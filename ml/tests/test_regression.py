from __future__ import annotations

import shutil
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ml.logreg import FEATURE_VIEWS, FeatureSet, grouped_folds
from ml.regression import (RegressionData, cross_talk, evaluate, evaluate_data,
                           fit_model, metric_values, target_from_metadata,
                           append_training_noise, validate_noise_training_index)
from ml.regression_report import write_report
from ml.regression_training import (training_mask, masked_squared_error, fit_filtered_candidate,
                                    fit_shared_candidate, predict_outputs)


def metadata(kind=None):
    result = {"simulation": {"noise": {"simplenoise_jy": 2.}}, "corruptions": []}
    if kind in ("amp", "phase"):
        result["corruptions"] = [{"solution": {"corruption_type": kind, "SNR_corr_target": 20.},
                                   "metrics": {"SNR_corr": 19.9}}]
    elif kind == "noise":
        result["simulation"]["noise_control"] = {
            "kind": "increased_noise", "reference_sigma_jy": 2., "final_total_sigma_jy": 2.5,
            "SNR_corr_target": 20., "SNR_corr_measured": 19.999,
        }
    return result


def toy_data():
    labels = np.tile([0, 1, 2, 3], 6)
    truth = np.array([[0., 0., 0.], [20., 0., 0.], [0., 20., 0.], [0., 0., 20.]])[labels]
    features = FeatureSet(np.random.default_rng(42).normal(size=(24, 6)), labels,
                          tuple(f"sample-{i}" for i in range(24)),
                          tuple(f"source-{i // 4}" for i in range(24)), tuple({} for _ in labels))
    return RegressionData(features, truth, np.array(["clean", "amp", "phase", "noise"])[labels],
                          np.where(labels == 0, 0., 20.), tuple({} for _ in labels))


class TypeSpecificTrainingTests(unittest.TestCase):
    def test_mask_and_objective_derivatives(self):
        data = toy_data()
        mask = training_mask(data.kinds)
        np.testing.assert_array_equal(mask[:4], [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pred = np.ones_like(data.targets)
        grad, hess = masked_squared_error(mask)(data.targets.ravel(), pred)
        def loss(p):
            return .5 * np.square(p - data.targets)[mask].sum()
        for row, col in ((0, 1), (1, 0), (1, 1), (3, 2)):
            step = np.zeros_like(pred); step[row, col] = .01
            numerical_grad = (loss(pred + step) - loss(pred - step)) / .02
            numerical_hess = (loss(pred + step) - 2 * loss(pred) + loss(pred - step)) / .0001
            self.assertAlmostEqual(grad[row, col], numerical_grad, places=7)
            self.assertAlmostEqual(hess[row, col], numerical_hess, places=7)
        np.testing.assert_array_equal(grad[~mask], 0)
        np.testing.assert_array_equal(hess[~mask], 0)

    def test_excluded_rows_do_not_train_the_output(self):
        data = toy_data(); X, y = data.features.X, data.targets
        mask = training_mask(data.kinds)
        params = {"n_estimators": 5, "max_depth": 2, "learning_rate": .1}
        changed_targets = y.copy(); changed_targets[~mask] = 1e6
        for fit in (fit_filtered_candidate, fit_shared_candidate):
            with self.subTest(fit=fit.__name__):
                model = fit(X, y, mask, params, jobs=1)
                changed = fit(X, changed_targets, mask, params, jobs=1)
                np.testing.assert_array_equal(predict_outputs(model, X), predict_outputs(changed, X))
        changed_X = X.copy(); changed_X[~mask[:, 0]] = 1e6
        original = fit_filtered_candidate(X, y, mask, params, jobs=1)
        changed = fit_filtered_candidate(changed_X, y, mask, params, jobs=1)
        np.testing.assert_array_equal(original[0].predict(X), changed[0].predict(X))

    def test_unmasked_custom_loss_matches_builtin_squared_error(self):
        from xgboost import XGBRegressor
        data = toy_data(); X, y = data.features.X, data.targets
        params = {"n_estimators": 5, "max_depth": 2, "learning_rate": .1}
        custom = fit_shared_candidate(X, y, np.ones_like(y, dtype=bool), params, jobs=1)
        builtin = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                               multi_strategy="multi_output_tree", base_score=y.mean(0).tolist(),
                               random_state=42, n_jobs=1, **params).fit(X, y)
        np.testing.assert_allclose(custom.predict(X), builtin.predict(X), rtol=0, atol=1e-6)


class TargetTests(unittest.TestCase):
    def test_baseline_and_pure_gain_have_normal_noise(self):
        self.assertEqual(target_from_metadata(metadata()), ([0., 0., 0.], "clean", 0.))
        self.assertEqual(target_from_metadata(metadata("amp")), ([19.9, 0., 0.], "amp", 20.))
        self.assertEqual(target_from_metadata(metadata("phase")), ([0., 19.9, 0.], "phase", 20.))

    def test_noise_uses_measured_extra_snr_not_total_sigma(self):
        self.assertEqual(target_from_metadata(metadata("noise")), ([0., 0., 19.999], "noise", 20.))

    def test_invalid_targets_and_mixtures_rejected(self):
        cases = []
        for bad in (None, 0., -1., float("nan"), float("inf"), True):
            sample = metadata("noise")
            sample["simulation"]["noise_control"]["reference_sigma_jy"] = bad
            cases.append(sample)
        sample = metadata("noise"); sample["simulation"]["noise_control"]["final_total_sigma_jy"] = 1.
        cases.append(sample)
        sample = metadata("noise"); sample["corruptions"] = metadata("amp")["corruptions"]
        cases.append(sample)
        sample = metadata("amp"); sample["corruptions"] *= 2
        cases.append(sample)
        sample = metadata("amp"); sample["corruptions"][0]["solution"]["corruption_type"] = "mixed"
        cases.append(sample)
        sample = metadata("amp"); sample["corruptions"][0]["metrics"]["SNR_corr"] = -1.
        cases.append(sample)
        for sample in cases:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                target_from_metadata(sample)


class EvaluationTests(unittest.TestCase):
    def test_constant_and_empty_r2_undefined(self):
        self.assertEqual(metric_values([1., 1.], [1., 2.]), {"mae": .5, "mse": .5, "rmse": np.sqrt(.5), "r2": None})
        self.assertIsNone(metric_values([1.], [2.])["r2"])
        self.assertEqual(metric_values([], []), {"mae": None, "mse": None, "rmse": None, "r2": None})
        self.assertEqual(metric_values([0., 2.], [0., 2.])["r2"], 1.)
        self.assertIsNone(metric_values([20. - 1e-8, 20. + 1e-8], [10., 30.])["r2"])
        self.assertIsNone(metric_values([5. - 1e-7, 5. + 1e-7], [10., 30.])["r2"])
        self.assertEqual(metric_values([20., 20.01], [20., 20.01])["r2"], 1.)

    def test_swaps_misses_and_clipping(self):
        truth = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 1.2]])
        pred = np.array([[0., 10., 0.], [-2., -1., 0.], [0., 0., .5]])
        result = evaluate(truth, pred)
        self.assertAlmostEqual(result["total"]["mae"], 10.7 / 3)
        self.assertAlmostEqual(result["mixture"]["mae"], 1 / 3)
        self.assertAlmostEqual(result["mixture"]["mse"], 1 / 3)
        self.assertEqual(result["coverage"], {"eligible": 3, "covered": 2, "missed": 1})
        self.assertEqual(result["mixture"]["r2"], 0.)
        self.assertAlmostEqual(result["main"]["noise"]["mae"], .7 / 3)

    def test_cross_talk_uses_full_extra_noise_snr(self):
        truth = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 1.5], [0., 0., 0.]])
        kinds = np.array(["amp", "phase", "noise", "clean"])
        rows = cross_talk(truth, truth, kinds)
        np.testing.assert_array_equal([row["values"] for row in rows], [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(rows[2]["denominator"], 1.5)
        self.assertEqual(rows[3]["n"], 1)

    def test_strength_slices_include_clean_and_count_sources(self):
        data = toy_data()
        result = evaluate_data(data, data.targets)
        group = result["by_strength"][0]
        self.assertEqual((group["strength"], group["n"], group["sources"]), (20., 24, 6))
        self.assertEqual(group["coverage"], {"eligible": 18, "covered": 18, "missed": 0})
        self.assertEqual(result["overall"]["mixture"]["r2"], 1.)
        self.assertNotIn('"r2"', json.dumps(result["by_strength"]))
        self.assertEqual(group["relative"]["n"], 18)

    def test_relative_errors_exclude_clean_and_mixture_includes_noise(self):
        data = toy_data()
        prediction = .8 * data.targets
        prediction[data.kinds == "clean"] = 100.  # Must not enter relative errors.
        result = evaluate_data(data, prediction)
        relative = result["by_strength"][0]["relative"]
        self.assertAlmostEqual(relative["total"]["mae"], .2)
        self.assertAlmostEqual(relative["total"]["rmse"], .2)
        self.assertAlmostEqual(relative["main"]["noise"]["mae"], .2 / 3)
        self.assertEqual(relative["mixture"]["mae"], 0.)
        prediction[data.kinds == "noise"] = [20., 0., 0.]
        result = evaluate_data(data, prediction)
        self.assertAlmostEqual(result["overall"]["mixture"]["mae"], 2 / 9)
        self.assertEqual(result["overall"]["coverage"]["eligible"], 18)


class FitAndReportTests(unittest.TestCase):
    def test_extra_index_rejects_held_out_and_baseline_samples(self):
        from scripts.preprocessing import TRAIN_IDS, VAL_IDS, TEST_IDS

        with tempfile.TemporaryDirectory() as directory:
            simulation = Path(directory) / "simulation.json"
            simulation.write_text(json.dumps(metadata("noise")["simulation"]))
            sample = SimpleNamespace(sample_id=TRAIN_IDS[0] + "_noise_snr_20", corruptions=[],
                                     label_name="not_corrupted", label_id=0, simulation=simulation)
            index = SimpleNamespace(samples=[Path("sample.json")], labels={"not_corrupted": 0})
            with patch("scripts.preprocessing.load_dataset_manifest", return_value=index), \
                 patch("scripts.preprocessing.load_sample_manifest", return_value=sample):
                validate_noise_training_index(Path("index.json"), {TRAIN_IDS[0]})
                for source in (VAL_IDS[0], TEST_IDS[0], TRAIN_IDS[1]):
                    sample.sample_id = source + "_noise_snr_20"
                    with self.assertRaisesRegex(ValueError, "outside main training"):
                        validate_noise_training_index(Path("index.json"), {TRAIN_IDS[0]})
                sample.sample_id = TRAIN_IDS[0] + "_not_corrupted"
                simulation.write_text(json.dumps(metadata()["simulation"]))
                with self.assertRaisesRegex(ValueError, "ordinary sample"):
                    validate_noise_training_index(Path("index.json"), {TRAIN_IDS[0]})

    def test_augmented_rows_remain_aligned_and_source_grouped(self):
        main = toy_data()
        features = FeatureSet(np.arange(36).reshape(6, 6), np.zeros(6),
                              tuple(f"noise-{i}" for i in range(6)), tuple(f"source-{i}" for i in range(6)),
                              tuple({"sample_kind": "increased_noise"} for _ in range(6)))
        extra = RegressionData(features, np.tile([0., 0., 20.], (6, 1)), np.array(["noise"] * 6),
                               np.full(6, 20.), tuple({"residual": f"noise-{i}"} for i in range(6)))
        merged = append_training_noise(main, extra)
        np.testing.assert_array_equal(merged.features.X[24:], extra.features.X)
        np.testing.assert_array_equal(merged.targets[24:], extra.targets)
        self.assertEqual(merged.features.sample_ids[24:], extra.features.sample_ids)
        self.assertEqual(merged.products[24:], extra.products)
        for train, validation in grouped_folds(merged.features, 3):
            self.assertFalse(set(np.asarray(merged.features.source_ids)[train]) & set(np.asarray(merged.features.source_ids)[validation]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            append_training_noise(merged, extra)

    def test_grouped_cv_and_model_roundtrip(self):
        from xgboost import XGBRegressor

        data = toy_data()
        folds = grouped_folds(data.features, 3)
        for train, validation in folds:
            self.assertFalse(set(np.asarray(data.features.source_ids)[train]) & set(np.asarray(data.features.source_ids)[validation]))
        for strategy in ("one_output_per_tree", "multi_output_tree"):
            with self.subTest(strategy=strategy), patch("ml.regression.GRID", {
                    "n_estimators": [3, 5], "max_depth": [2], "learning_rate": [.1]}):
                model, best, candidates = fit_model(data.features.X, data.targets, folds, jobs=1,
                                                     multi_strategy=strategy)
                self.assertEqual(len(candidates), 2)
                self.assertTrue(all(len(row["fold_scores"]) == 3 for row in candidates))
                self.assertEqual(best, min(candidates, key=lambda row: row["mean_scaled_mae"])["params"])
                pred = model.predict(data.features.X)
                self.assertEqual(pred.shape, data.targets.shape)
                self.assertTrue(np.any(pred[:, 2] > 0))
                learner = json.loads(model.get_booster().save_config())["learner"]
                self.assertEqual(learner["learner_train_param"]["multi_strategy"], strategy)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.json"
                    model.save_model(path)
                    restored = XGBRegressor(); restored.load_model(path)
                    np.testing.assert_array_equal(pred, restored.predict(data.features.X))

    @unittest.skipUnless(shutil.which("quarto"), "Quarto is needed for the HTML integration check")
    def test_report_renders_with_constant_targets_and_misses(self):
        data = toy_data()
        predictions = {name: data.targets.copy() for name in FEATURE_VIEWS}
        predictions["qa4"][1, :2] = 0.
        results = {name: evaluate_data(data, pred) for name, pred in predictions.items()}
        results["no_error"] = evaluate_data(data, np.tile([0., 0., 0.], (len(data.targets), 1)))
        matrices = {name: cross_talk(data.targets, pred, data.kinds) for name, pred in predictions.items()}
        config = {"quick": True, "noise_provisional": False, "best_params": {name: {"n_estimators": 20} for name in FEATURE_VIEWS}}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_report(run, {"train": data, "validation": data}, {"validation": predictions},
                         {"validation": results}, matrices, config)
            html = (run / "report.html").read_text()
            self.assertNotIn("Noise output is provisional", html)
            self.assertIn("17/18", html)
            self.assertIn("data:image/png;base64,", html)
            qmd = (run / "report.qmd").read_text()
            self.assertNotIn("NaN", qmd)
            self.assertNotIn("Share", qmd)
            self.assertNotIn("Higher gain", qmd)
            self.assertNotIn("Separate noise", qmd)
            self.assertIn("Mixture MSE", qmd)
            self.assertIn("Relative", qmd)
            self.assertGreater(qmd.index("## Training coverage"), qmd.index("## Hard samples"))


if __name__ == "__main__":
    unittest.main()
