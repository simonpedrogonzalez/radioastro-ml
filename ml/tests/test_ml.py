from __future__ import annotations

import math
import pickle
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.evaluate import evaluate_classification
from ml.logreg import (ALL_FEATURES, QA_FEATURES, FeatureSet,
                       features_from_dataloader, grouped_folds,
                       parity_features, tune)
from ml.task_evaluation import evaluate_noise_robustness, evaluate_task


def _qa(values: list[float]) -> dict:
    metrics: dict = {}
    validity: dict = {}
    for feature_name, value in zip(QA_FEATURES, values, strict=True):
        keys = feature_name.split(".")[1:]
        metric_parent = metrics
        validity_parent = validity
        for key in keys[:-1]:
            metric_parent = metric_parent.setdefault(key, {})
            validity_parent = validity_parent.setdefault(key, {})
        metric_parent[keys[-1]] = value
        validity_parent[keys[-1]] = True
    return {"metrics": metrics, "metric_validity": validity}


class FeatureTests(unittest.TestCase):
    def test_feature_order_and_invalid_values(self):
        values = [float(index) for index in range(1, len(QA_FEATURES) + 1)]
        torch.manual_seed(42)
        image = torch.randn(4, 256, 256)
        batch = {
            "image": image[None],
            "sample_id": ["0005+383_amp_snr_10"],
            "label": [1],
            "qa": [_qa(values)],
            "label_metadata": [
                {"corruption_snr": 10.1, "corruption_snr_target": 10.0,
                 "sample_kind": "gain"}
            ],
        }
        result = features_from_dataloader([batch])
        np.testing.assert_allclose(result.X[0, :4], values)
        np.testing.assert_allclose(result.X[0, 4:], parity_features(image))
        self.assertEqual(result.view(QA_FEATURES).tolist(), [values])
        self.assertEqual(result.X.shape[1], len(ALL_FEATURES))
        self.assertEqual(result.y.tolist(), [1])
        self.assertEqual(result.source_ids, ("0005+383",))

        invalid = _qa(values)
        invalid["metric_validity"]["residual"]["scaled_mad_jy_per_beam"] = False
        batch["qa"] = [invalid]
        with self.assertRaisesRegex(ValueError, "0005.*scaled_mad_jy_per_beam"):
            features_from_dataloader([batch])

        nonfinite = _qa(values)
        nonfinite["metrics"]["residual"]["scaled_mad_jy_per_beam"] = math.nan
        batch["qa"] = [nonfinite]
        with self.assertRaisesRegex(ValueError, "0005.*scaled_mad_jy_per_beam"):
            features_from_dataloader([batch])

    def test_grouped_tuning_for_both_estimators(self):
        rng = np.random.default_rng(42)
        groups = tuple(f"source-{index}" for index in range(9) for _ in range(3))
        y = np.tile(np.arange(3), 9)
        X = np.column_stack((y + rng.normal(0, .05, len(y)), rng.normal(size=len(y)),
                             rng.normal(size=(len(y), 4))))
        features = FeatureSet(X, y, tuple(map(str, range(len(y)))), groups,
                              tuple({} for _ in y))
        folds = grouped_folds(features, 3)
        for train, validation in folds:
            self.assertFalse(set(np.asarray(groups)[train]) & set(np.asarray(groups)[validation]))
        grids = {
            "logreg": {"logisticregression__C": [1.0],
                       "logisticregression__l1_ratio": [0.0],
                       "logisticregression__class_weight": ["balanced"]},
            "xgboost": {"n_estimators": [5], "max_depth": [2],
                        "learning_rate": [.1], "min_child_weight": [1],
                        "subsample": [1.0], "colsample_bytree": [1.0],
                        "reg_lambda": [1]},
        }
        for algorithm, grid in grids.items():
            model, summary, rows = tune(algorithm, X[:, :2], y, folds, n_jobs=1, grid=grid)
            probabilities = model.predict_proba(X[:2, :2])
            self.assertEqual(probabilities.shape, (2, 3))
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_allclose(restored.predict_proba(X[:2, :2]), probabilities)
            self.assertEqual(len(rows), 1)
            self.assertGreaterEqual(summary["mean_f1"], 0)

class EvaluationTests(unittest.TestCase):
    def test_generic_evaluator_is_label_and_class_count_agnostic(self):
        for labels in ((30, 10), ("z", "x", "y"), (9, 4, 7, 2)):
            truth = np.repeat(labels, 2)
            probabilities = np.full((len(truth), len(labels)), 0.1 / (len(labels) - 1))
            for row, label in enumerate(truth):
                probabilities[row, labels.index(label)] = 0.9
            result = evaluate_classification(truth, probabilities, labels)
            self.assertEqual(result["labels"], list(labels))
            self.assertEqual(result["confusion_matrix"], (np.eye(len(labels), dtype=int) * 2).tolist())
            self.assertTrue(all(result["macro"][metric] == 1 for metric in
                                ("precision", "recall", "f1", "auroc", "auprc")))

    def test_evaluator_line_limit_and_no_task_semantics(self):
        path = Path(__file__).parents[1] / "evaluate.py"
        source = path.read_text()
        self.assertLessEqual(len(source.splitlines()), 200)
        for forbidden in ("sample_kind", "corruption", "clean_label", "target_snr"):
            self.assertNotIn(forbidden, source)

    def test_overall_and_strength_views(self):
        truth = np.asarray([0, 1, 2, 1, 2])
        probabilities = np.asarray([
            [.1, .8, .1], [.1, .8, .1], [.8, .1, .1],
            [.1, .2, .7], [.1, .2, .7],
        ])
        metadata = [
            {"corruption_snr_target": level, "sample_kind": kind}
            for level, kind in (
                (0, "baseline"), (10, "gain"), (10, "gain"),
                (30, "gain"), (30, "gain"),
            )
        ]
        result = evaluate_task(
            truth, probabilities, (0, 1, 2), metadata, clean_label=0
        )

        self.assertEqual(result["evaluation_kind"], "classification")
        self.assertEqual(result["overall"]["main"]["confusion_matrix"],
                         [[0, 1, 0], [0, 1, 1], [1, 0, 1]])
        detection = result["overall"]["detection"]
        self.assertAlmostEqual(detection["precision"], .75)
        self.assertAlmostEqual(detection["recall"], .75)
        error = result["overall"]["error_identification"]
        self.assertEqual(error["detected_corruptions"], 3)
        self.assertAlmostEqual(error["detection_coverage"], .75)
        self.assertAlmostEqual(error["precision"], .75)
        self.assertEqual(set(result["by_corruption_snr_target"]), {"10", "30"})
        for views in result["by_corruption_snr_target"].values():
            self.assertEqual(views["main"]["sample_count"], 3)
            self.assertEqual([v["support"] for v in views["main"]["per_class"].values()], [1, 1, 1])

    def test_type_is_undefined_when_no_corruption_is_detected(self):
        truth = [0, 1, 2]
        probabilities = [[.9, .05, .05], [.8, .1, .1], [.7, .1, .2]]
        metadata = [
            {"corruption_snr_target": level, "sample_kind": kind}
            for level, kind in ((0, "baseline"), (10, "gain"), (10, "gain"))
        ]
        error = evaluate_task(
            truth, probabilities, (0, 1, 2), metadata, clean_label=0
        )["overall"]["error_identification"]
        self.assertEqual(error["detection_coverage"], 0.0)
        self.assertTrue(all(error[metric] is None for metric in
                            ("precision", "recall", "f1", "auroc", "auprc")))

    def test_increased_noise_controls_report_false_positive_rate(self):
        result = evaluate_task(
            [0, 0, 0, 0],
            [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.8, .1, .1]],
            (0, 1, 2),
            [
                {
                    "corruption_snr_target": 20.0,
                    "sample_kind": "increased_noise",
                }
            ]
            * 4,
            clean_label=0,
        )

        self.assertEqual(result["evaluation_kind"], "increased_noise_controls")
        level = result["by_noise_snr_target"]["20"]
        self.assertEqual(level["sample_count"], 4)
        self.assertEqual(level["false_positive_rate"], 0.5)
        self.assertEqual(level["correct_rejection_rate"], 0.5)
        self.assertEqual(level["predicted_counts"], {"0": 2, "1": 1, "2": 1})
        self.assertNotIn("auroc", level)

    def test_noise_controls_use_matched_source_baselines(self):
        result = evaluate_noise_robustness(
            [0, 0, 0, 0],
            [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.8, .1, .1]],
            (0, 1, 2),
            [{"corruption_snr_target": level, "sample_kind": "increased_noise"}
             for level in (10, 10, 20, 20)],
            ("a", "b", "a", "b"),
            [0, 1, 2, 0],
            [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.2, .7, .1]],
            [{"corruption_snr_target": level, "sample_kind": kind}
             for level, kind in ((0, "baseline"), (10, "gain"), (10, "gain"),
                                 (0, "baseline"))],
            ("a", "a", "a", "b"),
            clean_label=0,
        )
        self.assertEqual(result["evaluation_kind"], "noise_robustness")
        self.assertEqual(result["overall"]["false_positive_rate"], .5)
        self.assertEqual(result["overall"]["matched_baseline_false_positive_rate"], .5)
        self.assertEqual(result["overall"]["excess_false_positive_rate"], 0)
        self.assertEqual(result["by_noise_snr_target"]["10"]["false_positive_rate"], .5)

    def test_evaluation_rejects_mixed_gain_and_noise_controls(self):
        metadata = [
                {"corruption_snr_target": 10.0, "sample_kind": "gain"},
                {
                    "corruption_snr_target": 10.0,
                    "sample_kind": "increased_noise",
                },
            ]
        with self.assertRaisesRegex(ValueError, "Cannot pool"):
            evaluate_task(
                [1, 0], [[.1, .8, .1], [.8, .1, .1]], (0, 1, 2),
                metadata, clean_label=0,
            )

    def test_invalid_probabilities_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_classification(["a"], [[1, 0]], ("a", "b", "c"))
        with self.assertRaisesRegex(ValueError, "sum"):
            evaluate_classification(["a"], [[.2, .2]], ("a", "b"))
        with self.assertRaisesRegex(ValueError, "absent"):
            evaluate_classification(["c"], [[.5, .5]], ("a", "b"))


if __name__ == "__main__":
    unittest.main()
