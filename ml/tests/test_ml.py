from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from ml.evaluate import evaluate_classification
from ml.logreg import FEATURE_NAMES, features_from_dataloader
from ml.task_evaluation import evaluate_task


def _qa(values: list[float]) -> dict:
    metrics: dict = {}
    validity: dict = {}
    for feature_name, value in zip(FEATURE_NAMES, values, strict=True):
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
        values = [float(index) for index in range(1, len(FEATURE_NAMES) + 1)]
        batch = {
            "sample_id": ["0005+383_amp_snr_10"],
            "label": [1],
            "qa": [_qa(values)],
            "label_metadata": [
                {"corruption_snr": 10.1, "corruption_snr_target": 10.0}
            ],
        }
        result = features_from_dataloader([batch])
        self.assertEqual(result.X.tolist(), [values])
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
