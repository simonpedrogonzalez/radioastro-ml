from __future__ import annotations

import math
import unittest

from ml.evaluate import evaluate
from ml.logreg import FEATURE_NAMES, features_from_dataloader


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
    def test_overall_and_severity_metrics(self):
        sample_ids = [
            "0005+383_none",
            "0005+383_amp_10",
            "0005+383_phase_10",
            "0005+383_amp_30",
            "0005+383_phase_30",
        ]
        batch = {
            "sample_id": sample_ids,
            "label": [0, 1, 2, 1, 2],
            "label_metadata": [
                {"corruption_snr_target": level}
                for level in (0.0, 10.0, 10.0, 30.0, 30.0)
            ],
        }
        predictions = dict(zip(sample_ids, [1, 1, 0, 2, 2], strict=True))
        result = evaluate([batch], predictions)

        self.assertAlmostEqual(result["accuracy"], 0.4)
        self.assertAlmostEqual(result["balanced_accuracy"], 1 / 3)
        self.assertEqual(
            result["confusion_matrix"], [[0, 1, 0], [0, 1, 1], [1, 0, 1]]
        )
        levels = result["by_corruption_snr_target"]
        self.assertEqual(levels["0"]["false_positive_rate"], 1.0)
        self.assertEqual(levels["10"]["detection_recall"], 0.5)
        self.assertEqual(levels["10"]["corruption_type_accuracy"], 0.5)
        self.assertEqual(levels["10"]["type_accuracy_among_detected"], 1.0)
        self.assertEqual(levels["30"]["detection_recall"], 1.0)
        self.assertEqual(levels["30"]["corruption_type_accuracy"], 0.5)
        self.assertEqual(levels["30"]["type_accuracy_among_detected"], 0.5)

    def test_type_is_undefined_when_no_corruption_is_detected(self):
        batch = {"sample_id": ["a", "b"], "label": [1, 2],
                 "label_metadata": [{"corruption_snr_target": 10.0}] * 2}
        level = evaluate([batch], {"a": 0, "b": 0})["by_corruption_snr_target"]["10"]
        self.assertEqual(level["detection_recall"], 0.0)
        self.assertIsNone(level["type_accuracy_among_detected"])


if __name__ == "__main__":
    unittest.main()
