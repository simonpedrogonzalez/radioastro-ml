import unittest

import numpy as np
import torch

from ml.logreg import ALL_FEATURES, COMBINED8_FEATURES, SIMPLE_FEATURES, features_from_dataloader
from ml.simple_metrics import sigma_na_from_source, sigma_sna_units
from ml.tests.test_ml import _qa


class SigmaNAFeatureTests(unittest.TestCase):
    def test_source_rms_and_ratio(self):
        self.assertEqual(sigma_na_from_source({"predicted_image_rms_jy_per_beam": 2e-5}), 2e-5)
        self.assertAlmostEqual(sigma_sna_units(4e-5, 2e-5), 2)
        for value in (None, 0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                sigma_na_from_source({"predicted_image_rms_jy_per_beam": value})

    def test_extraction_preserves_six_features(self):
        image = torch.randn(1, 4, 256, 256, generator=torch.Generator().manual_seed(42))
        batch = {"image": image, "sample_id": ["0005+383_not_corrupted"], "label": [0],
                 "qa": [_qa([1., 2e-5, 3., 4.])], "label_metadata": [{}]}
        old = features_from_dataloader([batch])
        new = features_from_dataloader([batch], source_noise={"0005+383": {"predicted_image_rms_jy_per_beam": 1e-5}})
        self.assertEqual(new.feature_names, COMBINED8_FEATURES)
        np.testing.assert_array_equal(old.X, new.view(ALL_FEATURES))
        np.testing.assert_array_equal(new.view(SIMPLE_FEATURES)[0, -2:], [1e-5, 2.])
        with self.assertRaises(ValueError):
            features_from_dataloader([batch], source_noise={})
