"""Analytical checks for signed, partially observed recovery-damage loss."""

import numpy as np
import torch
import unittest

from ml.recovery_damage import damage_loss, torch_damage_loss


class DamageLossTests(unittest.TestCase):
    def test_dense_signed_loss(self):
        truth = np.array([[2., -1., 0.], [0., 0., 0.]])
        pred = np.array([[3., -3., 4.], [-2., 0., 1.]])
        observed = np.ones_like(truth)
        probability = np.ones_like(truth)
        total, per = damage_loss(pred, truth, observed, probability)
        np.testing.assert_allclose(per, [2.5, 2., 8.5])
        np.testing.assert_allclose(total, 13. / 3.)
        np.testing.assert_allclose(torch_damage_loss(torch.tensor(pred), torch.tensor(truth),
                                                     torch.tensor(observed), torch.tensor(probability)), total)


    def test_missing_active_label_and_inverse_probability(self):
        truth = np.array([[np.nan, -2., 0.], [4., np.nan, 0.]])
        pred = np.array([[100., -1., 1.], [2., 500., 3.]])
        observed = np.array([[0, 1, 1], [1, 0, 1]])
        probability = np.array([[0., .5, 1.], [.5, 0., 1.]])
        total, per = damage_loss(pred, truth, observed, probability)
        np.testing.assert_allclose(per, [4., 1., 5.])
        np.testing.assert_allclose(total, 10. / 3.)
        np.testing.assert_allclose(torch_damage_loss(torch.tensor(pred), torch.tensor(truth),
                                                     torch.tensor(observed), torch.tensor(probability)), total)
