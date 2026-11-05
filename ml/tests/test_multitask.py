"""Scaled sparse three-task objective and seven-output XGBoost checks."""
import unittest

import numpy as np
from scipy.special import expit

from ml.multitask import fit_xgb, mixed_derivatives, predict_xgb, raw_losses, scales, targets, task_metrics


class MultitaskTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 24
        b = rng.integers(0, 2, size=(n, 3))
        b[:3] = np.eye(3, dtype=int)
        o = np.ones((n, 3), int)
        pi = np.ones((n, 3), float)
        for i in range(n):
            active = np.flatnonzero(b[i])
            if len(active) > 1:
                o[i, active] = 0
                o[i, active[i % len(active)]] = 1
                pi[i, active] = 1 / len(active)
        d = rng.normal(size=(n, 3)) * .1
        d[o == 0] = np.nan
        self.data = dict(X=rng.normal(size=(n, 8)), b=b, o=o, pi=pi, d=d,
                         y_all=np.abs(rng.normal(size=n)))

    def test_gradient_hessian_and_unknown_target_mask(self):
        data = self.data
        scale = scales(data)
        target = targets(data, scale)
        rng = np.random.default_rng(9)
        raw = rng.normal(size=target.shape)
        grad, hess = mixed_derivatives(raw, target, data["o"], data["pi"])
        self.assertTrue(np.isfinite(grad).all())
        self.assertTrue(np.isfinite(hess).all())
        self.assertTrue(np.all(grad[:, :3][data["o"] == 0] == 0))
        self.assertTrue(np.all(hess[:, :3][data["o"] == 0] == 0))
        for i, j in ((0, 0), (2, 3), (1, 4)):
            delta = 1e-5
            plus, minus = raw.copy(), raw.copy()
            plus[i, j] += delta; minus[i, j] -= delta
            numeric = (raw_losses(plus, data, scale)["J"] - raw_losses(minus, data, scale)["J"]) / (2 * delta)
            self.assertAlmostEqual(numeric * len(raw), grad[i, j], places=5)

    def test_both_tree_families_seven_raw_outputs(self):
        data = self.data
        scale = scales(data)
        params = dict(n_estimators=2, max_depth=2, learning_rate=.1)
        for shared in (True, False):
            model = fit_xgb(data["X"], data, params, scale, shared=shared, jobs=1)
            raw = predict_xgb(model, data["X"])
            self.assertEqual(raw.shape, (len(data["X"]), 7))
            self.assertTrue(np.isfinite(raw).all())
            self.assertTrue(np.isfinite(raw_losses(raw, data, scale)["J"]))

    def test_active_absent_damage_split(self):
        # Hand case: absent rows have zero truth, so their RMSE is the prediction size.
        data = dict(X=np.zeros((4, 8)), b=np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]),
                    o=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]]),
                    pi=np.array([[1, 1, 1], [.5, .5, 1], [1, 1, 1], [1, 1, 1]]),
                    d=np.array([[.2, 0, 0], [.4, np.nan, 0], [0, 0, 0], [0, 0, 0]]),
                    y_all=np.ones(4))
        prediction = np.zeros((4, 7)); prediction[:, 0] = [.2, .0, .1, -.1]; prediction[:, 4:] = .5
        amp = task_metrics(prediction, data)["amp"]
        self.assertEqual(amp["damage_active"]["rows"], 2)
        # Active: weights 1 and 2 (1/pi) over 2 active rows: (0 + 2*.16)/2.
        self.assertAlmostEqual(amp["damage_active"]["rmse"], np.sqrt(.16))
        self.assertAlmostEqual(amp["damage_absent"]["rmse"], .1)
        self.assertAlmostEqual(amp["damage_absent"]["mean_prediction"], 0.)
        active, absent = raw_losses(np.zeros((4, 7)), data, np.ones(4))["L_dk_active"], None
        self.assertAlmostEqual(active[0], (.04 + 2 * .16) / 2)


if __name__ == "__main__":
    unittest.main()
