"""--part-weight / --scale-part in ml.multitask: weight 1 reproduces unweighted fits; scales come from one part."""
import unittest

import numpy as np

from ml import multitask as mt


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, (n, 3)).astype(float)
    b[:3] = np.eye(3)
    o = b.copy()
    d = np.where(o == 1, rng.normal(0, 1, (n, 3)), np.nan)
    return dict(X=rng.normal(size=(n, 4)), b=b, o=o, pi=np.ones((n, 3)), d=d, y_all=rng.gamma(2, .2, n),
                source_ids=np.repeat(np.arange(6), n // 6).astype(str),
                part=np.where(np.arange(n) < n // 2, "v2", "tail"), with_dirty=False)


class PartWeights(unittest.TestCase):
    def test_scale_part_uses_only_that_part(self):
        train = toy()
        train["y_all"][train["part"] == "tail"] *= 10
        mt.apply_part_options(train, None, "v2")
        expected = mt.scales(mt.subset(toy(), toy()["part"] == "v2"))
        np.testing.assert_allclose(mt.scales(train), expected)
        np.testing.assert_allclose(mt.scales(mt.subset(train, np.arange(10))), expected)   # OOF folds too

    def test_row_weights_and_errors(self):
        train = toy()
        mt.apply_part_options(train, {"tail": .25}, None)
        self.assertEqual(sorted(set(train["row_weight"])), [.25, 1.])
        self.assertTrue(np.all(train["row_weight"][train["part"] == "v2"] == 1))
        with self.assertRaises(ValueError):
            mt.apply_part_options(toy(), {"nope": .5}, None)
        plain = toy(); del plain["part"]
        with self.assertRaises(ValueError):
            mt.apply_part_options(plain, {"tail": .5}, None)

    def test_unit_weight_matches_unweighted(self):
        params = dict(max_depth=2, learning_rate=.3, n_estimators=5)
        a, b = toy(), toy()
        b["row_weight"] = np.ones(len(b["X"]))
        scale = mt.scales(a)
        for shared in (True, False):
            ra = mt.predict_xgb(mt.fit_xgb(a["X"], a, params, scale, shared=shared, jobs=1), a["X"])
            rb = mt.predict_xgb(mt.fit_xgb(b["X"], b, params, scale, shared=shared, jobs=1), b["X"])
            np.testing.assert_allclose(ra, rb, atol=1e-6)
        ma = mt.fit_mlp(a["X"], a, a["X"], a, fixed_epochs=3)
        mb = mt.fit_mlp(b["X"], b, b["X"], b, fixed_epochs=3)
        np.testing.assert_allclose(mt.predict_mlp(ma, a["X"]), mt.predict_mlp(mb, a["X"]), atol=1e-5)

    def test_downweighting_changes_fit(self):
        params = dict(max_depth=2, learning_rate=.3, n_estimators=5)
        a, b = toy(), toy()
        mt.apply_part_options(b, {"tail": .1}, None)
        scale = mt.scales(a)
        ra = mt.predict_xgb(mt.fit_xgb(a["X"], a, params, scale, shared=True, jobs=1), a["X"])
        rb = mt.predict_xgb(mt.fit_xgb(b["X"], b, params, scale, shared=True, jobs=1), b["X"])
        self.assertGreater(np.abs(ra - rb).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
