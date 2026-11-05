"""Small disjoint-source five-family fit and saved-prediction report smoke."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml import multitask


class MultitaskFitTest(unittest.TestCase):
    def test_five_families_and_report_only(self):
        rng = np.random.default_rng(33)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, groups in (("train", 6), ("val", 2)):
                n = groups * 12
                X = rng.normal(size=(n, 8))
                b = np.tile(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                      [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1],
                                      [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), (groups, 1))
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
                folder = root / split
                folder.mkdir()
                np.savez_compressed(folder / f"{split}.npz", X=X, b=b, o=o, pi=pi, d=d,
                                    y_all=np.abs(rng.normal(size=n)),
                                    row_ids=np.array([f"{split}-{i}" for i in range(n)]),
                                    source_ids=np.repeat([f"{split}-source-{j}" for j in range(groups)], 12))
            mlp = multitask.fit_mlp
            def small_mlp(*args, **kwargs):
                return mlp(*args, max_epochs=4, patience=2, **kwargs)
            with patch.object(multitask, "GRID", {"n_estimators": [2], "max_depth": [2], "learning_rate": [.1]}), \
                 patch.object(multitask, "fit_mlp", small_mlp):
                summary = multitask.fit(root / "train", root / "val", root / "models", jobs=1)
            self.assertEqual(set(summary["selected"]), set(multitask.NAMES))
            self.assertEqual(np.load(root / "models/stage_features.npz")["train_oof"].shape, (72, 14))
            multitask.report_only(root / "train", root / "val", root / "models")
            self.assertTrue((root / "models/report.html").is_file())


if __name__ == "__main__":
    unittest.main()
