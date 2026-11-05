"""Matched schema, feature ablations, missing inputs and source-held-out evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml import multitask as mt
from ml.matched_comparison import compare
from scripts.dataset_v2_matched_extract import FEATURES


def write_data(root, split, sources):
    rng = np.random.default_rng(4 if split == "train" else 5)
    n = sources * 8
    b = np.tile([[bool(m & (1 << k)) for k in range(3)] for m in range(8)], (sources, 1)).astype(int)
    x = rng.normal(size=(n, 20))
    x[:, 14] = np.nan  # An entirely unavailable model input must also survive MLP folds.
    x[::5, 19] = np.nan
    folder = root / split
    folder.mkdir()
    np.savez_compressed(folder / f"{split}.npz", X=x, feature_names=np.asarray(FEATURES),
                        d=.1 * b * x[:, :3], eC=np.abs(x[:, 0]) + .1,
                        eD=np.ones(n), p=b.astype(float), b=b, o=np.ones_like(b), pi=np.ones_like(b, float),
                        params=np.column_stack((np.exp(x[:, 0]), x[:, 1]*100, np.exp(x[:, 2]))),
                        row_ids=np.array([f"{split}-{i}" for i in range(n)]),
                        source_ids=np.repeat([f"{split}-source-{j}" for j in range(sources)], 8))
    return folder


class MatchedIntegration(unittest.TestCase):
    def test_views_five_families_missing_values_and_reload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tr, va = write_data(root, "train", 6), write_data(root, "val", 2)
            for view, width in zip(mt.FEATURE_VIEWS, (8, 14, 14, 20)):
                data = mt.load_split(tr, "train", view)
                self.assertEqual(data["X"].shape, (48, width))
                np.testing.assert_array_equal(data["y_all"], data["eC"])
                self.assertEqual(len(mt.subset(data, np.arange(width))["feature_names"]), width)
            original = mt.fit_mlp
            def small(*a, **kw): return original(*a, max_epochs=3, patience=2, **kw)
            with patch.object(mt, "GRID", dict(n_estimators=[2], max_depth=[2], learning_rate=[.1])), patch.object(mt, "fit_mlp", small):
                summary = mt.fit(tr, va, root / "models", jobs=1, feature_view="all20")
            self.assertEqual(len(summary["feature_names"]), 20)
            self.assertEqual(mt.load_frozen(root / "models", "mlp")["model"][0].in_features, 20)
            self.assertEqual(mt.load_frozen(root / "models", "xgb_mlp")["model"][0].in_features, 34)
            obj = mt.load_frozen(root / "models", "mlp")
            train = mt.load_split(tr, "train", "all20")
            self.assertEqual(obj["fill"][14], 0.)
            self.assertAlmostEqual(obj["fill"][19], np.nanmean(train["X"][:, 19]))
            self.assertEqual(summary["metrics"]["val"]["mlp"]["amp"]["leakage"]["phase"]["rows"], 4)
            mt.report_only(tr, va, root / "models")
            self.assertIn("False damage attributed", (root / "models/report.html").read_text())
            test_root = write_data(root, "test", 2)
            mt.evaluate_test(test_root, root / "models")
            frozen = json.loads((root / "models/test_results.json").read_text())
            self.assertEqual(frozen["test_rows"], 16)
            self.assertEqual(set(frozen["results"]), set(mt.NAMES))

    def test_nested_source_comparison_all_views(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); train = write_data(root, "train", 6)
            original = mt.fit_mlp
            def small(*a, **kw): return original(*a, max_epochs=2, patience=1, **kw)
            with patch.object(mt, "GRID", dict(n_estimators=[1], max_depth=[2], learning_rate=[.1])), \
                 patch.object(mt, "fit_mlp", small), patch.object(mt, "write_report"):
                result = compare(train, root / "comparison", folds=3, jobs=1, repeats=20)
            seen = []
            for fold in result["folds"]:
                fit, tune, evaluation = (set(fold[k]) for k in ("fit_sources", "tuning_sources", "evaluation_sources"))
                self.assertFalse(fit & tune or fit & evaluation or tune & evaluation)
                seen += list(evaluation)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(len(seen), 6)
            self.assertEqual(set(result["views"]), set(mt.FEATURE_VIEWS))
            for view in mt.FEATURE_VIEWS:
                self.assertIn("mask_3", result["views"][view]["xgb_shared"]["slices"])
                ci = result["views"][view]["xgb_shared"]["rmse_source_bootstrap"]["active"]["amp"]
                self.assertLessEqual(ci[0], ci[1])
            json.loads((root / "comparison/results.json").read_text())


if __name__ == "__main__":
    unittest.main()
