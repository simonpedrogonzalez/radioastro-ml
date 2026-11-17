"""Scoring helpers of the final-eC model report on pipeline QA cases (v5)."""
import unittest
from unittest.mock import patch

import numpy as np

from ml import pipeline_qa_models as q


def row(group, subgroup, result, recovery, after, mask=3, source="after-recipe evaluation", missing=None):
    return dict(group=group, subgroup=subgroup, result=result, recovery=recovery, after=after, mask=mask,
                after_source=source, missing=missing)


class RegressionTest(unittest.TestCase):
    def test_known_values(self):
        r = q.regression([1., 2., 3.], [1., 2., 4.])
        self.assertEqual(r["pairs"], 3)
        self.assertAlmostEqual(r["rmse"], np.sqrt(1 / 3))
        self.assertAlmostEqual(r["r2"], 1 - 1 / (14 / 3))           # SS_tot of [1, 2, 4] = 14/3

    def test_negative_and_degenerate(self):
        self.assertLess(q.regression([3., 1.], [1., 3.])["r2"], 0)   # negative R² kept
        self.assertEqual(q.regression([], []), dict(pairs=0, rmse=None, r2=None))
        single = q.regression([.2], [.1])
        self.assertEqual((single["pairs"], single["r2"]), (1, None)); self.assertAlmostEqual(single["rmse"], .1)
        self.assertIsNone(q.regression([.1, .2, .3], [.1, .1, .1])["r2"])      # constant target
        r = q.regression([np.nan, 1., 2.], [5., 1., np.inf])                    # nonfinite pairs dropped
        self.assertEqual((r["pairs"], r["rmse"]), (1, 0.))


class SectionTest(unittest.TestCase):
    def rows(self):
        return [row("improvement_expected", None, "success", "solved", .1),
                row("improvement_expected", None, "partial_success", "improved", .3),
                row("improvement_expected", None, "failure", "similar", .5),
                row("no_improvement_expected", "noise_only", "success", "similar", .2, mask=4,
                    source="archived case (input unchanged: nothing applied)"),
                row("no_improvement_expected", "cancelling", "failure", "degraded", .6, mask=5),
                row("no_improvement_expected", "small", "pending", "pending", None, missing="no result record (pending)")]

    def test_tables_pool_pairs_and_keep_accounting(self):
        rows = self.rows()
        pred = np.array([.1, .2, .5, .25, .4, np.nan])
        t = q.section(rows, pred, "result", q.RESULTS)
        (_, overall), = t["table1"]
        scores = dict(overall)
        self.assertEqual((scores["all"]["total"], scores["all"]["pairs"]), (6, 5))
        pooled = q.regression(pred[:5], [.1, .3, .5, .2, .6])
        self.assertAlmostEqual(scores["all"]["rmse"], pooled["rmse"])          # pooled, not averaged
        self.assertAlmostEqual(scores["all"]["r2"], pooled["r2"])
        self.assertEqual(scores["pending"]["missing"], {"no result record (pending)": 1})
        self.assertEqual(scores["all"]["reused_pairs"], 1)
        groups = dict(t["table2"])
        self.assertNotIn("partial_success", dict(groups["no_improvement_expected"]))   # not applicable there
        self.assertIn("partial_success", dict(groups["improvement_expected"]))
        sub = dict(t["table3"])
        self.assertEqual(list(sub), list(q.SUBGROUPS))                          # zero subgroups kept
        self.assertEqual(dict(sub["uncorrupted"])["all"]["total"], 0)
        rec = q.section(rows, pred, "recovery", q.RECOVERY)
        self.assertEqual(dict(rec["table1"][0][1])["degraded"]["total"], 1)

    def test_no_reliance_on_injected_labels(self):
        rows, pred = self.rows(), np.array([.1, .2, .5, .25, .4, np.nan])
        a = q.section(rows, pred, "recovery", q.RECOVERY)
        for r in rows:
            r["mask"] = 0
        self.assertEqual(a, q.section(rows, pred, "recovery", q.RECOVERY))


class AfterFeaturesTest(unittest.TestCase):
    NAMES = ["f1", "f2"]

    def test_reused_requires_unchanged_evidence(self):
        ev = dict(status="measured", evaluation_reused=True, reason="nothing applied", feature_names=self.NAMES,
                  before=dict(eC=.2), after=dict(eC=.2, X=[0.]))
        with patch("scripts.vla_pipeline_qa.archived_features", return_value=[1., 2.]) as archived, \
                patch("subprocess.run", side_effect=AssertionError("no execution")):
            names, x, source, why = q.after_features("src", "c1", ev)
            self.assertEqual((x, why), ([1., 2.], None)); self.assertIn("unchanged", source)
            archived.assert_called_once()
            ev["after"]["eC"] = .3                                    # not verified unchanged: no pair
            self.assertIsNone(q.after_features("src", "c1", ev)[1])

    def test_after_features_never_before(self):
        ev = dict(status="measured", evaluation_reused=False, feature_names=self.NAMES, before=dict(eC=.2, X=[9., 9.]),
                  after=dict(eC=.1, X=[3., 4.]))
        self.assertEqual(q.after_features("src", "c1", ev)[1], [3., 4.])
        self.assertIn("1 after features", q.after_features("src", "c1", dict(ev, after=dict(eC=.1, X=[3.])))[3])
        self.assertIn("no measured", q.after_features("src", "c1", dict(status="failed"))[3])

    def test_missing_features_stay_in_totals(self):
        ev = dict(status="measured", evaluation_reused=False, before=dict(eC=.2), after=dict(eC=.1))
        self.assertEqual(q.after_features("src", "c1", ev)[3], "no stored feature names")
        self.assertEqual(q.after_features("src", "c1", dict(ev, feature_names=self.NAMES))[3], "no stored after features")
        rows = [dict(row("improvement_expected", None, "success", "solved", .1), case="a", names=["f1"], X=[1.]),
                dict(row("improvement_expected", None, "failure", "similar", .3), case="b", names=self.NAMES, X=[1., 2.])]
        x = q.matrix(rows, self.NAMES)
        self.assertTrue(np.isnan(x[0]).all()); self.assertEqual(rows[0]["missing"], "stored features lack f2")
        scores = dict(q.section(rows, np.array([np.nan, .3]), "result", q.RESULTS)["table1"][0][1])
        self.assertEqual((scores["all"]["total"], scores["all"]["pairs"]), (2, 1))
        self.assertEqual(scores["success"]["missing"], {"stored features lack f2": 1})


if __name__ == "__main__":
    unittest.main()
