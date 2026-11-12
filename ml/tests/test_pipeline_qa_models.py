"""Scoring helpers of the multitask-on-pipeline-QA report."""
import unittest

import numpy as np

from ml import pipeline_qa_models as q


def row(category, mask, before, after, reference, label="s:r00:m1:slot0", d=None):
    return dict(category=category, mask=mask, before=before, after=after, reference=reference,
                boundary=reference + .05, label=label, d=d)


class PipelineQaModelsTest(unittest.TestCase):
    def test_classification_counts(self):
        c = q.classification([True, True, False, False], np.array([.9, .2, .7, .1]), .5)
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["flagged"], c["positives"]), (1, 1, 1, 2, 2))
        self.assertAlmostEqual(c["recall_yes"], .5)
        self.assertAlmostEqual(c["false_alarm"], .5)
        self.assertAlmostEqual(c["auroc"], .75)

    def test_crosstab_sets_and_boundary(self):
        rows = [row("solved", 3, .2, .01, 0.), row("unchanged", 3, .2, .2, 0.), row("unchanged", 5, .3, .3, .1)]
        p = np.array([[.1, .1, .1], [.9, .8, .1], [.1, .1, .9]])      # clean, amp+phase, noise
        t = q.crosstab(rows, p)
        self.assertEqual(t["solved"]["sets"]["clean"], 1)
        self.assertEqual(t["unchanged"]["total"], 2)
        self.assertEqual(t["unchanged"]["sets"]["amp+phase"], 1)
        self.assertEqual(t["unchanged"]["sets"]["noise"], 1)
        self.assertEqual(t["unchanged"]["gain_flagged"], 1)
        self.assertEqual(t["unchanged"]["above_after"], 2)
        self.assertEqual(t["solved"]["above_after"], 0)

    def test_crosstab_v4_partitions_each_dimension(self):
        rows = [dict(row("solved", 3, .2, .01, 0.), expectation="improvement_expected", pipeline="applied",
                     recovery="improved", image="cleaner"),
                dict(row("rejected", 3, .2, .2, 0.), expectation="improvement_expected", pipeline="rejected",
                     recovery="similar", image="unavailable"),
                dict(row("preserved", 4, .1, .1, .1), expectation="no_improvement_expected", pipeline="applied",
                     recovery="similar", image="similar")]
        t = q.crosstab_v4(rows, np.array([[.1, .1, .1], [.9, .1, .1], [.1, .1, .9]]))
        imp = t["improvement_expected"]
        self.assertEqual(imp["all"]["all"]["total"], 2)
        for dim in ("pipeline", "recovery", "image"):              # each dimension partitions the group
            self.assertEqual(sum(x["total"] for x in imp[dim].values()), 2)
        self.assertEqual(imp["pipeline"]["rejected"]["sets"]["amp"], 1)
        self.assertEqual(imp["image"]["unavailable"]["total"], 1)  # accounting state kept as its own row
        self.assertEqual(t["no_improvement_expected"]["recovery"]["similar"]["sets"]["noise"], 1)

    def test_before_truth_prefers_dataset_labels(self):
        rows = [row("solved", 1, .2, .01, 0., label="a", d=[.15, 0, 0]), row("solved", 2, .2, .01, 0., label="b",
                                                                              d=[0, .1, 0])]
        labels = {"a": (np.array([np.nan, 0, 0]), np.array([0, 1, 1]), np.array([.5, 1, 1]), np.array([1, 0, 0]))}
        t = q.before_truth(rows, labels)
        np.testing.assert_array_equal(t["o"], [[0, 1, 1], [1, 1, 1]])
        np.testing.assert_array_equal(t["pi"][0], [.5, 1, 1])
        np.testing.assert_allclose(t["d"][1], [0, .1, 0])
        with self.assertRaises(ValueError):                            # dataset presence must match the plan mask
            q.before_truth(rows, {"b": (np.zeros(3), np.ones(3), np.ones(3), np.array([1, 0, 0]))})


if __name__ == "__main__":
    unittest.main()
