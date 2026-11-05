"""Pure Dataset v2 planning and sparse-label contract checks."""
import copy
import unittest

import numpy as np

from scripts.dataset_v2 import DEFAULT, RANGES, canonical_case, join_rows, plan_source, sample_vectors


class DatasetV2PlanningTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(id="0449+113", nominal_source_snr=81.853,
                           reference_sha256="fixed-reference",
                           norms={"8": {"V_Ak_over_sigma_L2": 1.},
                                  "11": {"V_Ak_over_sigma_L2": 3.}})

    def test_default_budget_sparse_labels_and_lhs(self):
        plan = plan_source(self.source, DEFAULT, "train")
        self.assertEqual(plan, plan_source(self.source, DEFAULT, "train"))
        # Production budget: 4 vectors per nonempty mask, one noise slot.
        self.assertEqual((DEFAULT["vectors_per_mask"], DEFAULT["noise_slots"]), (4, 1))
        self.assertEqual(len(plan["vectors"]), 28)
        self.assertEqual(len(plan["rows"]), 29)
        self.assertLessEqual(len(plan["cases"]), 57)
        for mask in range(8):
            self.assertEqual(sum(r["mask"] == mask for r in plan["rows"]), 1 if mask == 0 else 4)
        # Both-gain masks get one vector per amp-branch/phase-sign combination.
        for mask in (3, 7):
            groups = {v["group"] for v in plan["vectors"] if v["mask"] == mask}
            self.assertEqual(len(groups), 4)
        for row in plan["rows"]:
            self.assertEqual(sum(row["b"]), row["mask"].bit_count())
            self.assertEqual(sum(bool(b and o) for b, o in zip(row["b"], row["o"])),
                             0 if row["mask"] == 0 else 1)
            self.assertEqual(row["o"], [int(not b or row["selected"] == kind)
                                         for b, kind in zip(row["b"], ("amp", "phase", "noise"))])
        self.assertTrue(all(v["a"] > 1 for v in plan["vectors"] if v["mask"] & 4))
        groups = {}
        for vector in plan["vectors"]:
            groups.setdefault(vector["group"], []).append(vector)
        for group in groups.values():
            n = len(group)
            for kind in group[0]["unit_cube"]:
                bins = sorted(int(v["unit_cube"][kind] * n) for v in group)
                self.assertEqual(bins, list(range(n)))

    def test_adjusted_ranges_preview_matches_exact_vectors(self):
        self.assertEqual(RANGES, {"phase_negative": (-180., -23.79),
                                  "phase_positive": (23.743, 180.),
                                  "amp_attenuation": (.01, .66391),
                                  "amp_amplification": (1.4309, 20.276),
                                  "noise_r": (.0097046, .19152)})
        preview = sample_vectors(self.source["id"], self.source["nominal_source_snr"], DEFAULT, "train")
        exact = plan_source(self.source, DEFAULT, "train")["vectors"]
        self.assertEqual(preview, [{k: value for k, value in v.items() if k != "antenna"} for v in exact])
        for v in preview:
            if v["amp_branch"] == "attenuation":
                lo, hi = RANGES["amp_attenuation"]
                self.assertAlmostEqual(v["A"], lo + v["unit_cube"]["amp"] * (hi - lo))
            if v["mask"] & 4:
                self.assertGreater(v["a"], 1.)
                self.assertLessEqual(v["r"], RANGES["noise_r"][1] * (1 + 1e-12))
                self.assertAlmostEqual(v["r"], v["a"] / self.source["nominal_source_snr"])
            else:
                self.assertEqual(v["a"], 1.)

    def test_equivalent_phase_endpoints_and_sparse_join(self):
        a = canonical_case("s", "ref", 0, 123, 1, -180, 1, 8)
        b = canonical_case("s", "ref", 0, 123, 1, 180, 1, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a["id"], canonical_case("s", "ref", 0, 124, 1, 180, 1, 8)["id"])
        plan = plan_source(self.source, DEFAULT, "train")
        records = {case["id"]: dict(y=.3, X=[1.] * 8) for case in plan["cases"]}
        rows = join_rows(plan, records)
        for row in rows:
            self.assertEqual([d is None for d in row["d"]],
                             [bool(b and not o) for b, o in zip(row["b"], row["o"])])
        broken = copy.deepcopy(records)
        broken.pop(plan["rows"][0]["full"])
        with self.assertRaises(ValueError):
            join_rows(plan, broken)

    def test_sparse_inverse_probability_expectation(self):
        # For a fixed three-active row, averaging the three possible observed
        # removals recovers the dense three-head squared-error sum.
        pred = np.array([.2, -.1, .4])
        dense = np.array([-.3, .2, .1])
        expected = np.square(pred - dense).sum() / 3
        sparse = []
        for selected in range(3):
            weight = np.zeros(3)
            weight[selected] = 3.
            sparse.append(np.dot(weight, np.square(pred - dense)) / 3)
        self.assertAlmostEqual(np.mean(sparse), expected)

    def test_worker_partition_is_disjoint_and_complete(self):
        from scripts.create_dataset_v2 import worker_sources
        ids = [f"s{i}" for i in range(11)]
        sizes = {sid: (i * 37) % 13 + 1 for i, sid in enumerate(ids)}
        parts = [worker_sources(ids, sizes, 4, w) for w in range(4)]
        self.assertEqual(sorted(sum(parts, [])), sorted(ids))
        self.assertEqual(sum(map(len, parts)), len(ids))
        self.assertEqual(parts, [worker_sources(ids, sizes, 4, w) for w in range(4)])


if __name__ == "__main__":
    unittest.main()
