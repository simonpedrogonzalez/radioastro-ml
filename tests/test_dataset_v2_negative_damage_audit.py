"""Negative-damage audit on a small saved-metric fixture (no CASA, no images)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import dataset_v2_negative_damage_audit as audit

# eC per mask of one recipe (slot 0). Expected labels:
#  1 amp:   d_amp -.02, D -.02          -> single-kind negative and gain-only negative D (exception)
#  3 a+p:   d_amp -.08, d_phase +.04, D +.02   -> negative individual with D >= 0, Small
#  5 a+n:   d_amp -5e-7 (tiny), D -5e-7  -> Cancelling, tiny
#  6 p+n:   D +.2                        -> Improvement expected
#  7 all:   d_amp -.25, d_phase -.0499995, D -.05 -> overlapping negatives, Cancelling
EC = {0: .10, 1: .08, 2: .20, 3: .12, 4: .30, 5: .2999995, 6: .50, 7: .25}
PARAMS = dict(A=2., phi_deg=30., a=5.)


def case(mask, seed=7):
    return dict(id=f"c{mask}", source="s", slot=0, noise_seed=seed, reference_sha256="r", antenna=3 if mask & 3 else None,
                A=PARAMS["A"] if mask & 1 else 1., phi_deg=PARAMS["phi_deg"] if mask & 2 else 0.,
                a=PARAMS["a"] if mask & 4 else 1.)


def write_dataset(root, sparse=(), seeds=None, drop_eC=()):
    root.mkdir(parents=True)
    rows, stored = [], []
    for m in range(8):
        removal = [f"c{m & ~(1 << k)}" if m >> k & 1 else None for k in range(3)]
        rows.append(dict(id=f"s:r00:m{m}:slot0" if m else "s:control:slot0", case=f"c{m}", mask=m, slot=0,
                         recipe="s:r00" if m else None, removal=removal, source="s", split="val"))
        d = [EC[m] - EC[m & ~(1 << k)] if m >> k & 1 else 0. for k in range(3)]
        stored.append(dict(case=f"c{m}", eC=EC[m], d=None if m in sparse else d))
        if m not in drop_eC:
            (root / "cases" / f"c{m}").mkdir(parents=True)
            (root / "cases" / f"c{m}" / "case.json").write_text(json.dumps(dict(eC=EC[m])))
    cases = [case(m, (seeds or {}).get(m, 7)) for m in range(8)]
    plan = dict(split="val", configuration_hash="h", source_plans=[dict(source="s", split="val", rows=rows, cases=cases)])
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "rows.json").write_text(json.dumps(dict(rows=stored)))
    return root


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.guard = patch("subprocess.run", side_effect=AssertionError("no execution"))
        self.guard.start()

    def tearDown(self):
        self.guard.stop()
        self.tmp.cleanup()

    def test_labels_groups_and_lists(self):
        recs = {r["mask"]: r for r in audit.audit_run(write_dataset(self.root / "d"), 1e-6)}
        self.assertTrue(all(not r["problems"] for r in recs.values()), [r["problems"] for r in recs.values()])
        self.assertEqual({m: r["bucket"] for m, r in recs.items()},
                         {0: "uncorrupted", 1: "exception", 2: "improvement_expected", 3: "small", 4: "noise_only",
                          5: "cancelling", 6: "improvement_expected", 7: "cancelling"})
        self.assertEqual(recs[7]["negative"], ["d_amp", "d_phase", "D"])          # overlapping negative labels
        self.assertEqual(recs[5]["tiny_negative"], ["d_amp", "D"])
        lists = list(audit.unexpected(list(recs.values())).values())
        self.assertEqual([r["mask"] for r in lists[0]], [1])                        # single-kind negative
        self.assertEqual([r["mask"] for r in lists[1]], [1])                        # gain-only negative D
        self.assertEqual(lists[2], [])                                              # control/noise D = 0
        self.assertEqual([r["mask"] for r in lists[3]], [3])                        # negative d_amp, D >= 0
        t = {r["mixture"]: r for r in audit.table(list(recs.values()))}
        self.assertEqual(len(t), 8)
        self.assertEqual((t["amp+phase+noise"]["negative_d_amp"], t["amp+phase+noise"]["negative_D"],
                          t["amp+phase+noise"]["any_negative_active"]), (1, 1, 1))  # unique case counted once
        self.assertEqual(t["amp+noise"]["tiny_negative"], 1)
        v = audit.verdict(list(recs.values()))
        self.assertTrue(v["reconciled"]); self.assertTrue(v["needs_investigation"])
        self.assertEqual((v["expected_noise_gain_joint_negatives"], v["unexpected_gain_only_joint_negatives"]), (2, 1))

    def test_sparse_invalid_and_missing(self):
        root = write_dataset(self.root / "d", sparse=(6,), seeds={4: 8}, drop_eC=(2,))
        recs = {r["mask"]: r for r in audit.audit_run(root, 1e-6)}
        self.assertIsNone(recs[6]["stored_d_phase"])                     # sparse stored label: recomputed, not zero
        self.assertIsNone(recs[2]["eC"]); self.assertEqual(recs[2]["bucket"], "unavailable")
        self.assertIsNone(recs[3]["d_amp"])                              # counterpart c2 has no eC: unavailable
        self.assertIsNone(recs[5]["D"]); self.assertEqual(recs[5]["bucket"], "unavailable")   # noise seed differs
        self.assertTrue(any("noise_seed" in p for p in recs[5]["problems"]))
        self.assertEqual(recs[4]["bucket"], "noise_only")                # metadata still establishes the group
        self.assertTrue(audit.verdict(list(recs.values()))["reconciled"])

    def test_stored_label_mismatch_and_dedup(self):
        root = write_dataset(self.root / "d")
        stored = json.loads((root / "rows.json").read_text())
        stored["rows"][3]["d"][0] += 1e-3
        (root / "rows.json").write_text(json.dumps(stored))
        recs = {r["mask"]: r for r in audit.audit_run(root, 1e-6)}
        self.assertTrue(any("stored d_amp" in p for p in recs[3]["problems"]))
        out = self.root / "out"
        with patch.object(sys, "argv", ["audit", "--output", str(out), str(root), str(root)]):
            audit.main()
        result = json.loads((out / "audit.json").read_text())
        self.assertEqual((result["duplicates"], result["verdict"]["cases"]), (8, 8))
        self.assertEqual(sum(r["total"] for r in result["tables"]["combined"]), 8)
        self.assertTrue((out / "cases.csv").exists() and (out / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
