"""Pure decision logic of the VLA pipeline QA experiment (no CASA): flag bookkeeping, B evaluation decision,
result unwrapping, plan option checks and recovery outcomes. Pipeline end-to-end checks live in
scripts/vla_pipeline_qa_checks.py (they need the CASA pipeline and take minutes)."""
import unittest

from scripts import vla_pipeline_qa as q


def summary(name, flagged, total=100.):
    return dict(name=name, flagged=flagged, total=total)


def app(vis="pipe_in.ms", calwt=False, column="CORRECTED_DATA"):
    return dict(vis=vis, any_calwt=calwt, registered_column=column)


UNCHANGED = dict(stage="hif_selfcal applycal", status="unchanged", added=0.)
ADDED = dict(stage="hif_selfcal applycal", status="added", added=5.)
UNKNOWN = dict(stage="hifv_flagtargetsdata", status="unknown", reason="missing summary")
SKIPPED = dict(stage="hifv_flagtargetsdata", status="skipped")


class FlagStageTests(unittest.TestCase):
    def test_unchanged_added(self):
        self.assertEqual(q.flag_stage("s", [summary("before", 10), summary("applycal", 10)], "before", "applycal")["status"],
                         "unchanged")
        st = q.flag_stage("s", [summary("before", 10), summary("applycal", 12)], "before", "applycal")
        self.assertEqual((st["status"], st["added"]), ("added", 2))

    def test_unknown_never_unchanged(self):
        for summaries in ([summary("before", 10)],                                   # missing after
                          [summary("before", 10), summary("applycal", 5)],           # decrease
                          [summary("before", 10), summary("applycal", 10, 99.)],     # different scope
                          [dict(name="before"), summary("applycal", 10)],            # malformed
                          None):
            self.assertEqual(q.flag_stage("s", summaries, "before", "applycal")["status"], "unknown", summaries)

    def test_named_not_positional_and_wrapped(self):
        st = q.flag_stage("s", [{"report1": summary("template", 3)}, {"report0": summary("before", 3)}],
                          "before", "template")
        self.assertEqual(st["status"], "unchanged")

    def test_all_flagged(self):
        self.assertTrue(q.flag_stage("s", [summary("before", 10), summary("applycal", 100)], "before", "applycal")["all_flagged"])


class DecisionTests(unittest.TestCase):
    """The reuse branch: no application, no added or unknown flags, no calwt -> after = before, no imaging."""

    def test_reuse_only_when_nothing_changed(self):
        d = q.decide_b([], [], [SKIPPED, UNCHANGED], flagging=False)
        self.assertIsNone(d["split"])
        self.assertTrue(d["record"]["evaluation_decision"]["reuse"])
        self.assertEqual(d["record"]["disposition"], "not_applied")

    def test_no_application_but_flags_added_is_imaged(self):
        d = q.decide_b([], [], [ADDED], flagging=True)
        self.assertEqual(d["split"], (None, "DATA"))
        self.assertEqual(d["record"]["evaluation_decision"], dict(reuse=False, dirty_reference="rebuild", reason="flags added"))

    def test_unknown_bookkeeping_is_imaged(self):
        d = q.decide_b([], [], [UNKNOWN, UNCHANGED], flagging=True)
        self.assertFalse(d["record"]["evaluation_decision"]["reuse"])
        self.assertEqual(d["record"]["evaluation_decision"]["dirty_reference"], "rebuild")

    def test_applied_is_imaged_from_registered_column(self):
        d = q.decide_b([app()], [], [UNCHANGED], flagging=False)
        self.assertEqual(d["split"], ("pipe_in.ms", "CORRECTED_DATA"))
        self.assertEqual(d["record"]["evaluation_decision"]["dirty_reference"], "reuse")
        self.assertEqual(d["record"]["disposition"], "applied")

    def test_calwt_rebuilds_reference(self):
        d = q.decide_b([app(calwt=True)], [], [UNCHANGED], flagging=False)
        self.assertEqual(d["record"]["evaluation_decision"]["dirty_reference"], "rebuild")
        self.assertTrue(d["record"]["weights_may_change"])

    def test_flagging_off_violation_labelled(self):
        self.assertTrue(q.decide_b([app()], [], [ADDED], flagging=False)["record"]["flagging_off_violated"])
        self.assertNotIn("flagging_off_violated", q.decide_b([app()], [], [ADDED], flagging=True)["record"])

    def test_inconsistent_records_fail(self):
        self.assertEqual(q.decide_b([app()], ["field mismatch"], [UNCHANGED], False)["record"]["disposition"], "runtime_failure")
        self.assertEqual(q.decide_b([app("a.ms"), app("b.ms")], [], [UNCHANGED], False)["record"]["disposition"],
                         "runtime_failure")

    def test_all_flagged_no_usable_output(self):
        stage = dict(ADDED, all_flagged=True)
        d = q.decide_b([app()], [], [stage], flagging=True)
        self.assertEqual((d["record"]["disposition"], d["split"]), ("no_usable_output", None))


class HelperTests(unittest.TestCase):
    def test_unwrap_results_list(self):
        class R:                      # carries the attribute
            summaries = []

        class ResultsList:            # iterable, not a Python list (like pipeline basetask.ResultsList)
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)
        self.assertEqual(len(q.unwrap(ResultsList([R(), ResultsList([R()])]), "summaries")), 2)
        self.assertEqual(q.unwrap(None, "summaries"), [])
        self.assertEqual(q.unwrap({"summaries": 1}, "summaries"), [])

    def test_imsize(self):
        self.assertEqual(q.check_imsize(512), 512)
        self.assertEqual(q.check_imsize(1260), 1260)
        for bad in (0, 511, 22, -4):
            with self.assertRaises(ValueError):
                q.check_imsize(bad)

    def test_outcome(self):
        e = lambda b, a: dict(before=dict(eC=b), after=dict(eC=a))
        self.assertEqual([q.outcome(e(*x)) for x in ((.1, .01), (.01, .02), (.2, .1), (.01, .2), (.05, .05))],
                         ["fixed", "already_ok", "still_bad", "degraded", "still_bad"])

    def test_mask_usage(self):
        self.assertEqual(q.mask_usage(["tclean( vis='a', usemask='user', mask='x.crtf')", 'usemask="auto-multithresh"',
                                       "no mask here"]), {"user": 1, "auto-multithresh": 1})

    def test_variant(self):
        self.assertIsNone(q.variant_label({"keep_work": True}))
        self.assertEqual(q.variant_label(dict(selfcal_imsize=512, selfcal_masking="center", calonly_model_fix=True)),
                         "imsize=512, masking=center")
        self.assertIn("fix OFF", q.variant_label(dict(selfcal_masking="auto", calonly_model_fix=False)))

    def test_calonly_fix_text(self):
        self.assertEqual(q.CALONLY_GUARD.strip(), 'if applymode == "calflag":')
        self.assertTrue(q.CALONLY_FIX.startswith('if applymode == "calflag" or '))


if __name__ == "__main__":
    unittest.main()
