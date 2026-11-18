"""Pure decision logic of the VLA pipeline QA experiment (no CASA): flag bookkeeping, B evaluation decision,
result unwrapping, plan option checks and recovery outcomes. Pipeline end-to-end checks live in
scripts/vla_pipeline_qa_checks.py (they need the CASA pipeline and take minutes)."""
import unittest

import numpy as np

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

    def test_my_criteria(self):
        def res(before, after, disposition="applied", reused=False, status="measured"):
            return dict(recipe_record=dict(disposition=disposition, pipeline_notes=["REASON: S/N decrease"] if disposition == "not_applied" else []), evaluation=dict(
                status=status, evaluation_reused=reused, before=dict(eC=before), after=dict(eC=after)))
        cat = lambda r, gain, ref=.10: q.my_criteria(r, gain, dict(eC=ref))["category"]
        # Design doc example table: reference 0.10, boundary 0.15, epsilon 0.005
        for before, after, disposition, expected in (
                (.12, .121, "applied", "preserved"), (.12, .121, "not_applied", "preserved"),
                (.12, .10, "applied", "improved"), (.20, .14, "applied", "solved"), (.20, .17, "applied", "improved"),
                (.12, .14, "applied", "degraded"), (.20, .20, "not_applied", "rejected"),
                (.20, .201, "applied", "unchanged"), (.151, .149, "applied", "unchanged")):
            self.assertEqual(cat(res(before, after, disposition), True), expected, (before, after, disposition))
        self.assertEqual(cat(res(.20, .14, "not_applied"), True), "improved")       # not solved without application
        self.assertEqual(cat(res(.20, .14), False, .20), "improved")                # no gain injection: never solved
        self.assertEqual(cat(res(.2, .2, "not_applied", True), False, .2), "preserved")  # noisy control
        self.assertEqual(cat(res(.2, .2), False, .2), "preserved")        # applied, no effect, pure noise
        self.assertEqual(cat(res(.2, .2, "not_run"), True), "unavailable")          # unknown action: no guess
        self.assertEqual(cat(res(.3, .1, "runtime_failure", status="unavailable"), True), "error")
        self.assertEqual(cat(res(.3, float("nan")), True), "unavailable")
        self.assertEqual(q.my_criteria(res(.3, .1), True, dict(reason="x"))["category"], "unavailable")

    def test_blocked_and_error_precedence(self):
        for before in (.01, .2):
            r = dict(recipe_record=dict(disposition="not_applied", selfcal_result=dict(targets=[dict(
                sc_lib=dict(Stop_Reason="Empty model for solint inf_EB"))])),
                evaluation=dict(status="measured", before=dict(eC=before), after=dict(eC=before)))
            self.assertEqual(q.my_criteria(r, True, dict(eC=0))["category"], "blocked")
            r["recipe_record"]["disposition"] = "runtime_failure"
            self.assertEqual(q.my_criteria(r, True, dict(eC=0))["category"], "error")
        r["recipe_record"] = dict(disposition="not_applied")
        self.assertEqual(q.my_criteria(r, True, dict(eC=.2))["category"], "unavailable")

    def test_structured_rejection_and_rollback(self):
        rec = dict(disposition="not_applied", selfcal_result=dict(targets=[dict(sc_lib=dict(
            Stop_Reason="S/N decrease", final_solint="None", inf_EB_SNR_decrease=True,
            ms={"inf_EB": dict(Pass=False, Fail_Reason="S/N decrease with no successful solints later")}))]))
        self.assertEqual(q.pipeline_action(rec), "Left unchanged")
        self.assertTrue(q.decision_evidence(rec)["rollback"])
        rec["selfcal_result"]["targets"][0]["sc_lib"]["Stop_Reason"] = "Empty model for solint inf"
        self.assertEqual(q.pipeline_action(rec), "Left unchanged")

    def test_matched_reference(self):
        case = lambda cid, A=1., phi=0., a=1.: dict(id=cid, A=A, phi_deg=phi, a=a, source="s", slot=0, noise_seed=7)
        row = lambda cid, mask, recipe: dict(case=cid, id=cid, mask=mask, recipe=recipe, slot=0)
        sp = dict(rows=[row("c", 0, None), row("g", 3, "r"), row("n", 4, "r"), row("gn", 7, "r")],
                  cases=[case("c"), case("g", 2., 10.), case("n", a=5.), case("gn", 2., 10., 5.)])
        eC = dict(c=.01, g=.5, n=.2, gn=.6).get
        self.assertEqual(q.matched_reference(sp, "g", eC)["id"], "c")    # gain-only -> nominal-noise control
        self.assertEqual(q.matched_reference(sp, "gn", eC)["id"], "n")   # gain+noise -> matched noise-only case
        self.assertEqual(q.matched_reference(sp, "n", eC)["eC"], .2)     # no gain -> itself
        sp["cases"][2]["noise_seed"] = 8
        self.assertEqual(q.matched_reference(sp, "gn", eC)["status"], "unavailable")

    def test_pipeline_action(self):
        self.assertEqual(q.pipeline_action(dict(disposition="applied")), "Applied")
        self.assertEqual(q.pipeline_action(dict(disposition="not_applied", pipeline_notes=["No regions found by automasking"])),
                         "Could not start")
        self.assertEqual(q.pipeline_action(dict(disposition="not_applied", pipeline_notes=[   # hif_selfcal empty mask
            "The mask file sc.X_dirty.mask is empty.", "Starting with solint: inf_EB", "Final solint: None",
            "No regions found by automasking"])), "Could not start")
        self.assertEqual(q.pipeline_action(dict(disposition="not_applied", pipeline_notes=[   # trial ran, rejected
            "Starting with solint: inf_EB", "REASON: S/N decrease", "The mask file sc.X_final.mask is empty."])),
            "Left unchanged")
        self.assertEqual(q.pipeline_action(dict(disposition="not_applied", pipeline_notes=[
            "Starting with solint: inf_EB", "REASON: S/N decrease"])), "Left unchanged")
        self.assertEqual(q.pipeline_action(dict(disposition="runtime_failure")), "Error")
        self.assertEqual(q.pipeline_action(dict(disposition="not_run")), "Unknown")

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


class ImageMetricTests(unittest.TestCase):
    def test_restored_metrics_and_pair(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from astropy.io import fits
        rng = np.random.default_rng(1)
        source = np.zeros((32, 32), bool); source[15:17, 15:17] = True
        annulus = np.zeros((32, 32), bool); annulus[:6] = True
        masks = dict(source=source, annulus=annulus)
        header = dict(BUNIT="Jy/beam", BMAJ=2e-4, BMIN=1e-4, BPA=0., CDELT1=-4e-5, CDELT2=4e-5, CRVAL1=1., CRVAL2=2.,
                      CRPIX1=17., CRPIX2=17.)

        def image(path, noise, peak, **change):
            data = rng.normal(0, noise, (32, 32)); data[16, 16] = peak
            hdu = fits.PrimaryHDU(data[None, None]); hdu.header.update({**header, **change})
            hdu.writeto(path)
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", side_effect=AssertionError("no execution")):
            b, a, other = (Path(tmp) / n for n in ("b.fits", "a.fits", "o.fits"))
            image(b, .01, 1.); image(a, .001, 1.); image(other, .001, 1., BMAJ=3e-4)
            m = q.restored_metrics(b, masks)
            off = fits.getdata(b).squeeze()[annulus]
            self.assertAlmostEqual(m["rms"], float(np.std(off)))
            self.assertAlmostEqual(m["snr"], (1. - off.mean()) / np.std(off))
            pair = q.image_pair(b, a, masks)
            self.assertEqual(pair["image"], "cleaner")
            self.assertGreater(pair["r_rms"], 0); self.assertGreater(pair["r_snr"], 0)
            self.assertEqual(q.image_pair(a, b, masks)["image"], "worse")
            bad = q.image_pair(b, other, masks)
            self.assertEqual(bad["image"], "unavailable"); self.assertIn("BMAJ", bad["image_reason"])
            self.assertEqual(q.image_pair(b, Path(tmp) / "missing.fits", masks)["image"], "unavailable")

    def _fits(self, path, noise, peak, header, drop=()):
        from astropy.io import fits
        data = np.random.default_rng(2).normal(0, noise, (32, 32)); data[16, 16] = peak
        hdu = fits.PrimaryHDU(data[None, None]); hdu.header.update(header)
        for k in drop:
            del hdu.header[k]
        hdu.writeto(path)

    def test_image_rejects_missing_metadata_and_nonpositive_snr(self):
        import tempfile
        from pathlib import Path
        source = np.zeros((32, 32), bool); source[15:17, 15:17] = True
        annulus = np.zeros((32, 32), bool); annulus[:6] = True
        masks = dict(source=source, annulus=annulus)
        header = dict(BUNIT="Jy/beam", BMAJ=2e-4, BMIN=1e-4, BPA=0., CDELT1=-4e-5, CDELT2=4e-5, CRVAL1=1., CRVAL2=2.,
                      CRPIX1=17., CRPIX2=17.)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            self._fits(t / "b.fits", .01, 1., header); self._fits(t / "a.fits", .001, 1., header)
            self._fits(t / "nb.fits", .01, 1., header, drop=("BUNIT", "BMAJ")); self._fits(t / "na.fits", .001, 1., header, drop=("BUNIT", "BMAJ"))
            self._fits(t / "neg.fits", .001, -1., header)
            missing = q.image_pair(t / "nb.fits", t / "na.fits", masks)          # missing on both: not a match
            self.assertEqual(missing["image"], "unavailable")
            self.assertIn("BUNIT", missing["image_reason"]); self.assertIn("BMAJ", missing["image_reason"])
            neg = q.image_pair(t / "b.fits", t / "neg.fits", masks)
            self.assertEqual(neg["image"], "unavailable"); self.assertIn("after", neg["image_reason"])
            self.assertEqual(q.image_pair(t / "b.fits", t / "a.fits", masks)["image"], "cleaner")
            loose = q.image_pair(t / "b.fits", t / "a.fits", masks, eps_rms=1e9, eps_snr=1e9)   # configurable
            self.assertEqual((loose["image"], loose["epsilon_rms"]), ("similar", 1e9))


class V5GroupTests(unittest.TestCase):
    """[USER] 2026-11-16 initial groups: confirmed precedence and exact threshold edges."""

    def bucket(self, mask, before, reference, tol=q.TOL_REFERENCE):
        g = q.initial_group(mask, before, reference, tol)
        return g["subgroup"] or g["group"]

    def test_precedence_and_edges(self):
        b = self.bucket
        self.assertEqual(b(0, None, None), "uncorrupted")                  # metadata alone, metrics missing
        self.assertEqual(b(4, .9, .9), "noise_only")                       # any amount of noise, D = 0
        self.assertEqual(b(4, None, None), "noise_only")
        self.assertEqual(b(5, .2, .2001), "cancelling")                    # any negative D, even tiny
        self.assertEqual(b(7, .2, .6), "cancelling")                       # large cancellation
        self.assertEqual(b(6, .2, .3), "cancelling")
        self.assertEqual(b(3, .2, .3), "exception")                        # gain only, D < 0: not Small/Cancelling
        self.assertEqual(b(1, .2, .3), "exception")
        self.assertEqual(b(3, .2, .2), "small")                            # D = 0
        self.assertEqual(b(5, .75, .5, tol=.25), "small")                  # D = tol exactly (exact binary numbers)
        self.assertEqual(b(5, .75, .5, tol=.125), "improvement_expected")  # D > tol
        self.assertEqual(b(2, .3, .1), "improvement_expected")
        self.assertEqual(b(3, None, .1), "unavailable")
        self.assertEqual(b(3, .3, float("nan")), "unavailable")
        self.assertEqual(b(None, .3, .1), "unavailable")                   # unknown metadata cannot establish a group

    def test_exclusive_and_exhaustive(self):
        known = {"improvement_expected", "uncorrupted", "noise_only", "cancelling", "small", "exception", "unavailable"}
        for mask in range(8):
            for D in (-.5, -1e-9, 0., 1e-9, .05, .0500001, .5):
                g = q.initial_group(mask, .5 + D, .5)
                self.assertIn(g["subgroup"] or g["group"], known)
                self.assertEqual(g["subgroup"] is not None, g["group"] == "no_improvement_expected")


class V5RecoveryTests(unittest.TestCase):
    def test_boundaries(self):
        r = lambda before, after, ref, t=.25: q.recovery_outcome(before, after, ref, t, t)["recovery"]
        self.assertEqual(r(.5, .75, 0.), "similar")            # change = +tol exactly
        self.assertEqual(r(.75, .5, .5), "similar")            # change = -tol exactly
        self.assertEqual(r(.5, .875, 0.), "degraded")
        self.assertEqual(r(1., .5, .25), "solved")              # remaining = tol exactly
        self.assertEqual(r(1., .5, .125), "improved")          # remaining > tol
        self.assertEqual(r(1., .125, .5), "solved")            # better than the reference
        self.assertEqual(r(.5, .875, None), "degraded")        # no reference: degraded/similar still known
        self.assertEqual(r(.5, .5, None), "similar")
        x = q.recovery_outcome(1., .5, None, .25, .25)
        self.assertEqual(x["recovery"], "unavailable"); self.assertEqual(x["change"], -.5)   # reduction kept
        self.assertEqual(r(None, .5, .1), "unavailable")
        d = q.recovery_outcome(.3, .2, .1)
        self.assertAlmostEqual(d["change"], -.1); self.assertAlmostEqual(d["remaining"], .1)

    def test_roundoff_at_default_boundaries(self):
        # 0.20 - 0.15 = 0.05000000000000002 in floating point: AT the 0.05 tolerance, not beyond it
        self.assertEqual(q.recovery_outcome(.15, .20, .15)["recovery"], "similar")
        self.assertEqual(q.recovery_outcome(.20, .15, .10)["recovery"], "similar")
        self.assertEqual(q.recovery_outcome(.30, .15, .10)["recovery"], "solved")      # remaining .05 (roundoff)
        self.assertEqual(q.recovery_outcome(.15, .2001, .15)["recovery"], "degraded")
        self.assertEqual(q.initial_group(3, .20, .15)["subgroup"], "small")
        self.assertEqual(q.initial_group(3, .2001, .15)["group"], "improvement_expected")
        self.assertEqual(q.initial_group(5, .15, .15 + 1e-15)["subgroup"], "cancelling")   # sign test stays exact
        self.assertEqual(q.image_outcome(1 - .95, 0), "similar")                         # 0.050000000000000044

    def test_solved_priority_for_improvement_expected(self):
        imp = "improvement_expected"
        r = lambda before, after, ref, group=imp: q.recovery_outcome(before, after, ref, group=group)["recovery"]
        # doc regression example: D = 0.0666266442, final excess 0.0171296447, reduction 0.0494969994 < 0.05
        ref = .1
        self.assertEqual(r(ref + .0666266442, ref + .0171296447, ref), "solved")
        self.assertEqual(q.pipeline_result(imp, r(ref + .0666266442, ref + .0171296447, ref)), "success")
        # D = 0.242335299, remaining 0.192734096 after a 0.049601203 reduction: still outside -> Similar, Failure
        self.assertEqual(r(ref + .242335299, ref + .192734096, ref), "similar")
        self.assertEqual(r(.30, .15, .10), "solved")                          # remaining exactly 0.05 (roundoff)
        self.assertEqual(r(.17, .15 + 1e-6, .10), "similar")                  # just outside, small reduction
        for group in ("no_improvement_expected", "exception", "unavailable", None):    # other groups unchanged
            self.assertEqual(r(ref + .0666266442, ref + .0171296447, ref, group), "similar")
        self.assertEqual(r(.12, .12, .10, "no_improvement_expected"), "similar")      # already within: not Solved
        self.assertEqual(r(.30, .15, None), "unavailable")                    # no reference: no inferred crossing
        x = q.v5_labels(3, .1666, .1, dict(recipe_record=dict(disposition="applied"), evaluation=dict(
            status="measured", before=dict(eC=.1666), after=dict(eC=.12))))
        self.assertEqual((x["group"], x["recovery"], x["result"]), (imp, "solved", "success"))

    def test_recovery_progress(self):
        pr = lambda b, a, ref: q.recovery_progress(b, a, ref)
        self.assertEqual(pr(.5, .5, .25)["progress_pct"], 0.)                 # unchanged
        self.assertEqual(pr(.5, .25, .25)["progress_pct"], 100.)              # fully recovered
        self.assertEqual(pr(.5, .75, .25)["progress_pct"], -100.)             # degraded: negative, not clamped
        self.assertEqual(pr(.5, .125, .25)["progress_pct"], 150.)             # below the reference: > 100
        self.assertAlmostEqual(pr(.5, .25, .25)["progress_required_pct"], 100 * (1 - .05 / .25))
        self.assertEqual(pr(.25, .2, .25)["progress_status"], "not_applicable")   # D = 0: no division
        self.assertEqual(pr(.2, .1, .25)["progress_status"], "not_applicable")    # D < 0: no abs(D)
        tiny = pr(.25 + 1e-6, .25, .25)
        self.assertAlmostEqual(tiny["progress_pct"], 100.); self.assertTrue(tiny["progress_small_D"])
        self.assertAlmostEqual(pr(.25 + 1e-6, .26, .25)["progress_pct"], -999900., places=0)   # extreme, kept
        self.assertEqual(pr(None, .2, .1)["progress_status"], "missing")
        self.assertEqual(pr(.3, .2, float("nan"))["progress_status"], "missing")
        self.assertEqual(q.v5_labels(3, .3, .1, None, finished=False)["progress_status"], "missing")

    def test_pipeline_result_mappings(self):
        p = q.pipeline_result
        imp, no = "improvement_expected", "no_improvement_expected"
        self.assertEqual([p(imp, k) for k in ("solved", "improved", "similar", "degraded")],
                         ["success", "partial_success", "failure", "failure"])
        self.assertEqual([p(no, k) for k in ("solved", "improved", "similar", "degraded")],
                         ["success", "success", "success", "failure"])
        self.assertEqual(p("exception", "solved"), "unavailable")
        self.assertEqual(p("unavailable", "similar"), "unavailable")
        self.assertEqual(p(imp, "unavailable"), "unavailable")
        self.assertEqual((p(imp, "pending"), p(no, "error")), ("pending", "error"))

    def test_labels_accounting(self):
        self.assertEqual(q.v5_labels(3, .3, .1, None, finished=False)["result"], "pending")
        self.assertEqual(q.v5_labels(3, .3, .1, None, finished=True)["application"], "error")
        res = dict(recipe_record=dict(disposition="applied"),
                   evaluation=dict(status="measured", before=dict(eC=.3), after=dict(eC=.12)))
        x = q.v5_labels(3, .3, .1, res)
        self.assertEqual((x["group"], x["application"], x["recovery"], x["result"]),
                         ("improvement_expected", "applied", "solved", "success"))
        res = dict(recipe_record=dict(disposition="runtime_failure"), evaluation=dict(status="failed"))
        x = q.v5_labels(3, .3, .1, res)
        self.assertEqual((x["application"], x["recovery"], x["result"]), ("error", "error", "error"))


class V5ImageTests(unittest.TestCase):
    def test_nine_combinations_and_edges(self):
        up, same, down = .06, 0., -.06              # benefit-oriented: positive r = better, tolerance 5%
        table = {(up, up): "cleaner", (up, same): "cleaner", (same, up): "cleaner", (same, same): "similar",
                 (down, down): "worse", (down, same): "worse", (same, down): "worse", (up, down): "mixed",
                 (down, up): "mixed"}
        for (r_rms, r_snr), want in table.items():
            self.assertEqual(q.image_outcome(r_rms, r_snr), want, (r_rms, r_snr))
        self.assertEqual(q.EPSILON_RMS, .05)
        self.assertEqual(q.image_outcome(.05, -.05), "similar")              # exactly at the tolerance
        self.assertEqual(q.image_outcome(.03, 0, .01, .01), "cleaner")       # 1% would call 3% cleaner
        self.assertEqual(q.image_outcome(.03, 0), "similar")
        self.assertEqual(q.image_outcome(None, 0), "unavailable")
        self.assertEqual(q.image_label(dict(r_rms=.5, r_snr=.5, image_reason="x"))["image"], "unavailable")

    def test_invalid_tolerances(self):
        for bad in (-.01, float("nan"), float("inf"), None, "0.05"):
            with self.assertRaises(ValueError):
                q.check_tolerance("eps", bad)
        self.assertEqual(q.check_tolerance("eps", 0), 0.)


class V5RunTests(unittest.TestCase):
    def lib(self, **kw):
        return dict(selfcal_result=dict(targets=[dict(sc_lib=kw)]))

    def test_run_outcomes(self):
        r = q.run_outcome
        # Applied, a later interval failed: still Applied
        rec = dict(disposition="applied", **self.lib(final_solint="inf_EB", Stop_Reason="S/N decrease",
                                                     inf=dict(Pass=False, Fail_Reason="S/N decrease")))
        self.assertEqual(r(rec)["application"], "applied")
        self.assertIn("S/N decrease", r(rec)["stop_reasons"])
        # complete rejection of the first trial
        rec = dict(disposition="not_applied", **self.lib(Stop_Reason=" S/N decrease; All sub-fields failed",
                                                         final_solint="None",
                                                         ms={"inf_EB": dict(Pass=False, Fail_Reason="S/N decrease")}))
        self.assertEqual((r(rec)["application"], r(rec)["blocked_reason"]), ("blocked", "solution_lowers_snr"))
        # provisional first correction revoked after a later failure
        rec = dict(disposition="not_applied", **self.lib(Stop_Reason="S/N decrease", final_solint="None",
                   inf_EB_SNR_decrease=True,
                   ms={"inf_EB": dict(Pass=False, Fail_Reason="S/N decrease with no successful solints later")}))
        self.assertEqual(r(rec)["blocked_reason"], "solution_lowers_snr")
        # initial low S/N: no trial
        rec = dict(disposition="not_applied", **self.lib(Stop_Reason="Estimated_SNR_too_low_for_solint inf_EB",
                                                         final_solint="None"))
        self.assertEqual(r(rec)["blocked_reason"], "low_initial_snr")
        # empty mask (log only) is not low S/N
        rec = dict(disposition="not_applied", pipeline_notes=["No regions found by automasking"])
        self.assertEqual(r(rec)["blocked_reason"], "empty_model")
        # RMS increase only: another reason, not an S/N decrease
        rec = dict(disposition="not_applied", **self.lib(Stop_Reason="RMS increase beyond 5%", final_solint="None",
                   ms={"inf_EB": dict(Pass=False, Fail_Reason="RMS increase beyond 5%")}))
        self.assertEqual(r(rec)["blocked_reason"], "other")
        self.assertEqual(r(dict(disposition="no_valid_solutions"))["blocked_reason"], "other")
        self.assertEqual(r(dict(disposition="not_applied"))["application"], "unknown")
        self.assertEqual(r(dict(disposition="runtime_failure"))["application"], "error")
        # QA scoring warning alone is not an execution failure
        rec = dict(disposition="applied", pipeline_notes=["QA execution failed with an exception: 'list' ..."])
        self.assertEqual(r(rec)["application"], "applied")


class ElapsedTests(unittest.TestCase):
    def test_killed_call_counts_until_last_case(self):
        from scripts.vla_pipeline_qa_compare import elapsed_hours
        calls = [dict(started=0.), dict(started=10000., finished=10000. + 3600.)]     # first call killed
        units = [1800., 3600., 10100.]                                                 # cases it completed
        self.assertAlmostEqual(elapsed_hours(calls, units), 2.)
        self.assertAlmostEqual(elapsed_hours([dict(started=0.)], []), 0.)
        self.assertIsNone(elapsed_hours([], []))


def case_item(run, i, group, subgroup, result, recovery, image, application, reason=None, progress=None):
    status = "applicable" if progress is not None else "not_applicable" if group != "improvement_expected" else "missing"
    return dict(progress_status=status, progress_pct=progress, progress_required_pct=50. if progress is not None else None,
                progress_small_D=False, eC_before=.3, eC_after=.2,
                case=f"c{i}", id=f"{run}|c{i}|B", label="s:r00:m3:slot0", corruption="amp+phase", mask=3, d=None,
                eC_archived=.3, group=group, subgroup=subgroup, group_reason="x", D=.1, reference=.1,
                application=application, blocked_reason=reason, recovery=recovery, recovery_reason=None,
                result=result, image=image, change=-.1, remaining=.1, description="d", seconds=60., action="Applied",
                legacy=dict(category="improved"))


class V5ReportTests(unittest.TestCase):
    CONFIG = "B · 256 px · center · flags on"

    def _rows(self):
        specs = [("improvement_expected", None, "success", "solved", "cleaner", "applied", None, 120.),
                 ("improvement_expected", None, "partial_success", "improved", "mixed", "applied", None, 60.),
                 ("improvement_expected", None, "failure", "similar", "similar", "blocked", "low_initial_snr", 2.),
                 ("improvement_expected", None, "failure", "similar", "similar", "applied", None, 2.),   # duplicate value
                 ("no_improvement_expected", "noise_only", "success", "similar", "cleaner", "blocked", "solution_lowers_snr"),
                 ("no_improvement_expected", "cancelling", "failure", "degraded", "worse", "applied", None),
                 ("no_improvement_expected", "uncorrupted", "success", "similar", "similar", "blocked", "other"),
                 ("no_improvement_expected", "small", "unavailable", "unavailable", "unavailable", "unknown", None, -40.),
                 ("exception", None, "unavailable", "improved", "cleaner", "applied", None),
                 ("improvement_expected", None, "pending", "pending", "pending", "pending", None)]
        # two different runs with the SAME folder name and configuration: identity is the absolute path
        return [dict(run="qa-run", run_id=f"/data/{where}/qa-run", tag="qa-run", config=self.CONFIG, key=where,
                     recipe="B:phase_amp:flagging_on",
                     cases=[case_item(f"/data/{where}/qa-run", i, *s) for i, s in enumerate(specs)], calls=[],
                     superseded=False)
                for where in ("a", "b")]

    def test_every_bar_selects_exactly_its_cases(self):
        import html, json, re, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import vla_pipeline_qa_compare as cmp
        rows = self._rows()
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", side_effect=AssertionError("no execution")):
            summaries = [cmp.summarize(r) for r in rows]
            cmp.write_report(Path(tmp), summaries, rows, [Path("run1"), Path("run2")], "r.html")
            text = (Path(tmp) / "r.html").read_text()
        trs = [{k: html.unescape(v) for k, v in re.findall(r"data-([a-z]+)='([^']*)'", tr)}
               for tr in re.findall(r"<tr data-run=[^>]*>", text)]
        self.assertEqual(len(trs), 20)
        bars = re.findall(r"class=bar [^>]*data-filter='([^']*)'><span class=lab>[^<]*</span><span[^>]*></span>"
                          r"<span>(\d+)", text)
        self.assertGreater(len(bars), 50)
        keys = set()
        for flt, shown in bars:                                   # exact-match semantics of the page script
            flt = json.loads(html.unescape(flt))
            self.assertIn("run", flt); self.assertIn("config", flt)
            keys |= set(flt)
            hit = [t["id"] for t in trs if all(t.get(k) == v for k, v in flt.items())]
            self.assertEqual(len(hit), int(shown), flt)
            self.assertTrue(flt["run"].startswith("/data/"))
        self.assertTrue({"group", "subgroup", "result", "recovery", "image", "application", "reason"} <= keys)
        for s in summaries:                                        # group/subgroup totals reconcile
            self.assertEqual(sum(t["planned"] for t in s["table2"].values()), s["table1"]["planned"])
            self.assertEqual(sum(t["planned"] for t in s["table3"].values()),
                             s["table2"]["no_improvement_expected"]["planned"])
            self.assertEqual(s["table3"]["small"]["planned"], 1)
            self.assertIn("exception", s["table2"])

    def test_progress_plots_one_image_per_row(self):
        import re, tempfile
        from pathlib import Path
        from scripts import vla_pipeline_qa_compare as cmp
        rows = self._rows()
        summaries = [cmp.summarize(r) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            cmp.write_report(Path(tmp), summaries, rows, [Path("run1"), Path("run2")], "r.html")
            text = (Path(tmp) / "r.html").read_text()
            cells = re.findall(r"<td class=progress>(.*?)</td>", text, re.S)
            imgs = [m for c in cells for m in re.findall(r"<img class=prog src='([^']+)'", c)]
            self.assertTrue(imgs and all((Path(tmp) / i).read_bytes()[:4] == b"\x89PNG" for i in imgs))
        self.assertEqual(len(cells), 2 * (1 + 3 + 4))                      # every row of Tables 1-3, both runs
        self.assertEqual(len(imgs), len(set(imgs)))                         # same-named runs do not collide
        self.assertIn(cmp.NO_PROGRESS, " ".join(cells))                     # noise only / cancelling / uncorrupted
        self.assertIn("plotted 5 / 10 · not applicable 4 · missing/invalid 1", cells[0])
        st = summaries[0]["table1"]["progress"]
        self.assertEqual((st["plotted"], st["total"], st["not_applicable"], st["missing"]), (5, 10, 4, 1))
        for dim, cats in (("result", q.RESULTS), ("recovery", q.RECOVERY)):   # every case once per panel, plus All
            groups = cmp.progress_groups(st["points"], dim, cats)
            self.assertEqual(groups[0], ("all", st["points"]))
            self.assertEqual(sorted(p["id"] for _, pts in groups[1:] for p in pts), sorted(p["id"] for p in st["points"]))
        self.assertEqual([k for k, _ in cmp.progress_groups(st["points"], "result", q.RESULTS)],
                         ["all", "success", "partial_success", "failure", "unavailable"])
        lo, hi = cmp.progress_range(summaries)
        self.assertTrue(lo < -40 and hi > 120)                              # nothing clipped, 0 and 100 included

    def test_tolerance_change_never_reuses_labels(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import vla_pipeline_qa_compare as cmp
        raw = dict(r_rms=.03, r_snr=0., image_reason=None, image_metric_version=q.IMAGE_METRIC_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for name in ("b.fits", "a.fits"):
                (t / name).write_text("x")
            np.savez(t / "m.npz", annulus=np.ones((2, 2), bool), source=np.ones((2, 2), bool))
            rows = [dict(cases=[dict(application="applied", clean_before=str(t / "b.fits"), clean_after=str(t / "a.fits"),
                                     masks=str(t / "m.npz"))])]
            with patch.object(cmp, "image_measure", return_value=dict(raw)) as measure:
                cmp.add_image_metrics(rows, t / "cache", .01, .01)
                self.assertEqual(rows[0]["cases"][0]["image"], "cleaner")
                cmp.add_image_metrics(rows, t / "cache", .05, .05)
                self.assertEqual(rows[0]["cases"][0]["image"], "similar")
                self.assertEqual(measure.call_count, 1)            # raw metrics cached, label recomputed

    def test_evidence_written_after_image_metrics(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import vla_pipeline_qa_compare as cmp
        rows = self._rows()[:1]
        rows[0]["cases"][0].update(image=None, evidence=dict(recipe_record={}, evaluation={}, reference={}))

        def measure(rows, cache, *a):
            rows[0]["cases"][0].update(image="cleaner", rms_before=2., rms_after=1., snr_before=10., snr_after=20.)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cmp, "load", return_value=rows), \
                patch.object(cmp, "add_image_metrics", side_effect=measure):
            cmp.build([Path("run1")], Path(tmp), "r.html")
            ev = json.loads(next((Path(tmp) / "descriptions").glob("*.json")).read_text())
        self.assertEqual((ev["v5"]["image"], ev["v5"]["rms_after"], ev["v5"]["snr_after"]), ("cleaner", 1., 20.))
        self.assertEqual(ev["thresholds"]["tol_change"], .05)
        self.assertEqual(ev["legacy_v3"]["category"], "improved")
        with self.assertRaises(ValueError):
            cmp.build([Path("run1")], Path("unused"), "r.html", eps_rms=-1.)


if __name__ == "__main__":
    unittest.main()
