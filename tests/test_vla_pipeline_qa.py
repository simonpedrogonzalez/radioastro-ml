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


class V4OutcomeTests(unittest.TestCase):
    def test_expectation_groups(self):
        e = q.expectation
        self.assertEqual(e(0, .1, .1)["expectation_reason"], "control")
        self.assertEqual(e(4, .2, .1)["expectation_reason"], "noise_only")          # noise only: never expected
        self.assertEqual(e(5, .09, .1)["expectation_reason"], "negative_joint_gain_damage")
        self.assertEqual(e(3, .1, .1)["expectation_reason"], "zero_joint_gain_damage")
        x = e(6, .1003, .1)
        self.assertEqual((x["expectation"], x["near_zero"]), ("improvement_expected", True))   # any D > 0, flagged
        self.assertEqual(e(1, .3, .1)["expectation"], "improvement_expected")
        self.assertEqual(e(1, None, .1)["expectation"], "expectation_unavailable")
        self.assertEqual(e(1, .3, float("nan"))["expectation"], "expectation_unavailable")

    def test_pipeline_and_recovery(self):
        self.assertEqual([q.pipeline_outcome(a) for a in ("Applied", "Could not start", "Left unchanged", "Error", "Unknown")],
                         ["applied", "blocked", "rejected", "error", "unknown"])
        eps = q.EPSILON
        self.assertEqual(q.recovery_outcome(.1, .1 - eps * 1.01), "improved")
        self.assertEqual(q.recovery_outcome(.1, .1 - eps * .99), "similar")
        self.assertEqual(q.recovery_outcome(.1, .1 + eps * 1.01), "degraded")
        self.assertEqual(q.recovery_outcome(.1, None), "unavailable")

    def test_image_table(self):
        up, same, down = .05, 0., -.05              # benefit-oriented: positive r = better
        table = {(up, up): "cleaner", (up, same): "cleaner", (same, up): "cleaner", (same, same): "similar",
                 (down, down): "worse", (down, same): "worse", (same, down): "worse", (up, down): "mixed",
                 (down, up): "mixed"}
        for (r_rms, r_snr), want in table.items():
            self.assertEqual(q.image_outcome(r_rms, r_snr), want, (r_rms, r_snr))
        self.assertEqual(q.image_outcome(q.EPSILON_RMS, 0), "similar")          # equality is similar
        self.assertEqual(q.image_outcome(None, 0), "unavailable")

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


class ElapsedTests(unittest.TestCase):
    def test_killed_call_counts_until_last_case(self):
        from scripts.vla_pipeline_qa_compare import elapsed_hours
        calls = [dict(started=0.), dict(started=10000., finished=10000. + 3600.)]     # first call killed
        units = [1800., 3600., 10100.]                                                 # cases it completed
        self.assertAlmostEqual(elapsed_hours(calls, units), 2.)
        self.assertAlmostEqual(elapsed_hours([dict(started=0.)], []), 0.)
        self.assertIsNone(elapsed_hours([], []))


class V4ReviewFixTests(unittest.TestCase):
    """Regression tests for the 2026-11-12 v4 review findings."""

    def test_recovery_ignores_missing_reference(self):
        ev = dict(status="measured", before=dict(eC=.3), after=dict(eC=.1))
        self.assertEqual(q.measured_recovery(ev), "improved")
        crit = q.my_criteria(dict(recipe_record=dict(disposition="applied"), evaluation=ev), True,
                             dict(status="unavailable", reason="no matched counterpart"))
        self.assertEqual(crit["category"], "unavailable")        # legacy needs the reference, Recovery does not
        self.assertEqual(q.measured_recovery(dict(status="failed")), "unavailable")

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

    def _rows(self):
        cases = lambda run: [dict(case=f"{run}{i}", id=f"{run}|c{i}|B", label="s:r00:m3:slot0", corruption="amp+phase",
                                  mask=3, action="Applied", expectation="improvement_expected", expectation_reason="x",
                                  D=.1, pipeline="applied", recovery="improved", image="cleaner", category="improved",
                                  description="d", seconds=60.) for i in range(1)]
        return [dict(run=run, config="B · 256 px · center · flags on", key=run, recipe="B:phase_amp:flagging_on",
                     cases=cases(run), calls=[], superseded=False) for run in ("run1", "run2")]

    def test_bar_filters_keep_run_identity(self):
        import html, json, re, tempfile
        from pathlib import Path
        from scripts import vla_pipeline_qa_compare as cmp
        rows = self._rows()
        with tempfile.TemporaryDirectory() as tmp:
            cmp.write_report(Path(tmp), [cmp.summarize(r) for r in rows], rows, [Path("run1"), Path("run2")], "r.html",
                             cmp.summarize_v4(rows))
            text = (Path(tmp) / "r.html").read_text()
        trs = [dict(re.findall(r"data-([a-z]+)='([^']*)'", tr)) for tr in re.findall(r"<tr data-run=[^>]*>", text)]
        trs = [{k: html.unescape(v) for k, v in t.items()} for t in trs]
        self.assertEqual(len(trs), 2)
        bars = re.findall(r"class=bar [^>]*data-filter='([^']*)'><span class=lab>[^<]*</span><span[^>]*></span><span>(\d+)",
                          text)
        self.assertTrue(bars)
        for flt, shown in bars:                                   # exact-match semantics of the page script
            flt = json.loads(html.unescape(flt))
            self.assertIn("run", flt)
            hit = sum(all(t.get(k) == v for k, v in flt.items() if v != "other") for t in trs)
            self.assertEqual(hit, int(shown), flt)
        self.assertIn('data-filter=\'{&quot;expectation&quot;: &quot;improvement_expected&quot;}\'', text)

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
        self.assertEqual((ev["v4"]["image"], ev["v4"]["rms_after"], ev["v4"]["snr_after"]), ("cleaner", 1., 20.))


if __name__ == "__main__":
    unittest.main()
