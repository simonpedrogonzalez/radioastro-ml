"""End-to-end checks of experiment B fixes (CASA pipeline, minutes per run). Unit tests of the pure decision
logic are in tests/test_vla_pipeline_qa.py.

Checks (each one small vla_pipeline_qa run on the smoke source, launched concurrently):
  model_fix         flagging_off with the calonly model-write fix, control + phase case, 512 px: a CLEAN model is
                    present before every gain solve and equals the known flux before the amplitude solves,
                    amplitude gains ~1, |flux error| and eC after are small.
                    --compare-unfixed also runs the unmodified pipeline and requires the +41% offset to reappear.
  calonly_preserve  same run (work kept): final FLAG/FLAG_ROW/WEIGHT/WEIGHT_SPECTRUM of the recipe output equal
                    the input's (direct column comparison, an independent check of the bookkeeping), the
                    pipeline flag stages say unchanged, and the flux matches a flagging_on reference run.
  reuse             control at the pipeline's own image size, flagging_off and flagging_on: when hif_selfcal
                    records no application and no flags are added, after == before with evaluation_reused and
                    no evaluation imaging. If the pipeline applies a calibration, the check is INCONCLUSIVE.

  python -m scripts.vla_pipeline_qa_checks --root runs/vla-pipeline-qa-checks-YYYYMMDD [--checks model_fix reuse]
Runs resume (completed cases are kept). Results: ROOT/checks.json. Exit code 1 if any check FAILS.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_RUN = "runs/dataset-v2-matched-smoke-20261105"
CONTROL, PHASE = "c76c678d422e5ba54aae", "7dc352f1d6659de668a6"
OFF, ON = "B:phase_amp:flagging_off", "B:phase_amp:flagging_on"
FLUX_TOL, GAIN_TOL, MODEL_TOL = 0.02, 0.02, 0.05
RUNS = {   # name: (recipe, cases, extra plan args)
    "fix": (OFF, [CONTROL, PHASE], ["--selfcal-imsize", "512", "--native-diagnostics", "--keep-work"]),
    "unfixed": (OFF, [CONTROL, PHASE], ["--selfcal-imsize", "512", "--native-diagnostics", "--no-calonly-model-fix"]),
    "flagging_on": (ON, [CONTROL, PHASE], ["--selfcal-imsize", "512"]),
    "reuse_off": (OFF, [CONTROL], []),
    "reuse_on": (ON, [CONTROL], []),
}
NEEDS = {"model_fix": ["fix"], "calonly_preserve": ["fix", "flagging_on"], "reuse": ["reuse_off", "reuse_on"]}


def ml(*args):
    return subprocess.run([str(ROOT / "bin/run"), "ml", "python", "-m", "scripts.vla_pipeline_qa", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def launch(root, names, parallel):
    procs = {}
    for name in names:
        out = root / name
        recipe, cases, extra = RUNS[name]
        if not (out / "plan.json").exists():
            res = ml("--output", out, "--plan", "--source-run", SOURCE_RUN, "--experiment", "B", "--recipes", recipe,
                     "--cases", ",".join(cases), *extra)
            if res.returncode:
                raise RuntimeError(f"plan {name} failed:\n{res.stdout[-2000:]}{res.stderr[-2000:]}")
        log = open(out / "execute.log", "a")
        procs[name] = subprocess.Popen([str(ROOT / "bin/run"), "ml", "python", "-m", "scripts.vla_pipeline_qa",
                                        "--output", str(out), "--execute", "--parallel", str(parallel)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print("RUN", name, out, flush=True)
    return {n: p.wait() for n, p in procs.items()}


def result(root, name, case):
    recipe = RUNS[name][0]
    path = root / name / "results" / f"{case}__{recipe.replace(':', '__')}.json"
    return json.loads(path.read_text()) if path.exists() else None


def known_flux(case):
    from scripts.vla_pipeline_qa import source_paths
    p = dict(source_run=str(ROOT / SOURCE_RUN))
    source_run, _, _, prepared = source_paths(p, case)
    return json.loads((source_run / "sources" / prepared["id"] / "recovery" / "reference.json").read_text())["flux_jy"]


class Check:
    def __init__(self, name):
        self.name, self.items = name, []

    def expect(self, label, ok, value=None):
        self.items.append(dict(label=label, ok=bool(ok), value=value))

    def status(self, inconclusive=False):
        if inconclusive:
            return "INCONCLUSIVE"
        return "PASS" if self.items and all(i["ok"] for i in self.items) else "FAIL"


def check_model_fix(root, compare):
    c = Check("model_fix")
    for case in (CONTROL, PHASE):
        r = result(root, "fix", case)
        if r is None:
            c.expect(f"{case[:8]} result exists", False)
            continue
        rec, ev = r["recipe_record"], r["evaluation"]
        c.expect(f"{case[:8]} fix applied in pipeline process", rec.get("calonly_model_fix", {}).get("applied"),
                 rec.get("calonly_model_fix"))
        solves = rec.get("solve_models", [])
        c.expect(f"{case[:8]} gain solves logged", solves, len(solves))
        c.expect(f"{case[:8]} model present and nonzero before every solve",
                 solves and all((s["model_before"].get("median_abs") or 0) > 0 for s in solves),
                 [s["model_before"].get("median_abs") for s in solves])
        flux = known_flux(case)
        amp = [s for s in solves if s.get("calmode") == "ap"]
        if amp:
            models = [s["model_before"]["median_abs"] for s in amp]
            gains = [s["gains"].get("median_amp") for s in amp]
            c.expect(f"{case[:8]} model before amplitude solves within {MODEL_TOL:.0%} of known flux {flux:.4f} Jy",
                     all(abs(m / flux - 1) < MODEL_TOL for m in models), models)
            c.expect(f"{case[:8]} median amplitude gains within {GAIN_TOL:.0%} of 1",
                     all(g is not None and abs(g - 1) < GAIN_TOL for g in gains), gains)
        c.expect(f"{case[:8]} evaluation measured", ev.get("status") == "measured", ev.get("status"))
        if ev.get("status") == "measured":
            c.expect(f"{case[:8]} |flux error after| < {FLUX_TOL}", abs(ev["after"]["flux_error"]) < FLUX_TOL,
                     ev["after"]["flux_error"])
            c.expect(f"{case[:8]} eC after < 0.05", ev["after"]["eC"] < 0.05, ev["after"]["eC"])
        if compare:
            u = result(root, "unfixed", case)
            if u is None:
                c.expect(f"{case[:8]} unfixed result exists", False)
                continue
            applied_amp = any("_ap" in g["gaintable"] for a in u["recipe_record"].get("applications", [])
                              for g in a["gaintables"])
            if applied_amp:
                c.expect(f"{case[:8]} unfixed: amplitude table applied without a model -> flux offset > 0.2 "
                         "(the bug is reproduced)", u["evaluation"]["after"]["flux_error"] > 0.2,
                         u["evaluation"]["after"]["flux_error"])
            else:
                c.expect(f"{case[:8]} unfixed: no amplitude table applied, offset not expected", True)
    return c


def casa_compare(a, b):
    """CASA: FLAG/FLAG_ROW/WEIGHT/WEIGHT_SPECTRUM hashes of two MSs (rows must match)."""
    from scripts.vla_pipeline_qa import table_hash, same_rows
    cols = ("FLAG", "FLAG_ROW", "WEIGHT", "WEIGHT_SPECTRUM")
    ha, hb = table_hash(a, cols), table_hash(b, cols)
    print("COMPARE " + json.dumps(dict(rows_match=same_rows(a, b),
                                       differ=sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k)),
                                       columns=sorted(set(ha) | set(hb)))))


def check_calonly_preserve(root, keep):
    c = Check("calonly_preserve")
    for case in (CONTROL, PHASE):
        r = result(root, "fix", case)
        if r is None:
            c.expect(f"{case[:8]} result exists", False)
            continue
        rec = r["recipe_record"]
        stages = {s["stage"]: s for s in rec.get("flag_stages", [])}
        c.expect(f"{case[:8]} hifv_flagtargetsdata skipped", stages.get("hifv_flagtargetsdata", {}).get("status") == "skipped")
        c.expect(f"{case[:8]} final applycal summary: no flags added",
                 stages.get("hif_selfcal applycal", {}).get("status") == "unchanged",
                 stages.get("hif_selfcal applycal", {}).get("added"))
        c.expect(f"{case[:8]} not labelled flagging_off_violated", not rec.get("flagging_off_violated"))
        c.expect(f"{case[:8]} applied tables have calwt False",
                 all(not g["calwt"] for a in rec.get("applications", []) for g in a["gaintables"]))
        work = root / "fix" / "work" / case
        out, observed = Path(rec.get("output_ms") or ""), work / "observed.ms"
        if rec.get("output_ms") and out.exists() and observed.exists():
            res = subprocess.run([str(ROOT / "bin/run"), "casa", "scripts/vla_pipeline_qa_checks.py", "--casa-compare",
                                  str(observed), str(out)], cwd=ROOT, capture_output=True, text=True)
            line = next((l for l in res.stdout.splitlines() if l.startswith("COMPARE ")), None)
            got = json.loads(line[8:]) if line else None
            c.expect(f"{case[:8]} output rows match input", got and got["rows_match"], got)
            c.expect(f"{case[:8]} final flags and weights identical to input (direct column check)",
                     got is not None and not got["differ"], got and got["differ"])
        else:
            c.expect(f"{case[:8]} work MSs kept for the column check", False, rec.get("output_ms"))
        ref = result(root, "flagging_on", case)
        if ref and ref["evaluation"].get("status") == "measured" and r["evaluation"].get("status") == "measured":
            a, b = r["evaluation"]["after"]["flux_error"], ref["evaluation"]["after"]["flux_error"]
            c.expect(f"{case[:8]} flux scale matches flagging_on within {FLUX_TOL}", abs(a - b) < FLUX_TOL, [a, b])
        else:
            c.expect(f"{case[:8]} flagging_on reference measured", False)
    if not keep:
        shutil.rmtree(root / "fix" / "work", ignore_errors=True)
    return c


def check_reuse(root):
    c = Check("reuse")
    exercised = 0
    for name in ("reuse_off", "reuse_on"):
        r = result(root, name, CONTROL)
        if r is None:
            c.expect(f"{name} result exists", False)
            continue
        rec, ev = r["recipe_record"], r["evaluation"]
        decision = rec.get("evaluation_decision", {})
        if rec.get("applications"):
            c.expect(f"{name}: pipeline applied a calibration, reuse branch not exercised (inconclusive)", True,
                     rec.get("disposition"))
            continue
        exercised += 1
        c.expect(f"{name}: no application recorded, disposition not_applied", rec.get("disposition") == "not_applied",
                 rec.get("disposition"))
        c.expect(f"{name}: every flag stage unchanged or skipped",
                 all(s["status"] in ("unchanged", "skipped") for s in rec.get("flag_stages", [])),
                 [(s["stage"], s["status"]) for s in rec.get("flag_stages", [])])
        c.expect(f"{name}: decision reuse", decision.get("reuse") is True, decision)
        c.expect(f"{name}: evaluation_reused and after == before",
                 ev.get("evaluation_reused") is True and ev.get("after") == ev.get("before"), ev.get("reason"))
        label = RUNS[name][0].replace(":", "__")
        imaged = (root / name / "cases" / CONTROL / label).exists() or \
            (root / name / "logs" / f"unit_{CONTROL}_{label}_image.log").exists()
        c.expect(f"{name}: no evaluation imaging and no output split", not imaged and not rec.get("output_ms"))
    return c, exercised == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--checks", nargs="*", default=list(NEEDS), choices=list(NEEDS))
    ap.add_argument("--compare-unfixed", action="store_true", help="model_fix: also run the unmodified pipeline")
    ap.add_argument("--no-run", action="store_true", help="Only evaluate existing runs")
    ap.add_argument("--keep-work", action="store_true", help="Keep the 'fix' work MSs after calonly_preserve")
    ap.add_argument("--parallel", type=int, default=2, help="Cases in parallel within each run")
    ap.add_argument("--casa-compare", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.casa_compare:
        return casa_compare(*a.casa_compare)
    if not a.root:
        ap.error("--root is required")
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = sorted({n for c in a.checks for n in NEEDS[c]} | ({"unfixed"} if a.compare_unfixed and "model_fix" in a.checks else set()))
    if not a.no_run:
        codes = launch(root, names, a.parallel)
        print("RUNS DONE", codes, flush=True)
    report = {}
    for name in a.checks:
        if name == "model_fix":
            c, status = check_model_fix(root, a.compare_unfixed), None
        elif name == "calonly_preserve":
            c, status = check_calonly_preserve(root, a.keep_work), None
        else:
            c, inconclusive = check_reuse(root)
            status = c.status(inconclusive)
        status = status or c.status()
        report[name] = dict(status=status, items=c.items)
        print(f"\n{status}  {name}")
        for i in c.items:
            print(f"  [{'ok' if i['ok'] else 'FAIL'}] {i['label']}" + ("" if i["value"] is None else f"  ({i['value']})"))
    (root / "checks.json").write_text(json.dumps(report, indent=2, default=str))
    sys.exit(1 if any(v["status"] == "FAIL" for v in report.values()) else 0)


if __name__ == "__main__":
    main()
