"""Compare several VLA pipeline QA runs (one experiment/variant each) in one report.

Each run of scripts/vla_pipeline_qa.py holds one experiment (A or B) with one set of self-calibration
options (image size, masking), and both flagging recipes. This reads the per-case results of several such
runs and writes one table of run variant x recipe: dispositions, outcomes at the eC threshold, median eC
before -> after, the same medians on the cases measured in every row (paired comparison), outcomes by
corruption type and runtime. Run folders are only read.

  bin/run ml scripts/vla_pipeline_qa_compare.py --output runs/vla-pipeline-qa-compare RUN [RUN ...]
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
from pathlib import Path

import numpy as np

from scripts.vla_pipeline_qa import ACCEPT, OUTCOMES, outcome

CORRUPTIONS = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise",
               7: "amp+phase+noise"}
DISPOSITIONS = ("applied", "not_applied", "no_valid_solutions", "no_usable_output", "unsupported", "runtime_failure")


def corruption(case):
    return CORRUPTIONS[(case["A"] != 1) + 2 * (case["phi_deg"] != 0) + 4 * (case["a"] != 1)]


def variant(p):
    """Readable experiment settings of a run (older plans have no options/variant)."""
    if p["experiment"] == "A":
        return "A: gaincal, true model"
    o = p.get("options") or {}
    size = o.get("selfcal_imsize") or "pipeline default"
    text = f"B: imsize {size}, masking {o.get('selfcal_masking', 'auto')}"
    return text + (", calonly model fix OFF" if o.get("calonly_model_fix") is False else "")


def superseded(result):
    """B flagging_off results without the calonly model-write fix carry the known flux-scale error."""
    if not result["recipe"].endswith("flagging_off") or not result["recipe"].startswith("B:"):
        return False
    fix = result["recipe_record"].get("calonly_model_fix")
    return not (isinstance(fix, dict) and fix.get("applied"))


def load(run):
    run = run.resolve()
    p = json.loads((run / "plan.json").read_text())
    results = [json.loads(x.read_text()) for x in sorted((run / "results").glob("*.json"))]
    units = {x.stem: json.loads(x.read_text()) for x in (run / "units").glob("*.json")}
    cases = {r["case"]: json.loads((Path(p["source_run"]) / "cases" / r["case"] / "case.json").read_text())["case"]
             for r in results}
    rows = []
    for recipe in p["recipes"]:
        xs = [r for r in results if r["recipe"] == recipe]
        if xs:
            rows.append(dict(run=run.name, variant=variant(p), recipe=recipe, results=xs, cases=cases, units=units,
                             superseded=any(superseded(r) for r in xs)))
    return rows


def measured(row):
    return {r["case"]: r["evaluation"] for r in row["results"] if r["evaluation"].get("status") == "measured"}


def summarize(row, common):
    ev = measured(row)
    disp = [r["recipe_record"].get("disposition") for r in row["results"]]
    outs = [outcome(e) for e in ev.values()]
    med = lambda xs: float(np.median(xs)) if xs else None
    recipe_s = [r["recipe_record"]["seconds"] for r in row["results"] if r["recipe_record"].get("seconds") is not None]
    case_s = [row["units"][c]["seconds"] for c in {r["case"] for r in row["results"]}
              if row["units"].get(c, {}).get("seconds") is not None]
    by_type = {}
    for c, e in ev.items():
        t = by_type.setdefault(corruption(row["cases"][c]), dict(n=0, bad=0, fixed=0))
        t["n"] += 1
        t["bad"] += e["before"]["eC"] >= ACCEPT
        t["fixed"] += outcome(e) == "fixed"
    return dict(
        run=row["run"], variant=row["variant"], recipe=row["recipe"], superseded=row["superseded"],
        attempted=len(row["results"]), measured=len(ev), dispositions={k: disp.count(k) for k in DISPOSITIONS},
        outcomes={k: outs.count(k) for k in OUTCOMES},
        eC_before=med([e["before"]["eC"] for e in ev.values()]), eC_after=med([e["after"]["eC"] for e in ev.values()]),
        common_n=len(common), common_eC_before=med([ev[c]["before"]["eC"] for c in common]),
        common_eC_after=med([ev[c]["after"]["eC"] for c in common]),
        recipe_minutes=med([s / 60 for s in recipe_s]), recipe_timed=len(recipe_s),
        case_minutes=med([s / 60 for s in case_s]), by_corruption=by_type)


def write_report(output, summaries, runs):
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    esc = htmllib.escape
    types = [t for t in CORRUPTIONS.values() if any(t in s["by_corruption"] for s in summaries)]
    common_n = summaries[0]["common_n"] if summaries else 0
    out = ["<!doctype html><meta charset=utf-8><title>VLA pipeline QA comparison</title><style>body{font-family:"
           "sans-serif;max-width:1500px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;margin:.6em 0}"
           "td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#f3f3f3}"
           "td:nth-child(-n+3){text-align:left}tr.old td{color:#999}</style>",
           f"<h1>VLA pipeline QA comparison</h1><p>{len(runs)} runs: {', '.join(esc(r.name) for r in runs)}. "
           f"eC: CLEAN recovery error at the common restoring beam (0 = perfect). Acceptable: eC &lt; {ACCEPT}. "
           "Fixed: before ≥ threshold, after &lt; threshold. Degraded: the reverse. Medians over measured cases. "
           + (f"Paired columns use the {common_n} case(s) measured in every row below. " if common_n else
              "No case was measured in every row, so the paired columns are empty. ") + "Runtime: median wall-clock "
           "minutes of the recipe's own CASA process, and of the whole case (regeneration, all recipes, evaluation "
           "imaging); runs with different <code>--parallel</code> are not directly comparable.</p>"]
    if any(s["superseded"] for s in summaries):
        out.append("<p><b>Grey rows</b>: B flagging_off without the calonly model-write fix (results superseded by "
                   "the flux-scale fix, kept only for reference).</p>")
    out.append("<h2>Outcomes</h2><table><tr><th>variant</th><th>recipe</th><th>run</th><th>attempted</th>"
               "<th>measured</th>" + "".join(f"<th>{k.replace('_', ' ')}</th>" for k in DISPOSITIONS) +
               "".join(f"<th>{k.replace('_', ' ')}</th>" for k in OUTCOMES) +
               "<th>median eC before</th><th>median eC after</th><th>paired eC before</th><th>paired eC after</th>"
               "<th>recipe min</th><th>case min</th></tr>")
    for s in summaries:
        out.append(f"<tr{' class=old' if s['superseded'] else ''}><td>{esc(s['variant'])}</td>"
                   f"<td>{esc(s['recipe'])}</td><td>{esc(s['run'])}</td><td>{s['attempted']}</td><td>{s['measured']}</td>" +
                   "".join(f"<td>{s['dispositions'][k]}</td>" for k in DISPOSITIONS) +
                   "".join(f"<td>{s['outcomes'][k]}</td>" for k in OUTCOMES) +
                   f"<td>{f(s['eC_before'])}</td><td>{f(s['eC_after'])}</td><td>{f(s['common_eC_before'])}</td>"
                   f"<td>{f(s['common_eC_after'])}</td><td>{f(s['recipe_minutes'], 1)}</td>"
                   f"<td>{f(s['case_minutes'], 1)}</td></tr>")
    out.append("</table><h2>Fixed by injected corruption</h2><p>fixed / initially unacceptable measured cases "
               "(n = measured cases of that type).</p><table><tr><th>variant</th><th>recipe</th><th>run</th>" +
               "".join(f"<th>{t}</th>" for t in types) + "</tr>")
    for s in summaries:
        cells = []
        for t in types:
            c = s["by_corruption"].get(t)
            cells.append("–" if c is None else f"{c['fixed']}/{c['bad']} (n={c['n']})")
        out.append(f"<tr{' class=old' if s['superseded'] else ''}><td>{esc(s['variant'])}</td><td>{esc(s['recipe'])}"
                   f"</td><td>{esc(s['run'])}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    out.append("</table>")
    (output / "comparison.html").write_text("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="vla_pipeline_qa output folders")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [row for run in args.runs for row in load(run)]
    rows.sort(key=lambda r: (r["variant"], r["recipe"], r["run"]))
    common = sorted(set.intersection(*(set(measured(r)) for r in rows))) if rows else []
    summaries = [summarize(r, common) for r in rows]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(dict(
        threshold=ACCEPT, runs=[str(r.resolve()) for r in args.runs], common_cases=common, rows=summaries), indent=1))
    write_report(args.output, summaries, args.runs)
    print("REPORT", (args.output / "comparison.html").resolve())


if __name__ == "__main__":
    main()
