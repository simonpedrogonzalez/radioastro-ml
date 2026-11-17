"""Read-only audit of negative ground-truth recovery damage in matched Dataset v2 runs.

[USER] design doc docs/vla_pipeline_qa_experiment_draft.md, "Required dataset sanity script: negative damage audit"
(2026-11-16). Checks where negative damage occurs and whether the v5 initial grouping covers the data, before group
counts are interpreted. It does not establish physical causes.

  individual damage  d_k = eC(case) - eC(same case with only kind k removed)        (stored labels in rows.json)
  joint gain damage  D   = eC(case) - eC(same case with amplitude AND phase removed, same noise)  (never a sum of d_k)

Per split and combined, one row per injection mixture (zero rows kept): unique cases, evaluable damages, negative
counts per kind and for D, unique cases with any negative active-kind damage, Cancelling cases. Lists the unexpected
cases (single-kind negatives, gain-only negative D, nonzero D of control/noise-only, negative individual gain damage
with D >= 0), magnitude summaries and the largest examples; reconciles every case with the five groups plus the
exception/unavailable buckets. Counts use unrounded values; tiny negatives (> -tiny, default 1e-6, a diagnostic
tolerance, not a validated numerical bound) are shown separately but never dropped. Missing/invalid counterparts are
unavailable, never zero damage.

Inputs are explicit dataset run folders (plan.json, rows.json, cases/<id>/case.json); split membership is unchanged.
No CASA, imaging, regeneration, MS access or model training.

  bin/run ml scripts/dataset_v2_negative_damage_audit.py --output runs/dataset-v2-negative-damage-audit-YYYYMMDD \\
      runs/dataset-v2-matched-train runs/dataset-v2-matched-val runs/dataset-v2-matched-test

Writes report.html, audit.json and cases.csv to --output.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.vla_pipeline_qa import gain_mask, initial_group, matched_reference

MIXTURES = {0: "control", 1: "amp", 2: "phase", 4: "noise", 3: "amp+phase", 5: "amp+noise", 6: "phase+noise",
            7: "amp+phase+noise"}
KINDS = ("amp", "phase", "noise")
NEUTRAL = dict(amp=("A", 1.), phase=("phi_deg", 0.), noise=("a", 1.))
LABEL_TOLERANCE = 1e-9     # stored d vs d recomputed from the case eC values (both float64 from the same records)
GROUP_BUCKETS = ("improvement_expected", "uncorrupted", "noise_only", "cancelling", "small", "exception", "unavailable")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def counterpart_problems(case, other, kind):
    """Why `other` is not `case` with only `kind` removed (same source, slot, noise seed and reference)."""
    problems = [k for k in ("source", "slot", "noise_seed", "reference_sha256") if case.get(k) != other.get(k)]
    for k, (param, neutral) in NEUTRAL.items():
        if k == kind:
            if other.get(param) != neutral:
                problems.append(f"{param} not removed")
        elif other.get(param) != case.get(param):
            problems.append(f"{param} differs")
    gains_left = other.get("A") != 1 or other.get("phi_deg") != 0
    if gains_left and other.get("antenna") != case.get("antenna"):
        problems.append("antenna differs")
    return problems


def audit_run(run, tiny):
    """One record per row (= case) of a dataset run."""
    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    rows_path = run / "rows.json"
    stored = {r["case"]: r for r in json.loads(rows_path.read_text())["rows"]} if rows_path.exists() else {}
    cache = {}

    def eC_of(case_id):
        if case_id not in cache:
            path = run / "cases" / case_id / "case.json"
            value = json.loads(path.read_text()).get("eC") if path.exists() else None
            cache[case_id] = float(value) if finite(value) else None
        return cache[case_id]

    out = []
    for sp in plan["source_plans"]:
        params = {c["id"]: c for c in sp["cases"]}
        for row in sp["rows"]:
            cid, case = row["case"], params[row["case"]]
            mask = gain_mask(case)
            rec = dict(dataset=str(run), identity=plan.get("configuration_hash"), split=sp.get("split", plan.get("split")),
                       source=sp["source"], row=row["id"], case=cid, mask=mask, mixture=MIXTURES[mask],
                       A=case.get("A"), phi_deg=case.get("phi_deg"), a=case.get("a"), antenna=case.get("antenna"),
                       slot=case.get("slot"), noise_seed=case.get("noise_seed"), eC=eC_of(cid), problems=[])
            if mask != row["mask"]:
                rec["problems"].append(f"plan mask {row['mask']} differs from injection parameters {mask}")
            label = stored.get(cid)
            if label is not None and finite(label.get("eC")) and rec["eC"] is not None and label["eC"] != rec["eC"]:
                rec["problems"].append("stored row eC differs from case eC")
            for k, kind in enumerate(KINDS):
                d_stored = None if label is None or label.get("d") is None else label["d"][k]
                rec[f"stored_d_{kind}"] = d_stored
                rec[f"d_{kind}"] = rec[f"{kind}_counterpart"] = rec[f"{kind}_counterpart_eC"] = None
                if not mask >> k & 1:                       # absent kind: not a damage event, check the convention
                    if d_stored not in (None, 0, 0.):
                        rec["problems"].append(f"absent {kind} has nonzero stored label {d_stored}")
                    continue
                removal = row["removal"][k]
                other = params.get(removal)
                bad = ["no removal counterpart"] if other is None else counterpart_problems(case, other, kind)
                ref = eC_of(removal) if other is not None else None
                rec[f"{kind}_counterpart"], rec[f"{kind}_counterpart_eC"] = removal, ref
                if bad or ref is None or rec["eC"] is None:
                    rec["problems"].append(f"{kind} counterpart unavailable: " + ", ".join(bad or ["missing eC"]))
                    continue
                rec[f"d_{kind}"] = rec["eC"] - ref
                if d_stored is not None and abs(d_stored - rec[f"d_{kind}"]) > LABEL_TOLERANCE:
                    rec["problems"].append(f"stored d_{kind} {d_stored} differs from recomputed {rec[f'd_{kind}']}")
            joint = matched_reference(sp, cid, eC_of) if rec["eC"] is not None else dict(
                status="unavailable", reason="missing case eC")
            rec.update(joint_counterpart=joint.get("id"), joint_counterpart_case=joint.get("case"),
                       joint_counterpart_eC=joint.get("eC"),
                       D=rec["eC"] - joint["eC"] if joint.get("status") != "unavailable" else None)
            if joint.get("status") == "unavailable":
                rec["problems"].append(f"joint counterpart unavailable: {joint.get('reason')}")
            g = initial_group(mask, rec["eC"], joint.get("eC"))
            rec.update(group=g["group"], subgroup=g["subgroup"], group_reason=g["group_reason"])
            rec["bucket"] = g["subgroup"] or g["group"]
            negatives = [n for n in ("d_amp", "d_phase", "d_noise", "D") if finite(rec.get(n)) and rec[n] < 0]
            rec["negative"] = negatives
            rec["tiny_negative"] = [n for n in negatives if rec[n] > -tiny]
            out.append(rec)
    return out


def table(records):
    """One row per mixture (zero rows kept)."""
    rows = []
    for mask in (0, 1, 2, 4, 3, 5, 6, 7):
        sub = [r for r in records if r["mask"] == mask]
        neg = lambda n: sum(finite(r.get(n)) and r[n] < 0 for r in sub)
        rows.append(dict(
            mixture=MIXTURES[mask], total=len(sub),
            **{f"evaluable_d_{k}": sum(finite(r.get(f"d_{k}")) for r in sub) for k in KINDS},
            evaluable_D=sum(finite(r.get("D")) for r in sub),
            any_negative_active=sum(any(n.startswith("d_") for n in r["negative"]) for r in sub),
            **{f"negative_d_{k}": neg(f"d_{k}") for k in KINDS}, negative_D=neg("D"),
            cancelling=sum(r["bucket"] == "cancelling" for r in sub),
            tiny_negative=sum(bool(r["tiny_negative"]) for r in sub),
            with_problems=sum(bool(r["problems"]) for r in sub)))
    return rows


def unexpected(records):
    """The case lists the design doc asks for."""
    return {
        "single-kind negative own damage (amp-only, phase-only, noise-only)": [
            r for r in records if r["mask"] in (1, 2, 4) and finite(r.get(f"d_{KINDS[r['mask'].bit_length() - 1]}"))
            and r[f"d_{KINDS[r['mask'].bit_length() - 1]}"] < 0],
        "gain-only negative joint D (amp, phase, amp+phase): exception bucket": [
            r for r in records if r["mask"] in (1, 2, 3) and finite(r.get("D")) and r["D"] < 0],
        "control/noise-only with nonzero D (counterpart should be the case itself)": [
            r for r in records if r["mask"] in (0, 4) and finite(r.get("D")) and r["D"] != 0],
        "negative individual amp/phase damage with D >= 0 (not Cancelling)": [
            r for r in records if finite(r.get("D")) and r["D"] >= 0 and any(
                finite(r.get(n)) and r[n] < 0 for n in ("d_amp", "d_phase"))],
        "negative noise damage in gain+noise mixtures (counted, not a Cancelling criterion)": [
            r for r in records if r["mask"] in (5, 6, 7) and finite(r.get("d_noise")) and r["d_noise"] < 0],
        "missing/invalid measurements or labels": [r for r in records if r["problems"]],
    }


def magnitude(records, key):
    values = np.asarray([r[key] for r in records if finite(r.get(key)) and r[key] < 0], float)
    return dict(count=int(values.size), minimum=float(values.min()) if values.size else None,
                median=float(np.median(values)) if values.size else None)


def verdict(records):
    lists = unexpected(records)
    keys = list(lists)
    groups = Counter(r["bucket"] for r in records)
    return dict(
        cases=len(records),
        expected_noise_gain_joint_negatives=groups.get("cancelling", 0),
        unexpected_single_kind_negatives=len(lists[keys[0]]),
        unexpected_gain_only_joint_negatives=len(lists[keys[1]]),
        nonzero_D_without_gains=len(lists[keys[2]]),
        missing_or_invalid=len(lists[keys[5]]),
        groups={b: groups.get(b, 0) for b in GROUP_BUCKETS},
        reconciled=sum(groups.get(b, 0) for b in GROUP_BUCKETS) == len(records),
        needs_investigation=bool(len(lists[keys[0]]) or len(lists[keys[1]]) or len(lists[keys[2]])
                                 or len(lists[keys[5]])))


EXPORT = ("split", "source", "row", "case", "mixture", "A", "phi_deg", "a", "antenna", "slot", "noise_seed", "eC",
          "d_amp", "d_phase", "d_noise", "stored_d_amp", "stored_d_phase", "stored_d_noise", "amp_counterpart",
          "amp_counterpart_eC", "phase_counterpart", "phase_counterpart_eC", "noise_counterpart", "noise_counterpart_eC",
          "D", "joint_counterpart", "joint_counterpart_eC", "group", "subgroup", "group_reason", "negative",
          "tiny_negative", "problems", "dataset")


def write_report(output, audit, tiny):
    e = html.escape
    g = lambda x: "—" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x))
    cols = list(audit["tables"]["combined"][0])
    parts = ["<!doctype html><meta charset=utf-8><title>Dataset v2 negative damage audit</title><style>body{font:14px "
             "system-ui;max-width:1500px;margin:auto;padding:1.5rem}table{border-collapse:collapse;margin:.4rem 0 1.2rem}"
             "th,td{border:1px solid #ccd;padding:.25rem .45rem;text-align:right}th{background:#f1f4f6}td.l{text-align:"
             "left}.scroll{overflow-x:auto}</style><h1>Dataset v2 negative damage audit</h1>",
             f"<p>Datasets: {', '.join(e(d) for d in audit['datasets'])}. Individual damage d_k = eC(case) − eC(only k "
             "removed); joint gain damage D = eC(case) − eC(amplitude and phase removed, same noise). Strict negatives "
             f"on unrounded values; tiny = negative above −{tiny:g} (diagnostic tolerance, not a validated bound). "
             f"Duplicates skipped: {audit['duplicates']}. Read-only; establishes where negative cases occur, not their "
             "physical cause.</p>"]
    v = audit["verdict"]
    parts.append("<h2>Verdict</h2><ul>"
                 f"<li>{v['cases']} unique cases; groups: "
                 + ", ".join(f"{k} {n}" for k, n in v["groups"].items())
                 + f" (reconciled: {v['reconciled']}).</li>"
                 f"<li>Expected noise+gain joint negatives (Cancelling): {v['expected_noise_gain_joint_negatives']}.</li>"
                 f"<li>Unexpected single-kind negatives: {v['unexpected_single_kind_negatives']}; unexpected gain-only "
                 f"joint negatives: {v['unexpected_gain_only_joint_negatives']}; nonzero D without gains: "
                 f"{v['nonzero_D_without_gains']}; missing/invalid: {v['missing_or_invalid']}.</li>"
                 + ("<li><b>Unexpected or invalid cases exist: the grouping assumption needs investigation (they are "
                    "listed below, not excluded or relabelled).</b></li>" if v["needs_investigation"] else
                    "<li>No unexpected or invalid cases: the grouping covers the data.</li>") + "</ul>")
    for split, rows in audit["tables"].items():
        parts.append(f"<h2>Mixtures: {e(split)}</h2><div class=scroll><table><tr>"
                     + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr>"
                     + "".join("<tr>" + "".join(f"<td{' class=l' if c == 'mixture' else ''}>{r[c]}</td>" for c in cols)
                               + "</tr>" for r in rows) + "</table></div>")
    parts.append("<h2>Magnitudes of negatives (combined)</h2><div class=scroll><table><tr><th>label</th><th>count</th>"
                 "<th>minimum</th><th>median</th></tr>" + "".join(
                     f"<tr><td class=l>{k}</td><td>{m['count']}</td><td>{g(m['minimum'])}</td><td>{g(m['median'])}</td></tr>"
                     for k, m in audit["magnitudes"].items()) + "</table></div>")
    show = ("split", "source", "row", "case", "mixture", "eC", "d_amp", "d_phase", "d_noise", "D", "joint_counterpart",
            "group", "problems")
    for title, cases in audit["unexpected"].items():
        parts.append(f"<h2>{e(title)}: {len(cases)}</h2>")
        if cases:
            key = "D" if "joint D" in title else None
            ordered = sorted(cases, key=lambda r: min([r[n] for n in ("d_amp", "d_phase", "d_noise", "D")
                                                       if finite(r.get(n))] or [0]) if key is None else r[key])
            parts.append("<p>Largest examples (all in cases.csv / audit.json):</p><div class=scroll><table><tr>"
                         + "".join(f"<th>{c}</th>" for c in show) + "</tr>" + "".join(
                             "<tr>" + "".join(f"<td class=l>{e(g(r.get(c)) if not isinstance(r.get(c), list) else '; '.join(r[c]))}</td>"
                                              for c in show) + "</tr>" for r in ordered[:8]) + "</table></div>")
    (output / "report.html").write_text("".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path, help="matched Dataset v2 run folders (train/val/test)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tiny", type=float, default=1e-6, help="diagnostic tolerance for tiny negatives (default 1e-6 eC)")
    a = ap.parse_args()
    if not (finite(a.tiny) and a.tiny >= 0):
        ap.error("--tiny must be a finite nonnegative number")
    records, seen, duplicates = [], set(), 0
    for run in a.datasets:
        for r in audit_run(run, a.tiny):
            key = (r["identity"], r["case"])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            records.append(r)
    splits = list(dict.fromkeys(r["split"] for r in records))
    lists = unexpected(records)
    audit = dict(datasets=[str(d.resolve()) for d in a.datasets], tiny=a.tiny, duplicates=duplicates,
                 tables={**{s: table([r for r in records if r["split"] == s]) for s in splits},
                         "combined": table(records)},
                 magnitudes={k: magnitude(records, k) for k in ("d_amp", "d_phase", "d_noise", "D")},
                 unexpected={k: [{c: r.get(c) for c in EXPORT} for r in v] for k, v in lists.items()},
                 verdict=verdict(records), verdict_by_split={s: verdict([r for r in records if r["split"] == s])
                                                             for s in splits})
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "audit.json").write_text(json.dumps(audit, indent=1))
    with open(a.output / "cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPORT)
        w.writeheader()
        for r in records:
            w.writerow({c: "; ".join(r[c]) if isinstance(r.get(c), list) else r.get(c) for c in EXPORT})
    write_report(a.output, audit, a.tiny)
    print(json.dumps(audit["verdict"], indent=1))
    print("REPORT", (a.output / "report.html").resolve())


if __name__ == "__main__":
    main()
