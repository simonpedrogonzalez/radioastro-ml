"""Pure Dataset v2 matched-combination sampling, unit planning, sharding and joins.

One recipe (A, phase, r, antenna) per sampling template; all seven nonzero
presence masks reuse that recipe, so every damage label has its matched
removal case inside the same template (or the shared per-source control).
See docs/dataset_v2_updated_plan.md.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.stats import qmc

from scripts.dataset_v2 import RANGES, TYPES, canonical_case, stream_seed

# Weak bands sit next to the strong ranges, between them and "no corruption".
# Provisional pilot values; freeze from a training-only pilot (plan §1).
WEAK_RANGES = {"phase_negative": (-23.79, -2.), "phase_positive": (2., 23.743),
               "amp_attenuation": (.66391, .98), "amp_amplification": (1.02, 1.4309)}
DEFAULT = dict(schema=3, design="matched_combination_groups",
               flux_policy="original_clean_model_sum_v2_adjusted",
               range_policy="v2_adjusted_noise_first_0.9_crossing_plus_weak_bands",
               recipes_per_source=12, noise_slots=1, weak_fraction=.25, seed=20261106,
               ranges=RANGES, weak_ranges=WEAK_RANGES,
               pilot_root="runs/simulation-parametrization-v2-20261104")
BRANCHES = [(amp, phase) for amp in ("attenuation", "amplification") for phase in ("negative", "positive")]


def _decode(u, lo, hi, logarithmic):
    if logarithmic:
        return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return float(lo + u * (hi - lo))


def _banded(u, strong, weak, fraction, logarithmic):
    """Map one LHS coordinate to the weak band (u < fraction) or the strong range."""
    if u < fraction:
        return _decode(u / fraction, *weak, logarithmic), "weak"
    return _decode((u - fraction) / (1 - fraction), *strong, logarithmic), "strong"


def sample_recipes(sid, snr0, ants, config):
    """Stratified recipes: equal counts per amp branch x phase sign, LHS within each."""
    snr0 = float(snr0)
    if not math.isfinite(snr0) or snr0 <= 0:
        raise ValueError(f"Invalid source SNR0: {sid}")
    if not ants:
        raise ValueError(f"No eligible antennas: {sid}")
    count = int(config["recipes_per_source"])
    if count <= 0 or count % len(BRANCHES):
        raise ValueError(f"recipes_per_source must be a positive multiple of {len(BRANCHES)}")
    w = float(config["weak_fraction"])
    if not 0 <= w < 1:
        raise ValueError("weak_fraction must lie in [0, 1)")
    ranges, weak = config["ranges"], config["weak_ranges"]
    r_lo, r_hi = ranges["noise_r"]
    admissible = max(r_lo, float(np.nextafter(1 / snr0, math.inf)))
    if admissible >= r_hi:
        raise ValueError(f"{sid}: no admissible active thermal-noise range")
    recipes = []
    for amp_branch, phase_sign in BRANCHES:
        group = f"{amp_branch}:{phase_sign}"
        cube = qmc.LatinHypercube(d=3, scramble=True, seed=stream_seed(
            config["seed"], sid, f"recipe:{group}")).random(count // len(BRANCHES))
        for ua, up, un in cube:
            key = f"amp_{amp_branch}"
            A, amp_band = _banded(float(ua), ranges[key], weak[key], w, amp_branch == "amplification")
            key = f"phase_{phase_sign}"
            phi, phase_band = _banded(float(up), ranges[key], weak[key], w, False)
            r = _decode(float(un), admissible, r_hi, True)
            recipes.append(dict(A=A, phi_deg=phi, r=r, a=r * snr0, amp_branch=amp_branch,
                                phase_sign=phase_sign, amp_band=amp_band, phase_band=phase_band,
                                group=group, unit_cube=dict(amp=float(ua), phase=float(up), noise=float(un)),
                                noise_bounds=[admissible, r_hi]))
    for j, recipe in enumerate(recipes):
        recipe["id"] = f"{sid}:r{j:02d}"
        recipe["index"] = j
        recipe["source"] = sid
        recipe["antenna"] = int(np.random.default_rng(
            stream_seed(config["seed"], sid, f"antenna:{recipe['id']}")).choice(sorted(ants)))
        if recipe["a"] <= 1:
            raise ValueError(f"{sid}: sampled active noise at a<=1")
    return recipes


def mask_case(sid, reference_hash, slot, noise_seed, recipe, mask):
    """Physical case of one recipe under one presence mask (mask 0 is the control)."""
    if mask == 0:
        return canonical_case(sid, reference_hash, slot, noise_seed, 1., 0., 1., None)
    return canonical_case(sid, reference_hash, slot, noise_seed,
                          recipe["A"] if mask & 1 else 1., recipe["phi_deg"] if mask & 2 else 0.,
                          recipe["a"] if mask & 4 else 1., recipe["antenna"])


def unit_file(unit_id):
    return unit_id.replace(":", "__")


def plan_source(source, config, split):
    """Recipes, work units, cases and fully observed rows for one prepared source."""
    sid = source["id"]
    ants = sorted(int(a) for a, n in source["norms"].items() if n["V_Ak_over_sigma_L2"] > 0)
    recipes = sample_recipes(sid, source["nominal_source_snr"], ants, config)
    ref = source["reference_sha256"]
    seeds = [stream_seed(config["seed"], sid, f"noise:{slot}") for slot in range(int(config["noise_slots"]))]
    units, rows, cases = [], [], {}
    for slot, seed in enumerate(seeds):
        control = mask_case(sid, ref, slot, seed, None, 0)
        cases[control["id"]] = control
        units.append(dict(id=f"{sid}:control:slot{slot}", source=sid, slot=slot, recipe=None,
                          cases=[control["id"]]))
        rows.append(dict(id=f"{sid}:control:slot{slot}", source=sid, split=split, slot=slot, recipe=None,
                         mask=0, b=[0, 0, 0], case=control["id"], removal=[None, None, None]))
        for recipe in recipes:
            by_mask = {m: mask_case(sid, ref, slot, seed, recipe, m) for m in range(1, 8)}
            by_mask[0] = control
            for c in by_mask.values():
                cases[c["id"]] = c
            units.append(dict(id=f"{recipe['id']}:slot{slot}", source=sid, slot=slot, recipe=recipe["id"],
                              cases=[by_mask[m]["id"] for m in range(1, 8)]))
            for m in range(1, 8):
                b = [int(bool(m & (1 << k))) for k in range(3)]
                rows.append(dict(id=f"{recipe['id']}:m{m}:slot{slot}", source=sid, split=split, slot=slot,
                                 recipe=recipe["id"], mask=m, b=b, case=by_mask[m]["id"],
                                 removal=[by_mask[m & ~(1 << k)]["id"] if b[k] else None for k in range(3)]))
    if len(cases) != len(rows):
        raise ValueError(f"{sid}: matched cases are not one per row")
    return dict(source=sid, split=split, recipes=recipes, units=units, rows=rows,
                cases=list(cases.values()), noise_seeds=seeds,
                counts=dict(recipes=len(recipes), units=len(units), rows=len(rows), cases=len(cases),
                            masks={str(k): v for k, v in Counter(r["mask"] for r in rows).items()}))


def assign_shards(units, cost, shards):
    """Deterministic size-balanced disjoint assignment: largest unit first to the lightest shard."""
    if shards < 1:
        raise ValueError("Expected at least one shard")
    loads, owner = [0.] * shards, {}
    for unit in sorted(units, key=lambda u: (-cost[u["id"]], u["id"])):
        target = min(range(shards), key=lambda s: (loads[s], s))
        owner[unit["id"]] = target
        loads[target] += cost[unit["id"]]
    return owner, loads


def join_rows(rows, records, dirty_change=None):
    """Complete labels: d_k = eC(case) - eC(case without k); absent types are exact zeros.

    ``dirty_change(case_id, removal_id)`` returns ||D_case - D_removal|| / ||D0||.
    """
    output = []
    for row in rows:
        full = records[row["case"]]
        y = float(full["eC"])
        d, p = [0., 0., 0.], [0., 0., 0.]
        for k in range(3):
            if row["b"][k]:
                removal = records[row["removal"][k]]
                d[k] = y - float(removal["eC"])
                if dirty_change is not None:
                    p[k] = float(dirty_change(row["case"], row["removal"][k]))
        if y < 0 or any(value > y + 1e-10 for value in d) or any(value < 0 for value in p):
            raise ValueError(f"Invalid recovery damage for {row['id']}")
        output.append(dict(row, eC=y, eD=float(full["eD"]), d=d, p=p, X=full["X"]))
    return output
