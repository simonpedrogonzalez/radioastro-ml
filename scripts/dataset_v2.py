"""Pure Dataset v2 sampling, sparse-label planning and compact row joins."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np
from scipy.stats import qmc

from scripts.injection_bounds_report import digest

TYPES = ("amp", "phase", "noise")
MASKS = tuple(range(8))
RANGES = {"phase_negative": (-180., -23.79), "phase_positive": (23.743, 180.),
          "amp_attenuation": (.01, .66391), "amp_amplification": (1.4309, 20.276),
          "noise_r": (.0097046, .19152)}
DEFAULT = dict(schema=2, flux_policy="original_clean_model_sum_v2_adjusted", range_policy="v2_adjusted_noise_first_0.9_crossing", vectors_per_mask=4, noise_slots=1, seed=20261104,
               ranges=RANGES, max_final_bytes=20_000_000_000,
               max_scratch_bytes=16_000_000_000,
               pilot_root="runs/simulation-parametrization-v2-20261104")


def stream_seed(base, source, purpose):
    raw = f"dataset-v2:{base}:{source}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2_147_483_646 + 1


def canonical_case(source, reference_hash, slot, noise_seed, A, phi, a, antenna):
    if not all(math.isfinite(float(v)) for v in (A, phi, a)) or A <= 0 or a < 1:
        raise ValueError("Invalid physical case")
    phi = 180. if abs(phi) == 180. else float(phi)
    antenna = None if A == 1 and phi == 0 else int(antenna)
    recipe = dict(source=source, reference_sha256=reference_hash, slot=int(slot),
                  noise_seed=int(noise_seed),
                  A=float(A), phi_deg=phi, a=float(a), antenna=antenna)
    return dict(id=digest(recipe)[:20], **recipe)


def _branches(mask):
    amp = ("attenuation", "amplification") if mask & 1 else ("inactive",)
    phase = ("negative", "positive") if mask & 2 else ("inactive",)
    return [(a, p) for a in amp for p in phase]


def _decode(u, lo, hi, logarithmic):
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))) if logarithmic else float(lo + u * (hi - lo))


def sample_vectors(sid, snr0, config, split):
    """Physical LHS design shared by pre-CASA preview and exact planning."""
    snr0 = float(snr0)
    if not math.isfinite(snr0) or snr0 <= 0:
        raise ValueError(f"Invalid source SNR0: {sid}")
    ranges = config["ranges"]
    r_lo, r_hi = ranges["noise_r"]
    admissible = max(r_lo, float(np.nextafter(1 / snr0, math.inf)))
    if admissible >= r_hi:
        raise ValueError(f"{sid}: no admissible active thermal-noise range")
    count = int(config["vectors_per_mask"])
    slots = int(config["noise_slots"])
    if count <= 0 or slots <= 0:
        raise ValueError("Positive vector and slot counts required")
    vectors = []
    for mask in range(1, 8):
        branch_groups = _branches(mask)
        if count % len(branch_groups):
            raise ValueError(f"Mask {mask} budget must divide {len(branch_groups)} branches")
        each = count // len(branch_groups)
        for amp_branch, phase_sign in branch_groups:
            active = [kind for bit, kind in enumerate(TYPES) if mask & (1 << bit)]
            group = f"mask{mask}:{amp_branch}:{phase_sign}"
            cube = qmc.LatinHypercube(d=len(active), scramble=True,
                                       seed=stream_seed(config["seed"], sid, f"sample:{group}")).random(each)
            for i, coordinates in enumerate(cube):
                coord = dict(zip(active, map(float, coordinates), strict=True))
                A = _decode(coord["amp"], *ranges[f"amp_{amp_branch}"], amp_branch == "amplification") if mask & 1 else 1.
                phi = _decode(coord["phase"], *ranges[f"phase_{phase_sign}"], False) if mask & 2 else 0.
                r = _decode(coord["noise"], admissible, r_hi, True) if mask & 4 else 1 / snr0
                a = r * snr0 if mask & 4 else 1.
                if mask & 4 and a <= 1:
                    raise ValueError(f"{sid}: sampled active noise at a<=1")
                vid = f"{sid}:m{mask}:{amp_branch}:{phase_sign}:{i}"
                vector = dict(id=vid, source=sid, split=split, mask=mask, A=A, phi_deg=phi,
                              a=a, r=r, amp_branch=amp_branch,
                              phase_sign=phase_sign, group=group, unit_cube=coord,
                              sample_bounds={kind: list(ranges[f"amp_{amp_branch}"] if kind == "amp" else
                                                        ranges[f"phase_{phase_sign}"] if kind == "phase" else
                                                        (admissible, r_hi)) for kind in active})
                vectors.append(vector)
    return vectors

def plan_source(source, config, split):
    """Attach antenna, noise and sparse-label recipes to the shared vectors."""
    sid = source["id"]
    snr0 = float(source["nominal_source_snr"])
    ants = sorted(int(a) for a, n in source["norms"].items() if n["V_Ak_over_sigma_L2"] > 0)
    if not ants:
        raise ValueError(f"No eligible antennas: {sid}")
    vectors = sample_vectors(sid, snr0, config, split)
    slots = int(config["noise_slots"])
    reference_hash = source["reference_sha256"]
    noise_seeds = [stream_seed(config["seed"], sid, f"noise:{slot}") for slot in range(slots)]
    cases, rows = {}, []
    for vector in vectors:
        mask = vector["mask"]
        active = [kind for bit, kind in enumerate(TYPES) if mask & (1 << bit)]
        vid = vector["id"]
        A, phi, a = vector["A"], vector["phi_deg"], vector["a"]
        antenna = int(np.random.default_rng(stream_seed(config["seed"], sid, f"antenna:{vid}")).choice(ants)) if mask & 3 else None
        vector["antenna"] = antenna
        for slot in range(slots):
            row_id = f"{vid}:slot{slot}"
            choose = np.random.default_rng(stream_seed(config["seed"], sid,
                                                         f"label:{split}:{row_id}"))
            selected = str(choose.choice(active))
            full = canonical_case(sid, reference_hash, slot, noise_seeds[slot], A, phi, a, antenna)
            removed = dict(A=A, phi=phi, a=a)
            removed[{"amp": "A", "phase": "phi", "noise": "a"}[selected]] = {"amp": 1., "phase": 0., "noise": 1.}[selected]
            removal = canonical_case(sid, reference_hash, slot, noise_seeds[slot], removed["A"],
                                     removed["phi"], removed["a"], antenna)
            for c in (full, removal):
                cases[c["id"]] = c
            b = [int(bool(mask & (1 << k))) for k in range(3)]
            o = [int(not present or kind == selected) for present, kind in zip(b, TYPES, strict=True)]
            pi = [1 / len(active) if present else 1. for present in b]
            rows.append(dict(id=row_id, vector=vid, source=sid, split=split,
                             slot=slot, mask=mask, b=b, o=o, pi=pi,
                             selected=selected, full=full["id"], removal=removal["id"],
                             label_seed=stream_seed(config["seed"], sid, f"label:{split}:{row_id}")))

    for slot in range(slots):
        control = canonical_case(sid, reference_hash, slot, noise_seeds[slot], 1., 0., 1., None)
        cases[control["id"]] = control
        rows.append(dict(id=f"{sid}:control:slot{slot}", vector=None, source=sid,
                         split=split, slot=slot, mask=0, b=[0, 0, 0], o=[1, 1, 1],
                         pi=[1., 1., 1.], selected=None, full=control["id"],
                         removal=None, label_seed=None))
    return dict(source=sid, split=split, vectors=vectors, rows=rows,
                cases=list(cases.values()), noise_seeds=noise_seeds,
                admissible_noise_r=[max(config["ranges"]["noise_r"][0], float(np.nextafter(1 / snr0, math.inf))), config["ranges"]["noise_r"][1]],
                counts=dict(rows=len(rows), vectors=len(vectors), unique_cases=len(cases),
                            masks={str(k): v for k, v in Counter(row["mask"] for row in rows).items()}))


def join_rows(plan, records):
    """Finalize sparse labels from committed full and selected-removal cases."""
    output = []
    for row in plan["rows"]:
        if row["full"] not in records or (row["removal"] and row["removal"] not in records):
            raise ValueError(f"Missing required case for {row['id']}")
        full = records[row["full"]]
        y = float(full["y"])
        d = [None, None, None]
        for k, observed in enumerate(row["o"]):
            if not row["b"][k]:
                d[k] = 0.
            elif observed:
                d[k] = y - float(records[row["removal"]]["y"])
        if y < 0 or any(value is not None and value > y + 1e-10 for value in d):
            raise ValueError(f"Invalid recovery damage for {row['id']}")
        output.append(dict(row, y_all=y, d=d, X=full["X"]))
    return output


def exact_counts(plans):
    return dict(sources=len(plans), vectors=sum(len(p["vectors"]) for p in plans),
                rows=sum(len(p["rows"]) for p in plans),
                unique_cases=sum(len(p["cases"]) for p in plans),
                upper_bound_cases=sum((len(p["rows"]) - len(p["noise_seeds"])) * 2 +
                                      len(p["noise_seeds"]) for p in plans))
