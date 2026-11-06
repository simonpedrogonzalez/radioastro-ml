import math
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from scripts.dataset_v2_matched import DEFAULT, assign_shards, join_rows, plan_source, sample_recipes


def source(sid="0000+000", snr=500.):
    return dict(id=sid, nominal_source_snr=snr, reference_sha256="ref",
                norms={"3": dict(V_Ak_over_sigma_L2=1.), "7": dict(V_Ak_over_sigma_L2=2.), "9": dict(V_Ak_over_sigma_L2=0.)})


class Sampling(unittest.TestCase):
    def test_recipes_are_stratified_banded_and_deterministic(self):
        recipes = sample_recipes("0000+000", 500., [3, 7], DEFAULT)
        self.assertEqual(recipes, sample_recipes("0000+000", 500., [3, 7], DEFAULT))
        self.assertEqual(len(recipes), DEFAULT["recipes_per_source"])
        for group in {r["group"] for r in recipes}:
            self.assertEqual(sum(r["group"] == group for r in recipes), DEFAULT["recipes_per_source"] // 4)
        for r in recipes:
            amp = (DEFAULT["weak_ranges"] if r["amp_band"] == "weak" else DEFAULT["ranges"])[f"amp_{r['amp_branch']}"]
            phase = (DEFAULT["weak_ranges"] if r["phase_band"] == "weak" else DEFAULT["ranges"])[f"phase_{r['phase_sign']}"]
            self.assertTrue(amp[0] <= r["A"] <= amp[1] and phase[0] <= r["phi_deg"] <= phase[1])
            self.assertTrue(r["a"] > 1 and r["antenna"] in (3, 7))
            self.assertTrue(r["noise_bounds"][0] <= r["r"] <= r["noise_bounds"][1])

    def test_budget_must_divide_strata(self):
        with self.assertRaises(ValueError):
            sample_recipes("0000+000", 500., [3], dict(DEFAULT, recipes_per_source=6))


class Planning(unittest.TestCase):
    def test_every_row_is_one_case_with_matched_removals(self):
        plan = plan_source(source(), DEFAULT, "train")
        R = DEFAULT["recipes_per_source"]
        self.assertEqual(plan["counts"]["rows"], 7 * R + 1)
        self.assertEqual(plan["counts"]["cases"], 7 * R + 1)
        self.assertEqual(plan["counts"]["units"], R + 1)
        cases = {c["id"]: c for c in plan["cases"]}
        self.assertEqual(len({c["noise_seed"] for c in cases.values()}), 1)
        recipes = {r["id"]: r for r in plan["recipes"]}
        for row in plan["rows"]:
            full = cases[row["case"]]
            for k, name in enumerate(("A", "phi_deg", "a")):
                if not row["b"][k]:
                    self.assertIsNone(row["removal"][k])
                    continue
                removal = cases[row["removal"][k]]
                neutral = (1., 0., 1.)[k]
                self.assertEqual(removal[name], neutral)
                for j, other in enumerate(("A", "phi_deg", "a")):
                    if j != k:
                        self.assertEqual(removal[other], full[other])
            if row["recipe"]:
                r = recipes[row["recipe"]]
                self.assertEqual(full["A"], r["A"] if row["mask"] & 1 else 1.)
        unit_cases = [c for u in plan["units"] for c in u["cases"]]
        self.assertEqual(sorted(unit_cases), sorted(cases))

    def test_noise_seed_never_depends_on_recipes(self):
        a = plan_source(source(), DEFAULT, "train")
        b = plan_source(source(), dict(DEFAULT, recipes_per_source=8), "train")
        self.assertEqual(a["noise_seeds"], b["noise_seeds"])

    def test_shards_are_disjoint_complete_and_deterministic(self):
        units = plan_source(source(), DEFAULT, "train")["units"] + plan_source(source("0001+000"), DEFAULT, "train")["units"]
        cost = {u["id"]: len(u["cases"]) * (2. if u["source"] == "0000+000" else 1.) for u in units}
        for shards in (1, 3, 7):
            owner, loads = assign_shards(units, cost, shards)
            self.assertEqual(set(owner), {u["id"] for u in units})
            self.assertTrue(all(0 <= s < shards for s in owner.values()))
            self.assertEqual(owner, assign_shards(units, cost, shards)[0])
            self.assertAlmostEqual(sum(loads), sum(cost.values()))

    def test_join_gives_complete_labels_and_exact_absent_zeros(self):
        plan = plan_source(source(), DEFAULT, "train")
        rng = np.random.default_rng(0)
        records = {c["id"]: dict(eC=float(rng.uniform(.1, .9)), eD=float(rng.uniform(0, 1)), X=[0.] * 20)
                   for c in plan["cases"]}
        rows = join_rows(plan["rows"], records, lambda a, b: abs(records[a]["eC"] - records[b]["eC"]))
        for row in rows:
            for k in range(3):
                if row["b"][k]:
                    self.assertAlmostEqual(row["d"][k], records[row["case"]]["eC"] - records[row["removal"][k]]["eC"])
                else:
                    self.assertEqual((row["d"][k], row["p"][k]), (0., 0.))


def header(n=64):
    h = fits.Header()
    h.update(NAXIS=2, NAXIS1=n, NAXIS2=n, CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CRVAL1=150., CRVAL2=30.,
             CRPIX1=n // 2 + 1, CRPIX2=n // 2 + 1, CDELT1=-0.1 / 3600, CDELT2=0.1 / 3600, CUNIT1="deg", CUNIT2="deg",
             BMAJ=1.0 / 3600, BMIN=0.6 / 3600, BPA=30., EQUINOX=2000.)
    return h


class Features(unittest.TestCase):
    def test_model6_identities_and_aperture(self):
        from scripts.dataset_v2_matched_extract import model6
        rng = np.random.default_rng(1)
        M = rng.normal(0, 1e-3, (64, 64))
        M[32, 32] = 1.
        omega = np.ones_like(M, bool)
        f, info = model6(M, omega, header(), restored_peak=.5)
        pos, neg, absolute, inside, outside, ratio = f
        self.assertAlmostEqual(pos - neg, inside + outside, places=12)
        self.assertAlmostEqual(absolute, pos + neg, places=12)
        self.assertAlmostEqual(ratio, M.sum() / .5, places=12)
        # FWHM ellipse area ~ pi*BMAJ*BMIN/4 in pixels (10 px x 6 px axes).
        self.assertTrue(abs(info["inside_pixels"] - math.pi * 10 * 6 / 4) < 8)
        self.assertGreater(inside, .99)
        self.assertTrue(math.isnan(model6(M, omega, header(), 0.)[0][5]))

    def test_dirty_qa_uses_the_existing_metric_functions(self):
        from scripts.dataset_v2_matched_extract import image_qa4, parity2
        from scripts.imaging.metrics import summarize_residual_pixels
        from scripts.imaging.models import Beam
        from ml.logreg import _parity_residual
        import torch
        rng = np.random.default_rng(2)
        image = rng.normal(0, 1, (256, 256))
        valid = np.ones_like(image, bool)
        h = header(256)
        q = image_qa4(image, valid, image, valid, h)
        m = summarize_residual_pixels(image.ravel(), pixel_area_arcsec2=.01,
                                      beam=Beam(h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))
        self.assertAlmostEqual(q[1], m.scaled_mad_jy_per_beam)
        self.assertAlmostEqual(q[2], m.peak_over_scaled_mad)
        self.assertAlmostEqual(q[0], image.max() / m.scaled_mad_jy_per_beam)
        self.assertEqual(parity2(image), list(_parity_residual(torch.from_numpy(image.astype(np.float32)))))


class Defects(unittest.TestCase):
    def test_missing_tclean_model_flux_is_none_not_an_error(self):
        from scripts.dataset_v2_matched_extract import tclean_model_flux
        self.assertIsNone(tclean_model_flux({"final_model_flux_jy": None}))
        self.assertIsNone(tclean_model_flux({}))
        self.assertIsNone(tclean_model_flux({"final_model_flux_jy": float("nan")}))
        self.assertEqual(tclean_model_flux({"final_model_flux_jy": .5}), .5)

    def test_missing_feature_is_json_null_then_nan(self):
        import json
        from scripts.create_dataset_v2_matched import features_array
        from scripts.dataset_v2_matched_extract import json_features
        from scripts.preprocessing.schema import atomic_write_json
        import tempfile
        X = json_features([1., 2., float("nan")])
        self.assertEqual(X, [1., 2., None])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            atomic_write_json(path, dict(X=X))
            back = json.loads(path.read_text())["X"]
        array = features_array([dict(X=back)])
        self.assertTrue(np.isnan(array[0, 2]) and array[0, 0] == 1.)

    def test_verify_rejects_missing_or_changed_products(self):
        import tempfile
        from scripts.create_dataset_v2 import file_sha
        from scripts.create_dataset_v2_matched import check_products
        keys = ("dirty", "clean", "residual", "psf", "model", "support")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            products = {k: f"case/{k}.bin" for k in keys}
            (root / "case").mkdir()
            for k in keys:
                (root / products[k]).write_bytes(k.encode())
            hashes = {k: file_sha(root / products[k]) for k in keys}
            check_products(root, products, hashes, "ok")
            (root / products["dirty"]).write_bytes(b"doubled")
            with self.assertRaisesRegex(ValueError, "dirty product changed"):
                check_products(root, products, hashes, "changed")
            (root / products["dirty"]).write_bytes(b"dirty")
            (root / products["model"]).unlink()
            with self.assertRaisesRegex(ValueError, "model product is missing"):
                check_products(root, products, hashes, "missing")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                check_products(root, {k: v for k, v in products.items() if k != "model"},
                               {k: v for k, v in hashes.items() if k != "model"}, "omitted")


class Clean2(unittest.TestCase):
    def test_counts_iterations_and_nonzero_pixels_of_either_sign(self):
        from scripts.dataset_v2_matched_extract import FEATURES, clean2
        model = np.zeros((8, 8)); model[1, 1] = .5; model[2, 3] = -.2; model[4, 4] = np.nan
        self.assertEqual(clean2(model, {"iterdone": 17}), [17., 2.])
        self.assertEqual(FEATURES[-2:], ("clean.iterations", "model.nonzero_pixels"))
        with self.assertRaises(ValueError):
            clean2(model, {})

    def test_old_case_needs_matching_sidecar(self):
        import json, tempfile
        from scripts.create_dataset_v2 import file_sha
        from scripts.create_dataset_v2_matched import EXTRA, _commit, case_features
        from scripts.dataset_v2_matched_extract import BASE20, CLEAN2, FEATURES
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "case.json"
            rec = dict(feature_names=list(BASE20), X=[0.] * 20, product_sha256=dict(model="m"))
            _commit(path, rec)
            with self.assertRaisesRegex(ValueError, "backfill"):
                case_features(path, rec)
            _commit(path.with_name(EXTRA), dict(feature_names=list(CLEAN2), X=[5., 3.], model_sha256="m", case_sha256=file_sha(path)))
            self.assertEqual(case_features(path, rec), [0.] * 20 + [5., 3.])
            _commit(path.with_name(EXTRA), dict(feature_names=list(CLEAN2), X=[5., 3.], model_sha256="other", case_sha256=file_sha(path)))
            with self.assertRaisesRegex(ValueError, "does not belong"):
                case_features(path, rec)
            new = dict(feature_names=list(FEATURES), X=[1.] * 22)
            self.assertEqual(case_features(path, new), [1.] * 22)


if __name__ == "__main__":
    unittest.main()
