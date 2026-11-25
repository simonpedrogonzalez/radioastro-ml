"""Synthetic checks of the image features (docs/image_features_plan.md §5); no CASA, no dataset files."""
import math
import unittest

import numpy as np

from scripts.features import energy, fourier, radial
from scripts.report_fourier_toy import phase_centered_fft, toy_header


def header(n):
    h = toy_header(p0=(n / 2, n / 2))
    h["CDELT2"], h["RADESYS"], h["EQUINOX"] = .6 / 3600, "FK5", 2000.   # square pixels, as in the dataset
    h["BMAJ"], h["BMIN"], h["BPA"] = 3 / 3600, 2 / 3600, 30.
    return h


def toy_planes(n=32, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n)) - n / 2
    model = np.zeros((n, n)); model[n // 2, n // 2] = 1.; model[n // 2 + 1, n // 2] = -.2
    return dict(C=rng.normal(size=(n, n)), R=rng.normal(size=(n, n)), omega=np.ones((n, n), bool),
                rho=np.hypot(x, y) / 2, header=header(n), s=2., model=model,
                psf=np.exp(-(x**2 + y**2) / 8) + .01 * rng.normal(size=(n, n)), psf_header=header(n))


class Energy(unittest.TestCase):
    def test_known_signed_image(self):
        x = np.array([2., -1., 0., 0.])
        rms, mean_abs, pos_ms, neg_ms, max_abs = energy.stats(x)
        self.assertAlmostEqual(rms, math.sqrt(5 / 4))
        self.assertAlmostEqual(mean_abs, 3 / 4)
        self.assertAlmostEqual(pos_ms, 1.)
        self.assertAlmostEqual(neg_ms, 1 / 4)
        self.assertEqual(max_abs, 2.)
        self.assertEqual(energy.stats(np.zeros(4)), [0., 0., 0., 0., 0.])

    def test_names_and_model_sums(self):
        values = energy.energy(toy_planes())
        self.assertEqual(tuple(values), energy.NAMES)
        self.assertEqual(len(energy.NAMES), 16)
        self.assertAlmostEqual(values["energy.log10_s"], math.log10(2.))
        self.assertAlmostEqual(values["energy.model.flux_positive_over_s"], .5)
        self.assertAlmostEqual(values["energy.model.flux_negative_magnitude_over_s"], .1)
        self.assertAlmostEqual(values["energy.model.flux_absolute_over_s"], .6)


class Radial(unittest.TestCase):
    def test_regions_frac_contrast_and_empty(self):
        p = toy_planes(32)   # rho reaches 11.3 < 16: the outer R region is empty
        p["R"] = np.where(np.random.default_rng(1).random((32, 32)) < .3, -1., 1.)
        values = radial.radial(p)
        self.assertEqual(len(values), 25)
        self.assertAlmostEqual(sum(values[f"radial.C.r{k}.frac"] for k in range(5)), 1.)
        for k in range(4):
            r = radial.region_masks(p["rho"], p["omega"], radial.R_REGIONS)[k]
            pos = values[f"radial.R.r{k}.pos_contrast"]
            self.assertAlmostEqual(pos + values[f"radial.R.r{k}.neg_contrast"], 1.)
            self.assertAlmostEqual(pos, float((p["R"][r] > 0).mean()))
        self.assertTrue(np.isnan(values["radial.R.r4.pos_contrast"]) and np.isnan(values["radial.R.r4.neg_contrast"]))

    def test_zero_model_gives_nan_frac_and_zero_energy(self):
        p = toy_planes(); p["C"] = np.zeros_like(p["C"])
        values = radial.radial(p)
        self.assertTrue(np.isnan(values["radial.C.r0.frac"]))
        self.assertEqual(values["radial.C.r0.pos_ms"], 0.)


class Fourier(unittest.TestCase):
    def test_independent_cell_count(self):
        for n in (8, 32, 256):
            self.assertEqual(fourier.independent_cells((n, n)).sum(), n * n // 2 + 1)

    def test_parseval_over_independent_cells(self):
        n = 32
        y = np.random.default_rng(2).normal(size=(n, n)); y -= y.mean()
        z, _, _ = phase_centered_fft(y, header(n))
        P = np.abs(z) ** 2
        self.assertAlmostEqual(P.sum(), np.square(y).sum())
        keep = fourier.independent_cells((n, n))
        self_conj = np.zeros((n, n), bool)
        for a in (0, n // 2):
            for b in (0, n // 2):
                self_conj[a, b] = True
        self_conj = np.fft.fftshift(self_conj)
        self.assertAlmostEqual(2 * P[keep & ~self_conj].sum() + P[keep & self_conj].sum(), np.square(y).sum())

    def test_cosine_fringe(self):
        n, h = 64, header(64)
        yy, xx = np.indices((n, n))
        x = np.cos(2 * np.pi * (3 * xx + 5 * yy) / n)
        P, u, v = fourier.power(x, h, np.ones((n, n), bool))
        keep = fourier.independent_cells((n, n))
        values, curve = fourier.spectrum(P, u, v, h, keep)
        self.assertAlmostEqual(values["log10_S.k1"], 0., places=9)
        self.assertAlmostEqual(values["angular_concentration"], 1., places=9)
        top = np.unravel_index(np.argmax(np.where(keep, P, -1)), P.shape)
        fy, fx = np.array(top) - n // 2
        self.assertIn((int(fy), int(fx)), {(5, 3), (-5, -3)})   # the fringe's frequency (or its Hermitian partner)
        beam = math.radians(math.sqrt(h["BMAJ"] * h["BMIN"]))
        self.assertAlmostEqual(values["peak_cycles_per_beam"], math.hypot(u[top], v[top]) * beam)
        self.assertAlmostEqual(math.atan2(v[top], u[top]) % math.pi,
                               math.atan2(*(np.linalg.inv(np.deg2rad(np.array([[-.6, 0], [0, .6]]) / 3600)).T
                                            @ np.array([3 / n, 5 / n]))[::-1]) % math.pi)
        self.assertTrue(np.all(np.diff(curve) >= -1e-12) and abs(curve[-1]) < 1e-12)

    def test_isotropic_noise_and_zero_power(self):
        n, h = 128, header(128)
        P, u, v = fourier.power(np.random.default_rng(3).normal(size=(n, n)), h, np.ones((n, n), bool))
        values, _ = fourier.spectrum(P, u, v, h, fourier.independent_cells((n, n)))
        self.assertLess(values["angular_concentration"], .05)
        values, curve = fourier.spectrum(np.zeros((n, n)), u, v, h, fourier.independent_cells((n, n)))
        self.assertTrue(all(np.isnan(x) for x in values.values()) and np.isnan(curve).all())

    def test_control_median_has_zero_excess(self):
        controls = np.random.default_rng(4).normal(size=(11, 3))
        np.testing.assert_allclose(fourier.excess(np.median(controls, 0), controls), 0.)
        z = fourier.excess(controls, controls)
        np.testing.assert_allclose(np.median(np.abs(z), 0), 1 / 1.4826)

    def test_whitened_residual_equal_to_psf_is_flat(self):
        p = toy_planes(64)
        P_R, _, _ = fourier.power(p["psf"], p["header"], p["omega"])
        keep = fourier.independent_cells((64, 64))
        q, ok = fourier.whitened(P_R, fourier.psf_power(p["psf"], p["psf_header"]), keep)
        self.assertTrue(ok.sum() > 100 and not ok[~keep].any())
        self.assertLess(np.ptp(q[ok]) / q[ok].mean(), 1e-9)

    def test_fourier_block_names(self):
        values, curves = fourier.fourier(toy_planes(32))
        self.assertEqual(tuple(values), fourier.NAMES)
        self.assertEqual(sum(1 for n in fourier.NAMES if not n.startswith("fft.Rq")), 16)
        self.assertEqual(set(curves), {"C", "R", "Rq"})
        self.assertTrue(all(len(c) == len(fourier.CURVE_FRACTIONS) for c in curves.values()))


if __name__ == "__main__":
    unittest.main()
