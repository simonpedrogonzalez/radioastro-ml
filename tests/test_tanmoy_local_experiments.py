import math
import unittest

import numpy as np

from scripts.tanmoy_local_experiments import (angular_cut, baseline_gain, central_crop, check_cached, construct, exact,
                                              fourier_cell, scaled_grid, storage_tolerance, uv_grid)


def toy(seed=0, rows=40, A=3.0):
    rng = np.random.default_rng(seed)
    ant1 = rng.integers(0, 6, rows)
    ant2 = (ant1 + rng.integers(1, 5, rows)) % 6
    V = (rng.normal(size=(2, 3, rows)) + 1j * rng.normal(size=(2, 3, rows))).astype(np.complex64).astype(np.complex128)
    n = 50 * (rng.normal(size=V.shape) + 1j * rng.normal(size=V.shape))
    return V, n, baseline_gain(ant1, ant2, 3, A), ant1, ant2


class VisibilityEquations(unittest.TestCase):
    def test_both_orders_match_their_equations(self):
        V, n, G, *_ = toy()
        for order in ("noise", "current", "alternative"):
            stored = construct(V, n, G, order)
            self.assertEqual(stored.dtype, np.complex64)
            want = exact(V, n, G, order)
            self.assertLessEqual(np.abs(stored - want).max(), storage_tolerance(want, n, V))
        self.assertTrue(np.allclose(exact(V, n, G, "current"), G[None, None, :] * V + n))
        self.assertTrue(np.allclose(exact(V, n, G, "alternative"), G[None, None, :] * (V + n)))

    def test_gain_convention(self):
        G = baseline_gain(np.array([3, 0, 3, 1]), np.array([0, 3, 3, 2]), 3, 2.0, 90.0)
        np.testing.assert_allclose(G, [2j, -2j, 4, 1], atol=1e-12)

    def test_unchanged_baselines_identical_and_difference_is_G_minus_1_times_n(self):
        V, n, G, *_ = toy(1)
        cur = construct(V, n, G, "current").astype(np.complex128)
        alt = construct(V, n, G, "alternative").astype(np.complex128)
        aff = G != 1
        self.assertTrue(aff.any() and (~aff).any())
        np.testing.assert_array_equal(cur[..., ~aff], alt[..., ~aff])
        np.testing.assert_array_equal(cur[..., ~aff], construct(V, n, G, "noise")[..., ~aff])
        want = (G[None, None, :] - 1) * n
        tol = storage_tolerance(G[None, None, :] * V, G[None, None, :] * n)
        self.assertLessEqual(np.abs((alt - cur) - want)[..., aff].max(), tol)

    def test_unit_gain_and_zero_noise_equivalence(self):
        V, n, _, ant1, ant2 = toy(2)
        G1 = baseline_gain(ant1, ant2, 3, 1.0)
        for order in ("current", "alternative"):
            np.testing.assert_array_equal(construct(V, n, G1, order), construct(V, n, G1, "noise"))
        _, _, G, *_ = toy(2)
        zero = np.zeros_like(V)
        np.testing.assert_array_equal(construct(V, zero, G, "current"), construct(V, zero, G, "alternative"))


class Bookkeeping(unittest.TestCase):
    def test_cache_rejects_changed_settings(self):
        self.assertFalse(check_cached(None, "k"))
        self.assertTrue(check_cached(dict(id="x", key="k", status="complete"), "k"))
        self.assertFalse(check_cached(dict(id="x", key="k", status="failed"), "k"))
        with self.assertRaises(ValueError):
            check_cached(dict(id="x", key="other", status="complete"), "k")

    def test_scaled_grids_share_angular_field_and_centre(self):
        ref = dict(beam=dict(major_arcsec=1.67), image_grid=dict(imsize=[256, 256], cell_arcsec=[0.3, 0.3],
                                                                    field_of_view_arcsec=[76.8, 76.8]))
        for imsize, factor in ((256, 1), (512, 2), (1024, 4)):
            g = scaled_grid(ref, imsize, factor)["image_grid"]
            self.assertAlmostEqual(g["field_of_view_arcsec"][0], 76.8)
            self.assertAlmostEqual(g["cell_arcsec"][0], 0.3 / factor)
            # CASA reference pixel nx//2 sits at the same angular offset from the field edge
            self.assertAlmostEqual((imsize // 2) * g["cell_arcsec"][0], 128 * 0.3)
        self.assertEqual(ref["image_grid"]["imsize"], [256, 256])   # reference left untouched

    def test_uv_grids_keep_pixels_and_refine_fourier_cell(self):
        ref = dict(beam=dict(major_arcsec=1.67), image_grid=dict(imsize=[256, 256], cell_arcsec=[0.29664, 0.29664],
                                                                    field_of_view_arcsec=[75.94, 75.94]))
        base = fourier_cell(256, 0.29664)
        self.assertAlmostEqual(base, 1 / (256 * 0.29664 * math.pi / 180 / 3600))
        self.assertAlmostEqual(base, 2716, delta=1)        # plan table: about 2716 wavelengths
        for n in (256, 512, 1024):
            g = uv_grid(ref, n)["image_grid"]
            self.assertEqual(g["cell_arcsec"], [0.29664, 0.29664])
            self.assertAlmostEqual(g["field_of_view_arcsec"][0], n * 0.29664)
            self.assertAlmostEqual(fourier_cell(n, 0.29664) * n / 256, base)
            s = central_crop(n, 256)
            self.assertEqual(s.stop - s.start, 256)
            self.assertEqual(n // 2 - s.start, 128)        # native centre pixel = crop centre pixel, no interpolation
            # same angular offset of every crop pixel from the phase centre on every grid
            np.testing.assert_allclose((np.arange(n)[s] - n // 2) * g["cell_arcsec"][0], (np.arange(256) - 128) * 0.29664)
        self.assertEqual(central_crop(256, 256), slice(0, 256))
        with self.assertRaises(ValueError):
            central_crop(128, 256)

    def test_angular_cut_is_conjugate_paired(self):
        ang = np.radians(np.arange(360) + 0.5)   # avoid exact boundary angles
        u, v = np.cos(ang), np.sin(ang)
        cut = angular_cut(u, v, 0.5)
        np.testing.assert_array_equal(cut, angular_cut(-u, -v, 0.5))
        self.assertAlmostEqual(cut.mean(), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
