"""Analytical coordinate/coverage checks; run with the ML Python environment."""
import unittest

import numpy as np
from astropy.wcs import WCS
from scipy import sparse

from scripts.report_fourier_toy import phase_centered_fft, run_checks, toy_header
from scripts.report_v2_pilots import bilinear_cells, score_map, source_info, visibility_display_limits


class VisibleFeatureTests(unittest.TestCase):
    def test_fft_matches_direct_sinusoids_and_scaled_power(self):
        checks = run_checks()
        self.assertEqual(len(checks), 6)
        self.assertLess(max(c['direct_error'] for c in checks), 1e-10)

    def test_bilinear_conjugate_fold_and_mass(self):
        header = toy_header()
        j = np.deg2rad(WCS(header).pixel_scale_matrix)
        q = np.linalg.inv(j).T @ (np.array([3.25, 4.75]) / 256)
        cells, weights, inside = bilinear_cells(
            np.array([q[0], -q[0]]), np.array([q[1], -q[1]]), header, (256,256))
        self.assertTrue(inside.all())
        np.testing.assert_allclose(weights.sum(axis=0), 1.)
        maps = [dict(zip(cells[:,i], weights[:,i])) for i in range(2)]
        self.assertEqual(set(maps[0]), set(maps[1]))
        for cell in maps[0]:
            self.assertAlmostEqual(maps[0][cell], maps[1][cell])
        expected = {(128+4)*256+128+3: .1875, (128+4)*256+128+4: .0625,
                    (128+5)*256+128+3: .5625, (128+5)*256+128+4: .1875}
        for cell, weight in expected.items():
            self.assertAlmostEqual(maps[0][cell], weight)

    def test_dc_nyquist_and_outside_are_not_wrapped(self):
        header = toy_header()
        j = np.deg2rad(WCS(header).pixel_scale_matrix)
        f = np.array([[0., .5, .75], [0., .1, .1]])
        q = np.linalg.inv(j).T @ f
        _, weights, inside = bilinear_cells(q[0], q[1], header, (256,256))
        np.testing.assert_array_equal(inside, [True,False,False])
        np.testing.assert_array_equal(weights, 0.)

    def test_coverage_weighted_scores_do_not_reward_count_alone(self):
        # Three independent cells; multiplying a baseline's occupancy must not
        # multiply its mean power. A completely flagged pair supplies no vote.
        z=np.array([[1.,2.,3.,100.]],complex)
        h=sparse.csr_matrix([[1.,0.,0.,0.],[0.,100.,0.,0.],[0.,0.,5.,0.],[0.,0.,0.,0.]])
        pairs=np.array([[0,1],[0,2],[1,2],[3,4]])
        scores,ants,summary=score_map(z,h,pairs)
        np.testing.assert_allclose(scores,[1,4,9,0])
        np.testing.assert_array_equal(ants,[0,1,2])
        np.testing.assert_allclose(summary,[2.5,5,6.5])

    def test_held_out_source_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError,'training-only'):
            source_info('1146+399')

    def test_invalid_fft_support_rejected(self):
        a=np.ones((256,256));a[0,0]=np.nan
        with self.assertRaises(ValueError):
            phase_centered_fft(a,toy_header())

    def test_visibility_limits_pool_all_cases_and_include_extremes(self):
        cases={'first':{'visibility':np.array([1+0j,2j])},
               'second':{'visibility':np.array([-3+0j,-4j])}}
        bounds=visibility_display_limits(cases)
        self.assertEqual((bounds['amplitude']['min'],bounds['amplitude']['max']),(1.,4.))
        self.assertEqual((bounds['phase']['min'],bounds['phase']['max']),(-90.,180.))
        for b in bounds.values():
            self.assertLess(b['axis_min'],b['min'])
            self.assertGreater(b['axis_max'],b['max'])


if __name__ == '__main__':
    unittest.main()
