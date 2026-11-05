"""Analytical recovery cases, coordinate geometry and report contracts."""
import tempfile
import json
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from scripts.clean_recovery_noise_sweep import geometry, measure, aggregate_training_metric, metric_grid, nominal_snr, METRICS
from scripts.imaging import Beam, DefaultImagingConfig
from scripts.preprocessing.fits import load_fits_plane, validate_fits_triplet


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        y,x=np.indices((9,9)); self.rho=np.hypot(x-4,y-4)/2
        self.T=np.exp(-4*np.log(2)*self.rho**2)
        self.omega=np.ones((9,9),bool);self.annulus=self.rho>1.5
        self.M=np.zeros((9,9));self.M[4,4]=1.
        self.zero=np.zeros((9,9))

    def metrics(self,I=None,R=None,M=None,regular=None):
        return measure(self.T if I is None else I,self.zero if R is None else R,
            self.M if M is None else M,self.T,self.rho,self.omega,self.annulus,1.,
            self.T if regular is None else regular,self.zero)

    def test_perfect_recovery_and_exact_regular_pair(self):
        m=self.metrics()
        for key,value in m.items():
            if key not in ('missing_detection','empty_raw_model','position_roundoff_warning'):self.assertEqual(value,0.,key)
        self.assertFalse(m['missing_detection'])

    def test_source_left_in_residual_has_correct_image_but_no_fitted_model(self):
        m=self.metrics(R=self.T,M=self.zero)
        self.assertEqual(m['e_TI_2'],0.)
        self.assertEqual(m['e_TM_2'],1.)
        self.assertEqual(m['e_Fin'],1.)
        self.assertTrue(m['missing_detection']);self.assertIsNone(m['e_pos'])

    def test_fixed_truth_denominator_and_absolute_false_flux(self):
        model=2*self.M;model[0,0]=.3;model[0,1]=-.2
        m=self.metrics(I=2*self.T,M=model)
        for key in ('e_TI_1','e_TI_2','e_TM_1','e_TM_2','e_Fin'):
            self.assertAlmostEqual(m[key],1.,msg=key)
        self.assertAlmostEqual(m['e_Fout'],.5)
        self.assertAlmostEqual(m['signed_flux_error'],1.)

    def test_full_image_includes_exterior_and_rejects_nonfinite_support(self):
        I=self.T.copy();I[0,0]+=1.
        self.assertGreater(self.metrics(I=I)['e_TI_2'],0.)
        I[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'): self.metrics(I=I)
        self.omega[0,0]=False;self.annulus[0,0]=False
        self.assertEqual(self.metrics(I=I)['e_TI_2'],0.)

    def test_peak_tie_chooses_closest_true_position(self):
        I=self.zero.copy();I[0,0]=2;I[4,5]=2
        self.assertEqual(self.metrics(I=I)['e_pos'],.5)

    def test_empty_model_roundoff_position_is_flagged_without_clipping(self):
        I=self.zero.copy();I[1,1]=np.finfo(np.float32).eps
        m=self.metrics(I=I,M=self.zero)
        self.assertTrue(m['empty_raw_model']);self.assertTrue(m['position_roundoff_warning'])
        self.assertFalse(m['missing_detection']);self.assertGreater(m['e_pos'],0.)

    def test_wcs_beam_rotation_uses_true_coordinate(self):
        w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crpix=[11,11]
        w.wcs.crval=[20.,30.];w.wcs.cdelt=[-.25/3600,.25/3600]
        w.wcs.radesys='FK5';w.wcs.equinox=2000.
        header=w.to_header();header['BMAJ']=2/3600;header['BMIN']=1/3600;header['BPA']=0.
        rho,T=geometry(header,(21,21),3.,np.deg2rad(20),np.deg2rad(30))
        self.assertLess(rho[10,10],1e-7);self.assertAlmostEqual(T[10,10],3.)
        self.assertAlmostEqual(rho[11,10],.25/2,places=6)
        self.assertAlmostEqual(rho[10,11],.25,places=6)
        header['BPA']=90.
        rho,_=geometry(header,(21,21),3.,np.deg2rad(20),np.deg2rad(30))
        self.assertAlmostEqual(rho[11,10],.25,places=6)


class ReportTests(unittest.TestCase):
    def test_nominal_snr_and_compact_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            folder=root/'sources'/'A'/'recovery';folder.mkdir(parents=True)
            (folder/'settings.json').write_text(json.dumps(dict(nominal_source_snr=100.)))
            self.assertEqual(nominal_snr(root,'A'),100.)
            levels=[0.,1.,10.]
            rows=[dict(seed=seed,noise_multiplier=a,**{key:a+seed for key in METRICS}) for seed in (0,1) for a in levels]
            import matplotlib.pyplot as plt
            actual_subplots=plt.subplots
            captured=[]
            def capture_axes(*args,**kwargs):
                figure,axes=actual_subplots(*args,**kwargs)
                captured.append(axes)
                return figure,axes
            with patch.object(plt,'subplots',side_effect=capture_axes):
                name=metric_grid(root,'A',rows,levels,[0,1],100.)
            self.assertTrue((root/name).is_file())
            self.assertEqual((root/name).suffix,'.png')
            self.assertEqual(captured[0].shape,(7,2))
            self.assertEqual(captured[0][-2,0].get_ylabel(),r'$e_{\mathrm{pos}}$ (beam FWHM)')
            self.assertEqual(captured[0][-1,0].get_ylabel(),'Robust residual RMS (Jy/beam)')
            self.assertFalse(captured[0][-2,1].get_visible())
            self.assertFalse(captured[0][-1,1].get_visible())

class TrainingAggregationTests(unittest.TestCase):
    def setUp(self):
        self.levels=[0.,1.,2.,4.,8.]
        self.rows=[dict(source=s,noise_multiplier=a,error=y) for s,values in
            (('A',[0,0,1,2,10]),('B',[0,0,3,4,20])) for a,y in zip(self.levels,values)]

    def test_equal_source_median_excludes_partial_source_at_every_level(self):
        rows=self.rows+[dict(source='pending',noise_multiplier=a,error=1e9) for a in (0.,1.)]
        result=aggregate_training_metric(rows,self.levels,['A','B','pending'],'error')
        self.assertEqual(result['complete_sources'],['A','B'])
        self.assertEqual(result['pending_sources'],['pending'])
        self.assertEqual(result['median_values'],[0.,0.,2.,3.,15.])
        self.assertEqual(result['source_counts'],[2]*5)

    def test_undefined_positions_do_not_remove_the_source_curve(self):
        self.rows[8]['error']=None
        result=aggregate_training_metric(self.rows,self.levels,['A','B'],'error')
        self.assertEqual(result['complete_sources'],['A','B'])
        self.assertEqual(result['source_counts'],[2,2,2,1,2])
        self.assertEqual(result['median_values'][3],2.)

    def test_duplicate_source_level_rejected_and_empty_group_supported(self):
        with self.assertRaisesRegex(ValueError,'one seed'):
            aggregate_training_metric(self.rows+[self.rows[0]],self.levels,['A','B'],'error')
        result=aggregate_training_metric([],self.levels,['A'],'error')
        self.assertEqual(result['source_counts'],[0]*5)


class InterfaceTests(unittest.TestCase):
    def test_restoring_beam_is_opt_in(self):
        self.assertNotIn('restoringbeam',DefaultImagingConfig._base_parameters())
        config=replace(DefaultImagingConfig,restoring_beam=Beam(2.,1.,35.))
        self.assertEqual(config._base_parameters()['restoringbeam'],['2arcsec','1arcsec','35deg'])
        with self.assertRaises(ValueError):replace(config,restoring_beam=Beam(0.,1.,0.))

    def test_jy_pixel_plane_allowed_but_image_triplet_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.fits'
            w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crpix=[3,3]
            w.wcs.crval=[20,30];w.wcs.cdelt=[-.001,.001]
            header=w.to_header();header['BUNIT']='Jy/pixel';header['BMAJ']=.001;header['BMIN']=.001
            fits.writeto(path,np.zeros((5,5)),header)
            self.assertEqual(load_fits_plane(path).unit,'Jy/pixel')
            with self.assertRaisesRegex(ValueError,'Jy/beam'):
                validate_fits_triplet(dict.fromkeys(('dirty','clean','residual'),path))


if __name__=='__main__':unittest.main()
