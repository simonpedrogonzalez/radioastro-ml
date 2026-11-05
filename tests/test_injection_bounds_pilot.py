"""Analytical convolution and protocol/scheduler regression tests."""
import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.wcs import WCS

from scripts.clean_recovery_noise_sweep import convolve_model, geometry, measure
from scripts.injection_bounds_report import (case, initial_cases, context_list, curve,
    analyze_curve, decisions, cases_at, digest, physical, cell, render)


class ConvolutionTests(unittest.TestCase):
    def setUp(self):
        w=WCS(naxis=2);w.wcs.crpix=[9,9];w.wcs.cdelt=[-1/3600,1/3600]
        w.wcs.crval=[180,30];w.wcs.ctype=['RA---SIN','DEC--SIN'];w.wcs.radesys='FK5';w.wcs.equinox=2000
        self.header=w.to_header();self.header.update(BMAJ=5/3600,BMIN=2/3600,BPA=37)
        self.model=np.zeros((17,17));self.model[8,8]=2

    def test_point_matches_rotated_analytic_truth_and_peak_not_sum_normalized(self):
        rho,truth=geometry(self.header,self.model.shape,2,math.pi,math.pi/6)
        smoothed=convolve_model(self.model,self.header)
        np.testing.assert_allclose(smoothed,truth,atol=2e-9,rtol=2e-7)
        self.assertAlmostEqual(smoothed[8,8],2)
        self.assertGreater(smoothed.sum(),self.model.sum())

    def test_negative_components_and_edge_without_wrap_or_pixel_shift(self):
        model=np.zeros_like(self.model);model[0,0]=-3
        C=convolve_model(model,self.header)
        self.assertAlmostEqual(C[0,0],-3)
        self.assertEqual(np.unravel_index(np.argmin(C),C.shape),(0,0))
        self.assertLess(abs(C[-1,-1]),1e-14)
        np.testing.assert_allclose(convolve_model(-self.model,self.header),-convolve_model(self.model,self.header),atol=1e-14)

    def test_zero_model_exactly_one_error_despite_restored_roundoff(self):
        rho,T=geometry(self.header,self.model.shape,2,math.pi,math.pi/6)
        zero=np.zeros_like(T);C=convolve_model(zero,self.header)
        self.assertFalse(C.any())
        m=measure(T+1e-8,T,zero,T,rho,np.ones_like(T,bool),rho>1,2,T,zero,
                  smoothed_model=C,regular_smoothed_model=T)
        self.assertEqual(m['e_TM_2'],1)
        self.assertIsNone(m['e_pos'])


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(seeds=[0,1,2],sources=[dict(id='a',selected_antennas=[1,4]),dict(id='b',selected_antennas=[1,4]),dict(id='c',selected_antennas=[1,4])])

    def rows(self, points, ctx=None):
        ctx=ctx or dict(source='a',antenna=None,branch='noise')
        result=[]
        for q,y in points:
            for c in cases_at(ctx,q,self.plan['seeds']):result.append(dict(c,y=y,d=y-.01,validity='valid'))
        return result

    def test_initial_grid_deduplication_and_count(self):
        cases=initial_cases(self.plan)
        self.assertEqual(len(cases),579)
        self.assertEqual(len({c['id'] for c in cases}),len(cases))
        self.assertEqual(case('a','phase',180,0,1),case('a','phase',-180,0,1))
        self.assertEqual(case('a','phase',0,0,1),case('a','amp',1,0,4))
        self.assertEqual(sum(c['kind']=='noise' and c['value']==1 for c in cases),9)
        self.assertEqual(sum(c['kind']=='noise' and c['value']==0 for c in cases),3)

    def test_missing_seed_cannot_establish_sustained_region(self):
        ctx=dict(source='a',antenna=None,branch='noise')
        rows=self.rows([(0,.01),(1,.95),(2,.98)])
        rows=[r for r in rows if not (r['value']==2 and r['seed']==1)]
        result=analyze_curve(curve(ctx,rows,self.plan['seeds']))
        self.assertIsNone(result['severe_q']);self.assertTrue(result['missing'])

    def test_nonmonotonic_multiple_crossings_and_boundary(self):
        ctx=dict(source='a',antenna=None,branch='noise')
        result=analyze_curve(curve(ctx,self.rows([(0,.01),(1,.1),(2,.02),(3,.95)]),self.plan['seeds']))
        self.assertEqual(len(result['brackets']['0.05']),3)
        self.assertIn('nonmonotonic',result['status']);self.assertIn('boundary crossing',result['status'])
        self.assertIsNone(result['severe_q'])

    def test_shared_weak_candidates_and_signed_amplitude_transform(self):
        rows=[]
        for ctx in context_list(self.plan):
            if ctx['branch']=='noise':rows+=self.rows([(0,.01),(.5,.1),(1,.2)],ctx)
        _,proposed=decisions(self.plan,rows,refine=True)
        half=[c for c in proposed if c['kind']=='noise' and c['value']==1.25]
        self.assertEqual(len(half),9)
        self.assertAlmostEqual(physical('amp_attenuation',math.log(2)),.5)
        self.assertAlmostEqual(physical('amp_amplification',math.log(2)),2)

    def test_baseline_degraded_not_used_for_bounds(self):
        rows=[]
        for ctx in context_list(self.plan):
            if ctx['branch']=='noise':rows+=self.rows([(0,.1),(.5,.2),(1,.95),(2,.96)],ctx)
        bounds,_=decisions(self.plan,rows)
        noise=next(b for b in bounds['recommendations'] if b['branch']=='noise')
        self.assertEqual(noise['baseline_good_contexts'],0);self.assertIsNone(noise['upper_q'])

    def test_sustained_region_stops_extensions_cap_reports_ceiling(self):
        ctx=dict(source='a',antenna=None,branch='noise')
        rows=self.rows([(0,.01),(.5,.02),(999,.95),(1999,.96)])
        result=analyze_curve(curve(ctx,rows,self.plan['seeds']));self.assertEqual(result['severe_q'],999)
        _,new=decisions(self.plan,rows,extend=True)
        self.assertFalse(any(c['source']=='a' and c['kind']=='noise' for c in new))
        rows=self.rows([(0,.01),(.5,.02),(999,.1),(1999,.2),(3999,.3)])
        bounds,new=decisions(self.plan,rows,extend=True)
        result=next(c for c in bounds['contexts'] if c['source']=='a' and c['branch']=='noise')
        self.assertEqual(result['ceiling_q'],3999);self.assertIn('no severe crossing',result['status'])
        self.assertFalse(any(c['source']=='a' and c['kind']=='noise' for c in new))

    def test_case_identity_extensible_and_recommendation_resume_noop(self):
        c=case('a','noise',2,0);h=digest(c)
        plan=dict(self.plan,arbitrary_new_grid=[4,8])
        self.assertEqual(h,digest(case('a','noise',2,0)))
        rows=self.rows([(0,.01),(.5,.02),(999,.95),(1999,.96)])
        _,first=decisions(plan,rows,refine=True)
        existing={c['id'] for c in first}
        _,second=decisions(plan,rows,refine=True)
        self.assertFalse([c for c in second if c['id'] not in existing])

    def test_partial_report_shows_seed_points_but_no_median_and_links_resolve(self):
        plan=dict(seeds=[0,1,2],sources=[dict(id='a',selected_antennas=[1])])
        cases=[case('a','noise',1,s) for s in plan['seeds']]+[case('a','noise',2,0)]
        rows=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for c in cases:
                valid=c['seed']==0
                r=dict(c,validity='valid' if valid else 'missing',products={},record=str((cell(root,c)/'record.json').relative_to(root)))
                if valid:r.update(y=.01 if c['value']==1 else .03,d=0 if c['value']==1 else .02,y0=.01,
                    raw_model_nonzero_count=1,iterations=4,stopcode=2,signed_total_model_flux_jy=1,e_Fout=0)
                rows.append(r);cell(root,c).mkdir(parents=True)
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'cases.json').write_text(json.dumps(dict(cases=cases,refinement_rounds=0,status='running')))
            with patch('scripts.injection_bounds_report.collect',return_value=rows):render(root)
            svg=(root/'plots/noise.svg').read_text()
            self.assertIn('../sources/a/recovery/cases/'+cases[0]['id']+'/index.html',svg)
            self.assertTrue((cell(root,cases[0])/'index.html').exists())
            bounds=json.loads((root/'bounds.json').read_text())
            self.assertFalse(bounds['complete'])
            ctx=next(c for c in bounds['contexts'] if c['branch']=='noise')
            self.assertIsNone(ctx['control_median']);self.assertIsNone(ctx['severe_q'])


if __name__=='__main__':unittest.main()
