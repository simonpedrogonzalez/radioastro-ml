"""Physical v2 noise coordinate, range tables and all-case slider mapping."""
import math
import tempfile
import unittest
from pathlib import Path

from scripts.injection_bounds_report import NOISE_R_V2, case, initial_cases
from scripts.simulation_parametrization_v2_report import (boundary_markers,
    range_tables, slider_manifest)


def source(sid, snr):
    return dict(id=sid, nominal_source_snr=snr, selected_antennas=[1,2],
                norms={'1':{'V_Ak_over_sigma_L2':10.},'2':{'V_Ak_over_sigma_L2':20.}})


class ParametrizationV2Tests(unittest.TestCase):
    def test_thermal_grid_is_common_in_r_across_original_fluxes(self):
        plan=dict(schema=2,sources=[source('low',560.),source('high',136588.)],seeds=[0,1,2])
        cases=initial_cases(plan)
        for item in plan['sources']:
            values=sorted({c['value']/item['nominal_source_snr'] for c in cases
                           if c['source']==item['id'] and c['kind']=='noise' and c['value']>1})
            for actual,wanted in zip(values,NOISE_R_V2,strict=True):
                self.assertAlmostEqual(actual,wanted,places=12)

    def test_range_table_uses_measured_thermal_r(self):
        item=source('one',100.)
        plan=dict(schema=2,sources=[item],seeds=[0,1,2])
        rows=[]
        for seed in plan['seeds']:
            for r,y in ((.01,.01),(.1,.04),(.2,.2),(.4,.95),(.8,1.)):
                a=r*100
                rows.append(dict(source='one',seed=seed,kind='noise',value=a,
                                 antenna=None,validity='valid',y=y,d=y-.01))
        full,summary,conclusion=range_tables(plan,rows)
        thermal=next(row for row in full if row['branch']=='noise')
        self.assertEqual(thermal['low_bracket'],[.1,.2])
        self.assertAlmostEqual(thermal['strong'],.4)
        bound=next(row for row in conclusion if row['branch']=='noise')
        self.assertAlmostEqual(bound['min'],.2)
        self.assertAlmostEqual(bound['max'],.4)

    def test_interpolated_branch_markers_take_earliest_and_outermost_curves(self):
        plan=dict(sources=[source('low',100.),source('high',200.)],seeds=[0,1])
        rows=[]
        for src in plan['sources']:
            sid=src['id'];snr=src['nominal_source_snr']
            for seed in plan['seeds']:
                rows.append(dict(source=sid,seed=seed,antenna=None,kind='noise',value=1.,
                                 validity='valid',y=0.,d=0.))
                for r in (.1,1.):
                    y=(.1 if sid=='low' and seed==0 else .02) if r==.1 else 1.2
                    rows.append(dict(source=sid,seed=seed,antenna=None,kind='noise',value=r*snr,
                                     validity='valid',y=y,d=y))
                for ant in src['selected_antennas']:
                    reactive=sid=='low' and ant==1 and seed==0
                    for value in (-90.,-30.,30.,90.):
                        y=abs(value)*(.1/30 if reactive else .2/90)
                        rows.append(dict(source=sid,seed=seed,antenna=ant,kind='phase',value=value,
                                         validity='valid',y=y,d=y))
                    for value in (.1,.5,2.,10.):
                        y=({.1:.12,.5:.1,2.:.1,10.:1.3} if reactive else
                           {.1:.08,.5:.02,2.:.02,10.:1.})[value]
                        rows.append(dict(source=sid,seed=seed,antenna=ant,kind='amp',value=value,
                                         validity='valid',y=y,d=y))
        global_markers,source_markers,adjusted=boundary_markers(plan,rows)
        phase=[m['value'] for m in global_markers['phase']]
        self.assertEqual(phase[0],-90.)
        self.assertAlmostEqual(phase[1],-15.)
        self.assertAlmostEqual(phase[2],15.)
        self.assertEqual(phase[3],90.)
        amp=[m['value'] for m in global_markers['amp']]
        self.assertEqual(amp[0],.1)
        self.assertAlmostEqual(amp[1],math.sqrt(.5))
        self.assertAlmostEqual(amp[2],math.sqrt(2.))
        self.assertGreater(amp[3],8.)
        self.assertLess(amp[3],10.)
        self.assertAlmostEqual(global_markers['noise'][0]['value'],math.sqrt(.01*.1))
        self.assertAlmostEqual(source_markers['high']['noise'][0]['value'],200*math.sqrt(.01*.1))
        self.assertFalse(global_markers['phase'][0]['reached'])
        self.assertTrue(global_markers['amp'][-1]['reached'])
        self.assertAlmostEqual(next(x for x in adjusted if x['branch']=='phase_negative')['max'],-15.)

    def test_all_cases_have_slider_stops_and_phase_endpoints_share_record(self):
        item=source('one',100.)
        plan=dict(schema=2,sources=[item],seeds=[0,1])
        cases=[case('one','noise',0)]
        for seed in plan['seeds']:
            cases += [case('one','noise',1,seed),case('one','noise',5,seed)]
            for ant in item['selected_antennas']:
                cases += [case('one','phase',180,seed,ant),case('one','amp',1.5,seed,ant)]
        cases += [case('one','phase',75,0,2)]
        manifest=slider_manifest(plan,cases)
        sliders=manifest['sliders']['one']
        phase=sliders['phase']
        left=next(stop for stop in phase if stop['value']==-180.)
        right=next(stop for stop in phase if stop['value']==180.)
        self.assertEqual(left['cases'],right['cases'])
        sparse=next(stop for stop in phase if stop['value']==75.)
        self.assertEqual(set(sparse['cases']),{'2:0'})
        ids={cid for kind in sliders.values() for stop in kind for cid in stop['cases'].values()}
        self.assertEqual(ids,{c['id'] for c in cases})
        zero=next(stop for stop in sliders['noise'] if stop['value']==0.)
        self.assertEqual(len(set(zero['cases'].values())),1)


if __name__=='__main__':unittest.main()
