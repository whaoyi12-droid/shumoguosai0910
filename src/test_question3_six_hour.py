"""Validate refund economics, causal vintage blending, rolling clock, and actual physics."""
import json
import unittest

import numpy as np
import pandas as pd

import solve_question3_six_hour as model


def independent_check(frame,inputs,d):
    price,load,pv,_,_=inputs
    n=(load[d]-pv[d])/6
    q=frame.plan_kwh.to_numpy();g=frame.final_regular_kwh.to_numpy()
    c=frame.charge_kwh.to_numpy();b=frame.discharge_kwh.to_numpy();e=frame.emergency_kwh.to_numpy()
    w=frame.surplus_kwh.to_numpy();s=frame.soc_start_kwh.to_numpy();end=frame.soc_end_kwh.to_numpy()
    np.testing.assert_allclose(g+b+e,n+c+w,atol=1e-5,rtol=0)
    np.testing.assert_allclose(end,s+.9*c-b/.9,atol=1e-5,rtol=0)
    np.testing.assert_allclose(s[1:],end[:-1],atol=1e-5,rtol=0)
    cost=price*q+price*1.5*np.maximum(g-q,0)-price*.5*np.maximum(q-g,0)+5*price*e
    np.testing.assert_allclose(frame.total_cost,cost,atol=1e-5,rtol=0)
    assert min(s.min(),end.min())>=1200-1e-5 and max(s.max(),end.max())<=10800+1e-5
    assert min(c.min(),b.min(),e.min(),w.min(),g.min(),q.min())>=-1e-5
    assert max(c.max(),b.max())<=5000/6+1e-5
    assert not np.any((c>1e-5)&((b>1e-5)|(e>1e-5)))


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=model.previous.load_inputs(model.ROOT)
        cls.report=model.blend.calibrate(cls.inputs)
        cls.weights=model.weight_sets(cls.report)
        cls.comparison=[]

    def test_refund_90_and_increase_130(self):
        for demand,expected in ((80.,90.),(120.,130.)):
            g,log=model.optimize([[demand]],[1.],1200.,0.,original=[100.])
            self.assertAlmostEqual(float(g[0]),demand,places=5)
            self.assertAlmostEqual(log['objective'],expected,places=5)

    def test_economics_choose_retain_or_refund(self):
        retained,_=model.optimize([[0.,8.1]],[1.,10.],1200.,0.,original=[10.,0.])
        returned,_=model.optimize([[0.,8.1]],[10.,1.],1200.,0.,original=[10.,0.])
        np.testing.assert_allclose(retained,[10.,0.],atol=1e-5)
        np.testing.assert_allclose(returned,[0.,8.1],atol=1e-5)

    def test_vintage_expiry(self):
        _,_,pv,dates,releases=self.inputs
        issue=pd.Timestamp('2025-03-20 18:00')
        mat=model.blend.vintage_matrix(issue,pv,dates,releases)
        np.testing.assert_array_equal(np.isfinite(mat).sum(axis=1),np.repeat([4,3,2,1],36))
        for age in range(4):
            v=issue-pd.Timedelta(hours=age*6)
            self.assertAlmostEqual(mat[5,age],releases[v][6*age],places=8)

    def test_weight_constraints(self):
        for w in self.weights:
            self.assertAlmostEqual(float(w.sum()),1.)
            self.assertTrue(np.all(w>=0))
            self.assertTrue(np.all(np.diff(w)<=1e-9))

    def test_forecast_and_matured_residual_causality(self):
        price,load,pv,dates,releases=self.inputs
        d=100
        for slot in (0,36,72,108):
            issue=dates[d]+pd.Timedelta(minutes=10*slot)
            before=model.make_scenarios(d,slot,self.inputs,7,self.weights)[0]
            changed_load=load.copy();changed_pv=pv.copy()
            changed_load[d,slot:]+=100000
            changed_load[d+1:]+=100000
            changed_pv[d,slot:]+=100000
            changed_pv[d+1:]+=100000
            changed_releases={t:v+100000 if t>issue else v.copy() for t,v in releases.items()}
            changed=(price,changed_load,changed_pv,dates,changed_releases)
            after=model.make_scenarios(d,slot,changed,7,self.weights)[0]
            np.testing.assert_array_equal(before,after)
            changed_releases[issue]+=1000
            newer=model.make_scenarios(d,slot,changed,7,self.weights)[0]
            self.assertGreater(float(np.max(np.abs(before-newer))),1.)

    def test_january_only_calibration(self):
        price,load,pv,dates,releases=self.inputs
        changed=pv.copy();changed[31:]+=100000
        future={t:v+100000 if t>=pd.Timestamp('2025-02-01') else v for t,v in releases.items()}
        report=model.blend.calibrate((price,load,changed,dates,future))
        self.assertEqual(self.report,report)

    def test_four_seasons(self):
        for date in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
            d=int(self.inputs[3].get_loc(date))
            for name,newest,adjust in (('weighted_refund',False,True),('newest_refund',True,True),
                                       ('weighted_no_adjustment',False,False)):
                df,versions,soc=model.simulate_day(d,6000.,self.inputs,7,self.weights,
                     newest_only=newest,adjustments=adjust)
                independent_check(df,self.inputs,d)
                self.assertEqual([v['first_slot'] for v in versions],[1,37,73,109])
                for v in versions:
                    self.assertEqual(v['lookahead_slots'],144)
                    self.assertEqual(len(v['forecast_kwh']),144)
                    start=v['first_slot']-1
                    np.testing.assert_allclose(df.final_regular_kwh.iloc[start:start+36],
                                               v['regular_draft_kwh'][:36],atol=1e-7)
                np.testing.assert_allclose(df.plan_kwh,versions[0]['regular_draft_kwh'],atol=1e-7)
                self.comparison.append({'date':date,'variant':name,'initial_soc':6000.,'final_soc':soc,
                     'total_cost':float(df.total_cost.sum()),'refund_amount':float(df.refund_amount.sum()),
                     'decrease_kwh':float(np.maximum(df.plan_kwh-df.final_regular_kwh,0).sum()),
                     'emergency_kwh':float(df.emergency_kwh.sum()),'surplus_kwh':float(df.surplus_kwh.sum())})

    def test_year_end(self):
        df,versions,_=model.simulate_day(364,6000.,self.inputs,7,self.weights)
        independent_check(df,self.inputs,364)
        self.assertEqual(len(versions[-1]['forecast_kwh']),144)
        self.assertEqual(df.interval_end.iloc[-1],'2026-01-01 00:00:00')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RevisionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=model.ROOT/'results/q3_revision_review'
    out.mkdir(exist_ok=True)
    model.previous.save_json(out/'revision_tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),
         'failures':[str(x) for x in result.failures+result.errors],
         'controlled_initial_soc':6000.,'seasonal_comparison':RevisionTests.comparison})
    raise SystemExit(not result.wasSuccessful())
