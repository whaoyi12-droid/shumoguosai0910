"""Forecast-startup and monthly calibration information-boundary checks."""
import unittest

import numpy as np
import pandas as pd

import analyze_question3_monthly_weights as monthly
import solve_question3_six_hour as model


class WeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=model.previous.load_inputs(model.ROOT)
        cls.blocks=monthly.collect_samples(cls.inputs)

    def test_first_three_releases_use_latest(self):
        for slot in (0,36,72):
            mixed=model.predict(0,slot,self.inputs,7,model.weight_sets())
            latest=model.predict(0,slot,self.inputs,7,model.weight_sets(),newest_only=True)
            np.testing.assert_array_equal(mixed,latest)

    def test_fourth_release_uses_valid_weighted_versions(self):
        price,load,pv,dates,releases=self.inputs
        # A synthetic version difference ensures the test detects ignored weights.
        releases={k:v.copy() for k,v in releases.items()}
        releases[pd.Timestamp('2025-01-01 18:00')]+=1000
        inputs=(price,load,pv,dates,releases)
        weights=model.weight_sets()
        result=model.predict(0,108,inputs,7,weights)
        matrix=model.blend.vintage_matrix(pd.Timestamp('2025-01-01 18:00'),pv,dates,releases)
        solar=np.concatenate([matrix[b*36:(b+1)*36,:4-b]@weights[b] for b in range(4)])
        # For origin_idx=0, teammate load forecast is the documented zero fallback.
        np.testing.assert_allclose(result,-solar/6,atol=1e-10)
        latest=model.predict(0,108,inputs,7,weights,newest_only=True)
        self.assertGreater(float(np.max(np.abs(result-latest))),1.)

    def test_mature_target_cutoff_and_weight_constraints(self):
        for month in range(2,13):
            cutoff=pd.Timestamp(2025,month,1)
            result=monthly.fit_asof(self.blocks,cutoff-pd.offsets.MonthBegin(1),cutoff)
            for record in result:
                self.assertLessEqual(pd.Timestamp(record['latest_available_at']),cutoff)
                w=np.asarray(record['weights'])
                self.assertAlmostEqual(float(w.sum()),1.)
                self.assertTrue(np.all(w>=0))
                self.assertTrue(np.all(np.diff(w)<=1e-8))

    def test_future_actuals_and_releases_do_not_change_march_weights(self):
        price,load,pv,dates,releases=self.inputs
        cutoff=pd.Timestamp('2025-03-01')
        before=monthly.fit_asof(self.blocks,'2025-02-01',cutoff)
        changed=pv.copy();changed[59:]+=1e6
        forecasts={k:v+1e6 if k>=cutoff else v.copy() for k,v in releases.items()}
        after_blocks=monthly.collect_samples((price,load,changed,dates,forecasts))
        after=monthly.fit_asof(after_blocks,'2025-02-01',cutoff)
        self.assertEqual(before,after)

    def test_prior_day_release_with_future_targets_is_not_mature(self):
        record=self.blocks[3]
        cutoff=pd.Timestamp('2025-03-01')
        chosen=monthly.select(record,'2025-02-01',cutoff)
        position=np.flatnonzero(record['issue']==pd.Timestamp('2025-02-28 18:00'))[0]
        self.assertFalse(chosen[position])

    def test_matched_february_weights_and_monthly_switch(self):
        schedules={mode:model.weight_schedule.build_schedule(self.inputs,mode)
                   for mode in ('january-full','previous-month','expanding')}
        reference=schedules['january-full']['2025-02']['blocks']
        self.assertEqual(reference,schedules['previous-month']['2025-02']['blocks'])
        self.assertEqual(reference,schedules['expanding']['2025-02']['blocks'])
        self.assertNotEqual(schedules['expanding']['2025-03']['blocks'],reference)
        self.assertEqual(schedules['january-full']['2025-03']['blocks'],reference)

    def test_march_weight_provenance_reaches_four_solves(self):
        schedule=model.weight_schedule.build_schedule(self.inputs,'expanding')
        entry=schedule['2025-03']
        weights=[np.asarray(b['weights']) for b in entry['blocks']]
        d=int(self.inputs[3].get_loc('2025-03-01'))
        frame,versions,_=model.simulate_day(d,6000.,self.inputs,7,weights,weight_metadata=entry)
        self.assertLess(model.verify_ledger(frame),1e-5)
        for version in versions:
            self.assertEqual(version['weight_training'],entry)
            self.assertEqual(version['weights'],[w.tolist() for w in weights])
            self.assertLessEqual(pd.Timestamp(entry['latest_training_target']),pd.Timestamp(version['issue_time']))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(WeightTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=model.ROOT/'results/q3_monthly_weight_review'
    out.mkdir(exist_ok=True)
    model.previous.save_json(out/'tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),
        'failures':[str(x) for x in result.failures+result.errors]})
    raise SystemExit(not result.wasSuccessful())
