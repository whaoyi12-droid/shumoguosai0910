"""Information boundary, physical dispatch, and seasonal integration validation for Q3."""
import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import solve_question3 as q3


class Question3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = q3.load_inputs(q3.ROOT)
        cls.results = []

    def test_copied_forecast_functions(self):
        source = q3.ROOT/'results/问题二_2_加速版.py'
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                    if isinstance(n, ast.FunctionDef)}
        original, copied = functions(source), functions(Path(q3.forecast.__file__))
        for name, definition in copied.items():
            self.assertEqual(definition, original[name], name)

    def test_no_future_leakage(self):
        _, load, pv, dates, releases = self.inputs
        d = 100
        for slot in (0, 36, 72, 108):
            baseline = q3.scenarios(d, slot, load, pv, dates, releases, 7)[0]
            altered_load, altered_pv = load.copy(), pv.copy()
            altered_load[d:] += 123456.
            altered_pv[d, slot:] += 23456.
            altered_pv[d+1:] += 34567.
            issue = dates[d]+pd.Timedelta(minutes=slot*10)
            altered_releases = {stamp: values+99999. if stamp>issue else values.copy()
                                for stamp, values in releases.items()}
            changed = q3.scenarios(d, slot, altered_load, altered_pv, dates, altered_releases, 7)[0]
            np.testing.assert_array_equal(baseline, changed)
            # A just-published forecast must actually affect the optimization input.
            altered_releases[issue] = releases[issue] + 100.
            changed = q3.scenarios(d, slot, load, pv, dates, altered_releases, 7)[0]
            self.assertGreater(float(np.max(np.abs(baseline-changed))), 1.)

    def test_hour_interpolation_and_cross_midnight(self):
        _, load, pv, dates, releases = self.inputs
        d, slot = 100, 108
        prediction = q3.predict(d, slot, load, pv, dates, releases, 7)
        loads = np.concatenate([q3.forecast.predict_load_day(d, j, load, dates, 7) for j in range(3)])[slot:]
        inferred_pv = loads[:144] - 6*prediction[:144]
        hourly = releases[dates[d]+pd.Timedelta(hours=18)]
        np.testing.assert_allclose(inferred_pv[5::6], hourly, atol=1e-9)
        self.assertAlmostEqual(inferred_pv[0], pv[d,107]*5/6+hourly[0]/6, places=8)

    def test_dispatch_extremes(self):
        for soc in (1200., 6000., 10800.):
            for net in (-2000., 0., 2000.):
                for regular in (0., 3000.):
                    c,b,e,w,end = q3.dispatch(net,regular,soc)
                    self.assertAlmostEqual(regular+b+e,net+c+w)
                    self.assertLessEqual(c*b,1e-8)
                    self.assertLessEqual(c*e,1e-8)
                    self.assertTrue(1200-1e-8<=end<=10800+1e-8)

    def test_locked_surplus_is_feasible(self):
        # Locked overpurchase must be allowed to become surplus even at full SOC.
        plan, log = q3.optimize(np.zeros((1, 2)), np.ones(2),10800.,1200.,0., np.full(2,2000.))
        np.testing.assert_allclose(plan, 2000.)
        self.assertEqual(log['status'],0)
        self.assertEqual(log['simultaneous_charge_discharge'],0)

    def test_bad_solver_input_is_rejected(self):
        with self.assertRaises(RuntimeError):
            q3.optimize(np.ones((1,2)),np.ones(2),6000.,20000.,0.)

    def test_seasonal_releases_and_accounting(self):
        for date in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
            d=int(self.inputs[3].get_loc(date))
            for hours in ((0,), (0,6), (0,12), (0,18), (0,6,12,18)):
                # Controlled single-day diagnostics, not annual inherited states.
                frame, versions, soc = q3.simulate_day(d,6000.,self.inputs,7,hours)
                self.assertLess(q3.verify_ledger(frame),1e-5)
                self.assertEqual([v['first_slot'] for v in versions], [h*6+1 for h in hours])
                for i, version in enumerate(versions):
                    start=version['first_slot']-1
                    end=versions[i+1]['first_slot']-1 if i+1<len(versions) else 144
                    np.testing.assert_allclose(frame.final_regular_kwh.iloc[start:end],version['regular_draft_kwh'][:end-start])
                np.testing.assert_allclose(frame.plan_kwh,versions[0]['regular_draft_kwh'][:144])
                for v in versions:
                    self.assertEqual(v['solver']['status'],0)
                self.results.append({'date':date,'hours':list(hours),'initial_soc':6000.,'final_soc':soc,
                                     'total_cost':float(frame.total_cost.sum()),'emergency_kwh':float(frame.emergency_kwh.sum()),
                                     'lp_simultaneous':sum(v['solver']['simultaneous_charge_discharge'] for v in versions),
                                     'lp_emergency_charge':sum(v['solver']['emergency_charge'] for v in versions)})

    def test_year_end_no_future_actual_required(self):
        frame, versions, _ = q3.simulate_day(364,6000.,self.inputs,7)
        self.assertEqual(len(frame),144)
        self.assertEqual(len(versions[-1]['forecast_kwh']),324)
        self.assertLess(q3.verify_ledger(frame),1e-5)


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Question3Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=q3.ROOT/'results/q3_validation'
    out.mkdir(exist_ok=True)
    q3.save_json(out/'tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),
        'failures':[str(x) for x in result.failures+result.errors],
        'note':'Seasonal costs use a controlled 6000 kWh start for each day; not annual results.',
        'seasonal_comparison':Question3Tests.results})
    raise SystemExit(not result.wasSuccessful())
