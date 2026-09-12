"""Validate temporal alignment and causal shared-January reproduction."""
import unittest

import numpy as np

import analyze_question3_time_and_warmup as review


class TimeAndWarmupTests(unittest.TestCase):
    def test_linear_hourly_points_and_energy_units(self):
        knots = np.arange(25, dtype=float)*600
        end = review.disaggregate(knots, 'linear_end')
        np.testing.assert_allclose(end[:6], [100, 200, 300, 400, 500, 600])
        np.testing.assert_allclose(end[5::6], knots[1:])
        self.assertAlmostEqual(float((end[:6]/6).sum()), 350.)
        mean = review.disaggregate(knots, 'linear_interval_mean')
        self.assertAlmostEqual(float((mean[:6]/6).sum()), 300.)
        self.assertEqual(len(end), 144)

    def test_warmup_lp_terminal_bound_and_balance(self):
        q, log = review.q2_warmup_lp(np.array([10.]), np.array([1.]), 6000., 6000., 0.)
        np.testing.assert_allclose(q, [10.], atol=1e-8)
        self.assertLess(log['absolute_dual_gap'], 1e-5)
        self.assertLess(log['max_constraint_violation'], 1e-5)

    def test_future_observations_do_not_change_january(self):
        inputs = review.model.previous.load_inputs(review.model.ROOT)
        original, _, original_soc = review.shared_january(inputs)
        price, load, pv, dates, releases = inputs
        changed_load, changed_pv = load.copy(), pv.copy()
        changed_load[31:] += 100000
        changed_pv[31:] += 200000
        changed, _, changed_soc = review.shared_january((price, changed_load, changed_pv, dates, releases))
        np.testing.assert_allclose(original.select_dtypes('number'), changed.select_dtypes('number'), atol=1e-8, rtol=0)
        self.assertAlmostEqual(original_soc, changed_soc, places=8)
        # Independently verify bus energy, efficiency, and all cross-day state links.
        start, end = original.soc_start_kwh.to_numpy(), original.soc_end_kwh.to_numpy()
        c, b = original.charge_kwh.to_numpy(), original.discharge_kwh.to_numpy()
        raw_net = ((load[7:31]-pv[7:31])/6).ravel()
        np.testing.assert_allclose(original.final_regular_kwh+b+original.emergency_kwh,
                                   raw_net+c+original.surplus_kwh, atol=1e-8, rtol=0)
        np.testing.assert_allclose(end, start+.9*c-b/.9, atol=1e-8, rtol=0)
        np.testing.assert_allclose(start, np.r_[6000., end[:-1]], atol=1e-8, rtol=0)
        self.assertFalse(np.any((c>1e-8)&(b>1e-8)))


if __name__ == '__main__':
    unittest.main()
