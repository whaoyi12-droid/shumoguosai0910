"""Synthetic checks of equations and interface; NOT a competition solver."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from model_contract import BATTERY, interval, latest_release_hour, next_soc, settlement, violations


def main():
    checks = []

    def record(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    day = datetime(2025, 2, 1)
    intervals = [interval(day, t) for t in range(1, 145)]
    record("144_records_cover_24_hours", sum((b - a).total_seconds() for a, b in intervals) == 86400)
    record("endpoint_labels_match_user_cycle", intervals[0][1] == day + timedelta(minutes=10)
           and intervals[-1][1] == day + timedelta(days=1))
    record("adjacent_records_and_day_boundary", all(intervals[t][1] == intervals[t + 1][0] for t in range(143))
           and intervals[-1][1] == interval(day + timedelta(days=1), 1)[0])
    record("six_hour_update_does_not_change_finished_segment", latest_release_hour(36) == 0
           and latest_release_hour(37) == 6 and latest_release_hour(72) == 6
           and latest_release_hour(73) == 12 and latest_release_hour(108) == 12
           and latest_release_hour(109) == 18 and latest_release_hour(144) == 18)
    record("power_to_energy_conversion", math.isclose(BATTERY.step_limit_kwh, 5000 / 6))
    record("90_percent_each_direction_means_81_percent_round_trip",
           math.isclose(next_soc(next_soc(6000, 100, 0), 0, 81), 6000))
    record("feasible_charge_balance", not violations(6000, 100, 0, 500, 500, 100, 0, 0))
    record("feasible_discharge_and_emergency", not violations(6000, 300, 19, 0, 400, 0, 81, 0))
    record("reject_soc_underflow", "final_soc_bound" in violations(1200, 0, 0, 0, 100, 0, 100, 0))
    record("reject_simultaneous_charge_discharge", "simultaneous_charge_discharge" in
           violations(6000, 0, 0, 0, 0, 100, 100, 0))
    record("reject_emergency_charging", "emergency_charging" in violations(6000, 0, 200, 0, 100, 100, 0, 0))
    record("reject_power_overrun", "power_limit" in violations(6000, 1000, 0, 0, 0, 1000, 0, 0))
    record("reject_energy_imbalance", "energy_balance" in violations(6000, 1, 0, 0, 0, 0, 0, 0))
    record("plan_paid_even_if_unused", settlement(100, 100, 0, 1)["total_cost"] == 100)
    record("downward_adjustment_without_refund", settlement(100, 80, 0, 1)["total_cost"] == 110)
    record("downward_adjustment_with_refund", settlement(100, 80, 0, 1, refund=True)["total_cost"] == 90)
    record("upward_adjustment_1_5_times", settlement(100, 120, 0, 1)["total_cost"] == 130)
    record("emergency_5_times_separate", settlement(100, 120, 10, 1)["total_cost"] == 180)
    # A future draft is revised from 120 to 110; only FINAL 110 is settled against original 100.
    record("final_revision_not_double_billed", settlement(100, 110, 0, 1)["total_cost"] == 115)
    record("downward_adjustment_dominated_with_free_surplus",
           not violations(6000, 80, 0, 0, 80, 0, 0, 0)
           and not violations(6000, 100, 0, 0, 80, 0, 0, 20)
           and settlement(100, 100, 0, 1)["total_cost"] < settlement(100, 80, 0, 1)["total_cost"])
    record("refund_convex_piecewise_formula", all(math.isclose(
        settlement(100, g, 0, 1, refund=True)["total_cost"], max(50 + 0.5 * g, 1.5 * g - 50))
        for g in (0, 80, 100, 120, 200)))
    # Independent scalar enumeration: uniform synthetic demand 0,...,100 has an 80th-percentile optimum.
    costs = {q: q + 5 * sum(max(d - q, 0) for d in range(101)) / 101 for q in range(101)}
    record("no_storage_newsvendor_80_percent_quantile", min(costs, key=costs.get) == 80)
    report = {"kind": "synthetic_algebra_and_interface_checks_not_optimization_results",
              "passed_count": len(checks), "checks": checks,
              "not_verified": ["solver feasibility or optimality", "forecast quality", "causal full-year replay",
                               "actual savings", "generated result workbooks"]}
    target = Path(__file__).resolve().parents[1] / "docs" / "model_checks.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed_count": len(checks), "data": "synthetic", "report": "docs/model_checks.json"}))


if __name__ == "__main__":
    main()
