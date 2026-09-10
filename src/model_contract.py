"""Shared mathematical conventions; this module is not an optimization solver."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 12000.0
    min_kwh: float = 1200.0
    max_kwh: float = 10800.0
    initial_kwh: float = 6000.0
    power_kw: float = 5000.0
    eta_charge: float = 0.9
    eta_discharge: float = 0.9
    step_hours: float = 1 / 6

    @property
    def step_limit_kwh(self):
        return self.power_kw * self.step_hours


BATTERY = Battery()
SLOTS_PER_CYCLE = 144


def interval(date: datetime, slot: int):
    """The input label is the END of its ten-minute interval."""
    if not 1 <= slot <= SLOTS_PER_CYCLE:
        raise ValueError("slot must be between 1 and 144")
    if date != date.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("date must be midnight")
    end = date + timedelta(minutes=10 * slot)
    return end - timedelta(minutes=10), end


def latest_release_hour(slot: int):
    if not 1 <= slot <= SLOTS_PER_CYCLE:
        raise ValueError("slot must be between 1 and 144")
    return ((slot - 1) // 36) * 6


def next_soc(soc, charge, discharge, battery=BATTERY):
    return soc + battery.eta_charge * charge - discharge / battery.eta_discharge


def settlement(plan, regular, emergency, price, refund=False):
    """One FINAL delivery settlement relative to the ORIGINAL daily plan."""
    if min(plan, regular, emergency) < 0 or price <= 0:
        raise ValueError("nonnegative energies and a positive price are required")
    increase = max(regular - plan, 0)
    reduction = max(plan - regular, 0)
    costs = {
        "plan_cost": price * plan,
        "increase_cost": 1.5 * price * increase,
        "reduction_cost": (-0.5 if refund else 0.5) * price * reduction,
        "emergency_cost": 5 * price * emergency,
    }
    return {**costs, "total_cost": sum(costs.values())}


def violations(soc, regular, emergency, pv, load, charge, discharge, surplus,
               battery=BATTERY, tol=1e-7):
    """Return physical violations for one realized segment (all energies in kWh)."""
    issues = []
    if min(regular, emergency, pv, load, charge, discharge, surplus) < -tol:
        issues.append("negative_energy")
    if max(charge, discharge) > battery.step_limit_kwh + tol:
        issues.append("power_limit")
    if charge > tol and discharge > tol:
        issues.append("simultaneous_charge_discharge")
    if emergency > tol and charge > tol:
        issues.append("emergency_charging")
    if emergency > max(load - pv, 0) + tol:
        issues.append("emergency_exceeds_net_load")
    if not battery.min_kwh - tol <= soc <= battery.max_kwh + tol:
        issues.append("initial_soc_bound")
    final = next_soc(soc, charge, discharge, battery)
    if not battery.min_kwh - tol <= final <= battery.max_kwh + tol:
        issues.append("final_soc_bound")
    residual = regular + emergency + pv + discharge - load - charge - surplus
    if abs(residual) > tol:
        issues.append("energy_balance")
    return issues
