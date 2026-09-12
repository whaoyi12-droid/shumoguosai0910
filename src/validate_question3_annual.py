"""Independent audit of a Q3 run: recompute from raw attachments, do not reuse the solver's checks.

Re-reads 附件1/附件2 directly, rebuilds the SOC chain and settlement from the per-day ledgers,
and aggregates the LP diagnostics stored in every *_versions.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH, LIMIT, ETA, TOL = 1200., 10800., 5000 / 6, .9, 1e-5
COLUMNS = ('plan_kwh', 'final_regular_kwh', 'emergency_kwh', 'charge_kwh', 'discharge_kwh',
           'surplus_kwh', 'net_kwh', 'price')


def raw_inputs(base):
    price = pd.read_excel(base / '附件/附件1.xlsx', sheet_name='Sheet1')['电价'].values[:144].astype(float)
    load = pd.read_excel(base / '附件/附件2.xlsx', sheet_name='小区负载', index_col=0, parse_dates=True)
    pv = pd.read_excel(base / '附件/附件2.xlsx', sheet_name='光伏发电实际功率', index_col=0, parse_dates=True)
    return price, load.values[:, :144].astype(float), np.maximum(pv.values[:, :144].astype(float), 0), \
        pd.DatetimeIndex(load.index)


def load_days(folder):
    days = {}
    for csv in sorted(folder.glob('*_ledger.csv')):
        day = csv.name[:10]
        frame = pd.read_csv(csv)
        if len(frame) != 144 or frame.slot.tolist() != list(range(1, 145)):
            raise ValueError(f'{day}: slot structure invalid')
        checkpoint = json.loads((folder / f'{day}_complete.json').read_text(encoding='utf-8'))
        if hashlib.sha256(csv.read_bytes()).hexdigest() != checkpoint['ledger_sha256']:
            raise ValueError(f'{day}: ledger hash does not match its checkpoint')
        days[day] = frame
    return days


def audit(folder, base):
    price, load, pv, dates = raw_inputs(base)
    days = load_days(folder)
    order = sorted(days)
    if order != pd.date_range(order[0], order[-1]).strftime('%Y-%m-%d').tolist():
        raise ValueError('Ledger days are not contiguous')

    issues, worst = [], {}
    metrics = {}
    for day in order:
        frame = days[day]
        d = int(dates.get_loc(pd.Timestamp(day)))
        # Independently rebuilt net load and price, straight from the attachments.
        metrics['net'] = max(metrics.get('net', 0), float(np.max(np.abs(frame.net_kwh - (load[d] - pv[d]) / 6))))
        metrics['price'] = max(metrics.get('price', 0), float(np.max(np.abs(frame.price - price))))
        q, g, e = (frame[name].to_numpy() for name in ('plan_kwh', 'final_regular_kwh', 'emergency_kwh'))
        c, b, w = (frame[name].to_numpy() for name in ('charge_kwh', 'discharge_kwh', 'surplus_kwh'))
        start, end = frame.soc_start_kwh.to_numpy(), frame.soc_end_kwh.to_numpy()
        metrics['balance'] = max(metrics.get('balance', 0), float(np.max(np.abs(g + b + e - frame.net_kwh - c - w))))
        metrics['soc'] = max(metrics.get('soc', 0), float(np.max(np.abs(end - start - ETA * c + b / ETA))))
        if len(start) > 1:
            metrics['chain'] = max(metrics.get('chain', 0), float(np.max(np.abs(start[1:] - end[:-1]))))
        expected = np.c_[frame.price * q, 1.5 * frame.price * np.maximum(g - q, 0),
                         .5 * frame.price * np.maximum(q - g, 0), 5 * frame.price * e]
        actual = frame[['plan_cost', 'increase_cost', 'reduction_cost', 'emergency_cost']].to_numpy()
        metrics['settlement'] = max(metrics.get('settlement', 0), float(np.max(np.abs(expected - actual))))
        metrics['total'] = max(metrics.get('total', 0), float(np.max(np.abs(expected.sum(1) - frame.total_cost))))
        if min(start.min(), end.min()) < LOW - TOL or max(start.max(), end.max()) > HIGH + TOL:
            issues.append(f'{day}: SOC outside [{LOW}, {HIGH}]')
        if min(c.min(), b.min(), e.min(), w.min(), q.min(), g.min()) < -TOL:
            issues.append(f'{day}: negative quantity')
        if max(c.max(), b.max()) > LIMIT + TOL:
            issues.append(f'{day}: power above {LIMIT}')
        if np.any((c > TOL) & ((b > TOL) | (e > TOL))):
            issues.append(f'{day}: simultaneous charge with discharge/emergency')
        if np.any(g < q - TOL):
            issues.append(f'{day}: final regular below the locked original plan')

    for a, b in zip(order, order[1:]):
        if abs(days[a].soc_end_kwh.iloc[-1] - days[b].soc_start_kwh.iloc[0]) > TOL:
            issues.append(f'{a}->{b}: SOC chain break')
    worst.update(metrics)

    annual = pd.concat([days[day] for day in order if day >= '2025-02-01'], ignore_index=True)
    if len(annual) != 334 * 144:
        raise ValueError('Annual window is not 334 days')

    lps, seconds, residuals, bound_violations = [], 0., 0., 0.
    two_sided = emergency_charge = 0
    nonoptimal = []
    for version_file in folder.glob('*_versions.json'):
        for record in json.loads(version_file.read_text(encoding='utf-8')):
            log = record['solver']
            lps.append(record)
            seconds += log['seconds']
            residuals = max(residuals, log['equality_residual'])
            bound_violations = max(bound_violations, log['bound_violation'])
            two_sided += log['simultaneous_charge_discharge']
            emergency_charge += log['emergency_charge']
            if log['status'] != 0:
                nonoptimal.append((version_file.name, log['status']))

    monthly = annual.groupby(annual.date.str[:7]).agg(
        days=('slot', 'size'), plan=('plan_cost', 'sum'), increase=('increase_cost', 'sum'),
        reduction=('reduction_cost', 'sum'), emergency=('emergency_cost', 'sum'),
        emergency_kwh=('emergency_kwh', 'sum'), surplus_kwh=('surplus_kwh', 'sum'),
        cost=('total_cost', 'sum'))
    monthly['days'] //= 144

    report = {
        'ledger_days': len(order), 'first_day': order[0], 'last_day': order[-1],
        'annual_days': int(len(annual) / 144),
        'annual_cost': float(annual.total_cost.sum()),
        'cost_breakdown': {name: float(annual[name].sum()) for name in
                           ('plan_cost', 'increase_cost', 'reduction_cost', 'emergency_cost')},
        'annual_emergency_kwh': float(annual.emergency_kwh.sum()),
        'annual_surplus_kwh': float(annual.surplus_kwh.sum()),
        'annual_charge_kwh': float(annual.charge_kwh.sum()),
        'annual_discharge_kwh': float(annual.discharge_kwh.sum()),
        'annual_net_kwh': float(annual.net_kwh.sum()),
        'initial_soc': float(annual.soc_start_kwh.iloc[0]),
        'final_soc': float(annual.soc_end_kwh.iloc[-1]),
        'max_residuals': worst,
        'issues': issues,
        'lp_count': len(lps), 'lp_total_seconds': seconds,
        'lp_max_seconds': max(r['solver']['seconds'] for r in lps),
        'lp_status_all_zero': not nonoptimal, 'lp_nonoptimal': nonoptimal[:10],
        'lp_max_equality_residual': residuals, 'lp_max_bound_violation': bound_violations,
        'lp_simultaneous_charge_discharge_slots': two_sided,
        'lp_emergency_charge_slots': emergency_charge,
        'lp_scenarios': sorted({r['solver']['scenarios'] for r in lps}),
        'lp_horizons': sorted({r['solver']['horizon'] for r in lps}),
        'monthly': monthly.round(4).reset_index().to_dict('records'),
    }
    return report, annual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--base-dir', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    report, annual = audit(args.folder, args.base_dir)
    out = args.out or args.folder / 'annual_audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # The audited annual ledger itself; the per-month aggregation is inside the JSON above.
    annual.to_csv(out.with_name('annual_audit_ledger.csv'), index=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'monthly'}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(report['monthly']).to_string(index=False))
