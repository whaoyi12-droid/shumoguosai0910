"""Audit hourly PV disaggregation and Q2-consistent January SOC; never overwrite annual runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import bmat, diags, eye

import solve_question3_six_hour as model


def q2_warmup_lp(net, price, soc, floor, lam):
    """Exact single-scenario Q2 warmup LP, expressed with SciPy instead of PuLP."""
    h = len(net)
    ident = eye(h, format='csr')
    recurrence = ident - diags(np.ones(h-1), -1, shape=(h, h))
    matrix = bmat([[ident, -ident, ident, ident, -ident, None],
                   [None, -.9*ident, ident/.9, None, None, recurrence]], format='csr')
    rhs = np.r_[net, soc, np.zeros(h-1)]
    cost = np.r_[price, np.full(2*h, 1e-5), 5*price, np.full(h, 1e-4), np.zeros(h)]
    cost[-1] = -lam
    low = np.r_[np.zeros(5*h), np.full(h, 1200.)]
    low[-1] = max(1200., floor)
    high = np.r_[np.full(h, np.inf), np.full(2*h, 5000/6),
                 np.maximum(net, 0), np.maximum(-net, 0), np.full(h, 10800.)]
    result = linprog(cost, A_eq=matrix, b_eq=rhs, bounds=np.c_[low, high], method='highs',
                     options={'time_limit': 60, 'primal_feasibility_tolerance': 1e-8,
                              'dual_feasibility_tolerance': 1e-8})
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f'Q2 January LP failed: {result.message}')
    x = result.x
    violation = max(np.max(np.abs(matrix@x-rhs)), np.max(np.maximum(low-x, 0)),
                    np.max(np.maximum(x-high, 0)))
    finite = np.isfinite(high)
    dual = rhs@result.eqlin.marginals + low@result.lower.marginals + high[finite]@result.upper.marginals[finite]
    dual_gap = abs(result.fun-dual)
    if violation > 1e-5 or dual_gap > 1e-5:
        raise RuntimeError('Q2 January LP primal/dual certificate failed')
    return x[:h], {'status': int(result.status), 'objective': float(result.fun),
                   'max_constraint_violation': float(violation), 'absolute_dual_gap': float(dual_gap),
                   'horizon': h, 'floor': float(floor), 'lambda': float(lam)}


def shared_january(inputs):
    price, load, pv, dates, _ = inputs
    soc = 6000.
    rows, records = [], []
    for d in range(7, 31):
        predicted = model.forecast.forecast_three_days(d, load, pv, dates, 7)
        floor = model.forecast.compute_terminal_soc_floor(d, load, pv, dates)
        q, log = q2_warmup_lp(predicted, np.tile(price, 3), soc, floor,
                             model.forecast.estimate_lambda(price))
        records.append({'date': str(dates[d].date()), 'initial_soc': soc,
                        'solver': log, 'plan_432_kwh': q.tolist()})
        for t in range(144):
            net = (load[d, t]-pv[d, t])/6
            c, b, e, w, end = model.previous.dispatch(net, q[t], soc)
            pc, ec = price[t]*q[t], 5*price[t]*e
            rows.append(dict(date=str(dates[d].date()), slot=t+1,
                             interval_start=str(dates[d]+pd.Timedelta(minutes=t*10)),
                             interval_end=str(dates[d]+pd.Timedelta(minutes=(t+1)*10)),
                             net_kwh=net, plan_kwh=q[t], final_regular_kwh=q[t],
                             charge_kwh=c, discharge_kwh=b, emergency_kwh=e, surplus_kwh=w,
                             soc_start_kwh=soc, soc_end_kwh=end, price=price[t],
                             plan_cost=pc, increase_cost=0., refund_amount=0., reduction_cost=0.,
                             emergency_cost=ec, total_cost=pc+ec))
            soc = end
    frame = pd.DataFrame(rows)
    model.verify_ledger(frame)
    return frame, records, soc


def disaggregate(knots, method):
    """Hourly point powers -> ten-minute representative powers; no forecast of SOC."""
    ends = np.interp(np.arange(1, 145)/6, np.arange(25), knots)
    if method == 'linear_end':
        return ends
    if method == 'linear_interval_mean':
        starts = np.interp(np.arange(144)/6, np.arange(25), knots)
        return (starts+ends)/2
    if method == 'next_hour_constant':
        return np.repeat(knots[1:], 6)
    raise ValueError(method)


def resolution_review(inputs, weights):
    _, _, pv, dates, releases = inputs
    methods = ['linear_end', 'linear_interval_mean', 'next_hour_constant']
    errors = {m: [] for m in methods}
    oracle_errors = {m: [] for m in methods}
    for d in range(31, 365):
        boundary = pv[d-1, -1]
        oracle = np.r_[boundary, pv[d, 5::6]]
        for method in methods:
            oracle_errors[method].append(disaggregate(oracle, method)-pv[d])
        for hour in (0, 6, 12, 18):
            issue = dates[d]+pd.Timedelta(hours=hour)
            candidates = {m: [] for m in methods}
            for age in range(4):
                vintage = issue-pd.Timedelta(hours=6*age)
                vi = int(dates.get_loc(vintage.normalize()))
                slot = vintage.hour*6
                boundary = pv[vi, slot-1] if slot else pv[vi-1, -1]
                knots = np.r_[boundary, releases[vintage]]
                for method in methods:
                    candidates[method].append(disaggregate(knots, method)[age*36:age*36+36])
            actual = pv[d, hour*6:hour*6+36]
            for method in methods:
                errors[method].append(weights@np.asarray(candidates[method])-actual)
    def scores(values):
        err = np.concatenate(values)
        return {'samples': int(err.size), 'rmse_kw': float(np.sqrt(np.mean(err**2))),
                'mae_kw': float(np.mean(abs(err))), 'bias_kw': float(err.mean()),
                'p95_absolute_kw': float(np.percentile(abs(err), 95)),
                'max_absolute_kw': float(np.max(abs(err))),
                'hourly_energy_rmse_kwh': float(np.sqrt(np.mean((err.reshape(-1, 6).sum(axis=1)/6)**2)))}
    return {'causal_fixed_weights_first_six_hours': {m: scores(errors[m]) for m in methods},
            'retrospective_exact_hourly_knots_not_executable_forecasts':
                {m: scores(oracle_errors[m]) for m in methods},
            'note': 'Oracle isolates within-hour reconstruction on observed hourly endpoints; '
                    'it is not a lower bound or an independent additive forecast-error component. '
                    'Alternative methods keep fixed January weights; they are sensitivity tests.'}


def run(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Choose a new audit directory')
    if not 1 <= args.days <= 334:
        raise ValueError('days must be 1..334')
    inputs = model.previous.load_inputs(model.ROOT)
    source = args.reference.resolve()
    saved_meta = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    for name, expected in saved_meta['hashes'].items():
        path = model.ROOT/('附件' if name.endswith('.xlsx') else 'src')/name
        if model.previous.digest(path) != expected:
            raise ValueError(f'Reference source or data changed: {name}')
    schedule = json.loads((source/'weight_schedule.json').read_text(encoding='utf-8'))
    january, logs, initial = shared_january(inputs)
    out.mkdir(parents=True)
    january.to_csv(out/'q2_shared_january_ledger.csv', index=False)
    model.previous.save_json(out/'q2_shared_january_lp.json', logs)
    print('Recomputed Q2 January final SOC:', initial, flush=True)
    weights = np.asarray(schedule['2025-02']['blocks'][0]['weights'])
    resolution = resolution_review(inputs, weights)
    model.previous.save_json(out/'resolution.json', resolution)
    actual_reference = pd.read_csv(source/'ledger.csv')
    frames, versions, daily = [], [], []
    soc, cache = initial, {}
    for d in range(31, 31+args.days):
        month = inputs[3][d].strftime('%Y-%m')
        record = schedule[month]
        ws = [np.asarray(b['weights']) for b in record['blocks']]
        frame, vv, soc = model.simulate_day(d, soc, inputs, saved_meta['selected_k'], ws, cache,
                                           weight_metadata=record)
        frames.append(frame);versions.extend(vv)
        ref = actual_reference[actual_reference.date == frame.date.iloc[0]]
        daily.append({'date': frame.date.iloc[0], 'shared_initial_soc': float(frame.soc_start_kwh.iloc[0]),
                      'shared_final_soc': soc, 'reference_initial_soc': float(ref.soc_start_kwh.iloc[0]),
                      'reference_final_soc': float(ref.soc_end_kwh.iloc[-1]),
                      'cost_difference': float(frame.total_cost.sum()-ref.total_cost.sum()),
                      'emergency_difference_kwh': float(frame.emergency_kwh.sum()-ref.emergency_kwh.sum()),
                      'max_soc_difference': float(np.max(np.abs(frame.soc_end_kwh.to_numpy()-ref.soc_end_kwh.to_numpy())))})
        print(daily[-1], flush=True)
    continuous = pd.concat([january]+frames, ignore_index=True)
    model.verify_ledger(continuous)
    feb = pd.concat(frames, ignore_index=True)
    feb.to_csv(out/'shared_initial_q3_ledger.csv', index=False)
    model.previous.save_json(out/'shared_initial_q3_versions.json', versions)
    pd.DataFrame(daily).to_csv(out/'initial_state_comparison.csv', index=False)
    q2_audit = json.loads((model.ROOT/'results/q2_teammate_review/audit.json').read_text(encoding='utf-8'))
    result = {'scope': 'Temporal-resolution diagnostics and common-Q2-January initial-state comparison',
              'days': args.days, 'annual_complete': args.days == 334,
              'january_initial_soc': 6000., 'january_idle_days': 7,
              'recomputed_february_soc': initial, 'saved_q2_february_soc': q2_audit['first_soc'],
              'q2_soc_reproduction_error_kwh': abs(initial-q2_audit['first_soc']),
              'reference_q3_february_soc': float(actual_reference.soc_start_kwh.iloc[0]),
              'comparison_cost_difference': float(sum(r['cost_difference'] for r in daily)),
              'comparison_emergency_difference_kwh': float(sum(r['emergency_difference_kwh'] for r in daily)),
              'comparison_final_soc_difference': daily[-1]['shared_final_soc']-daily[-1]['reference_final_soc'],
              'last_compared_date': daily[-1]['date'],
              'max_january_lp_residual': max(r['solver']['max_constraint_violation'] for r in logs),
              'max_january_lp_dual_gap': max(r['solver']['absolute_dual_gap'] for r in logs),
              'max_q3_mip_gap': max(v['solver']['mip_gap'] for v in versions),
              'hashes': {str(p.relative_to(model.ROOT)): model.previous.digest(p) for p in
                         [Path(__file__), model.ROOT/'src/solve_question2_three_day_teammate.py',
                          source/'ledger.csv', source/'weight_schedule.json']},
              'reference_metadata': saved_meta}
    model.previous.save_json(out/'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('hashes', 'reference_metadata')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=model.ROOT/'results/q3_time_warmup_review')
    parser.add_argument('--reference', type=Path, default=model.ROOT/'results/q3_monthly_fixed')
    parser.add_argument('--days', type=int, default=7)
    run(parser.parse_args())
