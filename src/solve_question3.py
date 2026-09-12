"""Q3 three-day stochastic LP and causal dispatch. See docs/17_question3_three_day_model.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import question3_forecast as forecast

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-5
LOW, HIGH, LIMIT, ETA = 1200., 10800., 5000 / 6, .9


def save_json(path, obj):
    # A completed day is committed only after its ledger and diagnostics exist.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(base):
    price, load, pv, dates = forecast.load_data(base)
    if (load.shape != (365, 144) or not dates.equals(pd.date_range('2025-01-01', '2025-12-31'))
            or not np.isfinite([load, pv]).all() or not np.isfinite(price).all() or np.any(price <= 0)):
        raise ValueError('Invalid dates, dimensions, numeric values or price')
    # Independently verify PV date alignment; teammate loader checks only shapes.
    pv_dates = pd.read_excel(base / '附件/附件2.xlsx', sheet_name='光伏发电实际功率', usecols=[0]).iloc[:, 0]
    if not pd.DatetimeIndex(pd.to_datetime(pv_dates)).equals(dates):
        raise ValueError('PV dates do not align with load dates')
    df = pd.read_excel(base / '附件/附件3.xlsx').replace({'': np.nan})
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    releases = {}
    for row in df.itertuples(index=False, name=None):
        hour = int(str(row[1]).split(':')[0])
        stamp = pd.Timestamp(row[0]).normalize() + pd.Timedelta(hours=hour)
        values = np.asarray(row[2:26], dtype=float)
        if hour not in (0, 6, 12, 18) or stamp in releases or len(values) != 24 or not np.isfinite(values).all():
            raise ValueError(f'Invalid forecast release: {stamp}')
        releases[stamp] = np.maximum(values, 0)
    expected = {day + pd.Timedelta(hours=h) for day in dates for h in (0, 6, 12, 18)}
    if set(releases) != expected:
        raise ValueError('Missing or extra forecast releases')
    return price, load, pv, dates, releases


def predict(d, kslot, load, pv, dates, releases, selected_k):
    """Return remaining path, using one issued forecast and only completed observations."""
    full = forecast.forecast_three_days(d, load, pv, dates, selected_k)
    issue = dates[d] + pd.Timedelta(minutes=kslot * 10)
    boundary = pv[d, kslot - 1] if kslot else (pv[d - 1, -1] if d else 0.)
    hourly = np.r_[boundary, releases[issue]]
    official = np.interp(np.arange(1, 145) / 6, np.arange(25), hourly)
    load3 = np.concatenate([forecast.predict_load_day(d, off, load, dates, selected_k) for off in range(3)])
    full[kslot:kslot + 144] = (load3[kslot:kslot + 144] - official) / 6
    return full[kslot:]


def scenarios(d, kslot, load, pv, dates, releases, selected_k, cache=None, single=False):
    cache = {} if cache is None else cache
    base = predict(d, kslot, load, pv, dates, releases, selected_k)
    origins = [] if single else list(range(max(1, d - 45), d - 2))[-3:]
    residuals = []
    for h in origins:
        key = (h, kslot, selected_k)
        if key not in cache:
            old = predict(h, kslot, load, pv, dates, releases, selected_k)
            actual = ((load[h:h + 3] - pv[h:h + 3]) / 6).ravel()[kslot:]
            cache[key] = actual - old
        residuals.append(cache[key])
    paths = base[None, :] + np.asarray(residuals) if residuals else base[None, :]
    return paths, base, origins


def optimize(paths, prices, soc, floor, lam, original=None, time_limit=60):
    """Continuous relaxation; reject non-optimal status or infeasible raw vectors."""
    paths, prices = np.asarray(paths), np.asarray(prices)
    m, horizon = paths.shape
    if prices.shape != (horizon,) or not np.isfinite(paths).all() or not LOW <= soc <= HIGH:
        raise ValueError('Invalid optimization input')
    n = horizon + 5 * m * horizon
    def ix(s, block, t):
        return horizon + (s * 5 + block) * horizon + t
    costs = np.zeros(n)
    costs[:horizon] = prices
    bounds = [(0, None)] * n
    if original is not None:
        for t, value in enumerate(original):
            bounds[t] = (float(value), None)
            costs[t] = 1.5 * prices[t]
    rows, cols, vals, rhs = [], [], [], []
    def equation(terms, value):
        row = len(rhs)
        for col, coefficient in terms:
            rows.append(row); cols.append(col); vals.append(coefficient)
        rhs.append(value)
    for s in range(m):
        for t in range(horizon):
            c, b, e, w, state = [ix(s, block, t) for block in range(5)]
            costs[c] = costs[b] = 1e-5 / m
            costs[e] = 5 * prices[t] / m
            bounds[c] = bounds[b] = (0, LIMIT)
            bounds[e] = (0, max(paths[s, t], 0))
            bounds[state] = (max(LOW, floor) if t == horizon - 1 else LOW, HIGH)
            equation([(t, 1), (b, 1), (e, 1), (c, -1), (w, -1)], paths[s, t])
            terms = [(state, 1), (c, -ETA), (b, 1 / ETA)]
            if t:
                terms.append((ix(s, 4, t - 1), -1))
            equation(terms, soc if t == 0 else 0)
        costs[ix(s, 4, horizon - 1)] = -lam / m
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n)).tocsr()
    began = time.perf_counter()
    result = linprog(costs, A_eq=matrix, b_eq=rhs, bounds=bounds, method='highs',
                     options={'time_limit': time_limit, 'primal_feasibility_tolerance': 1e-8,
                              'dual_feasibility_tolerance': 1e-8})
    if result.status != 0 or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f'LP rejected: {result.status}: {result.message}')
    x = result.x
    residual = float(np.max(np.abs(matrix @ x - rhs)))
    violations = [max(lo - v, 0, v - hi if hi is not None else 0) for v, (lo, hi) in zip(x, bounds)]
    if max(residual, max(violations)) > TOL:
        raise RuntimeError('Raw LP constraints failed')
    recourse = x[horizon:].reshape(m, 5, horizon)
    log = {'status': int(result.status), 'objective': float(result.fun),
           'seconds': time.perf_counter() - began, 'equality_residual': residual,
           'bound_violation': float(max(violations)), 'scenarios': m, 'horizon': horizon,
           'simultaneous_charge_discharge': int(np.sum((recourse[:, 0] > TOL) & (recourse[:, 1] > TOL))),
           'emergency_charge': int(np.sum((recourse[:, 0] > TOL) & (recourse[:, 2] > TOL)))}
    return np.maximum(x[:horizon], 0), log


def dispatch(net, regular, soc):
    gap = net - regular
    c = b = e = w = 0.
    if gap > 0:
        b = min(gap, LIMIT, max(0., ETA * (soc - LOW)))
        e = gap - b
    else:
        c = min(-gap, LIMIT, max(0., (HIGH - soc) / ETA))
        w = -gap - c
    end = soc + ETA * c - b / ETA
    if not LOW - TOL <= end <= HIGH + TOL or abs(regular + b + e - net - c - w) > TOL:
        raise RuntimeError('Dispatch physical check failed')
    return c, b, e, w, end


def simulate_day(d, soc, inputs, selected_k, hours=(0, 6, 12, 18), cache=None, single=False, time_limit=60):
    price, load, pv, dates, releases = inputs
    floor = forecast.compute_terminal_soc_floor(d, load, pv, dates)
    lam = forecast.estimate_lambda(price)
    versions, ledger = [], []
    plan = current = None
    for t in range(144):
        if t in [h * 6 for h in hours]:
            paths, base, origins = scenarios(d, t, load, pv, dates, releases, selected_k, cache, single)
            solution, log = optimize(paths, np.tile(price, 3)[t:], soc, floor, lam,
                                     None if t == 0 else plan[t:], time_limit)
            if t == 0:
                plan = solution[:144].copy()
                current = plan.copy()
            else:
                current[t:] = solution[:144 - t]
            versions.append({'issue_time': str(dates[d] + pd.Timedelta(minutes=10 * t)),
                             'first_slot': t + 1, 'soc': float(soc), 'selected_k': selected_k,
                             'residual_dates': [str(dates[h].date()) for h in origins],
                             'floor': floor, 'lambda': lam, 'forecast_kwh': base.tolist(),
                             'scenarios_kwh': paths.tolist(), 'regular_draft_kwh': solution.tolist(),
                             'solver': log})
        net = (load[d, t] - pv[d, t]) / 6
        c, b, e, w, end = dispatch(net, current[t], soc)
        pc = price[t] * plan[t]
        inc = price[t] * 1.5 * max(current[t] - plan[t], 0)
        reduction = price[t] * .5 * max(plan[t] - current[t], 0)
        ec = price[t] * 5 * e
        ledger.append({'date': str(dates[d].date()), 'slot': t + 1,
                       'interval_start': str(dates[d] + pd.Timedelta(minutes=10 * t)),
                       'interval_end': str(dates[d] + pd.Timedelta(minutes=10 * (t + 1))),
                       'net_kwh': net, 'plan_kwh': plan[t], 'final_regular_kwh': current[t],
                       'emergency_kwh': e, 'charge_kwh': c, 'discharge_kwh': b,
                       'surplus_kwh': w, 'soc_start_kwh': soc, 'soc_end_kwh': end,
                       'price': price[t], 'plan_cost': pc, 'increase_cost': inc,
                       'reduction_cost': reduction, 'emergency_cost': ec, 'total_cost': pc + inc + reduction + ec})
        soc = end
    frame = pd.DataFrame(ledger)
    verify_ledger(frame)
    return frame, versions, soc


def verify_ledger(df):
    if df.empty or len(df) % 144 or not np.isfinite(df.select_dtypes('number')).all().all():
        raise ValueError('Incomplete or non-finite ledger')
    for _, day in df.groupby('date', sort=False):
        if day.slot.tolist() != list(range(1, 145)):
            raise ValueError('Invalid slot order')
    c, b, e, w = [df[name].to_numpy() for name in ('charge_kwh', 'discharge_kwh', 'emergency_kwh', 'surplus_kwh')]
    q, g, p = [df[name].to_numpy() for name in ('plan_kwh', 'final_regular_kwh', 'price')]
    s, end = df.soc_start_kwh.to_numpy(), df.soc_end_kwh.to_numpy()
    expected = np.c_[p*q, 1.5*p*np.maximum(g-q, 0), .5*p*np.maximum(q-g, 0), 5*p*e]
    actual = df[['plan_cost', 'increase_cost', 'reduction_cost', 'emergency_cost']].to_numpy()
    errors = [np.max(np.abs(g+b+e-df.net_kwh-c-w)), np.max(np.abs(end-s-ETA*c+b/ETA)),
              np.max(np.abs(s[1:]-end[:-1])) if len(s)>1 else 0,
              np.max(np.abs(expected-actual)), np.max(np.abs(expected.sum(axis=1)-df.total_cost))]
    if (max(errors)>TOL or min(s.min(),end.min())<LOW-TOL or max(s.max(),end.max())>HIGH+TOL
            or min(c.min(),b.min(),e.min(),w.min(),q.min(),g.min()) < -TOL
            or max(c.max(),b.max())>LIMIT+TOL or np.any((c>TOL)&((b>TOL)|(e>TOL))) or np.any(g<q-TOL)):
        raise ValueError('Ledger physical/settlement validation failed')
    return float(max(errors))


def run(args):
    inputs = load_inputs(args.base_dir)
    price, load, pv, dates, _ = inputs
    hours = tuple(int(h) for h in args.release_hours.split(','))
    if not hours or hours[0] != 0 or tuple(sorted(set(hours))) != hours or not set(hours) <= {0,6,12,18}:
        raise ValueError('release-hours must be an ordered subset containing 0')
    stop = pd.Timestamp(args.end_date)
    if stop not in dates or stop < pd.Timestamp('2025-02-01'):
        raise ValueError('end-date must be in February--December 2025')
    selected_k, calibration = forecast.calibrate_k_on_january(dates, load, pv)
    fingerprint_files = [Path(__file__), Path(forecast.__file__), args.base_dir/'附件/附件1.xlsx',
                         args.base_dir/'附件/附件2.xlsx', args.base_dir/'附件/附件3.xlsx']
    metadata = {'model': 'Q3-three-day-LP-v1', 'hours': list(hours), 'selected_k': selected_k,
                'settlement': 'no-refund-final-vs-original', 'time_limit': args.time_limit,
                'hashes': {p.name: digest(p) for p in fingerprint_files}}
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('Use a new output directory, or --resume with identical inputs/code')
    out.mkdir(parents=True, exist_ok=True)
    mp = out/'metadata.json'
    if mp.exists():
        if json.loads(mp.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Resume metadata differs')
    else:
        if args.resume:
            raise ValueError('Missing metadata for resume')
        save_json(mp, metadata)
    calibration.to_csv(out/'k_calibration.csv', index=False)
    soc, cache, frames = 6000., {}, []
    for d in range(7, int(dates.get_loc(stop)) + 1):
        day = str(dates[d].date())
        complete, csv, versions = out/f'{day}_complete.json', out/f'{day}_ledger.csv', out/f'{day}_versions.json'
        if complete.exists():
            saved = json.loads(complete.read_text(encoding='utf-8'))
            if digest(csv) != saved['ledger_sha256'] or digest(versions) != saved['versions_sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {day}')
            frame = pd.read_csv(csv)
            verify_ledger(frame)
            if abs(frame.soc_start_kwh.iloc[0] - soc)>TOL:
                raise ValueError('Checkpoint SOC chain mismatch')
            soc = float(frame.soc_end_kwh.iloc[-1])
        else:
            try:
                frame, records, soc = simulate_day(d, soc, inputs, 7 if d < 31 else selected_k,
                    (0,6,12,18) if d < 31 else hours, cache, d < 31, args.time_limit)
            except Exception as exc:
                save_json(out/'last_failure.json', {'date': day, 'error': str(exc)})
                raise
            frame.to_csv(csv, index=False)
            save_json(versions, records)
            save_json(complete, {'date':day, 'ledger_sha256':digest(csv), 'versions_sha256':digest(versions)})
            print(f'{day}: cost={frame.total_cost.sum():.2f}, soc={soc:.4f}', flush=True)
        if d >= 31:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    error = verify_ledger(result)
    result.to_csv(out/'ledger.csv', index=False)
    summary = {name: float(result[name].sum()) for name in ('plan_cost','increase_cost','reduction_cost',
               'emergency_cost','total_cost','emergency_kwh','surplus_kwh')}
    summary.update(days=len(frames), start_date='2025-02-01', end_date=str(stop.date()),
                   annual_complete=len(frames)==334, initial_soc=float(result.soc_start_kwh.iloc[0]),
                   final_soc=soc, max_validation_error=error)
    save_json(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'results/q3_three_day')
    parser.add_argument('--end-date', default='2025-12-31')
    parser.add_argument('--release-hours', default='0,6,12,18')
    parser.add_argument('--time-limit', type=float, default=60)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
