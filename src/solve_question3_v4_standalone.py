"""第三问 v4 独立运行版：保持原版预测、MILP、调度和退款计算顺序。

只需此文件与原始附件1.xlsx、附件2.xlsx、附件3.xlsx；无需项目内其他代码。
附件可放在脚本同目录或同目录的“附件”子目录，也可用 --base-dir 指定。
运行：python solve_question3_v4_standalone.py
对照：python solve_question3_v4_standalone.py --reference 原版结果目录
续算：python solve_question3_v4_standalone.py --output 已有输出目录 --resume

原成果数值环境：Python 3.12.4 / numpy 2.1.3 / pandas 3.0.2 / scipy 1.17.1。
另需 openpyxl 读取原始 Excel。不同求解器版本不保证逐位一致。
输出逐日/全年 CSV、预测版本、汇总、断点及独立校验 JSON；不内置现成结果。
默认完整联合版；两个消融配置通过 --load-correction / --pv-interpolation 选择。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import PchipInterpolator
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import coo_matrix

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent if (SCRIPT.parent.parent / '附件').is_dir() else SCRIPT.parent
MODEL = 'Q3-six-hour-24h-MILP-refund-v4'
N_SLOTS = 144
DELTA = 1 / 6
MAX_TRAIN_DAYS = 31
RECENCY_DECAY = 0.92
K_CANDIDATES = [3, 5, 7, 10, 14, 21, 28]
DEFAULT_K = 7
TRAIN_START = pd.Timestamp('2025-01-01')
TRAIN_END = pd.Timestamp('2025-01-31')
ETA_D = 0.9
LOW, HIGH, LIMIT, ETA, TOL = 1200.0, 10800.0, 5000 / 6, 0.9, 1e-5
METHODS = ('linear', 'pchip', 'step')


def attachment_dir(base):
    """Resolve only input paths; never modify the source workbooks."""
    base = Path(base).resolve()
    for candidate in (base / '附件', base):
        if all((candidate / f'附件{i}.xlsx').is_file() for i in (1, 2, 3)):
            return candidate
    raise FileNotFoundError(f'附件1/2/3.xlsx must exist together in {base} or its 附件 directory')




# question3_forecast: extracted implementation; numerical operations preserved.


def load_data(base_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    读取附件1（电价）和附件2（负载、光伏）。

    返回：
        price:    (144,) 电价数组，元/kWh
        load_arr: (n_days, 144) 负载功率，kW
        pv_arr:   (n_days, 144) 光伏功率，kW（已 clip ≥ 0）
        dates:    (n_days,) 日期索引
    """
    f1 = attachment_dir(base_dir) / '附件1.xlsx'
    f2 = attachment_dir(base_dir) / '附件2.xlsx'
    if not f1.exists():
        raise FileNotFoundError(f'未找到附件1：{f1}')
    if not f2.exists():
        raise FileNotFoundError(f'未找到附件2：{f2}')
    df1 = pd.read_excel(f1, sheet_name='Sheet1')
    if '电价' not in df1.columns:
        raise ValueError('附件1 Sheet1 中未找到列。')
    price = df1['电价'].values[:N_SLOTS].astype(float)
    assert len(price) == N_SLOTS, f'电价数据应有 {N_SLOTS} 个时段'
    load_df = pd.read_excel(f2, sheet_name='小区负载', index_col=0, parse_dates=True)
    pv_df = pd.read_excel(f2, sheet_name='光伏发电实际功率', index_col=0, parse_dates=True)
    load_arr = load_df.iloc[:, :N_SLOTS].values.astype(float)
    pv_arr = pv_df.iloc[:, :N_SLOTS].values.astype(float)
    dates = pd.DatetimeIndex(pd.to_datetime(load_df.index))
    if load_arr.shape != pv_arr.shape:
        raise ValueError('负载数据与光伏数据形状不一致。')
    if load_arr.shape[1] != N_SLOTS:
        raise ValueError(f'每天必须有 {N_SLOTS} 个时段。')
    if len(dates) < 365:
        print(f'[WARN] 当前仅有 {len(dates)} 天数据。')
    pv_arr = np.maximum(pv_arr, 0.0)
    return (price, load_arr, pv_arr, dates)

def net_energy_day(day_idx: int, load_arr: np.ndarray, pv_arr: np.ndarray) -> np.ndarray:
    """
    计算某天144个时段的净负荷电量（kWh/10min）。
    净负荷 = (负载功率 - 光伏功率) × Δ
    可以为负（光伏 > 负载时）。
    """
    return (load_arr[day_idx] - pv_arr[day_idx]) * DELTA

def predict_load_day(origin_idx: int, target_day_offset: int, load_arr: np.ndarray, dates: pd.DatetimeIndex, k: int, max_train_days: int=MAX_TRAIN_DAYS) -> np.ndarray:
    """
    预测负载功率（kW），基于同星期加权历史均值 + 趋势修正。

    严格只使用 origin_idx 之前的数据。
    """
    if origin_idx <= 0:
        return np.zeros(N_SLOTS, dtype=float)
    target_idx = origin_idx + target_day_offset
    if target_idx < len(dates):
        target_weekday = dates[target_idx].weekday()
    else:
        last_weekday = dates[len(dates) - 1].weekday()
        target_weekday = (last_weekday + (target_idx - len(dates) + 1)) % 7
    hist_start = max(0, origin_idx - max_train_days)
    hist_indices = list(range(hist_start, origin_idx))
    if not hist_indices:
        return np.zeros(N_SLOTS, dtype=float)
    same_weekday = [h for h in hist_indices if dates[h].weekday() == target_weekday]
    if len(same_weekday) >= 2:
        selected = same_weekday[-min(k, len(same_weekday)):]
    else:
        selected = hist_indices[-min(k, len(hist_indices)):]
    ages = np.array([origin_idx - h for h in selected], dtype=float)
    weights = RECENCY_DECAY ** ages
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    pred = np.zeros(N_SLOTS, dtype=float)
    for w, h in zip(weights, selected):
        pred += w * load_arr[h]
    if len(same_weekday) >= 2:
        diffs = []
        for i in range(1, len(same_weekday)):
            h1, h0 = (same_weekday[i], same_weekday[i - 1])
            diffs.append(load_arr[h1] - load_arr[h0])
        if diffs:
            trend = np.median(np.vstack(diffs), axis=0)
            gamma = 0.3
            pred = pred + gamma * trend
    return pred

def predict_pv_day(origin_idx: int, target_day_offset: int, pv_arr: np.ndarray, dates: pd.DatetimeIndex, w_p: int=14) -> np.ndarray:
    """
    预测光伏功率（kW），基于近期加权均值。

    严格只使用 origin_idx 之前的数据。
    自然适应季节变化（日出日落时间变化）。
    """
    if origin_idx <= 0:
        return np.zeros(N_SLOTS, dtype=float)
    hist_start = max(0, origin_idx - w_p)
    selected = list(range(hist_start, origin_idx))
    if not selected:
        return np.zeros(N_SLOTS, dtype=float)
    ages = np.array([origin_idx - h for h in selected], dtype=float)
    weights = RECENCY_DECAY ** ages
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    pred = np.zeros(N_SLOTS, dtype=float)
    for w, h in zip(weights, selected):
        pred += w * pv_arr[h]
    pred = np.maximum(pred, 0.0)
    return pred

def forecast_net_energy(origin_idx: int, target_day_offset: int, load_arr: np.ndarray, pv_arr: np.ndarray, dates: pd.DatetimeIndex, k: int) -> np.ndarray:
    """
    预测某天的净负荷电量（kWh/10min）。
    分别预测负载和光伏，再合成。
    """
    pred_load = predict_load_day(origin_idx, target_day_offset, load_arr, dates, k)
    pred_pv = predict_pv_day(origin_idx, target_day_offset, pv_arr, dates)
    return (pred_load - pred_pv) * DELTA

def calibrate_k_on_january(dates: pd.DatetimeIndex, load_arr: np.ndarray, pv_arr: np.ndarray) -> Tuple[int, pd.DataFrame]:
    """用1月内部滚动回测选择 K。"""
    jan_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    jan_indices = np.where(jan_mask)[0]
    if len(jan_indices) < 10:
        print('[WARN] 1月样本不足，使用默认K。')
        return (DEFAULT_K, pd.DataFrame())
    first_idx = int(jan_indices[0])
    last_idx = int(jan_indices[-1])
    rows = []
    for k in K_CANDIDATES:
        sq_errors = []
        abs_errors = []
        val_start = first_idx + 7
        for d in range(val_start, last_idx + 1):
            pred = forecast_net_energy(origin_idx=d, target_day_offset=0, load_arr=load_arr, pv_arr=pv_arr, dates=dates, k=k)
            actual = net_energy_day(d, load_arr, pv_arr)
            err = actual - pred
            sq_errors.extend(np.square(err).tolist())
            abs_errors.extend(np.abs(err).tolist())
        if not sq_errors:
            continue
        rows.append({'K': k, 'RMSE': float(np.sqrt(np.mean(sq_errors))), 'MAE': float(np.mean(abs_errors))})
    if not rows:
        return (DEFAULT_K, pd.DataFrame())
    df = pd.DataFrame(rows).sort_values(['RMSE', 'MAE'])
    best_k = int(df.iloc[0]['K'])
    return (best_k, df.reset_index(drop=True))

def estimate_lambda(price: np.ndarray) -> float:
    """
    估算单位储能电量的未来价值 λ。
    λ ≈ 高价时段平均电价 × 放电效率
    """
    price_threshold = float(np.percentile(price, 75))
    high_price_mean = float(np.mean(price[price >= price_threshold]))
    lam = high_price_mean * ETA_D
    lam = float(np.clip(lam, 0.01, 5.0))
    return lam



# solve_question3: extracted implementation; numerical operations preserved.


def save_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_inputs_core(base):
    price, load, pv, dates = load_data(base)
    if load.shape != (365, 144) or not dates.equals(pd.date_range('2025-01-01', '2025-12-31')) or (not np.isfinite([load, pv]).all()) or (not np.isfinite(price).all()) or np.any(price <= 0):
        raise ValueError('Invalid dates, dimensions, numeric values or price')
    pv_dates = pd.read_excel(attachment_dir(base) / '附件2.xlsx', sheet_name='光伏发电实际功率', usecols=[0]).iloc[:, 0]
    if not pd.DatetimeIndex(pd.to_datetime(pv_dates)).equals(dates):
        raise ValueError('PV dates do not align with load dates')
    df = pd.read_excel(attachment_dir(base) / '附件3.xlsx').replace({'': np.nan})
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    releases = {}
    for row in df.itertuples(index=False, name=None):
        hour = int(str(row[1]).split(':')[0])
        stamp = pd.Timestamp(row[0]).normalize() + pd.Timedelta(hours=hour)
        values = np.asarray(row[2:26], dtype=float)
        if hour not in (0, 6, 12, 18) or stamp in releases or len(values) != 24 or (not np.isfinite(values).all()):
            raise ValueError(f'Invalid forecast release: {stamp}')
        releases[stamp] = np.maximum(values, 0)
    expected = {day + pd.Timedelta(hours=h) for day in dates for h in (0, 6, 12, 18)}
    if set(releases) != expected:
        raise ValueError('Missing or extra forecast releases')
    return (price, load, pv, dates, releases)

def dispatch(net, regular, soc):
    gap = net - regular
    c = b = e = w = 0.0
    if gap > 0:
        b = min(gap, LIMIT, max(0.0, ETA * (soc - LOW)))
        e = gap - b
    else:
        c = min(-gap, LIMIT, max(0.0, (HIGH - soc) / ETA))
        w = -gap - c
    end = soc + ETA * c - b / ETA
    if not LOW - TOL <= end <= HIGH + TOL or abs(regular + b + e - net - c - w) > TOL:
        raise RuntimeError('Dispatch physical check failed')
    return (c, b, e, w, end)



# analyze_question3_revision: extracted implementation; numerical operations preserved.


def vintage_matrix(issue, pv, dates, releases, horizon=144):
    """Rows=target ten-minute ends; columns=newest to oldest release, NaN if expired."""
    target = issue + pd.to_timedelta(np.arange(1, horizon + 1) * 10, unit='min')
    values = np.full((horizon, 4), np.nan)
    for age in range(4):
        vintage = issue - pd.Timedelta(hours=6 * age)
        if vintage not in releases:
            continue
        d = (vintage.normalize() - dates[0]).days
        slot = vintage.hour * 6
        boundary = pv[d, slot - 1] if slot else pv[d - 1, -1] if d else 0.0
        leads = np.asarray((target - vintage).total_seconds() / 3600)
        valid = (leads > 0) & (leads <= 24)
        values[valid, age] = np.interp(leads[valid], np.arange(25), np.r_[boundary, releases[vintage]])
    return values

def fit_monotone(x, y):
    """Nonnegative simplex weights, constrained newest >= older."""
    n = x.shape[1]
    if n == 1:
        return np.ones(1)
    scale = max(float(np.mean(y * y)), 1.0)
    gram = x.T @ x / len(x) / scale
    xy = x.T @ y / len(x) / scale
    result = minimize(lambda w: float(w @ gram @ w - 2 * xy @ w), np.ones(n) / n, jac=lambda w: 2 * (gram @ w - xy), method='SLSQP', bounds=[(0, 1)] * n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(n)}, {'type': 'ineq', 'fun': lambda w: w[:-1] - w[1:]}], options={'ftol': 1e-12, 'maxiter': 500})
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()



# analyze_question3_monthly_weights: extracted implementation; numerical operations preserved.


def select(record, start, cutoff):
    return (record['issue'] >= pd.Timestamp(start)) & (record['issue'] < pd.Timestamp(cutoff)) & (record['available_at'] <= pd.Timestamp(cutoff))

def fit_record(record, mask):
    x = record['x'][mask]
    if not len(x):
        raise ValueError('No mature training samples')
    return fit_monotone(x.reshape(-1, x.shape[-1]), record['y'][mask].ravel())

def fit_asof(blocks, start, cutoff):
    result = []
    for record in blocks:
        mask = select(record, start, cutoff)
        weights = fit_record(record, mask)
        result.append({'weights': weights.tolist(), 'training_blocks': int(mask.sum()), 'training_points': int(mask.sum()) * 36, 'latest_available_at': str(record['available_at'][mask].max())})
    return result



# solve_question3_six_hour: extracted implementation; numerical operations preserved.


def startup_weights(report=None):
    if report is None:
        return [0.92 ** np.arange(n) / (0.92 ** np.arange(n)).sum() for n in (4, 3, 2, 1)]
    return [np.asarray(block['fit_weights']) for block in report['blocks']]

def predict_warmup(d, slot, inputs, selected_k, weights, newest_only=False):
    _, load, pv, dates, releases = inputs
    issue = dates[d] + pd.Timedelta(minutes=10 * slot)
    matrix = vintage_matrix(issue, pv, dates, releases)
    cold_start = issue < dates[0] + pd.Timedelta(hours=18)
    solar = np.empty(144)
    for t in range(144):
        available = np.isfinite(matrix[t])
        w = np.zeros(4)
        if newest_only or cold_start:
            w[0] = 1.0
        else:
            w[:len(weights[t // 36])] = weights[t // 36]
        w[~available] = 0.0
        if w.sum() <= 0:
            raise ValueError('No valid released forecast for target')
        solar[t] = float(np.nan_to_num(matrix[t]) @ (w / w.sum()))
    demand = np.concatenate([predict_load_day(d, off, load, dates, selected_k) for off in (0, 1)])[slot:slot + 144]
    return (demand - solar) / 6

def make_warmup_scenarios(d, slot, inputs, selected_k, weights, cache=None, single=False, newest_only=False):
    base = predict_warmup(d, slot, inputs, selected_k, weights, newest_only)
    cache = {} if cache is None else cache
    origins = [] if single else list(range(max(3, d - 45), d))[-3:]
    residuals = []
    signature = tuple((tuple((float(x) for x in w)) for w in weights))
    net = ((inputs[1] - inputs[2]) / 6).ravel()
    for h in origins:
        key = (h, slot, selected_k, signature, newest_only)
        if key not in cache:
            start = h * 144 + slot
            historical = predict_warmup(h, slot, inputs, selected_k, weights, newest_only)
            cache[key] = net[start:start + 144] - historical
        residuals.append(cache[key])
    paths = base[None, :] + np.asarray(residuals) if residuals else base[None, :]
    return (paths, base, origins)

def optimize(paths, prices, soc, lam, original=None, time_limit=60, gap=0.0001):
    """Shared regular purchases; strict battery physics for each scenario path."""
    paths = np.asarray(paths, dtype=float)
    prices = np.asarray(prices, dtype=float)
    m, horizon = paths.shape
    if prices.shape != (horizon,) or not np.isfinite(paths).all() or (not np.isfinite(prices).all()) or np.any(prices <= 0) or (not LOW <= soc <= HIGH):
        raise ValueError('Invalid MILP inputs')
    original = np.asarray([] if original is None else original, dtype=float)
    if len(original) > horizon or not np.isfinite(original).all() or np.any(original < 0):
        raise ValueError('Invalid original plan')
    count = 2 * horizon + 6 * m * horizon
    cost = np.zeros(count)
    lo = np.zeros(count)
    hi = np.full(count, np.inf)
    integer = np.zeros(count, dtype=np.uint8)
    cost[:horizon] = prices
    hi[horizon:2 * horizon] = 0.0

    def index(s, block, t):
        return 2 * horizon + (6 * s + block) * horizon + t
    rows = []
    cols = []
    vals = []
    lower = []
    upper = []

    def add(terms, low=-np.inf, high=np.inf):
        r = len(lower)
        for c, v in terms:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lower.append(low)
        upper.append(high)
    for t, q in enumerate(original):
        z = horizon + t
        hi[z] = np.inf
        cost[z] = 1.0
        cost[t] = 0.0
        add([(z, 1), (t, -0.5 * prices[t])], low=0.5 * prices[t] * q)
        add([(z, 1), (t, -1.5 * prices[t])], low=-0.5 * prices[t] * q)
    for s in range(m):
        for t in range(horizon):
            c, b, e, w, state, y = [index(s, j, t) for j in range(6)]
            cost[c] = cost[b] = 1e-05 / m
            cost[e] = 5 * prices[t] / m
            hi[c] = hi[b] = LIMIT
            lo[state] = LOW
            hi[state] = HIGH
            hi[y] = 1
            integer[y] = 1
            nplus = max(paths[s, t], 0.0)
            hi[e] = nplus
            add([(t, 1), (b, 1), (e, 1), (c, -1), (w, -1)], paths[s, t], paths[s, t])
            terms = [(state, 1), (c, -ETA), (b, 1 / ETA)]
            if t:
                terms.append((index(s, 4, t - 1), -1))
            add(terms, soc if t == 0 else 0, soc if t == 0 else 0)
            add([(c, 1), (y, -LIMIT)], high=0.0)
            add([(b, 1), (y, LIMIT)], high=LIMIT)
            add([(e, 1), (y, nplus)], high=nplus)
        cost[index(s, 4, horizon - 1)] = -lam / m
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), count)).tocsc()
    began = time.perf_counter()
    result = milp(cost, integrality=integer, bounds=Bounds(lo, hi), constraints=LinearConstraint(matrix, lower, upper), options={'time_limit': time_limit, 'mip_rel_gap': gap})
    if result.x is None or not np.isfinite(result.x).all() or result.status != 0:
        raise RuntimeError(f'MILP rejected: {result.status}: {result.message}')
    relative = float(result.mip_gap)
    absolute = abs(float(result.fun) - float(result.mip_dual_bound))
    if not (np.isfinite(relative) and relative <= gap + 1e-10 or absolute <= 1e-06):
        raise RuntimeError('MILP optimality gap rejected')
    x = result.x
    ax = matrix @ x
    violation = max(float(np.max(np.maximum(np.asarray(lower) - ax, 0))), float(np.max(np.maximum(ax - np.asarray(upper), 0))), float(np.max(np.maximum(lo - x, 0))), float(np.max(np.maximum(x - hi, 0))), float(np.max(np.abs(x[integer == 1] - np.rint(x[integer == 1])))))
    if violation > TOL:
        raise RuntimeError(f'Raw MILP vector failed constraints: {violation}')
    recourse = x[2 * horizon:].reshape(m, 6, horizon)
    simultaneous = int(np.sum((recourse[:, 0] > TOL) & (recourse[:, 1] > TOL)))
    emergency_charge = int(np.sum((recourse[:, 0] > TOL) & (recourse[:, 2] > TOL)))
    if simultaneous or emergency_charge:
        raise RuntimeError('MILP scenario physical mode violation')
    log = {'status': int(result.status), 'objective': float(result.fun), 'mip_gap': relative if np.isfinite(relative) else None, 'absolute_gap': absolute, 'seconds': time.perf_counter() - began, 'max_constraint_violation': violation, 'horizon': horizon, 'scenarios': m, 'simultaneous_charge_discharge': simultaneous, 'emergency_charge': emergency_charge}
    return (np.maximum(x[:horizon], 0), log)

def simulate_warmup_day(d, soc, inputs, selected_k, weights, cache=None, single=False, newest_only=False, adjustments=True, time_limit=60, weight_metadata=None):
    price, load, pv, dates, _ = inputs
    lam = estimate_lambda(price)
    versions = []
    rows = []
    plan = None
    current = None
    for slot in range(144):
        if slot % 36 == 0:
            paths, base, origins = make_warmup_scenarios(d, slot, inputs, selected_k, weights, cache, single, newest_only)
            if slot == 0 or adjustments:
                prices = np.tile(price, 2)[slot:slot + 144]
                solution, log = optimize(paths, prices, soc, lam, None if slot == 0 else plan[slot:], time_limit)
                if slot == 0:
                    plan = solution.copy()
                    current = plan.copy()
                else:
                    current[slot:] = solution[:144 - slot]
            else:
                solution = current[slot:].copy()
                log = {'skipped': 'no-adjustment baseline'}
            versions.append({'issue_time': str(dates[d] + pd.Timedelta(minutes=10 * slot)), 'first_slot': slot + 1, 'executed_slots': 36, 'lookahead_slots': 144, 'soc': float(soc), 'selected_k': selected_k, 'weights': [w.tolist() for w in weights], 'weight_training': weight_metadata, 'residual_dates': [str(dates[h].date()) for h in origins], 'forecast_kwh': base.tolist(), 'scenarios_kwh': paths.tolist(), 'regular_draft_kwh': solution.tolist(), 'lambda': lam, 'solver': log})
        net = (load[d, slot] - pv[d, slot]) / 6
        c, b, e, w, end = dispatch(net, current[slot], soc)
        pc = price[slot] * plan[slot]
        increase = 1.5 * price[slot] * max(current[slot] - plan[slot], 0)
        refund = 0.5 * price[slot] * max(plan[slot] - current[slot], 0)
        emergency = 5 * price[slot] * e
        rows.append({'date': str(dates[d].date()), 'slot': slot + 1, 'interval_start': str(dates[d] + pd.Timedelta(minutes=slot * 10)), 'interval_end': str(dates[d] + pd.Timedelta(minutes=(slot + 1) * 10)), 'net_kwh': net, 'plan_kwh': plan[slot], 'final_regular_kwh': current[slot], 'emergency_kwh': e, 'charge_kwh': c, 'discharge_kwh': b, 'surplus_kwh': w, 'soc_start_kwh': soc, 'soc_end_kwh': end, 'price': price[slot], 'plan_cost': pc, 'increase_cost': increase, 'refund_amount': refund, 'reduction_cost': -refund, 'emergency_cost': emergency, 'total_cost': pc + increase - refund + emergency})
        soc = end
    frame = pd.DataFrame(rows)
    verify_ledger(frame)
    return (frame, versions, soc)

def verify_ledger(df):
    if df.empty or len(df) % 144 or (not np.isfinite(df.select_dtypes('number')).all().all()):
        raise ValueError('Incomplete or nonfinite ledger')
    for _, day in df.groupby('date', sort=False):
        if day.slot.tolist() != list(range(1, 145)):
            raise ValueError('Bad slot order')
    c, b, e, w = [df[n].to_numpy() for n in ('charge_kwh', 'discharge_kwh', 'emergency_kwh', 'surplus_kwh')]
    q, g, p = [df[n].to_numpy() for n in ('plan_kwh', 'final_regular_kwh', 'price')]
    start, end = (df.soc_start_kwh.to_numpy(), df.soc_end_kwh.to_numpy())
    expected = np.c_[p * q, 1.5 * p * np.maximum(g - q, 0), -0.5 * p * np.maximum(q - g, 0), 5 * p * e]
    actual = df[['plan_cost', 'increase_cost', 'reduction_cost', 'emergency_cost']].to_numpy()
    error = max(float(np.max(np.abs(g + b + e - df.net_kwh - c - w))), float(np.max(np.abs(end - start - ETA * c + b / ETA))), float(np.max(np.abs(start[1:] - end[:-1]))), float(np.max(np.abs(expected - actual))), float(np.max(np.abs(expected.sum(axis=1) - df.total_cost))), float(np.max(np.abs(df.refund_amount + df.reduction_cost))))
    if error > TOL or min(start.min(), end.min()) < LOW - TOL or max(start.max(), end.max()) > HIGH + TOL or (min(c.min(), b.min(), e.min(), w.min(), q.min(), g.min()) < -TOL) or (max(c.max(), b.max()) > LIMIT + TOL) or np.any((c > TOL) & ((b > TOL) | (e > TOL))):
        raise ValueError('Physical or refund-accounting violation')
    return error



# question3_forecast_v4: extracted implementation; numerical operations preserved.


def load_forecast(d, slot, inputs, k, mode):
    if mode not in ('none', 'prefix-ratio'):
        raise ValueError('Unknown load correction')
    _, load, _, dates, _ = inputs
    today = predict_load_day(d, 0, load, dates, k)
    tomorrow = predict_load_day(d, 1, load, dates, k)
    record = {'mode': mode, 'observed_slots': slot, 'factor': 1.0, 'bias_kw': 0.0}
    if slot and mode == 'prefix-ratio':
        expected = float(np.mean(today[:slot]))
        observed = float(np.mean(load[d, :slot]))
        if expected > 1e-06:
            factor = float(np.clip(observed / expected, 0.8, 1.2))
            today = np.maximum(today * factor, 0.0)
            record['factor'] = factor
        else:
            bias = observed - expected
            today = np.maximum(today + bias, 0.0)
            record['bias_kw'] = bias
    return (np.r_[today, tomorrow][slot:slot + 144], record)

class Predictor:
    """Caches belong to one immutable input/configuration; no global monkeypatching."""

    def __init__(self, inputs, load_mode='prefix-ratio', pv_mode='rolling'):
        if load_mode not in ('none', 'prefix-ratio') or pv_mode not in ('linear', 'rolling'):
            raise ValueError('Unknown forecast configuration')
        self.inputs = inputs
        self.load_mode, self.pv_mode = (load_mode, pv_mode)
        self.curves, self.scores, self.choices, self.residuals = ({}, {}, {}, {})

    def curve(self, issue, method):
        issue = pd.Timestamp(issue)
        key = (issue, method)
        if key not in self.curves:
            _, _, pv, dates, releases = self.inputs
            d = int((issue.normalize() - dates[0]).days)
            slot = issue.hour * 6
            anchor = pv[d, slot - 1] if slot else pv[d - 1, -1] if d else 0.0
            hourly = releases[issue]
            knots = np.r_[anchor, hourly]
            target = np.arange(1, 145) / 6.0
            if method == 'linear':
                values = np.interp(target, np.arange(25), knots)
            elif method == 'pchip':
                values = PchipInterpolator(np.arange(25), knots)(target)
            elif method == 'step':
                values = np.repeat(hourly, 6)
            else:
                raise ValueError('Unknown interpolation method')
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite interpolation')
            self.curves[key] = np.maximum(values, 0.0)
        return self.curves[key]

    def choose(self, issue):
        issue = pd.Timestamp(issue)
        if issue in self.choices:
            return self.choices[issue]
        record = {'mode': self.pv_mode, 'method': 'linear', 'windows': 0, 'latest_available_at': None, 'scores': {}}
        if self.pv_mode == 'rolling':
            _, _, pv, dates, releases = self.inputs
            candidates = [h for h in pd.date_range(issue - pd.Timedelta(days=30), issue - pd.Timedelta(hours=24), freq='6h') if h in releases]
            record['windows'] = len(candidates)
            if candidates:
                record['latest_available_at'] = str(candidates[-1] + pd.Timedelta(hours=24))
            if len(candidates) >= 8:
                for method in METHODS:
                    total = np.zeros(2)
                    for h in candidates:
                        key = (h, method)
                        if key not in self.scores:
                            start = int((h - dates[0]).total_seconds() / 600)
                            actual = pv.ravel()[start:start + 144]
                            if len(actual) != 144:
                                raise ValueError('Incomplete mature window')
                            err = self.curve(h, method) - actual
                            self.scores[key] = np.array([err @ err, np.abs(err).sum()])
                        total += self.scores[key]
                    record['scores'][method] = {'rmse_kw': float(np.sqrt(total[0] / (144 * len(candidates)))), 'mae_kw': float(total[1] / (144 * len(candidates)))}
                record['method'] = min(METHODS, key=lambda m: (record['scores'][m]['rmse_kw'], record['scores'][m]['mae_kw'], METHODS.index(m)))
        self.choices[issue] = record
        return record

    def matrix(self, issue):
        issue = pd.Timestamp(issue)
        method = self.choose(issue)['method']
        releases = self.inputs[4]
        matrix = np.full((144, 4), np.nan)
        for age in range(4):
            vintage = issue - pd.Timedelta(hours=age * 6)
            if vintage in releases:
                n = 144 - age * 36
                matrix[:n, age] = self.curve(vintage, method)[age * 36:]
        return matrix

    def january_weights(self):
        """Fit each interpolation policy on January only; do not reuse linear weights."""
        dates, releases = self.inputs[3:]
        actual = self.inputs[2].ravel()
        cutoff = pd.Timestamp('2025-02-01')
        records = [{'x': [], 'y': [], 'issue': [], 'available_at': []} for _ in range(4)]
        for issue in sorted((h for h in releases if h < cutoff)):
            matrix = self.matrix(issue)
            start = int((issue - dates[0]).total_seconds() / 600)
            for block, n in enumerate((4, 3, 2, 1)):
                a, b = (block * 36, (block + 1) * 36)
                mature = issue + pd.Timedelta(hours=6 * (block + 1))
                if mature > cutoff or not np.isfinite(matrix[a:b, :n]).all():
                    continue
                r = records[block]
                r['x'].append(matrix[a:b, :n])
                r['y'].append(actual[start + a:start + b])
                r['issue'].append(issue)
                r['available_at'].append(mature)
        for r in records:
            r['x'], r['y'] = (np.asarray(r['x']), np.asarray(r['y']))
            r['issue'], r['available_at'] = (pd.DatetimeIndex(r['issue']), pd.DatetimeIndex(r['available_at']))
        return fit_asof(records, '2025-01-01', cutoff)

    def predict(self, d, slot, k, weights):
        issue = self.inputs[3][d] + pd.Timedelta(minutes=10 * slot)
        matrix = self.matrix(issue)
        solar = np.empty(144)
        cold = issue < self.inputs[3][0] + pd.Timedelta(hours=18)
        for t in range(144):
            available = np.isfinite(matrix[t])
            w = np.zeros(4)
            if cold:
                w[0] = 1.0
            else:
                w[:len(weights[t // 36])] = weights[t // 36]
            w[~available] = 0.0
            if w.sum() <= 0:
                raise ValueError('No released prediction for target')
            solar[t] = np.nan_to_num(matrix[t]) @ (w / w.sum())
        demand, load_record = load_forecast(d, slot, self.inputs, k, self.load_mode)
        return ((demand - solar) / 6, {'load': load_record, 'pv': self.choose(issue)})

    def scenarios(self, d, slot, k, weights):
        base, record = self.predict(d, slot, k, weights)
        origins = list(range(max(3, d - 45), d))[-3:]
        signature = tuple((tuple((float(x) for x in w)) for w in weights))
        residuals = []
        net = ((self.inputs[1] - self.inputs[2]) / 6).ravel()
        for h in origins:
            key = (h, slot, k, signature)
            if key not in self.residuals:
                historical, _ = self.predict(h, slot, k, weights)
                start = h * 144 + slot
                self.residuals[key] = net[start:start + 144] - historical
            residuals.append(self.residuals[key])
        paths = base[None, :] + np.asarray(residuals) if residuals else base[None, :]
        return (paths, base, origins, record)



# solve_question3_v4: extracted implementation; numerical operations preserved.


def load_inputs(base):
    for filename, sheets, offset in [('附件2.xlsx', ('小区负载', '光伏发电实际功率'), 1), ('附件3.xlsx', (0,), 2)]:
        for sheet in sheets:
            frame = pd.read_excel(attachment_dir(base) / filename, sheet_name=sheet)
            values = frame.iloc[:, offset:].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'Invalid raw power in {filename}/{sheet}')
    return load_inputs_core(base)

def simulate_day(d, soc, inputs, k, weights, predictor, training, time_limit=60):
    price, load, pv, dates, _ = inputs
    lam = estimate_lambda(price)
    rows, versions = ([], [])
    plan = current = None
    for slot in range(144):
        if slot % 36 == 0:
            paths, base, origins, detail = predictor.scenarios(d, slot, k, weights)
            solution, log = optimize(paths, np.tile(price, 2)[slot:slot + 144], soc, lam, None if slot == 0 else plan[slot:], time_limit)
            if slot == 0:
                plan = solution.copy()
                current = plan.copy()
            else:
                current[slot:] = solution[:144 - slot]
            versions.append({'issue_time': str(dates[d] + pd.Timedelta(minutes=10 * slot)), 'first_slot': slot + 1, 'executed_slots': 36, 'lookahead_slots': 144, 'soc': float(soc), 'selected_k': k, 'weights': [w.tolist() for w in weights], 'weight_training': training, 'forecast_detail': detail, 'residual_dates': [str(dates[h].date()) for h in origins], 'forecast_kwh': base.tolist(), 'scenarios_kwh': paths.tolist(), 'regular_draft_kwh': solution.tolist(), 'lambda': lam, 'solver': log})
        net = (load[d, slot] - pv[d, slot]) / 6
        c, b, e, w, end = dispatch(net, current[slot], soc)
        pc = price[slot] * plan[slot]
        increase = 1.5 * price[slot] * max(current[slot] - plan[slot], 0.0)
        refund = 0.5 * price[slot] * max(plan[slot] - current[slot], 0.0)
        emergency = 5 * price[slot] * e
        rows.append({'date': str(dates[d].date()), 'slot': slot + 1, 'interval_start': str(dates[d] + pd.Timedelta(minutes=slot * 10)), 'interval_end': str(dates[d] + pd.Timedelta(minutes=(slot + 1) * 10)), 'net_kwh': net, 'plan_kwh': plan[slot], 'final_regular_kwh': current[slot], 'emergency_kwh': e, 'charge_kwh': c, 'discharge_kwh': b, 'surplus_kwh': w, 'soc_start_kwh': soc, 'soc_end_kwh': end, 'price': price[slot], 'plan_cost': pc, 'increase_cost': increase, 'refund_amount': refund, 'reduction_cost': -refund, 'emergency_cost': emergency, 'total_cost': pc + increase - refund + emergency})
        soc = end
    frame = pd.DataFrame(rows)
    verify_ledger(frame)
    return (frame, versions, soc)

def run(args):
    inputs = load_inputs(args.base_dir)
    _, load, pv, dates, _ = inputs
    stop = pd.Timestamp(args.end_date)
    if stop not in dates or stop < pd.Timestamp('2025-02-01'):
        raise ValueError('Invalid end date')
    predictor = Predictor(inputs, args.load_correction, args.pv_interpolation)
    records = predictor.january_weights()
    schedule = {f'2025-{m:02d}': {'mode': 'january-full', 'effective_at': str(pd.Timestamp(2025, m, 1)), 'training_issue_start': '2025-01-01 00:00:00', 'latest_training_target': max((r['latest_available_at'] for r in records)), 'blocks': records} for m in range(2, 13)}
    selected_k, k_table = calibrate_k_on_january(dates, load, pv)
    files = [SCRIPT] + [attachment_dir(args.base_dir) / f'附件{i}.xlsx' for i in (1, 2, 3)]
    meta = {'model': MODEL, 'settlement': '50-percent-net-refund', 'selected_k': selected_k, 'weight_mode': 'january-full', 'weight_schedule': schedule, 'newest_only': False, 'adjustments': True, 'load_correction': args.load_correction, 'pv_interpolation': args.pv_interpolation, 'warmup': 'unchanged-v3-january-8-31', 'time_limit': args.time_limit, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}, 'hashes': {p.name: digest(p) for p in files}}
    out = args.output.resolve()
    if out.exists() and (not args.resume):
        raise FileExistsError('Use a new output directory or matching --resume')
    out.mkdir(parents=True, exist_ok=True)
    mp = out / 'metadata.json'
    if args.resume:
        if not mp.exists() or json.loads(mp.read_text(encoding='utf-8')) != meta:
            raise ValueError('Resume metadata changed or missing')
    else:
        save_json(mp, meta)
    save_json(out / 'weight_schedule.json', schedule)
    k_table.to_csv(out / 'k_calibration.csv', index=False)
    if any((p.name[:10] > str(stop.date()) for p in out.glob('*_complete.json'))):
        raise ValueError('Resume end date precedes existing completed days')
    cache = {}
    soc = 6000.0
    frames = []
    for d in range(7, int(dates.get_loc(stop)) + 1):
        day = str(dates[d].date())
        csv = out / f'{day}_ledger.csv'
        vp = out / f'{day}_versions.json'
        cp = out / f'{day}_complete.json'
        if cp.exists():
            checkpoint = json.loads(cp.read_text(encoding='utf-8'))
            if digest(csv) != checkpoint['ledger_sha256'] or digest(vp) != checkpoint['versions_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            frame = pd.read_csv(csv)
            verify_ledger(frame)
            if frame.date.tolist() != [day] * 144 or abs(soc - frame.soc_start_kwh.iloc[0]) > TOL:
                raise ValueError('Resume day/SOC chain broken')
            soc = float(frame.soc_end_kwh.iloc[-1])
        else:
            try:
                if d < 31:
                    training = {'mode': 'startup-0.92', 'effective_at': '2025-01-01 00:00:00', 'latest_training_target': None}
                    frame, versions, soc = simulate_warmup_day(d, soc, inputs, 7, startup_weights(), cache, single=True, time_limit=args.time_limit, weight_metadata=training)
                    for v in versions:
                        v['forecast_detail'] = {'warmup': 'unchanged-v3'}
                else:
                    training = schedule[day[:7]]
                    weights = [np.asarray(r['weights']) for r in records]
                    frame, versions, soc = simulate_day(d, soc, inputs, selected_k, weights, predictor, training, args.time_limit)
            except Exception as exc:
                save_json(out / 'last_failure.json', {'date': day, 'error': str(exc)})
                raise
            frame.to_csv(csv, index=False)
            save_json(vp, versions)
            save_json(cp, {'date': day, 'ledger_sha256': digest(csv), 'versions_sha256': digest(vp)})
            if d % 7 == 0 or d in (30, 31, int(dates.get_loc(stop))):
                print(f'{day}: cost={frame.total_cost.sum():.2f}, emergency={frame.emergency_kwh.sum():.2f}', flush=True)
        if d >= 31:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    error = verify_ledger(result)
    result.to_csv(out / 'ledger.csv', index=False)
    summary = {n: float(result[n].sum()) for n in ('plan_cost', 'increase_cost', 'refund_amount', 'reduction_cost', 'emergency_cost', 'total_cost', 'emergency_kwh', 'surplus_kwh')}
    summary.update(days=len(frames), start_date='2025-02-01', end_date=str(stop.date()), annual_complete=len(frames) == 334, weight_mode='january-full', load_correction=args.load_correction, pv_interpolation=args.pv_interpolation, initial_soc=float(result.soc_start_kwh.iloc[0]), final_soc=soc, max_validation_error=error, decrease_kwh=float(np.maximum(result.plan_kwh - result.final_regular_kwh, 0).sum()), increase_kwh=float(np.maximum(result.final_regular_kwh - result.plan_kwh, 0).sum()), emergency_slots=int((result.emergency_kwh > TOL).sum()))
    save_json(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary



# validate_question3_annual: extracted implementation; numerical operations preserved.


def raw_inputs(base):
    price = pd.read_excel(attachment_dir(base) / '附件1.xlsx', sheet_name='Sheet1')['电价'].values[:144].astype(float)
    load = pd.read_excel(attachment_dir(base) / '附件2.xlsx', sheet_name='小区负载', index_col=0, parse_dates=True)
    pv = pd.read_excel(attachment_dir(base) / '附件2.xlsx', sheet_name='光伏发电实际功率', index_col=0, parse_dates=True)
    return (price, load.values[:, :144].astype(float), np.maximum(pv.values[:, :144].astype(float), 0), pd.DatetimeIndex(load.index))



# validate_question3_v4: extracted implementation; numerical operations preserved.


def audit(folder, base=ROOT):
    price, load, pv, dates = raw_inputs(base)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    if meta['model'] != 'Q3-six-hour-24h-MILP-refund-v4':
        raise ValueError('This audit requires the v4 runner')
    for name, expected in meta['hashes'].items():
        path = attachment_dir(base) / name if name.endswith('.xlsx') else SCRIPT
        if digest(path) != expected:
            raise ValueError(f'Input/source hash changed since run: {name}')
    monthly = json.loads((folder / 'weight_schedule.json').read_text(encoding='utf-8'))
    if monthly != meta['weight_schedule']:
        raise ValueError('Weight schedule differs from run metadata')
    expected_dates = pd.date_range('2025-01-08', summary['end_date']).strftime('%Y-%m-%d').tolist()
    files = sorted(folder.glob('*_ledger.csv'))
    if [p.name[:10] for p in files] != expected_dates:
        raise ValueError('Incomplete or extra dated ledger files')
    soc = 6000.0
    frames = []
    solves = 0
    max_gap = 0.0
    worst = {}

    def close(name, a, b):
        value = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        worst[name] = max(worst.get(name, 0.0), value)
        if not np.isfinite(value) or value > TOL:
            raise ValueError(f'{name} failed: {value}')
    for path in files:
        day = path.name[:10]
        d = int(dates.get_loc(day))
        cp = json.loads(path.with_name(f'{day}_complete.json').read_text(encoding='utf-8'))
        vf = path.with_name(f'{day}_versions.json')
        if digest(path) != cp['ledger_sha256'] or digest(vf) != cp['versions_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {day}')
        f = pd.read_csv(path)
        if f.date.tolist() != [day] * 144 or f.slot.tolist() != list(range(1, 145)):
            raise ValueError('Wrong day/slot keys')
        if not np.isfinite(f.select_dtypes('number')).all().all():
            raise ValueError('Nonfinite ledger')
        q, g, c, b, e, w = [f[n].to_numpy() for n in ('plan_kwh', 'final_regular_kwh', 'charge_kwh', 'discharge_kwh', 'emergency_kwh', 'surplus_kwh')]
        start, end = (f.soc_start_kwh.to_numpy(), f.soc_end_kwh.to_numpy())
        net = (load[d] - pv[d]) / 6
        close('raw_net', f.net_kwh, net)
        close('raw_price', f.price, price)
        close('balance', g + b + e, net + c + w)
        close('soc', end, start + 0.9 * c - b / 0.9)
        close('soc_chain', start, np.r_[soc, end[:-1]])
        soc = float(end[-1])
        if min(q.min(), g.min(), c.min(), b.min(), e.min(), w.min()) < -TOL or min(start.min(), end.min()) < 1200 - TOL or max(start.max(), end.max()) > 10800 + TOL or (max(c.max(), b.max()) > 5000 / 6 + TOL) or np.any((c > TOL) & ((b > TOL) | (e > TOL))):
            raise ValueError('Physical bounds/mode failed')
        amounts = np.c_[price * q, 1.5 * price * np.maximum(g - q, 0), -0.5 * price * np.maximum(q - g, 0), 5 * price * e]
        close('settlement', f[['plan_cost', 'increase_cost', 'reduction_cost', 'emergency_cost']], amounts)
        close('refund_amount', f.refund_amount, -amounts[:, 2])
        close('total', f.total_cost, amounts.sum(axis=1))
        versions = json.loads(vf.read_text(encoding='utf-8'))
        if [v['first_slot'] for v in versions] != [1, 37, 73, 109]:
            raise ValueError('Not four six-hour updates')
        close('frozen_q', q, versions[0]['regular_draft_kwh'])
        for v in versions:
            k = v['first_slot'] - 1
            if v['lookahead_slots'] != 144 or v['executed_slots'] != 36:
                raise ValueError('Wrong lookahead/execution length')
            if pd.Timestamp(v['issue_time']) != pd.Timestamp(day) + pd.Timedelta(minutes=k * 10):
                raise ValueError('Wrong issue timestamp')
            close('frozen_g', g[k:k + 36], v['regular_draft_kwh'][:36])
            if d >= 31:
                training = monthly[day[:7]]
                if v['weight_training'] != training:
                    raise ValueError('Wrong monthly training provenance')
                for actual, record in zip(v['weights'], training['blocks']):
                    close('weights', actual, record['weights'])
                    if pd.Timestamp(record['latest_available_at']) > pd.Timestamp(day[:7] + '-01'):
                        raise ValueError('Weight data not mature at month start')
            close('version_soc', v['soc'], start[k])
            if d < 31:
                if v['forecast_detail'] != {'warmup': 'unchanged-v3'}:
                    raise ValueError('Warmup changed')
            else:
                detail = v['forecast_detail']
                lp = detail['load']
                solar = detail['pv']
                if lp['mode'] != meta['load_correction'] or lp['observed_slots'] != k:
                    raise ValueError('Load forecast provenance mismatch')
                if not 0.8 - TOL <= lp['factor'] <= 1.2 + TOL or not np.isfinite(lp['bias_kw']):
                    raise ValueError('Invalid load adjustment')
                if (k == 0 or lp['mode'] == 'none') and (lp['factor'] != 1 or lp['bias_kw'] != 0):
                    raise ValueError('Forbidden load correction')
                if solar['mode'] != meta['pv_interpolation'] or solar['method'] not in ('linear', 'pchip', 'step'):
                    raise ValueError('Invalid PV interpolation')
                mature = solar['latest_available_at']
                if mature is not None and pd.Timestamp(mature) > pd.Timestamp(v['issue_time']):
                    raise ValueError('Interpolation selection uses immature targets')
                if solar['mode'] == 'linear' and solar['method'] != 'linear':
                    raise ValueError('Fixed interpolation changed')
                if solar['windows'] < 8 and solar['method'] != 'linear':
                    raise ValueError('Insufficient interpolation history')
                for r in v['residual_dates']:
                    if pd.Timestamp(r) + pd.Timedelta(minutes=k * 10, hours=24) > pd.Timestamp(v['issue_time']):
                        raise ValueError('Immature residual scenario')
                for weights in v['weights']:
                    a = np.asarray(weights)
                    if not np.isfinite(a).all() or np.any(a < -TOL) or abs(a.sum() - 1) > TOL or np.any(np.diff(a) > TOL):
                        raise ValueError('Invalid blend weights')
            log = v['solver']
            if 'skipped' in log:
                if d < 31 or meta['adjustments'] or k == 0:
                    raise ValueError('Unexpected skipped solve')
                continue
            solves += 1
            relative = log['mip_gap']
            absolute = log['absolute_gap']
            if log['status'] != 0 or not (relative is not None and relative <= 0.0001 + 1e-10 or absolute <= 1e-06) or log['max_constraint_violation'] > TOL or log['simultaneous_charge_discharge'] or log['emergency_charge']:
                raise ValueError('Uncertified or physically invalid MILP')
            if relative is not None:
                max_gap = max(max_gap, relative)
        if d >= 31:
            frames.append(f)
    annual = pd.concat(frames, ignore_index=True)
    saved = pd.read_csv(folder / 'ledger.csv')
    if saved[['date', 'slot']].to_dict('list') != annual[['date', 'slot']].to_dict('list'):
        raise ValueError('Combined ledger dates differ')
    close('combined_ledger', saved.select_dtypes('number'), annual.select_dtypes('number'))
    for name in ('plan_cost', 'increase_cost', 'reduction_cost', 'refund_amount', 'emergency_cost', 'total_cost', 'emergency_kwh', 'surplus_kwh'):
        close(f'summary_{name}', summary[name], annual[name].sum())
    close('summary_initial_soc', summary['initial_soc'], annual.soc_start_kwh.iloc[0])
    close('summary_final_soc', summary['final_soc'], soc)
    days = len(annual) // 144
    if summary['days'] != days or summary['annual_complete'] != (days == 334):
        raise ValueError('Incorrect completion status')
    report = {'load_correction': meta['load_correction'], 'pv_interpolation': meta['pv_interpolation'], 'passed': True, 'weight_mode': meta['weight_mode'], 'days': days, 'annual_complete': days == 334, 'newest_only': meta['newest_only'], 'adjustments': meta['adjustments'], 'settlement': meta['settlement'], 'start_date': '2025-02-01', 'end_date': summary['end_date'], 'initial_soc': summary['initial_soc'], 'final_soc': soc, 'solves': solves, 'max_mip_gap': max_gap, 'max_residuals': worst, **{n: float(annual[n].sum()) for n in ('plan_cost', 'increase_cost', 'refund_amount', 'emergency_cost', 'total_cost', 'emergency_kwh', 'surplus_kwh')}}
    return report

def compare_reference(folder, reference):
    """Compare recomputed results. References are never read by the optimizer.

    Permit a prefix smoke run; mark full-year reproduction only for all 334 days.
    Skip only solver wall-clock seconds and source hashes, which necessarily change.
    Numerical tolerance is absolute 1e-7; also report whether values match exactly.
    """
    folder, reference = Path(folder).resolve(), Path(reference).resolve()
    if folder == reference:
        raise ValueError('Reference must be a different directory')
    new_meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    ref_meta = json.loads((reference / 'metadata.json').read_text(encoding='utf-8'))
    for key in set(new_meta) | set(ref_meta):
        if key != 'hashes' and new_meta.get(key) != ref_meta.get(key):
            raise ValueError(f'Reference configuration/environment differs: {key}')
    for i in (1, 2, 3):
        name = f'附件{i}.xlsx'
        if new_meta['hashes'][name] != ref_meta['hashes'][name]:
            raise ValueError(f'Reference input differs: {name}')
    maxima = {'ledger': 0.0, 'versions': 0.0, 'summary': 0.0, 'calibration': 0.0}
    numeric_counts = {k: 0 for k in maxima}

    def compare(a, b, category, location):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                raise ValueError(f'Keys differ at {location}')
            for key in a:
                if category == 'versions' and key == 'seconds' and location.endswith('.solver'):
                    continue
                compare(a[key], b[key], category, f'{location}.{key}')
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError(f'Lengths differ at {location}')
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, category, f'{location}[{index}]')
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            error = abs(a - b)
            if not np.isfinite(error) or error > 1e-7:
                raise ValueError(f'Numerical mismatch at {location}: {a} versus {b}')
            maxima[category] = max(maxima[category], float(error))
            numeric_counts[category] += 1
        elif a != b:
            raise ValueError(f'Value mismatch at {location}: {a!r} versus {b!r}')

    def compare_csv(a, b, category):
        fa, fb = pd.read_csv(a), pd.read_csv(b)
        if fa.columns.tolist() != fb.columns.tolist() or len(fa) != len(fb):
            raise ValueError(f'CSV structure differs: {a.name}')
        for col in fa.columns:
            compare(fa[col].tolist(), fb[col].tolist(), category, f'{a.name}.{col}')

    files = sorted(folder.glob('*_ledger.csv'))
    if not files:
        raise ValueError('No computed daily ledgers')
    exact_csv = 0
    for path in files:
        other = reference / path.name
        compare_csv(path, other, 'ledger')
        exact_csv += digest(path) == digest(other)
        name = path.name.replace('_ledger.csv', '_versions.json')
        compare(json.loads((folder / name).read_text(encoding='utf-8')),
                json.loads((reference / name).read_text(encoding='utf-8')), 'versions', name)
    compare_csv(folder / 'k_calibration.csv', reference / 'k_calibration.csv', 'calibration')
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    ref_summary = json.loads((reference / 'summary.json').read_text(encoding='utf-8'))
    full_match = summary['end_date'] == ref_summary['end_date']
    if full_match:
        compare(summary, ref_summary, 'summary', 'summary')
        compare_csv(folder / 'ledger.csv', reference / 'ledger.csv', 'ledger')
    report = {'passed': True, 'reference': str(reference), 'days_including_warmup': len(files),
              'formal_days': summary['days'], 'full_annual_reproduction': full_match and summary['annual_complete'],
              'exact_daily_csv_files': exact_csv, 'max_absolute_difference': maxima,
              'numeric_values_compared': numeric_counts, 'all_compared_numbers_exact': all(v == 0 for v in maxima.values()),
              'absolute_tolerance': 1e-7, 'excluded': ['source hashes', 'solver.seconds'],
              'source_sha256': digest(SCRIPT)}
    save_json(folder / 'reproducibility.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir', type=Path, default=ROOT, help='Directory containing 附件1/2/3.xlsx or its 附件 child')
    p.add_argument('--output', type=Path, default=None, help='New result directory; defaults to results/q3_v4_standalone under base-dir')
    p.add_argument('--end-date', default='2025-12-31')
    p.add_argument('--time-limit', type=float, default=60)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--load-correction', choices=('none', 'prefix-ratio'), default='prefix-ratio')
    p.add_argument('--pv-interpolation', choices=('linear', 'rolling'), default='rolling')
    p.add_argument('--reference', type=Path, help='Optional old v4 result folder, read only after solving/auditing')
    p.add_argument('--audit-only', action='store_true', help='Validate existing output without solving')
    return p


def main():
    args = parser().parse_args()
    args.base_dir = args.base_dir.resolve()
    args.output = args.output or args.base_dir / 'results/q3_v4_standalone'
    args.output = args.output.resolve()
    if args.reference is not None and args.reference.resolve() == args.output:
        raise ValueError('Output and reference must differ')
    if not np.isfinite(args.time_limit) or args.time_limit <= 0:
        raise ValueError('time-limit must be finite and positive')
    if not args.audit_only:
        run(args)
    report = audit(args.output, args.base_dir)
    save_json(args.output / 'validation.json', report)
    print(f"Independent audit passed: {report['days']} days, {report['solves']} solves", flush=True)
    if args.reference is not None:
        compare_reference(args.output, args.reference)


if __name__ == '__main__':
    main()
