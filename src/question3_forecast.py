# Pure forecast functions copied from teammate Q2; no solver dependency.
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
N_SLOTS=144
DELTA=1/6
LOOKAHEAD_DAYS=3
MAX_TRAIN_DAYS=31
RECENCY_DECAY=0.92
K_CANDIDATES=[3,5,7,10,14,21,28]
DEFAULT_K=7
TRAIN_START=pd.Timestamp('2025-01-01')
TRAIN_END=pd.Timestamp('2025-01-31')
TERMINAL_SOC_FLOOR=6000.0
S_MIN=1200.0
S_MAX=10800.0
ETA_D=0.9

def load_data(base_dir: Path
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    读取附件1（电价）和附件2（负载、光伏）。

    返回：
        price:    (144,) 电价数组，元/kWh
        load_arr: (n_days, 144) 负载功率，kW
        pv_arr:   (n_days, 144) 光伏功率，kW（已 clip ≥ 0）
        dates:    (n_days,) 日期索引
    """
    f1 = base_dir / "附件" / "附件1.xlsx"
    f2 = base_dir / "附件" / "附件2.xlsx"

    if not f1.exists():
        raise FileNotFoundError(f"未找到附件1：{f1}")
    if not f2.exists():
        raise FileNotFoundError(f"未找到附件2：{f2}")

    # 电价
    df1 = pd.read_excel(f1, sheet_name="Sheet1")
    if "电价" not in df1.columns:
        raise ValueError("附件1 Sheet1 中未找到列。")
    price = df1["电价"].values[:N_SLOTS].astype(float)
    assert len(price) == N_SLOTS, f"电价数据应有 {N_SLOTS} 个时段"

    # 负载
    load_df = pd.read_excel(
        f2, sheet_name="小区负载",
        index_col=0, parse_dates=True,
    )
    # 光伏
    pv_df = pd.read_excel(
        f2, sheet_name="光伏发电实际功率",
        index_col=0, parse_dates=True,
    )

    load_arr = load_df.iloc[:, :N_SLOTS].values.astype(float)
    pv_arr = pv_df.iloc[:, :N_SLOTS].values.astype(float)
    dates = pd.DatetimeIndex(pd.to_datetime(load_df.index))

    # 校验
    if load_arr.shape != pv_arr.shape:
        raise ValueError("负载数据与光伏数据形状不一致。")
    if load_arr.shape[1] != N_SLOTS:
        raise ValueError(f"每天必须有 {N_SLOTS} 个时段。")
    if len(dates) < 365:
        print(f"[WARN] 当前仅有 {len(dates)} 天数据。")

    # 光伏不能为负
    pv_arr = np.maximum(pv_arr, 0.0)

    return price, load_arr, pv_arr, dates

def net_energy_day(day_idx: int,
                   load_arr: np.ndarray,
                   pv_arr: np.ndarray) -> np.ndarray:
    """
    计算某天144个时段的净负荷电量（kWh/10min）。
    净负荷 = (负载功率 - 光伏功率) × Δ
    可以为负（光伏 > 负载时）。
    """
    return (load_arr[day_idx] - pv_arr[day_idx]) * DELTA

def predict_load_day(origin_idx: int,
                     target_day_offset: int,
                     load_arr: np.ndarray,
                     dates: pd.DatetimeIndex,
                     k: int,
                     max_train_days: int = MAX_TRAIN_DAYS
                     ) -> np.ndarray:
    """
    预测负载功率（kW），基于同星期加权历史均值 + 趋势修正。

    严格只使用 origin_idx 之前的数据。
    """
    if origin_idx <= 0:
        return np.zeros(N_SLOTS, dtype=float)

    target_idx = origin_idx + target_day_offset

    # 用真实日历星期
    if target_idx < len(dates):
        target_weekday = dates[target_idx].weekday()
    else:
        last_weekday = dates[len(dates) - 1].weekday()
        target_weekday = (last_weekday + (target_idx - len(dates) + 1)) % 7

    hist_start = max(0, origin_idx - max_train_days)
    hist_indices = list(range(hist_start, origin_idx))

    if not hist_indices:
        return np.zeros(N_SLOTS, dtype=float)

    # 优先选同星期
    same_weekday = [
        h for h in hist_indices
        if dates[h].weekday() == target_weekday
    ]

    if len(same_weekday) >= 2:
        selected = same_weekday[-min(k, len(same_weekday)):]
    else:
        selected = hist_indices[-min(k, len(hist_indices)):]

    ages = np.array([origin_idx - h for h in selected], dtype=float)
    weights = RECENCY_DECAY ** ages
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)

    # 加权基线
    pred = np.zeros(N_SLOTS, dtype=float)
    for w, h in zip(weights, selected):
        pred += w * load_arr[h]

    # 趋势修正：同星期差分中位数
    if len(same_weekday) >= 2:
        diffs = []
        for i in range(1, len(same_weekday)):
            h1, h0 = same_weekday[i], same_weekday[i - 1]
            diffs.append(load_arr[h1] - load_arr[h0])
        if diffs:
            trend = np.median(np.vstack(diffs), axis=0)
            gamma = 0.3  # 收缩系数
            pred = pred + gamma * trend

    return pred

def predict_pv_day(origin_idx: int,
                   target_day_offset: int,
                   pv_arr: np.ndarray,
                   dates: pd.DatetimeIndex,
                   w_p: int = 14
                   ) -> np.ndarray:
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

    # 光伏不能为负
    pred = np.maximum(pred, 0.0)
    return pred

def forecast_net_energy(origin_idx: int,
                        target_day_offset: int,
                        load_arr: np.ndarray,
                        pv_arr: np.ndarray,
                        dates: pd.DatetimeIndex,
                        k: int
                        ) -> np.ndarray:
    """
    预测某天的净负荷电量（kWh/10min）。
    分别预测负载和光伏，再合成。
    """
    pred_load = predict_load_day(
        origin_idx, target_day_offset, load_arr, dates, k
    )
    pred_pv = predict_pv_day(
        origin_idx, target_day_offset, pv_arr, dates
    )
    return (pred_load - pred_pv) * DELTA

def forecast_three_days(origin_idx: int,
                        load_arr: np.ndarray,
                        pv_arr: np.ndarray,
                        dates: pd.DatetimeIndex,
                        k: int) -> np.ndarray:
    """预测未来3天共432个时段的净负荷电量。"""
    parts = []
    for offset in range(LOOKAHEAD_DAYS):
        parts.append(
            forecast_net_energy(
                origin_idx, offset, load_arr, pv_arr, dates, k
            )
        )
    return np.concatenate(parts)

def calibrate_k_on_january(dates: pd.DatetimeIndex,
                           load_arr: np.ndarray,
                           pv_arr: np.ndarray
                           ) -> Tuple[int, pd.DataFrame]:
    """用1月内部滚动回测选择 K。"""

    jan_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    jan_indices = np.where(jan_mask)[0]

    if len(jan_indices) < 10:
        print("[WARN] 1月样本不足，使用默认K。")
        return DEFAULT_K, pd.DataFrame()

    first_idx = int(jan_indices[0])
    last_idx = int(jan_indices[-1])

    rows = []
    for k in K_CANDIDATES:
        sq_errors = []
        abs_errors = []
        val_start = first_idx + 7

        for d in range(val_start, last_idx + 1):
            pred = forecast_net_energy(
                origin_idx=d,
                target_day_offset=0,
                load_arr=load_arr,
                pv_arr=pv_arr,
                dates=dates,
                k=k,
            )
            actual = net_energy_day(d, load_arr, pv_arr)
            err = actual - pred
            sq_errors.extend(np.square(err).tolist())
            abs_errors.extend(np.abs(err).tolist())

        if not sq_errors:
            continue

        rows.append({
            "K": k,
            "RMSE": float(np.sqrt(np.mean(sq_errors))),
            "MAE": float(np.mean(abs_errors)),
        })

    if not rows:
        return DEFAULT_K, pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["RMSE", "MAE"])
    best_k = int(df.iloc[0]["K"])
    return best_k, df.reset_index(drop=True)

def compute_terminal_soc_floor(day_idx: int,
                                load_arr: np.ndarray,
                                pv_arr: np.ndarray,
                                dates: pd.DatetimeIndex,
                                lookback: int = 14) -> float:
    """
    基于近期净负荷统计动态计算末端SOC下限。
    净负荷高（需求压力大）→ 末端SOC应高
    净负荷低（光伏充裕）→ 末端SOC可低
    """
    start = max(0, day_idx - lookback)
    if start >= day_idx:
        return TERMINAL_SOC_FLOOR

    daily_net = []
    for h in range(start, day_idx):
        daily_net.append(float(np.sum(net_energy_day(h, load_arr, pv_arr))))

    if len(daily_net) < 3:
        return TERMINAL_SOC_FLOOR

    # 用更长窗口建立参考分布
    ref_start = max(0, day_idx - 90)
    all_daily_net = []
    for h in range(ref_start, day_idx):
        all_daily_net.append(float(np.sum(net_energy_day(h, load_arr, pv_arr))))

    if len(all_daily_net) < 7:
        return TERMINAL_SOC_FLOOR

    p10 = float(np.percentile(all_daily_net, 10))
    p90 = float(np.percentile(all_daily_net, 90))

    if p90 <= p10:
        return TERMINAL_SOC_FLOOR

    mean_net = float(np.mean(daily_net))
    ratio = float(np.clip((mean_net - p10) / (p90 - p10), 0.0, 1.0))

    FLOOR_LOW = 4000.0
    FLOOR_HIGH = 8000.0
    BUFFER = 500.0

    floor = FLOOR_LOW + ratio * (FLOOR_HIGH - FLOOR_LOW)
    floor = float(np.clip(floor, S_MIN + BUFFER, S_MAX - BUFFER))
    return floor

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
