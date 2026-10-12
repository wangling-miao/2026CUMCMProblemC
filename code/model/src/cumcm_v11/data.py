"""Load attachments into slot-ordered arrays.

Slot t is the t-th 10-minute interval of the natural day; the raw timestamp
is treated as the interval end. Template headers are stored separately and
are not used as model clocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import DATA_DIR, DT_H, N_DAYS, T


@dataclass
class Data:
    price1: np.ndarray  # (T,) typical-day price
    load1: np.ndarray
    pv1: np.ndarray
    load: np.ndarray  # (365, T) kW
    pv: np.ndarray
    price: np.ndarray  # attachment 4
    pv_fc: np.ndarray  # (365, 4, 24) kW, issue 0/6/12/18
    dates: np.ndarray  # datetime64[D] length 365
    template_labels: list[str]
    year_headers: list[str]
    att1_stamps: list[str]


def load_all() -> Data:
    att1 = pd.read_excel(DATA_DIR / "附件1.xlsx", header=0)
    price1 = att1.iloc[:, 1].to_numpy(dtype=float)
    load1 = att1.iloc[:, 2].to_numpy(dtype=float)
    pv1 = att1.iloc[:, 3].to_numpy(dtype=float)
    att1_stamps = [str(x) for x in att1.iloc[:, 0].tolist()]
    assert price1.shape == (T,)

    load_df = pd.read_excel(DATA_DIR / "附件2.xlsx", sheet_name="小区负载", header=0)
    pv_df = pd.read_excel(DATA_DIR / "附件2.xlsx", sheet_name="光伏发电实际功率", header=0)
    load = load_df.iloc[:, 1 : 1 + T].to_numpy(dtype=float)
    pv = pv_df.iloc[:, 1 : 1 + T].to_numpy(dtype=float)
    dates = pd.to_datetime(load_df.iloc[:, 0]).dt.normalize().to_numpy()
    assert load.shape == (N_DAYS, T)

    price_df = pd.read_excel(DATA_DIR / "附件4.xlsx", header=0)
    price = price_df.iloc[:, 1 : 1 + T].to_numpy(dtype=float)
    assert price.shape == (N_DAYS, T)

    att3 = pd.read_excel(DATA_DIR / "附件3.xlsx", header=0)
    att3.iloc[:, 0] = att3.iloc[:, 0].replace("", pd.NA).ffill()
    pv_fc = np.zeros((N_DAYS, 4, 24), dtype=float)
    issue_map = {0: 0, 6: 1, 12: 2, 18: 3}
    date0 = pd.Timestamp(dates[0]).normalize()
    for _, row in att3.iterrows():
        d = pd.Timestamp(row.iloc[0])
        if pd.isna(d):
            continue
        di = (d.normalize() - date0).days
        if di < 0 or di >= N_DAYS:
            continue
        raw = row.iloc[1]
        if hasattr(raw, "hour"):
            hour = int(raw.hour)
        else:
            hour = int(str(raw).split(":")[0])
        ii = issue_map[hour]
        pv_fc[di, ii, :] = np.array(row.iloc[2:26], dtype=float)

    lab_df = pd.read_excel(DATA_DIR / "附件5" / "result1.xlsx", sheet_name="计划购电量", header=0)
    template_labels = [str(x) for x in lab_df.iloc[:, 0].tolist()]
    assert len(template_labels) == T

    ydf = pd.read_excel(DATA_DIR / "附件5" / "result2.xlsx", sheet_name="计划购电量", header=0, nrows=0)
    year_headers = [str(c) for c in ydf.columns[1 : 1 + T]]
    assert len(year_headers) == T

    return Data(
        price1=price1,
        load1=load1,
        pv1=pv1,
        load=load,
        pv=pv,
        price=price,
        pv_fc=pv_fc,
        dates=dates,
        template_labels=template_labels,
        year_headers=year_headers,
        att1_stamps=att1_stamps,
    )


def to_energy(power_kw: np.ndarray) -> np.ndarray:
    return power_kw * DT_H


def clock_label(minutes: int) -> str:
    if minutes <= 0:
        return "0:00"
    if minutes >= 1440:
        return "24:00"
    return f"{minutes // 60}:{minutes % 60:02d}"


def slot_interval(t: int) -> tuple[str, str]:
    """Internal natural-day interval [start, end) for slot t."""
    return clock_label(t * 10), clock_label((t + 1) * 10)


def time_map_rows(data: Data) -> list[dict]:
    rows = []
    for t in range(T):
        start, end = slot_interval(t)
        rows.append(
            {
                "slot": t,
                "raw_stamp": data.att1_stamps[t],
                "interval_start": start,
                "interval_end": end,
                "model_interval": f"[{start}, {end})",
                "template_result1_row": t + 2,
                "template_result1_header": data.template_labels[t],
                "template_year_col_index": t + 2,
                "template_year_header": data.year_headers[t],
            }
        )
    return rows


def interpolate_hourly(hourly_24: np.ndarray, v0: float) -> np.ndarray:
    """hourly_24[k] is power at hour k+1. v0 is power at hour 0 of the same origin."""
    knots_t = np.arange(25, dtype=float)
    knots_v = np.empty(25, dtype=float)
    knots_v[0] = v0
    knots_v[1:] = hourly_24
    end_hour = (np.arange(T) + 1) / 6.0
    return np.interp(end_hour, knots_t, knots_v)


def issued_calendar(hourly_24: np.ndarray, v0: float, issue_hour: int, n_days: int = 2) -> np.ndarray:
    """10-min power from an official 24h forecast. Out-of-coverage slots are NaN.

    `hourly_24[k]` is the issued value at issue_hour + (k+1) hours.
    `v0` is the known power at issue_hour.
    Row 0 is the issue calendar day; row 1 is the next day.
    """
    knots_t = issue_hour + np.arange(25, dtype=float)
    knots_v = np.empty(25, dtype=float)
    knots_v[0] = v0
    knots_v[1:] = hourly_24
    lo, hi = float(issue_hour), float(issue_hour + 24)
    out = np.full((n_days, T), np.nan, dtype=float)
    for nd in range(n_days):
        end_h = 24.0 * nd + (np.arange(T) + 1) / 6.0
        mask = (end_h >= lo - 1e-12) & (end_h <= hi + 1e-12)
        if not np.any(mask):
            continue
        out[nd, mask] = np.interp(end_h[mask], knots_t, knots_v)
    return out
