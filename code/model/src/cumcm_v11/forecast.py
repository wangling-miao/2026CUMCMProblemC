"""Causal forecasts with bias correction, official-forecast fusion, and error quantiles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .constants import (
    DEFAULT_FUSION_W,
    N_DAYS,
    NIGHT_HOURS,
    PV_NIGHT_KW,
    T,
)
from .data import Data, issued_calendar


def _median(xs: deque | list) -> float:
    if not xs:
        return 0.0
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _mae(xs: deque | list) -> float:
    if not xs:
        return float("nan")
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def _quantile(xs: deque | list, q: float, min_n: int) -> float:
    arr = np.asarray(list(xs), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_n:
        return 0.0
    return float(np.quantile(arr, q))


def _flatten_blocks(blocks: deque | list) -> np.ndarray:
    if not blocks:
        return np.zeros(0)
    parts = []
    for b in blocks:
        arr = np.asarray(b, dtype=float).ravel()
        parts.append(arr)
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)


@dataclass
class ForecastSnapshot:
    """Immutable prediction actually sent to every candidate at one decision."""

    snapshot_id: str
    day: int
    t0: int
    n: int
    decision_hour: int
    decision_origin: int
    pv_issue_used: int
    last_observed_slot: int
    actual_slot_kw: float
    load_kw: np.ndarray
    pv_kw: np.ndarray
    price: np.ndarray
    load_day: np.ndarray
    pv_day: np.ndarray
    price_day: np.ndarray

    @property
    def net_kwh(self) -> np.ndarray:
        from .data import to_energy

        return to_energy(self.load_kw - self.pv_kw)

    def as_log(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "day": int(self.day),
            "t0": int(self.t0),
            "n": int(self.n),
            "decision_hour": int(self.decision_hour),
            "decision_origin": int(self.decision_origin),
            "pv_issue_used": int(self.pv_issue_used),
            "last_observed_slot": int(self.last_observed_slot),
            "actual_slot_kw": float(self.actual_slot_kw),
            "load_kw": [float(x) for x in self.load_kw],
            "pv_kw": [float(x) for x in self.pv_kw],
            "price": [float(x) for x in self.price],
            "net_kwh": [float(x) for x in self.net_kwh],
        }


def blend(fc: np.ndarray, actual: np.ndarray, t0: int, decay: float) -> np.ndarray:
    """After seeing slot t0, shift remaining forecast by a decaying bias."""
    out = fc.copy()
    n = len(out)
    if t0 < 0:
        return out
    if t0 >= n:
        return out
    bias = actual[t0] - fc[t0]
    k = np.arange(1, n - t0)
    out[t0 + 1 :] = out[t0 + 1 :] + bias * (decay**k)
    out[t0] = actual[t0]
    return out


class ForecastEngine:
    """All forecasts are causal: actuals of day `asof` are not used for day `asof`."""

    def __init__(self, data: Data, cfg: SimConfig):
        self.data = data
        self.cfg = cfg
        self.load_week = np.zeros((N_DAYS, T))
        self.pv_hist = np.zeros((N_DAYS, T))
        self.price_week = np.zeros((N_DAYS, T))
        for d in range(N_DAYS):
            self.load_week[d] = data.load[d - 7] if d >= 7 else data.load1
            self.pv_hist[d] = self._pv_hist_asof(d, d)
            if cfg.price_mode == "att1":
                self.price_week[d] = data.price1
            else:
                self.price_week[d] = data.price[d - 7] if d >= 7 else data.price1

        win = cfg.err_window
        # Hourly series: in "days" mode each item is a 6-slot block from one completed day.
        self.net_err: list[list[deque]] = [
            [deque(maxlen=win) for _ in range(24)] for _ in range(4)
        ]
        self.iss_raw_err: list[list[deque]] = [
            [deque(maxlen=win) for _ in range(24)] for _ in range(4)
        ]
        self.iss_bc_err: list[list[deque]] = [
            [deque(maxlen=win) for _ in range(24)] for _ in range(4)
        ]
        self.hist_raw_err: list[deque] = [deque(maxlen=win) for _ in range(24)]
        self.hist_bc_err: list[deque] = [deque(maxlen=win) for _ in range(24)]
        self.iss_err = self.iss_raw_err  # alias used by fusion bias of the official series
        self.hist_pv_err = self.hist_raw_err
        self.load_diff: list[deque] = [deque(maxlen=win) for _ in range(T)]
        self.price_diff: list[deque] = [deque(maxlen=win) for _ in range(T)]
        self.pv_hist_err_slot: list[deque] = [deque(maxlen=win) for _ in range(T)]
        self.err_window_unit = getattr(cfg, "err_window_unit", "days")

        self.last_load_fc = np.zeros(T)
        self.last_pv_fc = np.zeros(T)
        self.last_price_fc = np.zeros(T)
        self.issue_net: dict[int, np.ndarray] = {}
        self.issue_pv: dict[int, np.ndarray] = {}

    def _pv_hist_asof(self, target: int, asof: int) -> np.ndarray:
        """Same-slot mean of up to 7 completed days strictly before `asof`."""
        end = min(asof, N_DAYS)
        start = max(0, end - 7)
        block = self.data.pv[start:end]
        if block.size == 0:
            return self.data.pv1.copy()
        if (not self.cfg.pv_weighted) or len(block) == 1:
            return block.mean(axis=0)
        w = np.arange(1, len(block) + 1, dtype=float)
        return (w / w.sum()) @ block

    def _week_bias(self, diffs: list[deque]) -> np.ndarray:
        if not self.cfg.load_bias:
            return np.zeros(T)
        out = np.zeros(T)
        for t in range(T):
            out[t] = _median(diffs[t])
        return out

    def load_fc(self, target: int, asof: int) -> np.ndarray:
        if target < 0 or target >= N_DAYS:
            base = self.data.load1.copy()
        elif target >= 7:
            base = self.data.load[target - 7].copy()
        else:
            base = self.data.load1.copy()
        bias = self._week_bias(self.load_diff) if self.cfg.load_bias else 0.0
        return np.maximum(base + bias, 0.0)

    def price_fc(self, target: int, asof: int) -> np.ndarray:
        if self.cfg.price_mode == "att1":
            return self.data.price1.copy()
        if target < 0 or target >= N_DAYS:
            base = self.data.price1.copy()
        elif target >= 7:
            base = self.data.price[target - 7].copy()
        else:
            base = self.data.price1.copy()
        if not self.cfg.price_bias:
            return np.maximum(base, 1e-4)
        bias = np.zeros(T)
        for t in range(T):
            bias[t] = _median(self.price_diff[t])
        return np.maximum(base + bias, 1e-4)

    def pv_hist_fc(self, target: int, asof: int) -> np.ndarray:
        hist = self._pv_hist_asof(target, asof)
        if not self.cfg.pv_bias:
            return np.maximum(hist, 0.0)
        bias = np.zeros(T)
        for t in range(T):
            bias[t] = _median(self.pv_hist_err_slot[t])
        return np.maximum(hist + bias, 0.0)

    def _issued_v0(self, day: int, issue: int, actual_today: np.ndarray | None) -> float:
        if issue == 0:
            return float(self.data.pv[day - 1, -1]) if day > 0 else 0.0
        last_obs = issue * 36 - 1
        if actual_today is not None:
            return float(actual_today[last_obs])
        return 0.0

    def issued_two_days(self, day: int, issue: int, actual_today: np.ndarray | None) -> np.ndarray:
        v0 = self._issued_v0(day, issue, actual_today)
        hourly = self.data.pv_fc[day, issue]
        return issued_calendar(hourly, v0, issue * 6, n_days=2)

    def _hour_values(self, deq: deque) -> list[float]:
        if self.err_window_unit == "days":
            return _flatten_blocks(deq).tolist()
        return list(deq)

    def _append_hour(self, deq: deque, hour: int, day_vals: np.ndarray) -> None:
        sl = day_vals[hour * 6 : hour * 6 + 6]
        if self.err_window_unit == "days":
            deq.append(np.asarray(sl, dtype=float).copy())
        else:
            for x in sl:
                deq.append(float(x))

    def fusion_wb(self, issue: int) -> tuple[np.ndarray, np.ndarray]:
        w = np.full(T, DEFAULT_FUSION_W, dtype=float)
        b = np.zeros(T)
        if not self.cfg.fusion:
            return w, b
        for t in range(T):
            hour = t // 6
            if hour in NIGHT_HOURS:
                w[t] = 0.5
                b[t] = 0.0
                continue
            raw_i = self._hour_values(self.iss_raw_err[issue][hour])
            raw_h = self._hour_values(self.hist_raw_err[hour])
            bc_i = self._hour_values(self.iss_bc_err[issue][hour])
            bc_h = self._hour_values(self.hist_bc_err[hour])
            b[t] = _median(raw_i)
            mae_i = _mae(bc_i if bc_i else raw_i)
            mae_h = _mae(bc_h if bc_h else raw_h)
            if not np.isfinite(mae_i) or not np.isfinite(mae_h):
                w[t] = DEFAULT_FUSION_W
            elif mae_i + mae_h < 1e-6:
                w[t] = DEFAULT_FUSION_W
            else:
                w[t] = float(np.clip(mae_h / (mae_i + mae_h + 1e-9), 0.0, 1.0))
        return w, b

    def pv_fused_today(
        self,
        day: int,
        issue: int,
        actual_today: np.ndarray | None,
        asof: int,
    ) -> np.ndarray:
        hist = self.pv_hist_fc(day, asof)
        if not self.cfg.use_att3:
            return hist
        cal = self.issued_two_days(day, issue, actual_today)
        issued = cal[0]
        if not self.cfg.fusion:
            out = np.where(np.isfinite(issued), issued, hist)
            return np.maximum(out, 0.0)
        w, b = self.fusion_wb(issue)
        fused = w * (issued + b) + (1.0 - w) * hist
        out = np.where(np.isfinite(issued), fused, hist)
        return np.maximum(out, 0.0)

    def pv_window(
        self,
        day: int,
        t0: int,
        n: int,
        issue: int,
        actual_today: np.ndarray | None,
        asof: int,
    ) -> np.ndarray:
        """n slots starting at (day, t0). Official coverage used only if issued."""
        today = self.pv_fused_today(day, issue, actual_today, asof) if self.cfg.use_att3 else self.pv_hist_fc(day, asof)
        hist_next = self.pv_hist_fc(day + 1, asof) if day + 1 < N_DAYS else self.data.pv1.copy()
        next_day = hist_next.copy()
        if self.cfg.use_att3 and issue in (1, 2, 3):
            cal = self.issued_two_days(day, issue, actual_today)
            issued_n = cal[1]
            if self.cfg.fusion:
                w, b = self.fusion_wb(issue)
                fused_n = w * (issued_n + b) + (1.0 - w) * hist_next
                next_day = np.where(np.isfinite(issued_n), fused_n, hist_next)
            else:
                next_day = np.where(np.isfinite(issued_n), issued_n, hist_next)
            next_day = np.maximum(next_day, 0.0)

        out = np.empty(n)
        d, t = day, t0
        for i in range(n):
            if d == day:
                out[i] = today[t]
            elif d == day + 1:
                out[i] = next_day[t]
            else:
                out[i] = self.pv_hist_fc(d, asof)[t] if 0 <= d < N_DAYS else self.data.pv1[t]
            t += 1
            if t >= T:
                t = 0
                d += 1
        if actual_today is not None and t0 > 0 and day < N_DAYS:
            # slots already in the window that belong to today before t0 should
            # not appear; window starts at t0.
            pass
        return np.maximum(out, 0.0)

    def load_window(self, day: int, t0: int, n: int, asof: int) -> np.ndarray:
        out = np.empty(n)
        d, t = day, t0
        cache: dict[int, np.ndarray] = {}
        for i in range(n):
            if d not in cache:
                cache[d] = self.load_fc(d, asof)
            out[i] = cache[d][t] if 0 <= d < N_DAYS else self.data.load1[t]
            t += 1
            if t >= T:
                t = 0
                d += 1
        return np.maximum(out, 0.0)

    def price_window(self, day: int, t0: int, n: int, asof: int) -> np.ndarray:
        out = np.empty(n)
        d, t = day, t0
        cache: dict[int, np.ndarray] = {}
        for i in range(n):
            if d not in cache:
                cache[d] = self.price_fc(d, asof)
            out[i] = cache[d][t] if 0 <= d < N_DAYS else self.data.price1[t]
            t += 1
            if t >= T:
                t = 0
                d += 1
        return np.maximum(out, 1e-4)

    def n_samples(self, issue: int, hour: int) -> int:
        return len(self.net_err[issue][hour])

    def error_quantile(self, issue: int, hour: int, alpha: float) -> float:
        vals = self._hour_values(self.net_err[issue][hour])
        min_n = self.cfg.min_err_samples
        if self.err_window_unit == "days":
            if len(self.net_err[issue][hour]) < min_n:
                return 0.0
            min_n = 1
        return _quantile(vals, alpha, min_n)

    def plan_buffer_quantile(self, issue: int, hour: int, alpha: float) -> float:
        """Day-ahead protection only. Archive stays 28 days; this reads a short tail."""
        deq = self.net_err[issue][hour]
        min_days = int(getattr(self.cfg, "plan_buffer_min_days", 8))
        if self.err_window_unit == "days" and len(deq) < min_days:
            return 0.0
        mode = getattr(self.cfg, "plan_buffer_mode", "days")
        if mode == "obs":
            vals = self._hour_values(deq)
            n_obs = int(getattr(self.cfg, "plan_buffer_obs", 28))
            vals = vals[-n_obs:]
            return _quantile(vals, alpha, 1 if vals else 10**9)
        if mode == "archive":
            return self.error_quantile(issue, hour, alpha)
        n_take = int(getattr(self.cfg, "plan_buffer_days", 5))
        blocks = list(deq)[-n_take:]
        vals = _flatten_blocks(blocks).tolist()
        return _quantile(vals, alpha, 1 if vals else 10**9)

    def protect_net(self, n_hat: np.ndarray, day: int, t0: int, issue: int, alpha_arr: np.ndarray) -> np.ndarray:
        """n_hat is the window net-load energy (or power); alpha_arr same length.

        Buffer is estimated in power (kW) by hour, then scaled if n_hat looks like energy.
        Callers pass power, then convert to energy. Uses plan_buffer_quantile, not the
        full 28-day archive quantile used by intra-day band_quantiles.
        """
        out = n_hat.copy()
        d, t = day, t0
        for i in range(len(n_hat)):
            buf = max(0.0, self.plan_buffer_quantile(issue, t // 6, float(alpha_arr[i])))
            out[i] = n_hat[i] + buf
            t += 1
            if t >= T:
                t = 0
                d += 1
        return out

    def band_quantiles(self, n_hat: np.ndarray, day: int, t0: int, issue: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        qlo = np.empty_like(n_hat)
        qhi = np.empty_like(n_hat)
        t = t0
        for i in range(len(n_hat)):
            hour = t // 6
            qlo[i] = n_hat[i] + self.error_quantile(issue, hour, lo)
            qhi[i] = n_hat[i] + self.error_quantile(issue, hour, hi)
            t += 1
            if t >= T:
                t = 0
        return np.maximum(qlo, 0.0), np.maximum(qhi, 0.0)

    def ingest_snapshot_net_err(self, snap: ForecastSnapshot, actual_L: np.ndarray, actual_V: np.ndarray) -> None:
        """Record kW net-load residuals against the frozen decision snapshot."""
        origin = int(snap.decision_origin)
        t0 = int(snap.t0)
        N = actual_L - actual_V
        pred = snap.load_day - snap.pv_day
        net_e = np.full(T, np.nan)
        for t in range(t0, T):
            if np.isfinite(pred[t]):
                net_e[t] = float(N[t] - pred[t])
        if origin == 0:
            return net_e
        for hour in range(t0 // 6, 24):
            sl = slice(hour * 6, hour * 6 + 6)
            if np.any(np.isfinite(net_e[sl])):
                pad = np.zeros(T)
                pad[sl] = np.where(np.isfinite(net_e[sl]), net_e[sl], 0.0)
                self._append_hour(self.net_err[origin][hour], hour, pad)
        return net_e

    def observe_day(
        self,
        day: int,
        load_fc0: np.ndarray,
        pv_fc0: np.ndarray,
        issue_log: dict[int, dict[str, np.ndarray]],
        snapshots: list[ForecastSnapshot] | None = None,
    ) -> dict[str, list]:
        L = self.data.load[day]
        V = self.data.pv[day]
        N = L - V
        hist_raw = self.pv_hist[day]
        hist_bc = self.pv_hist_fc(day, day)
        net0 = np.asarray(N - (load_fc0 - pv_fc0), dtype=float)
        for hour in range(24):
            self._append_hour(self.net_err[0][hour], hour, net0)
            self._append_hour(self.hist_raw_err[hour], hour, V - hist_raw)
            self._append_hour(self.hist_bc_err[hour], hour, V - hist_bc)
        for t in range(T):
            self.pv_hist_err_slot[t].append(float(V[t] - hist_raw[t]))
            if day >= 7:
                self.load_diff[t].append(float(L[t] - self.data.load[day - 7, t]))
                if self.cfg.price_mode != "att1":
                    self.price_diff[t].append(float(self.data.price[day, t] - self.data.price[day - 7, t]))
        for issue, rec in issue_log.items():
            issued = rec.get("issued")
            fused = rec.get("fused")
            if issued is None:
                continue
            issued = np.asarray(issued, dtype=float)
            _w, bb = self.fusion_wb(issue)
            issued_bc = issued + bb
            start_t = issue * 36
            raw_e = np.full(T, np.nan)
            bc_e = np.full(T, np.nan)
            for t in range(start_t, T):
                if t < len(issued) and np.isfinite(issued[t]):
                    raw_e[t] = float(V[t] - issued[t])
                    bc_e[t] = float(V[t] - issued_bc[t])
            for hour in range(start_t // 6, 24):
                sl = slice(hour * 6, hour * 6 + 6)
                if hour not in NIGHT_HOURS and np.any(np.isfinite(raw_e[sl])):
                    pad = np.zeros(T)
                    pad[sl] = np.where(np.isfinite(raw_e[sl]), raw_e[sl], 0.0)
                    self._append_hour(self.iss_raw_err[issue][hour], hour, pad)
                    pad[sl] = np.where(np.isfinite(bc_e[sl]), bc_e[sl], 0.0)
                    self._append_hour(self.iss_bc_err[issue][hour], hour, pad)
        ingested: dict[str, list] = {}
        if snapshots:
            for snap in snapshots:
                if snap.decision_origin == 0:
                    ingested[snap.snapshot_id] = [float(x) for x in net0]
                    continue
                net_e = self.ingest_snapshot_net_err(snap, L, V)
                ingested[snap.snapshot_id] = [
                    None if not np.isfinite(x) else float(x) for x in net_e
                ]
        return ingested
        return ingested
