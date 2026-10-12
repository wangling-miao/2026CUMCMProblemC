"""Independent cash settlement. Does not trust the solver objective."""

from __future__ import annotations

import numpy as np

from .constants import ADJUST_ABS_MULT, E_MAX, E_MIN, EMERGENCY_MULT, ETA_C, ETA_D, Q_BESS_MAX


def cash(price: np.ndarray, q: np.ndarray, a: np.ndarray, h: np.ndarray) -> dict[str, float]:
    """C = p a + 0.5 p |a-q| + 5 p h

    Equivalent breakdown used in the v1 read-only check:
    C = p q + 1.5 p (a-q)^+ - 0.5 p (q-a)^+ + 5 p h
    """
    base = price * a
    dev = ADJUST_ABS_MULT * price * np.abs(a - q)
    emg = EMERGENCY_MULT * price * h
    up = np.maximum(a - q, 0.0)
    down = np.maximum(q - a, 0.0)
    cost_plan = float((price * q).sum())
    cost_up = float((1.5 * price * up).sum())
    cost_down_refund = float((0.5 * price * down).sum())
    cost_h = float(emg.sum())
    cost = float((base + dev + emg).sum())
    return {
        "energy_q": float(q.sum()),
        "energy_a": float(a.sum()),
        "energy_h": float(h.sum()),
        "energy_up": float(up.sum()),
        "energy_down": float(down.sum()),
        "cost": cost,
        "cost_a": float(base.sum()),
        "cost_dev": float(dev.sum()),
        "cost_h": cost_h,
        "cost_plan": cost_plan,
        "cost_up": cost_up,
        "cost_down_refund": cost_down_refund,
    }


def audit_day(
    e0: float,
    a: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    load_e: np.ndarray,
    pv_e: np.ndarray,
    tol: float = 2e-3,
    q: np.ndarray | None = None,
    lock_until: int = 0,
    *,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> tuple[np.ndarray, list[str]]:
    n = len(a)
    E = np.empty(n + 1)
    E[0] = e0
    errs: list[str] = []
    for t in range(n):
        E[t + 1] = E[t] + eta_c * c[t] - d[t] / eta_d
        lhs = a[t] + h[t] + pv_e[t] + d[t]
        rhs = load_e[t] + c[t] + w[t]
        if lhs + tol < rhs:
            errs.append(f"t={t} short {rhs - lhs:.4f} kWh")
        if c[t] < -tol or d[t] < -tol or h[t] < -tol or a[t] < -tol:
            errs.append(f"t={t} negative power")
        if c[t] > Q_BESS_MAX + tol or d[t] > Q_BESS_MAX + tol:
            errs.append(f"t={t} bess power")
        if E[t + 1] < E_MIN - 1e-2 or E[t + 1] > E_MAX + 1e-2:
            errs.append(f"t={t} SOC {E[t + 1]:.2f}")
        if c[t] > 1e-4 and d[t] > 1e-4:
            errs.append(f"t={t} simultaneous {c[t]:.3f}/{d[t]:.3f}")
        if q is not None and t < lock_until and abs(a[t] - q[t]) > 1e-4:
            errs.append(f"t={t} locked a!=q")
    return E, errs


def split_waste(a: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Attribute unused energy: paid-but-unused vs remainder (often PV)."""
    paid = np.minimum(w, np.maximum(a, 0.0))
    return float(paid.sum()), float((w - paid).sum())
