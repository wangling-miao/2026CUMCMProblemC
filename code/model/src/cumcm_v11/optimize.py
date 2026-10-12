"""LP dispatch. Q1 path unchanged. v4 adds first-slot policy and nested cost caps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .constants import (
    ADJUST_ABS_MULT,
    E_MAX,
    E_MIN,
    EMERGENCY_MULT,
    EPS_LP,
    ETA_C,
    ETA_D,
    Q_BESS_MAX,
)

SRC_LP = 0
SRC_CLIP = 1
SRC_GREEDY = 2
SRC_GAMMA = 3


@dataclass
class Dispatch:
    a: np.ndarray
    c: np.ndarray
    d: np.ndarray
    w: np.ndarray
    h: np.ndarray
    E: np.ndarray
    cost: float
    ok: bool
    message: str = ""
    eq_dual: np.ndarray | None = None
    objective_primary: float = float("nan")
    objective_secondary: float = float("nan")
    primal_residual: float = float("nan")
    terminal_credit: float = float("nan")
    primary_final: float = float("nan")
    secondary_failure: str = ""


def _layout(n: int, use_z: bool) -> tuple[dict[str, slice], int]:
    i = 0
    sl: dict[str, slice] = {}
    for name in ("a", "c", "d", "w", "h"):
        sl[name] = slice(i, i + n)
        i += n
    if use_z:
        sl["z"] = slice(i, i + n)
        i += n
    sl["E"] = slice(i, i + n + 1)
    return sl, i + n + 1


def _eliminate_cycle(c: np.ndarray, d: np.ndarray, w: np.ndarray, *, eta_c: float = ETA_C, eta_d: float = ETA_D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = c.copy()
    d = d.copy()
    w = w.copy()
    prod = eta_c * eta_d
    for t in range(len(c)):
        if c[t] <= 1e-12 or d[t] <= 1e-12:
            continue
        delta = min(c[t], d[t] / prod)
        c[t] -= delta
        d[t] -= prod * delta
        w[t] += (1.0 - prod) * delta
    return c, d, w


def free_surplus_charge(a0: float, load0: float, pv0: float, e0: float, *, eta_c: float = ETA_C) -> tuple[float, float]:
    """Already-arrived surplus that should be absorbed. Returns (s0, c_free)."""
    s0 = float(a0 + pv0 - load0)
    if s0 <= 1e-12:
        return s0, 0.0
    room = (E_MAX - e0) / eta_c if eta_c > 0 else 0.0
    c_free = min(s0, Q_BESS_MAX, max(0.0, room))
    return s0, c_free


def solve_dispatch(
    price: np.ndarray,
    load_e: np.ndarray,
    pv_e: np.ndarray,
    e0: float,
    *,
    e_end: float | None = None,
    a_fixed: np.ndarray | None = None,
    a_ref: np.ndarray | None = None,
    a_lo: np.ndarray | None = None,
    a_hi: np.ndarray | None = None,
    h_allowed: bool = True,
    terminal_mu: float = 0.0,
    secondary_throughput: bool = True,
    first_slot_policy: bool = False,
    allow_emergency_charge: bool = False,
    primary_cap: float | None = None,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> Dispatch:
    """Minimize F = cash − ν E_H, then throughput subject to F ≤ F* + ε_LP."""
    n = int(len(price))
    assert load_e.shape == (n,) and pv_e.shape == (n,)
    use_z = a_ref is not None
    sl, nvar = _layout(n, use_z)

    def primary_vec() -> np.ndarray:
        cobj = np.zeros(nvar)
        cobj[sl["a"]] = price
        cobj[sl["h"]] = EMERGENCY_MULT * price
        if use_z:
            cobj[sl["z"]] = ADJUST_ABS_MULT * price
        cobj[sl["E"].stop - 1] = -terminal_mu
        return cobj

    c_primary = primary_vec()

    def apply_first_slot(bounds: list) -> None:
        if not first_slot_policy:
            return
        if a_fixed is None or not np.isfinite(a_fixed[0]):
            return
        a0 = float(max(0.0, a_fixed[0]))
        s0, c_free = free_surplus_charge(a0, float(load_e[0]), float(pv_e[0]), e0, eta_c=eta_c)
        if s0 >= -1e-9:
            hi_c = c_free if not allow_emergency_charge else min(Q_BESS_MAX, max(c_free, (E_MAX - e0) / eta_c))
            bounds[sl["c"].start] = (c_free, max(c_free, hi_c))
            bounds[sl["d"].start] = (0.0, 0.0)
            bounds[sl["h"].start] = (0.0, 0.0)
        else:
            need = -s0
            avail = eta_d * max(0.0, e0 - E_MIN)
            bounds[sl["c"].start] = (0.0, 0.0)
            bounds[sl["d"].start] = (0.0, min(need, Q_BESS_MAX, avail))

    def solve_level(throughput: bool, f_cap: float | None):
        cobj = np.zeros(nvar)
        if not throughput:
            cobj[:] = c_primary
        else:
            cobj[sl["c"]] = 1.0
            cobj[sl["d"]] = 1.0

        bounds = [(0.0, None)] * nvar
        for t in range(n):
            bounds[sl["c"].start + t] = (0.0, Q_BESS_MAX)
            bounds[sl["d"].start + t] = (0.0, Q_BESS_MAX)
            if not h_allowed:
                bounds[sl["h"].start + t] = (0.0, 0.0)
            lo = 0.0 if a_lo is None or not np.isfinite(a_lo[t]) else float(max(0.0, a_lo[t]))
            hi = None if a_hi is None or not np.isfinite(a_hi[t]) else float(max(lo, a_hi[t]))
            if a_fixed is not None and np.isfinite(a_fixed[t]):
                qv = float(max(0.0, a_fixed[t]))
                bounds[sl["a"].start + t] = (qv, qv)
            else:
                bounds[sl["a"].start + t] = (lo, hi)

        e_idx = list(range(sl["E"].start, sl["E"].stop))
        bounds[e_idx[0]] = (float(e0), float(e0))
        for k in range(1, n):
            bounds[e_idx[k]] = (E_MIN, E_MAX)
        if e_end is None:
            bounds[e_idx[-1]] = (E_MIN, E_MAX)
        else:
            bounds[e_idx[-1]] = (float(e_end), float(e_end))

        apply_first_slot(bounds)

        n_eq = 2 * n
        A_eq = np.zeros((n_eq, nvar))
        b_eq = np.zeros(n_eq)
        for t in range(n):
            r = t
            A_eq[r, sl["a"].start + t] = 1.0
            A_eq[r, sl["h"].start + t] = 1.0
            A_eq[r, sl["d"].start + t] = 1.0
            A_eq[r, sl["c"].start + t] = -1.0
            A_eq[r, sl["w"].start + t] = -1.0
            b_eq[r] = load_e[t] - pv_e[t]
            r2 = n + t
            A_eq[r2, e_idx[t + 1]] = 1.0
            A_eq[r2, e_idx[t]] = -1.0
            A_eq[r2, sl["c"].start + t] = -eta_c
            A_eq[r2, sl["d"].start + t] = 1.0 / eta_d

        A_ub = []
        b_ub = []
        if use_z:
            for t in range(n):
                if not np.isfinite(a_ref[t]):
                    bounds[sl["z"].start + t] = (0.0, 0.0)
                    continue
                row = np.zeros(nvar)
                row[sl["a"].start + t] = 1.0
                row[sl["z"].start + t] = -1.0
                A_ub.append(row)
                b_ub.append(float(a_ref[t]))
                row = np.zeros(nvar)
                row[sl["a"].start + t] = -1.0
                row[sl["z"].start + t] = -1.0
                A_ub.append(row)
                b_ub.append(float(-a_ref[t]))
        if f_cap is not None:
            A_ub.append(c_primary.copy())
            b_ub.append(float(f_cap))

        A_ub_m = np.vstack(A_ub) if A_ub else None
        b_ub_v = np.asarray(b_ub, dtype=float) if b_ub else None

        res = linprog(
            cobj,
            A_ub=A_ub_m,
            b_ub=b_ub_v,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 5.0},
        )
        if not res.success or res.x is None:
            return None, res.message, None, None
        dual = None
        eqlin = getattr(res, "eqlin", None)
        if eqlin is not None and getattr(eqlin, "marginals", None) is not None:
            dual = np.asarray(eqlin.marginals, dtype=float)[:n]
        resid = float(np.max(np.abs(A_eq @ res.x - b_eq))) if n_eq else 0.0
        return res.x, "ok", dual, resid

    x, msg, dual, resid = solve_level(False, primary_cap)
    if x is None:
        return Dispatch(
            a=np.zeros(n),
            c=np.zeros(n),
            d=np.zeros(n),
            w=np.zeros(n),
            h=np.zeros(n),
            E=np.full(n + 1, e0),
            cost=np.nan,
            ok=False,
            message=msg,
        )

    f_star = float(c_primary @ x)
    cash0 = float(np.dot(price, x[sl["a"]]) + EMERGENCY_MULT * np.dot(price, x[sl["h"]]))
    if use_z:
        cash0 += float(ADJUST_ABS_MULT * np.dot(price, x[sl["z"]]))
    term0 = float(terminal_mu * x[sl["E"].stop - 1])
    secondary = float("nan")
    secondary_failure = ""
    if secondary_throughput:
        x2, msg2, _, resid2 = solve_level(True, f_cap=f_star + EPS_LP)
        if x2 is not None and float(c_primary @ x2) <= f_star + EPS_LP + 1e-8:
            x = x2
            msg = msg2
            if resid2 is not None:
                resid = resid2
            secondary = float(np.sum(x[sl["c"]]) + np.sum(x[sl["d"]]))
        else:
            secondary_failure = msg2 if x2 is None else "Secondary solution exceeded the primary cost cap."

    a = np.array(x[sl["a"]], dtype=float)
    c = np.array(x[sl["c"]], dtype=float)
    d = np.array(x[sl["d"]], dtype=float)
    w = np.array(x[sl["w"]], dtype=float)
    h = np.array(x[sl["h"]], dtype=float)
    c, d, w = _eliminate_cycle(c, d, w, eta_c=eta_c, eta_d=eta_d)
    E = np.empty(n + 1)
    E[0] = e0
    for t in range(n):
        E[t + 1] = E[t] + eta_c * c[t] - d[t] / eta_d

    cash = float(np.dot(price, a) + EMERGENCY_MULT * np.dot(price, h))
    if use_z:
        ref = np.where(np.isfinite(a_ref), a_ref, a)
        cash += float(ADJUST_ABS_MULT * np.dot(price, np.abs(a - ref)))
    term = float(terminal_mu * E[-1])
    f_final = float(cash - term)
    A_eq = np.zeros((2 * n, nvar))
    b_eq = np.zeros(2 * n)
    e_idx = list(range(sl["E"].start, sl["E"].stop))
    for t in range(n):
        A_eq[t, sl["a"].start + t] = 1.0
        A_eq[t, sl["h"].start + t] = 1.0
        A_eq[t, sl["d"].start + t] = 1.0
        A_eq[t, sl["c"].start + t] = -1.0
        A_eq[t, sl["w"].start + t] = -1.0
        b_eq[t] = load_e[t] - pv_e[t]
        A_eq[n + t, e_idx[t + 1]] = 1.0
        A_eq[n + t, e_idx[t]] = -1.0
        A_eq[n + t, sl["c"].start + t] = -eta_c
        A_eq[n + t, sl["d"].start + t] = 1.0 / eta_d
    x_final = np.zeros(nvar)
    x_final[sl["a"]] = a
    x_final[sl["c"]] = c
    x_final[sl["d"]] = d
    x_final[sl["w"]] = w
    x_final[sl["h"]] = h
    x_final[sl["E"]] = E
    resid_final = float(np.max(np.abs(A_eq @ x_final - b_eq)))
    return Dispatch(
        a=a, c=c, d=d, w=w, h=h, E=E, cost=cash, ok=True, message=msg,
        eq_dual=dual, objective_primary=f_star, objective_secondary=secondary,
        primal_residual=resid_final,
        terminal_credit=term,
        primary_final=f_final,
        secondary_failure=secondary_failure,
    )


def greedy_slot(a: float, load_e: float, pv_e: float, e: float, *, eta_c: float = ETA_C, eta_d: float = ETA_D) -> tuple[float, float, float, float, float]:
    surplus = a + pv_e - load_e
    if surplus >= 0:
        room = (E_MAX - e) / eta_c if eta_c > 0 else 0.0
        c = min(surplus, Q_BESS_MAX, max(0.0, room))
        d = 0.0
        h = 0.0
        w = surplus - c
    else:
        need = -surplus
        avail = eta_d * max(0.0, e - E_MIN)
        d = min(need, Q_BESS_MAX, avail)
        c = 0.0
        h = need - d
        w = 0.0
    e2 = e + eta_c * c - d / eta_d
    e2 = min(E_MAX, max(E_MIN, e2))
    return c, d, h, w, e2


def mix_signed(c_g: float, d_g: float, c_m: float, d_m: float, gamma: float) -> tuple[float, float]:
    """b = d-c; b4 = (1-γ) b_G + γ b_M. Avoids simultaneous charge/discharge."""
    g = float(np.clip(gamma, 0.0, 1.0))
    b_g = d_g - c_g
    b_m = d_m - c_m
    b = (1.0 - g) * b_g + g * b_m
    if b >= 0:
        return 0.0, b
    return -b, 0.0


def realize_slot(
    a: float,
    load_e: float,
    pv_e: float,
    e: float,
    c: float,
    d: float,
    *,
    absorb_free: bool = True,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
    allow_emergency_charge: bool = False,
) -> tuple[float, float, float, float, float, str]:
    """Apply a candidate (c,d) with physics. Optionally force free-surplus charge."""
    surplus = a + pv_e - load_e
    room = max(0.0, (E_MAX - e) / eta_c if eta_c > 0 else 0.0)
    avail = eta_d * max(0.0, e - E_MIN)
    c = float(max(0.0, c))
    d = float(max(0.0, d))
    reason = "ok"
    if surplus >= -1e-12:
        s = max(0.0, surplus)
        c_free = min(s, Q_BESS_MAX, room)
        if absorb_free and c + 1e-9 < c_free:
            c = c_free
            reason = "free_surplus"
        c = min(c, s if not allow_emergency_charge else Q_BESS_MAX, Q_BESS_MAX, room)
        d = 0.0
        h = 0.0 if not allow_emergency_charge else max(0.0, c - s)
        w = max(0.0, s - c)
    else:
        need = -surplus
        c = 0.0
        d = min(d, need, Q_BESS_MAX, avail)
        h = need - d
        w = 0.0
    e2 = e + eta_c * c - d / eta_d
    e2 = min(E_MAX, max(E_MIN, e2))
    return c, d, h, w, e2, reason


execute_slot = greedy_slot
myopic_step = greedy_slot
