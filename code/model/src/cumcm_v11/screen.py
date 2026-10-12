"""Contract screening with common next-day continuation to 06:00."""

from __future__ import annotations

import numpy as np

from .constants import ADJUST_ABS_MULT, EMERGENCY_MULT, ETA_C, ETA_D, SCREEN_NEXT, T
from .optimize import greedy_slot


def greedy_to_horizon(
    a_today: np.ndarray,
    q_today: np.ndarray,
    q_ref: np.ndarray,
    price_today: np.ndarray,
    price_next: np.ndarray,
    net_today: np.ndarray,
    net_next: np.ndarray,
    e0: float,
    t0: int,
    *,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> tuple[float, float, float, float, float]:
    """Greedy roll [t0, T) on the candidate, then next morning on shared q_ref.

    Returns (cost_total, cost_today, cost_next, E_midnight, E_H).
    Next-day settlement is p*q_ref + 5p*h only; no previous-day adjust penalty.
    If q_ref is empty, evaluation stops at midnight (v5 fallback).
    """
    e = float(e0)
    cost_today = 0.0
    for t in range(t0, T):
        _c, _d, h, _w, e = greedy_slot(float(a_today[t]), float(net_today[t]), 0.0, e, eta_c=eta_c, eta_d=eta_d)
        p = float(price_today[t])
        aa = float(a_today[t])
        qq = float(q_today[t])
        cost_today += p * aa + ADJUST_ABS_MULT * p * abs(aa - qq) + EMERGENCY_MULT * p * h
    e_t = float(e)
    n_next = min(SCREEN_NEXT, len(q_ref), len(net_next), len(price_next))
    cost_next = 0.0
    for t in range(n_next):
        aa = float(q_ref[t])
        _c, _d, h, _w, e = greedy_slot(aa, float(net_next[t]), 0.0, e, eta_c=eta_c, eta_d=eta_d)
        p = float(price_next[t])
        cost_next += p * aa + EMERGENCY_MULT * p * h
    return cost_today + cost_next, cost_today, cost_next, e_t, float(e)


def mix_contract(a_old: np.ndarray, a_star: np.ndarray, lam: float, t0: int) -> np.ndarray:
    out = a_old.copy()
    out[t0:] = np.maximum(a_old[t0:] + lam * (a_star[t0:] - a_old[t0:]), 0.0)
    return out


def collect_paths(
    err_n: np.ndarray,
    err_n_next: np.ndarray,
    err_z: np.ndarray | None,
    err_z_next: np.ndarray | None,
    day: int,
    issue: int,
    t0: int,
    n_want: int,
    n_next_need: int = SCREEN_NEXT,
) -> list[dict]:
    """Causal blocks: today's remaining + next-morning residuals from the same issue."""
    n_next_need = max(0, int(n_next_need))
    out: list[dict] = []
    for d in range(day - 1, -1, -1):
        today = err_n[d, issue, t0:]
        if today.size == 0 or not np.all(np.isfinite(today)):
            continue
        nxt = err_n_next[d, issue]
        if n_next_need > 0:
            if nxt.size < n_next_need or not np.all(np.isfinite(nxt[:n_next_need])):
                continue
            eps_next = nxt[:n_next_need].copy()
        else:
            eps_next = np.zeros(0)
        rec = {"src_day": int(d), "eps": today, "eps_next": eps_next, "z": None, "z_next": None}
        if err_z is not None:
            zb = err_z[d, issue, t0:] if err_z.ndim == 3 else err_z[d, t0:]
            rec["z"] = zb if np.all(np.isfinite(zb)) else None
        if n_next_need > 0 and err_z_next is not None:
            zn = err_z_next[d, issue]
            rec["z_next"] = zn[:n_next_need].copy() if np.all(np.isfinite(zn[:n_next_need])) else None
        out.append(rec)
        if len(out) >= n_want:
            break
    return out


def terminal_shortfall(e_keep: float, e_cand: float) -> float:
    return max(0.0, float(e_keep) - float(e_cand))


def net_gain(g: float, e_keep: float, e_cand: float, kappa: float) -> float:
    """Ĝ = G − κ D, D = [E_keep − E_cand]^+. Extra leftover is not rewarded."""
    return float(g) - float(kappa) * terminal_shortfall(e_keep, e_cand)


def kappa_from_prices(p_bar: float, *, eta_d: float = ETA_D) -> float:
    return float(EMERGENCY_MULT * eta_d * max(p_bar, 0.0))


def screen_lambda(
    a_old: np.ndarray,
    a_star: np.ndarray,
    q: np.ndarray,
    n_hat: np.ndarray,
    n_hat_next: np.ndarray,
    p_hat: np.ndarray,
    p_hat_next: np.ndarray,
    q_ref: np.ndarray,
    e0: float,
    t0: int,
    paths: list[dict],
    *,
    eps_c: float,
    r: float,
    kappa: float,
    energy_gate: bool = False,
    eps_e: float = 100.0,
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0),
    extra_contracts: dict[float, np.ndarray] | None = None,
    worst_case: bool = False,
    eta_c: float = ETA_C,
    eta_d: float = ETA_D,
) -> tuple[float, dict]:
    """Path screen with continuation to H and terminal-storage opportunity cost.

    Official: Ĝ_s = C_keep − C − κ D_s, D_s = [E_H^keep − E_H]^+.
    Accept if mean Ĝ > ε_C; when worst_case=True also require min Ĝ ≥ −r.
    The min-path gate is a frozen ablation switch, not the official rule.
    The official 100 kWh energy gate is off.
    κ D is a screening debit only; it is not added to the real bill.
    """
    n_next = min(SCREEN_NEXT, len(q_ref), len(n_hat_next), len(p_hat_next))
    q_ref_use = np.asarray(q_ref[:n_next], dtype=float)
    n_hat_n = np.asarray(n_hat_next[:n_next], dtype=float)
    p_hat_n = np.asarray(p_hat_next[:n_next], dtype=float)
    rec: dict = {
        "n_paths": len(paths),
        "chosen": 0.0,
        "reason": "keep",
        "r": float(r),
        "eps_c": float(eps_c),
        "kappa": float(kappa),
        "energy_gate": bool(energy_gate),
        "worst_case": bool(worst_case),
        "eta_c": float(eta_c),
        "eta_d": float(eta_d),
        "eps_e": float(eps_e),
        "n_next": int(n_next),
        "q_ref_sum": float(np.sum(q_ref_use)) if n_next else 0.0,
        "q_ref": [float(x) for x in q_ref_use],
    }
    if not paths:
        rec["reason"] = "cold_keep"
        return 0.0, rec

    contracts: dict[float, np.ndarray] = {float(lam): mix_contract(a_old, a_star, lam, t0) for lam in lambdas}
    if extra_contracts:
        for key, arr in extra_contracts.items():
            contracts[float(key)] = np.asarray(arr, dtype=float)

    scores: dict[float, dict] = {}
    keep_cache: dict[int, tuple] = {}
    for lam, a_lam in contracts.items():
        gs = []
        nets = []
        ds = []
        ends_ok = True
        path_rows = []
        for path in paths:
            net = n_hat.copy()
            net[t0:] = n_hat[t0:] + path["eps"]
            eps_n = path["eps_next"]
            if n_next and len(eps_n) >= n_next:
                net_n = n_hat_n + np.asarray(eps_n[:n_next], dtype=float)
            else:
                net_n = n_hat_n.copy()
            price = p_hat.copy()
            price_n = p_hat_n.copy()
            if path.get("z") is not None:
                price[t0:] = np.maximum(p_hat[t0:] * np.exp(path["z"]), 1e-4)
            if n_next and path.get("z_next") is not None and len(path["z_next"]) >= n_next:
                price_n = np.maximum(p_hat_n * np.exp(np.asarray(path["z_next"][:n_next], dtype=float)), 1e-4)
            src = int(path["src_day"])
            if src not in keep_cache:
                keep_cache[src] = greedy_to_horizon(
                    a_old, q, q_ref_use, price, price_n, net, net_n, e0, t0, eta_c=eta_c, eta_d=eta_d
                )
            c0, c0_t, c0_n, e0_t, e0_h = keep_cache[src]
            if abs(float(lam)) < 1e-12:
                cl, cl_t, cl_n, el_t, el_h = c0, c0_t, c0_n, e0_t, e0_h
            else:
                cl, cl_t, cl_n, el_t, el_h = greedy_to_horizon(
                    a_lam, q, q_ref_use, price, price_n, net, net_n, e0, t0, eta_c=eta_c, eta_d=eta_d
                )
            g = c0 - cl
            d_h = terminal_shortfall(e0_h, el_h)
            g_net = g - float(kappa) * d_h
            gs.append(g)
            nets.append(g_net)
            ds.append(d_h)
            path_rows.append(
                {
                    "src_day": int(path["src_day"]),
                    "G": float(g),
                    "G_net": float(g_net),
                    "D": float(d_h),
                    "kappa": float(kappa),
                    "C0": float(c0),
                    "C": float(cl),
                    "C0_today": float(c0_t),
                    "C0_next": float(c0_n),
                    "C_today": float(cl_t),
                    "C_next": float(cl_n),
                    "E_T0": float(e0_t),
                    "E_T": float(el_t),
                    "E_H0": float(e0_h),
                    "E_H": float(el_h),
                }
            )
            if energy_gate and el_h < e0_h - eps_e:
                ends_ok = False
        g = np.asarray(gs, dtype=float)
        g_net = np.asarray(nets, dtype=float)
        mean_g = float(g.mean())
        min_g = float(g.min())
        mean_net = float(g_net.mean())
        min_net = float(g_net.min())
        max_d = float(np.max(ds)) if ds else 0.0
        ok = bool(mean_net > eps_c and (not worst_case or min_net >= -r))
        if energy_gate:
            ok = bool(ok and ends_ok)
        scores[float(lam)] = {
            "mean_g": mean_g,
            "min_g": min_g,
            "mean_g_net": mean_net,
            "min_g_net": min_net,
            "max_D": max_d,
            "ok": ok,
            "ends_ok": bool(ends_ok),
            "paths": path_rows,
        }

    rec["scores"] = scores
    passed = [lam for lam in contracts if lam > 0 and scores[float(lam)]["ok"]]
    if not passed:
        rec["reason"] = "screen_keep"
        rec["chosen"] = 0.0
        return 0.0, rec
    best = max(passed, key=lambda lam: (scores[float(lam)]["mean_g_net"], -lam))
    rec["chosen"] = float(best)
    rec["reason"] = f"lambda={best:g}"
    rec["mean_g_net"] = scores[float(best)]["mean_g_net"]
    rec["min_g_net"] = scores[float(best)]["min_g_net"]
    rec["max_D"] = scores[float(best)]["max_D"]
    return float(best), rec
