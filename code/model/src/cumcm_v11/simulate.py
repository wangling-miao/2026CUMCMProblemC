"""Year-long causal simulation: greedy baseline plus overnight continuation screening."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig
from .constants import (
    BAND_HI,
    BAND_LO,
    CLIP_FLOAT,
    CLIP_GAMMA,
    CLIP_GREEDY,
    CLIP_NONE,
    CLIP_PHYSICS,
    E0_JAN1,
    EMERGENCY_MULT,
    EXPORT_START,
    LAMBDA_V2,
    N_DAYS,
    SCREEN_NEXT,
    SLOT_TO_ISSUE,
    T,
)
from .data import Data, to_energy
from .forecast import ForecastEngine, ForecastSnapshot, blend
from .optimize import greedy_slot, mix_signed, realize_slot, solve_dispatch
from .screen import collect_paths, mix_contract, screen_lambda
from .settle import audit_day, cash, split_waste


@dataclass
class YearRun:
    q: np.ndarray
    a: np.ndarray
    c: np.ndarray
    d: np.ndarray
    h: np.ndarray
    w: np.ndarray
    E_start: np.ndarray
    E_end: np.ndarray
    E_slot: np.ndarray
    p_plan: np.ndarray
    p_set: np.ndarray
    fc_L0: np.ndarray
    fc_V0: np.ndarray
    a_snap: np.ndarray
    c_opt: np.ndarray
    d_opt: np.ndarray
    src: np.ndarray
    clip_reason: np.ndarray
    n_lp_fail: int = 0
    n_audit: int = 0
    n_extra_reopt: int = 0
    n_adjust: int = 0
    n_keep: int = 0
    n_adjust_334: int = 0
    n_keep_334: int = 0
    n_lp_act: int = 0
    n_clip: int = 0
    n_greedy: int = 0
    n_gamma: int = 0
    n_b_gt_a: int = 0
    n_half: int = 0
    n_full: int = 0
    n_half_334: int = 0
    n_full_334: int = 0
    n_cold_keep: int = 0
    n_v2: int = 0
    n_v2_334: int = 0
    gamma: float = 0.0
    day_cost: np.ndarray = field(default_factory=lambda: np.zeros(N_DAYS))
    day_metrics: list[dict] = field(default_factory=list)
    plan_events: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    cfg_name: str = ""
    failure_logs: list[dict] = field(default_factory=list)


def _settle_prices(data: Data, day: int, price_mode: str) -> np.ndarray:
    if price_mode == "att1":
        return data.price1.copy()
    return data.price[day].copy()


def _alpha_arr(n: int, t0: int, cfg: SimConfig) -> np.ndarray:
    out = np.full(n, cfg.alpha, dtype=float)
    flex = cfg.adjust == "selective" if cfg.protection_flex is None else cfg.protection_flex
    if not flex:
        return out
    t = t0
    for i in range(n):
        if t >= 36 and t < T:
            out[i] = cfg.alpha_flex
        t += 1
        if t >= T:
            t = 0
    return out


def _win_len(day: int, t: int, horizon: int) -> int:
    n_today = T - t
    extra = T if day < N_DAYS - 1 else 0
    return min(horizon, n_today + extra)


def simulate_year(data: Data, cfg: SimConfig, *, max_days: int = N_DAYS, log=print) -> YearRun:
    def emit(msg: str) -> None:
        try:
            log(msg, flush=True)
        except TypeError:
            log(msg)

    if not (0.0 < cfg.eta_c <= 1.0 and 0.0 < cfg.eta_d <= 1.0):
        raise ValueError("Charging and discharging efficiency must lie in (0, 1].")
    n_days = min(max_days, N_DAYS)
    q = np.zeros((n_days, T))
    a = np.zeros((n_days, T))
    c = np.zeros((n_days, T))
    d = np.zeros((n_days, T))
    h = np.zeros((n_days, T))
    w = np.zeros((n_days, T))
    E_start = np.zeros(n_days)
    E_end = np.zeros(n_days)
    E_slot = np.zeros((n_days, T + 1))
    p_plan_a = np.zeros((n_days, T))
    p_set_a = np.zeros((n_days, T))
    fc_L0 = np.zeros((n_days, T))
    fc_V0 = np.zeros((n_days, T))
    a_snap = np.zeros((n_days, 4, T))
    c_opt = np.zeros((n_days, T))
    d_opt = np.zeros((n_days, T))
    src_a = np.zeros((n_days, T), dtype=np.uint8)
    clip_r = np.zeros((n_days, T), dtype=np.uint8)
    day_cost = np.zeros(n_days)
    day_metrics: list[dict] = []
    plan_events: list[dict] = []
    snapshot_logs: list[dict] = []
    n_fail = 0
    failure_logs: list[dict] = []
    n_audit = 0
    n_extra = 0
    n_adjust = 0
    n_keep = 0
    n_adjust_334 = 0
    n_keep_334 = 0
    n_lp_act = 0
    n_clip = 0
    n_greedy = 0
    n_gamma = 0
    n_b_gt_a = 0
    n_half = 0
    n_full = 0
    n_half_334 = 0
    n_full_334 = 0
    n_cold = 0
    n_v2 = 0
    n_v2_334 = 0
    E = float(E0_JAN1)
    p_hist_max = 0.0
    p_max_att1 = float(np.max(data.price1))
    snap_n = np.full((n_days, 4, T), np.nan)
    snap_next = np.full((n_days, 4, SCREEN_NEXT), np.nan)
    snap_p_next = np.full((n_days, 4, SCREEN_NEXT), np.nan)
    err_n = np.full((n_days, 4, T), np.nan)
    err_n_next = np.full((n_days, 4, SCREEN_NEXT), np.nan)
    err_z = np.full((n_days, 4, T), np.nan)
    err_z_next = np.full((n_days, 4, SCREEN_NEXT), np.nan)
    p_hat_issue = np.full((n_days, 4, T), np.nan)
    engine = ForecastEngine(data, cfg)
    gamma = float(np.clip(cfg.gamma, 0.0, 1.0))

    def record_solver_failure(disp, day, t, stage):
        nonlocal n_fail
        failures = []
        if not disp.ok:
            failures.append(("primary", disp.message))
        if disp.secondary_failure:
            failures.append(("secondary", disp.secondary_failure))
        for level, message in failures:
            n_fail += 1
            failure_logs.append({"day": int(day), "slot": int(t), "stage": stage,
                                 "level": level, "message": message})
            emit(f"  {cfg.name} LP_FAIL day={day + 1} slot={t} stage={stage}/{level}: {message}")

    def window_pv(day: int, t: int, n: int, issue: int, actual, fc_V_base, just_fused: bool):
        V = engine.pv_window(day, t, n, issue, actual, day)
        n_today = min(n, T - t)
        if just_fused:
            V[:n_today] = fc_V_base[t : t + n_today]
        elif t > 0:
            V[:n_today] = blend(fc_V_base, actual, t - 1, cfg.decay)[t : t + n_today]
        else:
            V[:n_today] = fc_V_base[t : t + n_today]
        return np.maximum(V, 0.0)

    def window_load(day: int, t: int, n: int, actual_L, fc_L_base):
        L = engine.load_window(day, t, n, day)
        n_today = min(n, T - t)
        if t > 0:
            L[:n_today] = blend(fc_L_base, actual_L, t - 1, cfg.decay)[t : t + n_today]
        else:
            L[:n_today] = fc_L_base[t : t + n_today]
        return np.maximum(L, 0.0)

    def window_price(day: int, t: int, n: int, p_today):
        P = engine.price_window(day, t, n, day)
        n_today = min(n, T - t)
        P[:n_today] = p_today[t : t + n_today]
        return P

    def pin_first_obs(L, V, t, actual_L, actual_V):
        """Intra-slot feedback approximation: first window slot uses this slot's measurement.
        Declared as such; not used to rewrite already-passed contracts.
        """
        L = L.copy()
        V = V.copy()
        L[0] = float(actual_L[t])
        V[0] = float(actual_V[t])
        return L, V

    def solve_exec(day, t, Ecur, issue, actual_L, actual_V, fc_L_base, fc_V_base, p_today, a_today, just_fused):
        nonlocal n_fail
        n = _win_len(day, t, cfg.exec_horizon)
        if n <= 0:
            return None
        L = window_load(day, t, n, actual_L, fc_L_base)
        V = window_pv(day, t, n, issue, actual_V, fc_V_base, just_fused)
        L, V = pin_first_obs(L, V, t, actual_L, actual_V)
        P = window_price(day, t, n, p_today)
        n_today = min(n, T - t)
        a_fixed = np.full(n, np.nan)
        a_fixed[:n_today] = a_today[t : t + n_today]
        disp = solve_dispatch(
            P,
            to_energy(L),
            to_energy(V),
            Ecur,
            a_fixed=a_fixed,
            h_allowed=True,
            terminal_mu=cfg.terminal_mu,
            eta_c=cfg.eta_c, eta_d=cfg.eta_d,
            secondary_throughput=True,
            first_slot_policy=cfg.first_slot_policy,
            allow_emergency_charge=cfg.allow_emergency_charge,
        )
        record_solver_failure(disp, day, t, "execution")
        if not disp.ok:
            return None
        return disp

    def freeze_snapshot(day, t, hour, decision_origin, pv_issue, actual_L, actual_V, fc_L, fc_V, p_today, just_fused):
        n = _win_len(day, t, cfg.exec_horizon)
        L = window_load(day, t, n, actual_L, fc_L)
        V = window_pv(day, t, n, pv_issue, actual_V, fc_V, just_fused)
        P = window_price(day, t, n, p_today)
        n_today = min(n, T - t)
        load_day = np.full(T, np.nan)
        pv_day = np.full(T, np.nan)
        price_day = np.full(T, np.nan)
        if t > 0:
            load_day[:t] = actual_L[:t]
            pv_day[:t] = actual_V[:t]
            price_day[:t] = p_today[:t]
        load_day[t : t + n_today] = L[:n_today]
        pv_day[t : t + n_today] = V[:n_today]
        price_day[t : t + n_today] = P[:n_today]
        snap_n[day, decision_origin] = to_energy(load_day - pv_day)
        take = min(SCREEN_NEXT, max(0, n - n_today))
        if take > 0:
            snap_next[day, decision_origin, :take] = to_energy(L[n_today : n_today + take] - V[n_today : n_today + take])
            snap_p_next[day, decision_origin, :take] = P[n_today : n_today + take]
        snap = ForecastSnapshot(
            snapshot_id=f"d{day:03d}-h{hour:02d}-o{decision_origin}-pv{pv_issue}",
            day=int(day),
            t0=int(t),
            n=int(n),
            decision_hour=int(hour),
            decision_origin=int(decision_origin),
            pv_issue_used=int(pv_issue),
            last_observed_slot=int(t - 1),
            actual_slot_kw=float(actual_L[t]),
            load_kw=np.asarray(L, dtype=float),
            pv_kw=np.asarray(V, dtype=float),
            price=np.asarray(P, dtype=float),
            load_day=load_day,
            pv_day=pv_day,
            price_day=price_day,
        )
        return snap

    def fill_next_errors(day, actual_L, actual_V, p_set):
        if day <= 0:
            return
        actual_n0 = to_energy(actual_L[:SCREEN_NEXT] - actual_V[:SCREEN_NEXT])
        for iss in range(4):
            if np.all(np.isfinite(snap_next[day - 1, iss])):
                err_n_next[day - 1, iss] = actual_n0 - snap_next[day - 1, iss]
            if np.all(np.isfinite(snap_p_next[day - 1, iss])) and np.all(snap_p_next[day - 1, iss] > 0):
                with np.errstate(divide="ignore", invalid="ignore"):
                    err_z_next[day - 1, iss] = np.log(
                        np.maximum(p_set[:SCREEN_NEXT], 1e-8) / np.maximum(snap_p_next[day - 1, iss], 1e-8)
                    )

    def current_kappa(t, P_window, p_set):
        if cfg.kappa_fixed is not None:
            return float(cfg.kappa_fixed)
        if cfg.price_mode == "att1":
            return float(EMERGENCY_MULT * cfg.eta_d * p_max_att1)
        realized = p_hist_max
        if t > 0:
            realized = max(realized, float(np.max(p_set[:t])))
        fc = float(np.max(P_window)) if len(P_window) else 0.0
        return float(EMERGENCY_MULT * cfg.eta_d * max(realized, fc, 1e-6))

    def v2_adjust_star(snap: ForecastSnapshot, Ecur, q0, a_cur):
        """Quantile-band LP on the frozen snapshot, not an archived a-column."""
        n = snap.n
        t = snap.t0
        L = snap.load_kw
        V = snap.pv_kw
        P = snap.price
        n_today = min(n, T - t)
        q70_p, q90_p = engine.band_quantiles(L - V, snap.day, t, snap.decision_origin, BAND_LO, BAND_HI)
        q70 = to_energy(q70_p)
        q90 = to_energy(q90_p)
        a_lo = np.zeros(n)
        a_hi = np.full(n, np.inf)
        a_ref = np.full(n, np.nan)
        a_ref[:n_today] = q0[t : t + n_today]
        for i in range(n_today):
            hour_i = (t + i) // 6
            if engine.n_samples(snap.decision_origin, hour_i) < cfg.min_err_samples:
                a_lo[i] = a_cur[t + i]
                a_hi[i] = a_cur[t + i]
                continue
            qt = float(q0[t + i])
            lo = float(q70[i])
            hi = float(max(lo, q90[i]))
            if qt < lo - 1e-9:
                a_lo[i] = qt
                a_hi[i] = np.inf
            elif qt > hi + 1e-9:
                a_lo[i] = max(0.0, hi)
                a_hi[i] = qt
            else:
                a_lo[i] = qt
                a_hi[i] = qt
        plan = solve_dispatch(
            P,
            to_energy(L),
            to_energy(V),
            Ecur,
            a_ref=a_ref,
            a_lo=a_lo,
            a_hi=a_hi,
            h_allowed=True,
            terminal_mu=cfg.terminal_mu,
            eta_c=cfg.eta_c, eta_d=cfg.eta_d,
            secondary_throughput=False,
            first_slot_policy=False,
            allow_emergency_charge=False,
        )
        record_solver_failure(plan, snap.day, t, "v2_candidate")
        if not plan.ok:
            return None
        out = a_cur.copy()
        out[t : t + n_today] = np.maximum(plan.a[:n_today], 0.0)
        return out

    def maybe_adjust(snap: ForecastSnapshot, Ecur, q0, a_cur, p_set):
        nonlocal n_fail, n_adjust, n_keep, n_adjust_334, n_keep_334, n_b_gt_a, n_half, n_full, n_half_334, n_full_334, n_cold, n_v2, n_v2_334
        day = snap.day
        t = snap.t0
        hour = snap.decision_hour
        issue = snap.decision_origin
        n = snap.n
        L = snap.load_kw
        V = snap.pv_kw
        P = snap.price
        n_today = min(n, T - t)
        a_fixed = np.full(n, np.nan)
        a_fixed[:n_today] = a_cur[t : t + n_today]
        a_ref = np.full(n, np.nan)
        a_ref[:n_today] = q0[t : t + n_today]
        le = to_energy(L)
        ve = to_energy(V)
        common = dict(
            h_allowed=True,
            terminal_mu=cfg.terminal_mu,
            eta_c=cfg.eta_c, eta_d=cfg.eta_d,
            secondary_throughput=True,
            first_slot_policy=cfg.first_slot_policy,
            allow_emergency_charge=cfg.allow_emergency_charge,
        )
        plan_a = solve_dispatch(P, le, ve, Ecur, a_fixed=a_fixed, a_ref=a_ref, **common)
        record_solver_failure(plan_a, day, t, "keep_contract")
        cap = None
        if plan_a.ok:
            cap = float(plan_a.objective_primary) + 1e-4
        plan_b = solve_dispatch(P, le, ve, Ecur, a_ref=a_ref, primary_cap=cap, **common)
        record_solver_failure(plan_b, day, t, "adjust_contract")
        raw_a = None if not plan_a.ok else float(plan_a.objective_primary)
        raw_b = None if not plan_b.ok else float(plan_b.objective_primary)
        raw_gap = None if raw_a is None or raw_b is None else float(raw_a - raw_b)
        fallback = None
        if not plan_b.ok:
            if plan_a.ok:
                plan_b = plan_a
                fallback = "B_fail_use_A"
        new_a = a_cur.copy()
        decision = "keep"
        save = float("nan")
        da = 0.0
        lam_ch = 0.0
        screen_rec: dict = {}
        mu_mean = None
        if plan_b.ok and plan_b.eq_dual is not None and len(plan_b.eq_dual) >= n_today:
            mu_mean = float(np.mean(plan_b.eq_dual[:n_today]))
        if plan_a.ok and plan_b.ok:
            save = float(plan_a.objective_primary - plan_b.objective_primary)
            if save < -1e-4:
                n_b_gt_a += 1
                fallback = "B_gt_A"
                save = 0.0
                plan_b = plan_a
            da = float(np.abs(plan_b.a[:n_today] - a_cur[t : t + n_today]).sum())
            a_star = a_cur.copy()
            a_star[t : t + n_today] = np.maximum(plan_b.a[:n_today], 0.0)
            use_b = save >= cfg.min_adjust_save and da >= cfg.min_adjust_kwh
            if cfg.screen:
                n_hat = to_energy(L - V)
                p_day = np.zeros(T)
                p_day[t : t + n_today] = P[:n_today]
                net_day = np.zeros(T)
                net_day[t:] = n_hat[:n_today]
                take = min(SCREEN_NEXT, max(0, n - n_today))
                if take > 0 and plan_a.ok and len(plan_a.a) >= n_today + take:
                    q_ref = np.maximum(plan_a.a[n_today : n_today + take], 0.0)
                else:
                    q_ref = np.zeros(0)
                if take > 0:
                    n_next = n_hat[n_today : n_today + take].copy()
                    p_next = P[n_today : n_today + take].copy()
                else:
                    n_next = np.zeros(0)
                    p_next = np.zeros(0)
                paths = collect_paths(
                    err_n, err_n_next,
                    err_z if cfg.price_mode != "att1" else None,
                    err_z_next if cfg.price_mode != "att1" else None,
                    day, issue, t, cfg.n_screen_paths,
                    n_next_need=take,
                )
                kap = current_kappa(t, P, p_set)
                extra = None
                if cfg.v2_candidate:
                    a_v2 = v2_adjust_star(snap, Ecur, q0, a_cur)
                    if a_v2 is not None:
                        extra = {LAMBDA_V2: a_v2}
                if len(paths) < cfg.min_screen_paths:
                    use_b = False
                    lam_ch = 0.0
                    n_cold += 1
                    screen_rec = {
                        "reason": "cold_keep",
                        "n_paths": len(paths),
                        "n_next": int(take),
                        "kappa": float(kap),
                        "q_ref": [float(x) for x in q_ref],
                        "q_ref_sum": float(np.sum(q_ref)) if q_ref.size else 0.0,
                    }
                else:
                    lam_ch, screen_rec = screen_lambda(
                        a_cur, a_star, q0, net_day, n_next, p_day, p_next, q_ref,
                        Ecur, t, paths,
                        eps_c=cfg.eps_c, r=cfg.r_budget, kappa=kap,
                        energy_gate=cfg.energy_gate, eps_e=cfg.eps_e,
                        extra_contracts=extra,
                        worst_case=cfg.screen_worst_case,
                        eta_c=cfg.eta_c, eta_d=cfg.eta_d,
                    )
                    use_b = lam_ch > 0
                    if use_b:
                        if abs(lam_ch - LAMBDA_V2) < 1e-9 and extra is not None:
                            new_a = extra[LAMBDA_V2]
                        else:
                            new_a = mix_contract(a_cur, a_star, lam_ch, t)
            elif use_b:
                new_a[t : t + n_today] = a_star[t : t + n_today]
                lam_ch = 1.0
            if use_b:
                if abs(lam_ch - LAMBDA_V2) < 1e-9:
                    decision = "v2"
                    n_v2 += 1
                    if day >= EXPORT_START:
                        n_v2_334 += 1
                elif abs(lam_ch - 0.5) < 1e-9:
                    decision = "half"
                    n_half += 1
                    if day >= EXPORT_START:
                        n_half_334 += 1
                else:
                    decision = "adjust"
                    n_full += 1
                    if day >= EXPORT_START:
                        n_full_334 += 1
                n_adjust += 1
                if day >= EXPORT_START:
                    n_adjust_334 += 1
            else:
                n_keep += 1
                if day >= EXPORT_START:
                    n_keep_334 += 1
        else:
            n_keep += 1
            if day >= EXPORT_START:
                n_keep_334 += 1
        reason = (
            f"{decision} lam={lam_ch:g} save={save:.2f} da={da:.1f} "
            f"A={'ok' if plan_a.ok else plan_a.message} B={'ok' if plan_b.ok else plan_b.message} "
            f"fb={fallback} scr={screen_rec.get('reason')}"
        )
        event = {
            "day": int(day),
            "hour": int(hour),
            "issue": int(issue),
            "decision_origin": int(snap.decision_origin),
            "pv_issue_used": int(snap.pv_issue_used),
            "snapshot_id": snap.snapshot_id,
            "decision": decision,
            "reason": reason,
            "lambda": lam_ch,
            "raw_obj_A": raw_a,
            "raw_obj_B": raw_b,
            "raw_gap": raw_gap,
            "fallback": fallback,
            "cost_A": None if not plan_a.ok else float(plan_a.cost),
            "cost_B": None if not plan_b.ok else float(plan_b.cost),
            "obj_A": raw_a,
            "obj_B": raw_b,
            "save": None if not np.isfinite(save) else save,
            "da_kwh": da,
            "mu_mean": mu_mean,
            "screen": screen_rec,
            "q_remain": float(q0[t:].sum()),
            "a_before": float(a_cur[t:].sum()),
            "a_after": float(new_a[t:].sum()),
            "E": float(Ecur),
            "obs_until": int(t - 1),
            "issue_time": int(hour),
            "last_observed_slot": int(t - 1),
            "fc_L0": float(L[0]),
            "actual_L0": float(snap.actual_slot_kw),
            "used_current_actual": abs(float(L[0]) - float(snap.actual_slot_kw)) < 1e-9,
            "F_A": None if not plan_a.ok else float(plan_a.objective_primary),
            "F_B": None if not plan_b.ok else float(plan_b.objective_primary),
            "F_A_final": None if not plan_a.ok else float(plan_a.primary_final),
            "F_B_final": None if not plan_b.ok else float(plan_b.primary_final),
            "primal_A": None if not plan_a.ok else float(plan_a.primal_residual),
            "primal_B": None if not plan_b.ok else float(plan_b.primal_residual),
        }
        return new_a, event

    for day in range(n_days):
        E_start[day] = E
        E_slot[day, 0] = E
        p_today = engine.price_fc(day, day)
        p_set = _settle_prices(data, day, cfg.price_mode)
        p_plan_a[day] = p_today
        p_set_a[day] = p_set
        actual_L = data.load[day]
        actual_V = data.pv[day]

        fc_L_base = engine.load_fc(day, day)
        if cfg.use_att3 and 0 in cfg.att3_issues:
            fc_V_base = engine.pv_fused_today(day, 0, None, day)
            current_issue = 0
        else:
            fc_V_base = engine.pv_hist_fc(day, day)
            current_issue = 0
        fc_L0[day] = fc_L_base
        fc_V0[day] = fc_V_base
        p_hat_issue[day, 0] = p_today.copy()
        day_snaps: list[ForecastSnapshot] = []
        snap0 = freeze_snapshot(day, 0, 0, 0, 0, actual_L, actual_V, fc_L_base, fc_V_base, p_today, True)
        day_snaps.append(snap0)
        issue_pv_log: dict[int, dict[str, np.ndarray]] = {}
        if cfg.use_att3 and 0 in cfg.att3_issues:
            issued0 = engine.issued_two_days(day, 0, None)[0]
            issue_pv_log[0] = {"issued": issued0.copy(), "fused": fc_V_base.copy()}

        n_plan = T if day >= N_DAYS - 1 else min(cfg.plan_horizon, T * 2)
        L_win = engine.load_window(day, 0, n_plan, day)
        V_win = engine.pv_window(day, 0, n_plan, current_issue, None, day)
        V_win[:T] = fc_V_base
        L_win[:T] = fc_L_base
        P_win = engine.price_window(day, 0, n_plan, day)
        N_hat = L_win - V_win
        N_til = (
            engine.protect_net(N_hat, day, 0, 0, _alpha_arr(n_plan, 0, cfg))
            if cfg.plan_protection else N_hat.copy()
        )
        L_use = V_win + N_til
        plan = solve_dispatch(
            P_win,
            to_energy(L_use),
            to_energy(V_win),
            E,
            h_allowed=False,
            terminal_mu=cfg.terminal_mu,
            eta_c=cfg.eta_c, eta_d=cfg.eta_d,
            secondary_throughput=False,
            first_slot_policy=False,
        )
        record_solver_failure(plan, day, 0, "day_ahead")
        if plan.ok:
            q[day] = np.maximum(plan.a[:T], 0.0)
        else:
            q[day] = np.maximum(to_energy(fc_L_base) - to_energy(fc_V_base), 0.0)
        a[day] = q[day].copy()
        a_snap[day, 0] = a[day]
        plan_events.append(
            {
                "day": int(day),
                "hour": 0,
                "issue": 0,
                "decision": "plan",
                "reason": "day-ahead" if plan.ok else f"day-ahead failure: {plan.message}",
                "cost_A": None if not plan.ok else float(plan.cost),
                "cost_B": None,
                "save": None,
                "da_kwh": 0.0,
                "mu_mean": None,
                "q_remain": float(q[day].sum()),
                "a_before": float(q[day].sum()),
                "a_after": float(a[day].sum()),
                "E": float(E),
                "obs_until": -1,
            }
        )

        disp = None
        disp_t0 = 0
        t = 0
        just_fused = True
        pv_issue_used = 0
        while t < T:
            if t in SLOT_TO_ISSUE and t > 0:
                hour = {36: 6, 72: 12, 108: 18}[t]
                decision_origin = SLOT_TO_ISSUE[t]
                if cfg.price_mode != "att1":
                    p_today = blend(p_today.copy(), p_set, t - 1, cfg.decay)
                p_hat_issue[day, decision_origin] = p_today.copy()
                if cfg.use_att3 and hour in cfg.att3_issues:
                    issued_raw = engine.issued_two_days(day, decision_origin, actual_V)[0]
                    fc_V_base = engine.pv_fused_today(day, decision_origin, actual_V, day)
                    fc_V_base[:t] = actual_V[:t]
                    issue_pv_log[decision_origin] = {"issued": issued_raw.copy(), "fused": fc_V_base.copy()}
                    pv_issue_used = decision_origin
                    current_issue = decision_origin
                    just_fused = True
                else:
                    just_fused = False
                if t == 36:
                    fill_next_errors(day, actual_L, actual_V, p_set)
                fc_L_now = fc_L_base if t == 0 else blend(fc_L_base, actual_L, t - 1, cfg.decay)
                snap = freeze_snapshot(
                    day, t, hour, decision_origin, pv_issue_used,
                    actual_L, actual_V, fc_L_now, fc_V_base, p_today, just_fused,
                )
                day_snaps.append(snap)
                if t > 0:
                    p_today = p_today.copy()
                    p_today[:t] = p_set[:t]
                if cfg.adjust == "selective" and hour in cfg.adjust_hours:
                    a[day], event = maybe_adjust(snap, E, q[day], a[day], p_set)
                    plan_events.append(event)
                    disp = None
                a_snap[day, decision_origin] = a[day].copy()

            if cfg.economic_exec and (disp is None or (t % cfg.control_step == 0)):
                disp = solve_exec(
                    day, t, E, current_issue, actual_L, actual_V,
                    fc_L_base, fc_V_base, p_today, a[day], just_fused,
                )
                disp_t0 = t
                just_fused = False
            elif not cfg.economic_exec:
                just_fused = False

            le = float(actual_L[t] * (1.0 / 6.0))
            ve = float(actual_V[t] * (1.0 / 6.0))
            ae = float(a[day, t])
            cg, dg, hg, wg, _ = greedy_slot(ae, le, ve, E, eta_c=cfg.eta_c, eta_d=cfg.eta_d)
            k = t - disp_t0
            reason_code = CLIP_NONE
            src = "greedy"
            if cfg.economic_exec and disp is not None and 0 <= k < len(disp.c):
                c_m = float(max(0.0, disp.c[k]))
                d_m = float(max(0.0, disp.d[k]))
                c_opt[day, t] = c_m
                d_opt[day, t] = d_m
                if abs(gamma - 1.0) <= 1e-12:
                    c_use, d_use = c_m, d_m
                    src = "lp"
                elif abs(gamma) <= 1e-12:
                    c_use, d_use = cg, dg
                    src = "greedy"
                    reason_code = CLIP_GREEDY
                else:
                    c_use, d_use = mix_signed(cg, dg, c_m, d_m, gamma)
                    src = "gamma"
                    reason_code = CLIP_GAMMA
                    n_gamma += 1
            else:
                c_opt[day, t] = 0.0
                d_opt[day, t] = 0.0
                c_use, d_use = cg, dg
                src = "greedy"
                reason_code = CLIP_GREEDY

            cs, ds, hs, ws, E, rsn = realize_slot(
                ae, le, ve, E, c_use, d_use,
                absorb_free=True,
                eta_c=cfg.eta_c, eta_d=cfg.eta_d,
                allow_emergency_charge=cfg.allow_emergency_charge,
            )
            if rsn == "free_surplus" and src == "lp":
                reason_code = CLIP_PHYSICS
                src = "clip"
            elif src == "lp":
                if abs(cs - c_use) > 1e-6 or abs(ds - d_use) > 1e-6:
                    if abs(cs - c_use) < 1e-3 and abs(ds - d_use) < 1e-3:
                        reason_code = CLIP_FLOAT
                    else:
                        reason_code = CLIP_PHYSICS
                        src = "clip"

            if src == "lp":
                n_lp_act += 1
            elif src == "clip":
                n_clip += 1
            elif src == "gamma":
                pass
            else:
                n_greedy += 1
            src_a[day, t] = {"lp": 0, "clip": 1, "greedy": 2, "gamma": 3}[src]
            clip_r[day, t] = reason_code
            c[day, t], d[day, t], h[day, t], w[day, t] = cs, ds, hs, ws
            E_slot[day, t + 1] = E
            t += 1

        E_end[day] = E
        lock_until = 36 if cfg.adjust == "selective" else T
        _, errs = audit_day(
            E_start[day],
            a[day],
            c[day],
            d[day],
            h[day],
            w[day],
            to_energy(actual_L),
            to_energy(actual_V),
            q=q[day],
            lock_until=lock_until,
            eta_c=cfg.eta_c, eta_d=cfg.eta_d,
        )
        n_audit += len(errs)
        met = cash(p_set, q[day], a[day], h[day])
        met["waste"] = float(w[day].sum())
        paid_w, pv_w = split_waste(a[day], w[day])
        met["waste_paid"] = paid_w
        met["waste_pv"] = pv_w
        met["charge"] = float(c[day].sum())
        met["discharge"] = float(d[day].sum())
        met["E0"] = float(E_start[day])
        met["E24"] = float(E_end[day])
        met["E16"] = float(E_slot[day, 96])
        met["audit_n"] = len(errs)
        day_cost[day] = met["cost"]
        day_metrics.append(met)
        actual_n = to_energy(actual_L - actual_V)
        for iss in range(4):
            if np.any(np.isfinite(snap_n[day, iss])):
                err_n[day, iss] = actual_n - snap_n[day, iss]
        if cfg.price_mode != "att1":
            with np.errstate(divide="ignore", invalid="ignore"):
                for iss in range(4):
                    hat = p_hat_issue[day, iss]
                    if np.any(np.isfinite(hat)):
                        err_z[day, iss] = np.log(np.maximum(p_set, 1e-8) / np.maximum(hat, 1e-8))
        p_hist_max = max(p_hist_max, float(np.max(p_set)))
        ingested = engine.observe_day(day, fc_L0[day], fc_V0[day], issue_pv_log, snapshots=day_snaps)
        for snap in day_snaps:
            rec = snap.as_log()
            rec["residual_kw"] = ingested.get(snap.snapshot_id)
            snapshot_logs.append(rec)
        if (day + 1) % 30 == 0 or day == 0 or day == n_days - 1:
            emit(
                f"  {cfg.name} day {day + 1:3d}/{n_days}  cost={met['cost']:.1f}  "
                f"h={met['energy_h']:.1f}  E={E:.1f}  fail={n_fail}  audit={n_audit}  "
                f"adj={n_adjust}/{n_keep} half={n_half} full={n_full}  src=g{n_greedy}/lp{n_lp_act}"
            )

    return YearRun(
        q=q, a=a, c=c, d=d, h=h, w=w,
        E_start=E_start, E_end=E_end, E_slot=E_slot,
        p_plan=p_plan_a, p_set=p_set_a, fc_L0=fc_L0, fc_V0=fc_V0, a_snap=a_snap,
        c_opt=c_opt, d_opt=d_opt, src=src_a, clip_reason=clip_r,
        n_lp_fail=n_fail, n_audit=n_audit, n_extra_reopt=n_extra,
        n_adjust=n_adjust, n_keep=n_keep,
        n_adjust_334=n_adjust_334, n_keep_334=n_keep_334,
        n_lp_act=n_lp_act, n_clip=n_clip, n_greedy=n_greedy, n_gamma=n_gamma,
        n_b_gt_a=n_b_gt_a, n_half=n_half, n_full=n_full,
        n_half_334=n_half_334, n_full_334=n_full_334, n_cold_keep=n_cold,
        n_v2=n_v2, n_v2_334=n_v2_334, gamma=gamma,
        day_cost=day_cost, day_metrics=day_metrics, plan_events=plan_events,
        snapshots=snapshot_logs, cfg_name=cfg.name,
        failure_logs=failure_logs,
    )
