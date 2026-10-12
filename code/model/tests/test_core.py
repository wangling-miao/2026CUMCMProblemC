"""Unit checks for v7 settlement, Q1, continuation screening, terminal κ pricing."""

from __future__ import annotations

import numpy as np

from cumcm_v11.constants import E_MAX, KAPPA_Q3, Q1_BASELINE, Q1_COST, SCREEN_NEXT, T
from cumcm_v11.data import load_all, slot_interval, time_map_rows, to_energy
from cumcm_v11.optimize import greedy_slot
from cumcm_v11.q1 import run_q1
from cumcm_v11.screen import greedy_to_horizon, kappa_from_prices, mix_contract, net_gain, screen_lambda
from cumcm_v11.settle import cash


def test_settlement_examples() -> None:
    p = np.array([1.0])
    q = np.array([100.0])
    assert abs(cash(p, q, np.array([100.0]), np.array([0.0]))["cost"] - 100.0) < 1e-9
    assert abs(cash(p, q, np.array([80.0]), np.array([0.0]))["cost"] - 90.0) < 1e-9
    assert abs(cash(p, q, np.array([120.0]), np.array([0.0]))["cost"] - 130.0) < 1e-9


def test_mix_contract_locks_past() -> None:
    a_old = np.ones(T) * 10.0
    a_star = np.ones(T) * 4.0
    out = mix_contract(a_old, a_star, 1.0, 36)
    assert np.allclose(out[:36], 10.0)
    assert np.allclose(out[36:], 4.0)
    half = mix_contract(a_old, a_star, 0.5, 36)
    assert np.allclose(half[36:], 7.0)


def test_continuation_uses_shared_ref() -> None:
    a = np.ones(T) * 50.0
    q = a.copy()
    q_ref = np.ones(SCREEN_NEXT) * 40.0
    price = np.ones(T) * 1.0
    price_n = np.ones(SCREEN_NEXT) * 1.0
    net = np.ones(T) * 10.0
    net_n = np.ones(SCREEN_NEXT) * 10.0
    c1, ct1, cn1, et, eh = greedy_to_horizon(a, q, q_ref, price, price_n, net, net_n, 6000.0, 120)
    c2, ct2, cn2, et2, eh2 = greedy_to_horizon(a, q, q_ref, price, price_n, net, net_n, 6000.0, 120)
    assert abs(c1 - c2) < 1e-9
    assert abs(eh - eh2) < 1e-9
    assert eh >= 1200.0
    assert abs(c1 - ct1 - cn1) < 1e-9
    # next-day cost is p*q_ref + 5p*h, no |a-q| vs yesterday's q
    assert cn1 >= 0.0
    empty = np.zeros(0)
    c3, ct3, cn3, et3, eh3 = greedy_to_horizon(a, q, empty, price, empty, net, empty, 6000.0, 120)
    assert abs(cn3) < 1e-12
    assert abs(eh3 - et3) < 1e-9


def test_screen_keeps_when_down_causes_emergency() -> None:
    t0 = 120
    a_old = np.ones(T) * 200.0
    a_star = a_old.copy()
    a_star[t0:] = 0.0
    q = a_old.copy()
    n_hat = np.ones(T) * 50.0
    n_next = np.ones(SCREEN_NEXT) * 50.0
    p = np.ones(T) * 1.0
    p_next = np.ones(SCREEN_NEXT) * 1.0
    q_ref = np.ones(SCREEN_NEXT) * 50.0
    paths = [
        {
            "src_day": i,
            "eps": np.ones(T - t0) * 180.0,
            "eps_next": np.ones(SCREEN_NEXT) * 0.0,
            "z": None,
            "z_next": None,
        }
        for i in range(8)
    ]
    lam, rec = screen_lambda(
        a_old, a_star, q, n_hat, n_next, p, p_next, q_ref, 6000.0, t0, paths,
        eps_c=5.0, r=5.0, kappa=KAPPA_Q3,
    )
    assert lam == 0.0


def test_next_day_has_no_adjust_penalty() -> None:
    a = np.ones(T) * 80.0
    q = np.ones(T) * 200.0
    q_ref = np.ones(SCREEN_NEXT) * 10.0
    price = np.ones(T) * 1.0
    price_n = np.ones(SCREEN_NEXT) * 2.0
    net = np.ones(T) * 80.0
    net_n = np.zeros(SCREEN_NEXT)
    _tot, _ct, cn, _et, _eh = greedy_to_horizon(a, q, q_ref, price, price_n, net, net_n, 6000.0, 140)
    expected = float(np.sum(price_n * q_ref))
    assert abs(cn - expected) < 1e-6


def test_v5_passer_still_passes_with_continuation() -> None:
    t0 = 120
    a_old = np.zeros(T)
    a_star = np.ones(T) * 200.0
    q = a_old.copy()
    n_hat = np.ones(T) * 200.0
    n_next = np.ones(SCREEN_NEXT) * 50.0
    p = np.ones(T) * 1.0
    p_next = np.ones(SCREEN_NEXT) * 1.0
    q_ref = np.ones(SCREEN_NEXT) * 50.0
    paths = [
        {
            "src_day": i,
            "eps": np.zeros(T - t0),
            "eps_next": np.zeros(SCREEN_NEXT),
            "z": None,
            "z_next": None,
        }
        for i in range(8)
    ]
    lam, rec = screen_lambda(
        a_old, a_star, q, n_hat, n_next, p, p_next, q_ref, 1200.0, t0, paths,
        eps_c=5.0, r=5.0, kappa=KAPPA_Q3,
    )
    assert lam == 1.0
    row = rec["scores"][1.0]["paths"][0]
    assert row["E_T"] + 1e-6 >= row["E_T0"] - 100.0
    assert rec["scores"][1.0]["ends_ok"]


def test_midnight_short_soc_can_recover_by_morning() -> None:
    t0 = 140
    a_old = np.ones(T) * 80.0
    a_star = a_old.copy()
    a_star[t0:] = 0.0
    q = a_old.copy()
    n_hat = np.ones(T) * 80.0
    n_next = np.zeros(SCREEN_NEXT)
    p = np.ones(T) * 1.0
    p_next = np.ones(SCREEN_NEXT) * 1.0
    q_ref = np.ones(SCREEN_NEXT) * 200.0
    paths = [
        {
            "src_day": i,
            "eps": np.zeros(T - t0),
            "eps_next": np.zeros(SCREEN_NEXT),
            "z": None,
            "z_next": None,
        }
        for i in range(8)
    ]
    lam, rec = screen_lambda(
        a_old, a_star, q, n_hat, n_next, p, p_next, q_ref, 10700.0, t0, paths,
        eps_c=5.0, r=5.0, kappa=KAPPA_Q3,
    )
    row = rec["scores"][1.0]["paths"][0]
    assert row["E_T0"] - row["E_T"] > 100.0
    assert row["E_H"] + 1e-6 >= row["E_H0"] - 100.0
    assert rec["scores"][1.0]["mean_g"] > 5.0
    assert lam == 1.0
    assert abs(row["E_H"] - E_MAX) < 1e-6
    assert abs(row["E_H0"] - E_MAX) < 1e-6


def test_mean_only_accepts_when_one_path_is_bad() -> None:
    """Official screen: one analog-day loss must not veto a positive mean-gain candidate."""
    t0 = 120
    a_old = np.zeros(T)
    a_star = np.ones(T) * 200.0
    q = a_old.copy()
    n_hat = np.ones(T) * 200.0
    n_next = np.ones(SCREEN_NEXT) * 50.0
    p = np.ones(T) * 1.0
    p_next = np.ones(SCREEN_NEXT) * 1.0
    q_ref = np.ones(SCREEN_NEXT) * 50.0
    paths = [
        {
            "src_day": i,
            "eps": np.zeros(T - t0),
            "eps_next": np.zeros(SCREEN_NEXT),
            "z": None,
            "z_next": None,
        }
        for i in range(7)
    ]
    paths.append(
        {
            "src_day": 7,
            "eps": np.full(T - t0, -200.0),
            "eps_next": np.zeros(SCREEN_NEXT),
            "z": None,
            "z_next": None,
        }
    )
    lam_mean, rec_mean = screen_lambda(
        a_old, a_star, q, n_hat, n_next, p, p_next, q_ref, 1200.0, t0, paths,
        eps_c=5.0, r=5.0, kappa=KAPPA_Q3, worst_case=False,
    )
    lam_min, rec_min = screen_lambda(
        a_old, a_star, q, n_hat, n_next, p, p_next, q_ref, 1200.0, t0, paths,
        eps_c=5.0, r=5.0, kappa=KAPPA_Q3, worst_case=True,
    )
    s = rec_mean["scores"][1.0]
    assert s["mean_g_net"] > 5.0
    assert s["min_g_net"] < -5.0
    assert lam_mean == 1.0
    assert lam_min == 0.0
    assert rec_min["worst_case"] is True


def test_risk_budget_r_not_double_counted() -> None:
    t0 = 140
    a_old = np.ones(T) * 10.0
    a_star = a_old.copy()
    q = a_old.copy()
    n_hat = np.ones(T) * 10.0
    n_next = np.zeros(SCREEN_NEXT)
    p = np.ones(T)
    p_next = np.ones(SCREEN_NEXT)
    q_ref = np.zeros(SCREEN_NEXT)
    paths = [
        {
            "src_day": i,
            "eps": np.zeros(T - t0),
            "eps_next": np.zeros(SCREEN_NEXT),
            "z": None,
            "z_next": None,
        }
        for i in range(8)
    ]
    lam, rec = screen_lambda(
        a_old, a_star, q, n_hat, n_next, p, p_next, q_ref, 6000.0, t0, paths,
        eps_c=5.0, r=5.0, kappa=KAPPA_Q3,
    )
    assert lam == 0.0
    assert rec["r"] == 5.0
    assert rec["eps_c"] == 5.0


def test_kappa_q3_and_report_examples() -> None:
    assert abs(KAPPA_Q3 - 6.2784) < 1e-12
    # 3月11日12时完整候选：高收益、末端亏约106 kWh，计价后仍应通过
    g_mean, g_min, d = 8666.92, 3892.09, 105.879806
    assert net_gain(g_mean, d, 0.0, KAPPA_Q3) > 5
    assert net_gain(g_min, d, 0.0, KAPPA_Q3) >= -5
    # 6月27日12时半幅：收益约249、亏约75 kWh，扣价后应变保持
    g2, d2 = 248.626767, 74.552587
    assert net_gain(g2, d2, 0.0, KAPPA_Q3) < -5
    # 多留电不给正奖励
    assert abs(net_gain(10.0, 6000.0, 6500.0, KAPPA_Q3) - 10.0) < 1e-9


def test_q4_kappa_is_causal() -> None:
    year_max = 1.7936
    observed_or_fc = 1.0
    k_causal = kappa_from_prices(observed_or_fc)
    k_leak = kappa_from_prices(year_max)
    assert abs(k_causal - 5.0 * 0.9 * 1.0) < 1e-12
    assert k_causal < k_leak


def test_greedy_absorbs_surplus() -> None:
    c, d, h, w, e2 = greedy_slot(100.0, 20.0, 0.0, 6000.0)
    assert d == 0.0 and h == 0.0
    assert c + w == 80.0


def test_time_map() -> None:
    data = load_all()
    rows = time_map_rows(data)
    assert len(rows) == T
    start, end = slot_interval(143)
    assert start == "23:50" and end == "24:00"


def test_q2_two_days_greedy() -> None:
    from dataclasses import replace
    from cumcm_v11.config import official_jobs
    from cumcm_v11.simulate import simulate_year

    data = load_all()
    cfg = replace(official_jobs()[0], economic_exec=False)
    run = simulate_year(data, cfg, max_days=2, log=lambda *a, **k: None)
    assert run.n_audit == 0
    assert np.allclose(run.a, run.q)
    assert run.n_greedy == 2 * T
    assert abs(run.E_end[0] - run.E_start[1]) < 1e-8


def test_q3_two_days_lock() -> None:
    from dataclasses import replace
    from cumcm_v11.config import official_jobs
    from cumcm_v11.simulate import simulate_year

    data = load_all()
    cfg = replace([c for c in official_jobs() if c.name == "q3"][0], screen=True)
    run = simulate_year(data, cfg, max_days=2, log=lambda *a, **k: None)
    assert run.n_audit == 0
    assert np.allclose(run.a[:, :36], run.q[:, :36])
    assert run.n_cold_keep >= 0


def test_blend_bias_not_zeroed() -> None:
    from cumcm_v11.forecast import blend

    fc = np.ones(8)
    actual = np.ones(8)
    actual[2] = 3.0
    out = blend(fc, actual, 2, 0.5)
    assert abs(out[2] - 3.0) < 1e-12
    assert abs(out[3] - (1.0 + 2.0 * 0.5)) < 1e-12


def test_contract_does_not_pin_current_slot() -> None:
    from dataclasses import replace
    from cumcm_v11.config import official_jobs
    from cumcm_v11.simulate import simulate_year

    data = load_all()
    cfg = replace([c for c in official_jobs() if c.name == "q3"][0], screen=True)
    run = simulate_year(data, cfg, max_days=2, log=lambda *a, **k: None)
    adj = [e for e in run.plan_events if e.get("hour") in (6, 12, 18)]
    assert adj
    mismatched = [e for e in adj if abs(e["fc_L0"] - e["actual_L0"]) > 1e-6]
    assert mismatched, "forecast first slot should typically differ from unfinished actual"
    assert all(not e.get("used_current_actual") for e in mismatched)
    assert all(e.get("last_observed_slot") == e["hour"] * 6 - 1 or e.get("obs_until") == e["hour"] * 6 - 1 for e in adj)


def test_secondary_keeps_full_primary() -> None:
    from cumcm_v11.constants import EPS_LP, TERMINAL_MU
    from cumcm_v11.optimize import solve_dispatch

    n = 8
    price = np.ones(n)
    load_e = np.ones(n) * 20.0
    pv_e = np.zeros(n)
    d1 = solve_dispatch(price, load_e, pv_e, 6000.0, h_allowed=True, terminal_mu=TERMINAL_MU, secondary_throughput=False)
    d2 = solve_dispatch(price, load_e, pv_e, 6000.0, h_allowed=True, terminal_mu=TERMINAL_MU, secondary_throughput=True)
    assert d1.ok and d2.ok
    assert d2.primary_final <= d1.objective_primary + EPS_LP + 1e-6


def test_night_net_error_is_recorded() -> None:
    from cumcm_v11.config import official_jobs
    from cumcm_v11.forecast import ForecastEngine, ForecastSnapshot

    data = load_all()
    cfg = [c for c in official_jobs() if c.name == "q3"][0]
    eng = ForecastEngine(data, cfg)
    issued = eng.issued_two_days(0, 1, data.pv[0])[0]
    fused = eng.pv_fused_today(0, 1, data.pv[0], 0)
    load = eng.load_fc(0, 0)
    snap = ForecastSnapshot(
        snapshot_id="t",
        day=0,
        t0=36,
        n=T - 36,
        decision_hour=6,
        decision_origin=1,
        pv_issue_used=1,
        last_observed_slot=35,
        actual_slot_kw=float(data.load[0, 36]),
        load_kw=load[36:],
        pv_kw=fused[36:],
        price=np.ones(T - 36),
        load_day=load,
        pv_day=fused,
        price_day=np.ones(T),
    )
    eng.observe_day(0, load, fused, {1: {"issued": issued, "fused": fused}}, snapshots=[snap])
    assert eng.n_samples(1, 20) >= 1


def test_e18_keeps_decision_and_pv_issue() -> None:
    from dataclasses import replace
    from cumcm_v11.config import q3_ablations
    from cumcm_v11.simulate import simulate_year

    data = load_all()
    cfg = replace([c for c in q3_ablations() if c.name == "q3-E"][0])
    run = simulate_year(data, cfg, max_days=3, log=lambda *a, **k: None)
    ev18 = [e for e in run.plan_events if e.get("hour") == 18]
    assert ev18
    for e in ev18:
        assert e["decision_origin"] == 3
        assert e["pv_issue_used"] == 2


def test_snapshot_residual_matches_err_n() -> None:
    from dataclasses import replace
    from cumcm_v11.config import official_jobs
    from cumcm_v11.data import to_energy
    from cumcm_v11.simulate import simulate_year

    data = load_all()
    cfg = replace([c for c in official_jobs() if c.name == "q3"][0])
    run = simulate_year(data, cfg, max_days=5, log=lambda *a, **k: None)
    snaps = [s for s in run.snapshots if s["decision_origin"] == 1 and s["day"] == 2]
    assert snaps
    s = snaps[0]
    t0 = s["t0"]
    assert s["last_observed_slot"] == t0 - 1
    pred0 = float(s["net_kwh"][0])
    win0 = float(to_energy(np.array([s["load_kw"][0] - s["pv_kw"][0]]))[0])
    assert abs(pred0 - win0) < 1e-9
    actual0 = float(to_energy(data.load[2, t0 : t0 + 1] - data.pv[2, t0 : t0 + 1])[0])
    residual = actual0 - pred0
    assert np.isfinite(residual)


def test_plan_buffer_reads_last_five_days() -> None:
    from cumcm_v11.forecast import _flatten_blocks, _quantile

    blocks = [np.full(6, float(i)) for i in range(10)]
    q_all = _quantile(_flatten_blocks(blocks).tolist(), 0.8, 1)
    q_tail = _quantile(_flatten_blocks(blocks[-5:]).tolist(), 0.8, 1)
    assert q_tail > q_all
    flat = _flatten_blocks(blocks).tolist()
    q_obs28 = _quantile(flat[-28:], 0.8, 1)
    assert q_obs28 >= q_all - 1e-9


def test_plan_buffer_modes_and_eight_day_gate() -> None:
    from dataclasses import replace
    from cumcm_v11.config import official_jobs
    from cumcm_v11.forecast import ForecastEngine

    data = load_all()
    cfg = replace([c for c in official_jobs() if c.name == "q2"][0])
    eng = ForecastEngine(data, cfg)
    for i in range(7):
        eng.net_err[0][10].append(np.full(6, float(i + 1)))
    assert eng.plan_buffer_quantile(0, 10, 0.8) == 0.0
    eng.net_err[0][10].append(np.full(6, 20.0))
    q_obs = eng.plan_buffer_quantile(0, 10, 0.8)
    assert q_obs > 0.0
    eng.cfg = replace(cfg, plan_buffer_mode="days", plan_buffer_days=5)
    q_days = eng.plan_buffer_quantile(0, 10, 0.8)
    assert q_days > 0.0
    eng.cfg = replace(cfg, plan_buffer_mode="archive")
    q_arch = eng.plan_buffer_quantile(0, 10, 0.8)
    assert q_arch > 0.0


def test_ingested_residual_matches_snapshot() -> None:
    from dataclasses import replace
    from cumcm_v11.config import official_jobs
    from cumcm_v11.simulate import simulate_year

    data = load_all()
    cfg = replace([c for c in official_jobs() if c.name == "q3"][0])
    run = simulate_year(data, cfg, max_days=4, log=lambda *a, **k: None)
    snaps = [s for s in run.snapshots if s["decision_origin"] == 1 and s["day"] == 2 and s.get("residual_kw")]
    assert snaps
    s = snaps[0]
    t0 = int(s["t0"])
    pred = float(s["load_kw"][0] - s["pv_kw"][0])
    actual = float(data.load[2, t0] - data.pv[2, t0])
    stored = s["residual_kw"][t0]
    assert stored is not None
    assert abs(float(stored) - (actual - pred)) < 1e-8


def test_q1_regression() -> None:
    data = load_all()
    rec = run_q1(data)
    assert abs(rec["cost"] - Q1_COST) < 1e-4
    assert abs(rec["baseline"] - Q1_BASELINE) < 1e-4
    assert rec["h_sum"] < 1e-6
    assert not rec["audit"]
    le = to_energy(data.load1)
    ve = to_energy(data.pv1)
    assert abs(float(np.dot(data.price1, np.maximum(le - ve, 0))) - Q1_BASELINE) < 1e-4


if __name__ == "__main__":
    test_settlement_examples()
    print("settlement ok")
    test_mix_contract_locks_past()
    print("mix ok")
    test_continuation_uses_shared_ref()
    print("continuation ok")
    test_next_day_has_no_adjust_penalty()
    print("next-day settle ok")
    test_v5_passer_still_passes_with_continuation()
    print("v5 passer ok")
    test_midnight_short_soc_can_recover_by_morning()
    print("recover-by-morning ok")
    test_risk_budget_r_not_double_counted()
    print("r budget ok")
    test_kappa_q3_and_report_examples()
    print("kappa examples ok")
    test_q4_kappa_is_causal()
    print("q4 kappa causal ok")
    test_screen_keeps_when_down_causes_emergency()
    print("screen keep ok")
    test_greedy_absorbs_surplus()
    print("greedy ok")
    test_time_map()
    print("time map ok")
    test_q2_two_days_greedy()
    print("q2 two days ok")
    test_q3_two_days_lock()
    print("q3 two days ok")
    test_blend_bias_not_zeroed()
    print("blend bias ok")
    test_contract_does_not_pin_current_slot()
    print("no pin ok")
    test_secondary_keeps_full_primary()
    print("nested F ok")
    test_night_net_error_is_recorded()
    print("night net err ok")
    test_e18_keeps_decision_and_pv_issue()
    print("E18 labels ok")
    test_snapshot_residual_matches_err_n()
    print("snapshot residual ok")
    test_plan_buffer_reads_last_five_days()
    print("plan buffer tail ok")
    test_plan_buffer_modes_and_eight_day_gate()
    print("plan buffer modes ok")
    test_ingested_residual_matches_snapshot()
    print("ingested residual ok")
    test_q1_regression()
    print("q1 regression ok")
