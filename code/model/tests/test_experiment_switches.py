"""Mechanism isolation and physical consistency for the experiment switches."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cumcm_v11.config import official_jobs
from cumcm_v11.constants import E_MIN, E_MAX, SCREEN_NEXT, T
from cumcm_v11.data import load_all
from cumcm_v11.forecast import ForecastEngine
from cumcm_v11.optimize import (
    _eliminate_cycle, free_surplus_charge, greedy_slot, realize_slot, solve_dispatch,
)
from cumcm_v11.screen import greedy_to_horizon, kappa_from_prices, screen_lambda
from cumcm_v11.settle import audit_day
from cumcm_v11.simulate import _alpha_arr, simulate_year


@pytest.fixture(scope="module")
def data():
    return load_all()


def test_nondefault_efficiencies_agree_in_plan_execution_and_audit():
    physics = dict(eta_c=0.8, eta_d=0.85)
    a = np.array([200.0, 0.0])
    load = np.array([100.0, 50.0])
    pv = np.zeros(2)
    plan = solve_dispatch(np.ones(2), load, pv, E_MIN, a_fixed=a,
                          h_allowed=False, terminal_mu=1.0, **physics)
    assert plan.ok and not plan.secondary_failure
    assert plan.c[0] == pytest.approx(100.0, abs=2e-6)
    assert plan.d[1] == pytest.approx(50.0, abs=2e-6)
    assert plan.E[-1] == pytest.approx(E_MIN + 80.0 - 50.0 / 0.85, abs=2e-6)
    executed = []
    e = E_MIN
    for t in range(2):
        row = realize_slot(a[t], load[t], pv[t], e, plan.c[t], plan.d[t], **physics)
        executed.append(row[:4])
        e = row[4]
    c, d, h, w = np.asarray(executed).T
    audit_e, errors = audit_day(E_MIN, a, c, d, h, w, load, pv, **physics)
    assert not errors
    np.testing.assert_allclose(audit_e, plan.E, atol=2e-6)
    assert e == pytest.approx(audit_e[-1])
    assert greedy_slot(0.0, 100.0, 0.0, E_MIN + 100.0, **physics)[1] == 85.0
    assert free_surplus_charge(100.0, 0.0, 0.0, E_MAX - 8.0, eta_c=0.8)[1] == 10.0
    first = solve_dispatch(np.ones(1), np.zeros(1), np.zeros(1), E_MAX - 8.0,
                           a_fixed=np.array([100.0]), first_slot_policy=True, **physics)
    assert first.ok and first.c[0] == pytest.approx(10.0)


def test_cycle_elimination_preserves_custom_efficiency_soc_and_balance():
    c, d, w = np.array([100.0]), np.array([34.0]), np.array([2.0])
    c2, d2, w2 = _eliminate_cycle(c, d, w, eta_c=0.8, eta_d=0.85)
    np.testing.assert_allclose(0.8 * c2 - d2 / 0.85, 0.8 * c - d / 0.85)
    np.testing.assert_allclose(c2 + w2 - d2, c + w - d)
    assert d2[0] == pytest.approx(0.0)


def test_screen_continuation_uses_custom_efficiencies():
    a = np.zeros(T)
    a[-1] = 100.0
    net = np.zeros(T)
    # Charge 100 on the last slot, then discharge 50 on next day's first slot.
    result = greedy_to_horizon(a, a, np.zeros(1), np.ones(T), np.ones(1), net,
                               np.array([50.0]), E_MIN, T - 1, eta_c=0.8, eta_d=0.85)
    assert result[3] == pytest.approx(E_MIN + 80.0)
    assert result[4] == pytest.approx(E_MIN + 80.0 - 50.0 / 0.85)
    assert result[2] == pytest.approx(0.0)
    assert kappa_from_prices(1.4, eta_d=0.85) == pytest.approx(5.0 * 0.85 * 1.4)


def test_disable_worst_case_keeps_four_candidates_paths_and_mean_gate():
    old = np.full(T, 10.0)
    star = old.copy()
    star[-1] = 0.0
    extra = old.copy()
    extra[-1] = 7.5
    empty = np.zeros(0)
    paths = [dict(src_day=i, eps=np.array([0.0 if i < 19 else 10.0]),
                  eps_next=empty, z=None, z_next=None) for i in range(20)]
    args = (old, star, old, np.zeros(T), empty, np.ones(T), empty, empty, E_MIN, T - 1, paths)
    kw = dict(eps_c=1.0, r=1.0, kappa=0.0, extra_contracts={2.0: extra}, eta_c=0.8, eta_d=0.85)
    chosen_safe, safe = screen_lambda(*args, worst_case=True, **kw)
    chosen_open, opened = screen_lambda(*args, worst_case=False, **kw)
    assert chosen_safe == 0.0 and chosen_open == 1.0
    assert set(safe["scores"]) == set(opened["scores"]) == {0.0, 0.5, 1.0, 2.0}
    for key, row in safe["scores"].items():
        assert {k: v for k, v in row.items() if k != "ok"} == {
            k: v for k, v in opened["scores"][key].items() if k != "ok"}
    assert opened["scores"][1.0]["mean_g_net"] == pytest.approx(2.5)
    assert opened["scores"][1.0]["min_g_net"] == pytest.approx(-45.0)
    assert opened["scores"][2.0]["ok"] is False  # mean gate still rejects.
    _, debited = screen_lambda(*args, **{**kw, "kappa": 2.0}, worst_case=False)
    assert debited["scores"][1.0]["mean_g_net"] < 0  # terminal debit still applies.


def test_no_adjust_can_keep_identical_flexible_protection_schedule():
    q3 = official_jobs()[1]
    no_adjust = replace(q3, adjust="none", adjust_hours=(), protection_flex=True)
    np.testing.assert_array_equal(_alpha_arr(2 * T, 0, q3), _alpha_arr(2 * T, 0, no_adjust))
    np.testing.assert_array_equal(_alpha_arr(T, 0, replace(q3, adjust="none")), q3.alpha)
    assert _alpha_arr(T, 0, no_adjust)[36] == q3.alpha_flex


def test_disable_plan_protection_bypasses_positive_buffer(data, monkeypatch):
    import cumcm_v11.simulate as sim
    calls, inputs = [], []
    original_solver = sim.solve_dispatch

    def positive_buffer(self, n_hat, *args):
        calls.append(n_hat.copy())
        return n_hat + 600.0

    def spy_solver(price, load_e, pv_e, e0, **kwargs):
        if not kwargs.get("h_allowed", True):
            inputs.append(load_e - pv_e)
        return original_solver(price, load_e, pv_e, e0, **kwargs)

    monkeypatch.setattr(ForecastEngine, "protect_net", positive_buffer)
    monkeypatch.setattr(sim, "solve_dispatch", spy_solver)
    cfg = official_jobs()[0]
    simulate_year(data, cfg, max_days=1, log=lambda *a, **k: None)
    simulate_year(data, replace(cfg, plan_protection=False), max_days=1, log=lambda *a, **k: None)
    assert len(calls) == 1
    np.testing.assert_allclose(inputs[1], calls[0] / 6.0, atol=1e-9)
    np.testing.assert_allclose(inputs[0] - inputs[1], 100.0, atol=1e-9)


def test_simulation_passes_efficiency_and_terminal_credit_to_every_solver(data, monkeypatch):
    import cumcm_v11.simulate as sim
    calls = []
    original_solver = sim.solve_dispatch

    def spy(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(sim, "solve_dispatch", spy)
    cfg = replace(official_jobs()[1], eta_c=0.8, eta_d=0.85, terminal_mu=0.3,
                  kappa_fixed=None)
    run = simulate_year(data, cfg, max_days=1, log=lambda *a, **k: None)
    assert len(calls) >= 10  # day-ahead, keep, adjust and fourth candidate at each issue.
    assert all(row["eta_c"] == 0.8 and row["eta_d"] == 0.85 and row["terminal_mu"] == 0.3
               for row in calls)
    assert run.n_audit == 0
    np.testing.assert_allclose(np.diff(run.E_slot[0]), 0.8 * run.c[0] - run.d[0] / 0.85, atol=1e-8)
    events = [e for e in run.plan_events if e["hour"] > 0]
    assert events and all(e["screen"]["kappa"] == pytest.approx(5 * 0.85 * max(data.price1)) for e in events)


def test_secondary_solver_failure_is_retained(monkeypatch):
    import cumcm_v11.optimize as opt
    original = opt.linprog
    count = 0

    def failing_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            return SimpleNamespace(success=False, x=None, message="injected secondary failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(opt, "linprog", failing_second)
    plan = solve_dispatch(np.ones(2), np.ones(2), np.zeros(2), E_MIN)
    assert plan.ok
    assert plan.secondary_failure == "injected secondary failure"


def test_simulation_failure_logs_survive_fallback(data, monkeypatch):
    import cumcm_v11.simulate as sim
    from cumcm_v11.optimize import Dispatch

    def fail(price, load_e, pv_e, e0, **kwargs):
        n = len(price)
        return Dispatch(*(np.zeros(n) for _ in range(5)), np.full(n + 1, e0),
                        float("nan"), False, "injected primary failure")

    monkeypatch.setattr(sim, "solve_dispatch", fail)
    logs = []
    run = simulate_year(data, official_jobs()[0], max_days=1, log=lambda m, **kw: logs.append(m))
    assert run.n_lp_fail == 1
    assert run.failure_logs[0]["message"] == "injected primary failure"
    assert any("injected primary failure" in line for line in logs)
    assert "injected primary failure" in run.plan_events[0]["reason"]
