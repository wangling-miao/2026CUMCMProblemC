"""Simulation presets. Official tasks plus Q3 mechanism ablations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .constants import (
    ETA_C,
    ETA_D,
    KAPPA_Q3,
    PLAN_BUFFER_DAYS,
    PLAN_BUFFER_MIN_DAYS,
    PLAN_BUFFER_OBS,
    SCREEN_EPS_C,
    SCREEN_EPS_E,
    SCREEN_MIN_PATHS,
    SCREEN_PATHS,
    SCREEN_R,
    SCREEN_WORST_CASE,
    TERMINAL_MU,
)


@dataclass
class SimConfig:
    name: str
    use_att3: bool = False
    adjust: str = "none"  # none | selective
    adjust_hours: tuple[int, ...] = (6, 12, 18)
    att3_issues: tuple[int, ...] = (0, 6, 12, 18)
    price_mode: str = "att1"  # att1 | forecast_att4
    plan_horizon: int = 288
    exec_horizon: int = 144
    alpha: float = 0.80
    alpha_flex: float = 0.70
    protection_flex: bool | None = None  # None follows adjust; explicit override isolates adjustments
    plan_protection: bool = True
    terminal_mu: float = TERMINAL_MU
    eta_c: float = ETA_C
    eta_d: float = ETA_D
    load_bias: bool = True
    pv_weighted: bool = True
    pv_bias: bool = True
    fusion: bool = False
    price_bias: bool = False
    slot_correct: bool = True
    extra_reopt: bool = True
    control_step: int = 1
    min_adjust_save: float = 5.0
    min_adjust_kwh: float = 20.0
    soc_dev: float = 400.0
    decay: float = 0.90
    err_window: int = 28
    err_window_unit: str = "days"  # archive: 28 completed days of 6-slot blocks
    min_err_samples: int = 8
    plan_buffer_mode: str = "days"  # days | obs | archive
    plan_buffer_days: int = PLAN_BUFFER_DAYS
    plan_buffer_obs: int = PLAN_BUFFER_OBS
    plan_buffer_min_days: int = PLAN_BUFFER_MIN_DAYS
    economic_exec: bool = False  # v5 default: greedy real-time baseline
    first_slot_policy: bool = True
    allow_emergency_charge: bool = False
    gamma: float = 0.0
    screen: bool = False
    screen_worst_case: bool = SCREEN_WORST_CASE  # official: mean Ĝ only; True restores min Ĝ ≥ −r
    n_screen_paths: int = SCREEN_PATHS
    min_screen_paths: int = SCREEN_MIN_PATHS
    eps_c: float = SCREEN_EPS_C
    r_budget: float = SCREEN_R
    eps_e: float = SCREEN_EPS_E
    terminal_price: bool = True  # v7: debit κD instead of the 100 kWh gate
    kappa_fixed: float | None = KAPPA_Q3  # None → causal κ from realized/forecast prices
    energy_gate: bool = False
    v2_candidate: bool = False
    template: str = "result2.xlsx"
    outfile: str = "result2.xlsx"
    with_adjust_sheet: bool = False


def official_jobs() -> list[SimConfig]:
    q2 = SimConfig(
        name="q2",
        use_att3=False,
        adjust="none",
        price_mode="att1",
        alpha=0.80,
        alpha_flex=0.80,
        fusion=False,
        price_bias=False,
        economic_exec=False,
        screen=False,
        plan_buffer_mode="obs",
        plan_buffer_obs=28,
        template="result2.xlsx",
        outfile="result2.xlsx",
        with_adjust_sheet=False,
    )
    q3 = SimConfig(
        name="q3",
        use_att3=True,
        adjust="selective",
        att3_issues=(0, 6, 12, 18),
        adjust_hours=(6, 12, 18),
        price_mode="att1",
        alpha=0.80,
        alpha_flex=0.70,
        fusion=True,
        price_bias=False,
        economic_exec=False,
        screen=True,
        screen_worst_case=False,
        terminal_price=True,
        kappa_fixed=KAPPA_Q3,
        energy_gate=False,
        v2_candidate=True,
        plan_buffer_mode="days",
        plan_buffer_days=5,
        template="result3.xlsx",
        outfile="result3.xlsx",
        with_adjust_sheet=True,
    )
    q42 = replace(
        q2,
        name="q4-2",
        price_mode="forecast_att4",
        price_bias=True,
        template="result4-2.xlsx",
        outfile="result4-2.xlsx",
    )
    q43 = replace(
        q3,
        name="q4-3",
        price_mode="forecast_att4",
        price_bias=True,
        kappa_fixed=None,
        template="result4-3.xlsx",
        outfile="result4-3.xlsx",
    )
    return [q2, q3, q42, q43]


def q3_ablations() -> list[SimConfig]:
    """A/B/D/E plus q3-ns (same base, no risk screen)."""
    base = [c for c in official_jobs() if c.name == "q3"][0]
    ns = replace(
        base,
        name="q3-ns",
        screen=False,
        outfile="ablation_q3ns.xlsx",
        with_adjust_sheet=True,
    )
    a = replace(
        base,
        name="q3-A",
        adjust="none",
        att3_issues=(0,),
        adjust_hours=(),
        alpha_flex=0.80,
        screen=False,
        outfile="ablation_q3A.xlsx",
        with_adjust_sheet=False,
    )
    b = replace(
        base,
        name="q3-B",
        adjust="none",
        att3_issues=(0, 6, 12, 18),
        adjust_hours=(),
        alpha_flex=0.80,
        screen=False,
        outfile="ablation_q3B.xlsx",
        with_adjust_sheet=False,
    )
    d = replace(
        base,
        name="q3-D",
        adjust="selective",
        att3_issues=(0, 6, 12),
        adjust_hours=(6, 12),
        outfile="ablation_q3D.xlsx",
        with_adjust_sheet=True,
    )
    e = replace(
        base,
        name="q3-E",
        adjust="selective",
        att3_issues=(0, 6, 12),
        adjust_hours=(6, 12, 18),
        outfile="ablation_q3E.xlsx",
        with_adjust_sheet=True,
    )
    q43ns = replace(
        [c for c in official_jobs() if c.name == "q4-3"][0],
        name="q4-3-ns",
        screen=False,
        outfile="ablation_q43ns.xlsx",
        with_adjust_sheet=True,
    )
    q3c = replace(
        base,
        name="q3-3cand",
        v2_candidate=False,
        outfile="ablation_q33cand.xlsx",
        with_adjust_sheet=True,
    )
    q43c = replace(
        [c for c in official_jobs() if c.name == "q4-3"][0],
        name="q4-3-3cand",
        v2_candidate=False,
        outfile="ablation_q433cand.xlsx",
        with_adjust_sheet=True,
    )
    q2 = [c for c in official_jobs() if c.name == "q2"][0]
    q42 = [c for c in official_jobs() if c.name == "q4-2"][0]
    q43 = [c for c in official_jobs() if c.name == "q4-3"][0]
    q2b0 = replace(q2, name="q2-B0", plan_buffer_mode="archive", outfile="ablation_q2B0.xlsx")
    q2b2 = replace(q2, name="q2-B2", plan_buffer_mode="days", plan_buffer_days=5, outfile="ablation_q2B2.xlsx")
    q42b0 = replace(q42, name="q4-2-B0", plan_buffer_mode="archive", outfile="ablation_q42B0.xlsx")
    q42b2 = replace(q42, name="q4-2-B2", plan_buffer_mode="days", plan_buffer_days=5, outfile="ablation_q42B2.xlsx")
    q3b0 = replace(base, name="q3-B0", plan_buffer_mode="archive", outfile="ablation_q3B0.xlsx", with_adjust_sheet=True)
    q43b0 = replace(q43, name="q4-3-B0", plan_buffer_mode="archive", outfile="ablation_q43B0.xlsx", with_adjust_sheet=True)
    return [ns, a, b, d, e, q43ns, q3c, q43c, q2b0, q2b2, q42b0, q42b2, q3b0, q43b0]
