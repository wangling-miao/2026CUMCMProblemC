"""Problem 1: deterministic typical-day LP. Unchanged from v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import (
    E0_JAN1,
    Q1_BASELINE,
    Q1_CHARGE,
    Q1_COST,
    Q1_DISCHARGE,
    Q1_ENERGY,
    RESULT_DIR,
)
from .data import Data, to_energy
from .export import export_result1
from .optimize import solve_dispatch
from .settle import audit_day, cash


def run_q1(data: Data, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    price = data.price1
    load_e = to_energy(data.load1)
    pv_e = to_energy(data.pv1)
    baseline = float(np.dot(price, np.maximum(load_e - pv_e, 0.0)))
    disp = solve_dispatch(
        price,
        load_e,
        pv_e,
        E0_JAN1,
        e_end=E0_JAN1,
        h_allowed=False,
        terminal_mu=0.0,
        secondary_throughput=True,
    )
    if not disp.ok:
        raise RuntimeError(f"Q1 LP failed: {disp.message}")
    E, errs = audit_day(E0_JAN1, disp.a, disp.c, disp.d, disp.h, disp.w, load_e, pv_e)
    met = cash(price, disp.a, disp.a, disp.h)
    rec = {
        "ok": disp.ok,
        "cost": met["cost"],
        "energy_q": met["energy_q"],
        "charge": float(disp.c.sum()),
        "discharge": float(disp.d.sum()),
        "E0": float(E[0]),
        "E24": float(E[-1]),
        "E_min": float(E.min()),
        "E_max": float(E.max()),
        "h_sum": float(disp.h.sum()),
        "waste": float(disp.w.sum()),
        "baseline": baseline,
        "save_pct": 100.0 * (1.0 - met["cost"] / baseline) if baseline else 0.0,
        "audit": errs[:10],
        "delta_cost": met["cost"] - Q1_COST,
        "delta_energy": met["energy_q"] - Q1_ENERGY,
        "delta_charge": float(disp.c.sum()) - Q1_CHARGE,
        "delta_discharge": float(disp.d.sum()) - Q1_DISCHARGE,
        "delta_baseline": baseline - Q1_BASELINE,
        "message": disp.message,
    }
    export_result1(
        out_dir / "result1.xlsx",
        data.template_labels,
        disp.a,
        disp.c,
        disp.d,
        E[0],
        E[-1],
    )
    (out_dir / "q1_check.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec
