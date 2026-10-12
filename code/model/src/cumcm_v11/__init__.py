"""CUMCM 2026 problem C, code version v11."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimConfig, official_jobs, q3_ablations
from .constants import EXPORT_START, RESULT_DIR, SPECIFIED_DATES, V2_COST_334, V10_COST_334
from .data import load_all
from .export import (
    export_audit,
    export_plans_jsonl,
    export_snapshots_jsonl,
    export_specified_csv,
    export_time_map,
    export_trace_npz,
    export_year,
)
from .q1 import run_q1
from .simulate import YearRun, simulate_year


def _summarize(name: str, run: YearRun, data) -> dict:
    n = run.q.shape[0]
    dates = pd.to_datetime(data.dates[:n])
    sl = slice(EXPORT_START, min(n, 365))
    if sl.start >= n:
        sl = slice(0, n)
    overlap = float(np.minimum(np.maximum(run.q[sl] - run.a[sl], 0.0), run.h[sl]).sum())
    mets = run.day_metrics[sl.start : sl.stop]
    tot = {
        "name": name,
        "n_days": n,
        "cost_334": float(run.day_cost[sl].sum()),
        "cost_365": float(run.day_cost.sum()),
        "cost_plan_334": float(sum(m.get("cost_plan", 0.0) for m in mets)),
        "cost_up_334": float(sum(m.get("cost_up", 0.0) for m in mets)),
        "cost_down_refund_334": float(sum(m.get("cost_down_refund", 0.0) for m in mets)),
        "cost_h_334": float(sum(m.get("cost_h", 0.0) for m in mets)),
        "h_334": float(run.h[sl].sum()),
        "q_334": float(run.q[sl].sum()),
        "a_334": float(run.a[sl].sum()),
        "up_334": float(np.maximum(run.a[sl] - run.q[sl], 0).sum()),
        "down_334": float(np.maximum(run.q[sl] - run.a[sl], 0).sum()),
        "waste_334": float(run.w[sl].sum()),
        "overlap_down_h_334": overlap,
        "E_feb1": float(run.E_start[EXPORT_START]) if n > EXPORT_START else float(run.E_start[0]),
        "E_dec31": float(run.E_end[-1]),
        "lp_fail": run.n_lp_fail,
        "audit": run.n_audit,
        "n_adjust_334": run.n_adjust_334,
        "n_keep_334": run.n_keep_334,
        "n_half": run.n_half,
        "n_full": run.n_full,
        "n_half_334": run.n_half_334,
        "n_full_334": run.n_full_334,
        "n_cold_keep": run.n_cold_keep,
        "n_v2_334": run.n_v2_334,
        "n_b_gt_a": run.n_b_gt_a,
        "n_greedy": run.n_greedy,
        "n_lp_act": run.n_lp_act,
    }
    cmp_name = {
        "q3-3cand": "q3-3cand",
        "q4-3-3cand": "q4-3-3cand",
        "q3-4cand": "q3",
        "q4-3-4cand": "q4-3",
    }.get(name, name)
    if name in ("q3", "q4-3"):
        v8_key = name
    else:
        v8_key = cmp_name
    if name in V2_COST_334 and n > EXPORT_START:
        tot["v2_cost_334"] = V2_COST_334[name]
        tot["vs_v2"] = tot["cost_334"] - V2_COST_334[name]
        tot["vs_v2_pct"] = 100.0 * tot["vs_v2"] / V2_COST_334[name]
    elif name in ("q3-3cand",) and n > EXPORT_START:
        tot["v2_cost_334"] = V2_COST_334["q3"]
        tot["vs_v2"] = tot["cost_334"] - V2_COST_334["q3"]
        tot["vs_v2_pct"] = 100.0 * tot["vs_v2"] / V2_COST_334["q3"]
    elif name in ("q4-3-3cand",) and n > EXPORT_START:
        tot["v2_cost_334"] = V2_COST_334["q4-3"]
        tot["vs_v2"] = tot["cost_334"] - V2_COST_334["q4-3"]
        tot["vs_v2_pct"] = 100.0 * tot["vs_v2"] / V2_COST_334["q4-3"]
    if name in V10_COST_334 and n > EXPORT_START:
        tot["v10_cost_334"] = V10_COST_334[name]
        tot["vs_v10"] = tot["cost_334"] - V10_COST_334[name]
        tot["vs_v10_pct"] = 100.0 * tot["vs_v10"] / V10_COST_334[name]
    spec = {}
    for s in SPECIFIED_DATES:
        idx = np.where(dates == np.datetime64(s))[0]
        if len(idx) == 0:
            continue
        day = int(idx[0])
        spec[s] = {
            "cost": float(run.day_cost[day]),
            "q": float(run.q[day].sum()),
            "a": float(run.a[day].sum()),
            "h": float(run.h[day].sum()),
            "E0": float(run.E_start[day]),
            "E24": float(run.E_end[day]),
        }
    tot["specified"] = spec
    return tot


def _write_metrics(path: Path, q1: dict, summaries: list[dict]) -> None:
    lines = ["# v11 结果", "", "## 问题 1 回归", ""]
    lines.append(f"- 购电费 {q1['cost']:.6f} 元（差 {q1['delta_cost']:.6e}）")
    lines.append(f"- 节省 {q1['save_pct']:.4f}%")
    lines.append("")
    lines.append("## 问题 2–4（2.1–12.31，334 天）")
    lines.append("")
    lines.append(
        "| 任务 | 总购电费 | 相对v2 | 相对v10来源 | 紧急电量 | 紧急费 | 下调∩紧急 | "
        "保持/半幅/完整/分位数(334) | B>A | 审计 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        vs = s.get("vs_v2_pct")
        vs_s = f"{vs:+.2f}%" if vs is not None else "—"
        v10 = s.get("vs_v10_pct")
        v10_s = f"{v10:+.2f}%" if v10 is not None else "—"
        lines.append(
            f"| {s['name']} | {s['cost_334']:.2f} | {vs_s} | {v10_s} | {s['h_334']:.1f} | "
            f"{s['cost_h_334']:.1f} | {s['overlap_down_h_334']:.1f} | "
            f"{s['n_keep_334']}/{s.get('n_half_334', s['n_half'])}/"
            f"{s.get('n_full_334', s['n_full'])}/{s.get('n_v2_334', 0)} | {s['n_b_gt_a']} | {s['audit']} |"
        )
    lines.append("")
    lines.append("v11 定稿：问题2/4-2 用 B1（末 28 观测）；问题3/4-3 用 B2（5 日）四候选。")
    lines.append("不是全局最优证明。B0/B2 消融保留。result2/result4-2 来自 B1 轨迹。")
    lines.append("")
    for s in summaries:
        vs2 = s.get("vs_v2")
        extra = f"；相对v2 {vs2:+.2f} 元" if vs2 is not None else ""
        lines.append(
            f"- {s['name']}: 原计划 {s['cost_plan_334']:.2f}，紧急 {s['cost_h_334']:.2f}；"
            f"年末 SOC {s['E_dec31']:.1f}{extra}"
        )
    lines.append("")
    lines.append("## 指定日")
    for s in summaries:
        lines.append(f"### {s['name']}")
        for day, m in s.get("specified", {}).items():
            lines.append(
                f"- {day}: 费 {m['cost']:.2f}，计划 {m['q']:.1f}，调整后 {m['a']:.1f}，"
                f"紧急 {m['h']:.1f}，SOC {m['E0']:.1f}→{m['E24']:.1f}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_run(out: Path, cfg: SimConfig, run: YearRun, data, official: bool) -> dict:
    costs = run.day_cost.copy()
    qtys = run.q.sum(axis=1)
    if official and run.q.shape[0] > EXPORT_START:
        export_year(out / cfg.outfile, cfg.template, run, costs, qtys, with_adjust=cfg.with_adjust_sheet)
    diag = out / "diag"
    diag.mkdir(parents=True, exist_ok=True)
    export_trace_npz(diag / f"{cfg.name}_trace.npz", run, data)
    export_plans_jsonl(diag / f"{cfg.name}_plans.jsonl", run)
    export_snapshots_jsonl(diag / f"{cfg.name}_snapshots.jsonl", run)
    export_specified_csv(diag / f"{cfg.name}_specified.csv", run, data, SPECIFIED_DATES)
    return _summarize(cfg.name, run, data)


def _run_job(payload: tuple) -> tuple[str, YearRun]:
    data, cfg, max_days = payload
    run = simulate_year(data, cfg, max_days=max_days, log=print)
    return cfg.name, run


def main() -> None:
    parser = argparse.ArgumentParser(description="CUMCM 2026 C v11")
    parser.add_argument("--only", choices=["q1", "q2", "q3", "q4", "all", "ablation"], default="all")
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--max-days", type=int, default=365)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("loading attachments...", flush=True)
    data = load_all()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    export_time_map(out / "time_map.csv", data)

    q1 = None
    if args.only in ("q1", "all", "q2", "q3", "q4", "ablation"):
        print("=== Q1 ===", flush=True)
        q1 = run_q1(data, out)
        print(f"Q1 cost={q1['cost']:.6f}  d={q1['delta_cost']:.3e}", flush=True)
        if args.only == "q1":
            _write_metrics(out / "metrics.md", q1, [])
            return

    jobs: list[SimConfig] = []
    if args.only in ("q2", "all"):
        jobs.extend([c for c in official_jobs() if c.name == "q2"])
    if args.only in ("q3", "all"):
        jobs.extend([c for c in official_jobs() if c.name == "q3"])
    if args.only in ("q4", "all"):
        jobs.extend([c for c in official_jobs() if c.name.startswith("q4")])
    if args.only == "ablation" or args.ablation:
        extra = q3_ablations()
        if args.only == "ablation":
            jobs = extra + [c for c in official_jobs() if c.name == "q3"]
        else:
            jobs.extend(extra)

    seen = set()
    uniq: list[SimConfig] = []
    for c in jobs:
        if c.name in seen:
            continue
        seen.add(c.name)
        uniq.append(replace(c, plan_horizon=args.horizon))
    jobs = uniq

    summaries = []
    dump = {}
    official_names = {c.name for c in official_jobs()}
    (out / "run_config.json").write_text(
        json.dumps([asdict(c) for c in jobs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.jobs > 1 and len(jobs) > 1:
        print(f"running {len(jobs)} tasks with {args.jobs} processes", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run_job, (data, cfg, args.max_days)): cfg for cfg in jobs}
            results = {}
            for fut in as_completed(futs):
                name, run = fut.result()
                results[name] = run
                print(f"{name} done cost={run.day_cost[EXPORT_START:].sum():.2f}", flush=True)
        for cfg in jobs:
            run = results[cfg.name]
            summ = _export_run(out, cfg, run, data, official=cfg.name in official_names)
            summaries.append(summ)
            dump[cfg.name] = summ
    else:
        for cfg in jobs:
            print(f"=== {cfg.name} screen={cfg.screen} ===", flush=True)
            t1 = time.time()
            run = simulate_year(data, cfg, max_days=args.max_days, log=print)
            summ = _export_run(out, cfg, run, data, official=cfg.name in official_names)
            summaries.append(summ)
            dump[cfg.name] = summ
            print(f"{cfg.name} cost={summ['cost_334']:.2f}  in {time.time() - t1:.1f}s", flush=True)

    if q1 is None:
        q1 = run_q1(data, out)
    _write_metrics(out / "metrics.md", q1, summaries)
    export_audit(out / "audit.json", summaries)
    (out / "summary.json").write_text(
        json.dumps({"q1": q1, "year": dump}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("done in", f"{time.time() - t0:.1f}s", "->", out, flush=True)


if __name__ == "__main__":
    main()
