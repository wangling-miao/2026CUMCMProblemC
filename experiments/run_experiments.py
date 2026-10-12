"""Frozen retrospective experiments; results never overwrite official workbooks."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/model/src"))
import numpy as np
import pandas as pd
import scipy
from cumcm_v11.config import SimConfig, official_jobs
from cumcm_v11.constants import E_MIN, E_MAX, E0_JAN1, T, Q_BESS_MAX, EXPORT_START, P_MAX_ATT1
from cumcm_v11.data import load_all
from cumcm_v11.export import export_trace_npz
from cumcm_v11.simulate import simulate_year

OUT = ROOT / "results/experiments"
REQUIRED_OUTPUTS = {"trace.npz", "daily.csv", "monthly.csv", "decisions.jsonl.gz", "snapshots.jsonl.gz", "solver_failures.json", "audit.json", "case.json"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_record_valid(record, path, signature=None):
    try:
        if record.get("status") != "PASS" or set(record.get("output_hashes", {})) != REQUIRED_OUTPUTS:
            return False
        if signature is not None and record.get("signature") != signature:
            return False
        return all((path / name).is_file() and file_sha(path / name) == digest for name, digest in record["output_hashes"].items())
    except (OSError, ValueError, TypeError):
        return False


def build_matrix():
    official = {c.name: c for c in official_jobs()}
    cases = []

    def add(case_id, cfg, family, factor, value, label, reference, archived=None):
        cfg = replace(cfg, name=case_id)
        cases.append(dict(case_id=case_id, config=asdict(cfg), family=family, factor=factor,
                          value=value, label=label, reference=reference, archived_reference=archived,
                          price_environment="固定电价" if cfg.price_mode == "att1" else "波动电价"))

    for code, case_id, ref in [("q2", "official_q2", "official_q2"),
                               ("q3", "full_fixed", "full_fixed"),
                               ("q4-2", "official_q42", "official_q42"),
                               ("q4-3", "full_variable", "full_variable")]:
        cfg = official[code]
        # 66-group matrix keeps the with-min-path control as "完整方案".
        if case_id in ("full_fixed", "full_variable"):
            cfg = replace(cfg, screen_worst_case=True)
        add(case_id, cfg, "baseline", "baseline", 0, "正式完整方案" if "full" in case_id else "正式不可调整方案", ref, code)

    removals = [
        ("M0", dict(use_att3=False, fusion=False, adjust="none", adjust_hours=(),
                    plan_protection=False, terminal_mu=0.0, protection_flex=True), "历史预测与确定性线性规划基线"),
        ("A1_no_intraday_pv", dict(att3_issues=(0,)), "仅使用0时光伏预报"),
        ("A2_no_day_ahead_protection", dict(plan_protection=False), "无日前净负荷保护"),
        ("A3_no_nu", dict(terminal_mu=0.0), "无规划末端储电价值"),
        ("A4_no_kappa", dict(kappa_fixed=0.0), "无筛选末端短缺惩罚"),
        ("A5_no_tail", dict(screen_worst_case=False), "无最坏情景损失约束"),
        ("A6_no_adjust", dict(adjust="none", adjust_hours=(), protection_flex=True), "无日内合同调整"),
        ("A7_no_fusion", dict(fusion=False), "仅用已发布光伏预报而不组合"),
    ]
    for suffix, key, ref in [("fixed", "q3", "full_fixed"), ("variable", "q4-3", "full_variable")]:
        base = replace(official[key], screen_worst_case=True)
        for tag, changes, label in removals:
            add(tag + "_" + suffix, replace(base, **changes), "ablation", tag, 0, label, ref)
        grids = {
            "alpha": [0.70, 0.75, 0.85, 0.90],
            "nu": [0.30, 0.375, 0.525, 0.60],
            "eta": [0.85, 0.875, 0.925, 0.95],
            "H": [144, 216],
            "r": [0.0, 2.5, 10.0, 20.0],
        }
        for factor, values in grids.items():
            for value in values:
                if factor == "alpha":
                    changes = dict(alpha=value, alpha_flex=round(value - 0.10, 3))
                elif factor == "nu":
                    changes = dict(terminal_mu=value)
                elif factor == "eta":
                    changes = dict(eta_c=value, eta_d=value)
                    if key == "q3":
                        changes["kappa_fixed"] = 5.0 * value * P_MAX_ATT1
                elif factor == "H":
                    changes = dict(plan_horizon=int(value))
                else:
                    changes = dict(r_budget=value)
                tag = str(value).replace(".", "p")
                add(f"sensitivity_{factor}_{tag}_{suffix}", replace(base, **changes), "sensitivity",
                    factor, value, f"{factor}={value}", ref)

    for key, ref in [("q2", "official_q2"), ("q4-2", "official_q42")]:
        for value in [0.70, 0.75, 0.85, 0.90]:
            add(f"sensitivity_alpha_{str(value).replace('.', 'p')}_{key.replace('-', '')}",
                replace(official[key], alpha=value, alpha_flex=value), "sensitivity",
                "alpha", value, f"不可调整方案 alpha={value}", ref)
    for key, ref in [("q4-2", "official_q42"), ("q4-3", "full_variable")]:
        add(f"A8_no_price_bias_{key.replace('-', '')}", replace(official[key], price_bias=False),
            "ablation", "A8_no_price_bias", 0, "周同期电价预测不作中位数偏差校正", ref)
    assert len(cases) == 66 and len({c["case_id"] for c in cases}) == 66
    return cases


def independent_audit(run, data, cfg):
    n = len(run.q)
    array_shapes = {key:(n, T) for key in ["q", "a", "c", "d", "h", "w", "p_set", "p_plan", "fc_L0", "fc_V0"]}
    array_shapes.update(E_slot=(n, T+1), a_snap=(n, 4, T), day_cost=(n,))
    invalid_arrays = [key for key, shape in array_shapes.items() if np.shape(getattr(run, key)) != shape or not np.all(np.isfinite(getattr(run, key)))]
    if invalid_arrays:
        return {"pass":False, "invalid_shape_or_nonfinite_arrays":invalid_arrays}
    snapshot_keys = [(s["day"], s["decision_origin"], s["t0"], s["decision_hour"]) for s in run.snapshots]
    expected_snapshots = {(day, origin, origin*36, origin*6) for day in range(n) for origin in range(4)}
    snapshot_coverage_ok = len(snapshot_keys) == 4*n and set(snapshot_keys) == expected_snapshots
    event_keys = [(e["day"], e["hour"]) for e in run.plan_events]
    expected_events = {(day, 0) for day in range(n)}
    if cfg.adjust == "selective":
        expected_events |= {(day, hour) for day in range(n) for hour in cfg.adjust_hours}
    event_coverage_ok = len(event_keys) == len(expected_events) and set(event_keys) == expected_events
    balance = run.a + run.h + data.pv[:n] / 6.0 + run.d - data.load[:n] / 6.0 - run.c - run.w
    state = run.E_slot[:, 1:] - run.E_slot[:, :-1] - cfg.eta_c * run.c + run.d / cfg.eta_d
    recalculated = (run.p_set * run.a + 0.5 * run.p_set * np.abs(run.a - run.q) + 5.0 * run.p_set * run.h).sum(axis=1)
    freeze_errors = 0.0
    for day in range(n):
        for issue in range(4):
            stop = issue * 36
            if issue == 0:
                freeze_errors = max(freeze_errors, float(np.max(np.abs(run.a_snap[day, issue] - run.q[day]))))
            else:
                freeze_errors = max(freeze_errors, float(np.max(np.abs(run.a_snap[day, issue, :stop] - run.a_snap[day, issue - 1, :stop]))))
        freeze_errors = max(freeze_errors, float(np.max(np.abs(run.a[day] - run.a_snap[day, 3]))))
    bad_origin = sum(int(s["last_observed_slot"] != s["t0"] - 1 or s["pv_issue_used"] > s["decision_origin"]) for s in run.snapshots)
    if cfg.use_att3:
        bad_origin += sum(int(s["pv_issue_used"] * 6 not in cfg.att3_issues) for s in run.snapshots)
    decisions = [e for e in run.plan_events if e.get("hour", 0) > 0]
    future_paths = sum(int(row["src_day"] >= e["day"]) for e in decisions for score in e.get("screen", {}).get("scores", {}).values() for row in score.get("paths", []))
    violations = {
        "energy_balance_max_kwh": float(np.max(np.abs(balance))),
        "state_equation_max_kwh": float(np.max(np.abs(state))),
        "cash_recalculation_max_yuan": float(np.max(np.abs(recalculated - run.day_cost))),
        "continuity_max_kwh": float(np.max(np.abs(run.E_slot[1:, 0] - run.E_slot[:-1, -1]))) if n > 1 else 0.0,
        "soc_min_kwh": float(run.E_slot.min()), "soc_max_kwh": float(run.E_slot.max()),
        "power_excess_kwh": float(max(0.0, run.c.max() - Q_BESS_MAX, run.d.max() - Q_BESS_MAX)),
        "simultaneous_slots": int(np.sum((run.c > 1e-5) & (run.d > 1e-5))),
        "negative_quantity_min": float(min(run.q.min(), run.a.min(), run.c.min(), run.d.min(), run.h.min(), run.w.min())),
        "locked_contract_max_kwh": freeze_errors,
        "bad_snapshot_origins": bad_origin, "future_scenario_paths": future_paths,
        "lp_failures": int(run.n_lp_fail), "model_audit_errors": int(run.n_audit),
        "initial_soc_error_kwh": float(abs(run.E_slot[0,0]-E0_JAN1)),
        "snapshot_coverage_ok": snapshot_coverage_ok, "decision_coverage_ok": event_coverage_ok,
    }
    checks = [violations[k] <= 1e-5 for k in ["energy_balance_max_kwh", "state_equation_max_kwh", "cash_recalculation_max_yuan", "continuity_max_kwh", "power_excess_kwh", "locked_contract_max_kwh"]]
    checks += [violations["soc_min_kwh"] >= E_MIN - 1e-5, violations["soc_max_kwh"] <= E_MAX + 1e-5,
               violations["negative_quantity_min"] >= -1e-5]
    checks += [violations[k] == 0 for k in ["simultaneous_slots", "bad_snapshot_origins", "future_scenario_paths", "lp_failures", "model_audit_errors"]]
    checks += [violations["initial_soc_error_kwh"] <= 1e-8, snapshot_coverage_ok, event_coverage_ok]
    violations["pass"] = bool(all(checks))
    return violations


def run_case(payload):
    case, max_days, fingerprint = payload
    job_id = case["case_id"]
    cfg = SimConfig(**case["config"])
    signature = hashlib.sha256(json.dumps([case, max_days, fingerprint], sort_keys=True).encode()).hexdigest()
    path = OUT / ("runs" if max_days == 365 else f"pilot_{max_days}") / job_id / signature[:12]
    path.mkdir(parents=True, exist_ok=True)
    existing = path / "summary.json"
    if existing.exists():
        try:
            saved = json.loads(existing.read_text(encoding="utf-8"))
            if completed_record_valid(saved, path, signature):
                return saved
        except (OSError, ValueError, TypeError):
            pass
    previous_files = [p for p in path.iterdir() if p.is_file()]
    if previous_files:
        history = path / "previous_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        history.mkdir(parents=True, exist_ok=False)
        for previous in previous_files:
            shutil.copy2(previous, history / previous.name)
    start = time.perf_counter()
    write_json(path / "case.json", {**case, "signature": signature, "fingerprint": fingerprint, "max_days": max_days})
    write_json(path / "status.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        data = load_all()
        with (path / "run.log").open("w", encoding="utf-8") as logfile:
            def log(msg, **_kwargs):
                logfile.write(str(msg) + "\n")
                logfile.flush()
                match = re.search(r"day\s+(\d+)/", str(msg))
                write_json(path / "status.json", {"status": "RUNNING", "pid": os.getpid(), "day": int(match.group(1)) if match else None, "elapsed_seconds": time.perf_counter() - start, "message": str(msg)})
            run = simulate_year(data, cfg, max_days=max_days, log=log)
        audit = independent_audit(run, data, cfg)
        export_trace_npz(path / "trace.npz", run, data)
        for filename, rows in [("decisions.jsonl.gz", run.plan_events), ("snapshots.jsonl.gz", run.snapshots)]:
            with gzip.open(path / filename, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        n = len(run.q)
        annual = n == 365
        first = EXPORT_START if n > EXPORT_START else 0
        frame = pd.DataFrame(run.day_metrics)
        frame.insert(0, "date", pd.to_datetime(data.dates[:n]).strftime("%Y-%m-%d"))
        frame["evaluation"] = np.arange(n) >= first
        frame["boundary_adjusted_cost"] = frame["cost"] - 0.45 * (frame["E24"] - frame["E0"])
        frame["n_adjust"] = 0
        for event in run.plan_events:
            if event.get("hour", 0) > 0 and event.get("lambda", 0) > 0:
                frame.loc[event["day"], "n_adjust"] += 1
        issue_errors = {hour:[] for hour in [0,6,12,18]}
        for snap in run.snapshots:
            day, slot, hour = int(snap["day"]), int(snap["t0"]), int(snap["decision_hour"])
            if day >= first:
                error = np.asarray(snap["pv_kw"][:36]) - data.pv[day, slot:slot+36]
                issue_errors[hour].extend(np.abs(error).tolist())
        frame.to_csv(path / "daily.csv", index=False, encoding="utf-8-sig")
        monthly = frame.loc[frame.evaluation].assign(month=lambda f: f.date.str.slice(0, 7)).groupby("month").agg(
            cost=("cost", "sum"), emergency_cost=("cost_h", "sum"), emergency_kwh=("energy_h", "sum"),
            waste_kwh=("waste", "sum"), charge_kwh=("charge", "sum"), discharge_kwh=("discharge", "sum"),
            adjustments=("n_adjust", "sum"), boundary_adjusted_cost=("boundary_adjusted_cost", "sum"))
        monthly.to_csv(path / "monthly.csv", encoding="utf-8-sig")
        window = frame.loc[frame.evaluation]
        summary = {k: case[k] for k in ["case_id", "family", "factor", "value", "label", "reference", "archived_reference", "price_environment"]}
        write_json(path / "solver_failures.json", getattr(run, "failure_logs", []))
        summary.update(dict(status="PASS" if audit["pass"] else "FAILED_AUDIT", path=str(path.relative_to(ROOT)).replace("\\", "/"),
                            signature=signature, fingerprint_sha256=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
                            n_days=n, evaluation_days=len(window), evaluation_start=window.date.iloc[0], evaluation_end=window.date.iloc[-1],
                            complete_annual=annual, elapsed_seconds=time.perf_counter()-start, config=asdict(cfg),
                            cost_yuan=float(window.cost.sum()), cost_365_yuan=float(frame.cost.sum()),
                            boundary_adjusted_cost_yuan=float(window.boundary_adjusted_cost.sum()),
                            emergency_cost_yuan=float(window.cost_h.sum()), emergency_kwh=float(window.energy_h.sum()),
                            waste_kwh=float(window.waste.sum()), throughput_kwh=float(window.charge.sum()+window.discharge.sum()),
                            adjustment_up_kwh=float(window.energy_up.sum()), adjustment_down_kwh=float(window.energy_down.sum()),
                            adjustments=int(window.n_adjust.sum()), start_soc_kwh=float(run.E_slot[first, 0]), end_soc_kwh=float(run.E_slot[-1, -1]),
                            load_mae_kw=float(np.mean(np.abs(run.fc_L0[first:] - data.load[first:n]))),
                            pv_mae_kw=float(np.mean(np.abs(run.fc_V0[first:] - data.pv[first:n]))),
                            price_mae=float(np.mean(np.abs(run.p_plan[first:] - run.p_set[first:]))), audit=audit))
        summary["pv_next6h_mae_kw_by_issue"] = {str(hour):float(np.mean(errors)) for hour,errors in issue_errors.items()}
        summary["pv_intraday_next6h_mae_kw"] = float(np.mean(sum([issue_errors[h] for h in [6,12,18]], [])))
        if case.get("archived_reference"):
            archive = np.load(ROOT / f"results/diagnostics/{case['archived_reference']}_trace.npz")
            differences = {key: float(np.max(np.abs(getattr(run, key) - archive[key][:n]))) for key in ["q", "a", "c", "d", "h", "E_slot", "day_cost"]}
            summary["archived_max_abs_differences"] = differences
            if max(differences.values()) > 1e-4:
                summary["status"] = "FAILED_BASELINE_REGRESSION"
        write_json(path / "audit.json", audit)
        summary["output_hashes"] = {f: file_sha(path / f) for f in sorted(REQUIRED_OUTPUTS)}
        write_json(existing, summary)
        write_json(path / "status.json", {"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"]})
        return summary
    except Exception:
        failure = {**{k:case[k] for k in ["case_id","family","factor","value","reference"]}, "status": "FAILED_EXCEPTION", "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter()-start, "path": str(path.relative_to(ROOT)).replace("\\", "/")}
        write_json(path / "failure.json", failure)
        write_json(path / "status.json", failure)
        return failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["freeze", "pilot", "baseline", "all"], default="freeze")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    cases = clean(build_matrix())
    fingerprint = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                   "model_files": {p.name:file_sha(p) for p in sorted((ROOT / "code/model/src/cumcm_v11").glob("*.py"))},
                   "runner_sha256": file_sha(__file__),
                   "archive_traces": {key:file_sha(ROOT / f"results/diagnostics/{key}_trace.npz") for key in ["q2", "q3", "q4-2", "q4-3"]},
                   "data_files": {str(p.relative_to(ROOT)).replace("\\", "/"):file_sha(p) for p in sorted((ROOT / "data/raw").rglob("*.xlsx"))}}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), experiment_count=len(cases), fingerprint=fingerprint,
                    evaluation_period="2025-02-01 through 2025-12-31, 334 days; January warms forecast histories and SOC from 6000 kWh",
                    interpretation="Retrospective fixed-configuration analysis on previously used 2025 data. No independent-year generalization claim. Official parameters and workbooks are not replaced.", cases=cases)
    freeze = ROOT / "experiments/experiment_matrix.json"
    if args.stage == "freeze" or not freeze.exists():
        write_json(freeze, manifest)
    else:
        frozen = json.loads(freeze.read_text(encoding="utf-8"))
        if frozen["fingerprint"] != fingerprint or frozen["cases"] != cases:
            raise SystemExit("Source/configuration changed after freeze. Review and explicitly refreeze before running.")
    print(f"FROZEN {len(cases)} cases", flush=True)
    if args.stage == "freeze":
        return
    selected = cases
    days = 35 if args.stage == "pilot" else 365
    if args.stage in ["pilot", "baseline"]:
        selected = [c for c in cases if c["family"] == "baseline"]
    if args.cases:
        selected = [c for c in selected if c["case_id"] in args.cases]
        if len(selected) != len(args.cases):
            raise SystemExit("Unknown or excluded case")
    if args.stage == "all":
        baseline_path = OUT / "baseline_index.json"
        if not baseline_path.exists():
            raise SystemExit("Run and pass all four baseline regressions first.")
        passed = json.loads(baseline_path.read_text(encoding="utf-8"))
        current_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        expected_ids = {c["case_id"] for c in cases if c["family"] == "baseline"}
        valid_ids = len(passed) == 4 and {s.get("case_id") for s in passed} == expected_ids
        valid_records = True
        for s in passed:
            try:
                directory = (ROOT / s["path"]).resolve()
                saved = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                valid_records = valid_records and directory.is_relative_to(OUT.resolve()) and saved == s and completed_record_valid(s, directory)
            except (OSError, ValueError, KeyError, TypeError):
                valid_records = False
        if not valid_ids or not valid_records or any(not s.get("complete_annual") or s.get("evaluation_days") != 334 or s.get("fingerprint_sha256") != current_hash for s in passed):
            raise SystemExit("Baseline regression gate failed.")
    results = []
    index = OUT / f"{args.stage}_index.json"
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_case, (case, days, fingerprint)):case["case_id"] for case in selected}
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            write_json(index, sorted(results, key=lambda x:x["case_id"]))
            print(f"[{len(results)}/{len(selected)}] {result['case_id']} {result['status']} cost={result.get('cost_yuan')} seconds={result.get('elapsed_seconds',0):.1f}", flush=True)
    print(f"STAGE {args.stage} finished in {time.perf_counter()-started:.1f}s; PASS {sum(s['status']=='PASS' for s in results)}/{len(results)}", flush=True)
    if any(s["status"] != "PASS" for s in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
