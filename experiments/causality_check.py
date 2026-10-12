"""Future-input perturbation check of Q3 and Q4-3 midnight / 06:00 decisions.

Run from any directory; defaults to checking both configurations. --scenario
q4-3 runs only its two replays and preserves a previously recorded Q3 check.
Writes only results/experiments/causality_check.json.
The inspected run retains the existing interval-average execution convention.
"""

from __future__ import annotations

import os

# Set before NumPy/SciPy imports. The installed threadpoolctl is not used.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import copy
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "model" / "src"))

import numpy as np
import scipy

from cumcm_v11.config import official_jobs
from cumcm_v11.data import load_all
from cumcm_v11.simulate import simulate_year


DAY = 14  # 2025-01-15, zero-based; evaluate midnight and 06:00 of this day.
CUT = 36  # At the 06:00 decision, only slots 0..35 are completed observations.
OUTPUT = ROOT / "results" / "experiments" / "causality_check.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    files = sorted((ROOT / "code" / "model" / "src" / "cumcm_v11").glob("*.py"))
    files += [Path(__file__), ROOT / "experiments" / "experiment_matrix.json"]
    return {p.relative_to(ROOT).as_posix(): sha_file(p) for p in files}


def compare(label: str, baseline, changed) -> dict:
    a, b = np.asarray(baseline), np.asarray(changed)
    same_shape = a.shape == b.shape
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    max_diff = float(np.max(np.abs(a - b))) if same_shape and finite and a.size else 0.0
    exact = bool(same_shape and np.array_equal(a, b))
    return {"field": label, "shape": list(a.shape), "same_shape": same_shape,
            "all_finite": finite, "max_abs_difference": max_diff,
            "exactly_equal": exact, "pass": exact and finite}


def run_check(scenario: str) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    source_before = source_hashes()
    cfg = next(config for config in official_jobs() if config.name == scenario)
    original = load_all()
    perturbed = copy.deepcopy(original)
    # Positive affine transformations make the perturbation reproducible and
    # conspicuous while preserving nonnegative power and positive prices.
    transformations = {
        "load": (1.37, 173.0),
        "pv": (0.61, 71.0),
        "price": (1.29, 0.17),
    }
    perturbations = []
    for name, (multiplier, offset) in transformations.items():
        src, dst = getattr(original, name), getattr(perturbed, name)
        dst[DAY, CUT:] = multiplier * src[DAY, CUT:] + offset
        dst[DAY + 1:] = multiplier * src[DAY + 1:] + offset
        perturbations.append({
            "field": name,
            "selection": "day_index=14, slots>=36; and every slot of day_index>=15",
            "formula": f"new = {multiplier} * original + {offset}",
            "changed_elements": int(np.count_nonzero(src != dst)),
            "max_abs_change": float(np.max(np.abs(src - dst))),
            "baseline_sha256": sha_array(src), "perturbed_sha256": sha_array(dst),
        })
    perturbed.pv_fc[DAY, 2:4] = 1.53 * original.pv_fc[DAY, 2:4] + 137.0
    perturbations.append({
        "field": "pv_fc", "selection": "day_index=14, issue_index=2/3 (12:00 / 18:00)",
        "formula": "new = 1.53 * original + 137.0",
        "changed_elements": int(np.count_nonzero(original.pv_fc != perturbed.pv_fc)),
        "max_abs_change": float(np.max(np.abs(original.pv_fc - perturbed.pv_fc))),
        "baseline_sha256": sha_array(original.pv_fc), "perturbed_sha256": sha_array(perturbed.pv_fc),
    })
    preserved = [
        compare("load_before_day15_06", original.load[:DAY].ravel(), perturbed.load[:DAY].ravel()),
        compare("load_day15_slots0to35", original.load[DAY, :CUT], perturbed.load[DAY, :CUT]),
        compare("pv_before_day15", original.pv[:DAY], perturbed.pv[:DAY]),
        compare("pv_day15_slots0to35", original.pv[DAY, :CUT], perturbed.pv[DAY, :CUT]),
        compare("price_before_day15", original.price[:DAY], perturbed.price[:DAY]),
        compare("price_day15_slots0to35", original.price[DAY, :CUT], perturbed.price[DAY, :CUT]),
        compare("published_day15_00_06_pv_fc", original.pv_fc[DAY, :2], perturbed.pv_fc[DAY, :2]),
        compare("pv_fc_before_day15", original.pv_fc[:DAY], perturbed.pv_fc[:DAY]),
        compare("price1_fixed_tariff", original.price1, perturbed.price1),
    ]
    runs, timings, logs = [], {}, {}
    for name, data in (("baseline", original), ("future_perturbed", perturbed)):
        messages = []
        begin = time.perf_counter()
        run = simulate_year(data, cfg, max_days=DAY + 1,
                            log=lambda message, **kwargs: messages.append(str(message)))
        timings[name] = time.perf_counter() - begin
        logs[name] = messages
        runs.append(run)
        print(f"{scenario}/{name}: {timings[name]:.3f}s, LP failures={run.n_lp_fail}, audits={run.n_audit}", flush=True)
    baseline, changed = runs
    comparisons, diagnostic_exclusions = [], []
    for hour, origin in ((0, 0), (6, 1)):
        snaps = []
        for run in runs:
            matches = [row for row in run.snapshots if row["day"] == DAY and row["decision_origin"] == origin]
            if len(matches) != 1:
                raise AssertionError(f"Expected one snapshot for day {DAY}, origin {origin}; got {len(matches)}")
            snaps.append(matches[0])
        for field in ("load_kw", "pv_kw", "price", "net_kwh"):
            comparisons.append(compare(f"day15_{hour:02d}:00.snapshot.{field}", snaps[0][field], snaps[1][field]))
        comparisons.append(compare(f"day15_{hour:02d}:00.contract.a_snap", baseline.a_snap[DAY, origin], changed.a_snap[DAY, origin]))
        for meta in ("day", "t0", "n", "decision_hour", "decision_origin", "pv_issue_used", "last_observed_slot"):
            comparisons.append(compare(f"day15_{hour:02d}:00.snapshot.{meta}", snaps[0][meta], snaps[1][meta]))
        diagnostic_exclusions.append({
            "field": f"day15_{hour:02d}:00.snapshot.actual_slot_kw",
            "baseline": snaps[0]["actual_slot_kw"], "perturbed": snaps[1]["actual_slot_kw"],
            "reason": "Diagnostic measurement only; not a candidate forecast input. At 06:00 this lies in the perturbed unfinished slot.",
        })
    comparisons += [
        compare("all_contracts_before_day15", baseline.a_snap[:DAY], changed.a_snap[:DAY]),
        compare("day15_SOC_up_to_06_before_slot36", baseline.E_slot[DAY, :CUT + 1], changed.E_slot[DAY, :CUT + 1]),
    ]
    for field in ("a", "c", "d", "h", "w"):
        comparisons.append(compare(f"day15.executed_{field}_slots0to35", getattr(baseline, field)[DAY, :CUT], getattr(changed, field)[DAY, :CUT]))
    source_after = source_hashes()
    source_unchanged = source_before == source_after
    numerical_ok = all(run.n_lp_fail == 0 and run.n_audit == 0 for run in runs)
    causal_pass = all(row["pass"] for row in comparisons + preserved)
    return {
        "schema": "future-perturbation-causality-check/v1",
        "status": "PASS" if causal_pass and source_unchanged and numerical_ok else "FAIL",
        "causality_comparisons_pass": causal_pass,
        "source_unchanged_during_check": source_unchanged,
        "numerical_runs_pass": numerical_ok,
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": str(np.datetime_as_string(original.dates[DAY], unit="D")),
        "day_index": DAY, "max_days": DAY + 1,
        "configuration": asdict(cfg),
        "visible_information": {
            "00:00": "Completed observations through day14; day15 00:00 issued PV forecast is available.",
            "06:00": "Completed observations through day15 slot35; day15 06:00 issued PV forecast is available.",
            "not_visible_at_either_decision": "day15 actual slots>=36, later-day actuals, day15 12:00/18:00 issued PV forecasts.",
        },
        "scope": [
            f"Exact invariance at the tested 00:00/06:00 {scenario} decisions under this deterministic future perturbation.",
            ("Q3 uses fixed attachment1 tariffs; its price invariance alone does not validate Q4 dynamic-price causality."
             if scenario == "q3" else
             "Q4-3 uses forecast_att4 dynamic prices: the test checks that actual prices from the unfinished 06:00 slot onward and future days do not alter the tested forecasts/contracts; unpublished 12:00/18:00 PV forecasts are also perturbed."),
            "The check does not require equality after 06:00; actual_slot_kw is excluded because it is diagnostic, not a planning input.",
            "This does not change or validate the declared interval-average feedback approximation as a deployable real-time measurement model.",
            "A passing perturbation test is evidence for these fields and cutoffs, not a proof covering every possible future perturbation.",
        ],
        "input_perturbations": perturbations,
        "preserved_visible_inputs": preserved,
        "comparisons": comparisons,
        "excluded_diagnostic_fields": diagnostic_exclusions,
        "runs": {name: {"elapsed_seconds": timings[name], "lp_failures": run.n_lp_fail,
                        "audit_errors": run.n_audit, "failure_logs": run.failure_logs,
                        "execution_log": logs[name]}
                 for name, run in zip(("baseline", "future_perturbed"), runs)},
        "source_sha256": source_before,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "thread_limits": {name: os.environ[name] for name in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", "q3", "q4-3"), default="all")
    args = parser.parse_args()
    previous = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    scenarios = dict(previous.get("scenarios", {}))
    if "configuration" in previous:
        # Preserve the original Q3 record, including its original script hash.
        scenarios[previous["configuration"]["name"]] = previous
    selected = ("q3", "q4-3") if args.scenario == "all" else (args.scenario,)
    for scenario in selected:
        try:
            scenarios[scenario] = run_check(scenario)
        except Exception as exc:
            scenarios[scenario] = {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc(),
                                   "source_sha256": source_hashes(), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    both_present = all(name in scenarios for name in ("q3", "q4-3"))
    status = "PASS" if both_present and all(scenarios[name]["status"] == "PASS" for name in ("q3", "q4-3")) else "FAIL"
    report = {
        "schema": "q3-q43-future-perturbation-causality-check/v2",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios_rerun_this_invocation": list(selected),
        "record_preservation": "Unselected scenario records remain unchanged, including their original script hashes and timestamps.",
        "source_sha256": source_hashes(),
        "reproduce_all": "python -X utf8 experiments/causality_check.py --scenario all",
        "scenarios": scenarios,
        "scope": "Each scenario supports only its tested day15 00:00/06:00 cutoffs and input fields. Q3 has fixed prices; Q4-3 explicitly exercises dynamic-price forecast and contract inputs. Later execution need not remain equal.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {OUTPUT}", flush=True)
    sys.exit(0 if report["status"] == "PASS" else 1)
