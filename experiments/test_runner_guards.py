"""Failure-injection checks for the frozen experiment runner; no model replay.

Run with ``python -B -X utf8 experiments/test_runner_guards.py``.  This invokes
pytest without its cache provider and writes only the named JSON test report.
Filesystem mutations during guard tests are replaced by in-memory fixtures.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import run_experiments as runner


def valid_run():
    """Two days, known settlement, constant valid SOC, full decision coverage."""
    n, t = 2, runner.T
    cfg = runner.official_jobs()[1]
    one = np.ones((n, t))
    zero = np.zeros((n, t))
    snapshots = [dict(day=d, decision_origin=o, t0=o*36,
                      decision_hour=o*6, last_observed_slot=o*36-1,
                      pv_issue_used=o) for d in range(n) for o in range(4)]
    events = [dict(day=d, hour=h) for d in range(n) for h in (0, 6, 12, 18)]
    run = SimpleNamespace(q=one.copy(), a=one.copy(), c=zero.copy(),
                          d=zero.copy(), h=zero.copy(), w=zero.copy(),
                          E_slot=np.full((n, t+1), runner.E0_JAN1),
                          p_set=one.copy(), p_plan=one.copy(),
                          fc_L0=6*one, fc_V0=zero.copy(),
                          day_cost=np.full(n, float(t)),
                          a_snap=np.ones((n, 4, t)), snapshots=snapshots,
                          plan_events=events, n_lp_fail=0, n_audit=0)
    data = SimpleNamespace(load=6*one, pv=zero.copy())
    return run, data, cfg


def test_valid_physical_control_passes():
    run, data, cfg = valid_run()
    assert runner.independent_audit(run, data, cfg)["pass"] is True


def test_previously_accepted_counterexample_is_rejected():
    run, data, cfg = valid_run()
    run.E_slot[:] += 1
    run.snapshots = []
    run.plan_events = []
    audit = runner.independent_audit(run, data, cfg)
    assert audit["pass"] is False
    assert audit["initial_soc_error_kwh"] == 1
    assert not audit["snapshot_coverage_ok"]
    assert not audit["decision_coverage_ok"]


@pytest.mark.parametrize("fault,metric", [
    ("initial_soc", "initial_soc_error_kwh"),
    ("missing_snapshot", "snapshot_coverage_ok"),
    ("duplicate_snapshot", "snapshot_coverage_ok"),
    ("wrong_snapshot_slot", "snapshot_coverage_ok"),
    ("future_pv_issue", "bad_snapshot_origins"),
    ("current_slot_observation", "bad_snapshot_origins"),
    ("missing_event", "decision_coverage_ok"),
    ("duplicate_event", "decision_coverage_ok"),
    ("future_scenario", "future_scenario_paths"),
    ("cash", "cash_recalculation_max_yuan"),
    ("balance", "energy_balance_max_kwh"),
    ("state", "state_equation_max_kwh"),
    ("continuity", "continuity_max_kwh"),
    ("locked_contract", "locked_contract_max_kwh"),
    ("solver_failure", "lp_failures"),
])
def test_each_independent_audit_guard_rejects_fault(fault, metric):
    run, data, cfg = valid_run()
    if fault == "initial_soc": run.E_slot[:] += 1
    elif fault == "missing_snapshot": run.snapshots.pop()
    elif fault == "duplicate_snapshot": run.snapshots[-1] = copy.deepcopy(run.snapshots[0])
    elif fault == "wrong_snapshot_slot": run.snapshots[-1]["t0"] -= 1
    elif fault == "future_pv_issue": run.snapshots[0]["pv_issue_used"] = 1
    elif fault == "current_slot_observation": run.snapshots[0]["last_observed_slot"] = 0
    elif fault == "missing_event": run.plan_events.pop()
    elif fault == "duplicate_event": run.plan_events[-1] = copy.deepcopy(run.plan_events[0])
    elif fault == "future_scenario":
        run.plan_events[-1]["screen"] = {"scores": {"1": {"paths": [{"src_day": 1}]}}}
    elif fault == "cash": run.day_cost[0] += 1
    elif fault == "balance": run.h[0, 0] += 1
    elif fault == "state": run.E_slot[0, 1] += 1
    elif fault == "continuity": run.E_slot[1, :] += 1
    elif fault == "locked_contract": run.a_snap[0, 1, 0] += 1
    elif fault == "solver_failure": run.n_lp_fail = 1
    audit = runner.independent_audit(run, data, cfg)
    assert audit["pass"] is False
    assert not audit[metric] if metric.endswith("_ok") else audit[metric] > 0


@pytest.mark.parametrize("fault", ["nonfinite", "shape"])
def test_invalid_numeric_arrays_fail_before_statistics(fault):
    run, data, cfg = valid_run()
    if fault == "nonfinite": run.q[0, 0] = np.nan
    else: run.a = run.a[:, :-1]
    audit = runner.independent_audit(run, data, cfg)
    assert audit["pass"] is False
    assert ("q" if fault == "nonfinite" else "a") in audit["invalid_shape_or_nonfinite_arrays"]


class MemoryFile:
    def __init__(self, files, name=""):
        self.files, self.name = files, name
    def __truediv__(self, name):
        return MemoryFile(self.files, name)
    def is_file(self):
        return self.name in self.files


@pytest.mark.parametrize("fault", ["valid", "no_hashes", "empty_hashes", "missing_hash",
                                  "extra_hash", "missing_file", "changed_file", "wrong_signature",
                                  "failed_status", "io_error"])
def test_completed_record_cache_requires_all_verified_outputs(monkeypatch, fault):
    files = {name: "verified" for name in runner.REQUIRED_OUTPUTS}
    record = {"status": "PASS", "signature": "signature", "output_hashes": files.copy()}
    name = sorted(files)[0]
    if fault == "no_hashes": del record["output_hashes"]
    elif fault == "empty_hashes": record["output_hashes"] = {}
    elif fault == "missing_hash": del record["output_hashes"][name]
    elif fault == "extra_hash": record["output_hashes"]["unexpected"] = "verified"
    elif fault == "missing_file": del files[name]
    elif fault == "changed_file": files[name] = "corrupted"
    elif fault == "wrong_signature": record["signature"] = "stale"
    elif fault == "failed_status": record["status"] = "FAILED_AUDIT"
    def digest(path):
        if fault == "io_error": raise OSError("injected unreadable output")
        return path.files[path.name]
    monkeypatch.setattr(runner, "file_sha", digest)
    assert runner.completed_record_valid(record, MemoryFile(files), "signature") is (fault == "valid")


class GateAccepted(Exception):
    """Raised before any process pool or simulation is started."""


@pytest.mark.parametrize("fault", ["valid", "duplicate_ids", "short_evaluation", "stale_fingerprint",
                                  "summary_disagreement", "missing_output", "outside_results"])
def test_annual_baseline_gate_rejects_incomplete_or_stale_evidence(monkeypatch, fault):
    cases = runner.clean(runner.build_matrix())
    fake_sha = "test_fixture_sha"
    fingerprint = dict(python=platform.python_version(), numpy=np.__version__, scipy=runner.scipy.__version__,
                       pandas=runner.pd.__version__,
                       model_files={p.name: fake_sha for p in sorted((runner.ROOT / "code/model/src/cumcm_v11").glob("*.py"))},
                       runner_sha256=fake_sha,
                       archive_traces={k: fake_sha for k in ("q2", "q3", "q4-2", "q4-3")},
                       data_files={str(p.relative_to(runner.ROOT)).replace("\\", "/"): fake_sha
                                   for p in sorted((runner.ROOT / "data/raw").rglob("*.xlsx"))})
    fp_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    records = [dict(case_id=c["case_id"], status="PASS", complete_annual=True, evaluation_days=334,
                    fingerprint_sha256=fp_hash, signature="fixture", path=f"results/experiments/guard_fixture/{c['case_id']}",
                    output_hashes={name: fake_sha for name in runner.REQUIRED_OUTPUTS})
               for c in cases if c["family"] == "baseline"]
    if fault == "duplicate_ids": records[-1]["case_id"] = records[0]["case_id"]
    elif fault == "short_evaluation": records[0]["evaluation_days"] = 333
    elif fault == "stale_fingerprint": records[0]["fingerprint_sha256"] = "stale"
    elif fault == "outside_results": records[0]["path"] = "experiments/guard_fixture/outside"
    texts = {runner.ROOT / "experiments/experiment_matrix.json": json.dumps(dict(fingerprint=fingerprint, cases=cases)),
             runner.OUT / "baseline_index.json": json.dumps(records)}
    for record in records:
        saved = copy.deepcopy(record)
        if fault == "summary_disagreement" and record is records[0]: saved["signature"] = "different"
        texts[runner.ROOT / record["path"] / "summary.json"] = json.dumps(saved)
    actual_read_text = Path.read_text
    actual_exists = Path.exists
    actual_is_file = Path.is_file
    def read_text(path, *args, **kwargs):
        return texts[path] if path in texts else actual_read_text(path, *args, **kwargs)
    def is_file(path):
        if "guard_fixture" in path.parts:
            return not (fault == "missing_output" and path.name == "trace.npz")
        return actual_is_file(path)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "exists", lambda path: path in texts or actual_exists(path))
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(runner, "file_sha", lambda path: fake_sha)
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--stage", "all"])
    def stop_before_pool(**kwargs): raise GateAccepted()
    monkeypatch.setattr(runner, "ProcessPoolExecutor", stop_before_pool)
    if fault == "valid":
        with pytest.raises(GateAccepted): runner.main()
    else:
        with pytest.raises(SystemExit, match="Baseline regression gate failed"):
            runner.main()


if __name__ == "__main__":
    class Reporter:
        def __init__(self): self.results = []
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed:
                self.results.append(dict(case=report.nodeid, phase=report.when,
                                         outcome=report.outcome, seconds=report.duration,
                                         detail=str(report.longrepr) if report.failed else None))
    reporter = Reporter()
    started = datetime.now(timezone.utc).isoformat()
    exit_code = int(pytest.main([str(Path(__file__).resolve()), "-q", "-p", "no:cacheprovider"], plugins=[reporter]))
    source = Path(runner.__file__).resolve()
    report = dict(status="PASS" if exit_code == 0 else "FAILED", pytest_exit_code=exit_code,
                  started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                  command="python -B -X utf8 experiments/test_runner_guards.py",
                  pytest_arguments=[str(Path(__file__).resolve()), "-q", "-p", "no:cacheprovider"],
                  python=sys.version, runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Synthetic failure injection for audits, cached artifacts and annual baseline gate; no annual model rerun.",
                  cases=reporter.results, passed=sum(r["outcome"] == "passed" for r in reporter.results),
                  failed=sum(r["outcome"] == "failed" for r in reporter.results))
    output = runner.ROOT / "results/experiments/runner_guard_tests.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GUARD_REPORT {output}")
    raise SystemExit(exit_code)
