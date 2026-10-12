"""Explicit opt-in recomputation; official archived results are never the target."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser(description="主动运行作者原模型；默认写入新目录。")
    p.add_argument("--confirm-recompute", action="store_true", required=True)
    p.add_argument("--only", choices=["q1", "q2", "q3", "q4", "all", "ablation"], default="all")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--max-days", type=int, default=365)
    a = p.parse_args()
    if a.jobs < 1 or not 1 <= a.max_days <= 365:
        p.error("jobs必须为正整数，max-days须在1至365之间。")
    env = os.environ.copy()
    src = str(ROOT / "code/model/src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", "from cumcm_v11 import main; main()",
        "--only", a.only, "--jobs", str(a.jobs), "--max-days", str(a.max_days),
        "--out", str(ROOT / "results/recomputed")], cwd=ROOT, env=env, check=True)
if __name__ == "__main__":
    main()
