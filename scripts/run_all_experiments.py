#!/usr/bin/env python3
"""
Run all experiment_run.py files one-by-one and log their output.

This script:
 - searches for `experiment_run.py` under `results/` and `results_full_history/` directories
 - runs each script sequentially using the same Python interpreter
 - writes logs to `run_logs/<path>/experiment_run.log`
 - can be configured to `--dry-run` or to `--continue-on-error`

This avoids running them in parallel (some experiments are heavy) and captures output for debugging.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def find_experiments(root: Path) -> List[Path]:
    patterns = ["results/**/experiment_run.py", "results_full_history/**/experiment_run.py"]
    files = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    # Deduplicate
    unique = sorted(set(files))
    return unique


def run_experiment(script: Path, python_exe: str, env: dict, log_root: Path, dry_run: bool = False) -> int:
    rel = script.relative_to(Path.cwd()) if script.is_relative_to(Path.cwd()) else script
    log_path = log_root / rel.parent
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / (script.stem + ".log")

    if dry_run:
        print(f"DRY-RUN: would run {script} and write to {log_file}")
        return 0

    print(f"Running {script} -> {log_file}")

    start = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        # Use Popen to stream output to console and log
        proc = subprocess.Popen([python_exe, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()), text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            # echo to stdout and write to file
            print(line, end="")
            f.write(line)
            f.flush()
        rc = proc.wait()

    elapsed = time.time() - start
    print(f"Done: {script} (exit {rc}) in {elapsed:.1f}s. Log: {log_file}")
    return rc


def main():
    parser = argparse.ArgumentParser(description="Run all experiment_run.py scripts sequentially")
    parser.add_argument("--dry-run", action="store_true", help="List experiments but do not execute them")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running next experiments on non-zero return codes")
    parser.add_argument("--disable-wandb", action="store_true", help="Set WANDB_MODE=disabled in child processes")
    parser.add_argument("--log-root", default="run_logs", help="Directory to write experiment logs")
    parser.add_argument("--interpreter", default=sys.executable, help="Python interpreter to use (default: current)")
    args = parser.parse_args()

    root = Path.cwd()
    experiments = find_experiments(root)
    if not experiments:
        print("No experiment_run.py found.")
        return 1

    print(f"Found {len(experiments)} experiments to run")

    env = os.environ.copy()
    if args.disable_wandb:
        env["WANDB_MODE"] = "disabled"

    failures = []
    for ex in experiments:
        rc = run_experiment(ex, args.interpreter, env, Path(args.log_root), dry_run=args.dry_run)
        if rc != 0:
            failures.append((ex, rc))
            if not args.continue_on_error:
                print(f"Stopping on first failure: {ex} (exit {rc})")
                break

    print("\nSummary:")
    print(f"Total experiments discovered: {len(experiments)}")
    print(f"Failures: {len(failures)}")
    for ex, rc in failures:
        print(f"  - {ex} -> exit {rc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
