#!/usr/bin/env python3
"""Collect independent Criterion process runs with a machine manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def capture(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def machine_manifest(command: list[str], repetitions: int) -> dict[str, object]:
    cpu = ""
    if platform.system() == "Darwin":
        cpu = capture(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif shutil.which("lscpu"):
        cpu = capture(["lscpu"])

    return {
        "schema": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "repetitions": repetitions,
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "rustc": capture(["rustc", "-vV"]),
        "cargo": capture(["cargo", "-V"]),
        "python": sys.version,
        "platform": platform.platform(),
        "uname": capture(["uname", "-a"]),
        "os_version": capture(["sw_vers"]) if platform.system() == "Darwin" else "",
        "cpu": cpu,
        "cargo_target_dir": os.environ.get("CARGO_TARGET_DIR", "target"),
        "rustflags": os.environ.get("RUSTFLAGS", ""),
    }


def copy_fresh_criterion_results(
    criterion_root: Path, destination: Path, started_ns: int
) -> list[str]:
    copied: list[str] = []
    for estimates in criterion_root.glob("**/new/estimates.json"):
        if estimates.stat().st_mtime_ns < started_ns:
            continue
        source_new = estimates.parent
        relative_new = source_new.relative_to(criterion_root)
        target_new = destination / "criterion" / relative_new
        shutil.copytree(source_new, target_new, dirs_exist_ok=False)
        copied.append(str(relative_new.parent))
    return sorted(copied)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Criterion suite in independent processes and preserve raw samples."
    )
    parser.add_argument("--suite", required=True, choices=["field_ops", "binding", "lookup_tables", "sumcheck"])
    parser.add_argument("--filter", required=True, help="Criterion benchmark filter or regular expression")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--label", required=True, help="Short output-directory label")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow an uncommitted source tree (for smoke tests only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    dirty_status = capture(["git", "status", "--short"])
    if dirty_status and not args.allow_dirty:
        raise SystemExit(
            "refusing to collect paper data from a dirty tree; commit the benchmark "
            "revision or pass --allow-dirty for a non-publishable smoke test"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_root / f"{timestamp}-{args.label}"
    output.mkdir(parents=True, exist_ok=False)

    command = [
        "cargo",
        "bench",
        "--bench",
        args.suite,
        "--",
        args.filter,
        "--noplot",
    ]
    (output / "manifest.json").write_text(
        json.dumps(machine_manifest(command, args.repetitions), indent=2) + "\n"
    )

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target_dir.is_absolute():
        target_dir = ROOT / target_dir
    criterion_root = target_dir / "criterion"

    for index in range(1, args.repetitions + 1):
        run_dir = output / f"run-{index:02d}"
        run_dir.mkdir()
        started_ns = time.time_ns()
        print(f"[{index}/{args.repetitions}] {' '.join(command)}", flush=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (run_dir / "cargo-bench.log").write_text(result.stdout)
        if result.returncode != 0:
            (run_dir / "FAILED").write_text(f"exit code {result.returncode}\n")
            print(f"benchmark process {index} failed; see {run_dir}", file=sys.stderr)
            return result.returncode

        copied = copy_fresh_criterion_results(criterion_root, run_dir, started_ns)
        (run_dir / "benchmarks.json").write_text(json.dumps(copied, indent=2) + "\n")
        if not copied:
            print(
                f"no fresh Criterion estimates found after process {index}; "
                f"check the filter and {run_dir / 'cargo-bench.log'}",
                file=sys.stderr,
            )
            return 2

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
