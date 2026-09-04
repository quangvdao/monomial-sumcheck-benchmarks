#!/usr/bin/env python3
"""Collect independent Criterion process runs with a machine manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
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


def load_snapshot() -> dict[str, object]:
    load1, load5, load15 = os.getloadavg()
    logical_cpus = os.cpu_count() or 1
    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "load_average_1m": load1,
        "load_average_5m": load5,
        "load_average_15m": load15,
        "logical_cpus": logical_cpus,
        "load_1m_per_logical_cpu": load1 / logical_cpus,
    }


def machine_manifest(
    commands: list[list[str]], repetitions: int, initial_load: dict[str, object]
) -> dict[str, object]:
    cpu = ""
    if platform.system() == "Darwin":
        cpu = capture(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif shutil.which("lscpu"):
        cpu = capture(["lscpu"])

    return {
        "schema": 2,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commands": commands,
        "repetitions": repetitions,
        "pairing": "one fresh cargo process per case; cyclically balanced case order",
        "initial_load": initial_load,
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
        description="Run Criterion cases in independent processes and preserve raw samples."
    )
    parser.add_argument("--suite", required=True, choices=["field_ops", "binding", "lookup_tables", "sumcheck"])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--filter",
        help="Criterion filter or regular expression for an exploratory single-process run",
    )
    selection.add_argument(
        "--case",
        action="append",
        help="exact Criterion benchmark ID; repeat to collect paired cases in separate processes",
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--label", required=True, help="Short output-directory label")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow an uncommitted source tree (for smoke tests only)",
    )
    parser.add_argument(
        "--max-load-per-cpu",
        type=float,
        default=0.25,
        help="maximum one-minute load average per logical CPU (default: 0.25)",
    )
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="run above the load threshold (for smoke tests only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.max_load_per_cpu <= 0:
        raise SystemExit("--max-load-per-cpu must be positive")
    dirty_status = capture(["git", "status", "--short"])
    if dirty_status and not args.allow_dirty:
        raise SystemExit(
            "refusing to collect paper data from a dirty tree; commit the benchmark "
            "revision or pass --allow-dirty for a non-publishable smoke test"
        )
    initial_load = load_snapshot()
    if (
        float(initial_load["load_1m_per_logical_cpu"]) > args.max_load_per_cpu
        and not args.allow_busy
    ):
        raise SystemExit(
            "refusing to collect paper data on a busy host: one-minute load per "
            f"logical CPU is {initial_load['load_1m_per_logical_cpu']:.2f}, above "
            f"the {args.max_load_per_cpu:.2f} threshold; wait for the host to become "
            "idle or pass --allow-busy for a non-publishable smoke test"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_root / f"{timestamp}-{args.label}"
    output.mkdir(parents=True, exist_ok=False)

    if args.case:
        if len(set(args.case)) != len(args.case):
            raise SystemExit("--case values must be unique")
        filters = [f"^{re.escape(case)}$" for case in args.case]
    else:
        filters = [args.filter]

    commands = [
        ["cargo", "bench", "--bench", args.suite, "--", criterion_filter, "--noplot"]
        for criterion_filter in filters
    ]
    (output / "manifest.json").write_text(
        json.dumps(machine_manifest(commands, args.repetitions, initial_load), indent=2)
        + "\n"
    )

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target_dir.is_absolute():
        target_dir = ROOT / target_dir
    criterion_root = target_dir / "criterion"

    for index in range(1, args.repetitions + 1):
        run_dir = output / f"run-{index:02d}"
        run_dir.mkdir()
        offset = (index - 1) % len(commands)
        ordered_commands = commands[offset:] + commands[:offset]
        (run_dir / "command-order.json").write_text(
            json.dumps(ordered_commands, indent=2) + "\n"
        )

        copied: list[str] = []
        load_records: list[dict[str, object]] = []
        for case_index, command in enumerate(ordered_commands, start=1):
            before_load = load_snapshot()
            if (
                float(before_load["load_1m_per_logical_cpu"])
                > args.max_load_per_cpu
                and not args.allow_busy
            ):
                (run_dir / "FAILED").write_text(
                    "host load exceeded the configured threshold before "
                    f"case {case_index}\n"
                )
                (run_dir / "system-load.json").write_text(
                    json.dumps(load_records + [{"before": before_load}], indent=2)
                    + "\n"
                )
                print(
                    f"host became busy before repetition {index}, case {case_index}; "
                    f"see {run_dir}",
                    file=sys.stderr,
                )
                return 3
            started_ns = time.time_ns()
            print(
                f"[{index}/{args.repetitions}, case {case_index}/{len(ordered_commands)}] "
                f"{' '.join(command)}",
                flush=True,
            )
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            after_load = load_snapshot()
            load_records.append(
                {
                    "case_index": case_index,
                    "command": command,
                    "before": before_load,
                    "after": after_load,
                }
            )
            (run_dir / "system-load.json").write_text(
                json.dumps(load_records, indent=2) + "\n"
            )
            log_name = (
                "cargo-bench.log"
                if len(ordered_commands) == 1
                else f"cargo-bench-{case_index:02d}.log"
            )
            (run_dir / log_name).write_text(result.stdout)
            if result.returncode != 0:
                (run_dir / "FAILED").write_text(
                    f"case {case_index} exit code {result.returncode}\n"
                )
                print(
                    f"benchmark repetition {index}, case {case_index} failed; see {run_dir}",
                    file=sys.stderr,
                )
                return result.returncode

            fresh = copy_fresh_criterion_results(criterion_root, run_dir, started_ns)
            if not fresh:
                print(
                    f"no fresh Criterion estimates found after repetition {index}, "
                    f"case {case_index}; check {run_dir / log_name}",
                    file=sys.stderr,
                )
                return 2
            copied.extend(fresh)

        (run_dir / "benchmarks.json").write_text(
            json.dumps(sorted(set(copied)), indent=2) + "\n"
        )

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
