#!/usr/bin/env python3
"""
Run the Python scenario generator repeatedly with stable scenario names.

Examples:
  python3 src/scripts/run_generation_batch.py --count 100
  python3 src/scripts/run_generation_batch.py --count 100 --random-seed 1234 --prefix train
  python3 src/scripts/run_generation_batch.py --count 100 --min-drones 5 --max-drones 10 -- --n-obstacles 100
  python3 src/scripts/run_generation_batch.py --generator src/scripts/gen_scenario.py -- --n-drones 8
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Batch-run the Python scenario generator with deterministic output names.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=100, help="number of scenarios to generate")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="optional seed for the batch runner itself; use this to reproduce the sampled per-run seeds",
    )
    parser.add_argument(
        "--min-drones",
        type=int,
        default=5,
        help="minimum drone count for batch generation",
    )
    parser.add_argument(
        "--max-drones",
        type=int,
        default=10,
        help="maximum drone count for batch generation",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="src/scripts/gen_scenario_decentralized.py",
        help="generator entrypoint to run",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/scenarios_batch",
        help="root directory for the generated scenario folders",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="scene",
        help="scenario folder prefix; final names look like <prefix>_000_seed0000",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop immediately if one scenario generation fails",
    )
    return parser.parse_known_args()


def main() -> int:
    args, extra_args = parse_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.min_drones <= 0 or args.max_drones < args.min_drones:
        raise ValueError("Require 0 < min_drones <= max_drones")

    repo_root = Path(__file__).resolve().parents[2]
    generator_path = Path(args.generator)
    if not generator_path.is_absolute():
        generator_path = repo_root / generator_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    failures = 0
    drone_span = args.max_drones - args.min_drones + 1
    rng = random.Random(args.random_seed)
    used_seeds: set[int] = set()

    for idx in range(args.count):
        seed = rng.randrange(0, 1_000_000_000)
        while seed in used_seeds:
            seed = rng.randrange(0, 1_000_000_000)
        used_seeds.add(seed)
        n_drones = args.min_drones + (idx % drone_span)
        scenario_name = f"{args.prefix}_{idx:03d}_d{n_drones}_seed{seed:09d}"
        scenario_dir = output_root / scenario_name
        cmd = [
            sys.executable,
            str(generator_path),
            "--n-drones",
            str(n_drones),
            "--seed",
            str(seed),
            "--scenario-index",
            str(idx),
            "--scenario-name",
            scenario_name,
            "--output-dir",
            str(scenario_dir),
            *extra_args,
        ]

        print(f"[{idx + 1}/{args.count}] {scenario_name}")
        result = subprocess.run(cmd, cwd=repo_root)
        manifest.append(
            {
                "index": idx,
                "n_drones": n_drones,
                "seed": seed,
                "scenario_name": scenario_name,
                "output_dir": str(scenario_dir),
                "returncode": result.returncode,
            }
        )
        if result.returncode != 0:
            failures += 1
            print(f"  failed with return code {result.returncode}")
            if args.stop_on_error:
                break

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count_requested": args.count,
                "count_completed": len(manifest),
                "batch_random_seed": args.random_seed,
                "generator": str(generator_path),
                "extra_args": extra_args,
                "entries": manifest,
            },
            f,
            indent=2,
        )

    print(f"Manifest written to {manifest_path}")
    if failures:
        print(f"Completed with {failures} failures")
        return 1
    print("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
