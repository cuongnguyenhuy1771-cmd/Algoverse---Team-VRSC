#!/usr/bin/env python3
"""
Run the full reward-hacking-contagion experiment end to end.

The original artifact was a single Jupyter notebook meant to be run
top-to-bottom in one kernel: later cells rely on globals (config
constants, helper functions — including underscore-prefixed ones,
loaded model clients, etc.) defined by earlier cells. Splitting the
notebook into pipeline/01_*.py .. pipeline/08_*.py keeps each stage's
purpose readable in its own file, but does not change that
dependency: these are still stage *scripts*, not self-contained
importable modules.

This runner reproduces the notebook's execution model faithfully: it
executes each pipeline/*.py file in order inside one shared
namespace, so stage N sees every name defined by stages 1..N-1,
exactly as it would inside one Jupyter kernel.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python run_pipeline.py                 # run every stage in order
    python run_pipeline.py --until 04      # stop after Stage 0 seed construction
    python run_pipeline.py --from 06       # resume from Stage 1 onward

Stopping partway is expected and normal: Stage 0 (04), the smoke test
(05), and Stage 1 (06) are each meant to be reviewed before you move
on to the next one — see the repo README and docs/expected_outputs.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent / "pipeline"


def discover_stages() -> list[Path]:
    stages = sorted(PIPELINE_DIR.glob("[0-9][0-9]_*.py"))
    if not stages:
        raise SystemExit(f"No pipeline stages found under {PIPELINE_DIR}")
    return stages


def stage_id(path: Path) -> str:
    return path.name[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_id", default=None, help="two-digit stage id to start from, e.g. 06")
    parser.add_argument("--until", dest="until_id", default=None, help="two-digit stage id to stop after, e.g. 04")
    args = parser.parse_args()

    stages = discover_stages()
    if args.from_id:
        stages = [s for s in stages if stage_id(s) >= args.from_id]
    if args.until_id:
        stages = [s for s in stages if stage_id(s) <= args.until_id]
    if not stages:
        raise SystemExit("No stages selected — check --from/--until values.")

    namespace: dict = {"__name__": "__main__"}
    for stage_path in stages:
        print(f"\n{'=' * 78}\nRunning {stage_path.relative_to(Path(__file__).parent)}\n{'=' * 78}\n")
        code = compile(stage_path.read_text(encoding="utf-8"), str(stage_path), "exec")
        try:
            exec(code, namespace)
        except Exception:
            print(f"\nPipeline stopped while running {stage_path.name}.", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
