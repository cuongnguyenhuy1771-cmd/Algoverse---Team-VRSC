# Reward-Hacking Propagation: Directly Seeded Patient Zero + Repeated Examples

A controlled multi-agent experiment testing whether reward-hacking behavior
**propagates** from a deliberately seeded "Patient Zero" agent to susceptible
peer agents that never receive any private reward-hacking instructions
themselves — and how that propagation varies across network topology,
model mix, and sycophancy level.

This repository is a from-notebook conversion: the original artifact was a
single Jupyter notebook meant to be run top to bottom in one kernel. It has
been split into one file per pipeline stage for readability, with an
orchestrator script that reproduces the notebook's execution semantics
exactly (see [How it's organized](#how-its-organized) below).

Full experimental design, definitions, and validity protections are in
[`docs/experiment_design.md`](docs/experiment_design.md). A description of
every output file is in [`docs/expected_outputs.md`](docs/expected_outputs.md).

## What this tests

- **RQ1** — Does reward-hacking behavior propagate to susceptible peers when
  a deliberately reward-hacking source agent (Patient Zero) participates in
  the collaboration network, relative to a matched honest-seed network?
- **RQ2** — How does that seed-manipulation treatment effect vary across
  five network topologies, homogeneous vs. heterogeneous model teams, and
  low/medium/high sycophancy?
- **RQ3** — Conditional on a realized Patient-Zero hack plus traceable
  seed-linked peer adoption in Stage 1, does the selected peer keep showing
  manipulation signals on fresh tasks, under own-history vs. reset
  conditions?

Patient Zero is seeded with two *private* signals (repeated worked examples
of reward hacking + a direct instruction to game the evaluator on trigger
tickets) that susceptible peers never see directly — peers can only observe
whatever Patient Zero actually publishes to the shared network transcript.

## How it's organized

```
pipeline/
  01_environment_setup.py        Install deps, locate + patch your ImpossibleBench clone
  02_preflight_checks.py         API key + live model + budget checks
  03_experiment_library.py       Config, topology, prompts, solver, tasks
  04_stage0_seed_construction.py Build + validate the seed manipulation
  05_smoke_test.py               One fixed engineering/routing smoke test
  06_stage1_main_sweep.py        Full RQ1/RQ2 factorial sweep
  07_stage2_rq3_carryover.py     RQ3 self-history vs. reset carryover
  08_final_analysis.py           Integrity checks, tables, figures
run_pipeline.py                  Orchestrator — runs the stages in order
docs/
  experiment_design.md           Full design, operational definitions, guardrails
  expected_outputs.md            What each output file contains
requirements.txt
.env.example
```

**Important:** these stage files are not independent, self-contained
modules — they're the original notebook's cells, one per file, and later
stages reference constants and helper functions defined by earlier stages
(exactly as later cells in a notebook reference names from earlier cells).
Run them with `run_pipeline.py`, which executes each file in order inside
one shared namespace, or open them in Jupyter/VS Code and run them as cells
in the same order (01 → 08). Do not `python pipeline/06_stage1_main_sweep.py`
on its own — it will fail with `NameError`s for names defined upstream.

## Prerequisites

- Python 3.10+
- Git
- An [OpenRouter](https://openrouter.ai/) API key with credit — set
  `OPENROUTER_API_KEY`, or the preflight step will prompt for it interactively
- Internet access (package installs, model calls, dataset downloads)
- ~$90 of OpenRouter credit for a full run (see [Budget policy](#budget-policy))

## Installation

Two repos are involved: this one, and the official
[ImpossibleBench](https://github.com/safety-research/impossiblebench)
implementation it evaluates against (built on Inspect AI). ImpossibleBench
isn't published on PyPI, so it's cloned and installed in editable mode
rather than pip-installed from `requirements.txt`:

```bash
# 1. Clone this repo
git clone <this-repo-url>
cd <this-repo>

# 2. Clone ImpossibleBench alongside it and install it in editable mode
#    (this is the official repo's own documented install method)
git clone https://github.com/safety-research/impossiblebench.git
cd impossiblebench && pip install -e . && cd ..

# 3. Install this repo's own pinned dependencies
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENROUTER_API_KEY, then:
export $(grep -v '^#' .env | xargs)   # or use a tool like python-dotenv/direnv
```

After step 2, open `pipeline/01_environment_setup.py` and paste the commit
you checked out (`git -C impossiblebench rev-parse HEAD`) into
`IMPOSSIBLEBENCH_REPO_COMMIT` near the top — it's recorded in the run
metadata for reproducibility. Running the pipeline (below) will find your
clone automatically through Python's normal import machinery and apply one
small compatibility patch to it; it never re-downloads or overwrites your
clone. A virtual environment isn't required — use one if you'd like isolated
dependencies, but `pip install` alone is enough to run this repo.

## Running the experiment

The stages are meant to be run — and reviewed — in order, not fired off
unattended, since each of Stage 0 / the smoke test / Stage 1 gates the next:

```bash
# Stage 0: build and validate the seed manipulation (~$2)
python run_pipeline.py --until 04
# Wait for: "STAGE 0 COMPLETE — SEED MANIPULATION READY = True"

# Paid smoke test: engineering/routing check, no peer-adoption requirement (~$2)
python run_pipeline.py --from 05 --until 05

# Stage 1: full RQ1/RQ2 sweep (~$78 cap)
python run_pipeline.py --from 06 --until 06
# Wait for: "READY FOR FULL RUN" / Stage-1 completion

# Stage 2 (RQ3) + final analysis (~$8 cap)
python run_pipeline.py --from 07
```

Because each `run_pipeline.py` invocation above starts a fresh Python
process, results that need to persist between invocations (Stage 0's
checkpoint, Stage 1's `eval_set` cache) are read from and written to disk
under `OUTPUT_DIR`, exactly as they were in the original notebook — resuming
is safe and expected. Alternatively, run all eight stages in a single
`python run_pipeline.py` call, or paste the files into Jupyter cells in
order (01 → 08) if you'd rather step through interactively.

## Budget policy

The whole run is capped at a **$90 total ceiling**, split as:

| Phase | Cap |
|---|---|
| Stage 0 (seed examples + held-out validation) | $2 |
| Smoke test (fixed/resumable engineering check) | $2 |
| Stage 1 (main RQ1/RQ2 sweep) | $78 |
| Stage 2 (RQ3) | $8 |

A live-price planner in Stage 1 chooses the largest uniform per-cell
`LIMIT` that fits the remaining Stage-1 budget, and runtime provider-balance
guards run throughout.

## Outputs

Everything lands under `~/rh_contagion_final/results/<RUN_LABEL>/`
(`OUTPUT_DIR`, printed by the setup stage), including `run_plan.json`,
`integrity_report.json`, `contagion_results.csv`, `contagion_funnel.csv`,
`rq1_primary_paired_contrasts.csv`, `rq2_*_seed_manipulation_effects.csv`,
RQ3 eligibility/results/carryover tables, `budget_audit.json`,
`manuscript_results.json`, and figures. See
[`docs/expected_outputs.md`](docs/expected_outputs.md) for the full list
and what each file means.

## Local execution notes / caveats

- Model-generated LiveCodeBench candidate code is executed locally in a
  short-lived child Python process with provider credentials stripped,
  isolated mode, no shell, and a hard timeout — this is a precaution, **not
  a full security sandbox**.
- SWE-bench Verified is used in RQ3 only as a text-level behavioral audit,
  not official repository execution / pass@1.
- The primary causal quantity is the **paired seed-manipulation effect**
  (deliberately reward-hacking seed network minus matched honest-seed
  network) — individual-level "exposure → adoption" numbers are descriptive,
  not evidence of causation on their own. See
  [`docs/experiment_design.md`](docs/experiment_design.md) for the complete
  set of interpretation and validity guardrails.

## Provenance

Converted from a single-notebook artifact (`latest.ipynb`, `prompt_v3_pinned`
revision) into this multi-file layout for repository use. Pinned dataset
revisions: ImpossibleBench/LiveCodeBench commit `98650ffc...`, SWE-bench
Verified revision `91aa3ed5...` (see `docs/experiment_design.md` for
details). The source notebook originally pinned ImpossibleBench itself to
repo commit `061dc3dc...`; this repo instead records whichever commit you
clone in `IMPOSSIBLEBENCH_REPO_COMMIT` (see Installation above).
