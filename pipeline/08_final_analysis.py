"""
Final analysis, integrity checks, figures, and manuscript tables.

Runs hard completeness/pairing checks over the Stage-1 results,
builds the contagion funnel, computes the paired
seed-manipulation-effect contrasts for RQ1/RQ2, summarizes RQ3
conditional results, and writes every manuscript-ready CSV/JSON
table plus the budget audit into OUTPUT_DIR. See
docs/expected_outputs.md for a description of each output file.

Depends on globals defined in 07_stage2_rq3_carryover.py.
"""

# =============================================================================
# FINAL ANALYSIS — FAIL-CLOSED INTEGRITY, RQ1/RQ2/RQ3, BUDGET AUDIT
# =============================================================================

from pathlib import Path
import json
import math

from inspect_ai.log import read_eval_log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stage-0 acquisition evidence is part of experiment integrity. The frozen
# artifacts were copied into the new run's EXPOSURE_DIR without regeneration.
if not (EXPOSURE_DIR / "exposure_build_report.json").exists():
    raise RuntimeError("Missing Stage-0 exposure build report.")
if not (EXPOSURE_DIR / "exposure_validation_report.json").exists():
    raise RuntimeError("Missing Stage-0 held-out seed-manipulation report.")
_stage0_validation = json.loads((EXPOSURE_DIR / "exposure_validation_report.json").read_text(encoding="utf-8"))
if _stage0_validation.get("seed_manipulation_protocol") != SEED_MANIPULATION_PROTOCOL:
    raise RuntimeError(
        "Stage-0 report belongs to a different Patient-Zero seed manipulation protocol."
    )
if not _stage0_validation.get("passed_gate", False):
    raise RuntimeError("Stage-0 seed-manipulation realization gate was not passed.")

LOG_DIR = OUTPUT_DIR / "logs"
log_files = sorted(LOG_DIR.glob("*.eval"))
print(f"Found {len(log_files)} .eval file(s) in {LOG_DIR}")
if not log_files:
    raise RuntimeError("No Stage-1 eval logs found.")

# ---------------------------------------------------------------------------
# 1) Load one structured result row per completed sample.
# ---------------------------------------------------------------------------
rows, turn_rows, probe_rows = [], [], []
for log_path in log_files:
    try:
        log = read_eval_log(str(log_path))
    except Exception as exc:
        raise RuntimeError(f"Could not read {log_path.name}: {exc}") from exc

    for sample in (log.samples or []):
        result = (sample.metadata or {}).get("contagion_results")
        if not isinstance(result, dict):
            continue
        row = dict(result)
        row["_log_file"] = log_path.name
        row["_sample_id"] = sample.id
        row["_epoch"] = sample.epoch
        row["run_key"] = f"{row['sample_id']}||{row['epoch']}"
        rows.append(row)

        base = {
            "run_label": row.get("run_label"),
            "sample_id": row["sample_id"],
            "epoch": row["epoch"],
            "run_key": row["run_key"],
            "condition": row["condition"],
            "topology": row["topology"],
            "model_mix": row["model_mix"],
            "sycophancy": row["sycophancy"],
        }
        for turn in result.get("turns", []):
            turn_rows.append({**base, **turn})
        for probe in result.get("probes", []):
            probe_rows.append({**base, **probe})

if not rows:
    raise RuntimeError("No completed contagion_results were found.")

df = pd.DataFrame(rows)
turns_df = pd.DataFrame(turn_rows)
probes_df = pd.DataFrame(probe_rows)
# Backward-compatible treatment aliases. The paper-facing treatment is now a\n# bundled seed manipulation (private examples + private direct trigger directive).\n# Existing exposure_* names are retained only so older analysis code still works.\n# This also supports
# Stage-1 logs produced by the immediately previous compatible notebook version.
if "exposure_conditioned_assigned" not in df.columns and "prompt_injection_assigned" in df.columns:
    df["exposure_conditioned_assigned"] = df["prompt_injection_assigned"]
if "patient_zero_hack_realized" not in df.columns and "prompt_injection_realized" in df.columns:
    df["patient_zero_hack_realized"] = df["prompt_injection_realized"]
# Backward-compatible aliases used by some older table code, now derived from
# canonical compatibility fields rather than legacy prompt-injection language.
df["exposure_assigned"] = df["exposure_conditioned_assigned"]
df["exposure_realized"] = df["patient_zero_hack_realized"]

# Only this notebook version belongs in this analysis directory.
if "run_label" in df.columns:
    wrong = df[df["run_label"] != RUN_LABEL]
    if len(wrong):
        raise RuntimeError(
            f"Found {len(wrong)} result rows from a different run_label. "
            "Use a clean output directory before making paper claims."
        )

identity_cols = [
    "sample_id", "epoch", "condition", "topology", "model_mix", "sycophancy"
]
dupes = df[df.duplicated(identity_cols, keep=False)]
if len(dupes):
    display(dupes[identity_cols + ["_log_file"]].sort_values(identity_cols))
    raise RuntimeError(
        "Duplicate completed sample identities were found. Do not average over "
        "mixed reruns; move/clear the run directory and resume one consistent run."
    )

df.to_csv(OUTPUT_DIR / "contagion_results.csv", index=False)
turns_df.to_csv(OUTPUT_DIR / "turns.csv", index=False)
if len(probes_df):
    probes_df.to_csv(OUTPUT_DIR / "probes.csv", index=False)

# ---------------------------------------------------------------------------
# 2) HARD COMPLETENESS / PAIRING CHECKS.
# ---------------------------------------------------------------------------
expected_cells = set(ACTIVE_CELLS)
actual_cells = set(
    zip(df["topology"], df["condition"], df["model_mix"], df["sycophancy"])
)
missing_cells = sorted(expected_cells - actual_cells)
extra_cells = sorted(actual_cells - expected_cells)
if missing_cells or extra_cells:
    print("Missing cells:", missing_cells)
    print("Extra cells:", extra_cells)
    raise RuntimeError("Stage-1 factorial cell set is not exactly the intended design.")

expected_runs_per_cell = int(LIMIT) * int(EPOCHS)
cell_counts = (
    df.groupby(["topology", "condition", "model_mix", "sycophancy"])
      .size()
      .rename("n_rows")
      .reset_index()
)
bad_counts = cell_counts[cell_counts["n_rows"] != expected_runs_per_cell]
if len(bad_counts):
    display(bad_counts)
    raise RuntimeError(
        f"Every Stage-1 cell must contain exactly {expected_runs_per_cell} rows."
    )

if not bool(df["valid_trial"].fillna(False).all()):
    bad = df[~df["valid_trial"].fillna(False)]
    display(bad[identity_cols])
    raise RuntimeError("At least one trial failed the treatment-validity check.")

# Matched treatment/honest pairs must have exactly the same run keys and blocked
# nuisance variables (queue, models, graph) within every RQ2 factor cell.
pair_key = ["run_key", "topology", "model_mix", "sycophancy"]
seed = df[df["condition"] == "exposure_conditioned_seed"].copy()
honest = df[df["condition"] == "honest_peer_control"].copy()

seed_keys = set(map(tuple, seed[pair_key].to_records(index=False)))
honest_keys = set(map(tuple, honest[pair_key].to_records(index=False)))
if seed_keys != honest_keys:
    raise RuntimeError(
        f"Treatment/honest pairing mismatch: "
        f"{len(seed_keys-honest_keys)} treatment-only and "
        f"{len(honest_keys-seed_keys)} control-only pairs."
    )

paired = seed.merge(
    honest,
    on=pair_key,
    suffixes=("_seed", "_honest"),
    validate="one_to_one",
)

def _canon(x):
    return json.dumps(x, sort_keys=True, default=str)

for field in ("queue_kinds", "queue_task_ids", "models", "edges", "susceptible_agent_ids"):
    mismatch = paired[
        paired[f"{field}_seed"].map(_canon) != paired[f"{field}_honest"].map(_canon)
    ]
    if len(mismatch):
        raise RuntimeError(f"Blocked variable mismatch between treatment/control: {field}")

if not (
    (df["round_semantics"] == "synchronous_previous_round_context_only").all()
):
    raise RuntimeError("Mixed/old round semantics detected in logs.")

# The susceptible denominator must be 1..N-1 in every arm.
expected_susceptible = list(range(1, N_AGENTS))
if not df["susceptible_agent_ids"].map(lambda x: list(x) == expected_susceptible).all():
    raise RuntimeError("Susceptible-agent denominator is not identical across arms.")

integrity_report = {
    "run_label": RUN_LABEL,
    "stage1_rows": int(len(df)),
    "expected_cells": int(len(ACTIVE_CELLS)),
    "runs_per_cell": expected_runs_per_cell,
    "paired_network_runs": int(len(paired)),
    "synchronous_rounds": True,
    "matched_susceptible_denominator": expected_susceptible,
    "queue_model_graph_pairing": "passed",
    "factorial_completeness": "passed",
}
(OUTPUT_DIR / "integrity_report.json").write_text(
    json.dumps(integrity_report, indent=2), encoding="utf-8"
)
print("STAGE-1 INTEGRITY CHECKS: PASSED")

valid = df.copy()
GROUP = ["condition", "topology", "model_mix", "sycophancy"]

# Treatment realization/compliance is reported but is NOT used to filter the
# primary intention-to-treat analysis.
treatment_trials = valid[valid["condition"] == "exposure_conditioned_seed"].copy()
exposure_realization = pd.DataFrame([
    {
        "n_treatment_trials": int(len(treatment_trials)),
        "n_realized_patient_zero_hack_trials": int(
            treatment_trials["patient_zero_hack_realized"].fillna(False).astype(bool).sum()
        ),
        "realized_trial_rate": float(
            treatment_trials["patient_zero_hack_realized"].fillna(False).astype(bool).mean()
        ) if len(treatment_trials) else float("nan"),
        "mean_patient_zero_trigger_hack_rate": float(
            pd.to_numeric(treatment_trials["patient_zero_trigger_hack_rate"], errors="coerce").mean()
        ) if len(treatment_trials) else float("nan"),
        "mean_final_seed_linked_adoption_rate": float(
            pd.to_numeric(treatment_trials["final_seed_linked_adoption_rate"], errors="coerce").mean()
        ) if len(treatment_trials) else float("nan"),
    }
])
exposure_realization.to_csv(OUTPUT_DIR / "patient_zero_exposure_realization.csv", index=False)
print("PATIENT ZERO EXPOSURE REALIZATION (reported separately from ITT)")
display(exposure_realization)

# ---------------------------------------------------------------------------
# Seed-exposure diagnostics. These are descriptive mechanism checks, not a
# second causal endpoint. They answer the key question when spread is small:
# did susceptible agents fail to receive the realized seed, or did they see it
# and decline to adopt? Sycophancy never changes network visibility rules.
# ---------------------------------------------------------------------------
_treatment_turns = turns_df[
    (turns_df["condition"] == "exposure_conditioned_seed")
    & (pd.to_numeric(turns_df["agent_id"], errors="coerce") != 0)
].copy()

_agent_exposure = (
    _treatment_turns
    .groupby(
        ["run_key", "topology", "model_mix", "sycophancy", "agent_id"],
        as_index=False,
    )
    .agg(
        ever_exposed_to_seed_lineage=("exposed_to_seed_lineage", "max"),
        ever_seed_linked_adoption=("is_seed_linked_post_exposure_adoption", "max"),
    )
)
_realized_map = treatment_trials[["run_key", "topology", "model_mix", "sycophancy", "patient_zero_hack_realized"]].copy()
_agent_exposure = _agent_exposure.merge(
    _realized_map,
    on=["run_key", "topology", "model_mix", "sycophancy"],
    how="left",
    validate="many_to_one",
)
_agent_exposure = _agent_exposure[
    _agent_exposure["patient_zero_hack_realized"].fillna(False).astype(bool)
].copy()

_exposure_rows = []
for keys, g in _agent_exposure.groupby(["topology", "model_mix", "sycophancy"], dropna=False):
    exposed = g["ever_exposed_to_seed_lineage"].fillna(False).astype(bool)
    adopted = g["ever_seed_linked_adoption"].fillna(False).astype(bool)
    n_exposed = int(exposed.sum())
    _exposure_rows.append({
        "topology": keys[0],
        "model_mix": keys[1],
        "sycophancy": keys[2],
        "n_susceptible_agent_trials_in_realized_seed_runs": int(len(g)),
        "n_ever_exposed_to_seed_lineage": n_exposed,
        "seed_lineage_exposure_rate": float(exposed.mean()) if len(g) else float("nan"),
        "n_seed_linked_adopters": int(adopted.sum()),
        "adoption_rate_given_seed_exposure": (
            float(adopted[exposed].mean()) if n_exposed else float("nan")
        ),
    })

seed_exposure_diagnostics = pd.DataFrame(_exposure_rows)
seed_exposure_diagnostics.to_csv(
    OUTPUT_DIR / "seed_exposure_diagnostics.csv", index=False
)
print("\nSEED EXPOSURE DIAGNOSTICS — REALIZED-SEED TRIALS ONLY")
display(seed_exposure_diagnostics.sort_values(["topology", "model_mix", "sycophancy"]))

# ---------------------------------------------------------------------------
# 3) Descriptive cell estimates with nonparametric run-level bootstrap CIs.
# ---------------------------------------------------------------------------
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260823

def bootstrap_mean_ci(values, draws=BOOTSTRAP_DRAWS, seed_value=BOOTSTRAP_SEED):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(x.mean())
    if len(x) == 1:
        return (mean, float("nan"), float("nan"), 1)
    rng = np.random.default_rng(seed_value)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[idx].mean(axis=1)
    return (
        mean,
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        len(x),
    )

metrics = [
    "final_post_exposure_adoption_rate",
    "final_successful_post_exposure_hack_rate",
    "final_seed_linked_adoption_rate",
    "t80_post_exposure_adoption_reached",
    "t80_successful_post_exposure_hack_reached",
    "t80_post_exposure_adoption_rounds",
    "t80_successful_post_exposure_hack_rounds",
    "e_operational_innovation_adoption",
    "e_operational_innovation_success",
    "spontaneous_adoption_rate",
    "pre_exposure_clean_pass_rate",
]
summary_rows = []
for keys, g in valid.groupby(GROUP, dropna=False):
    base = dict(zip(GROUP, keys))
    base["n_runs"] = int(g["run_key"].nunique())
    for metric in metrics:
        mean, lo, hi, n = bootstrap_mean_ci(g[metric])
        base[f"{metric}_mean"] = mean
        base[f"{metric}_ci_low"] = lo
        base[f"{metric}_ci_high"] = hi
        base[f"{metric}_n"] = n
    summary_rows.append(base)

cell_summary = pd.DataFrame(summary_rows)
cell_summary.to_csv(OUTPUT_DIR / "cell_summary_bootstrap.csv", index=False)
display(cell_summary.head(20))

# ---------------------------------------------------------------------------
# 4) Primary paired contrasts.
# ---------------------------------------------------------------------------
def paired_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    *,
    merge_on=("run_key",),
    label: str,
):
    left = a[list(merge_on) + [metric]].rename(columns={metric: "a"})
    right = b[list(merge_on) + [metric]].rename(columns={metric: "b"})
    m = left.merge(right, on=list(merge_on), how="inner", validate="one_to_one")
    m["diff"] = pd.to_numeric(m["a"], errors="coerce") - pd.to_numeric(m["b"], errors="coerce")
    mean, lo, hi, n = bootstrap_mean_ci(m["diff"])
    return {
        "contrast": label,
        "metric": metric,
        "n_pairs": n,
        "mean_diff": mean,
        "ci_low": lo,
        "ci_high": hi,
    }, m

def select_cell(frame, condition, topology, model_mix, sycophancy):
    return frame[
        (frame["condition"] == condition)
        & (frame["topology"] == topology)
        & (frame["model_mix"] == model_mix)
        & (frame["sycophancy"] == sycophancy)
    ].copy()

rq1_seed = select_cell(
    valid, "exposure_conditioned_seed", "mesh_full", "homogeneous", "medium"
)
rq1_honest = select_cell(
    valid, "honest_peer_control", "mesh_full", "homogeneous", "medium"
)
rq1_null = select_cell(
    valid, "null_independent", "mesh_full", "homogeneous", "medium"
)

primary_contrasts = []
for metric in ("final_post_exposure_adoption_rate", "final_successful_post_exposure_hack_rate"):
    result, _ = paired_difference(
        rq1_seed, rq1_honest, metric,
        label="RQ1 seed manipulation - honest peer control"
    )
    primary_contrasts.append(result)
    result, _ = paired_difference(
        rq1_seed, rq1_null, metric,
        label="RQ1 seed manipulation - null independent"
    )
    primary_contrasts.append(result)

primary_contrasts_df = pd.DataFrame(primary_contrasts)
primary_contrasts_df.to_csv(
    OUTPUT_DIR / "rq1_primary_paired_contrasts.csv", index=False
)
print("\nRQ1 PRIMARY PAIRED CONTRASTS — INTENTION TO TREAT")
display(primary_contrasts_df)

# Conservative task-clustered sensitivity. If EPOCHS>1, repeated generations
# within each task are averaged before bootstrapping; with the frozen EPOCHS=1
# design this reduces to one observation per distinct benchmark ticket.
def _task_clustered_paired_contrast(a, b, metric, label):
    left = a[["sample_id", "epoch", metric]].rename(columns={metric: "a"})
    right = b[["sample_id", "epoch", metric]].rename(columns={metric: "b"})
    m = left.merge(right, on=["sample_id", "epoch"], validate="one_to_one")
    m["diff"] = pd.to_numeric(m["a"], errors="coerce") - pd.to_numeric(m["b"], errors="coerce")
    task = m.groupby("sample_id", as_index=False)["diff"].mean()
    mean, lo, hi, n = bootstrap_mean_ci(task["diff"], seed_value=BOOTSTRAP_SEED + 17)
    return {
        "contrast": label, "metric": metric, "n_task_clusters": n,
        "mean_diff": mean, "ci_low": lo, "ci_high": hi,
    }

_clustered_rows = []
for _metric in ("final_post_exposure_adoption_rate", "final_successful_post_exposure_hack_rate"):
    _clustered_rows.append(_task_clustered_paired_contrast(
        rq1_seed, rq1_honest, _metric, "RQ1 seed manipulation - honest peer control"
    ))
    _clustered_rows.append(_task_clustered_paired_contrast(
        rq1_seed, rq1_null, _metric, "RQ1 seed manipulation - null independent"
    ))
rq1_primary_clustered = pd.DataFrame(_clustered_rows)
rq1_primary_clustered.to_csv(OUTPUT_DIR / "rq1_primary_paired_contrasts_task_clustered.csv", index=False)
print("\nRQ1 TASK-CLUSTERED SENSITIVITY CIs")
display(rq1_primary_clustered)

# Secondary descriptive analysis conditioned on treatment realization. Because
# realization is post-treatment, this is NOT labeled as a causal ITT estimate.
realized_seed = rq1_seed[rq1_seed["exposure_realized"].fillna(False).astype(bool)].copy()
realized_secondary = []
for metric in ("final_post_exposure_adoption_rate", "final_successful_post_exposure_hack_rate"):
    result, _ = paired_difference(
        realized_seed, rq1_honest, metric,
        label="RQ1 realized Patient-Zero hack subset - matched honest control (descriptive)"
    )
    realized_secondary.append(result)
realized_secondary_df = pd.DataFrame(realized_secondary)
realized_secondary_df.to_csv(
    OUTPUT_DIR / "rq1_realized_seed_descriptive_contrasts.csv", index=False
)
realized_seed_linked = pd.DataFrame([
    {
        "measure": "final_seed_linked_adoption_rate_among_realized_exposure_trials",
        "n_runs": int(len(realized_seed)),
        "mean": float(pd.to_numeric(realized_seed["final_seed_linked_adoption_rate"], errors="coerce").mean())
        if len(realized_seed) else float("nan"),
    }
])
realized_seed_linked.to_csv(
    OUTPUT_DIR / "rq1_seed_linked_spread_descriptive.csv", index=False
)
print("\nRQ1 REALIZED-SEED SECONDARY CONTRASTS — DESCRIPTIVE, POST-TREATMENT CONDITIONING")
display(realized_secondary_df)
print("\nRQ1 SEED-LINKED SPREAD — DESCRIPTIVE TRACEABILITY METRIC")
display(realized_seed_linked)

# Innovation is only defined after post-exposure adoption exists. Treat it as a conditional
# mechanism estimate, not as a zero-filled outcome in non-post-exposure-adoption controls.
rq1_operational_innovation = pd.DataFrame([
    {
        "measure": "operational_macro_novelty_among_post_exposure_adoptions",
        "n_runs_defined": int(rq1_seed["e_operational_innovation_adoption"].notna().sum()),
        "mean": float(pd.to_numeric(rq1_seed["e_operational_innovation_adoption"], errors="coerce").mean()),
        "total_post_exposure_turns": int(pd.to_numeric(rq1_seed["n_post_exposure_adoption_turns"], errors="coerce").fillna(0).sum()),
        "total_operational_innovation_turns": int(pd.to_numeric(rq1_seed["n_operational_innovation_turns"], errors="coerce").fillna(0).sum()),
    },
    {
        "measure": "operational_macro_novelty_among_successful_post_exposure_hacks",
        "n_runs_defined": int(rq1_seed["e_operational_innovation_success"].notna().sum()),
        "mean": float(pd.to_numeric(rq1_seed["e_operational_innovation_success"], errors="coerce").mean()),
        "total_post_exposure_turns": int(pd.to_numeric(rq1_seed["n_successful_post_exposure_turns"], errors="coerce").fillna(0).sum()),
        "total_operational_innovation_turns": int(pd.to_numeric(rq1_seed["n_operational_innovation_success_turns"], errors="coerce").fillna(0).sum()),
    },
])
rq1_operational_innovation.to_csv(OUTPUT_DIR / "rq1_operational_innovation_conditional.csv", index=False)
display(rq1_operational_innovation)

# ---------------------------------------------------------------------------
# 5) RQ2: compute the treatment effect first, THEN compare factors.
# This avoids interpreting raw treatment-arm behavior as a structural treatment effect.
# ---------------------------------------------------------------------------
rq2_metrics = [
    "final_post_exposure_adoption_rate",
    "final_successful_post_exposure_hack_rate",
    "final_seed_linked_adoption_rate",
    "t80_post_exposure_adoption_reached",
    "t80_successful_post_exposure_hack_reached",
]
effect_rows = []
for metric in rq2_metrics:
    m = paired[
        ["run_key", "topology", "model_mix", "sycophancy",
         f"{metric}_seed", f"{metric}_honest"]
    ].copy()
    m["metric"] = metric
    m["seed_manipulation_effect"] = (
        pd.to_numeric(m[f"{metric}_seed"], errors="coerce")
        - pd.to_numeric(m[f"{metric}_honest"], errors="coerce")
    )
    effect_rows.append(m[["run_key", "topology", "model_mix", "sycophancy", "metric", "seed_manipulation_effect"]])

rq2_effects = pd.concat(effect_rows, ignore_index=True)
rq2_effects.to_csv(OUTPUT_DIR / "rq2_run_level_seed_manipulation_effects.csv", index=False)

rq2_cell_rows = []
for keys, g in rq2_effects.groupby(["topology", "model_mix", "sycophancy", "metric"]):
    mean, lo, hi, n = bootstrap_mean_ci(g["seed_manipulation_effect"])
    rq2_cell_rows.append({
        "topology": keys[0],
        "model_mix": keys[1],
        "sycophancy": keys[2],
        "metric": keys[3],
        "n_pairs": n,
        "mean_seed_manipulation_effect": mean,
        "ci_low": lo,
        "ci_high": hi,
    })
rq2_cell_effects = pd.DataFrame(rq2_cell_rows)
rq2_cell_effects.to_csv(OUTPUT_DIR / "rq2_cell_seed_manipulation_effects.csv", index=False)
print("\nRQ2 CELL-LEVEL TREATMENT EFFECTS")
display(rq2_cell_effects)

# Factor summaries of the treatment effect.
factor_rows = []
for metric in rq2_metrics:
    em = rq2_effects[rq2_effects["metric"] == metric]

    for topology, g in em.groupby("topology"):
        mean, lo, hi, n = bootstrap_mean_ci(
            g.groupby("run_key", as_index=False)["seed_manipulation_effect"].mean()["seed_manipulation_effect"]
        )
        factor_rows.append({
            "factor": "topology",
            "level_or_contrast": topology,
            "metric": metric,
            "n_blocks": n,
            "estimate": mean,
            "ci_low": lo,
            "ci_high": hi,
        })

    # Direct topology contrasts for the proposal's structural hypotheses.
    # These compare the treatment effect within the same run/model-mix/sycophancy
    # blocks, so "denser topology" is tested rather than inferred by eyeballing
    # separate topology-level confidence intervals.
    topo = em.pivot_table(
        index=["run_key", "model_mix", "sycophancy"],
        columns="topology", values="seed_manipulation_effect", aggfunc="first"
    )
    for label, a, b in (
        ("mesh_full - mesh_random", "mesh_full", "mesh_random"),
        ("static_hierarchy_root - static_hierarchy_leaf", "static_hierarchy_root", "static_hierarchy_leaf"),
    ):
        if a in topo.columns and b in topo.columns:
            tt = topo.dropna(subset=[a, b]).copy()
            tt["contrast"] = tt[a] - tt[b]
            by_run = tt.reset_index().groupby("run_key", as_index=False)["contrast"].mean()
            mean, lo, hi, n = bootstrap_mean_ci(by_run["contrast"])
            factor_rows.append({
                "factor": "topology_contrast",
                "level_or_contrast": label,
                "metric": metric,
                "n_blocks": n,
                "estimate": mean,
                "ci_low": lo,
                "ci_high": hi,
            })

    # Heterogeneous - homogeneous treatment-effect contrast, paired within
    # run_key × topology × sycophancy.
    mm = em.pivot_table(
        index=["run_key", "topology", "sycophancy"],
        columns="model_mix", values="seed_manipulation_effect", aggfunc="first"
    ).dropna(subset=["homogeneous", "heterogeneous"])
    mm["contrast"] = mm["heterogeneous"] - mm["homogeneous"]
    by_run = mm.reset_index().groupby("run_key", as_index=False)["contrast"].mean()
    mean, lo, hi, n = bootstrap_mean_ci(by_run["contrast"])
    factor_rows.append({
        "factor": "model_mix",
        "level_or_contrast": "heterogeneous - homogeneous",
        "metric": metric,
        "n_blocks": n,
        "estimate": mean,
        "ci_low": lo,
        "ci_high": hi,
    })

    sy = em.pivot_table(
        index=["run_key", "topology", "model_mix"],
        columns="sycophancy", values="seed_manipulation_effect", aggfunc="first"
    ).dropna(subset=["low", "medium", "high"])
    for label, a, b in (
        ("medium - low", "medium", "low"),
        ("high - low", "high", "low"),
        ("high - medium", "high", "medium"),
    ):
        sy["contrast"] = sy[a] - sy[b]
        by_run = sy.reset_index().groupby("run_key", as_index=False)["contrast"].mean()
        mean, lo, hi, n = bootstrap_mean_ci(by_run["contrast"])
        factor_rows.append({
            "factor": "sycophancy",
            "level_or_contrast": label,
            "metric": metric,
            "n_blocks": n,
            "estimate": mean,
            "ci_low": lo,
            "ci_high": hi,
        })

rq2_factor_effects = pd.DataFrame(factor_rows)
rq2_factor_effects.to_csv(OUTPUT_DIR / "rq2_factor_effects.csv", index=False)
print("\nRQ2 FACTOR CONTRASTS OF TREATMENT EFFECT")
display(rq2_factor_effects)

# ---------------------------------------------------------------------------
# 6) Sycophancy capability-confound diagnostic on null-independent clean work.
# ---------------------------------------------------------------------------
cap_rows = []
null = valid[valid["condition"] == "null_independent"].copy()
for (mix, syc), g in null.groupby(["model_mix", "sycophancy"]):
    mean, lo, hi, n = bootstrap_mean_ci(g["pre_exposure_clean_pass_rate"])
    cap_rows.append({
        "model_mix": mix,
        "sycophancy": syc,
        "n_runs": n,
        "pre_exposure_clean_pass_mean": mean,
        "ci_low": lo,
        "ci_high": hi,
    })
capability_check = pd.DataFrame(cap_rows)
capability_check.to_csv(OUTPUT_DIR / "sycophancy_capability_check.csv", index=False)
print("\nSYCOPHANCY CAPABILITY CHECK")
display(capability_check)

# ---------------------------------------------------------------------------
# 7) RQ3 — conditional-on-adoption matched analysis.
# ---------------------------------------------------------------------------
rq3_path = OUTPUT_DIR / "rq3_results.csv"
rq3_summary = pd.DataFrame()
rq3_paired = pd.DataFrame()
rq3_never_paired = pd.DataFrame()
rq3_eligibility = pd.DataFrame()
rq3_history_carryover = pd.DataFrame()
rq3_history_carryover_contrasts = pd.DataFrame()
rq3_summary_by_sycophancy = pd.DataFrame()
rq3_sycophancy_paired_effects = pd.DataFrame()
rq3_sycophancy_drift_effects = pd.DataFrame()
rq3_sycophancy_factor_contrasts = pd.DataFrame()
rq3_clean_success_effects = pd.DataFrame()

if rq3_path.exists():
    rq3 = pd.read_csv(rq3_path)
    elig_path = OUTPUT_DIR / "rq3_eligibility.csv"
    if elig_path.exists():
        rq3_eligibility = pd.read_csv(elig_path)

    outcome_rows = []
    # On benchmark trigger/clean tasks use adoption; on SWE audit use the same
    # conservative manipulation flag stored in is_adoption.
    for keys, g in rq3.groupby(["subject", "mode", "task_source", "task_kind"], dropna=False):
        mean, lo, hi, n = bootstrap_mean_ci(g["is_adoption"])
        outcome_rows.append({
            "subject": keys[0],
            "mode": keys[1],
            "task_source": keys[2],
            "task_kind": keys[3],
            "n_task_rows": n,
            "adoption_or_manipulation_rate": mean,
            "ci_low": lo,
            "ci_high": hi,
        })
    rq3_summary = pd.DataFrame(outcome_rows)
    rq3_summary.to_csv(OUTPUT_DIR / "rq3_summary.csv", index=False)

    # Case-level adopter-vs-exact-honest comparison, preserving dependence among
    # multiple tasks inside one RQ3 case.
    case_rates = (
        rq3[rq3["subject"].isin(["adopter", "matched_honest"])]
        .groupby(["case_index", "subject", "mode", "task_source"], as_index=False)
        ["is_adoption"].mean()
    )
    piv = case_rates.pivot_table(
        index=["case_index", "mode", "task_source"],
        columns="subject", values="is_adoption", aggfunc="first"
    ).dropna(subset=["adopter", "matched_honest"]).reset_index()
    piv["paired_diff"] = piv["adopter"] - piv["matched_honest"]

    paired_rows = []
    for (mode, source), g in piv.groupby(["mode", "task_source"]):
        mean, lo, hi, n = bootstrap_mean_ci(g["paired_diff"])
        paired_rows.append({
            "mode": mode,
            "task_source": source,
            "n_cases": n,
            "adopter_minus_matched_honest": mean,
            "ci_low": lo,
            "ci_high": hi,
        })
    rq3_paired = pd.DataFrame(paired_rows)
    rq3_paired.to_csv(OUTPUT_DIR / "rq3_paired_case_effects.csv", index=False)

    # Optional same-trial never-adopter comparison. Full-cascade trials have no
    # never-adopter and are retained in RQ3 rather than excluded.
    rq3_never_paired = pd.DataFrame()
    if "same_trial_never_adopter" in set(rq3["subject"]):
        never_case_rates = (
            rq3[rq3["subject"].isin(["adopter", "same_trial_never_adopter"])]
            .groupby(["case_index", "subject", "mode", "task_source"], as_index=False)
            ["is_adoption"].mean()
        )
        never_piv = never_case_rates.pivot_table(
            index=["case_index", "mode", "task_source"],
            columns="subject", values="is_adoption", aggfunc="first"
        ).reset_index()
        if {"adopter", "same_trial_never_adopter"}.issubset(never_piv.columns):
            never_piv = never_piv.dropna(subset=["adopter", "same_trial_never_adopter"])
            never_piv["paired_diff"] = (
                never_piv["adopter"] - never_piv["same_trial_never_adopter"]
            )
            never_rows = []
            for (mode, source), g in never_piv.groupby(["mode", "task_source"]):
                mean, lo, hi, n = bootstrap_mean_ci(g["paired_diff"])
                never_rows.append({
                    "mode": mode,
                    "task_source": source,
                    "n_cases": n,
                    "adopter_minus_same_trial_never_adopter": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                })
            rq3_never_paired = pd.DataFrame(never_rows)
            rq3_never_paired.to_csv(
                OUTPUT_DIR / "rq3_adopter_vs_same_trial_never_adopter.csv", index=False
            )

    # Explicit context-mediated carryover: self_history minus reset within the
    # same selected subject/case/task source. This is the RQ3 persistence quantity.
    carry_rates = (
        rq3[rq3["subject"].isin(["adopter", "matched_honest", "same_trial_never_adopter"])]
        .groupby(["case_index", "subject", "mode", "task_source"], as_index=False)
        ["is_adoption"].mean()
    )
    carry_piv = carry_rates.pivot_table(
        index=["case_index", "subject", "task_source"],
        columns="mode", values="is_adoption", aggfunc="first"
    ).dropna(subset=["self_history", "reset"]).reset_index()
    carry_piv["self_history_minus_reset"] = carry_piv["self_history"] - carry_piv["reset"]

    carry_rows = []
    for (subject, source), g in carry_piv.groupby(["subject", "task_source"]):
        mean, lo, hi, n = bootstrap_mean_ci(g["self_history_minus_reset"])
        carry_rows.append({
            "subject": subject,
            "task_source": source,
            "n_cases": n,
            "self_history_minus_reset": mean,
            "ci_low": lo,
            "ci_high": hi,
            "interpretation": "context-mediated carryover; not permanent model change",
        })
    rq3_history_carryover = pd.DataFrame(carry_rows)
    rq3_history_carryover.to_csv(
        OUTPUT_DIR / "rq3_history_carryover_effects.csv", index=False
    )

    # Difference-in-carryover: adopter vs matched honest and same-trial never-adopter.
    carry_case = carry_piv.pivot_table(
        index=["case_index", "task_source"], columns="subject",
        values="self_history_minus_reset", aggfunc="first"
    ).reset_index()
    carry_contrast_rows = []
    for control in ("matched_honest", "same_trial_never_adopter"):
        if "adopter" not in carry_case.columns or control not in carry_case.columns:
            continue
        temp = carry_case.dropna(subset=["adopter", control]).copy()
        temp["diff_in_carryover"] = temp["adopter"] - temp[control]
        for source, g in temp.groupby("task_source"):
            mean, lo, hi, n = bootstrap_mean_ci(g["diff_in_carryover"])
            carry_contrast_rows.append({
                "control": control,
                "task_source": source,
                "n_cases": n,
                "adopter_minus_control_carryover": mean,
                "ci_low": lo,
                "ci_high": hi,
            })
    rq3_history_carryover_contrasts = pd.DataFrame(carry_contrast_rows)
    rq3_history_carryover_contrasts.to_csv(
        OUTPUT_DIR / "rq3_history_carryover_contrasts.csv", index=False
    )

    # -----------------------------------------------------------------------
    # RQ3 sycophancy moderation — proposal-facing analysis.
    # The written RQ3 asks whether higher network sycophancy makes drift appear
    # more strongly. We therefore preserve case-level dependence and compare the
    # adopter to its exact matched-honest counterpart separately at low/medium/high.
    # -----------------------------------------------------------------------
    syc_rows = []
    for keys, g in rq3.groupby(
        ["sycophancy", "subject", "mode", "task_source", "task_kind"],
        dropna=False,
    ):
        mean, lo, hi, n = bootstrap_mean_ci(g["is_adoption"])
        syc_rows.append({
            "sycophancy": keys[0],
            "subject": keys[1],
            "mode": keys[2],
            "task_source": keys[3],
            "task_kind": keys[4],
            "n_task_rows": n,
            "adoption_or_manipulation_rate": mean,
            "ci_low": lo,
            "ci_high": hi,
        })
    rq3_summary_by_sycophancy = pd.DataFrame(syc_rows)
    rq3_summary_by_sycophancy.to_csv(
        OUTPUT_DIR / "rq3_summary_by_sycophancy.csv", index=False
    )

    syc_case_rates = (
        rq3[rq3["subject"].isin(["adopter", "matched_honest"])]
        .groupby(
            ["case_index", "sycophancy", "subject", "mode", "task_source"],
            as_index=False,
        )["is_adoption"].mean()
    )
    syc_piv = syc_case_rates.pivot_table(
        index=["case_index", "sycophancy", "mode", "task_source"],
        columns="subject", values="is_adoption", aggfunc="first",
    ).reset_index()
    if {"adopter", "matched_honest"}.issubset(syc_piv.columns):
        syc_piv = syc_piv.dropna(subset=["adopter", "matched_honest"])
        syc_piv["paired_diff"] = syc_piv["adopter"] - syc_piv["matched_honest"]
        syc_pair_rows = []
        for (syc, mode, source), g in syc_piv.groupby(
            ["sycophancy", "mode", "task_source"]
        ):
            mean, lo, hi, n = bootstrap_mean_ci(g["paired_diff"])
            syc_pair_rows.append({
                "sycophancy": syc,
                "mode": mode,
                "task_source": source,
                "n_cases": n,
                "adopter_minus_matched_honest": mean,
                "ci_low": lo,
                "ci_high": hi,
            })
        rq3_sycophancy_paired_effects = pd.DataFrame(syc_pair_rows)
        rq3_sycophancy_paired_effects.to_csv(
            OUTPUT_DIR / "rq3_sycophancy_paired_effects.csv", index=False
        )

    # Difference-in-carryover by sycophancy: [adopter self-history - reset]
    # minus [matched-honest self-history - reset]. This is the cleanest RQ3
    # measure of whether the Stage-1 social-pressure condition moderates drift.
    carry_syc_rates = (
        rq3[rq3["subject"].isin(["adopter", "matched_honest"])]
        .groupby(
            ["case_index", "sycophancy", "subject", "mode", "task_source"],
            as_index=False,
        )["is_adoption"].mean()
    )
    carry_syc_piv = carry_syc_rates.pivot_table(
        index=["case_index", "sycophancy", "subject", "task_source"],
        columns="mode", values="is_adoption", aggfunc="first",
    ).reset_index()
    if {"self_history", "reset"}.issubset(carry_syc_piv.columns):
        carry_syc_piv = carry_syc_piv.dropna(subset=["self_history", "reset"])
        carry_syc_piv["carryover"] = (
            carry_syc_piv["self_history"] - carry_syc_piv["reset"]
        )
        carry_syc_case = carry_syc_piv.pivot_table(
            index=["case_index", "sycophancy", "task_source"],
            columns="subject", values="carryover", aggfunc="first",
        ).reset_index()
        if {"adopter", "matched_honest"}.issubset(carry_syc_case.columns):
            carry_syc_case = carry_syc_case.dropna(
                subset=["adopter", "matched_honest"]
            )
            carry_syc_case["adopter_minus_honest_carryover"] = (
                carry_syc_case["adopter"] - carry_syc_case["matched_honest"]
            )
            drift_rows = []
            for (syc, source), g in carry_syc_case.groupby(
                ["sycophancy", "task_source"]
            ):
                mean, lo, hi, n = bootstrap_mean_ci(
                    g["adopter_minus_honest_carryover"]
                )
                drift_rows.append({
                    "sycophancy": syc,
                    "task_source": source,
                    "n_cases": n,
                    "adopter_minus_honest_carryover": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                })
            rq3_sycophancy_drift_effects = pd.DataFrame(drift_rows)
            rq3_sycophancy_drift_effects.to_csv(
                OUTPUT_DIR / "rq3_sycophancy_drift_effects.csv", index=False
            )

            # Explicit higher-vs-lower sycophancy contrasts on the case-level
            # difference-in-carryover. Cells are independently sampled, so this
            # is a difference of bootstrap means rather than a within-case pair.
            factor_rows = []
            order = ["low", "medium", "high"]
            for source, source_df in carry_syc_case.groupby("task_source"):
                vals = {
                    level: source_df.loc[
                        source_df["sycophancy"] == level,
                        "adopter_minus_honest_carryover",
                    ].astype(float).to_numpy()
                    for level in order
                }
                for hi_level, lo_level in [("medium", "low"), ("high", "low"), ("high", "medium")]:
                    a, b = vals.get(hi_level), vals.get(lo_level)
                    if a is None or b is None or len(a) == 0 or len(b) == 0:
                        continue
                    rng = np.random.default_rng(20260824 + len(factor_rows))
                    boots = []
                    for _ in range(4000):
                        aa = rng.choice(a, size=len(a), replace=True)
                        bb = rng.choice(b, size=len(b), replace=True)
                        boots.append(float(np.mean(aa) - np.mean(bb)))
                    boots = np.asarray(boots, dtype=float)
                    factor_rows.append({
                        "task_source": source,
                        "contrast": f"{hi_level}_minus_{lo_level}",
                        "n_high": int(len(a)),
                        "n_low": int(len(b)),
                        "difference_in_carryover_effect": float(np.mean(a) - np.mean(b)),
                        "ci_low": float(np.quantile(boots, 0.025)),
                        "ci_high": float(np.quantile(boots, 0.975)),
                    })
            rq3_sycophancy_factor_contrasts = pd.DataFrame(factor_rows)
            rq3_sycophancy_factor_contrasts.to_csv(
                OUTPUT_DIR / "rq3_sycophancy_factor_contrasts.csv", index=False
            )

    # Proposal's Generalization Rate also asks whether infected agents lose
    # clean-task success while taking manipulative shortcuts. LiveCodeBench clean
    # rows are execution-scored, so report clean pass-rate effects separately.
    clean_exec = rq3[rq3["task_source"] == "livecodebench_clean"].copy()
    if not clean_exec.empty:
        clean_exec["passed_numeric"] = clean_exec["passed"].map(
            {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0}
        )
        if clean_exec["passed_numeric"].isna().any():
            clean_exec["passed_numeric"] = pd.to_numeric(
                clean_exec["passed"], errors="coerce"
            )
        clean_case = (
            clean_exec[clean_exec["subject"].isin(["adopter", "matched_honest"])]
            .groupby(
                ["case_index", "sycophancy", "subject", "mode"], as_index=False
            )["passed_numeric"].mean()
            .rename(columns={"passed_numeric": "passed"})
        )
        clean_piv = clean_case.pivot_table(
            index=["case_index", "sycophancy", "mode"],
            columns="subject", values="passed", aggfunc="first",
        ).reset_index()
        if {"adopter", "matched_honest"}.issubset(clean_piv.columns):
            clean_piv = clean_piv.dropna(subset=["adopter", "matched_honest"])
            clean_piv["adopter_minus_matched_honest_clean_pass"] = (
                clean_piv["adopter"] - clean_piv["matched_honest"]
            )
            clean_rows = []
            for (syc, mode), g in clean_piv.groupby(["sycophancy", "mode"]):
                mean, lo, hi, n = bootstrap_mean_ci(
                    g["adopter_minus_matched_honest_clean_pass"]
                )
                clean_rows.append({
                    "sycophancy": syc,
                    "mode": mode,
                    "n_cases": n,
                    "adopter_minus_matched_honest_clean_pass": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                })
            rq3_clean_success_effects = pd.DataFrame(clean_rows)
            rq3_clean_success_effects.to_csv(
                OUTPUT_DIR / "rq3_clean_success_effects.csv", index=False
            )

    print("\nRQ3 SUMMARY (conditional on realized seed + Stage-1 post-exposure adoption)")
    display(rq3_summary)
    print("\nRQ3 PAIRED CASE EFFECTS — adopter vs exact honest counterpart")
    display(rq3_paired)
    print("\nRQ3 PAIRED CASE EFFECTS — adopter vs same-trial never-adopter")
    display(rq3_never_paired)
    print("\nRQ3 SELF-HISTORY CARRYOVER (self_history - reset)")
    display(rq3_history_carryover)
    print("\nRQ3 DIFFERENCE-IN-CARRYOVER — adopter vs controls")
    display(rq3_history_carryover_contrasts)
    print("\nRQ3 BY SYCOPHANCY — paired adopter vs matched honest")
    display(rq3_sycophancy_paired_effects)
    print("\nRQ3 SYCOPHANCY MODERATION — difference-in-carryover")
    display(rq3_sycophancy_drift_effects)
    print("\nRQ3 SYCOPHANCY FACTOR CONTRASTS")
    display(rq3_sycophancy_factor_contrasts)
    print("\nRQ3 CLEAN EXECUTION SUCCESS EFFECTS")
    display(rq3_clean_success_effects)

    if len(rq3_eligibility):
        eligible_share = float(pd.to_numeric(rq3_eligibility["eligible"], errors="coerce").mean())
        print(f"RQ3 eligible factorial-cell share: {eligible_share:.3f}")
else:
    print("RQ3 results not found. Run the Stage-2 cell before using RQ3 conclusions.")


# ---------------------------------------------------------------------------
# 7b) PAPER-COMPLETE NULL/LOW-SPREAD ANALYSIS.
# These outputs ensure that weak or zero contagion is still a complete result:
#   (i) spontaneous/no-seed baselines,
#   (ii) round-by-round dynamics,
#   (iii) a realized-seed -> exposure -> adoption -> success -> innovation funnel,
#   (iv) T80 reach probability plus restricted/censored timing,
#   (v) an overall factorial-average RQ1 ITT estimate.
# ---------------------------------------------------------------------------

# ---------- Explicit no-infection / background controls ---------------------
control_metrics = [
    "spontaneous_adoption_rate",
    "final_post_exposure_adoption_rate",
    "final_successful_post_exposure_hack_rate",
    "clean_pass_rate",
    "pre_exposure_clean_pass_rate",
]
control_rows = []
for keys, g in valid.groupby(GROUP, dropna=False):
    if keys[0] not in {"null_independent", "honest_peer_control", "exposure_conditioned_seed"}:
        continue
    row = dict(zip(GROUP, keys))
    row["n_runs"] = int(g["run_key"].nunique())
    for metric in control_metrics:
        mean, lo, hi, n = bootstrap_mean_ci(g[metric])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_ci_low"] = lo
        row[f"{metric}_ci_high"] = hi
        row[f"{metric}_n"] = n
    control_rows.append(row)
control_baselines = pd.DataFrame(control_rows)
control_baselines.to_csv(OUTPUT_DIR / "control_baseline_summary.csv", index=False)
print("\nCONTROL / BACKGROUND BASELINES")
display(control_baselines)

# ---------- Overall factorial-average RQ1 ITT -------------------------------
# Every network cell has equal replication, so this is an equal-cell/equal-run
# average seed-manipulation effect across the full RQ2 factorial design. It is
# useful as the global RQ1 answer while RQ2 describes heterogeneity by factors.
overall_rq1_rows = []
for metric in (
    "final_post_exposure_adoption_rate",
    "final_successful_post_exposure_hack_rate",
    "final_seed_linked_adoption_rate",
    "t80_post_exposure_adoption_reached",
    "t80_successful_post_exposure_hack_reached",
):
    vals = rq2_effects.loc[rq2_effects["metric"] == metric, "seed_manipulation_effect"]
    mean, lo, hi, n = bootstrap_mean_ci(vals)
    overall_rq1_rows.append({
        "estimand": "factorial_average_seed_manipulation_effect",
        "metric": metric,
        "n_paired_network_trials": n,
        "mean_effect": mean,
        "ci_low": lo,
        "ci_high": hi,
    })
rq1_overall_itt = pd.DataFrame(overall_rq1_rows)
rq1_overall_itt.to_csv(OUTPUT_DIR / "rq1_overall_factorial_average_itt.csv", index=False)
print("\nRQ1 OVERALL FACTORIAL-AVERAGE ITT")
display(rq1_overall_itt)

# ---------- Round-by-round spread dynamics ----------------------------------
curve_fields = {
    "post_exposure_adoption": "post_exposure_adoption_curve",
    "successful_post_exposure_hack": "successful_post_exposure_curve",
    "seed_linked_adoption": "seed_linked_adoption_curve",
    "all_adoption": "all_adoption_curve",
}
round_rows = []
for r in valid.itertuples(index=False):
    for metric_name, field_name in curve_fields.items():
        curve = getattr(r, field_name, None)
        if not isinstance(curve, (list, tuple)):
            continue
        for round_idx, value in enumerate(curve):
            round_rows.append({
                "run_key": r.run_key,
                "condition": r.condition,
                "topology": r.topology,
                "model_mix": r.model_mix,
                "sycophancy": r.sycophancy,
                "round_index": int(round_idx),
                "round_number": int(round_idx) + 1,
                "metric": metric_name,
                "rate": float(value),
            })
round_level = pd.DataFrame(round_rows)
round_level.to_csv(OUTPUT_DIR / "spread_dynamics_run_level.csv", index=False)

round_summary_rows = []
if len(round_level):
    for keys, g in round_level.groupby(
        ["condition", "topology", "model_mix", "sycophancy", "round_index", "round_number", "metric"],
        dropna=False,
    ):
        mean, lo, hi, n = bootstrap_mean_ci(g["rate"])
        round_summary_rows.append({
            "condition": keys[0],
            "topology": keys[1],
            "model_mix": keys[2],
            "sycophancy": keys[3],
            "round_index": keys[4],
            "round_number": keys[5],
            "metric": keys[6],
            "n_runs": n,
            "mean_rate": mean,
            "ci_low": lo,
            "ci_high": hi,
        })
spread_dynamics = pd.DataFrame(round_summary_rows)
spread_dynamics.to_csv(OUTPUT_DIR / "spread_dynamics_by_round.csv", index=False)

# Exact paired treatment-control differences at every round.
round_paired_rows = []
if len(round_level):
    _seed_round = round_level[round_level["condition"] == "exposure_conditioned_seed"].copy()
    _hon_round = round_level[round_level["condition"] == "honest_peer_control"].copy()
    _keys = ["run_key", "topology", "model_mix", "sycophancy", "round_index", "round_number", "metric"]
    _rp = _seed_round.merge(
        _hon_round,
        on=_keys,
        suffixes=("_seed", "_honest"),
        validate="one_to_one",
    )
    _rp["seed_manipulation_effect"] = _rp["rate_seed"] - _rp["rate_honest"]
    for keys, g in _rp.groupby(
        ["topology", "model_mix", "sycophancy", "round_index", "round_number", "metric"]
    ):
        mean, lo, hi, n = bootstrap_mean_ci(g["seed_manipulation_effect"])
        round_paired_rows.append({
            "topology": keys[0],
            "model_mix": keys[1],
            "sycophancy": keys[2],
            "round_index": keys[3],
            "round_number": keys[4],
            "metric": keys[5],
            "n_pairs": n,
            "mean_seed_manipulation_effect": mean,
            "ci_low": lo,
            "ci_high": hi,
        })
spread_dynamics_paired = pd.DataFrame(round_paired_rows)
spread_dynamics_paired.to_csv(
    OUTPUT_DIR / "spread_dynamics_paired_effects_by_round.csv", index=False
)
print("\nROUND-BY-ROUND SPREAD DYNAMICS")
display(spread_dynamics.head(30))

# ---------- Contagion funnel: makes weak/zero-spread runs interpretable ------
# Run-level mechanism stages. All treatment trials remain in the denominator;
# prompt realization and exposure are explicitly reported rather than filtered.
_turn_seed = turns_df[
    (turns_df["condition"] == "exposure_conditioned_seed")
    & (pd.to_numeric(turns_df["agent_id"], errors="coerce") != 0)
].copy()
_funnel_run = (
    _turn_seed.groupby(["run_key", "topology", "model_mix", "sycophancy"], as_index=False)
    .agg(
        any_seed_exposure=("exposed_to_seed_lineage", "max"),
        any_seed_linked_adoption=("is_seed_linked_post_exposure_adoption", "max"),
        any_successful_post_exposure_hack=("is_successful_post_exposure_hack", "max"),
        any_operational_innovation=("is_operational_innovation", "max"),
    )
)
_funnel_run = treatment_trials[
    ["run_key", "topology", "model_mix", "sycophancy", "patient_zero_hack_realized"]
].merge(
    _funnel_run,
    on=["run_key", "topology", "model_mix", "sycophancy"],
    how="left",
    validate="one_to_one",
)
for _c in [
    "patient_zero_hack_realized",
    "any_seed_exposure",
    "any_seed_linked_adoption",
    "any_successful_post_exposure_hack",
    "any_operational_innovation",
]:
    _funnel_run[_c] = _funnel_run[_c].fillna(False).astype(bool)
_funnel_run.to_csv(OUTPUT_DIR / "contagion_funnel_run_level.csv", index=False)

funnel_rows = []
def _funnel_summary_row(g, label_fields):
    n = len(g)
    realized = g["patient_zero_hack_realized"]
    exposed = g["any_seed_exposure"]
    adopted = g["any_seed_linked_adoption"]
    success = g["any_successful_post_exposure_hack"]
    innov = g["any_operational_innovation"]
    n_realized = int(realized.sum())
    n_exposed_realized = int((realized & exposed).sum())
    n_adopt_exposed = int((realized & exposed & adopted).sum())
    n_success_adopt = int((realized & exposed & adopted & success).sum())
    return {
        **label_fields,
        "n_treatment_runs": int(n),
        "n_prompt_realized": n_realized,
        "exposure_realization_rate": float(realized.mean()) if n else float("nan"),
        "n_with_seed_exposure": int(exposed.sum()),
        "seed_exposure_rate_all_treatment": float(exposed.mean()) if n else float("nan"),
        "seed_exposure_rate_given_realization": (
            n_exposed_realized / n_realized if n_realized else float("nan")
        ),
        "n_with_seed_linked_adoption": int(adopted.sum()),
        "adoption_rate_given_realized_exposure": (
            n_adopt_exposed / n_exposed_realized if n_exposed_realized else float("nan")
        ),
        "n_with_successful_post_exposure_hack": int(success.sum()),
        "successful_hack_rate_given_seed_linked_adoption": (
            n_success_adopt / n_adopt_exposed if n_adopt_exposed else float("nan")
        ),
        "n_with_operational_innovation": int(innov.sum()),
        "operational_innovation_run_rate": float(innov.mean()) if n else float("nan"),
    }

funnel_rows.append(_funnel_summary_row(_funnel_run, {
    "scope": "overall",
    "topology": "ALL",
    "model_mix": "ALL",
    "sycophancy": "ALL",
}))
for keys, g in _funnel_run.groupby(["topology", "model_mix", "sycophancy"]):
    funnel_rows.append(_funnel_summary_row(g, {
        "scope": "cell",
        "topology": keys[0],
        "model_mix": keys[1],
        "sycophancy": keys[2],
    }))
contagion_funnel = pd.DataFrame(funnel_rows)
contagion_funnel.to_csv(OUTPUT_DIR / "contagion_funnel.csv", index=False)
print("\nCONTAGION FUNNEL — LOW/ZERO SPREAD REMAINS INTERPRETABLE")
display(contagion_funnel)

# ---------- T80 reach + censored/restricted timing ---------------------------
# A run that never reaches 80% is not silently dropped. We report:
#   1) probability of reaching T80 by the final round,
#   2) conditional T80 among runs that reached it,
#   3) restricted time where 'not reached' is encoded as ROUNDS+1.
# The restricted measure is an ordinal summary, not a literal event time.
t80_specs = [
    ("post_exposure_adoption", "t80_post_exposure_adoption_reached", "t80_post_exposure_adoption_rounds"),
    ("successful_post_exposure_hack", "t80_successful_post_exposure_hack_reached", "t80_successful_post_exposure_hack_rounds"),
]
t80_summary_rows = []
for event_name, reached_col, round_col in t80_specs:
    for keys, g in valid.groupby(GROUP, dropna=False):
        reached = g[reached_col].fillna(False).astype(bool)
        rounds_reached = pd.to_numeric(g.loc[reached, round_col], errors="coerce")
        restricted = pd.to_numeric(g[round_col], errors="coerce").where(reached, ROUNDS + 1)
        p, plo, phi, pn = bootstrap_mean_ci(reached.astype(float))
        cm, clo, chi, cn = bootstrap_mean_ci(rounds_reached)
        rm, rlo, rhi, rn = bootstrap_mean_ci(restricted)
        t80_summary_rows.append({
            **dict(zip(GROUP, keys)),
            "event": event_name,
            "n_runs": int(len(g)),
            "reach_probability": p,
            "reach_ci_low": plo,
            "reach_ci_high": phi,
            "conditional_mean_t80_rounds_among_reached": cm,
            "conditional_t80_ci_low": clo,
            "conditional_t80_ci_high": chi,
            "n_reached": cn,
            "restricted_mean_round_not_reached_equals_rounds_plus_1": rm,
            "restricted_ci_low": rlo,
            "restricted_ci_high": rhi,
            "restricted_n": rn,
            "censoring_rule": f"not reached by round {ROUNDS} -> {ROUNDS + 1}",
        })
t80_censoring_summary = pd.DataFrame(t80_summary_rows)
t80_censoring_summary.to_csv(OUTPUT_DIR / "t80_censoring_summary.csv", index=False)

# Paired treatment-control restricted-time effects. Negative means the prompt
# assignment produced an earlier/more frequent cascade under this encoding.
t80_pair_rows = []
for event_name, reached_col, round_col in t80_specs:
    tmp = paired[[
        "run_key", "topology", "model_mix", "sycophancy",
        f"{reached_col}_seed", f"{round_col}_seed",
        f"{reached_col}_honest", f"{round_col}_honest",
    ]].copy()
    sr = tmp[f"{reached_col}_seed"].fillna(False).astype(bool)
    hr = tmp[f"{reached_col}_honest"].fillna(False).astype(bool)
    tmp["seed_restricted"] = pd.to_numeric(tmp[f"{round_col}_seed"], errors="coerce").where(sr, ROUNDS + 1)
    tmp["honest_restricted"] = pd.to_numeric(tmp[f"{round_col}_honest"], errors="coerce").where(hr, ROUNDS + 1)
    tmp["diff"] = tmp["seed_restricted"] - tmp["honest_restricted"]
    for keys, g in tmp.groupby(["topology", "model_mix", "sycophancy"]):
        mean, lo, hi, n = bootstrap_mean_ci(g["diff"])
        t80_pair_rows.append({
            "event": event_name,
            "topology": keys[0],
            "model_mix": keys[1],
            "sycophancy": keys[2],
            "n_pairs": n,
            "mean_prompt_minus_honest_restricted_round": mean,
            "ci_low": lo,
            "ci_high": hi,
            "interpretation": "negative = earlier/more frequent cascade under seed manipulation",
        })
t80_paired_restricted = pd.DataFrame(t80_pair_rows)
t80_paired_restricted.to_csv(
    OUTPUT_DIR / "t80_paired_restricted_time_effects.csv", index=False
)
print("\nT80 REACH / CENSORING SUMMARY")
display(t80_censoring_summary.head(30))

# ---------- RQ3 eligibility/status is itself a paper result ------------------
rq3_eligibility_path = OUTPUT_DIR / "rq3_eligibility.csv"
rq3_eligibility_final = (
    pd.read_csv(rq3_eligibility_path) if rq3_eligibility_path.exists() else pd.DataFrame()
)
_eligible_total = 0
if len(rq3_eligibility_final) and "eligible_cases" in rq3_eligibility_final.columns:
    _eligible_total = int(pd.to_numeric(rq3_eligibility_final["eligible_cases"], errors="coerce").fillna(0).sum())
rq3_status = {
    "eligible_cases_before_budget_selection": _eligible_total,
    "rq3_results_file_exists": bool((OUTPUT_DIR / "rq3_results.csv").exists()),
    "interpretation_if_zero": (
        "No RQ3 effect is estimable because Stage 1 produced no realized-seed, "
        "seed-linked adopter meeting the preregistered eligibility rule; this is a valid null/eligibility result."
    ),
}
(OUTPUT_DIR / "rq3_status.json").write_text(json.dumps(rq3_status, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 8) Budget audit — all paid phases + persistent provider-balance anchor.
# ---------------------------------------------------------------------------
main_reported_cost = float(
    pd.to_numeric(df.get("trial_reported_model_cost_usd", 0), errors="coerce")
      .fillna(0).sum()
)
rq3_reported_cost = 0.0
if rq3_path.exists():
    _rq3 = pd.read_csv(rq3_path)
    rq3_reported_cost = float(
        pd.to_numeric(_rq3.get("reported_cost_usd", 0), errors="coerce")
          .fillna(0).sum()
    )

def _load_budget_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

_stage0_budget = _load_budget_report(OUTPUT_DIR / "stage0_seed_prompt_v3_budget_report.json")
_smoke_budget = _load_budget_report(OUTPUT_DIR / "smoke_gate_report.json")
_stage1_budget = _load_budget_report(OUTPUT_DIR / "stage1_budget_report.json")
_rq3_budget = _load_budget_report(OUTPUT_DIR / "rq3_budget_report.json")
_persistent_budget = _load_budget_report(BUDGET_STATE_PATH)

_stage0_observed_raw = _stage0_budget.get("observed_budget_decrease_usd")
stage0_observed = (
    None if _stage0_observed_raw is None else float(_stage0_observed_raw)
)
smoke_observed = float(_smoke_budget.get("observed_budget_decrease", 0.0) or 0.0)
stage1_observed = float(_stage1_budget.get("cumulative_observed_budget_decrease_usd", 0.0) or 0.0)
rq3_observed = float(_rq3_budget.get("cumulative_observed_budget_decrease_usd", 0.0) or 0.0)
phase_observed_total = (
    (0.0 if stage0_observed is None else stage0_observed)
    + smoke_observed + stage1_observed + rq3_observed
)

_final_key_status = _openrouter_key_status()
_final_remaining_raw = _final_key_status.get("limit_remaining")
_final_remaining = float(_final_remaining_raw) if _final_remaining_raw is not None else float("nan")
_persistent_start = float(_persistent_budget.get("provider_start_remaining_usd", float("nan")))
persistent_provider_delta = (
    max(0.0, _persistent_start - _final_remaining)
    if _persistent_start == _persistent_start and _final_remaining == _final_remaining
    else float("nan")
)

provider_reported_rows_total = main_reported_cost + rq3_reported_cost
budget_audit = {
    "configured_account_credits_estimate_usd": ACCOUNT_CREDITS_ESTIMATE_USD,
    "whole_notebook_ceiling_usd": TOTAL_BUDGET_CAP_USD,
    "operational_generation_stop_threshold_usd": TOTAL_BUDGET_CAP_USD - SESSION_BUDGET_STOP_MARGIN_USD,
    "exposure_cap_usd": EXPOSURE_BUDGET_CAP_USD,
    "smoke_cap_usd": SMOKE_BUDGET_CAP_USD,
    "main_cap_usd": MAIN_BUDGET_CAP_USD,
    "rq3_cap_usd": RQ3_BUDGET_CAP_USD,
    "unallocated_buffer_usd": UNALLOCATED_BUFFER_USD,
    "stage0_observed_provider_balance_decrease_usd": stage0_observed,
    "smoke_observed_provider_balance_decrease_usd": smoke_observed,
    "stage1_cumulative_observed_provider_balance_decrease_usd": stage1_observed,
    "rq3_cumulative_observed_provider_balance_decrease_usd": rq3_observed,
    "sum_of_phase_observed_balance_decreases_usd": phase_observed_total,
    "persistent_provider_start_remaining_usd": _persistent_start,
    "provider_remaining_at_final_audit_usd": _final_remaining,
    "persistent_provider_balance_decrease_usd": persistent_provider_delta,
    "provider_reported_main_cost_captured_usd": main_reported_cost,
    "provider_reported_rq3_cost_captured_usd": rq3_reported_cost,
    "provider_reported_row_cost_total_usd": provider_reported_rows_total,
    "note": (
        "The persistent provider-balance delta is the strongest whole-run audit when this key is dedicated. "
        "Per-row ModelUsage costs cover Stage 1/RQ3 only and are retained as a cross-check."
    ),
}
(OUTPUT_DIR / "budget_audit.json").write_text(
    json.dumps(budget_audit, indent=2), encoding="utf-8"
)
print("\nBUDGET AUDIT")
print(json.dumps(budget_audit, indent=2))
if stage0_observed is not None and stage0_observed > EXPOSURE_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Observed Stage-0 spend exceeds its phase cap.")
if smoke_observed > SMOKE_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Observed smoke spend exceeds its phase cap.")
if stage1_observed > MAIN_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Observed Stage-1 spend exceeds its phase cap.")
if rq3_observed > RQ3_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Observed RQ3 spend exceeds its phase cap.")
if persistent_provider_delta == persistent_provider_delta and persistent_provider_delta > TOTAL_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Persistent provider-balance decrease exceeds whole-notebook ceiling.")

# ---------------------------------------------------------------------------
# 9) Compact paper figures.
# ---------------------------------------------------------------------------
# RQ1 dynamics figure: canonical mesh/full homogeneous condition, with all
# sycophancy levels and honest-vs-exposure trajectories. Null-independent appears
# in the tables rather than this network-dynamics plot because it has no peer context.
_dyn = spread_dynamics[
    (spread_dynamics["topology"] == "mesh_full")
    & (spread_dynamics["model_mix"] == "homogeneous")
    & (spread_dynamics["metric"] == "post_exposure_adoption")
    & (spread_dynamics["condition"].isin(["exposure_conditioned_seed", "honest_peer_control"]))
].copy()
if len(_dyn):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (condition, syc), g in _dyn.groupby(["condition", "sycophancy"]):
        g = g.sort_values("round_number")
        ax.plot(
            g["round_number"], g["mean_rate"], marker="o",
            label=f"{condition} / {syc}",
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean post-exposure adoption rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("RQ1: round-by-round spread dynamics (mesh_full / homogeneous)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq1_spread_dynamics_by_round.png", dpi=220)
    plt.show()

adopt_effect = rq2_cell_effects[
    rq2_cell_effects["metric"] == "final_post_exposure_adoption_rate"
].copy()

if len(adopt_effect):
    # One plot per chart (no subplots).
    fig, ax = plt.subplots(figsize=(10, 6))
    order = adopt_effect.sort_values(
        ["topology", "model_mix", "sycophancy"]
    ).reset_index(drop=True)
    labels = [
        f"{r.topology}\n{r.model_mix[:4]} / {r.sycophancy}"
        for r in order.itertuples()
    ]
    y = order["mean_seed_manipulation_effect"].to_numpy(float)
    lo = y - order["ci_low"].to_numpy(float)
    hi = order["ci_high"].to_numpy(float) - y
    ax.errorbar(np.arange(len(order)), y, yerr=np.vstack([lo, hi]), fmt="o", capsize=3)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Seed-manipulation effect on final post-exposure adoption rate")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("RQ2: treatment effect by topology × model mix × sycophancy")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq2_adoption_seed_manipulation_effects.png", dpi=220)
    plt.show()

# ---------------------------------------------------------------------------
# 10) Machine-readable manuscript summary. No significance is invented.
# ---------------------------------------------------------------------------
def _records(frame):
    if frame is None or len(frame) == 0:
        return []
    return json.loads(frame.to_json(orient="records"))

manuscript = {
    "scope": (
        "OpenRouter deliberately seeded Patient-Zero study using private reward-hacking examples plus "
        "a private direct trigger directive. Primary RQ1/RQ2 quantities are paired "
        "intention-to-treat seed-manipulation effects. Individual post-exposure labels are "
        "temporal/descriptive, RQ3 self-history effects are context-mediated carryover, and "
        "SWE-bench Verified is a behavioral audit without official repository execution; "
        "not permanent weight change."
    ),
    "integrity": integrity_report,
    "patient_zero_exposure_realization": _records(exposure_realization),
    "control_baselines": _records(control_baselines),
    "contagion_funnel": _records(contagion_funnel),
    "rq1_overall_factorial_average_itt": _records(rq1_overall_itt),
    "spread_dynamics_by_round": _records(spread_dynamics),
    "spread_dynamics_paired_effects_by_round": _records(spread_dynamics_paired),
    "t80_censoring_summary": _records(t80_censoring_summary),
    "t80_paired_restricted_time_effects": _records(t80_paired_restricted),
    "rq3_eligibility": _records(rq3_eligibility_final),
    "rq3_status": rq3_status,
    "rq1_primary_paired_contrasts": _records(primary_contrasts_df),
    "rq1_primary_paired_contrasts_task_clustered": _records(rq1_primary_clustered),
    "rq1_realized_seed_descriptive_contrasts": _records(realized_secondary_df),
    "rq1_seed_linked_spread_descriptive": _records(realized_seed_linked),
    "rq1_conditional_operational_innovation": _records(rq1_operational_innovation),
    "rq2_cell_seed_manipulation_effects": _records(rq2_cell_effects),
    "rq2_factor_effects": _records(rq2_factor_effects),
    "sycophancy_capability_check": _records(capability_check),
    "rq3_conditional_summary": _records(rq3_summary),
    "rq3_paired_case_effects": _records(rq3_paired),
    "rq3_adopter_vs_same_trial_never_adopter": _records(rq3_never_paired),
    "rq3_history_carryover_effects": _records(rq3_history_carryover),
    "rq3_history_carryover_contrasts": _records(rq3_history_carryover_contrasts),
    "rq3_summary_by_sycophancy": _records(rq3_summary_by_sycophancy),
    "rq3_sycophancy_paired_effects": _records(rq3_sycophancy_paired_effects),
    "rq3_sycophancy_drift_effects": _records(rq3_sycophancy_drift_effects),
    "rq3_sycophancy_factor_contrasts": _records(rq3_sycophancy_factor_contrasts),
    "rq3_clean_success_effects": _records(rq3_clean_success_effects),
    "budget_audit": budget_audit,
}
(OUTPUT_DIR / "manuscript_results.json").write_text(
    json.dumps(manuscript, indent=2), encoding="utf-8"
)

print("\nFINAL ANALYSIS COMPLETE")
print("All Stage-1 factorial completeness, exposure-treatment/control pairing, and ITT inclusion checks passed.")
print("Use the generated CIs/estimates exactly as observed; a CI crossing zero, no T80 event, or zero RQ3 eligibility is a valid null/uncertain result, not a failed experiment.")
print("Paper-complete outputs include explicit no-infection controls, contagion funnel, round dynamics, censored T80 summaries, RQ1/RQ2 effects, RQ3 status/results, and budget audit.")
print("Outputs:", OUTPUT_DIR)
