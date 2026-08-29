"""
Stage 1: Main RQ1/RQ2 factorial sweep.

Runs the full frozen factorial design (topology x model-mix x
sycophancy x condition) comparing the reward-hacking-seeded Patient
Zero network against the matched honest-seed network, to measure
propagation to susceptible peers (RQ1) and how the treatment effect
varies across the design cells (RQ2).

Run only after Stage 0 and the smoke test both pass.

Depends on globals defined in 05_smoke_test.py.
"""

# =============================================================================
# 16. RUN
# =============================================================================

import inspect as _stdlib_inspect

# FULL-SWEEP SAFETY GATE
if not globals().get("FULL_RUN_UNLOCKED", False):
    raise RuntimeError(
        "Full sweep is locked. Run the paid smoke-test cell immediately above "
        "and continue only after it prints READY FOR FULL RUN."
    )


# Crash-safe Stage-1 phase budget. Persist the phase-start provider balance BEFORE
# any Stage-1 calls, so an interrupted eval_set cannot reset the $78 phase window.
_STAGE1_BUDGET_PATH = OUTPUT_DIR / "stage1_budget_report.json"

def _stage1_atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)

_stage1_remaining_before = float(_openrouter_key_status().get("limit_remaining"))
if _STAGE1_BUDGET_PATH.exists():
    _stage1_previous = json.loads(_STAGE1_BUDGET_PATH.read_text(encoding="utf-8"))
    _stage1_phase_start = float(
        _stage1_previous.get("phase_start_remaining_usd", _stage1_remaining_before)
    )
else:
    _stage1_phase_start = _stage1_remaining_before
    _stage1_previous = {
        "phase_start_remaining_usd": _stage1_phase_start,
        "cumulative_observed_budget_decrease_usd": 0.0,
        "phase_cap_usd": MAIN_BUDGET_CAP_USD,
        "status": "running",
    }
    _stage1_atomic_json_write(_STAGE1_BUDGET_PATH, _stage1_previous)

_stage1_spent_before_resume = max(0.0, _stage1_phase_start - _stage1_remaining_before)
if _stage1_spent_before_resume > MAIN_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Stage-1 observed provider-balance decrease already exceeds its phase cap.")
print(f"Stage-1 cumulative spend before this invocation: ${_stage1_spent_before_resume:.4f}")

EVAL_KWARGS = dict(
    model=HOMOGENEOUS_MODEL,   # default; per-agent models are resolved in-solver
    epochs=EPOCHS,
    temperature=TEMPERATURE,
    max_tokens=max(
        [MAX_OUTPUT_TOKENS, SEED_POSITION_TRIGGER_MAX_OUTPUT_TOKENS]
        + list(MODEL_MAX_OUTPUT_TOKENS.values())
    ),
    # These are Inspect sample-level limits. They intentionally live here
    # rather than in impossible_livecodebench().
    token_limit=1_500_000,
    time_limit=3600,
    max_connections=MAX_CONNECTIONS,
    max_samples=MAX_SAMPLES,
    max_tasks=MAX_TASKS,
    max_subprocesses=MAX_SUBPROCESSES,
    max_retries=MAX_RETRIES,
    cache=False,
    # Plain display is robust in local Jupyter terminals/notebook frontends.
    display="plain",

    # CRITICAL WINDOWS FIX:
    # Disable realtime event logging so Inspect never constructs its
    # SampleBufferDatabase (SQLite). Keep detailed sample logging enabled and
    # flush every completed sample immediately for interruption safety.
    log_samples=True,
    log_realtime=False,
    log_buffer=WINDOWS_LOG_BUFFER_SAMPLES,
)

if USE_EVAL_SET:
    # Resumable: re-running this cell after a disconnect skips finished tasks.
    from inspect_ai import eval_set

    _eval_set_kwargs = dict(
        log_dir=str(LOG_DIR),
        retry_attempts=3,
        **EVAL_KWARGS,
    )
    _eval_set_params = set(_stdlib_inspect.signature(eval_set).parameters)
    if "ctl_server" in _eval_set_params:
        _eval_set_kwargs["ctl_server"] = False

    SUCCESS, EVAL_LOGS = eval_set(TASKS, **_eval_set_kwargs)
    print(f"\neval_set complete (all tasks succeeded: {SUCCESS})")
    if not SUCCESS:
        raise RuntimeError(
            "At least one Stage-1 task is incomplete/failed. Re-run this cell "
            "until eval_set reports success before running RQ3 or analysis."
        )
else:
    _direct_eval_kwargs = dict(
        fail_on_error=False,
        log_dir=str(LOG_DIR),
        **EVAL_KWARGS,
    )
    _inspect_eval_params = set(_stdlib_inspect.signature(inspect_eval).parameters)
    if "ctl_server" in _inspect_eval_params:
        _direct_eval_kwargs["ctl_server"] = False

    EVAL_LOGS = inspect_eval(TASKS, **_direct_eval_kwargs)

_stage1_remaining_after = float(_openrouter_key_status().get("limit_remaining"))
_stage1_invocation_spend = max(0.0, _stage1_remaining_before - _stage1_remaining_after)
_stage1_cumulative = max(0.0, _stage1_phase_start - _stage1_remaining_after)
_stage1_budget_report = {
    "phase_start_remaining_usd": _stage1_phase_start,
    "last_provider_remaining_before": _stage1_remaining_before,
    "last_provider_remaining_after": _stage1_remaining_after,
    "last_invocation_budget_decrease_usd": _stage1_invocation_spend,
    "cumulative_observed_budget_decrease_usd": _stage1_cumulative,
    "phase_cap_usd": MAIN_BUDGET_CAP_USD,
    "status": "complete",
}
_stage1_atomic_json_write(_STAGE1_BUDGET_PATH, _stage1_budget_report)
print(f"Stage-1 cumulative provider-balance decrease: ${_stage1_cumulative:.4f} / ${MAIN_BUDGET_CAP_USD:.2f}")
if _stage1_cumulative > MAIN_BUDGET_CAP_USD + 1e-9:
    raise RuntimeError("Stage-1 observed provider-balance decrease exceeded its phase cap; do not start RQ3.")

print("\nRUN COMPLETE. Proceed to the analysis cell.")
print(f"Logs: {LOG_DIR}")
