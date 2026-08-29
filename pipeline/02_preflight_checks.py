"""
Stage: Preflight diagnostics.

Confirms OPENROUTER_API_KEY is present (prompting for it if not),
lists candidate models, runs one tiny real Inspect/OpenRouter
generation to confirm the API path works, checks the session budget
ceiling, and confirms the local child-Python code-verification path
is functional. Costs approximately one small model call.

Depends on globals defined in 01_environment_setup.py.
"""

# =============================================================================
# LOCAL DIAGNOSTIC — RUN BEFORE THE EXPERIMENT CONFIGURATION
# =============================================================================

import getpass
import json
import os
import re
import subprocess
import urllib.request

FAILED = []
WARNINGS = []

# 1) API key: environment first, then hidden prompt.
key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not key:
    key = getpass.getpass("Paste OPENROUTER_API_KEY (hidden): ").strip()
if not key:
    FAILED.append("OPENROUTER_API_KEY is missing.")
else:
    os.environ["OPENROUTER_API_KEY"] = key
    print("1. OPENROUTER_API_KEY: present (hidden)")

# 2) Live catalog.
CANDIDATE_MODELS = [
    "openai/gpt-oss-120b",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.6-luna",
    "deepseek/deepseek-v4-flash-0731",
    "z-ai/glm-4.5-air",
    "google/gemini-3.7-flash",
    "mistralai/mistral-small-2603",
]
available = set()
try:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        catalog_payload = json.loads(response.read())
    available = {entry["id"] for entry in catalog_payload.get("data", [])}
    print(f"2. OpenRouter catalog reachable: {len(available)} models")
    for slug in CANDIDATE_MODELS:
        print(f"   {'OK' if slug in available else 'NOT FOUND':9s} {slug}")
except Exception as exc:
    FAILED.append(f"OpenRouter catalog request failed: {type(exc).__name__}: {exc}")

# 2b) Research-program/shared-key budget policy.
# The provider key itself may have a cap above $90 because the user may not own
# key-management permissions. We therefore snapshot provider remaining balance
# and enforce a separate notebook-session ceiling.
#
# Preliminary balance floor only. The configuration cell performs the exact
# catalog-price check against the cached-Stage-0 pending plan before paid work.
SESSION_BUDGET_CEILING_USD = 90.0
SESSION_BUDGET_STOP_MARGIN_USD = 1.0
MIN_REQUIRED_REMAINING_USD = 80.0

try:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        key_status = (json.loads(response.read()) or {}).get("data") or {}

    key_limit = key_status.get("limit")
    remaining = key_status.get("limit_remaining")
    if remaining is None:
        raise RuntimeError(
            "This OpenRouter key does not expose `limit_remaining`, so the notebook "
            "cannot safely enforce the research-session budget."
        )

    key_limit = float(key_limit) if key_limit is not None else float("nan")
    remaining = float(remaining)

    # These globals persist into the experiment configuration cell in the same
    # Jupyter kernel and anchor all later session-spend checks.
    SESSION_START_REMAINING_USD = remaining
    SESSION_PROVIDER_KEY_LIMIT_USD = key_limit

    print(
        f"   Provider key cap: "
        f"{('$' + format(key_limit, '.2f')) if key_limit == key_limit else 'not reported'}; "
        f"remaining under key cap: ${remaining:.2f}"
    )
    if key_limit == key_limit and key_limit > SESSION_BUDGET_CEILING_USD:
        WARNINGS.append(
            f"Provider key cap is ${key_limit:.2f}, above this notebook's "
            f"${SESSION_BUDGET_CEILING_USD:.2f} session ceiling. This is allowed "
            "for an externally managed research-program key; notebook-side "
            "session checks will enforce the experiment budget instead."
        )

    if remaining < MIN_REQUIRED_REMAINING_USD:
        raise RuntimeError(
            f"Only ${remaining:.2f} remains under this key, below the "
            f"${MIN_REQUIRED_REMAINING_USD:.2f} preliminary launch floor. "
            "The exact pending-plan check runs in the configuration cell."
        )
except Exception as exc:
    FAILED.append(f"Provider/session budget check failed: {type(exc).__name__}: {exc}")

# 3) 100% local LiveCodeBench execution preflight. ImpossibleBench documents
# Docker as optional (recommended) for LiveCodeBench; this notebook deliberately
# uses local child-Python execution because Docker is unavailable.
try:
    import tempfile
    from pathlib import Path

    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python {sys.version.split()[0]} is too old; use Python 3.10+."
        )

    # Use the exact Jupyter-kernel interpreter, not whatever a shell happens to
    # resolve as `python`. -I enables Python isolated mode for the child.
    safe_env_keys = (
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
        "HOME", "USERPROFILE", "LANG", "LC_ALL",
    )
    child_env = {k: os.environ[k] for k in safe_env_keys if k in os.environ}
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONNOUSERSITE"] = "1"
    # Never expose provider credentials to code under evaluation.
    for secret_name in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "GEMINI_API_KEY",
    ):
        child_env.pop(secret_name, None)

    with tempfile.TemporaryDirectory(prefix="rh_local_probe_") as td:
        probe_code = (
            "import os, pathlib; "
            "assert os.environ.get('OPENROUTER_API_KEY') is None; "
            "assert pathlib.Path.cwd() == pathlib.Path.cwd().resolve(); "
            "print('local-runner-ok')"
        )
        probe = subprocess.run(
            [sys.executable, "-I", "-c", probe_code],
            cwd=td,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode != 0 or "local-runner-ok" not in probe.stdout:
            raise RuntimeError(probe.stderr.strip() or probe.stdout.strip())

    print(f"3. Local verification Python: {sys.executable}")
    print("   Temporary child-process probe: PASSED")
    print("   Provider-key scrubbing probe: PASSED")
except Exception as exc:
    FAILED.append(
        "Local LiveCodeBench child-process verification failed. "
        f"Details: {type(exc).__name__}: {exc}"
    )

# 4) One tiny real Inspect/OpenRouter generation. This is intentionally placed
# after all zero-cost local checks so a broken local setup does not spend API credit.
if not FAILED:
    SMOKE_MODEL = "openrouter/openai/gpt-oss-120b"
    try:
        if available and SMOKE_MODEL.removeprefix("openrouter/") not in available:
            raise RuntimeError(f"{SMOKE_MODEL} is not in the live catalog.")
        from inspect_ai.model import GenerateConfig, get_model
        # GPT-OSS is reasoning-enabled. A 16-token cap can be consumed before
        # visible answer text is emitted, producing an empty completion even when
        # the API request itself succeeded. Use the same low-reasoning policy as
        # the experiment and give this one diagnostic call ample output headroom.
        reply = await get_model(SMOKE_MODEL).generate(
            "Reply with exactly the word: ready",
            config=GenerateConfig(
                max_tokens=1536,
                temperature=0,
                reasoning_effort="low",
            ),
        )
        text = (reply.completion or "").strip()
        print(f"4. Live generation: {text!r}")
        if not text:
            FAILED.append("The smoke model returned empty content.")
        usage = getattr(reply, "usage", None)
        if usage is not None:
            print("   Smoke-call reported cost:", getattr(usage, "total_cost", None))
    except Exception as exc:
        FAILED.append(f"Inspect/OpenRouter generation failed: {type(exc).__name__}: {exc}")
else:
    print("4. Live generation: SKIPPED because a zero-cost diagnostic already failed.")

print("\n" + "=" * 72)
for warning in WARNINGS:
    print("WARNING:", warning)
if FAILED:
    print("BLOCKERS FOUND:")
    for item in FAILED:
        print(" -", item)
    raise RuntimeError("Diagnostic failed; do not start the paid sweep.")
print("DIAGNOSTIC PASSED — API and local LiveCodeBench execution layers are healthy.")
