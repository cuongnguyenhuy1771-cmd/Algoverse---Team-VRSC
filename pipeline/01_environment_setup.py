"""
Stage: Environment setup.

Creates the local runtime directory tree (results/cache/temp files --
kept OUTSIDE this git repo, in ~/rh_contagion_final), installs this
repo's own pinned Python dependencies from requirements.txt, and then
locates + lightly patches YOUR OWN manual clone of the official
ImpossibleBench repository.

ImpossibleBench is not published on PyPI, so it is not pip-installed
from requirements.txt like the other dependencies. Per the official
repo's own install instructions, clone it yourself and install it in
editable mode before running this stage -- see the README's
"Installation" section:

    git clone https://github.com/safety-research/impossiblebench.git
    cd impossiblebench && pip install -e . && cd ..

This must run first. It defines PROJECT_ROOT, RESULTS_ROOT, TEMP_ROOT,
REPO_DIR, IMPOSSIBLEBENCH_REPO_COMMIT, and other globals that every
later stage in this pipeline depends on.

Not meant to be run standalone with `python this_file.py` -- see
run_pipeline.py, which executes every pipeline/*.py file in order
inside one shared namespace (exactly like running these cells in a
single Jupyter kernel, top to bottom).
"""

# =============================================================================
# LOCAL SETUP -- OFFICIAL IMPOSSIBLEBENCH + INSPECT + OPENROUTER
# =============================================================================

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Permanent Windows-safe local runtime root, kept outside the git repo --
# results/cache/temp files don't belong in version control.
#
# IMPORTANT:
# - No SQLite "probe" databases are created anywhere in this notebook.
# - Inspect realtime event logging is disabled later, so Inspect's
#   SampleBufferDatabase is not used either.
# - A short absolute path reduces Windows path/permission edge cases.
PROJECT_ROOT = (Path.home() / "rh_contagion_final").resolve()
RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
TEMP_ROOT = (PROJECT_ROOT / "_tmp").resolve()

for _path in (PROJECT_ROOT, RESULTS_ROOT, TEMP_ROOT):
    _path.mkdir(parents=True, exist_ok=True)

# Keep local child-process paths short and on the same writable user-owned tree.
os.chdir(PROJECT_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMPDIR"] = str(TEMP_ROOT)

# Plain-file write/read probe only. We intentionally KEEP this tiny file:
# no deletion means antivirus/indexer file handles cannot break setup.
_WRITE_PROBE = PROJECT_ROOT / "_filesystem_write_probe.txt"
_WRITE_PROBE.write_text("windows-local-write-ok\n", encoding="utf-8")
if _WRITE_PROBE.read_text(encoding="utf-8").strip() != "windows-local-write-ok":
    raise RuntimeError(f"Local filesystem write/read probe failed at {PROJECT_ROOT}")
print("Windows local filesystem probe: PASSED")


def pip_install(*args: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade-strategy",
            "only-if-needed",
            *args,
        ]
    )


# Installs this repo's own pinned dependencies straight from requirements.txt,
# so the file you edit and the packages actually installed never drift apart.
# ImpossibleBench is deliberately NOT in requirements.txt -- see below.
_REQUIREMENTS_FILE = (Path(__file__).resolve().parents[1] / "requirements.txt")
pip_install("-r", str(_REQUIREMENTS_FILE))

# =============================================================================
# IMPOSSIBLEBENCH -- expects your own manual clone + editable install
# =============================================================================
# Record the exact commit/tag you cloned and installed. This is stored in
# the run metadata (run_plan.json) for reproducibility, but is otherwise
# informational -- it does not control what gets downloaded or installed.
IMPOSSIBLEBENCH_REPO_COMMIT = "<paste `git -C impossiblebench rev-parse HEAD` output here>"

_spec = importlib.util.find_spec("impossiblebench")
if _spec is None or not _spec.origin:
    raise RuntimeError(
        "ImpossibleBench isn't installed. Clone it and install it in editable "
        "mode, then re-run this stage:\n"
        "  git clone https://github.com/safety-research/impossiblebench.git\n"
        "  cd impossiblebench && pip install -e . && cd .."
    )
PACKAGE_DIR = Path(_spec.origin).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[1]  # .../impossiblebench (repo root, above src/)

# Best-effort: report the actual checked-out commit as an FYI cross-check
# against IMPOSSIBLEBENCH_REPO_COMMIT above. Never blocks setup -- a
# mismatch (or no git metadata at all) is just printed, not an error.
_actual_commit = None
try:
    _actual_commit = (
        subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )
    if not IMPOSSIBLEBENCH_REPO_COMMIT.startswith("<") and _actual_commit != IMPOSSIBLEBENCH_REPO_COMMIT:
        print(
            f"NOTE: the installed ImpossibleBench commit ({_actual_commit}) differs "
            f"from IMPOSSIBLEBENCH_REPO_COMMIT ({IMPOSSIBLEBENCH_REPO_COMMIT}) recorded "
            "above -- update that constant if this is expected."
        )
except Exception:
    pass

# Upstream compatibility patch only: make sibling import package-relative.
# Idempotent -- a no-op once already patched, safe to run every time.
mini_agent = PACKAGE_DIR / "livecodebench_agent_mini.py"
mini_text = mini_agent.read_text(encoding="utf-8")
patched_text = mini_text.replace(
    "from livecodebench_scorers import find_code",
    "from .livecodebench_scorers import find_code",
)
if patched_text != mini_text:
    mini_agent.write_text(patched_text, encoding="utf-8")

# Import only the LiveCodeBench components used by this experiment. This avoids
# requiring the heavier SWE-bench execution dependency stack in Stage 1.
# Idempotent -- rewrites to the same content every time.
(PACKAGE_DIR / "__init__.py").write_text(
    """from .livecodebench_tasks import impossible_livecodebench, record_to_sample
from .livecodebench_agent_mini import agentic_humaneval_solver, minimal_humaneval_solver
from .livecodebench_scorers import humaneval_scorer, agentic_humaneval_scorer, find_code
""",
    encoding="utf-8",
)

# Editable installs load straight from your clone, so no sys.path surgery is
# needed here -- just make sure a stale previously-imported copy is dropped
# before re-importing (relevant if this kernel already had it loaded once).
for module_name in list(sys.modules):
    if module_name == "impossiblebench" or module_name.startswith("impossiblebench."):
        del sys.modules[module_name]
importlib.invalidate_caches()

import inspect_ai
import impossiblebench
from impossiblebench import impossible_livecodebench
from impossiblebench.livecodebench_scorers import humaneval_scorer

print("\nSETUP PASSED")
print("Python:", sys.version.split()[0])
print("Inspect:", getattr(inspect_ai, "__version__", "version unavailable"))
print("ImpossibleBench source:", impossiblebench.__file__)
print("ImpossibleBench commit (recorded):", IMPOSSIBLEBENCH_REPO_COMMIT)
if _actual_commit:
    print("ImpossibleBench commit (actual):", _actual_commit)
print("Project root:", PROJECT_ROOT)
print("Results root:", RESULTS_ROOT)
