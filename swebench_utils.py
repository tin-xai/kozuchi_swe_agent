"""
SWE-bench instance loading and Docker-based evaluation.
Uses the official swebench package for environment setup and test running.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

console = Console()


# ── loading instances ─────────────────────────────────────────────────────────

def load_instances(
    split: str = "verified",
    instance_ids: list[str] | None = None,
) -> list[dict]:
    """
    Load SWE-bench instances from HuggingFace.
    split: "verified" (500 instances) or "lite" (300 instances)
    """
    hf_name = "princeton-nlp/SWE-bench_Verified" if split == "verified" else "princeton-nlp/SWE-bench_Lite"
    console.print(f"Loading {hf_name} from HuggingFace...")
    ds = load_dataset(hf_name, split="test")
    instances = [dict(row) for row in ds]
    if instance_ids:
        instances = [i for i in instances if i["instance_id"] in instance_ids]
    console.print(f"Loaded {len(instances)} instance(s)")
    return instances


# ── lightweight git-based repo setup (no Docker) ─────────────────────────────

def setup_repo(instance: dict, base_dir: str) -> str | None:
    """
    Clone the repo at the base_commit SHA into base_dir/<instance_id>.
    Returns the repo path or None on failure.

    This is the lightweight path (no Docker). For full SWE-bench evaluation
    with Docker isolation, use evaluate_with_docker() instead.
    """
    iid = instance["instance_id"]
    repo_name = instance["repo"]          # e.g. "django/django"
    base_commit = instance["base_commit"]

    repo_dir = Path(base_dir) / iid / "repo"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists():
        console.print(f"  [dim]Repo already exists at {repo_dir}[/dim]")
        return str(repo_dir)

    # Clone from GitHub
    github_url = f"https://github.com/{repo_name}.git"
    console.print(f"  Cloning {repo_name}@{base_commit[:7]}...")
    r = subprocess.run(
        ["git", "clone", "--depth=200", github_url, str(repo_dir)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        console.print(f"  [red]Clone failed: {r.stderr[:200]}[/red]")
        return None

    # Checkout base commit
    r2 = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    if r2.returncode != 0:
        # Try fetching more history
        subprocess.run(["git", "fetch", "--unshallow"], cwd=str(repo_dir), capture_output=True)
        r2 = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(repo_dir), capture_output=True, text=True,
        )
        if r2.returncode != 0:
            console.print(f"  [red]Checkout failed: {r2.stderr[:200]}[/red]")
            return None

    # setuptools provides distutils shim for Python 3.12+ where distutils was removed
    subprocess.run(["pip", "install", "setuptools", "--quiet"], capture_output=True)

    # Install the repo in development mode if setup.py / pyproject.toml exists
    for setup_file in ["setup.py", "pyproject.toml"]:
        if (repo_dir / setup_file).exists():
            subprocess.run(
                ["pip", "install", "-e", ".", "--quiet"],
                cwd=str(repo_dir), capture_output=True,
            )
            break

    console.print(f"  [green]Repo ready at {repo_dir}[/green]")
    return str(repo_dir)


# ── repo structure context ────────────────────────────────────────────────────

def repo_structure_hint(repo_path: str) -> str:
    """
    Return a concise directory tree and test file locations to orient the agent.
    Injected once at the start of the run so the agent never guesses wrong paths.
    """
    rp = Path(repo_path)
    lines: list[str] = []

    # Top-level directories
    top_dirs = sorted(p.name for p in rp.iterdir() if p.is_dir() and not p.name.startswith("."))
    lines.append(f"Top-level dirs: {', '.join(top_dirs)}")

    # Find test directories (up to depth 3)
    test_dirs: list[str] = []
    for p in sorted(rp.rglob("*")):
        if p.is_dir() and "test" in p.name.lower():
            rel = str(p.relative_to(rp))
            if rel.count("/") <= 2:
                test_dirs.append(rel)
    if test_dirs:
        lines.append(f"Test directories: {', '.join(test_dirs[:12])}")

    # Find a sample of test files relevant to auth/validators (common SWE-bench targets)
    sample_tests: list[str] = []
    for p in sorted(rp.rglob("test_*.py")):
        rel = str(p.relative_to(rp))
        if rel.count("/") <= 4:
            sample_tests.append(rel)
    if sample_tests:
        lines.append(f"Test files (sample, up to 20):")
        for t in sample_tests[:20]:
            lines.append(f"  {t}")

    return "\n".join(lines)


# ── Docker-based evaluation (official SWE-bench) ─────────────────────────────

def evaluate_with_docker(
    predictions: list[dict],
    split: str = "verified",
    run_id: str = "harness_run",
    max_workers: int = 4,
) -> dict:
    """
    Run official SWE-bench Docker evaluation.
    predictions: list of {"instance_id": ..., "model_patch": ..., "model_name_or_path": ...}
    Returns evaluation results dict.
    """
    try:
        from swebench.harness.run_evaluation import main as swe_eval
    except ImportError:
        console.print("[red]swebench package not installed. Run: pip install swebench[/red]")
        return {}

    # Write predictions to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
        pred_file = f.name

    results_dir = tempfile.mkdtemp(prefix="swe_results_")
    try:
        swe_eval(
            dataset_name=f"princeton-nlp/SWE-bench_{'Verified' if split=='verified' else 'Lite'}",
            split="test",
            instance_ids=None,
            predictions_path=pred_file,
            max_workers=max_workers,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_flag=False,
            run_id=run_id,
            timeout=1800,
        )
        # Load results
        result_file = Path(results_dir) / f"{run_id}.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return {}
    finally:
        os.unlink(pred_file)


# ── patch formatting ──────────────────────────────────────────────────────────

def detect_repo_env(instance: dict, repo_path: str) -> dict[str, str]:
    """
    Return environment variables needed to run tests in this repo.
    Injected into every bash/line_trace call so the agent never hits
    missing-settings errors.
    """
    repo = instance.get("repo", "")
    rp = Path(repo_path)
    env: dict[str, str] = {}

    # Add repo root to PYTHONPATH so local imports always resolve
    existing = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_path}:{existing}" if existing else repo_path

    # Django: requires DJANGO_SETTINGS_MODULE to run any test or import
    if "django/django" in repo:
        for candidate in ["tests/test_sqlite.py", "tests/settings.py"]:
            if (rp / candidate).exists():
                env["DJANGO_SETTINGS_MODULE"] = candidate.replace("/", ".").removesuffix(".py")
                break
        else:
            env["DJANGO_SETTINGS_MODULE"] = "tests.test_sqlite"

    return env


def format_prediction(instance_id: str, patch: str, model_name: str) -> dict:
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": model_name,
    }


def extract_patch_from_repo(repo_path: str) -> str:
    """Generate git diff of all uncommitted changes."""
    r = subprocess.run(
        ["git", "diff"],
        cwd=repo_path, capture_output=True, text=True,
    )
    return r.stdout
