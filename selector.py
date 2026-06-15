"""
TTS@N candidate selection.
Runs N independent harness candidates and selects the best patch using:
  score = 0.3 * F2P_pass_rate + 0.7 * P2P_pass_rate
Ties broken by shortest patch (raw bytes).
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

import config

console = Console()


@dataclass
class Candidate:
    run_id: int
    patch: str
    work_dir: str
    f2p_tests: str = ""   # content of fail_to_pass_tests.py
    p2p_tests: str = ""   # content of pass_to_pass_tests.txt (node IDs)
    f2p_results: dict[int, float] = field(default_factory=dict)  # run_id → pass_rate
    p2p_results: dict[int, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if not self.f2p_results and not self.p2p_results:
            return 0.0
        f2p = sum(self.f2p_results.values()) / max(len(self.f2p_results), 1)
        p2p = sum(self.p2p_results.values()) / max(len(self.p2p_results), 1)
        return config.TTS_F2P_WEIGHT * f2p + config.TTS_P2P_WEIGHT * p2p


def _run_tests(patch: str, tests_file: str, repo_path: str, timeout: int = 120) -> float:
    """Apply patch, run tests, return pass rate, restore."""
    try:
        # Apply patch
        r = subprocess.run(
            ["git", "apply", "--check"],
            input=patch, text=True, cwd=repo_path, capture_output=True,
        )
        if r.returncode != 0:
            return 0.0
        subprocess.run(["git", "apply"], input=patch, text=True, cwd=repo_path, check=True)

        # Run tests
        result = subprocess.run(
            [sys.executable, "-m", "pytest", tests_file, "-q", "--tb=no", "--no-header"],
            cwd=repo_path, capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout
        # Parse "X passed, Y failed"
        passed = failed = 0
        for line in out.splitlines():
            if "passed" in line or "failed" in line:
                import re
                m_pass = re.search(r"(\d+) passed", line)
                m_fail = re.search(r"(\d+) failed", line)
                if m_pass:
                    passed += int(m_pass.group(1))
                if m_fail:
                    failed += int(m_fail.group(1))
        total = passed + failed
        return passed / total if total > 0 else 0.0

    except Exception:
        return 0.0
    finally:
        subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)


def run_p2p_tests(patch: str, node_ids: list[str], repo_path: str, timeout: int = 120) -> float:
    """Run a list of pytest node IDs after applying patch."""
    if not node_ids:
        return 1.0
    try:
        r = subprocess.run(
            ["git", "apply", "--check"],
            input=patch, text=True, cwd=repo_path, capture_output=True,
        )
        if r.returncode != 0:
            return 0.0
        subprocess.run(["git", "apply"], input=patch, text=True, cwd=repo_path, check=True)

        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + node_ids + ["-q", "--tb=no", "--no-header"],
            cwd=repo_path, capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout
        import re
        passed = failed = 0
        for line in out.splitlines():
            m_pass = re.search(r"(\d+) passed", line)
            m_fail = re.search(r"(\d+) failed", line)
            if m_pass:
                passed += int(m_pass.group(1))
            if m_fail:
                failed += int(m_fail.group(1))
        total = passed + failed
        return passed / total if total > 0 else 0.0
    except Exception:
        return 0.0
    finally:
        subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)


def cross_test_and_select(
    candidates: list[Candidate],
    repo_path: str,
) -> Candidate:
    """
    Cross-test matrix: apply each candidate's tests against every other candidate's patch.
    Select best by weighted score; tie-break on shortest patch.
    """
    console.print(f"\n[bold]Cross-testing {len(candidates)} candidates...[/bold]")

    for i, evaluator in enumerate(candidates):
        f2p_file = Path(evaluator.work_dir) / "_share" / "fail_to_pass_tests.py"
        p2p_file = Path(evaluator.work_dir) / "_share" / "pass_to_pass_tests.txt"
        f2p_tests = str(f2p_file) if f2p_file.exists() else None
        p2p_ids = (
            [l.strip() for l in p2p_file.read_text().splitlines() if l.strip()]
            if p2p_file.exists() else []
        )

        for j, subject in enumerate(candidates):
            if not subject.patch:
                continue
            f2p_rate = _run_tests(subject.patch, f2p_tests, repo_path) if f2p_tests else 0.0
            p2p_rate = run_p2p_tests(subject.patch, p2p_ids, repo_path) if p2p_ids else 1.0

            subject.f2p_results[i] = f2p_rate
            subject.p2p_results[i] = p2p_rate
            console.print(f"  run {j} vs tests-{i}: F2P={f2p_rate:.2f} P2P={p2p_rate:.2f}")

    # Print leaderboard
    table = Table(title="Candidate Scores")
    table.add_column("Run", style="cyan")
    table.add_column("F2P (avg)")
    table.add_column("P2P (avg)")
    table.add_column("Score", style="bold")
    table.add_column("Patch size")
    for c in candidates:
        f2p_avg = sum(c.f2p_results.values()) / max(len(c.f2p_results), 1)
        p2p_avg = sum(c.p2p_results.values()) / max(len(c.p2p_results), 1)
        table.add_row(
            str(c.run_id),
            f"{f2p_avg:.3f}",
            f"{p2p_avg:.3f}",
            f"{c.score:.3f}",
            str(len(c.patch)),
        )
    console.print(table)

    # Select: highest score, tie-break shortest patch
    best = max(candidates, key=lambda c: (c.score, -len(c.patch)))
    console.print(f"[green]Selected run {best.run_id} (score={best.score:.3f})[/green]")
    return best
