"""
Self-Harness: automated harness improvement loop.

Mines failure patterns from past runs, proposes minimal prompt improvements
via LLM, and provides LearnedInstanceHarness that injects them at runtime.
Zero changes to any existing file — this module is entirely additive.

Workflow:
  python self_harness.py mine --results-dir results/
  python self_harness.py improve
  python self_harness.py solve --instance django__django-11099
  python self_harness.py show
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import click
from openai import OpenAI
from rich.console import Console

import config
import phases as phases_module
from harness import InstanceHarness
from phases import Phase
from selector import Candidate, cross_test_and_select
from swebench_utils import (
    extract_patch_from_repo,
    format_prediction,
    load_instances,
    setup_repo,
)

console = Console()

LEARNED_CONFIG_PATH = Path("self_harness_config.json")
FAILURES_PATH = Path("self_harness_failures.json")
_EMPTY_CONFIG: dict = {"base_additions": "", "phase_additions": {}, "version": 0, "reasoning": ""}


# ── learned config I/O ────────────────────────────────────────────────────────

def load_learned_config() -> dict:
    if LEARNED_CONFIG_PATH.exists():
        return json.loads(LEARNED_CONFIG_PATH.read_text())
    return _EMPTY_CONFIG.copy()


def save_learned_config(cfg: dict) -> None:
    existing = load_learned_config()
    cfg["version"] = existing.get("version", 0) + 1
    LEARNED_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    console.print(f"[green]Saved learned config v{cfg['version']} → {LEARNED_CONFIG_PATH}[/green]")


# ── LearnedInstanceHarness ────────────────────────────────────────────────────

class LearnedInstanceHarness(InstanceHarness):
    """
    Subclass of InstanceHarness that injects learned prompt improvements
    into the initial message of each phase. harness.py is not modified.

    If no learned config exists yet (fresh run), behaviour is identical
    to the base InstanceHarness with zero overhead.
    """

    def __init__(self, instance: dict, repo_path: str, work_dir: str) -> None:
        super().__init__(instance, repo_path, work_dir)
        self._learned = load_learned_config()

    def _extra_for_phase(self, phase_name: str) -> str:
        parts: list[str] = []
        base = self._learned.get("base_additions", "")
        if base:
            parts.append(base)
        phase_extra = self._learned.get("phase_additions", {}).get(phase_name, "")
        if phase_extra:
            parts.append(phase_extra)
        return "\n".join(parts)

    def _run_phase(self, phase: Phase) -> bool:
        extra = self._extra_for_phase(phase.value)
        if not extra:
            # No learned content — identical to base class, no overhead
            return super()._run_phase(phase)

        # Temporarily augment context_summary to append learned improvements.
        # We patch the bound method on the instance (not the class) so the
        # original is fully restored in the finally block regardless of exceptions.
        _orig = self.state.context_summary
        self.state.context_summary = lambda: (
            _orig() + f"\n\n<learned_improvements>\n{extra}\n</learned_improvements>"
        )
        try:
            return super()._run_phase(phase)
        finally:
            self.state.context_summary = _orig


# ── failure mining ────────────────────────────────────────────────────────────

def mine_failures(results_dir: str) -> dict:
    """
    Scan all work dirs for phase_log.jsonl files and extract failure patterns.
    Supports both results/<instance_id>/run_*/_share/ and results/run_*/_share/.
    """
    rp = Path(results_dir)
    phase_gave_up: Counter = Counter()
    phase_completed: Counter = Counter()
    missing_assets: Counter = Counter()
    total = 0

    log_files = list(rp.glob("*/run_*/_share/phase_log.jsonl"))
    if not log_files:
        log_files = list(rp.glob("run_*/_share/phase_log.jsonl"))

    for log_file in log_files:
        total += 1
        try:
            for raw in log_file.read_text().splitlines():
                if not raw.strip():
                    continue
                ev = json.loads(raw)
                phase = ev.get("phase", "")
                event = ev.get("event", "")

                if event == "complete":
                    phase_completed[phase] += 1
                elif event.startswith("gave_up"):
                    phase_gave_up[phase] += 1
                    m = re.search(r"missing=\[([^\]]*)\]", event)
                    if m:
                        for asset in re.findall(r"'([^']+)'", m.group(1)):
                            missing_assets[asset] += 1
        except Exception:
            continue

    completion_rate = {
        p: round(phase_completed[p] / max(total, 1), 2)
        for p in phase_completed
    }

    return {
        "total_runs": total,
        "phase_gave_up": dict(phase_gave_up.most_common()),
        "phase_completion_rate": completion_rate,
        "missing_assets": dict(missing_assets.most_common()),
    }


# ── LLM improvement proposal ──────────────────────────────────────────────────

def propose_improvements(failure_report: dict) -> dict:
    """
    Ask the LLM to propose minimal, targeted additions to the harness prompts
    based on the mined failure report. Returns the raw proposal dict.
    """
    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )

    phase_list = ", ".join(p.value for p in phases_module.PHASE_ORDER)
    current_base = phases_module.CONDUCTOR_BASE_SYSTEM

    prompt = f"""You are improving an LLM agent harness for SWE-bench (resolving real GitHub issues).
The agent runs through 8 sequential phases: {phase_list}

Failure analysis from {failure_report["total_runs"]} past runs:
{json.dumps(failure_report, indent=2)}

Current base system prompt (first 2000 chars):
---
{current_base[:2000]}
---

Propose MINIMAL targeted additions to fix the most frequent failures.
Rules:
- Do NOT rewrite existing instructions — only add short targeted clarifications
- Each addition must be under 100 words
- Focus on phases with the highest gave_up count and most-missing assets
- If all gave_up counts are zero, return empty strings (no improvement needed)
- Be specific: if an asset is missing often, say exactly how to produce it

Return JSON only:
{{
  "base_additions": "text to append to the base system prompt, or empty string",
  "phase_additions": {{
    "PHASE_NAME": "text to append to that specific phase prompt"
  }},
  "reasoning": "one sentence: the main failure pattern and what you changed"
}}"""

    resp = client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


# ── shared repo reset ─────────────────────────────────────────────────────────

def _reset_repo(repo_path: str) -> None:
    subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_path, capture_output=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Self-Harness: mine → improve → solve with learned prompt additions."""


@cli.command()
@click.option("--results-dir", default="results", show_default=True,
              help="Directory containing past run results")
def mine(results_dir: str):
    """Analyse failure patterns from past runs and save to self_harness_failures.json."""
    console.print(f"[cyan]Mining failures from {results_dir}/...[/cyan]")
    report = mine_failures(results_dir)

    if report["total_runs"] == 0:
        console.print("[yellow]No phase_log.jsonl files found. Run some instances first.[/yellow]")
        return

    console.print(json.dumps(report, indent=2))
    FAILURES_PATH.write_text(json.dumps(report, indent=2))
    console.print(f"[green]Saved → {FAILURES_PATH}[/green]")


@cli.command()
@click.option("--failures-file", default=str(FAILURES_PATH), show_default=True)
def improve(failures_file: str):
    """Propose harness improvements from mined failures and save learned config."""
    fp = Path(failures_file)
    if not fp.exists():
        console.print(f"[red]Run `mine` first — {failures_file} not found[/red]")
        return

    report = json.loads(fp.read_text())
    console.print(
        f"[cyan]Proposing improvements from {report['total_runs']} run(s)...[/cyan]"
    )

    proposal = propose_improvements(report)

    console.print(f"\n[bold]Reasoning:[/bold] {proposal.get('reasoning', '(none)')}")
    base_add = proposal.get("base_additions", "")
    phase_adds = proposal.get("phase_additions", {})
    console.print(f"[bold]Base addition:[/bold] {(base_add[:120] + '…') if len(base_add) > 120 else base_add or '(none)'}")
    console.print(f"[bold]Phase additions:[/bold] {list(phase_adds.keys()) or '(none)'}")

    save_learned_config(proposal)


@cli.command()
@click.option("--instance", "-i", required=True, help="SWE-bench instance_id")
@click.option("--split", default=config.SWEBENCH_SPLIT, show_default=True)
@click.option("--tts", default=1, type=int, show_default=True,
              help="Number of TTS candidate runs")
@click.option("--work-dir", default=None,
              help="Base dir (default: ./results_learned/<instance_id>)")
@click.option("--output", "-o", default="predictions_learned.jsonl", show_default=True)
def solve(instance: str, split: str, tts: int, work_dir: str | None, output: str):
    """Solve one instance using the learned harness."""
    cfg = load_learned_config()
    version = cfg.get("version", 0)
    if not cfg.get("base_additions") and not cfg.get("phase_additions"):
        console.print("[yellow]No learned config yet — running with baseline prompts.[/yellow]")
        console.print("[yellow]Run `mine` then `improve` to build one.[/yellow]")
    else:
        console.print(f"[green]Using learned config v{version}[/green]")
        console.print(f"[dim]Reasoning: {cfg.get('reasoning', '')}[/dim]")

    instances = load_instances(split=split, instance_ids=[instance])
    if not instances:
        console.print(f"[red]Instance '{instance}' not found in split '{split}'[/red]")
        return

    inst = instances[0]
    base_dir = work_dir or str(Path("results_learned") / instance)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    repo_path = setup_repo(inst, base_dir)
    if not repo_path:
        console.print("[red]Repo setup failed[/red]")
        return

    candidates: list[Candidate] = []
    for run_id in range(tts):
        console.print(f"\n[bold cyan]══ TTS run {run_id + 1}/{tts} ══[/bold cyan]")
        run_work = Path(base_dir) / f"run_{run_id}"
        run_work.mkdir(parents=True, exist_ok=True)
        _reset_repo(repo_path)

        harness = LearnedInstanceHarness(inst, repo_path, str(run_work))
        patch = harness.run() or extract_patch_from_repo(repo_path)
        candidates.append(Candidate(run_id=run_id, patch=patch or "", work_dir=str(run_work)))

    best = candidates[0] if tts == 1 else cross_test_and_select(candidates, repo_path)
    model_tag = f"{config.MODEL}+sh-v{version}"
    pred = format_prediction(instance, best.patch, model_tag)

    with open(output, "a") as f:
        f.write(json.dumps(pred) + "\n")

    console.print(f"\n[green]Prediction written to {output}[/green]")
    if best.patch:
        console.print(f"Patch preview:\n{best.patch[:400]}")
    else:
        console.print("[yellow]Warning: empty patch[/yellow]")


@cli.command()
def show():
    """Show current learned_config.json."""
    cfg = load_learned_config()
    if cfg.get("version", 0) == 0:
        console.print("[yellow]No learned config yet.[/yellow]")
    else:
        console.print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    cli()
