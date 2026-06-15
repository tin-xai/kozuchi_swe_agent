"""
CLI entry point for the SWE harness.

Usage examples:
  # Run one instance (fast, TTS@1)
  python run.py solve --instance django__django-11099 --tts 1

  # Run with TTS@8 (full, slower)
  python run.py solve --instance django__django-11099

  # Run a batch from a file (one instance_id per line)
  python run.py batch --ids-file my_ids.txt --tts 1

  # List available verified instances (first 20)
  python run.py list
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import click
from rich.console import Console

import config
from harness import InstanceHarness
from selector import Candidate, cross_test_and_select
from swebench_utils import (
    extract_patch_from_repo,
    format_prediction,
    load_instances,
    setup_repo,
)

console = Console()


@click.group()
def cli():
    """SWE-bench agent harness using OpenRouter."""


@cli.command()
@click.option("--instance", "-i", required=True, help="SWE-bench instance_id")
@click.option("--split", default=config.SWEBENCH_SPLIT, help="verified or lite")
@click.option("--tts", default=config.TTS_N, type=int, help="Number of TTS candidates")
@click.option("--work-dir", default=None, help="Base working directory (default: ./results/<instance_id>)")
@click.option("--output", "-o", default="predictions.jsonl", help="Output predictions file")
def solve(instance: str, split: str, tts: int, work_dir: str | None, output: str):
    """Solve a single SWE-bench instance."""
    instances = load_instances(split=split, instance_ids=[instance])
    if not instances:
        console.print(f"[red]Instance '{instance}' not found in {split}[/red]")
        return

    inst = instances[0]
    # Default to ./results/<instance_id> so data persists across macOS temp cleanups
    base_dir = work_dir or str(Path("results") / instance)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    console.print(f"Working directory: {base_dir}")

    # Setup repo once
    repo_path = setup_repo(inst, base_dir)
    if not repo_path:
        console.print("[red]Repo setup failed[/red]")
        return

    candidates: list[Candidate] = []

    for run_id in range(tts):
        console.print(f"\n[bold cyan]══ TTS run {run_id+1}/{tts} ══[/bold cyan]")

        # Each run gets its own work dir (shared assets)
        run_work = Path(base_dir) / f"run_{run_id}"
        run_work.mkdir(parents=True, exist_ok=True)

        # Reset repo to base commit for each run
        _reset_repo(repo_path)

        harness = InstanceHarness(
            instance=inst,
            repo_path=repo_path,
            work_dir=str(run_work),
        )
        patch = harness.run()

        if patch is None:
            # Try extracting from git diff as fallback
            patch = extract_patch_from_repo(repo_path)

        console.print(f"Run {run_id}: patch size = {len(patch or '')} chars")
        candidates.append(Candidate(
            run_id=run_id,
            patch=patch or "",
            work_dir=str(run_work),
        ))

    # Select best candidate
    if tts == 1:
        best = candidates[0]
        console.print(f"Single run — using run 0 directly.")
    else:
        best = cross_test_and_select(candidates, repo_path)

    # Write prediction
    pred = format_prediction(instance, best.patch, config.MODEL)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a") as f:
        f.write(json.dumps(pred) + "\n")

    # Write manifest so visualize.py can auto-discover the last run
    import time as _time
    manifest = {
        "instance_id": instance,
        "run_dir": str(Path(base_dir) / "run_0"),
        "base_dir": str(base_dir),
        "predictions": str(output),
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path("last_run.json").write_text(json.dumps(manifest, indent=2))

    console.print(f"\n[green]Prediction written to {output}[/green]")
    console.print(f"[dim]Run data at: {base_dir}[/dim]")
    if best.patch:
        console.print(f"Patch preview (first 500 chars):\n{best.patch[:500]}")
    else:
        console.print("[yellow]Warning: empty patch[/yellow]")


@cli.command()
@click.option("--ids-file", required=True, type=click.Path(exists=True))
@click.option("--split", default=config.SWEBENCH_SPLIT)
@click.option("--tts", default=1, type=int, help="TTS candidates per instance (default 1 for batch)")
@click.option("--work-dir", default=None)
@click.option("--output", "-o", default="predictions.jsonl")
def batch(ids_file: str, split: str, tts: int, work_dir: str | None, output: str):
    """Solve a batch of instances from a file (one instance_id per line)."""
    ids = [l.strip() for l in Path(ids_file).read_text().splitlines() if l.strip()]
    console.print(f"Batch: {len(ids)} instance(s), TTS={tts}")

    base_dir = work_dir or tempfile.mkdtemp(prefix="swe_batch_")
    instances = load_instances(split=split, instance_ids=ids)

    results = []
    for i, inst in enumerate(instances):
        console.print(f"\n[bold]═══ [{i+1}/{len(instances)}] {inst['instance_id']} ═══[/bold]")
        iid = inst["instance_id"]
        inst_dir = Path(base_dir) / iid
        inst_dir.mkdir(parents=True, exist_ok=True)

        repo_path = setup_repo(inst, str(inst_dir))
        if not repo_path:
            console.print("[red]Skipping — repo setup failed[/red]")
            continue

        candidates = []
        for run_id in range(tts):
            run_work = inst_dir / f"run_{run_id}"
            run_work.mkdir(parents=True, exist_ok=True)
            _reset_repo(repo_path)

            harness = InstanceHarness(inst, repo_path, str(run_work))
            patch = harness.run() or extract_patch_from_repo(repo_path)
            candidates.append(Candidate(run_id, patch or "", str(run_work)))

        best = candidates[0] if tts == 1 else cross_test_and_select(candidates, repo_path)
        pred = format_prediction(iid, best.patch, config.MODEL)
        results.append(pred)

        with open(output, "a") as f:
            f.write(json.dumps(pred) + "\n")

    console.print(f"\n[green]Done. {len(results)}/{len(instances)} predictions in {output}[/green]")


@cli.command("list")
@click.option("--split", default=config.SWEBENCH_SPLIT)
@click.option("--n", default=20, type=int, help="How many to show")
def list_instances(split: str, n: int):
    """List available SWE-bench instances."""
    instances = load_instances(split=split)
    for inst in instances[:n]:
        console.print(f"  {inst['instance_id']}  ({inst['repo']})")
    console.print(f"\n... total {len(instances)} instances")


@cli.command()
@click.option("--predictions", "-p", required=True, type=click.Path(exists=True))
@click.option("--split", default=config.SWEBENCH_SPLIT)
@click.option("--run-id", default="harness_eval")
def evaluate(predictions: str, split: str, run_id: str):
    """Run official SWE-bench Docker evaluation on a predictions file."""
    from swebench_utils import evaluate_with_docker

    preds = [json.loads(l) for l in Path(predictions).read_text().splitlines() if l.strip()]
    console.print(f"Evaluating {len(preds)} prediction(s) with Docker...")
    results = evaluate_with_docker(preds, split=split, run_id=run_id)
    console.print(json.dumps(results, indent=2))


def _reset_repo(repo_path: str):
    """Reset repo to committed state (discard any agent changes)."""
    subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_path, capture_output=True)


if __name__ == "__main__":
    cli()
