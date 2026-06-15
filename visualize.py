"""
Result visualiser for the SWE harness.

Commands:
  python visualize.py run   --work-dir <run_0 dir>       # single run dashboard
  python visualize.py batch --predictions predictions.jsonl --work-dir <base dir>
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

PHASE_COLORS = {
    "ISSUE_REPRODUCT": "#3498db",
    "TEST_SYNTHSIZE":  "#9b59b6",
    "CODE_LOCALIZE":   "#e67e22",
    "TEST_LOCALIZE":   "#1abc9c",
    "CODE_FIX":        "#e74c3c",
    "VERIFY_PATCH":    "#f39c12",
    "ISSUE_CLOSE":     "#2ecc71",
    "FINAL_REPORT":    "#95a5a6",
}


# ── data loaders ─────────────────────────────────────────────────────────────

def load_phase_log(work_dir: Path) -> list[dict]:
    p = work_dir / "_share" / "phase_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def parse_phase_timings(events: list[dict]) -> list[dict]:
    """Return [{phase, start, end, duration_s, status}]."""
    pairs: dict[str, dict] = {}
    for e in events:
        pairs.setdefault(e["phase"], {})[e["event"]] = datetime.fromisoformat(e["ts"])
    result = []
    for phase, times in pairs.items():
        if "start" in times and "complete" in times:
            dur = (times["complete"] - times["start"]).total_seconds()
            result.append({"phase": phase, "start": times["start"],
                            "end": times["complete"], "duration_s": dur, "status": "complete"})
        elif "start" in times:
            result.append({"phase": phase, "start": times["start"],
                            "end": times["start"], "duration_s": 0, "status": "incomplete"})
    return result


def parse_patch(patch: str) -> dict:
    """Extract stats from a unified diff."""
    files, added, removed = [], 0, 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"files": files, "added": added, "removed": removed, "total": added + removed}


def load_predictions(pred_file: Path) -> list[dict]:
    if not pred_file.exists():
        return []
    return [json.loads(l) for l in pred_file.read_text().splitlines() if l.strip()]


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_phase_gantt(timings: list[dict], ax: plt.Axes, title: str = "Phase Timeline"):
    if not timings:
        ax.text(0.5, 0.5, "No phase data", ha="center", va="center")
        return

    t0 = min(t["start"] for t in timings)
    phases = [t["phase"] for t in timings]
    y_pos = range(len(phases))

    for i, t in enumerate(timings):
        start_s = (t["start"] - t0).total_seconds()
        dur = max(t["duration_s"], 1)
        color = PHASE_COLORS.get(t["phase"], "#bdc3c7")
        ax.barh(i, dur, left=start_s, color=color, alpha=0.85, height=0.6)
        mins = int(dur // 60)
        secs = int(dur % 60)
        label = f"{mins}m{secs:02d}s"
        ax.text(start_s + dur + 2, i, label, va="center", fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([t["phase"].replace("_", "\n") for t in timings], fontsize=8)
    ax.set_xlabel("Elapsed seconds")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    total = sum(t["duration_s"] for t in timings)
    ax.set_xlim(right=total * 1.15)


def plot_phase_durations_bar(timings: list[dict], ax: plt.Axes, title: str = "Phase Durations"):
    if not timings:
        ax.text(0.5, 0.5, "No phase data", ha="center", va="center")
        return

    phases = [t["phase"] for t in timings]
    durations_min = [t["duration_s"] / 60 for t in timings]
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]
    bars = ax.bar(range(len(phases)), durations_min, color=colors, alpha=0.85)

    for bar, dur in zip(bars, durations_min):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{dur:.1f}m", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=7)
    ax.set_ylabel("Minutes")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def plot_patch_stats(patch_info: dict, ax: plt.Axes, instance_id: str = ""):
    if not patch_info["files"]:
        ax.text(0.5, 0.5, "No patch", ha="center", va="center")
        ax.axis("off")
        return

    files = patch_info["files"]
    # Per-file line counts not available from summary — show aggregate
    categories = ["Lines added", "Lines removed", "Net change"]
    values = [patch_info["added"], patch_info["removed"],
              patch_info["added"] - patch_info["removed"]]
    colors = ["#2ecc71", "#e74c3c", "#3498db" if values[2] >= 0 else "#e74c3c"]

    bars = ax.bar(categories, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.5,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    title = f"Patch Stats — {instance_id}" if instance_id else "Patch Stats"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Lines")
    ax.grid(axis="y", alpha=0.3)

    # File list as text below
    file_text = "Files changed:\n" + "\n".join(f"  • {f}" for f in files[:8])
    if len(files) > 8:
        file_text += f"\n  ... +{len(files)-8} more"
    ax.text(0.98, 0.98, file_text, transform=ax.transAxes,
            fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.8))


def plot_llm_call_estimate(timings: list[dict], ax: plt.Axes):
    if not timings:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    phases = [t["phase"] for t in timings]
    # ~8.8s per LLM call based on observed average
    calls = [round(t["duration_s"] / 8.8) for t in timings]
    colors = [PHASE_COLORS.get(p, "#bdc3c7") for p in phases]

    bars = ax.barh(range(len(phases)), calls, color=colors, alpha=0.85)
    for bar, c in zip(bars, calls):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=8)

    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([p.replace("_", "\n") for p in phases], fontsize=7)
    ax.set_xlabel("Estimated LLM calls")
    ax.set_title("LLM Calls per Phase (~8.8s/call)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(right=max(calls) * 1.15)


def plot_tts_scores(candidates: list[dict], ax: plt.Axes):
    """candidates: [{run_id, f2p, p2p, score, patch_size}]"""
    if not candidates:
        ax.text(0.5, 0.5, "TTS@1 — no comparison", ha="center", va="center")
        ax.axis("off")
        return

    ids = [f"Run {c['run_id']}" for c in candidates]
    f2p = [c.get("f2p", 0) for c in candidates]
    p2p = [c.get("p2p", 0) for c in candidates]
    scores = [c.get("score", 0) for c in candidates]

    x = np.arange(len(ids))
    w = 0.25
    ax.bar(x - w, f2p, w, label="F2P (×0.3)", color="#e74c3c", alpha=0.8)
    ax.bar(x,     p2p, w, label="P2P (×0.7)", color="#2ecc71", alpha=0.8)
    ax.bar(x + w, scores, w, label="Score", color="#3498db", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("TTS Candidate Scores", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ── single run dashboard ──────────────────────────────────────────────────────

def _resolve_run_dir(work_dir: Path | None) -> tuple[Path, str]:
    """Return (run_dir, instance_id), auto-discovering from last_run.json if needed."""
    if work_dir is None:
        manifest = Path("last_run.json")
        if not manifest.exists():
            raise click.ClickException("No --work-dir given and last_run.json not found. Run solve first.")
        m = json.loads(manifest.read_text())
        return Path(m["run_dir"]), m["instance_id"]

    # Work out instance_id from directory name
    wd = work_dir
    if wd.name.startswith("run_"):
        # e.g. results/django__django-11099/run_0  → parent is instance_id
        instance_id = wd.parent.name
    else:
        instance_id = wd.name
    return wd, instance_id


def single_run_dashboard(work_dir: Path | None, output: Path):
    run_dir, instance_id = _resolve_run_dir(work_dir)
    events = load_phase_log(run_dir)
    timings = parse_phase_timings(events)

    patch_text = (run_dir / "_share" / "patch.diff").read_text() \
        if (run_dir / "_share" / "patch.diff").exists() else ""
    patch_info = parse_patch(patch_text)

    console.print(f"Loading from: {run_dir}")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"SWE Harness Run — {instance_id}", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_phase_gantt(timings, fig.add_subplot(gs[0, :2]), "Phase Timeline (Gantt)")
    plot_phase_durations_bar(timings, fig.add_subplot(gs[1, 0]))
    plot_llm_call_estimate(timings, fig.add_subplot(gs[1, 1]))
    plot_patch_stats(patch_info, fig.add_subplot(gs[0, 2]), instance_id)

    # Summary text box
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.axis("off")
    total_s = sum(t["duration_s"] for t in timings)
    complete = sum(1 for t in timings if t["status"] == "complete")
    summary = (
        f"Instance:   {instance_id}\n"
        f"Phases:     {complete}/8 complete\n"
        f"Total time: {int(total_s//60)}m {int(total_s%60):02d}s\n"
        f"Est. calls: ~{round(total_s/8.8) if total_s else 0}\n\n"
        f"Patch:\n"
        f"  Files:    {len(patch_info['files'])}\n"
        f"  Added:    +{patch_info['added']} lines\n"
        f"  Removed:  -{patch_info['removed']} lines\n"
        f"\nRun dir:\n  {str(run_dir)[-40:]}"
    )
    ax_sum.text(0.05, 0.95, summary, transform=ax_sum.transAxes,
                fontsize=9, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.9))
    ax_sum.set_title("Summary", fontsize=11, fontweight="bold")

    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved: {output}[/green]")


# ── batch dashboard ───────────────────────────────────────────────────────────

def batch_dashboard(pred_file: Path, base_dir: Path | None, output: Path):
    preds = load_predictions(pred_file)
    if not preds:
        console.print("[red]No predictions found.[/red]")
        return

    # Collect stats per instance
    rows = []
    for p in preds:
        iid = p["instance_id"]
        patch_info = parse_patch(p.get("model_patch", ""))
        total_s = 0
        phases_done = 0

        if base_dir:
            # Try to find the run_0 dir
            for candidate in [
                base_dir / iid / "run_0",
                base_dir / "run_0",
            ]:
                if candidate.exists():
                    events = load_phase_log(candidate)
                    timings = parse_phase_timings(events)
                    total_s = sum(t["duration_s"] for t in timings)
                    phases_done = sum(1 for t in timings if t["status"] == "complete")
                    break

        rows.append({
            "instance_id": iid,
            "phases": phases_done,
            "duration_m": total_s / 60,
            "files": len(patch_info["files"]),
            "added": patch_info["added"],
            "removed": patch_info["removed"],
            "has_patch": patch_info["total"] > 0,
        })

    # ── rich table ──
    table = Table(title=f"Batch Results ({len(rows)} instances)")
    table.add_column("Instance", style="cyan")
    table.add_column("Phases", justify="right")
    table.add_column("Time (m)", justify="right")
    table.add_column("Files Δ", justify="right")
    table.add_column("+Lines", justify="right", style="green")
    table.add_column("-Lines", justify="right", style="red")
    table.add_column("Patch?")
    for r in rows:
        table.add_row(
            r["instance_id"], str(r["phases"]),
            f"{r['duration_m']:.1f}", str(r["files"]),
            f"+{r['added']}", f"-{r['removed']}",
            "✓" if r["has_patch"] else "✗",
        )
    console.print(table)

    # ── plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Batch Results — {len(rows)} instances", fontsize=13, fontweight="bold")

    # 1. Phase completion
    phases_counts = [r["phases"] for r in rows]
    axes[0].bar(range(len(rows)), phases_counts, color=[
        "#2ecc71" if p == 8 else "#e67e22" if p >= 5 else "#e74c3c"
        for p in phases_counts
    ], alpha=0.85)
    axes[0].axhline(8, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels([r["instance_id"].split("__")[-1][:12] for r in rows],
                             rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Phases completed")
    axes[0].set_title("Phase Completion per Instance")
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Duration distribution
    durations = [r["duration_m"] for r in rows if r["duration_m"] > 0]
    if durations:
        axes[1].hist(durations, bins=min(10, len(durations)), color="#3498db", alpha=0.8, edgecolor="white")
        axes[1].axvline(np.mean(durations), color="#e74c3c", linestyle="--",
                        label=f"mean {np.mean(durations):.1f}m")
        axes[1].set_xlabel("Duration (minutes)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Run Duration Distribution")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

    # 3. Patch size distribution
    patch_sizes = [r["added"] + r["removed"] for r in rows if r["has_patch"]]
    if patch_sizes:
        axes[2].hist(patch_sizes, bins=min(10, len(patch_sizes)),
                     color="#9b59b6", alpha=0.8, edgecolor="white")
        axes[2].axvline(np.mean(patch_sizes), color="#e74c3c", linestyle="--",
                        label=f"mean {np.mean(patch_sizes):.0f} lines")
        axes[2].set_xlabel("Total lines changed")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Patch Size Distribution")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved: {output}[/green]")


# ── predictions visualiser ───────────────────────────────────────────────────

def predictions_dashboard(pred_file: Path, output: Path):
    preds = load_predictions(pred_file)
    if not preds:
        console.print(f"[red]No predictions in {pred_file}[/red]")
        return

    # Group by instance_id (multiple runs may produce duplicates)
    from collections import defaultdict
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        grouped[p["instance_id"]].append(p)

    # ── rich table ──
    table = Table(title=f"predictions.jsonl — {len(preds)} entries, {len(grouped)} unique instances")
    table.add_column("#", justify="right", style="dim")
    table.add_column("instance_id", style="cyan")
    table.add_column("model")
    table.add_column("files Δ", justify="right")
    table.add_column("+lines", justify="right", style="green")
    table.add_column("-lines", justify="right", style="red")
    table.add_column("patch size", justify="right")
    table.add_column("duplicate", justify="center")

    rows = []
    for idx, p in enumerate(preds):
        info = parse_patch(p.get("model_patch", ""))
        is_dup = len(grouped[p["instance_id"]]) > 1
        rows.append({
            "idx": idx + 1,
            "instance_id": p["instance_id"],
            "model": p.get("model_name_or_path", "?"),
            "files": len(info["files"]),
            "added": info["added"],
            "removed": info["removed"],
            "total": info["total"],
            "has_patch": info["total"] > 0,
            "is_dup": is_dup,
            "patch": p.get("model_patch", ""),
            "patch_info": info,
        })
        table.add_row(
            str(idx + 1),
            p["instance_id"],
            p.get("model_name_or_path", "?")[-30:],
            str(len(info["files"])),
            f"+{info['added']}",
            f"-{info['removed']}",
            f"{info['total']} lines",
            "[yellow]yes[/yellow]" if is_dup else "",
        )
    console.print(table)

    if not rows:
        return

    # ── figure layout ──
    n_unique = len(grouped)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"predictions.jsonl — {len(preds)} predictions, {n_unique} unique instances",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Patch size per prediction (bar, coloured by duplicate)
    ax1 = fig.add_subplot(gs[0, :2])
    labels = [f"{r['idx']}. {r['instance_id'].split('__')[-1][:18]}" for r in rows]
    sizes = [r["total"] for r in rows]
    colors = ["#e67e22" if r["is_dup"] else "#3498db" for r in rows]
    bars = ax1.bar(range(len(rows)), sizes, color=colors, alpha=0.85)
    for bar, sz in zip(bars, sizes):
        if sz > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(sz), ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Total lines changed")
    ax1.set_title("Patch Size per Prediction")
    ax1.grid(axis="y", alpha=0.3)
    legend_handles = [
        mpatches.Patch(color="#3498db", label="unique"),
        mpatches.Patch(color="#e67e22", label="duplicate instance"),
    ]
    ax1.legend(handles=legend_handles, fontsize=8)

    # 2. Added vs removed split
    ax2 = fig.add_subplot(gs[0, 2])
    added_vals = [r["added"] for r in rows]
    removed_vals = [r["removed"] for r in rows]
    x = np.arange(len(rows))
    w = 0.4
    ax2.bar(x - w/2, added_vals, w, color="#2ecc71", alpha=0.8, label="+added")
    ax2.bar(x + w/2, removed_vals, w, color="#e74c3c", alpha=0.8, label="-removed")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(r["idx"]) for r in rows], fontsize=8)
    ax2.set_xlabel("Prediction #")
    ax2.set_ylabel("Lines")
    ax2.set_title("Added vs Removed Lines")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Diff preview for each prediction (text boxes)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")

    # Show diff hunks side by side (up to 4 predictions)
    preview_rows = rows[:4]
    n = len(preview_rows)
    for i, r in enumerate(preview_rows):
        x_start = i / n
        x_end = (i + 1) / n - 0.01
        # Render first ~20 lines of diff
        diff_lines = r["patch"].splitlines()[:22]
        diff_text = "\n".join(diff_lines)
        if len(r["patch"].splitlines()) > 22:
            diff_text += "\n..."

        # Colour header
        header_color = "#e67e22" if r["is_dup"] else "#2c3e50"
        ax3.text(
            x_start + 0.005, 0.97,
            f"#{r['idx']} {r['instance_id']}",
            transform=ax3.transAxes,
            fontsize=8, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=header_color, alpha=0.9),
        )
        ax3.text(
            x_start + 0.005, 0.88,
            diff_text,
            transform=ax3.transAxes,
            fontsize=6.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#f8f9fa", alpha=0.9),
            clip_on=True,
        )

    ax3.set_title("Diff Preview (first 4 predictions)", fontsize=11, fontweight="bold", pad=8)

    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved: {output}[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Visualise SWE harness results."""


@cli.command()
@click.option("--work-dir", default=None, type=click.Path(),
              help="Path to run_N directory (auto-detected from last_run.json if omitted)")
@click.option("--output", "-o", default="run_dashboard.png")
def run(work_dir: str | None, output: str):
    """Dashboard for a single run. Uses last_run.json if --work-dir is omitted."""
    single_run_dashboard(Path(work_dir) if work_dir else None, Path(output))


@cli.command()
@click.option("--predictions", "-p", default="predictions.jsonl", type=click.Path())
@click.option("--work-dir", default=None, type=click.Path(),
              help="Base directory containing per-instance subdirs (optional, for timing data)")
@click.option("--output", "-o", default="batch_dashboard.png")
def batch(predictions: str, work_dir: str | None, output: str):
    """Dashboard for a batch of predictions."""
    batch_dashboard(Path(predictions), Path(work_dir) if work_dir else None, Path(output))


@cli.command()
@click.option("--predictions", "-p", default="predictions.jsonl", type=click.Path())
@click.option("--output", "-o", default="predictions_dashboard.png")
def preds(predictions: str, output: str):
    """Visualise predictions.jsonl — patch sizes, diffs, duplicates."""
    predictions_dashboard(Path(predictions), Path(output))


# ── instance comparison (Figure 6 style) ─────────────────────────────────────

def _diff_lines(diff_text: str, max_lines: int = 30) -> list[tuple[str, str, str]]:
    """Parse diff into (text, bg_color, fg_color) tuples."""
    result = []
    for line in diff_text.splitlines()[:max_lines]:
        if line.startswith("+") and not line.startswith("+++"):
            result.append((line, "#d4edda", "#155724"))
        elif line.startswith("-") and not line.startswith("---"):
            result.append((line, "#f8d7da", "#721c24"))
        elif line.startswith("@@"):
            result.append((line, "#cce5ff", "#004085"))
        elif line.startswith(("---", "+++")):
            result.append((line, "#e2e3e5", "#383d41"))
        else:
            result.append((line, "#f8f9fa", "#212529"))
    return result


def _render_lines(ax: plt.Axes, lines: list[tuple[str, str, str]],
                  title: str, subtitle: str = "", fontsize: int = 7):
    """Render colored lines into an axes as a fake code panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06,
                                facecolor="#2c3e50", transform=ax.transAxes, clip_on=False))
    ax.text(0.01, 0.97, title, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="white", va="center")

    if subtitle:
        ax.add_patch(plt.Rectangle((0, 0.89), 1, 0.05,
                                    facecolor="#ecf0f1", transform=ax.transAxes, clip_on=False))
        ax.text(0.01, 0.915, subtitle, transform=ax.transAxes,
                fontsize=8, color="#555", va="center", fontfamily="monospace")

    body_top = 0.88 if subtitle else 0.93
    n = max(len(lines), 1)
    line_h = body_top / n

    for i, (text, bg, fg) in enumerate(lines):
        y_top = body_top - i * line_h
        y_bot = y_top - line_h
        ax.add_patch(plt.Rectangle((0, y_bot), 1, line_h,
                                    facecolor=bg, transform=ax.transAxes, clip_on=True))
        ax.text(0.008, y_top - line_h * 0.15, text[:95],
                transform=ax.transAxes, fontsize=fontsize,
                fontfamily="monospace", color=fg, va="top", clip_on=True)

    # Border
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#bdc3c7")
        ax.spines[spine].set_linewidth(0.8)


def _render_text_panel(ax: plt.Axes, sections: list[tuple[str, str, int]],
                        panel_title: str):
    """
    Render a model-input style panel.
    sections: [(section_title, body_text, bullet_count), ...]
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06,
                                facecolor="#2c3e50", transform=ax.transAxes, clip_on=False))
    ax.text(0.01, 0.97, panel_title, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="white", va="center")

    y = 0.91
    for sec_title, body, bullet in sections:
        ax.text(0.01, y, f"▼ {sec_title}",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#2c3e50", va="top")
        if bullet:
            ax.text(0.97, y, f"• {bullet} lines",
                    transform=ax.transAxes, fontsize=8, color="#888", va="top", ha="right")
        y -= 0.04

        lines = body.splitlines()[:14]
        for line in lines:
            is_code = line.startswith((" ", "\t", "def ", "class ", "   "))
            ax.text(0.015, y, line[:88],
                    transform=ax.transAxes, fontsize=8,
                    fontfamily="monospace" if is_code else "sans-serif",
                    color="#333", va="top", clip_on=True)
            y -= 0.032
            if y < 0.02:
                ax.text(0.015, y + 0.032, "...", transform=ax.transAxes,
                        fontsize=8, color="#888", va="top")
                break
        y -= 0.015

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#bdc3c7")
        ax.spines[spine].set_linewidth(0.8)


def _render_test_results(ax: plt.Axes, verify_log: str, title: str = "Test Results"):
    """Render PASSED/FAILED test lines like the paper's bottom-right panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0.92), 1, 0.08,
                                facecolor="#2c3e50", transform=ax.transAxes, clip_on=False))
    ax.text(0.01, 0.96, title, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="white", va="center")

    if not verify_log.strip():
        ax.text(0.5, 0.5, "No test results", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontsize=9)
        return

    lines = [l for l in verify_log.splitlines() if l.strip()][:18]
    n = max(len(lines), 1)
    line_h = 0.90 / n
    y_top = 0.90

    for i, line in enumerate(lines):
        y = y_top - i * line_h
        if "PASSED" in line or "passed" in line.lower():
            badge_col, text_col, bg = "#27ae60", "#155724", "#d4edda"
            badge = "PASSED"
        elif "FAILED" in line or "failed" in line.lower() or "error" in line.lower():
            badge_col, text_col, bg = "#c0392b", "#721c24", "#f8d7da"
            badge = "FAILED"
        elif line.startswith("="):
            badge_col, text_col, bg = "#7f8c8d", "#555", "#f0f0f0"
            badge = ""
        else:
            badge_col, text_col, bg = "#95a5a6", "#555", "#f8f9fa"
            badge = ""

        ax.add_patch(plt.Rectangle((0, y - line_h), 1, line_h,
                                    facecolor=bg, transform=ax.transAxes, clip_on=True))

        x_offset = 0.01
        if badge:
            ax.add_patch(plt.FancyBboxPatch((x_offset, y - line_h * 0.85), 0.10, line_h * 0.7,
                                             boxstyle="round,pad=0.01",
                                             facecolor=badge_col, transform=ax.transAxes,
                                             clip_on=True))
            ax.text(x_offset + 0.05, y - line_h * 0.5, badge,
                    transform=ax.transAxes, fontsize=6.5, fontweight="bold",
                    color="white", va="center", ha="center")
            x_offset = 0.13

        rest = re.sub(r"(PASSED|FAILED)\s*", "", line).strip()
        ax.text(x_offset, y - line_h * 0.2, rest[:85],
                transform=ax.transAxes, fontsize=8,
                fontfamily="monospace", color=text_col, va="top", clip_on=True)

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#bdc3c7")
        ax.spines[spine].set_linewidth(0.8)


def instance_comparison(instance_id: str, pred_file: Path,
                         run_dir: Path | None, output: Path):
    """Render Figure-6-style comparison: input | gold patch | generated patch | tests."""
    # Load prediction
    preds = [p for p in load_predictions(pred_file) if p["instance_id"] == instance_id]
    if not preds:
        console.print(f"[red]No prediction found for {instance_id}[/red]")
        return
    pred = preds[-1]   # use latest if duplicates
    gen_patch = pred.get("model_patch", "")

    # Load SWE-bench instance for gold patch + issue
    try:
        from swebench_utils import load_instances
        instances = load_instances(instance_ids=[instance_id])
        inst = instances[0] if instances else {}
    except Exception:
        inst = {}

    gold_patch   = inst.get("patch", "")
    issue_text   = inst.get("problem_statement", "No issue text available.")
    repo         = inst.get("repo", "")

    # Load verify log
    verify_log = ""
    if run_dir:
        vlog = run_dir / "_share" / "verify_log.txt"
        if vlog.exists():
            verify_log = vlog.read_text()
    if not verify_log:
        verify_log = "(No test results — run VERIFY_PATCH phase to generate)"

    # Load code context (first file in gold patch)
    code_context = ""
    if run_dir:
        loc = run_dir / "_share" / "localize_notes.md"
        if loc.exists():
            code_context = loc.read_text()[:400]

    # ── figure ──
    fig = plt.figure(figsize=(26, 16))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"SWE-bench Instance: {instance_id}  |  model: {pred.get('model_name_or_path','')}",
                 fontsize=13, fontweight="bold", color="#2c3e50", y=0.995)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.01, right=0.99, top=0.97, bottom=0.01,
                           hspace=0.08, wspace=0.035,
                           width_ratios=[1, 1.4])

    # Left panel: model input (spans all 3 rows)
    ax_input = fig.add_subplot(gs[:, 0])
    issue_short = issue_text[:600]
    sections = [
        ("Instructions", "You will be provided with a partial code base and an issue\n"
                          "statement explaining a problem to resolve.", 1),
        ("Issue", issue_short, len(issue_text.splitlines())),
    ]
    if code_context:
        sections.append(("Localization Notes", code_context, len(code_context.splitlines())))
    _render_text_panel(ax_input, sections, "Model Input")

    # Top-right: gold patch
    ax_gold = fig.add_subplot(gs[0, 1])
    gold_file = next((l[6:] for l in gold_patch.splitlines() if l.startswith("+++ b/")), "")
    _render_lines(ax_gold, _diff_lines(gold_patch, 22), "Gold Patch", gold_file, fontsize=8)

    # Mid-right: generated patch
    ax_gen = fig.add_subplot(gs[1, 1])
    gen_file = next((l[6:] for l in gen_patch.splitlines() if l.startswith("+++ b/")), "")
    _render_lines(ax_gen, _diff_lines(gen_patch, 22), "Generated Patch", gen_file, fontsize=8)

    # Bottom-right: test results
    ax_tests = fig.add_subplot(gs[2, 1])
    _render_test_results(ax_tests, verify_log, "Generated Patch Test Results")

    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    console.print(f"[green]Saved: {output}[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--instance", "-i", required=True, help="SWE-bench instance_id")
@click.option("--predictions", "-p", default="predictions.jsonl")
@click.option("--work-dir", default=None, type=click.Path(),
              help="run_N directory for verify log (auto-detected from last_run.json)")
@click.option("--output", "-o", default=None)
def compare(instance: str, predictions: str, work_dir: str | None, output: str | None):
    """Figure-6-style comparison: model input | gold | generated | test results."""
    if work_dir is None and Path("last_run.json").exists():
        m = json.loads(Path("last_run.json").read_text())
        work_dir = m.get("run_dir")
    out = Path(output) if output else Path(f"compare_{instance}.png")
    instance_comparison(instance, Path(predictions),
                         Path(work_dir) if work_dir else None, out)


# ── HTML comparison ───────────────────────────────────────────────────────────

def _html_diff(diff_text: str) -> str:
    """Convert unified diff to HTML with coloured lines."""
    import html as _html
    lines_html = []
    for line in diff_text.splitlines():
        escaped = _html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            cls = "add"
        elif line.startswith("-") and not line.startswith("---"):
            cls = "del"
        elif line.startswith("@@"):
            cls = "hunk"
        elif line.startswith(("---", "+++")):
            cls = "meta"
        else:
            cls = "ctx"
        lines_html.append(f'<div class="diff-line {cls}">{escaped}</div>')
    return "\n".join(lines_html)


def _html_tests(verify_log: str) -> str:
    import html as _html
    if not verify_log.strip():
        return '<p class="muted">No test results — run VERIFY_PATCH phase to generate.</p>'
    rows = []
    for line in verify_log.splitlines():
        if not line.strip():
            continue
        escaped = _html.escape(line)
        if "PASSED" in line:
            badge = '<span class="badge pass">PASSED</span>'
            rest = escaped.replace("PASSED", "").strip()
            rows.append(f'<div class="test-row pass-row">{badge} {rest}</div>')
        elif "FAILED" in line or "ERROR" in line:
            badge = '<span class="badge fail">FAILED</span>'
            rest = escaped.replace("FAILED", "").replace("ERROR", "").strip()
            rows.append(f'<div class="test-row fail-row">{badge} {rest}</div>')
        elif line.startswith("="):
            rows.append(f'<div class="test-summary">{escaped}</div>')
        else:
            rows.append(f'<div class="test-row">{escaped}</div>')
    return "\n".join(rows)


def _html_issue(text: str) -> str:
    import html as _html
    lines = []
    for line in text.splitlines():
        escaped = _html.escape(line)
        if line.startswith(("def ", "class ", "    ", "\t")):
            lines.append(f'<code>{escaped}</code><br>')
        elif line.startswith("###"):
            lines.append(f'<strong>{escaped}</strong><br>')
        elif not line.strip():
            lines.append('<br>')
        else:
            lines.append(f'{escaped}<br>')
    return "\n".join(lines)


def instance_comparison_html(instance_id: str, pred_file: Path,
                              run_dir: Path | None, output: Path):
    """Render a Figure-6-style HTML comparison report."""
    import html as _html

    preds = [p for p in load_predictions(pred_file) if p["instance_id"] == instance_id]
    if not preds:
        console.print(f"[red]No prediction found for {instance_id}[/red]")
        return
    pred = preds[-1]
    gen_patch = pred.get("model_patch", "")

    try:
        from swebench_utils import load_instances
        instances = load_instances(instance_ids=[instance_id])
        inst = instances[0] if instances else {}
    except Exception:
        inst = {}

    gold_patch  = inst.get("patch", "")
    issue_text  = inst.get("problem_statement", "No issue text.")
    repo        = inst.get("repo", "")
    base_commit = inst.get("base_commit", "")[:7]

    verify_log = ""
    loc_notes  = ""
    if run_dir:
        vlog = run_dir / "_share" / "verify_log.txt"
        loc  = run_dir / "_share" / "localize_notes.md"
        if vlog.exists(): verify_log = vlog.read_text()
        if loc.exists():  loc_notes  = loc.read_text()

    gold_file = next((l[6:] for l in gold_patch.splitlines() if l.startswith("+++ b/")), "")
    gen_file  = next((l[6:] for l in gen_patch.splitlines()  if l.startswith("+++ b/")), "")

    patch_info = parse_patch(gen_patch)
    gold_info  = parse_patch(gold_patch)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SWE-bench: {_html.escape(instance_id)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f0f2f5; color: #212529; font-size: 14px; }}

  .page-header {{
    background: #2c3e50; color: white; padding: 14px 20px;
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 16px;
  }}
  .page-header h1 {{ font-size: 15px; font-weight: 600; }}
  .chip {{ background: #34495e; border-radius: 4px; padding: 3px 8px;
           font-size: 12px; font-family: monospace; }}

  .layout {{
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 12px;
    padding: 12px;
    height: calc(100vh - 52px);
  }}

  /* ── left panel ── */
  .left-panel {{ display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }}

  /* ── right panels ── */
  .right-panels {{
    display: grid;
    grid-template-rows: 1fr 1fr auto;
    gap: 10px;
    overflow: hidden;
  }}

  /* ── panel card ── */
  .card {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.1);
    display: flex; flex-direction: column;
    overflow: hidden;
  }}
  .card-header {{
    padding: 8px 12px;
    font-size: 13px; font-weight: 700;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid #dee2e6;
    flex-shrink: 0;
  }}
  .card-header.gold   {{ background: #fff3cd; color: #856404; }}
  .card-header.gen    {{ background: #d1ecf1; color: #0c5460; }}
  .card-header.tests  {{ background: #d4edda; color: #155724; }}
  .card-header.input  {{ background: #e2e3e5; color: #383d41; }}
  .card-body {{ overflow-y: auto; flex: 1; padding: 0; }}
  .card-body.padded {{ padding: 12px; }}

  .file-path {{ font-family: monospace; font-size: 11px; background: #f8f9fa;
                border: 1px solid #dee2e6; border-radius: 3px; padding: 4px 8px;
                margin: 8px 10px; color: #555; flex-shrink: 0; }}

  /* ── diff ── */
  .diff-line {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px;
                padding: 1px 10px; white-space: pre; line-height: 1.55; }}
  .diff-line.add  {{ background: #d4edda; color: #155724; }}
  .diff-line.del  {{ background: #f8d7da; color: #721c24; }}
  .diff-line.hunk {{ background: #cce5ff; color: #004085; }}
  .diff-line.meta {{ background: #e2e3e5; color: #383d41; }}
  .diff-line.ctx  {{ background: #fafafa; color: #212529; }}

  /* ── test results ── */
  .test-row {{ font-family: monospace; font-size: 12px; padding: 4px 10px;
               display: flex; align-items: center; gap: 8px; border-bottom: 1px solid #f0f0f0; }}
  .test-row.pass-row {{ background: #f6fff8; }}
  .test-row.fail-row {{ background: #fff5f5; }}
  .test-summary {{ font-family: monospace; font-size: 12px; padding: 6px 10px;
                   background: #f8f9fa; color: #555; border-top: 1px solid #dee2e6;
                   font-weight: bold; }}
  .badge {{ font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px;
            min-width: 52px; text-align: center; flex-shrink: 0; }}
  .badge.pass {{ background: #27ae60; color: white; }}
  .badge.fail {{ background: #c0392b; color: white; }}
  .muted {{ color: #888; padding: 12px; font-style: italic; }}

  /* ── issue text ── */
  .section-label {{
    font-size: 12px; font-weight: 700; color: #495057;
    padding: 8px 12px 4px; border-bottom: 1px solid #f0f0f0;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .section-label .bullet {{ font-size: 11px; color: #aaa; font-weight: 400; }}
  .issue-body {{ padding: 10px 12px; font-size: 13px; line-height: 1.6; color: #333; }}
  .issue-body code {{ font-family: monospace; font-size: 12px; background: #f8f9fa;
                      padding: 1px 4px; border-radius: 3px; }}

  /* ── stats bar ── */
  .stats {{ display: flex; gap: 12px; padding: 6px 12px; background: #f8f9fa;
            border-bottom: 1px solid #eee; font-size: 11px; color: #555; flex-shrink: 0; }}
  .stats .add {{ color: #27ae60; font-weight: 600; }}
  .stats .del {{ color: #c0392b; font-weight: 600; }}
</style>
</head>
<body>

<div class="page-header">
  <h1>SWE-bench: {_html.escape(instance_id)}</h1>
  <span class="chip">{_html.escape(repo)}@{base_commit}</span>
  <span class="chip">model: {_html.escape(pred.get("model_name_or_path",""))}</span>
</div>

<div class="layout">

  <!-- ── LEFT: Model Input ── -->
  <div class="left-panel">
    <div class="card">
      <div class="card-header input">📋 Model Input</div>

      <div class="section-label">
        ▼ Instructions <span class="bullet">• 1 line</span>
      </div>
      <div class="issue-body">
        You will be provided with a partial code base and an issue statement explaining a problem to resolve.
      </div>

      <div class="section-label">
        ▼ Issue <span class="bullet">• {len(issue_text.splitlines())} lines</span>
      </div>
      <div class="issue-body">{_html_issue(issue_text[:1200])}</div>

      {"" if not loc_notes else f'''
      <div class="section-label">▼ Localisation Notes</div>
      <div class="issue-body"><code>{_html.escape(loc_notes[:500])}</code></div>
      '''}
    </div>
  </div>

  <!-- ── RIGHT panels ── -->
  <div class="right-panels">

    <!-- Gold Patch -->
    <div class="card">
      <div class="card-header gold">🥇 Gold Patch</div>
      {"" if not gold_file else f'<div class="file-path">{_html.escape(gold_file)}</div>'}
      <div class="stats">
        <span class="add">+{gold_info["added"]} additions</span>
        <span class="del">-{gold_info["removed"]} deletions</span>
        <span>{len(gold_info["files"])} file(s)</span>
      </div>
      <div class="card-body">
        {_html_diff(gold_patch)}
      </div>
    </div>

    <!-- Generated Patch -->
    <div class="card">
      <div class="card-header gen">🤖 Generated Patch</div>
      {"" if not gen_file else f'<div class="file-path">{_html.escape(gen_file)}</div>'}
      <div class="stats">
        <span class="add">+{patch_info["added"]} additions</span>
        <span class="del">-{patch_info["removed"]} deletions</span>
        <span>{len(patch_info["files"])} file(s)</span>
      </div>
      <div class="card-body">
        {_html_diff(gen_patch)}
      </div>
    </div>

    <!-- Test Results -->
    <div class="card">
      <div class="card-header tests">✅ Generated Patch Test Results</div>
      <div class="card-body">
        {_html_tests(verify_log)}
      </div>
    </div>

  </div>
</div>

</body>
</html>
"""

    output.write_text(html)
    console.print(f"[green]Saved: {output}[/green]")


@cli.command()
@click.option("--instance", "-i", required=True, help="SWE-bench instance_id")
@click.option("--predictions", "-p", default="predictions.jsonl")
@click.option("--work-dir", default=None, type=click.Path(),
              help="run_N directory (auto-detected from last_run.json)")
@click.option("--output", "-o", default=None)
def html(instance: str, predictions: str, work_dir: str | None, output: str | None):
    """Figure-6-style HTML report: crisp, zoomable, scrollable."""
    if work_dir is None and Path("last_run.json").exists():
        m = json.loads(Path("last_run.json").read_text())
        work_dir = m.get("run_dir")
    out = Path(output) if output else Path(f"compare_{instance}.html")
    instance_comparison_html(instance, Path(predictions),
                              Path(work_dir) if work_dir else None, out)


if __name__ == "__main__":
    cli()
