"""
Filesystem-based shared state for a single instance run.
All intermediate artifacts live in <work_dir>/_share/ so they persist
across phase handovers and context compressions.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class SharedState:
    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.share_dir = self.work_dir / "_share"
        self.share_dir.mkdir(parents=True, exist_ok=True)

    # ── generic read/write ───────────────────────────────────────────────────

    def write(self, filename: str, content: str):
        (self.share_dir / filename).write_text(content)

    def read(self, filename: str) -> str | None:
        p = self.share_dir / filename
        return p.read_text() if p.exists() else None

    def exists(self, filename: str) -> bool:
        return (self.share_dir / filename).exists()

    def list_assets(self) -> list[str]:
        return [p.name for p in self.share_dir.iterdir() if p.is_file()]

    # ── handover memo ────────────────────────────────────────────────────────

    def write_handover(self, phase: str, memo: str):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self.write(f"handover_{phase}_{ts}.md", memo)

    def read_latest_handover(self) -> str:
        memos = sorted(self.share_dir.glob("handover_*.md"))
        if not memos:
            return ""
        return memos[-1].read_text()

    # ── phase log ────────────────────────────────────────────────────────────

    def append_phase_log(self, phase: str, event: str):
        log_path = self.share_dir / "phase_log.jsonl"
        with log_path.open("a") as f:
            f.write(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "phase": phase,
                "event": event,
            }) + "\n")

    def read_phase_log(self) -> list[dict]:
        log_path = self.share_dir / "phase_log.jsonl"
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text().splitlines()]

    # ── visual memory (graphs) ───────────────────────────────────────────────

    @property
    def graphs_dir(self) -> Path:
        d = self.share_dir / "graphs"
        d.mkdir(exist_ok=True)
        return d

    def graph_path(self, graph_type: str) -> Path:
        return self.graphs_dir / f"{graph_type}_graph.pkl"

    def graph_built(self, graph_type: str) -> bool:
        return self.graph_path(graph_type).exists()

    # ── summary for context ──────────────────────────────────────────────────

    def context_summary(self) -> str:
        """Brief summary of what assets exist — injected at phase start."""
        assets = self.list_assets()
        if not assets:
            return "No shared assets yet."
        lines = ["Existing _share/ assets:"]
        for name in sorted(assets):
            p = self.share_dir / name
            lines.append(f"  {name} ({p.stat().st_size} bytes)")
        return "\n".join(lines)
