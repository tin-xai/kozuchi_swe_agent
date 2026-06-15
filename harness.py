"""
InstanceHarness: runs one SWE-bench instance through all 8 phases.
Orchestrates the Conductor + ToolSpecialist dual-agent loop with
filesystem state, context compression, and phase transitions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

import config
import tools as tool_dispatch
from agents import ConductorAgent, ToolSpecialistAgent, count_tokens
from swebench_utils import detect_repo_env, repo_structure_hint
from visual_memory import build_call_graph, build_import_graph, save_graph
from phases import (
    PHASE_ORDER,
    Phase,
    assets_satisfied,
    conductor_system_prompt,
    tool_specialist_system_prompt,
)
from state import SharedState
from tools import TOOL_SCHEMAS

console = Console()


HANDOVER_REQUEST = """\
You are about to exhaust the context budget for this phase. Write a concise handover memo
(plain text, under 800 words) that captures:
1. What has been accomplished so far in this phase.
2. Key findings (file paths, line numbers, hypotheses confirmed/rejected).
3. What still needs to be done to complete this phase.
4. Any file paths written to _share/ and what they contain.

Write the memo now — it will replace your full context in the next segment.
"""


class InstanceHarness:
    def __init__(self, instance: dict, repo_path: str, work_dir: str):
        self.instance = instance
        self.instance_id: str = instance["instance_id"]
        self.issue_text: str = f"{instance['problem_statement']}"
        self.repo_path = repo_path
        self.state = SharedState(work_dir)
        self.repo_env = detect_repo_env(instance, repo_path)
        self.repo_structure = repo_structure_hint(repo_path)
        # Symlink repo_path/_share → work_dir/_share so bash commands like
        # "python _share/fail_to_pass_tests.py" resolve correctly inside the repo.
        share_link = Path(repo_path) / "_share"
        if not share_link.exists():
            share_link.symlink_to(self.state.share_dir)

    # ── main entry ───────────────────────────────────────────────────────────

    def run(self) -> str | None:
        """Run all phases. Returns the generated patch or None on failure."""
        console.print(Panel(f"[bold]Instance: {self.instance_id}[/bold]", style="blue"))

        if config.USE_VISUAL_MEMORY:
            self._build_visual_memory()

        for phase in PHASE_ORDER:
            success = self._run_phase(phase)
            if not success:
                console.print(f"[red]Phase {phase.value} failed — aborting instance.[/red]")
                return None

        patch = self.state.read("patch.diff")
        return patch

    # ── visual memory ────────────────────────────────────────────────────────

    def _build_visual_memory(self):
        """Build call and import graphs once at the start of the run."""
        graphs_dir = self.state.graphs_dir
        for graph_type, builder in [("call", build_call_graph), ("import", build_import_graph)]:
            out = graphs_dir / f"{graph_type}_graph.pkl"
            if out.exists():
                console.print(f"  [dim]{graph_type} graph already built[/dim]")
                continue
            console.print(f"  Building {graph_type} graph...")
            try:
                G = builder(self.repo_path)
                save_graph(G, out)
                console.print(f"  [green]{graph_type} graph: {len(G)} nodes, {G.number_of_edges()} edges[/green]")
            except Exception as e:
                console.print(f"  [yellow]Could not build {graph_type} graph: {e}[/yellow]")

    # ── phase loop ───────────────────────────────────────────────────────────

    def _run_phase(self, phase: Phase) -> bool:
        console.print(f"\n[yellow]→ Phase: {phase.value}[/yellow]")
        self.state.append_phase_log(phase.value, "start")

        max_turns = config.MAX_TURNS.get(phase.value, 32)
        retries = 0
        max_retries = 2

        conductor = ConductorAgent(
            system_prompt=conductor_system_prompt(phase, self.issue_text, self.instance_id),
            tools=TOOL_SCHEMAS,
        )
        tool_spec = ToolSpecialistAgent(
            system_prompt=tool_specialist_system_prompt(phase),
            tools=TOOL_SCHEMAS,
        )

        # Inject context from previous phases
        context_summary = self.state.context_summary()
        latest_handover = self.state.read_latest_handover()

        env_hint = "\n".join(f"  {k}={v}" for k, v in self.repo_env.items())
        initial_message = (
            f"<repo_env>\n{env_hint}\n</repo_env>\n"
            f"<repo_structure>\n{self.repo_structure}\n</repo_structure>\n"
        )

        if config.USE_VISUAL_MEMORY:
            graphs_available = [t for t in ("call", "import") if self.state.graph_built(t)]
            if graphs_available:
                graph_hint = (
                    f"Visual memory graphs available: {', '.join(graphs_available)}. "
                    "Use view_graph(focus=[...], graph_type='call'|'import') to get a "
                    "VLM-generated structural description of any function's connections."
                )
                initial_message += f"<visual_memory>\n{graph_hint}\n</visual_memory>\n"

        initial_message += f"<share_context>\n{context_summary}\n</share_context>"
        if latest_handover:
            initial_message += f"\n\n<previous_handover>\n{latest_handover}\n</previous_handover>"
        initial_message += "\n\nBegin this phase."

        conductor.step(initial_message)

        for turn in range(max_turns):
            last_response = conductor.history[-1]

            # Check if we're over the token budget
            if conductor.token_count() > config.MAX_PROMPT_TOKENS - config.CONTEXT_MARGIN:
                console.print(f"  [dim]Token budget reached at turn {turn} — compressing[/dim]")
                memo = self._request_handover(conductor, tool_spec, phase)
                self.state.write_handover(phase.value, memo)
                conductor.compress(memo)
                conductor.step("Continue. The handover memo above summarises where we are.")

            # Process tool calls from last conductor response
            tool_calls = last_response.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    # Route write_file to share_dir if path starts with _share/
                    if fn_name == "write_file":
                        fp = fn_args.get("file_path", "")
                        if fp.startswith("_share/"):
                            content = fn_args.get("content", "")
                            self.state.write(fp[len("_share/"):], content)
                            result = f"[ok] written to {fp}"
                        else:
                            result = tool_dispatch.dispatch(fn_name, fn_args, self.repo_path, self.repo_env, self.state.share_dir)
                    elif fn_name in ("line_trace", "caller_trace", "coedit_localize"):
                        # Use ToolSpecialist to validate before running
                        validated = tool_spec.validate_and_emit(
                            f"Execute: {fn_name}({fn_args})",
                            context=f"Conductor called {fn_name} — validate args are safe and correct.",
                        )
                        if validated.get("tool_calls"):
                            vtc = validated["tool_calls"][0]
                            try:
                                vargs = json.loads(vtc["function"]["arguments"])
                            except json.JSONDecodeError:
                                vargs = fn_args
                            result = tool_dispatch.dispatch(fn_name, vargs, self.repo_path)
                        else:
                            result = tool_dispatch.dispatch(fn_name, fn_args, self.repo_path, self.repo_env, self.state.share_dir)
                    else:
                        result = tool_dispatch.dispatch(fn_name, fn_args, self.repo_path, self.repo_env, self.state.share_dir)

                    console.print(f"  [dim]{fn_name}[/dim] → {result[:120].replace(chr(10), ' ')}")
                    conductor.add_tool_result(tc["id"], result)

                # Continue conductor after tool results
                next_resp = conductor.step("Continue.")
                continue

            # No tool calls — conductor is reasoning; prompt it to proceed or check assets
            ok, missing = assets_satisfied(phase, self.state.share_dir)
            if ok:
                console.print(f"  [green]Phase {phase.value} complete (all assets present)[/green]")
                self.state.append_phase_log(phase.value, "complete")
                return True

            # Nudge conductor
            if missing:
                nudge = (
                    f"Missing required assets: {missing}. "
                    "Complete them before this phase can exit. Continue."
                )
            else:
                nudge = "Continue working on this phase."

            conductor.step(nudge)

        # Max turns reached — check assets one more time
        ok, missing = assets_satisfied(phase, self.state.share_dir)
        if ok:
            self.state.append_phase_log(phase.value, "complete")
            return True

        if retries < max_retries:
            console.print(f"  [yellow]Max turns reached, missing {missing} — retrying phase[/yellow]")
            retries += 1
            return self._run_phase(phase)

        console.print(f"  [red]Phase {phase.value} gave up after retries. Missing: {missing}[/red]")
        self.state.append_phase_log(phase.value, f"gave_up missing={missing}")
        return False

    # ── handover compression ─────────────────────────────────────────────────

    def _request_handover(
        self,
        conductor: ConductorAgent,
        tool_spec: ToolSpecialistAgent,
        phase: Phase,
    ) -> str:
        resp = conductor.step(HANDOVER_REQUEST)
        memo = resp.get("content", "")
        if not memo:
            # Fallback: ask tool specialist to summarise from available context
            ts_resp = tool_spec.validate_and_emit(
                "Write a brief handover memo summarising what has been done and what remains.",
                context=f"Phase: {phase.value}\nAssets: {self.state.list_assets()}",
            )
            memo = ts_resp.get("content", "No memo produced.")
        return memo
