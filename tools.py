"""
Tool implementations for the SWE harness.
All tools run inside an isolated repo directory passed as `repo_path`.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: str, cwd: str, timeout: int = 60, env: dict | None = None) -> tuple[int, str, str]:
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, env=merged_env,
    )
    return result.returncode, result.stdout, result.stderr


def _clean_path(file_path: str) -> str:
    """Strip common wrong prefixes the agent adds (repo/, ./repo/, etc.)."""
    for prefix in ("repo/", "./repo/", "/repo/"):
        if file_path.startswith(prefix):
            return file_path[len(prefix):]
    return file_path


def _truncate(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [truncated {len(text)-max_chars} chars] ...\n" + text[-half:]


# ── tool: bash ───────────────────────────────────────────────────────────────

def bash(command: str, repo_path: str, timeout: int = 60, repo_env: dict | None = None) -> str:
    """Run an arbitrary shell command inside the repo. Returns stdout+stderr."""
    try:
        rc, out, err = _run(command, cwd=repo_path, timeout=timeout, env=repo_env)
        combined = out + (f"\n[stderr]\n{err}" if err.strip() else "")
        return _truncate(f"[exit {rc}]\n{combined}")
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except Exception as e:
        return f"[error] {e}"


# ── tool: line_trace ─────────────────────────────────────────────────────────

def line_trace(test_script: str, repo_path: str, repo_env: dict | None = None) -> str:
    """
    Run test_script under Python trace coverage and return which lines
    were executed in repo source files, grouped by file.
    """
    trace_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=repo_path
    )
    trace_script.write(f"""
import trace, sys, os

tracer = trace.Trace(
    count=True, trace=False,
    ignoredirs=[sys.prefix, os.path.join(sys.prefix, 'lib')],
)
tracer.runscript("{test_script}")
r = tracer.results()

repo = "{repo_path}"
for (fn, lineno), count in sorted(r.counts.items()):
    if fn.startswith(repo) and count > 0:
        print(f"{{fn[len(repo)+1:]}}:{{lineno}}")
""")
    trace_script.close()
    try:
        rc, out, err = _run(
            f"{sys.executable} {trace_script.name}",
            cwd=repo_path, timeout=120, env=repo_env,
        )
        return _truncate(f"[exit {rc}]\n{out}" + (f"\n[stderr]\n{err}" if err.strip() else ""))
    except subprocess.TimeoutExpired:
        return "[timeout after 120s]"
    finally:
        os.unlink(trace_script.name)


# ── tool: caller_trace ───────────────────────────────────────────────────────

def caller_trace(func_name: str, repo_path: str) -> str:
    """
    Statically find all callers of func_name within the repo using AST.
    Returns file:line pairs.
    """
    results: list[str] = []
    repo = Path(repo_path)

    for py_file in repo.rglob("*.py"):
        try:
            source = py_file.read_text(errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name == func_name:
                    rel = str(py_file.relative_to(repo))
                    results.append(f"{rel}:{node.lineno}")

    if not results:
        return f"No callers of '{func_name}' found in repo."
    return "\n".join(results[:200])


# ── tool: coedit_localize ────────────────────────────────────────────────────

def coedit_localize(target_file: str, repo_path: str, top_n: int = 20) -> str:
    """
    Use git log to find files most frequently co-edited with target_file.
    Ranks by co-edit frequency as a proxy for coupling.
    """
    rc, commits, _ = _run(
        f"git log --follow --pretty=format:%H -- {target_file}",
        cwd=repo_path,
    )
    if rc != 0 or not commits.strip():
        return f"No git history found for {target_file}"

    commit_list = commits.strip().split("\n")[:100]
    coedits: dict[str, int] = {}

    for commit in commit_list:
        _, files, _ = _run(
            f"git diff-tree --no-commit-id -r --name-only {commit}",
            cwd=repo_path,
        )
        for f in files.strip().split("\n"):
            f = f.strip()
            if f and f != target_file and f.endswith(".py"):
                coedits[f] = coedits.get(f, 0) + 1

    ranked = sorted(coedits.items(), key=lambda x: -x[1])[:top_n]
    if not ranked:
        return f"No co-edited Python files found for {target_file}"
    lines = [f"{count:4d}  {f}" for f, count in ranked]
    return f"Co-edit partners for {target_file} (commits={len(commit_list)}):\n" + "\n".join(lines)


# ── tool: line_edit ──────────────────────────────────────────────────────────

def line_edit(
    file_path: str,
    line_number: int,
    expected_content: str,
    new_content: str,
    repo_path: str,
) -> str:
    """
    Edit a single line (1-indexed) in file_path, validating against expected_content
    to catch off-by-one errors. Replaces the line with new_content.
    """
    file_path = _clean_path(file_path)
    abs_path = Path(repo_path) / file_path
    if not abs_path.exists():
        return f"[error] file not found: {file_path}"

    lines = abs_path.read_text().splitlines(keepends=True)
    idx = line_number - 1
    if idx < 0 or idx >= len(lines):
        return f"[error] line {line_number} out of range (file has {len(lines)} lines)"

    actual = lines[idx].rstrip("\n")
    if actual != expected_content:
        return (
            f"[error] expected_content mismatch at line {line_number}:\n"
            f"  expected: {repr(expected_content)}\n"
            f"  actual:   {repr(actual)}"
        )

    lines[idx] = new_content + ("\n" if not new_content.endswith("\n") else "")
    abs_path.write_text("".join(lines))
    return f"[ok] line {line_number} in {file_path} updated."


# ── tool: view_file ──────────────────────────────────────────────────────────

def view_file(file_path: str, repo_path: str, start: int = 1, end: int | None = None) -> str:
    """View a file with line numbers, optionally restricted to a range."""
    file_path = _clean_path(file_path)
    abs_path = Path(repo_path) / file_path
    if not abs_path.exists():
        return f"[error] file not found: {file_path}"
    lines = abs_path.read_text(errors="replace").splitlines()
    start = max(1, start)
    end = min(len(lines), end or len(lines))
    numbered = "\n".join(f"{i+1:5d}  {lines[i]}" for i in range(start - 1, end))
    return f"=== {file_path} (lines {start}-{end}/{len(lines)}) ===\n{numbered}"


# ── tool: str_replace ────────────────────────────────────────────────────────

def str_replace(file_path: str, old_str: str, new_str: str, repo_path: str) -> str:
    """Replace first occurrence of old_str with new_str in file."""
    file_path = _clean_path(file_path)
    abs_path = Path(repo_path) / file_path
    if not abs_path.exists():
        return f"[error] file not found: {file_path}"
    content = abs_path.read_text()
    if old_str not in content:
        return f"[error] old_str not found in {file_path}"
    count = content.count(old_str)
    if count > 1:
        return f"[error] old_str appears {count} times — be more specific"
    abs_path.write_text(content.replace(old_str, new_str, 1))
    return f"[ok] replaced in {file_path}"


# ── tool: view_graph ─────────────────────────────────────────────────────────

def view_graph(
    focus: list[str],
    repo_path: str,
    share_dir: "Path | None" = None,
    graph_type: str = "call",
    depth: int = 2,
) -> str:
    """
    Render a subgraph centred on `focus` functions/modules and return a
    VLM-generated natural-language description of the structural relationships.
    Falls back to text topology if the VLM is unavailable.
    """
    from pathlib import Path as _Path
    from visual_memory import get_graph_description, describe_topology, load_graph

    if share_dir is None:
        return "[view_graph] share_dir not provided — graph not available."

    sd = _Path(share_dir)
    graph_path = sd / "graphs" / f"{graph_type}_graph.pkl"

    if not graph_path.exists():
        return (
            f"[view_graph] {graph_type} graph has not been built yet. "
            "It is built automatically at the start of the run."
        )

    return get_graph_description(sd, graph_type, focus, depth=depth)


# ── JSON schemas for LLM tool definitions ────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repository root. Returns stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View a file with line numbers, optionally a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start": {"type": "integer", "description": "First line (1-indexed)"},
                    "end": {"type": "integer", "description": "Last line (inclusive)"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace the first occurrence of old_str with new_str in a file. Fails if old_str appears more than once or not at all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["file_path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "line_edit",
            "description": "Edit a single line by number, with expected-content validation to prevent off-by-one errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line_number": {"type": "integer"},
                    "expected_content": {"type": "string", "description": "Current line content (without newline) for validation"},
                    "new_content": {"type": "string", "description": "Replacement line content"},
                },
                "required": ["file_path", "line_number", "expected_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "line_trace",
            "description": "Run a Python test script under execution trace to reveal which source lines are actually executed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_script": {"type": "string", "description": "Absolute path to the test script to run"},
                },
                "required": ["test_script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "caller_trace",
            "description": "Statically find all callers of a function name within the repo using AST analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "func_name": {"type": "string"},
                },
                "required": ["func_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "coedit_localize",
            "description": "Use git history to find files most frequently co-edited with the target file — proxy for coupling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string"},
                    "top_n": {"type": "integer", "description": "How many results to return (default 20)"},
                },
                "required": ["target_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_graph",
            "description": (
                "Render a subgraph of the codebase's call or import graph centred on the "
                "given functions/modules, then get a VLM description of structural relationships "
                "(hubs, blast radius, unexpected couplings). Use during CODE_LOCALIZE to quickly "
                "understand how a function is connected without reading every file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Function or module names to centre the subgraph on.",
                    },
                    "graph_type": {
                        "type": "string",
                        "enum": ["call", "import"],
                        "description": "'call' for function call relationships, 'import' for file-level dependencies.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Hop depth around focus nodes (default 2).",
                    },
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
]


def write_file(file_path: str, content: str, repo_path: str) -> str:
    """Write content to file_path inside repo."""
    file_path = _clean_path(file_path)
    abs_path = Path(repo_path) / file_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content)
    return f"[ok] wrote {len(content)} chars to {file_path}"


def dispatch(
    tool_name: str,
    args: dict,
    repo_path: str,
    repo_env: dict | None = None,
    share_dir: "Path | None" = None,
) -> str:
    """Dispatch a tool call by name with repo_path, repo_env, and share_dir injected."""
    fn_map = {
        "bash": lambda a: bash(a["command"], repo_path, a.get("timeout", 60), repo_env),
        "view_file": lambda a: view_file(a["file_path"], repo_path, a.get("start", 1), a.get("end")),
        "str_replace": lambda a: str_replace(a["file_path"], a["old_str"], a["new_str"], repo_path),
        "line_edit": lambda a: line_edit(a["file_path"], a["line_number"], a["expected_content"], a["new_content"], repo_path),
        "line_trace": lambda a: line_trace(a["test_script"], repo_path, repo_env),
        "caller_trace": lambda a: caller_trace(a["func_name"], repo_path),
        "coedit_localize": lambda a: coedit_localize(a["target_file"], repo_path, a.get("top_n", 20)),
        "view_graph": lambda a: view_graph(a["focus"], repo_path, share_dir, a.get("graph_type", "call"), a.get("depth", 2)),
        "write_file": lambda a: write_file(a["file_path"], a["content"], repo_path),
    }
    fn = fn_map.get(tool_name)
    if fn is None:
        return f"[error] unknown tool: {tool_name}"
    try:
        return fn(args)
    except Exception as e:
        return f"[error] {tool_name}: {e}"
