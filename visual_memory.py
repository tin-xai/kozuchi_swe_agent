"""
Visual memory for the SWE harness.

Pipeline:
  1. build_call_graph / build_import_graph  — parse repo into networkx graphs
  2. render_subgraph                         — draw a focused subgraph to PNG
  3. describe_with_vlm                       — send PNG to VLM, get text back
  4. describe_topology (fallback)            — pure-text description if VLM fails

The conductor is text-only, so the VLM acts as a translator:
  image → natural-language structural description → conductor context
"""
from __future__ import annotations

import ast
import base64
import io
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from openai import OpenAI

import config

# ── graph builders ────────────────────────────────────────────────────────────

_SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".tox", "venv", ".venv", "node_modules"}


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def build_call_graph(repo_path: str) -> nx.DiGraph:
    """
    AST-based function call graph.
    Nodes: "rel/path.py::func_name"
    Edges: caller → callee (callee may be unqualified if cross-file)
    """
    G = nx.DiGraph()
    repo = Path(repo_path)

    # First pass: register all defined functions with their full qualified name
    func_index: dict[str, str] = {}   # bare_name → qualified (last-wins for duplicates)
    for py in repo.rglob("*.py"):
        if _should_skip(py):
            continue
        try:
            tree = ast.parse(py.read_text(errors="replace"))
        except SyntaxError:
            continue
        rel = str(py.relative_to(repo))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = f"{rel}::{node.name}"
                func_index[node.name] = qname
                G.add_node(qname, file=rel, func=node.name)

    # Second pass: extract calls
    for py in repo.rglob("*.py"):
        if _should_skip(py):
            continue
        try:
            tree = ast.parse(py.read_text(errors="replace"))
        except SyntaxError:
            continue
        rel = str(py.relative_to(repo))

        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            caller = f"{rel}::{func_node.name}"
            for child in ast.walk(func_node):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Attribute):
                    callee_name = child.func.attr
                elif isinstance(child.func, ast.Name):
                    callee_name = child.func.id
                else:
                    continue
                callee = func_index.get(callee_name, callee_name)
                if caller != callee:
                    G.add_edge(caller, callee)

    return G


def build_import_graph(repo_path: str) -> nx.DiGraph:
    """
    File-level import dependency graph.
    Nodes: "rel/path.py"
    Edges: importer → imported_module
    """
    G = nx.DiGraph()
    repo = Path(repo_path)

    for py in repo.rglob("*.py"):
        if _should_skip(py):
            continue
        rel = str(py.relative_to(repo))
        G.add_node(rel)
        try:
            tree = ast.parse(py.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    G.add_edge(rel, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    G.add_edge(rel, node.module)

    return G


# ── subgraph extraction & rendering ──────────────────────────────────────────

def _extract_subgraph(G: nx.DiGraph, focus_nodes: list[str], depth: int) -> nx.DiGraph:
    """Extract nodes within `depth` hops of any focus node (in both directions)."""
    keep: set = set()
    for node in focus_nodes:
        # exact match or suffix match (bare name)
        matches = [n for n in G.nodes if n == node or n.endswith(f"::{node}") or n == node]
        if not matches:
            matches = [n for n in G.nodes if node in n]
        for m in matches:
            keep.add(m)
            # predecessors (callers)
            for pred in nx.ancestors(G, m) if m in G else []:
                if nx.shortest_path_length(G, pred, m) <= depth:
                    keep.add(pred)
            # successors (callees)
            for succ in nx.descendants(G, m) if m in G else []:
                if nx.shortest_path_length(G, m, succ) <= depth:
                    keep.add(succ)
    return G.subgraph(keep).copy() if keep else G.copy()


def render_subgraph(
    G: nx.DiGraph,
    focus_nodes: list[str],
    depth: int = 2,
    max_nodes: int = 60,
) -> bytes:
    """Render a focused subgraph to PNG bytes."""
    sub = _extract_subgraph(G, focus_nodes, depth)

    # If still too large, keep only the highest-degree nodes
    if len(sub) > max_nodes:
        top = sorted(sub.nodes, key=lambda n: sub.degree(n), reverse=True)[:max_nodes]
        sub = sub.subgraph(top).copy()

    if len(sub) == 0:
        # Empty graph — return a small placeholder image
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No nodes found for focus terms", ha="center", va="center")
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Layout
        try:
            pos = nx.kamada_kawai_layout(sub)
        except Exception:
            pos = nx.spring_layout(sub, seed=42)

        # Colour: red = focus, blue = callers (in-edges), green = callees (out-edges), grey = rest
        focus_set = {n for n in sub.nodes
                     if any(f == n or n.endswith(f"::{f}") or f in n for f in focus_nodes)}
        color_map = []
        for n in sub.nodes:
            if n in focus_set:
                color_map.append("#e74c3c")
            elif any(sub.has_edge(p, n) and p in focus_set for p in sub.predecessors(n)):
                color_map.append("#3498db")
            elif any(sub.has_edge(n, s) and s in focus_set for s in sub.successors(n)):
                color_map.append("#2ecc71")
            else:
                color_map.append("#95a5a6")

        # Short labels for readability
        labels = {n: n.split("::")[-1] if "::" in n else n.split("/")[-1] for n in sub.nodes}

        nx.draw_networkx(
            sub, pos=pos, ax=ax, labels=labels,
            node_color=color_map, node_size=800,
            font_size=7, arrows=True, arrowsize=12,
            edge_color="#bdc3c7", width=0.8,
        )
        ax.set_title(f"Subgraph around: {', '.join(focus_nodes)} (depth={depth})", fontsize=10)

        legend = [
            mpatches.Patch(color="#e74c3c", label="focus"),
            mpatches.Patch(color="#3498db", label="callers"),
            mpatches.Patch(color="#2ecc71", label="callees"),
            mpatches.Patch(color="#95a5a6", label="other"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=8)
        ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── VLM description ──────────────────────────────────────────────────────────

def describe_with_vlm(image_bytes: bytes, focus_nodes: list[str], graph_type: str = "call") -> str:
    """
    Send the rendered graph image to the VLM and get a structural description back.
    Returns plain text for injection into the conductor's context.
    """
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        f"This is a {graph_type} graph of a Python codebase. "
        f"The red nodes are the focus functions/modules: {', '.join(focus_nodes)}.\n"
        f"Blue nodes call into the focus nodes (callers). "
        f"Green nodes are called by the focus nodes (callees).\n\n"
        f"Please describe:\n"
        f"1. Which focus nodes are hubs (many connections) vs leaves (few connections)?\n"
        f"2. What is the blast radius — how many callers would be affected by changing the focus nodes?\n"
        f"3. Are there any unexpected connections or clusters worth investigating?\n"
        f"Keep the description under 200 words, focused on what a developer fixing a bug needs to know."
    )

    if not config.VLM_MODEL:
        return describe_topology(None, focus_nodes)

    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )
    try:
        resp = client.chat.completions.create(
            model=config.VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        # VLM failed (model may be text-only) — fall back to topology text
        return f"[VLM unavailable: {e}]\n" + describe_topology(None, focus_nodes)


def describe_topology(G: nx.DiGraph | None, focus_nodes: list[str], depth: int = 2) -> str:
    """
    Pure-text fallback: derive structural description directly from the graph.
    Used when the VLM call fails or the model is text-only.
    """
    if G is None or len(G) == 0:
        return "Graph not available."

    lines: list[str] = []
    for raw in focus_nodes:
        matches = [n for n in G.nodes if n == raw or n.endswith(f"::{raw}") or raw in n]
        for node in matches[:3]:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            callers = list(G.predecessors(node))[:10]
            callees = list(G.successors(node))[:10]
            lines.append(
                f"  {node.split('::')[-1]}: "
                f"{in_deg} callers, {out_deg} callees"
            )
            if callers:
                lines.append(f"    called by: {', '.join(c.split('::')[-1] for c in callers)}")
            if callees:
                lines.append(f"    calls: {', '.join(c.split('::')[-1] for c in callees)}")
    return "\n".join(lines) if lines else f"No nodes found matching: {focus_nodes}"


# ── persistence helpers ───────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f)


def load_graph(path: Path) -> nx.DiGraph | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


# ── main entry point ──────────────────────────────────────────────────────────

def get_graph_description(
    share_dir: Path,
    graph_type: str,
    focus_nodes: list[str],
    depth: int = 2,
) -> str:
    """
    Load pre-built graph from share_dir, render a focused subgraph,
    send to VLM, return text description.
    """
    graph_path = share_dir / "graphs" / f"{graph_type}_graph.pkl"
    G = load_graph(graph_path)

    if G is None:
        return f"[visual_memory] {graph_type} graph not built yet. Run build first."

    image_bytes = render_subgraph(G, focus_nodes, depth=depth)

    # Save the rendered image for inspection
    img_path = share_dir / "graphs" / f"{graph_type}_{'_'.join(focus_nodes[:2])}.png"
    img_path.write_bytes(image_bytes)

    return describe_with_vlm(image_bytes, focus_nodes, graph_type)
