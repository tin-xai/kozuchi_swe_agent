# How the SWE-bench Harness Works

Based on Fujitsu's "Orchestra" approach from their April 2026 blog post, this harness
achieves strong SWE-bench scores by treating issue resolution as a structured pipeline
— not a single long conversation.

---

## The Core Idea

A typical LLM agent given a GitHub issue will: read the issue, poke around the repo,
make some edits, and hope for the best. The Fujitsu insight is that this is too
unconstrained. Instead, split the work into **8 mandatory phases**, each with a clear
goal and required output files. The agent cannot skip ahead — it must produce specific
artifacts before moving to the next phase.

```
Issue → REPRODUCE → WRITE TESTS → LOCALIZE → SELECT REGRESSION TESTS
      → WRITE FIX  → VERIFY → REVIEW → REPORT → Patch submitted
```

---

## The Two-Agent "Orchestra" Pattern

Each turn of the agent loop uses **two LLM calls**, not one.

### Conductor (temperature = 0.6)
- Does the *thinking*: reads the issue, forms hypotheses, decides what to investigate next.
- Higher temperature lets it explore different angles rather than converging too early.
- Produces natural-language intent: "I want to run the test suite to see which tests fail."

### Tool Specialist (temperature = 0.0)
- Does the *execution*: takes the conductor's intent and produces a valid, deterministic
  tool call with correct syntax.
- Zero temperature = no hallucinated flags, no mistyped paths.
- Used specifically for the tricky tools (`line_trace`, `caller_trace`, `coedit_localize`)
  where command syntax errors would waste a turn.

Both agents use the same model (e.g. `qwen/qwen3-14b` via OpenRouter). The split is
purely in **sampling temperature** and **system prompt role**.

---

## The 8 Phases

Each phase has a system prompt, a turn budget, and **required assets** — files that must
exist in `_share/` before the phase is allowed to exit.

| # | Phase | Goal | Required output |
|---|-------|------|-----------------|
| 1 | `ISSUE_REPRODUCT` | Reproduce the bug with a minimal script | `repro_notes.md` |
| 2 | `TEST_SYNTHSIZE` | Write pytest tests that currently FAIL | `fail_to_pass_tests.py`, `test_notes.md` |
| 3 | `CODE_LOCALIZE` | Find the root cause: exact file + line | `localize_notes.md` |
| 4 | `TEST_LOCALIZE` | Pick existing tests that must keep passing | `pass_to_pass_tests.txt` |
| 5 | `CODE_FIX` | Write the patch | `patch.diff` |
| 6 | `VERIFY_PATCH` | Confirm fix works, regressions don't break | `verify_log.txt` |
| 7 | `ISSUE_CLOSE` | Pre-submission review | `close_review.md` |
| 8 | `FINAL_REPORT` | Document everything | `final_report.md` |

The harness enforces this with `assets_satisfied()` in [phases.py](phases.py): if the
required files aren't written, the conductor gets nudged to keep working.

---

## Filesystem-Based State (`_share/`)

The biggest architectural decision: **intermediate work lives on disk, not in the
conversation history**.

```
work_dir/
└── _share/
    ├── repro_notes.md           ← written in phase 1
    ├── fail_to_pass_tests.py    ← written in phase 2
    ├── localize_notes.md        ← written in phase 3
    ├── pass_to_pass_tests.txt   ← written in phase 4
    ├── patch.diff               ← written in phase 5
    ├── verify_log.txt           ← written in phase 6
    ├── handover_PHASE_*.md      ← context compression memos
    └── phase_log.jsonl          ← event log
```

When the conductor moves from phase 3 to phase 4, the new conversation does not need
to remember everything from phase 3 — it just reads `localize_notes.md` from disk.
This is how the system handles issues that would otherwise exhaust the context window.

---

## Context Compression (Handover)

Each phase has a **turn budget** (32–48 turns). When the budget is nearly exhausted
*or* the token count approaches the limit, the harness triggers a handover:

1. It asks the conductor: *"Write a handover memo summarising what you've done and what remains."*
2. The memo is saved to `_share/handover_PHASE_timestamp.md`.
3. The conductor's entire conversation history is discarded.
4. A new conversation starts with only: system prompt + handover memo.

This is the same trick humans use in long projects: write a status update, hand it to
the next shift, don't repeat all the work.

---

## The 4 Custom Tools

Beyond standard `bash`, `view_file`, `str_replace`, the harness adds:

### `line_trace`
Runs a test script under Python's `trace` module and reports which source lines were
actually executed. Instead of guessing where a bug is, the agent can see the exact
execution path through the suspect code.

```
django/forms/fields.py:142   ← this line ran
django/forms/fields.py:143   ← this line ran
django/forms/fields.py:157   ← jumped here, skipped lines 144-156
```

### `caller_trace`
Statically walks the entire repo's AST and finds every call site of a given function.
Used to understand the "blast radius" before making a change — how many places call
this function, and which ones matter.

### `coedit_localize`
Queries `git log` to find files that have historically been edited in the same commit
as the target file. High co-edit count = high coupling. If you're touching
`MultiValueField`, this might surface `MultiWidget` as frequently co-edited.

### `line_edit`
A safe single-line editor that requires you to specify the *expected current content*
of the line before replacing it. If the line doesn't match, the edit is rejected.
Prevents off-by-one errors that `str_replace` on large blocks can miss.

---

## Test-Time Scaling (TTS@N)

The harness runs the entire 8-phase pipeline **N independent times** (default: 8).
Each run uses a different random seed (via temperature=0.6 for the conductor) and may
find a different patch.

Then a **cross-test matrix** is built:

```
           patch-0  patch-1  patch-2  ...  patch-7
tests-0:   F2P=1.0  F2P=0.8  F2P=1.0       F2P=0.9
           P2P=0.99 P2P=1.0  P2P=0.95      P2P=1.0
tests-1:   ...
```

Each candidate's tests are run against *every* candidate's patch. A patch that passes
most test suites — not just its own — is more likely to be genuinely correct.

**Selection rule** (from the blog):
```
score = 0.3 × avg(F2P pass rates) + 0.7 × avg(P2P pass rates)
```
P2P (regression) is weighted 70% because a patch that breaks existing tests is worse
than one that doesn't fix everything. Ties are broken by **shortest patch** (simpler
= safer).

---

## OpenRouter Integration

OpenRouter provides an OpenAI-compatible API endpoint at `https://openrouter.ai/api/v1`.
The harness uses the `openai` Python SDK with a custom `base_url` — no OpenRouter-
specific SDK needed.

```python
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
```

Any model on OpenRouter works. Swap `MODEL` in `.env`:

| Model | Notes |
|-------|-------|
| `qwen/qwen3-14b` | Default — good balance of speed and capability |
| `qwen/qwen3-8b:free` | Free tier, slower |
| `qwen/qwen3-30b` | Stronger, closer to the blog's Qwen3.5-27B |
| `meta-llama/llama-3.3-70b-instruct` | Strong open-weight alternative |

---

## Data Flow for One Instance

```
1. load_instances()          ← fetch from HuggingFace datasets
2. setup_repo()              ← git clone at base_commit SHA
3. for run in range(TTS_N):
     InstanceHarness.run()
     └─ for phase in PHASE_ORDER:
          conductor.step(phase_prompt + share_context)
          loop:
            if tool_calls → dispatch → result → conductor.step(result)
            if no tool_calls → check assets → nudge or advance phase
            if token_limit → handover memo → compress
4. cross_test_and_select()   ← pick best patch
5. write predictions.jsonl   ← {"instance_id", "model_patch", "model_name_or_path"}
6. evaluate_with_docker()    ← official SWE-bench Docker evaluation
```

---

## File Map

| File | What it does |
|------|-------------|
| [config.py](config.py) | API keys, model name, temperatures, turn budgets |
| [agents.py](agents.py) | `ConductorAgent` and `ToolSpecialistAgent` wrappers around OpenRouter |
| [tools.py](tools.py) | All tool implementations + JSON schemas for function calling |
| [phases.py](phases.py) | Phase enum, system prompts, required-asset definitions |
| [state.py](state.py) | `SharedState` — reads/writes `_share/` directory |
| [harness.py](harness.py) | `InstanceHarness` — the main phase+turn loop |
| [selector.py](selector.py) | TTS@N cross-test matrix and candidate selection |
| [swebench_utils.py](swebench_utils.py) | HuggingFace loading, git repo setup, Docker eval |
| [run.py](run.py) | CLI: `solve`, `batch`, `list`, `evaluate` commands |
