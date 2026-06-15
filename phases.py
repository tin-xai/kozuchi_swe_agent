"""
Phase definitions: 8 sequential phases, each with a system prompt,
required assets (files that must exist before phase exit), and
a max turn budget from config.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path


class Phase(str, Enum):
    ISSUE_REPRODUCT = "ISSUE_REPRODUCT"
    TEST_SYNTHSIZE  = "TEST_SYNTHSIZE"
    CODE_LOCALIZE   = "CODE_LOCALIZE"
    TEST_LOCALIZE   = "TEST_LOCALIZE"
    CODE_FIX        = "CODE_FIX"
    VERIFY_PATCH    = "VERIFY_PATCH"
    ISSUE_CLOSE     = "ISSUE_CLOSE"
    FINAL_REPORT    = "FINAL_REPORT"


PHASE_ORDER = [
    Phase.ISSUE_REPRODUCT,
    Phase.TEST_SYNTHSIZE,
    Phase.CODE_LOCALIZE,
    Phase.TEST_LOCALIZE,
    Phase.CODE_FIX,
    Phase.VERIFY_PATCH,
    Phase.ISSUE_CLOSE,
    Phase.FINAL_REPORT,
]

# Files that must be present in share_dir before exiting each phase
REQUIRED_ASSETS: dict[Phase, list[str]] = {
    Phase.ISSUE_REPRODUCT: ["repro_notes.md"],
    Phase.TEST_SYNTHSIZE:  ["fail_to_pass_tests.py", "test_notes.md"],
    Phase.CODE_LOCALIZE:   ["localize_notes.md"],
    Phase.TEST_LOCALIZE:   ["pass_to_pass_tests.txt"],
    Phase.CODE_FIX:        ["patch.diff"],
    Phase.VERIFY_PATCH:    ["verify_log.txt"],
    Phase.ISSUE_CLOSE:     ["close_review.md"],
    Phase.FINAL_REPORT:    ["final_report.md"],
}


def _phase_intro(phase: Phase) -> str:
    intros = {
        Phase.ISSUE_REPRODUCT: (
            "## Phase: ISSUE_REPRODUCT\n"
            "Your goal is to reproduce the bug described in the issue.\n"
            "- Read the issue carefully, understand the expected vs. actual behaviour.\n"
            "- Write a minimal reproduction script and run it to confirm the failure.\n"
            "- Write reproduction notes to `_share/repro_notes.md`.\n"
            "- Exit this phase once you can reliably trigger the bug."
        ),
        Phase.TEST_SYNTHSIZE: (
            "## Phase: TEST_SYNTHSIZE\n"
            "Synthesise FAIL_TO_PASS tests that will fail now and pass after the fix.\n"
            "- Build on your reproduction from the previous phase.\n"
            "- Write clean pytest tests to `_share/fail_to_pass_tests.py`.\n"
            "- Document your test rationale in `_share/test_notes.md`.\n"
            "- Confirm the tests FAIL on the unmodified repo before exiting."
        ),
        Phase.CODE_LOCALIZE: (
            "## Phase: CODE_LOCALIZE\n"
            "Identify the root cause and the precise files/lines that need changing.\n"
            "- Use `line_trace` on your repro script to see executed lines.\n"
            "- Use `caller_trace` to understand call chains.\n"
            "- Use `coedit_localize` to surface coupled files.\n"
            "- Write your findings to `_share/localize_notes.md` (file, line, root cause)."
        ),
        Phase.TEST_LOCALIZE: (
            "## Phase: TEST_LOCALIZE\n"
            "Select PASS_TO_PASS regression tests from the existing test suite.\n"
            "- Identify test files related to the changed code area.\n"
            "- Run them to confirm they currently PASS.\n"
            "- Write their pytest node IDs (one per line) to `_share/pass_to_pass_tests.txt`."
        ),
        Phase.CODE_FIX: (
            "## Phase: CODE_FIX\n"
            "Implement the minimal patch that fixes the issue.\n"
            "- Apply changes using `str_replace` or `line_edit` (prefer `str_replace`).\n"
            "- Run the FAIL_TO_PASS tests from `_share/fail_to_pass_tests.py` — they must pass.\n"
            "- Run the PASS_TO_PASS tests — they must remain passing.\n"
            "- Generate the patch with `git diff > _share/patch.diff` and verify it is non-empty."
        ),
        Phase.VERIFY_PATCH: (
            "## Phase: VERIFY_PATCH\n"
            "Final verification before submission.\n"
            "- Apply the patch on a clean checkout (git stash, apply, test).\n"
            "- Run both F2P and P2P tests and record results.\n"
            "- Write a verify log to `_share/verify_log.txt` with exit codes."
        ),
        Phase.ISSUE_CLOSE: (
            "## Phase: ISSUE_CLOSE\n"
            "Pre-submission review.\n"
            "- Review the patch for correctness, style, and completeness.\n"
            "- Check for unintended side effects.\n"
            "- Write a short sign-off to `_share/close_review.md`."
        ),
        Phase.FINAL_REPORT: (
            "## Phase: FINAL_REPORT\n"
            "Document the full resolution.\n"
            "- Summarise: issue, root cause, fix approach, test results.\n"
            "- Write to `_share/final_report.md`.\n"
            "- This phase produces no code changes — documentation only."
        ),
    }
    return intros[phase]


CONDUCTOR_BASE_SYSTEM = """\
You are an expert software engineer solving a real GitHub issue from the SWE-bench dataset.

IMPORTANT — file paths: All tool calls use paths relative to the repository root.
Do NOT prefix paths with "repo/", "./repo/", or any leading directory.
For bash commands use "." as the working directory, never "repo".
Correct:   django/contrib/auth/validators.py
Wrong:     repo/django/contrib/auth/validators.py

You work methodically through structured phases. In each phase you:
1. Reason about the problem and form a hypothesis.
2. Call tools to investigate or apply changes.
3. Interpret results and update your understanding.
4. Advance toward the phase exit criteria.

Tools available: bash, view_file, str_replace, line_edit, line_trace, caller_trace,
coedit_localize, write_file.

All tool calls must be through the function-calling interface — never output raw shell commands
in your text. Use `write_file` to write _share/ assets.

Be systematic and minimal: make the smallest change that fixes the issue.
"""

TOOL_SPECIALIST_BASE_SYSTEM = """\
You are a tool-syntax specialist. The conductor agent has described what it wants to do.
Your job is to emit a valid, correctly-structured tool call (or sequence of calls) to
accomplish that intent. Do not reason about the strategy — just produce correct tool calls.

Rules:
- Paths must be relative to the repo root (no leading slash unless absolute is required).
- Shell commands must be safe for the current repo environment.
- If the conductor's intent is ambiguous, choose the safest interpretation.
"""


def conductor_system_prompt(phase: Phase, issue_text: str, instance_id: str) -> str:
    return (
        f"{CONDUCTOR_BASE_SYSTEM}\n\n"
        f"Instance: {instance_id}\n\n"
        f"Issue:\n{issue_text}\n\n"
        f"{_phase_intro(phase)}"
    )


def tool_specialist_system_prompt(phase: Phase) -> str:
    return f"{TOOL_SPECIALIST_BASE_SYSTEM}\n\nCurrent phase: {phase.value}"


def assets_satisfied(phase: Phase, share_dir: Path) -> tuple[bool, list[str]]:
    required = REQUIRED_ASSETS.get(phase, [])
    missing = [f for f in required if not (share_dir / f).exists()]
    return len(missing) == 0, missing
