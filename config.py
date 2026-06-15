import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Small model to use — swap freely, OpenRouter supports hundreds of models
MODEL = os.environ.get("MODEL", "qwen/qwen3-14b")

# VLM used to interpret rendered code graphs and produce text descriptions
# for the text-only conductor. Must support vision content blocks.
# Leave unset (or set to empty string) to disable visual memory entirely.
VLM_MODEL = os.environ.get("VLM_MODEL") or None   # None = visual memory off
USE_VISUAL_MEMORY = VLM_MODEL is not None

# Orchestra dual-agent temperatures
CONDUCTOR_TEMP = 0.6      # strategic reasoning, allows exploration
TOOL_SPECIALIST_TEMP = 0.0  # deterministic command generation

# Phase turn budgets (max turns before forced compression/handover)
MAX_TURNS = {
    "ISSUE_REPRODUCT": 32,
    "TEST_SYNTHSIZE":  32,
    "CODE_LOCALIZE":   48,
    "TEST_LOCALIZE":   24,
    "CODE_FIX":        48,
    "VERIFY_PATCH":    32,
    "ISSUE_CLOSE":     16,
    "FINAL_REPORT":    16,
}

# Token budget — leave this much headroom for responses
MAX_PROMPT_TOKENS = 60_000
CONTEXT_MARGIN = 8_000

# Test-time scaling: number of independent candidate generations
TTS_N = int(os.environ.get("TTS_N", "8"))

# TTS selection weights
TTS_F2P_WEIGHT = 0.3
TTS_P2P_WEIGHT = 0.7

# SWE-bench split to use
SWEBENCH_SPLIT = "verified"   # "verified" (500) or "lite" (300)
