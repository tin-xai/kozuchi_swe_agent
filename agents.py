"""
Orchestra dual-agent pattern: Conductor (strategic) + ToolSpecialist (syntactic).
Both share the same base model on OpenRouter, differ only in temperature.
"""
from __future__ import annotations

import json
import time
from typing import Any

import tiktoken
from openai import OpenAI

import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )
    return _client


def count_tokens(messages: list[dict]) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return sum(len(str(m)) // 4 for m in messages)
    total = 0
    for msg in messages:
        total += 4
        for val in msg.values():
            if isinstance(val, str):
                total += len(enc.encode(val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "text" in item:
                        total += len(enc.encode(item["text"]))
    return total


def _call_llm(
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    retries: int = 3,
) -> dict:
    client = get_client()
    kwargs: dict[str, Any] = dict(
        model=config.MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            return {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
            }
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[LLM] error (attempt {attempt+1}): {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")


class ConductorAgent:
    """Strategic reasoning agent. Generates hypotheses and plans."""

    def __init__(self, system_prompt: str, tools: list[dict]):
        self.system_prompt = system_prompt
        self.tools = tools
        self.history: list[dict] = []
        self._reset_history()

    def _reset_history(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def step(self, user_content: str) -> dict:
        self.history.append({"role": "user", "content": user_content})
        response = _call_llm(self.history, self.tools, config.CONDUCTOR_TEMP)
        self.history.append(response)
        return response

    def add_tool_result(self, tool_call_id: str, result: str):
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })

    def compress(self, handover_memo: str):
        """Rebuild history from a handover memo, discarding prior turns."""
        self._reset_history()
        self.history.append({
            "role": "user",
            "content": f"<handover_memo>\n{handover_memo}\n</handover_memo>\nContinue from where the previous context left off.",
        })

    def token_count(self) -> int:
        return count_tokens(self.history)


class ToolSpecialistAgent:
    """Validates and stabilises command syntax at temperature=0."""

    def __init__(self, system_prompt: str, tools: list[dict]):
        self.system_prompt = system_prompt
        self.tools = tools
        self.history: list[dict] = []
        self._reset_history()

    def _reset_history(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def validate_and_emit(self, conductor_intent: str, context: str = "") -> dict:
        """Given conductor's stated intent, produce a concrete tool call."""
        prompt = conductor_intent
        if context:
            prompt = f"{context}\n\n{conductor_intent}"
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.history.append({"role": "user", "content": prompt})
        response = _call_llm(self.history, self.tools, config.TOOL_SPECIALIST_TEMP)
        self.history.append(response)
        return response

    def add_tool_result(self, tool_call_id: str, result: str):
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })
