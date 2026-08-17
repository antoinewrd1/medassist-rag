"""LLM abstraction: OpenAI, or a deterministic offline stand-in.

The fake is not a simulation of intelligence. It selects the highest-scoring
retrieved passage, returns its first sentence with a citation, and emits a
conservative triage suggestion. It exists so that tests, CI, and the eval
harness run with no key, no network, and no spend -- and so a reviewer can
clone the repo and see the full request path work in one command.

Any metric produced under the fake backend measures the harness and the
guardrails, not a model. The eval report records which backend produced it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.config import get_settings


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = get_settings().chat_model

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,     # reproducible for eval comparison
            "max_tokens": 600,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        u = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            usage={
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
            },
        )


class FakeLLM(BaseLLM):
    """Deterministic. Extractive, never generative -- see the module docstring."""

    name = "fake"

    _PASSAGE_RE = re.compile(r"\[(doc-[\w-]+)\]\s*(.+)")

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        passages = self._PASSAGE_RE.findall(user)
        if not passages:
            return LLMResponse(
                json.dumps(
                    {
                        "summary": "INSUFFICIENT_CONTEXT",
                        "citations": [],
                        "suggested_triage": "routine",
                        "next_steps": [],
                    }
                ),
                model="fake",
            )

        doc_id, body = passages[0]
        first = body.split(". ")[0].strip().rstrip(".")
        return LLMResponse(
            json.dumps(
                {
                    "summary": f"{first}. [{doc_id}]",
                    "citations": [doc_id],
                    # The stub never de-escalates; safety.clamp() decides finally.
                    "suggested_triage": "routine",
                    "next_steps": [
                        "Discuss these symptoms with a clinician",
                        "Seek care sooner if symptoms worsen",
                    ],
                }
            ),
            model="fake",
        )


def get_llm() -> BaseLLM:
    backend = get_settings().llm_backend
    if backend == "openai":
        return OpenAILLM()
    if backend == "fake":
        return FakeLLM()
    raise ValueError(f"unknown MA_LLM_BACKEND={backend!r}")
