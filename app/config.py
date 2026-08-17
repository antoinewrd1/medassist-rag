"""Environment-driven settings. One place decides which backends are live."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


@dataclass(frozen=True)
class Settings:
    llm_backend: str        # "fake" | "openai"
    embed_backend: str      # "local" | "openai"
    db_url: str
    chat_model: str
    embed_model: str
    top_k: int
    min_score: float
    index_dir: Path

    @property
    def offline(self) -> bool:
        """True when no network or API key is required to serve a request."""
        return self.llm_backend == "fake" and self.embed_backend == "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        llm_backend=_env("MA_LLM_BACKEND", "fake").lower(),
        embed_backend=_env("MA_EMBED_BACKEND", "local").lower(),
        db_url=_env("MA_DB_URL", "sqlite:///./medassist.db"),
        chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=_env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        top_k=int(_env("MA_TOP_K", "3")),
        min_score=float(_env("MA_MIN_SCORE", "0.12")),
        index_dir=Path(_env("MA_INDEX_DIR", "_index")),
    )
