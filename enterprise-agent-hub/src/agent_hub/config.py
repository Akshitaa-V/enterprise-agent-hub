"""Paths and runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("HUB_DATA_DIR", ROOT / "data")))
    var_dir: Path = field(default_factory=lambda: Path(os.getenv("HUB_VAR_DIR", ROOT / "var")))
    # "scripted" runs fully offline and deterministically; "openai" / "anthropic" use a hosted model.
    llm_provider: str = field(default_factory=lambda: os.getenv("HUB_LLM_PROVIDER", "scripted"))
    llm_model: str = field(default_factory=lambda: os.getenv("HUB_LLM_MODEL", ""))
    max_hops: int = field(default_factory=lambda: int(os.getenv("HUB_MAX_HOPS", "6")))
    max_tool_calls_per_agent: int = field(default_factory=lambda: int(os.getenv("HUB_MAX_TOOL_CALLS", "4")))
    # Credit notes above this amount need the head of finance (see data/docs/credit_notes.md).
    finance_approval_threshold: float = 500.0

    @property
    def erp_db(self) -> Path:
        return self.var_dir / "erp.db"

    @property
    def dwh_db(self) -> Path:
        return self.var_dir / "dwh.db"

    @property
    def runs_db(self) -> Path:
        return self.var_dir / "runs.db"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def legacy_export(self) -> Path:
        return self.data_dir / "legacy" / "inventory_export.txt"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_settings() -> Settings:
    return Settings()
