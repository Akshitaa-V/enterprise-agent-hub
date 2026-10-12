import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_hub.config import Settings  # noqa: E402
from agent_hub.seed import seed_all  # noqa: E402


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Fresh, seeded databases for every test, in a temporary directory."""
    s = replace(Settings(), var_dir=tmp_path, llm_provider="scripted")
    seed_all(s)
    return s
