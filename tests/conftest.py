import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def index():
    from app.rag.index import GuidanceIndex
    return GuidanceIndex.build()


@pytest.fixture
def fake_llm():
    from app.llm import FakeLLM
    return FakeLLM()
