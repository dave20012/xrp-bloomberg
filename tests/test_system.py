import os
import sys

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db import create_tables, engine  # noqa: E402
from core.redis_client import cache_snapshot, get_snapshot  # noqa: E402
from core.signals import build_signals  # noqa: E402


def test_create_tables():
    create_tables()
    inspector = engine.inspect(engine)
    tables = inspector.get_table_names()
    assert "flows" in tables
    assert "ohlcv" in tables
    assert "openinterest" in tables
    assert "scores" in tables
    assert "news" in tables


def test_redis_cache_snapshot():
    payload = {"hello": "world"}
    cache_snapshot("pytest:key", payload)
    cached = get_snapshot("pytest:key")
    assert cached == payload


def test_signals_deterministic():
    result = build_signals(
        volumes=[100, 120, 130, 125, 140, 150],
        flows=[{"volume": 100000, "direction": "inflow"}],
        prices=[0.5, 0.51, 0.515, 0.52],
        open_interest=1_000_000,
        funding_rates=[0.0001, 0.0002],
        long_short_ratios=[1.05, 1.02],
        depth_imbalance=0.1,
        spoofing_score=0.2,
    )
    assert -1.0 <= result.composite <= 1.0
    assert result.manipulation_score >= 0
