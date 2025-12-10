from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis

from core.config import get_settings

logger = logging.getLogger(__name__)
_CLIENT: redis.Redis | InMemoryRedis | None = None


class InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[bytes]:
        value = self._store.get(key)
        return value.encode() if isinstance(value, str) else value

    def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        if isinstance(value, bytes):
            value = value.decode()
        self._store[key] = value
        return True

    def ping(self) -> bool:
        return True


def _build_client():
    settings = get_settings()
    try:
        client = redis.from_url(settings.redis_url, decode_responses=False)
        client.ping()
        return client
    except Exception:
        logger.warning("Falling back to in-memory Redis client")
        return InMemoryRedis()


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_client()
    return _CLIENT


def cache_snapshot(key: str, payload: Any, expire: int = 600) -> None:
    client = get_client()
    serialized = json.dumps(payload, default=str)
    client.set(key, serialized, ex=expire)


def get_snapshot(key: str) -> Any:
    client = get_client()
    raw = client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


__all__ = ["get_client", "cache_snapshot", "get_snapshot"]
