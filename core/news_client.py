from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

import httpx

from core.config import get_settings
from core.utils import retry

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2"


class NewsClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = httpx.Client(timeout=10.0)

    @retry()
    def fetch_headlines(self, query: str = "XRP") -> List[Dict[str, Any]]:
        if not self.settings.news_api_key:
            return self._fallback_headlines()
        try:
            response = self.session.get(
                f"{NEWS_API_BASE}/everything",
                params={"q": query, "apiKey": self.settings.news_api_key, "language": "en", "pageSize": 10},
            )
            response.raise_for_status()
            data = response.json().get("articles", [])
            return [self._format_article(article) for article in data]
        except Exception as exc:
            logger.warning("News API request failed: %s", exc)
            return self._fallback_headlines()

    def _format_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "headline": article.get("title", ""),
            "source": article.get("source", {}).get("name", "unknown"),
            "url": article.get("url", ""),
            "published_at": self._parse_date(article.get("publishedAt")),
            "summary": article.get("description", ""),
        }

    def _parse_date(self, value: str | None) -> datetime:
        if not value:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return datetime.utcnow()

    def _fallback_headlines(self) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        return [
            {
                "headline": "XRP liquidity stabilizes across major venues",
                "source": "DeterministicFeed",
                "url": "https://example.com/xrp-liquidity",
                "published_at": now,
                "summary": "Placeholder headline generated offline.",
            }
        ]
