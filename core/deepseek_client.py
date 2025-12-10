from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from core.config import get_settings
from core.utils import retry

logger = logging.getLogger(__name__)

DEEPSEEK_BASE = "https://api.deepseek.com"


class DeepSeekClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = httpx.Client(timeout=10.0)

    @retry()
    def enrich(self, text: str) -> Dict[str, Any]:
        if not self.settings.deepseek_api_key:
            return self._fallback(text)
        try:
            response = self.session.post(
                f"{DEEPSEEK_BASE}/v1/enrich",
                headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                json={"text": text},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("DeepSeek enrichment failed: %s", exc)
            return self._fallback(text)

    def _fallback(self, text: str) -> Dict[str, Any]:
        return {"text": text, "sentiment": "neutral", "confidence": 0.5, "topics": ["xrp", "macro"]}
