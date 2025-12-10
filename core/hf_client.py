from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from core.config import get_settings
from core.utils import retry

logger = logging.getLogger(__name__)

HF_API_BASE = "https://api-inference.huggingface.co/models"
MODEL_NAME = "distilbert-base-uncased"


class HFClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = httpx.Client(timeout=10.0)

    @retry()
    def classify(self, text: str) -> Dict[str, Any]:
        if not self.settings.hf_token:
            return self._fallback(text)
        try:
            response = self.session.post(
                f"{HF_API_BASE}/{MODEL_NAME}",
                headers={"Authorization": f"Bearer {self.settings.hf_token}"},
                json={"inputs": text},
            )
            response.raise_for_status()
            payload = response.json()
            label = payload[0][0].get("label", "neutral") if isinstance(payload, list) else "neutral"
            score = payload[0][0].get("score", 0.5) if isinstance(payload, list) else 0.5
            return {"label": label, "score": score}
        except Exception as exc:
            logger.warning("HF classification failed: %s", exc)
            return self._fallback(text)

    def _fallback(self, text: str) -> Dict[str, Any]:
        keyword_map = {
            "sec": "regulatory",
            "court": "regulatory",
            "inflation": "macro",
            "fed": "macro",
            "liquidity": "market",
        }
        label = "market"
        for key, mapped in keyword_map.items():
            if key in text.lower():
                label = mapped
                break
        return {"label": label, "score": 0.5}
