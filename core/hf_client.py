from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any, Dict, List

import httpx

from core.config import get_settings
from core.utils import retry

logger = logging.getLogger(__name__)

HF_API_BASE = "https://api-inference.huggingface.co/models"
DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class HFClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = httpx.Client(timeout=10.0)
        self.model_name = self.settings.hf_model or DEFAULT_MODEL
        self.remote_disabled = False

    @retry()
    def classify(self, text: str) -> Dict[str, Any]:
        if self.remote_disabled or not self.settings.hf_token:
            return self._fallback(text)
        try:
            model_endpoint = f"{HF_API_BASE}/{self.model_name}"
            response = self.session.post(
                model_endpoint,
                headers={"Authorization": f"Bearer {self.settings.hf_token}"},
                json={"inputs": text},
            )
            if response.status_code == HTTPStatus.GONE:
                logger.warning(
                    "HF model %s unavailable (410). Disabling remote classification.",
                    self.model_name,
                )
                self.remote_disabled = True
                return self._fallback(text)

            response.raise_for_status()
            payload = response.json()
            label, score = self._parse_response(payload)
            return {"label": label, "score": score}
        except Exception as exc:
            logger.warning("HF classification failed: %s", exc)
            return self._fallback(text)

    def _parse_response(self, payload: Any) -> tuple[str, float]:
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            candidate = payload[0][0] if payload[0] else {}
        elif isinstance(payload, list) and payload:
            candidate = payload[0]
        else:
            candidate = {}

        label = candidate.get("label", "neutral") if isinstance(candidate, dict) else "neutral"
        score = candidate.get("score", 0.5) if isinstance(candidate, dict) else 0.5
        return label, score

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
