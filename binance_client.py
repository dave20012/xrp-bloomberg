from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx

from core.config import get_settings
from core.utils import retry

logger = logging.getLogger(__name__)

BINANCE_API_BASE = "https://api.binance.com"
BINANCE_FAPI_BASE = "https://fapi.binance.com"


class BinanceClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = httpx.Client(timeout=10.0)

    @retry()
    def _get(self, url: str, params: dict | None = None) -> Any:
        headers = {}
        if self.settings.binance_api_key:
            headers["X-MBX-APIKEY"] = self.settings.binance_api_key
        try:
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("Binance request failed: %s", exc)
            return None

    def fetch_agg_trades(self, symbol: str = "XRPUSDT", limit: int = 200) -> List[Dict[str, Any]]:
        data = self._get(f"{BINANCE_API_BASE}/api/v3/aggTrades", params={"symbol": symbol, "limit": limit})
        if not data:
            return self._fallback_trades(limit)
        trades: List[Dict[str, Any]] = []
        for item in data:
            trades.append(
                {
                    "price": float(item.get("p", 0)),
                    "quantity": float(item.get("q", 0)),
                    "timestamp": datetime.fromtimestamp(item.get("T", 0) / 1000),
                    "is_buyer_maker": bool(item.get("m", False)),
                }
            )
        return trades

    def fetch_klines(self, symbol: str = "XRPUSDT", interval: str = "1h", limit: int = 200) -> List[Dict[str, Any]]:
        data = self._get(
            f"{BINANCE_API_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not data:
            return self._fallback_klines(limit)
        candles: List[Dict[str, Any]] = []
        for kline in data:
            candles.append(
                {
                    "open_time": datetime.fromtimestamp(kline[0] / 1000),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                }
            )
        return candles

    def fetch_futures_open_interest(self, symbol: str = "XRPUSDT") -> Dict[str, Any]:
        data = self._get(f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist", params={"symbol": symbol, "period": "5m", "limit": 30})
        if not data:
            return {"symbol": symbol, "openInterest": 1000000, "timestamp": datetime.utcnow()}
        latest = data[-1]
        return {
            "symbol": symbol,
            "openInterest": float(latest.get("sumOpenInterest", 0)),
            "timestamp": datetime.fromtimestamp(int(latest.get("timestamp", 0)) / 1000),
        }

    def fetch_funding_rates(self, symbol: str = "XRPUSDT") -> List[Dict[str, Any]]:
        data = self._get(f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 100})
        if not data:
            now = datetime.utcnow()
            return [
                {
                    "fundingRate": 0.0001,
                    "fundingTime": now - timedelta(hours=i * 8),
                }
                for i in range(5)
            ]
        return [
            {
                "fundingRate": float(item.get("fundingRate", 0)),
                "fundingTime": datetime.fromtimestamp(item.get("fundingTime", 0) / 1000),
            }
            for item in data
        ]

    def fetch_long_short_ratio(self, symbol: str = "XRPUSDT", period: str = "5m") -> List[Dict[str, Any]]:
        data = self._get(
            f"{BINANCE_FAPI_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": 50},
        )
        if not data:
            now = datetime.utcnow()
            return [
                {"longShortRatio": 1.0 + i * 0.01, "timestamp": now - timedelta(minutes=5 * i)}
                for i in range(10)
            ]
        return [
            {
                "longShortRatio": float(item.get("longShortRatio", 0)),
                "timestamp": datetime.fromtimestamp(int(item.get("timestamp", 0)) / 1000),
            }
            for item in data
        ]

    def _fallback_trades(self, limit: int) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        return [
            {
                "price": 0.5 + i * 0.0001,
                "quantity": 100 + i,
                "timestamp": now - timedelta(seconds=i * 15),
                "is_buyer_maker": i % 2 == 0,
            }
            for i in range(limit)
        ]

    def _fallback_klines(self, limit: int) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        candles = []
        base_price = 0.5
        for i in range(limit):
            open_time = now - timedelta(hours=limit - i)
            candles.append(
                {
                    "open_time": open_time,
                    "open": base_price + i * 0.001,
                    "high": base_price + i * 0.0015,
                    "low": base_price + i * 0.0005,
                    "close": base_price + i * 0.001,
                    "volume": 100000 + i * 500,
                }
            )
        return candles
