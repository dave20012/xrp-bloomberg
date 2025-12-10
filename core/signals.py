from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np

from core.exchange_addresses import EXCHANGE_ADDRESSES
from core.utils import ewma, zscore


@dataclass
class SignalResult:
    anomaly_z: float
    accumulation_score: float
    flow_pressure: float
    leverage_regime: float
    manipulation_score: float
    composite: float


def volume_anomaly(volumes: List[float]) -> float:
    if len(volumes) < 5:
        return 0.0
    return float(zscore(volumes)[-1])


def accumulation_distribution(flows: List[Dict[str, float]], prices: List[float]) -> float:
    if not flows or not prices:
        return 0.0
    flow_volume = sum(item.get("volume", 0) for item in flows)
    price_change = prices[-1] - prices[0] if len(prices) > 1 else 0
    divergence = price_change - flow_volume * 1e-6
    return max(min(divergence, 1.0), -1.0)


def flow_pressure(flows: List[Dict[str, float]]) -> float:
    if not flows:
        return 0.0
    exchange_weights = {addr: 1 for addresses in EXCHANGE_ADDRESSES.values() for addr in addresses}
    pressure = 0.0
    for flow in flows:
        direction = flow.get("direction", "inflow")
        sign = 1 if direction == "inflow" else -1
        pressure += sign * flow.get("volume", 0) * 1e-6
    normalized = max(min(pressure, 1.0), -1.0)
    return normalized


def leverage_regime(open_interest: float, funding_rates: List[float], long_short_ratios: List[float]) -> float:
    if open_interest <= 0:
        return 0.0
    funding_bias = np.mean(funding_rates) if funding_rates else 0
    lsr_bias = np.mean(long_short_ratios) - 1 if long_short_ratios else 0
    raw = np.tanh(open_interest * 1e-8 + funding_bias + lsr_bias)
    return float(raw)


def manipulation_heuristic(depth_imbalance: float = 0.0, spoofing_score: float = 0.0) -> float:
    combined = depth_imbalance * 0.6 + spoofing_score * 0.4
    return max(min(combined, 1.0), -1.0)


def composite_score(values: Dict[str, float]) -> float:
    weights = {
        "anomaly": 0.2,
        "accumulation": 0.2,
        "flow_pressure": 0.2,
        "leverage": 0.2,
        "manipulation": 0.2,
    }
    score = 0.0
    for key, weight in weights.items():
        score += values.get(key, 0.0) * weight
    return max(min(score, 1.0), -1.0)


def build_signals(
    volumes: List[float],
    flows: List[Dict[str, float]],
    prices: List[float],
    open_interest: float,
    funding_rates: List[float],
    long_short_ratios: List[float],
    depth_imbalance: float = 0.0,
    spoofing_score: float = 0.0,
) -> SignalResult:
    anomaly = volume_anomaly(volumes)
    accumulation = accumulation_distribution(flows, prices)
    pressure = flow_pressure(flows)
    leverage = leverage_regime(open_interest, funding_rates, long_short_ratios)
    manipulation = manipulation_heuristic(depth_imbalance, spoofing_score)
    composite = composite_score(
        {
            "anomaly": anomaly,
            "accumulation": accumulation,
            "flow_pressure": pressure,
            "leverage": leverage,
            "manipulation": manipulation,
        }
    )
    return SignalResult(
        anomaly_z=anomaly,
        accumulation_score=accumulation,
        flow_pressure=pressure,
        leverage_regime=leverage,
        manipulation_score=manipulation,
        composite=composite,
    )
