"""State space construction utilities for the predictive geometry layer.

This module treats incoming market data as a compact, normalized vector with
explicit composite axes. It intentionally keeps the feature engineering simple
and transparent so downstream geometry and swarm layers can operate without a
heavyweight modeling dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Mapping

import numpy as np


@dataclass
class MarketState:
    """Container for a single market state snapshot."""

    timestamp: datetime
    raw_features: Dict[str, float]
    normalized_features: Dict[str, float]
    composite_axes: Dict[str, float]
    vector: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
            "composite_axes": self.composite_axes,
            "vector": self.vector,
        }


def _zscore(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def _normalize_features(
    raw_inputs: Mapping[str, float],
    rolling_stats: Mapping[str, tuple[float, float]] | None = None,
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in raw_inputs.items():
        if rolling_stats and key in rolling_stats:
            mean, std = rolling_stats[key]
        else:
            mean, std = value, abs(value) if value != 0 else 1.0
        normalized[key] = _zscore(value, mean, std)
    return normalized


def _composite_axes(normalized_features: Mapping[str, float]) -> Dict[str, float]:
    flow_axis = np.mean(
        [
            normalized_features.get("net_flow", 0.0),
            normalized_features.get("exchange_concentration", 0.0),
            normalized_features.get("stablecoin_rotation", 0.0),
        ]
    )
    leverage_axis = np.mean(
        [
            normalized_features.get("open_interest", 0.0),
            normalized_features.get("funding_skew", 0.0),
            normalized_features.get("perp_basis", 0.0),
        ]
    )
    pressure_axis = np.mean(
        [
            normalized_features.get("orderbook_imbalance", 0.0),
            normalized_features.get("aggressive_volume", 0.0),
        ]
    )
    headline_axis = np.mean(
        [
            normalized_features.get("headline_risk", 0.0),
            normalized_features.get("headline_count", 0.0),
            normalized_features.get("headline_recency", 0.0),
        ]
    )

    return {
        "flow_axis": float(flow_axis),
        "leverage_axis": float(leverage_axis),
        "pressure_axis": float(pressure_axis),
        "headline_axis": float(headline_axis),
    }


def build_state_vector(
    timestamp: datetime,
    raw_inputs: Mapping[str, float],
    rolling_stats: Mapping[str, tuple[float, float]] | None = None,
) -> MarketState:
    normalized_features = _normalize_features(raw_inputs, rolling_stats)
    composite = _composite_axes(normalized_features)

    vector = [*normalized_features.values(), *composite.values()]
    return MarketState(
        timestamp=timestamp,
        raw_features=dict(raw_inputs),
        normalized_features=normalized_features,
        composite_axes=composite,
        vector=[float(v) for v in vector],
    )


def load_state_matrix(states: Iterable[MarketState]) -> np.ndarray:
    vectors: List[List[float]] = [state.vector for state in states if state.vector]
    if not vectors:
        return np.zeros((0, 0))
    return np.vstack(vectors)


__all__ = ["MarketState", "build_state_vector", "load_state_matrix"]
