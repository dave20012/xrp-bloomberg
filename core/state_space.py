"""State space construction utilities for the predictive geometry layer.

This module treats incoming market data as a compact, normalized vector with
explicit composite axes. It intentionally keeps the feature engineering simple
and transparent so downstream geometry and swarm layers can operate without a
heavyweight modeling dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

N_FEATURES = 18
N_COMPONENTS = 2

RAW_FEATURE_KEYS: List[str] = [
    "spot_price",
    "returns",
    "realized_vol",
    "net_flow",
    "exchange_concentration",
    "stablecoin_rotation",
    "open_interest",
    "funding_skew",
    "perp_basis",
    "orderbook_imbalance",
    "aggressive_volume",
    "headline_risk",
    "headline_count",
    "headline_recency",
]

COMPOSITE_KEYS: List[str] = [
    "flow_axis",
    "leverage_axis",
    "pressure_axis",
    "headline_axis",
]


def _validate_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size != N_FEATURES:
        raise ValueError(f"State vector must have {N_FEATURES} features; received {vector.size}.")
    return vector


@dataclass
class MarketState:
    """Container for a single market state snapshot."""

    timestamp: datetime
    raw_features: Dict[str, float]
    normalized_features: Dict[str, float]
    composite_axes: Dict[str, float]
    vector: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES, dtype=float))

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
            "composite_axes": self.composite_axes,
            "vector": self.vector.tolist(),
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
    for key in RAW_FEATURE_KEYS:
        value = float(raw_inputs.get(key, 0.0))
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


def _vector_from_features(
    normalized_features: Mapping[str, float],
    composite_axes: Mapping[str, float],
) -> np.ndarray:
    values = [float(normalized_features.get(key, 0.0)) for key in RAW_FEATURE_KEYS]
    values.extend(float(composite_axes.get(key, 0.0)) for key in COMPOSITE_KEYS)
    return _validate_vector(np.array(values, dtype=float))


def build_state_vector(
    raw_inputs: Mapping[str, float] | Sequence[float],
    rolling_stats: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Construct a flat state vector of fixed dimensionality.

    Accepts either a mapping of raw feature values or a precomputed sequence,
    enforcing a strict shape of ``(N_FEATURES,)``.
    """

    if isinstance(raw_inputs, Mapping):
        normalized_features = _normalize_features(raw_inputs, rolling_stats)
        composite = _composite_axes(normalized_features)
        return _vector_from_features(normalized_features, composite)

    return _validate_vector(np.asarray(raw_inputs, dtype=float))


def build_market_state(
    timestamp: datetime,
    raw_inputs: Mapping[str, float],
    rolling_stats: Mapping[str, tuple[float, float]] | None = None,
) -> MarketState:
    normalized_features = _normalize_features(raw_inputs, rolling_stats)
    composite = _composite_axes(normalized_features)
    vector = _vector_from_features(normalized_features, composite)
    return MarketState(
        timestamp=timestamp,
        raw_features=dict(raw_inputs),
        normalized_features=normalized_features,
        composite_axes=composite,
        vector=vector,
    )


def load_state_matrix(states: Iterable[MarketState | np.ndarray | Sequence[float]]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for state in states:
        if isinstance(state, MarketState):
            vector = state.vector
        else:
            vector = state
        try:
            vectors.append(_validate_vector(vector))
        except ValueError:
            continue
    if not vectors:
        return np.zeros((0, N_FEATURES))
    return np.vstack(vectors)


__all__ = [
    "MarketState",
    "N_COMPONENTS",
    "N_FEATURES",
    "build_market_state",
    "build_state_vector",
    "load_state_matrix",
]
