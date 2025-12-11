"""Swarm predictor layer with lightweight agents and volume aggregation.

This module also includes a "connectome" abstraction that lets swarm agents
share information along a lightweight graph, mirroring the structure → dynamics
→ function pattern described in the predictive spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from core.state_space import MarketState, RAW_FEATURE_KEYS


@dataclass
class SwarmVote:
    direction: str
    strength: float
    horizon: str
    target: str


@dataclass
class SwarmSnapshot:
    per_horizon: Dict[str, Dict[str, float]]
    agent_breakdown: List[Dict[str, str]]
    motif_id: Optional[str] = None
    connectome_support: Dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "per_horizon": self.per_horizon,
            "agent_breakdown": self.agent_breakdown,
            "motif_id": self.motif_id,
            "connectome_support": self.connectome_support,
        }


@dataclass
class ConnectomeGraph:
    adjacency: Dict[str, Dict[str, float]]

    @classmethod
    def default(cls) -> "ConnectomeGraph":
        adjacency: Dict[str, Dict[str, float]] = {}
        for feature in RAW_FEATURE_KEYS:
            adjacency.setdefault(feature, {})
        # Couple composite axes to their primary drivers.
        composite_links = {
            "flow_axis": ["net_flow", "exchange_concentration", "stablecoin_rotation"],
            "leverage_axis": ["open_interest", "funding_skew", "perp_basis"],
            "pressure_axis": ["orderbook_imbalance", "aggressive_volume"],
            "headline_axis": ["headline_risk", "headline_count", "headline_recency"],
        }
        for composite, drivers in composite_links.items():
            adjacency.setdefault(composite, {})
            for driver in drivers:
                adjacency[composite][driver] = 0.33
                adjacency[driver][composite] = 0.33
        return cls(adjacency=adjacency)

    def neighborhood(self, node: str) -> Dict[str, float]:
        return self.adjacency.get(node, {})


@dataclass
class SwarmAgentConfig:
    name: str
    feature_subset: List[str]
    horizon: str
    target: str
    threshold: float = 0.5
    direction_labels: tuple[str, str] = ("UP", "DOWN")


class SwarmAgent:
    """Simple linear agent that emits a vote when confident enough."""

    def __init__(self, config: SwarmAgentConfig, coefficients: List[float], intercept: float = 0.0):
        self.config = config
        self.coefficients = np.array(coefficients)
        self.intercept = intercept

    def _feature_vector(self, state: MarketState) -> np.ndarray:
        values = []
        for name in self.config.feature_subset:
            values.append(
                state.normalized_features.get(name)
                or state.composite_axes.get(name)
                or state.raw_features.get(name)
                or 0.0
            )
        return np.array(values)

    def predict(self, state: MarketState) -> Optional[SwarmVote]:
        features = self._feature_vector(state)
        if features.size == 0:
            return None
        margin = float(np.dot(features, self.coefficients) + self.intercept)
        strength = abs(margin)
        if strength < self.config.threshold:
            return None
        direction = self.config.direction_labels[0] if margin >= 0 else self.config.direction_labels[1]
        return SwarmVote(direction=direction, strength=strength, horizon=self.config.horizon, target=self.config.target)


class SwarmEnsemble:
    def __init__(self, agents: Iterable[SwarmAgent]) -> None:
        self.agents = list(agents)
        self.persistence_state: Dict[str, float] = {}

    def _aggregate_votes(self, votes: List[SwarmVote]) -> Dict[str, Dict[str, float]]:
        by_horizon: Dict[str, Dict[str, float]] = {}
        for vote in votes:
            horizon_bucket = by_horizon.setdefault(vote.horizon, {"up_strength": 0.0, "down_strength": 0.0, "total_votes": 0})
            if vote.direction in ("UP", "EVENT_YES"):
                horizon_bucket["up_strength"] += vote.strength
            else:
                horizon_bucket["down_strength"] += vote.strength
            horizon_bucket["total_votes"] += 1

        for horizon, metrics in by_horizon.items():
            denom = metrics["up_strength"] + metrics["down_strength"] or 1.0
            raw_score = (metrics["up_strength"] - metrics["down_strength"]) / denom
            previous = self.persistence_state.get(horizon, 0.0)
            persistence = 0.7 * previous + 0.3 * raw_score
            metrics.update({"swarm_score": raw_score, "persistence": persistence})
            self.persistence_state[horizon] = persistence
        return by_horizon

    def predict(self, state: MarketState, motif_id: str | None = None) -> SwarmSnapshot:
        votes = []
        breakdown = []
        for agent in self.agents:
            vote = agent.predict(state)
            if not vote:
                continue
            votes.append(vote)
            breakdown.append({"name": agent.config.name, "horizon": agent.config.horizon, "target": agent.config.target})

        per_horizon = self._aggregate_votes(votes)
        return SwarmSnapshot(per_horizon=per_horizon, agent_breakdown=breakdown, motif_id=motif_id)


class SwarmPredictor:
    """Geometry-aware connectomic swarm forecaster."""

    def __init__(
        self,
        connectome: ConnectomeGraph,
        ensemble: SwarmEnsemble,
        iterations: int = 3,
        damping: float = 0.25,
    ) -> None:
        self.connectome = connectome
        self.ensemble = ensemble
        self.iterations = iterations
        self.damping = damping

    def _connectome_messages(self, state: MarketState) -> Dict[str, float]:
        messages: Dict[str, float] = {}
        for node, neighbors in self.connectome.adjacency.items():
            base_value = state.normalized_features.get(node) or state.raw_features.get(node) or 0.0
            for neighbor, weight in neighbors.items():
                messages[neighbor] = messages.get(neighbor, 0.0) + base_value * weight
        return messages

    def _geometry_bias(self, coords: Optional[np.ndarray]) -> float:
        if coords is None or coords.size == 0:
            return 0.0
        radius = float(np.linalg.norm(coords))
        return float(np.tanh(radius))

    def forecast(
        self, state: MarketState, coords: Optional[np.ndarray] = None, motif_id: Optional[str] = None
    ) -> SwarmSnapshot:
        snapshot = self.ensemble.predict(state, motif_id=motif_id)
        messages = self._connectome_messages(state)
        geometry_bias = self._geometry_bias(coords)

        support_score = 0.0
        if messages:
            support_score = float(np.tanh(np.mean(list(messages.values())) + geometry_bias))

        # Iteratively blend support into each horizon bucket.
        for _ in range(self.iterations):
            for horizon, metrics in snapshot.per_horizon.items():
                base = metrics.get("swarm_score", 0.0)
                metrics["swarm_score"] = (1 - self.damping) * base + self.damping * support_score
                metrics["connectome_support"] = support_score

        snapshot.connectome_support = {"support_score": support_score, "geometry_bias": geometry_bias}
        return snapshot


__all__ = [
    "ConnectomeGraph",
    "SwarmAgent",
    "SwarmAgentConfig",
    "SwarmEnsemble",
    "SwarmPredictor",
    "SwarmSnapshot",
    "SwarmVote",
]
