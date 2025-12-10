"""Geometry layer: project market states into a low-dimensional field.

The implementation keeps dependencies light while surfacing the conceptual
objects described in the predictive spec: a projection into geometric space,
coarse motif inference, and local drift estimation that can be visualized as a
vector field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from core.state_space import MarketState, load_state_matrix


@dataclass
class GeometrySnapshot:
    coords: List[float]
    motif_id: str
    motif_transition_probs: Dict[str, float]
    local_vector: List[float]

    def to_dict(self) -> dict:
        return {
            "coords": self.coords,
            "motif_id": self.motif_id,
            "motif_transition_probs": self.motif_transition_probs,
            "local_vector": self.local_vector,
        }


class GeometryModel:
    """Lightweight PCA-style projection with motif and drift helpers."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self._components: np.ndarray | None = None
        self._means: np.ndarray | None = None

    def fit(self, history: Iterable[MarketState]) -> None:
        matrix = load_state_matrix(history)
        if matrix.size == 0:
            self._components = None
            self._means = None
            return

        self._means = matrix.mean(axis=0)
        centered = matrix - self._means
        u, _, _ = np.linalg.svd(centered, full_matrices=False)
        self._components = u[:, : self.n_components].T

    def transform(self, state: MarketState) -> np.ndarray:
        vector = np.array(state.vector)
        if vector.size == 0:
            return np.zeros(self.n_components)
        if self._components is None or self._means is None:
            # Fall back to the first n components of the raw vector.
            padded = np.zeros(self.n_components)
            padded[: min(self.n_components, vector.size)] = vector[: self.n_components]
            return padded
        centered = vector - self._means
        return centered @ self._components.T

    def infer_motif(self, coords: Sequence[float]) -> str:
        radius = np.linalg.norm(coords)
        if radius < 0.5:
            return "calm_leverage_build"
        if coords[0] >= 0 and coords[1] >= 0:
            return "grinding_squeeze"
        if coords[0] < 0 and coords[1] < 0:
            return "panic_unwind"
        return "neutral_balance"

    def estimate_local_drift(self, coords_history: np.ndarray) -> np.ndarray:
        if coords_history.shape[0] < 2:
            return np.zeros_like(coords_history[0]) if coords_history.size else np.zeros(self.n_components)
        recent = coords_history[-3:]
        diffs = np.diff(recent, axis=0)
        return diffs.mean(axis=0)

    def transition_probabilities(self, motif: str) -> Dict[str, float]:
        # Placeholder transitions emphasize clarity over sophistication.
        if motif == "grinding_squeeze":
            return {"5m": 0.6, "1h": 0.65, "4h": 0.5}
        if motif == "panic_unwind":
            return {"5m": 0.4, "1h": 0.35, "4h": 0.3}
        if motif == "calm_leverage_build":
            return {"5m": 0.55, "1h": 0.6, "4h": 0.55}
        return {"5m": 0.5, "1h": 0.5, "4h": 0.5}

    def snapshot(self, state: MarketState, history: List[MarketState]) -> GeometrySnapshot:
        coords_history = np.array([self.transform(s) for s in history]) if history else np.zeros((0, self.n_components))
        coords = self.transform(state)
        motif = self.infer_motif(coords)
        drift = self.estimate_local_drift(np.vstack([coords_history, coords]) if coords_history.size else np.array([coords]))
        transitions = self.transition_probabilities(motif)

        return GeometrySnapshot(
            coords=[float(v) for v in coords],
            motif_id=motif,
            motif_transition_probs=transitions,
            local_vector=[float(v) for v in drift],
        )


__all__ = ["GeometryModel", "GeometrySnapshot"]
