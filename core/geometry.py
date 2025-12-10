"""Geometry layer: project market states into a low-dimensional field.

The implementation keeps dependencies light while surfacing the conceptual
objects described in the predictive spec: a projection into geometric space,
coarse motif inference, and local drift estimation that can be visualized as a
vector field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.state_space import N_COMPONENTS, N_FEATURES, load_state_matrix


@dataclass
class GeometrySnapshot:
    coords: List[float]
    motif_id: Optional[str]
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

    def __init__(self, n_components: int = N_COMPONENTS, n_features: int = N_FEATURES) -> None:
        self.n_components = n_components
        self.n_features = n_features
        self._components: np.ndarray | None = None
        self._means: np.ndarray | None = None

    def _validate_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float).reshape(-1)
        if arr.size != self.n_features:
            raise ValueError(
                f"State vector has {arr.size} dimensions; expected {self.n_features}."
            )
        return arr

    def fit(self, history: Iterable[Sequence[float]]) -> None:
        matrix = load_state_matrix(history)
        if matrix.shape[0] < max(self.n_components, 10):
            self._components = None
            self._means = None
            return

        self._means = matrix.mean(axis=0)
        centered = matrix - self._means
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[: self.n_components]
        self._components = components

    def transform(self, state_vector: Sequence[float]) -> np.ndarray:
        vector = self._validate_vector(state_vector)
        if self._components is None or self._means is None:
            return np.zeros(self.n_components)
        centered = vector - self._means
        return centered @ self._components.T

    def infer_motif(self, coords: Sequence[float]) -> Optional[str]:
        if not coords:
            return None
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
            return np.zeros(self.n_components)
        recent = coords_history[-3:]
        diffs = np.diff(recent, axis=0)
        return diffs.mean(axis=0)

    def transition_probabilities(self, motif: Optional[str]) -> Dict[str, float]:
        if motif is None:
            return {}
        if motif == "grinding_squeeze":
            return {"5m": 0.6, "1h": 0.65, "4h": 0.5}
        if motif == "panic_unwind":
            return {"5m": 0.4, "1h": 0.35, "4h": 0.3}
        if motif == "calm_leverage_build":
            return {"5m": 0.55, "1h": 0.6, "4h": 0.55}
        return {"5m": 0.5, "1h": 0.5, "4h": 0.5}

    def snapshot(self, state_vector: Sequence[float], history: List[Sequence[float]]) -> GeometrySnapshot:
        history_matrix = load_state_matrix(history)
        if history_matrix.shape[0] < max(self.n_components, 10):
            zero_coords = [0.0] * self.n_components
            return GeometrySnapshot(
                coords=zero_coords,
                motif_id=None,
                motif_transition_probs={},
                local_vector=zero_coords,
            )

        coords_history = np.array([self.transform(vec) for vec in history])
        coords = self.transform(state_vector)
        motif = self.infer_motif(coords)
        drift = self.estimate_local_drift(np.vstack([coords_history, coords]))
        transitions = self.transition_probabilities(motif)

        return GeometrySnapshot(
            coords=[float(v) for v in coords],
            motif_id=motif,
            motif_transition_probs=transitions,
            local_vector=[float(v) for v in drift],
        )


__all__ = ["GeometryModel", "GeometrySnapshot"]
