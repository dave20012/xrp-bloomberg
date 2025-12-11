"""Geometry layer: project market states into a low-dimensional field.

The implementation keeps dependencies light while surfacing the conceptual
objects described in the predictive spec: a projection into geometric space,
coarse motif inference, and local drift estimation that can be visualized as a
vector field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from core.state_space import (
    N_FEATURES,
    N_GEOMETRY_COMPONENTS,
    build_state_vector,
    load_state_matrix,
)


@dataclass
class GeometrySnapshot:
    coords: np.ndarray
    motif_id: Optional[str]
    motif_transition_probs: Dict[str, float]
    local_drift: np.ndarray

    def to_dict(self) -> dict:
        return {
            "coords": self.coords.tolist(),
            "motif_id": self.motif_id,
            "motif_transition_probs": self.motif_transition_probs,
            "local_drift": self.local_drift.tolist(),
        }


class GeometryModel:
    """Lightweight PCA-style projection with motif and drift helpers."""

    def __init__(
        self,
        n_components: int = N_GEOMETRY_COMPONENTS,
        n_features: int = N_FEATURES,
    ) -> None:
        self.n_components = n_components
        self.n_features = n_features
        self._components: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._coords_history: np.ndarray | None = None

    def _validate_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = build_state_vector(vector)
        return arr.reshape((self.n_features,))

    def _zero_snapshot(self) -> GeometrySnapshot:
        zeros = np.zeros((self.n_components,))
        return GeometrySnapshot(
            coords=zeros,
            motif_id=None,
            motif_transition_probs={},
            local_drift=zeros,
        )

    def fit(self, history: Iterable[Sequence[float]]) -> None:
        matrix = load_state_matrix(history)
        if matrix.shape[0] < 10:
            self._components = None
            self._means = None
            self._coords_history = None
            return

        if matrix.shape[1] != self.n_features:
            raise ValueError(
                f"Geometry fit expected {self.n_features} features; received {matrix.shape[1]}."
            )

        self._means = matrix.mean(axis=0).astype(float)
        centered = matrix - self._means
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[: self.n_components]
        assert components.shape == (self.n_components, self.n_features), (
            "Geometry components shape mismatch; expected "
            f"{(self.n_components, self.n_features)}, got {components.shape}."
        )
        self._components = components
        self._coords_history = centered @ self._components.T

    def transform(self, state_vector: Sequence[float]) -> np.ndarray:
        vector = self._validate_vector(state_vector)
        if self._components is None or self._means is None:
            return np.zeros((self.n_components,))
        assert vector.shape == (self.n_features,), (
            f"Transform received vector of shape {vector.shape}, expected {(self.n_features,)}."
        )
        assert self._components.shape == (self.n_components, self.n_features), (
            "Geometry components shape mismatch; expected "
            f"{(self.n_components, self.n_features)}, got {self._components.shape}."
        )
        centered = vector - self._means
        return centered @ self._components.T

    def infer_motif(self, coords: Sequence[float]) -> Optional[str]:
        if coords is None:
            return None
        radius = np.linalg.norm(coords)
        if radius < 0.5:
            return "calm_leverage_build"
        if coords[0] >= 0 and coords[1] >= 0:
            return "grinding_squeeze"
        if coords[0] < 0 and coords[1] < 0:
            return "panic_unwind"
        return "neutral_balance"

    def estimate_local_drift(self, coords: np.ndarray) -> np.ndarray:
        if self._coords_history is None:
            return np.zeros((self.n_components,))
        history = np.vstack([self._coords_history, coords])
        if history.shape[0] < 2:
            return np.zeros((self.n_components,))
        recent = history[-3:]
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

    def snapshot(self, state_vector: Sequence[float]) -> GeometrySnapshot:
        try:
            state = self._validate_vector(state_vector)
        except ValueError:
            return self._zero_snapshot()

        if self._components is None or self._means is None:
            return self._zero_snapshot()

        assert state.shape == (self.n_features,), (
            f"Snapshot received vector of shape {state.shape}, expected {(self.n_features,)}."
        )
        assert self._components.shape == (self.n_components, self.n_features), (
            "Geometry components shape mismatch; expected "
            f"{(self.n_components, self.n_features)}, got {self._components.shape}."
        )

        centered = state - self._means
        coords = centered @ self._components.T

        motif = self.infer_motif(coords)
        drift = self.estimate_local_drift(coords)
        transitions = self.transition_probabilities(motif)

        return GeometrySnapshot(
            coords=np.asarray(coords, dtype=float),
            motif_id=motif,
            motif_transition_probs=transitions,
            local_drift=np.asarray(drift, dtype=float),
        )


__all__ = ["GeometryModel", "GeometrySnapshot"]
