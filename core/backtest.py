"""Backtesting helpers for swarm reliability analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from core.swarm import SwarmSnapshot


@dataclass
class SwarmPerformance:
    hit_rate: float
    average_payoff: float
    horizon: str


def compute_hit_rate(predictions: Iterable[float], outcomes: Iterable[float]) -> float:
    preds = np.array(list(predictions))
    outs = np.array(list(outcomes))
    if preds.size == 0 or outs.size == 0:
        return 0.0
    agreement = np.sign(preds) == np.sign(outs)
    return float(agreement.mean())


def evaluate_swarm(snapshots: List[SwarmSnapshot], realized_returns: List[float]) -> List[SwarmPerformance]:
    performances: List[SwarmPerformance] = []
    if not snapshots or not realized_returns:
        return performances

    # Align by index for simplicity.
    for horizon in snapshots[-1].per_horizon.keys():
        preds = [snap.per_horizon.get(horizon, {}).get("swarm_score", 0.0) for snap in snapshots]
        hit_rate = compute_hit_rate(preds, realized_returns[: len(preds)])
        average_payoff = float(np.mean(np.array(preds) * np.array(realized_returns[: len(preds)])))
        performances.append(SwarmPerformance(hit_rate=hit_rate, average_payoff=average_payoff, horizon=horizon))
    return performances


__all__ = ["compute_hit_rate", "evaluate_swarm", "SwarmPerformance", "run_backtests"]


def run_backtests() -> None:
    """Placeholder entrypoint for batch backtest execution."""
    # Future implementations can load historical snapshots and realized returns
    # to populate ``evaluate_swarm``. For now we avoid raising to keep the
    # worker callable.
    return None
