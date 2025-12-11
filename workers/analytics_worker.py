import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
from sqlalchemy.orm import Session

from core.db import ScoreRecord, SessionLocal, create_tables
from core.redis_client import cache_snapshot, get_snapshot
from core.signals import build_signals
from core.state_space import build_market_state, restore_market_state
from core.swarm import (
    ConnectomeGraph,
    SwarmAgent,
    SwarmAgentConfig,
    SwarmEnsemble,
    SwarmPredictor,
)


class AnalyticsWorker:
    def __init__(self) -> None:
        connectome = ConnectomeGraph.default()
        agents = self._default_agents()
        self.predictor = SwarmPredictor(connectome=connectome, ensemble=SwarmEnsemble(agents))

    def _default_agents(self) -> List[SwarmAgent]:
        configs = [
            SwarmAgentConfig(
                name="flow-anomaly", feature_subset=["net_flow", "flow_axis"], horizon="5m", target="flow"
            ),
            SwarmAgentConfig(
                name="derivatives-pressure",
                feature_subset=["open_interest", "leverage_axis", "pressure_axis"],
                horizon="1h",
                target="derivatives",
            ),
            SwarmAgentConfig(
                name="headline-heat", feature_subset=["headline_axis", "headline_risk"], horizon="4h", target="headline"
            ),
        ]
        coefficients: Dict[str, List[float]] = {
            "flow-anomaly": [0.6, 0.4],
            "derivatives-pressure": [0.3, 0.4, 0.3],
            "headline-heat": [0.5, 0.5],
        }
        intercepts = {"flow-anomaly": 0.0, "derivatives-pressure": 0.1, "headline-heat": -0.05}
        agents: List[SwarmAgent] = []
        for cfg in configs:
            agents.append(SwarmAgent(cfg, coefficients[cfg.name], intercept=intercepts[cfg.name]))
        return agents

    def _hydrate_state(self, flows_snapshot) -> object:
        cached_state = get_snapshot("state:latest") or {}
        state = restore_market_state(cached_state)
        if state:
            return state

        flows = flows_snapshot.get("flows", []) if flows_snapshot else []
        ohlcv = flows_snapshot.get("ohlcv", []) if flows_snapshot else []
        open_interest = flows_snapshot.get("open_interest") if flows_snapshot else {}
        volumes = [flow["volume"] for flow in flows if flow.get("direction") == "inflow"]
        raw_inputs: Dict[str, float] = {
            "spot_price": float(ohlcv[-1]["close"]) if ohlcv else 0.0,
            "returns": float(ohlcv[-1]["close"] - ohlcv[-2]["close"]) if len(ohlcv) >= 2 else 0.0,
            "realized_vol": float(open_interest.get("value", 0.0)) if open_interest else 0.0,
            "net_flow": float(sum(volumes)) if volumes else 0.0,
            "exchange_concentration": 0.0,
            "stablecoin_rotation": 0.0,
            "open_interest": float(open_interest.get("value", 0.0)) if open_interest else 0.0,
            "funding_skew": 0.0,
            "perp_basis": 0.0,
            "orderbook_imbalance": 0.0,
            "aggressive_volume": float(volumes[0]) if volumes else 0.0,
            "headline_risk": 0.0,
            "headline_count": 0.0,
            "headline_recency": 0.0,
        }
        return build_market_state(timestamp=datetime.utcnow(), raw_inputs=raw_inputs)

    def _geometry_context(self):
        snapshot = get_snapshot("geometry:latest") or {}
        coords = np.array(snapshot.get("coords", []), dtype=float) if snapshot else np.array([])
        motif = snapshot.get("motif_id") if snapshot else None
        return coords, motif

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            flows_snapshot = get_snapshot("flows:latest") or {}
            flows = flows_snapshot.get("flows", [])
            ohlcv = flows_snapshot.get("ohlcv", [])
            open_interest_data = flows_snapshot.get("open_interest", {})

            if not ohlcv:
                ohlcv = [
                    {
                        "close": 0.5,
                        "volume": 100000,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ]

            volumes = [candle.get("volume", 0) for candle in ohlcv]
            prices = [candle.get("close", 0) for candle in ohlcv]
            open_interest_value = float(open_interest_data.get("value", 0))

            funding_rates = [0.0001] * 5
            long_short = [1.05] * 5

            signals = build_signals(
                volumes=volumes,
                flows=flows,
                prices=prices,
                open_interest=open_interest_value,
                funding_rates=funding_rates,
                long_short_ratios=long_short,
            )

            record = ScoreRecord(
                composite=signals.composite,
                flow_pressure=signals.flow_pressure,
                leverage_regime=signals.leverage_regime,
                accumulation=signals.accumulation_score,
                manipulation=signals.manipulation_score,
                anomaly=signals.anomaly_z,
                timestamp=datetime.utcnow(),
            )
            db.add(record)
            db.commit()

            market_state = self._hydrate_state(flows_snapshot)
            coords, motif = self._geometry_context()
            swarm_snapshot = self.predictor.forecast(market_state, coords=coords, motif_id=motif)

            cache_snapshot(
                "scores:latest",
                {
                    "composite": signals.composite,
                    "flow_pressure": signals.flow_pressure,
                    "leverage_regime": signals.leverage_regime,
                    "accumulation": signals.accumulation_score,
                    "manipulation": signals.manipulation_score,
                    "anomaly": signals.anomaly_z,
                    "timestamp": record.timestamp.isoformat(),
                    "swarm": swarm_snapshot.to_dict(),
                },
            )
            cache_snapshot("swarm:latest", swarm_snapshot.to_dict())
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytics worker")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=600, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = AnalyticsWorker()
    if args.loop:
        while True:
            worker.run_once()
            time.sleep(args.interval)
    else:
        worker.run_once()


if __name__ == "__main__":
    main()
