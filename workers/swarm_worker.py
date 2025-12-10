import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import GeometrySnapshotRecord, MarketStateSnapshot, SessionLocal, SwarmSnapshotRecord, create_tables
from core.redis_client import cache_snapshot
from core.state_space import MarketState
from core.swarm import SwarmAgent, SwarmAgentConfig, SwarmEnsemble


class SwarmWorker:
    """Runs swarm agents against the latest state and stores aggregated votes."""

    def __init__(self) -> None:
        self.ensemble = SwarmEnsemble(self._bootstrap_agents())

    def _bootstrap_agents(self) -> List[SwarmAgent]:
        # These defaults encode the "many cheap scouts" idea without depending on an
        # external model registry. Coefficients are hand-tuned placeholders.
        return [
            SwarmAgent(
                config=SwarmAgentConfig(
                    name="flow_plus_price",
                    feature_subset=["net_flow", "returns", "flow_axis"],
                    horizon="5m",
                    target="direction",
                    threshold=0.2,
                ),
                coefficients=[0.4, 0.6, 0.3],
                intercept=0.0,
            ),
            SwarmAgent(
                config=SwarmAgentConfig(
                    name="leverage_balance",
                    feature_subset=["open_interest", "leverage_axis"],
                    horizon="1h",
                    target="direction",
                    threshold=0.15,
                ),
                coefficients=[0.5, 0.5],
                intercept=-0.05,
            ),
            SwarmAgent(
                config=SwarmAgentConfig(
                    name="headline_tension",
                    feature_subset=["headline_axis", "realized_vol"],
                    horizon="4h",
                    target="tail_event",
                    threshold=0.25,
                    direction_labels=("EVENT_YES", "EVENT_NO"),
                ),
                coefficients=[0.7, 0.2],
                intercept=0.0,
            ),
        ]

    def _latest_state(self, db: Session) -> MarketState | None:
        row = db.query(MarketStateSnapshot).order_by(MarketStateSnapshot.timestamp.desc()).first()
        if not row:
            return None
        vector = json.loads(row.state_vector) if row.state_vector else []
        composite = json.loads(row.composite_axes) if row.composite_axes else {}
        return MarketState(
            timestamp=row.timestamp,
            raw_features={},
            normalized_features={},
            composite_axes=composite,
            vector=[float(v) for v in vector],
        )

    def _latest_motif(self, db: Session) -> str | None:
        row = db.query(GeometrySnapshotRecord).order_by(GeometrySnapshotRecord.timestamp.desc()).first()
        return row.motif_id if row else None

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            state = self._latest_state(db)
            if not state:
                cache_snapshot("swarm:latest", {})
                return

            motif = self._latest_motif(db)
            snapshot = self.ensemble.predict(state, motif_id=motif)

            record = SwarmSnapshotRecord(
                timestamp=datetime.now(timezone.utc),
                motif_id=motif,
                per_horizon=json.dumps(snapshot.per_horizon),
                agent_breakdown=json.dumps(snapshot.agent_breakdown),
            )
            db.add(record)
            db.commit()

            cache_snapshot("swarm:latest", snapshot.to_dict())
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm aggregation worker")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=600, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = SwarmWorker()
    if args.loop:
        while True:
            worker.run_once()
            time.sleep(args.interval)
    else:
        worker.run_once()


if __name__ == "__main__":
    main()
