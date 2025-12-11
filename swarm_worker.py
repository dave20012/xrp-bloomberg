import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict

import numpy as np
from sqlalchemy.orm import Session

from core.db import (
    GeometrySnapshotRecord,
    MarketStateSnapshot,
    SessionLocal,
    SwarmSnapshotRecord,
    create_tables,
)
from core.redis_client import cache_snapshot
from core.state_space import MarketState
from core.swarm import SwarmAgent, SwarmAgentConfig, SwarmEnsemble


class SwarmWorker:
    def __init__(self) -> None:
        agent_config = SwarmAgentConfig()
        self.agents = [SwarmAgent(config=agent_config) for _ in range(3)]
        self.ensemble = SwarmEnsemble(self.agents)

    def _load_state(self, db: Session) -> MarketState | None:
        row = (
            db.query(MarketStateSnapshot)
            .order_by(MarketStateSnapshot.timestamp.desc())
            .limit(1)
            .first()
        )
        if row is None:
            return None
        try:
            vector = json.loads(row.state_vector)
            return MarketState(
                timestamp=row.timestamp,
                raw_features={},
                normalized_features={},
                composite_axes=json.loads(row.composite_axes),
                vector=np.array(vector),
            )
        except Exception:
            return None

    def _load_geometry(self, db: Session) -> Dict:
        row = (
            db.query(GeometrySnapshotRecord)
            .order_by(GeometrySnapshotRecord.timestamp.desc())
            .limit(1)
            .first()
        )
        if row is None:
            return {}
        try:
            return {
                "coords": json.loads(row.coords),
                "motif_id": row.motif_id,
                "transition_probs": json.loads(row.transition_probs),
                "local_vector": json.loads(row.local_vector),
            }
        except Exception:
            return {}

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            state = self._load_state(db)
            geometry = self._load_geometry(db)
            snapshot = self.ensemble.evaluate(state, geometry)
            record = SwarmSnapshotRecord(
                timestamp=datetime.now(timezone.utc),
                snapshot=json.dumps(snapshot.to_dict()),
            )
            db.add(record)
            db.commit()
            cache_snapshot("swarm:latest", snapshot.to_dict())
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm evaluation worker")
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
