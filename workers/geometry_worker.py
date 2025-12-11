import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime, timezone
from typing import List, Sequence

from sqlalchemy.orm import Session

from core.db import GeometrySnapshotRecord, MarketStateSnapshot, SessionLocal, create_tables
from core.geometry import GeometryModel
from core.redis_client import cache_snapshot
from core.state_space import N_FEATURES, build_state_vector


class GeometryWorker:
    """Projects market states into the geometry space and persists snapshots."""

    def __init__(self) -> None:
        self.model = GeometryModel()

    def _load_vector(self, raw_vector: Sequence[float]) -> Sequence[float] | None:
        try:
            vector = build_state_vector(raw_vector)
        except ValueError:
            return None
        if vector.shape != (N_FEATURES,):
            return None
        return vector

    def _fetch_state_history(self, db: Session, limit: int = 500) -> List[Sequence[float]]:
        rows = (
            db.query(MarketStateSnapshot)
            .order_by(MarketStateSnapshot.timestamp.desc())
            .limit(limit)
            .all()
        )
        raw_history: List[Sequence[float]] = []
        for row in reversed(rows):
            raw_vector = json.loads(row.state_vector) if row.state_vector else []
            raw_history.append(raw_vector)

        history: List[Sequence[float]] = []
        for raw in raw_history:
            vector = self._load_vector(raw)
            if vector is not None:
                history.append(vector)
        return history

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            history = self._fetch_state_history(db)
            current_state = history[-1] if history else build_state_vector([0.0] * N_FEATURES)

            self.model.fit(history)
            snapshot = self.model.snapshot(current_state)

            record = GeometrySnapshotRecord(
                timestamp=datetime.now(timezone.utc),
                coords=json.dumps(snapshot.coords.tolist()),
                motif_id=snapshot.motif_id,
                transition_probs=json.dumps(snapshot.motif_transition_probs),
                local_vector=json.dumps(snapshot.local_drift.tolist()),
            )
            db.add(record)
            db.commit()

            cache_snapshot(
                "geometry:latest",
                {
                    "coords": snapshot.coords.tolist(),
                    "motif_id": snapshot.motif_id,
                    "motif_transition_probs": snapshot.motif_transition_probs,
                    "local_drift": snapshot.local_drift.tolist(),
                },
            )
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry projection worker")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=600, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = GeometryWorker()
    if args.loop:
        while True:
            worker.run_once()
            time.sleep(args.interval)
    else:
        worker.run_once()


if __name__ == "__main__":
    main()
