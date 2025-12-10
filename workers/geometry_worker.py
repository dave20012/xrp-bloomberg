import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import GeometrySnapshotRecord, MarketStateSnapshot, SessionLocal, create_tables
from core.geometry import GeometryModel
from core.redis_client import cache_snapshot
from core.state_space import N_COMPONENTS, N_FEATURES, build_state_vector


class GeometryWorker:
    """Projects market states into the geometry space and persists snapshots."""

    def __init__(self) -> None:
        self.model = GeometryModel(n_components=N_COMPONENTS, n_features=N_FEATURES)

    def _load_vector(self, raw_vector: Sequence[float]) -> Sequence[float] | None:
        try:
            return build_state_vector(raw_vector)
        except ValueError:
            return None

    def _fetch_state_history(self, db: Session, limit: int = 500) -> List[Sequence[float]]:
        rows = (
            db.query(MarketStateSnapshot)
            .order_by(MarketStateSnapshot.timestamp.desc())
            .limit(limit)
            .all()
        )
        history: List[Sequence[float]] = []
        for row in reversed(rows):
            raw_vector = json.loads(row.state_vector) if row.state_vector else []
            vector = self._load_vector(raw_vector)
            if vector is not None:
                history.append(vector)
        return history

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            history = self._fetch_state_history(db)
            if not history:
                cache_snapshot("geometry:latest", {})
                return

            current_state = history[-1] if history and len(history[-1]) == N_FEATURES else None
            if current_state is None:
                cache_snapshot("geometry:latest", {})
                return

            self.model.fit(history)
            snapshot = self.model.snapshot(current_state, history)

            record = GeometrySnapshotRecord(
                timestamp=datetime.now(timezone.utc),
                coords=json.dumps(snapshot.coords),
                motif_id=snapshot.motif_id,
                transition_probs=json.dumps(snapshot.motif_transition_probs),
                local_vector=json.dumps(snapshot.local_vector),
            )
            db.add(record)
            db.commit()

            cache_snapshot("geometry:latest", snapshot.to_dict())
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
