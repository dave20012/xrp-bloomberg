import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import (
    FlowRecord,
    MarketStateSnapshot,
    OHLCVRecord,
    OpenInterestRecord,
    ScoreRecord,
    SessionLocal,
    create_tables,
)
from core.redis_client import cache_snapshot, get_snapshot
from core.state_space import build_market_state


class StateWorker:
    """Builds normalized market state vectors from persisted data."""

    def _latest_flows(self, db: Session) -> List[FlowRecord]:
        return (
            db.query(FlowRecord)
            .order_by(FlowRecord.timestamp.desc())
            .limit(200)
            .all()
        )

    def _latest_ohlcv(self, db: Session) -> List[OHLCVRecord]:
        return (
            db.query(OHLCVRecord)
            .order_by(OHLCVRecord.timestamp.desc())
            .limit(200)
            .all()
        )

    def _latest_open_interest(self, db: Session) -> OpenInterestRecord | None:
        return db.query(OpenInterestRecord).order_by(OpenInterestRecord.timestamp.desc()).first()

    def _latest_score(self, db: Session) -> ScoreRecord | None:
        return db.query(ScoreRecord).order_by(ScoreRecord.timestamp.desc()).first()

    def _raw_inputs_from_sources(self, db: Session) -> Dict[str, float]:
        flows = self._latest_flows(db)
        ohlcv = self._latest_ohlcv(db)
        open_interest = self._latest_open_interest(db)
        score = self._latest_score(db)

        price = ohlcv[0].close if ohlcv else 0.0
        prev_price = ohlcv[1].close if len(ohlcv) > 1 else price
        returns = (price - prev_price) / prev_price if prev_price else 0.0

        volumes = [c.volume for c in ohlcv[:20]] if ohlcv else [0.0]
        realized_vol = float((sum((v - sum(volumes) / len(volumes)) ** 2 for v in volumes) / len(volumes)) ** 0.5)

        inflow = sum(f.volume for f in flows if f.direction == "inflow")
        outflow = sum(f.volume for f in flows if f.direction == "outflow")
        net_flow = inflow - outflow

        headline_snapshot = get_snapshot("news:latest") or []
        headline_count = float(len(headline_snapshot))

        return {
            "spot_price": price,
            "returns": returns,
            "realized_vol": realized_vol,
            "net_flow": net_flow,
            "exchange_concentration": 0.0,
            "stablecoin_rotation": 0.0,
            "open_interest": float(open_interest.value) if open_interest else 0.0,
            "funding_skew": 0.0,
            "perp_basis": 0.0,
            "orderbook_imbalance": 0.0,
            "aggressive_volume": float(volumes[0]) if volumes else 0.0,
            "headline_risk": float(score.anomaly) if score else 0.0,
            "headline_count": headline_count,
            "headline_recency": 1.0 if headline_count else 0.0,
        }

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            raw_inputs = self._raw_inputs_from_sources(db)
            now = datetime.now(timezone.utc)
            state = build_market_state(timestamp=now, raw_inputs=raw_inputs)

            record = MarketStateSnapshot(
                timestamp=now,
                state_vector=json.dumps(state.vector),
                composite_axes=json.dumps(state.composite_axes),
            )
            db.add(record)
            db.commit()

            cache_snapshot("state:latest", state.to_dict())
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State construction worker")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = StateWorker()
    if args.loop:
        while True:
            worker.run_once()
            time.sleep(args.interval)
    else:
        worker.run_once()


if __name__ == "__main__":
    main()
