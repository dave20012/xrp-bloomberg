import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict

from sqlalchemy.orm import Session

from core.db import MarketStateSnapshot, SessionLocal, create_tables
from core.redis_client import cache_snapshot, get_snapshot
from core.state_space import build_market_state


class StateWorker:
    def _load_flows(self, db: Session, limit: int = 100):
        rows = db.query(MarketStateSnapshot).order_by(MarketStateSnapshot.timestamp.desc()).limit(limit)
        return rows

    def _load_signals(self):
        return get_snapshot("scores:latest")

    def _load_news(self):
        return get_snapshot("news:latest")

    def _load_flows_snapshot(self):
        return get_snapshot("flows:latest")

    def _raw_inputs_from_sources(self, db: Session) -> Dict[str, float]:
        flows_snapshot = self._load_flows_snapshot() or {}
        flows = flows_snapshot.get("flows", [])
        ohlcv = flows_snapshot.get("ohlcv", [])
        open_interest = flows_snapshot.get("open_interest")

        signals = self._load_signals() or {}
        headlines = self._load_news() or []
        headline_count = len(headlines)

        volumes = [flow["volume"] for flow in flows if flow.get("direction") == "inflow"]
        raw_inputs: Dict[str, float] = {
            "spot_price": float(ohlcv[-1]["close"]) if ohlcv else 0.0,
            "returns": float(ohlcv[-1]["close"] - ohlcv[-2]["close"]) if len(ohlcv) >= 2 else 0.0,
            "realized_vol": float(open_interest.get("value", 0.0)) if open_interest else 0.0,
            "net_flow": float(sum(volumes)) if volumes else 0.0,
            "exchange_concentration": float(signals.get("flow_pressure", 0.0)),
            "stablecoin_rotation": float(signals.get("accumulation", 0.0)),
            "open_interest": float(open_interest.get("value", 0.0)) if open_interest else 0.0,
            "funding_skew": 0.0,
            "perp_basis": 0.0,
            "orderbook_imbalance": 0.0,
            "aggressive_volume": float(volumes[0]) if volumes else 0.0,
            "headline_risk": float(signals.get("anomaly", 0.0)) if signals else 0.0,
            "headline_count": headline_count,
            "headline_recency": 1.0 if headline_count else 0.0,
        }
        return raw_inputs

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            raw_inputs = self._raw_inputs_from_sources(db)
            now = datetime.now(timezone.utc)
            state = build_market_state(timestamp=now, raw_inputs=raw_inputs)

            record = MarketStateSnapshot(
                timestamp=now,
                state_vector=json.dumps(state.vector.tolist()),
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
