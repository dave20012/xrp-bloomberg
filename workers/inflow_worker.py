import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from core.binance_client import BinanceClient
from core.cc_client import CryptoCompareClient
from core.db import FlowRecord, OHLCVRecord, OpenInterestRecord, SessionLocal, create_tables
from core.redis_client import cache_snapshot


class InflowWorker:
    def __init__(self) -> None:
        self.binance = BinanceClient()
        self.cc = CryptoCompareClient()

    def _fetch_flows(self):
        return self.binance.get_recent_flows()

    def _fetch_ohlcv(self):
        return self.cc.get_recent_ohlcv()

    def _fetch_open_interest(self):
        return self.binance.get_open_interest()

    def _save_flows(self, db: Session, flows):
        for flow in flows:
            record = FlowRecord(
                volume=flow["volume"],
                direction=flow["direction"],
                timestamp=datetime.fromtimestamp(flow["timestamp"], tz=timezone.utc),
            )
            db.add(record)

    def _save_ohlcv(self, db: Session, ohlcv):
        for candle in ohlcv:
            record = OHLCVRecord(
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
                timestamp=datetime.fromtimestamp(candle["timestamp"], tz=timezone.utc),
            )
            db.add(record)

    def _save_open_interest(self, db: Session, open_interest):
        record = OpenInterestRecord(value=open_interest, timestamp=datetime.now(timezone.utc))
        db.add(record)

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            flows = self._fetch_flows()
            ohlcv = self._fetch_ohlcv()
            open_interest = self._fetch_open_interest()

            self._save_flows(db, flows)
            self._save_ohlcv(db, ohlcv)
            self._save_open_interest(db, open_interest)

            db.commit()

            cache_snapshot(
                "flows:latest",
                {
                    "flows": flows,
                    "ohlcv": ohlcv,
                    "open_interest": {"value": open_interest, "timestamp": datetime.now(timezone.utc).isoformat()},
                },
            )
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exchange inflow/outflow worker")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = InflowWorker()
    if args.loop:
        while True:
            worker.run_once()
            time.sleep(args.interval)
    else:
        worker.run_once()


if __name__ == "__main__":
    main()
