import argparse
import time
from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from core.binance_client import BinanceClient
from core.cc_client import CryptoCompareClient
from core.db import FlowRecord, OHLCVRecord, OpenInterestRecord, create_tables, SessionLocal
from core.redis_client import cache_snapshot


class InflowWorker:
    def __init__(self) -> None:
        self.binance = BinanceClient()
        self.cc = CryptoCompareClient()

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            trades = self.binance.fetch_agg_trades()
            klines = self.cc.fetch_ohlcv()
            oi = self.binance.fetch_futures_open_interest()

            flows_data = self._persist_flows(db, trades)
            ohlcv_data = self._persist_ohlcv(db, klines)
            oi_value = self._persist_open_interest(db, oi)

            snapshot = {
                "flows": flows_data,
                "ohlcv": ohlcv_data[-50:],
                "open_interest": oi_value,
                "timestamp": datetime.utcnow().isoformat(),
            }
            cache_snapshot("flows:latest", snapshot)
        finally:
            db.close()

    def _persist_flows(self, db: Session, trades: List[dict]) -> List[dict]:
        flows = []
        for trade in trades:
            direction = "inflow" if trade.get("is_buyer_maker") else "outflow"
            record = FlowRecord(
                exchange="binance",
                direction=direction,
                volume=trade.get("quantity", 0),
                price=trade.get("price", 0),
                timestamp=trade.get("timestamp", datetime.utcnow()),
            )
            db.add(record)
            flows.append(
                {
                    "exchange": record.exchange,
                    "direction": record.direction,
                    "volume": record.volume,
                    "price": record.price,
                    "timestamp": record.timestamp.isoformat(),
                }
            )
        db.commit()
        return flows[-200:]

    def _persist_ohlcv(self, db: Session, klines: List[dict]) -> List[dict]:
        candles = []
        for candle in klines:
            record = OHLCVRecord(
                open=candle.get("open", 0),
                high=candle.get("high", 0),
                low=candle.get("low", 0),
                close=candle.get("close", 0),
                volume=candle.get("volume", 0),
                timestamp=candle.get("time") or candle.get("open_time") or datetime.utcnow(),
            )
            db.add(record)
            candles.append(
                {
                    "open": record.open,
                    "high": record.high,
                    "low": record.low,
                    "close": record.close,
                    "volume": record.volume,
                    "timestamp": record.timestamp.isoformat(),
                }
            )
        db.commit()
        return candles[-200:]

    def _persist_open_interest(self, db: Session, oi: dict) -> dict:
        record = OpenInterestRecord(
            symbol=oi.get("symbol", "XRPUSDT"),
            value=oi.get("openInterest", 0),
            timestamp=oi.get("timestamp", datetime.utcnow()),
        )
        db.add(record)
        db.commit()
        return {"symbol": record.symbol, "value": record.value, "timestamp": record.timestamp.isoformat()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inflow worker")
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
