import argparse
import time
from datetime import datetime

from sqlalchemy.orm import Session

from core.db import OHLCVRecord, OpenInterestRecord, ScoreRecord, SessionLocal, create_tables
from core.redis_client import cache_snapshot, get_snapshot
from core.signals import build_signals


class AnalyticsWorker:
    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            snapshot = get_snapshot("flows:latest") or {}
            flows = snapshot.get("flows", [])
            ohlcv = snapshot.get("ohlcv", [])
            open_interest_data = snapshot.get("open_interest", {})

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
                },
            )
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
