import argparse
import time

from workers.analytics_worker import AnalyticsWorker
from workers.inflow_worker import InflowWorker
from workers.news_worker import NewsWorker


def run_cycle() -> None:
    InflowWorker().run_once()
    AnalyticsWorker().run_once()
    NewsWorker().run_once()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheduler for XRP analytics")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=900, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.loop:
        while True:
            run_cycle()
            time.sleep(args.interval)
    else:
        run_cycle()


if __name__ == "__main__":
    main()
