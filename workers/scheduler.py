import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
import time

WORKERS = [
    "inflow_worker.py",
    "state_worker.py",
    "analytics_worker.py",
    "geometry_worker.py",
    "news_worker.py",
    "swarm_worker.py",
]


class Scheduler:
    def __init__(self, interval: int) -> None:
        self.interval = interval

    def run_once(self) -> None:
        for worker in WORKERS:
            subprocess.Popen(["python", f"workers/{worker}"])

    def run_loop(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker scheduler")
    parser.add_argument("--interval", type=int, default=600, help="Interval between launches")
    parser.add_argument("--loop", action="store_true", help="Run scheduler in a loop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler = Scheduler(interval=args.interval)
    if args.loop:
        scheduler.run_loop()
    else:
        scheduler.run_once()


if __name__ == "__main__":
    main()
