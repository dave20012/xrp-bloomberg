import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from core.db import NewsRecord, SessionLocal, create_tables
from core.hf_client import HFClient
from core.news_client import NewsClient
from core.redis_client import cache_snapshot


class NewsWorker:
    def __init__(self) -> None:
        self.news_client = NewsClient()
        self.hf_client = HFClient()

    def _fetch_news(self):
        return self.news_client.fetch_recent_news()

    def _summarize(self, headlines):
        return [self.hf_client.summarize(item["headline"]) for item in headlines]

    def _save_news(self, db: Session, headlines, summaries):
        for item, summary in zip(headlines, summaries):
            record = NewsRecord(
                headline=item["headline"],
                source=item["source"],
                published_at=datetime.fromtimestamp(item["timestamp"], tz=timezone.utc),
                tag=item.get("tag", ""),
                summary=summary,
            )
            db.add(record)

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            headlines = self._fetch_news()
            summaries = self._summarize(headlines)

            self._save_news(db, headlines, summaries)
            db.commit()

            cache_snapshot(
                "news:latest",
                [
                    {
                        "headline": h["headline"],
                        "source": h["source"],
                        "published_at": datetime.fromtimestamp(h["timestamp"], tz=timezone.utc).isoformat(),
                        "tag": h.get("tag", ""),
                        "summary": s,
                    }
                    for h, s in zip(headlines, summaries)
                ],
            )
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News ingestion worker")
    parser.add_argument("--loop", action="store_true", help="Loop execution")
    parser.add_argument("--interval", type=int, default=1800, help="Loop interval seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = NewsWorker()
    if args.loop:
        while True:
            worker.run_once()
            time.sleep(args.interval)
    else:
        worker.run_once()


if __name__ == "__main__":
    main()
