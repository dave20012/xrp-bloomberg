import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import NewsRecord, SessionLocal, create_tables
from core.hf_client import HFClient
from core.news_client import NewsClient
from core.redis_client import cache_snapshot


class NewsWorker:
    def __init__(self) -> None:
        self.news_client = NewsClient()
        self.hf_client = HFClient()

    def run_once(self) -> None:
        create_tables()
        db: Session = SessionLocal()
        try:
            articles = self.news_client.fetch_headlines()
            enriched = []
            for article in articles:
                tag_result = self.hf_client.classify(article.get("headline", ""))
                tag = tag_result.get("label", "market")
                record = NewsRecord(
                    headline=article.get("headline", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    tag=tag,
                    summary=article.get("summary", ""),
                    published_at=article.get("published_at", datetime.now(timezone.utc)),
                )
                db.add(record)
                enriched.append(
                    {
                        "headline": record.headline,
                        "source": record.source,
                        "url": record.url,
                        "tag": record.tag,
                        "summary": record.summary,
                        "published_at": record.published_at.isoformat(),
                    }
                )
            db.commit()
            cache_snapshot("news:latest", enriched)
        finally:
            db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News worker")
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
