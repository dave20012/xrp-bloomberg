from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Flow(BaseModel):
    exchange: str
    direction: str
    volume: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


class OpenInterest(BaseModel):
    symbol: str = "XRPUSDT"
    value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Score(BaseModel):
    composite: float
    flow_pressure: float
    leverage_regime: float
    accumulation: float
    manipulation: float
    anomaly: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NewsItem(BaseModel):
    headline: str
    source: str
    url: str
    tag: str
    summary: str
    published_at: datetime = Field(default_factory=datetime.utcnow)


class DashboardSnapshot(BaseModel):
    flows: List[Flow]
    scores: Optional[Score]
    news: List[NewsItem]
