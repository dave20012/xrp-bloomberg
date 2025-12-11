from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, inspect
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config import get_settings

settings = get_settings()


def _build_engine(url: str):
    try:
        return create_engine(url, future=True)
    except (NoSuchModuleError, ModuleNotFoundError):
        fallback = "sqlite:///./local.db"
        return create_engine(fallback, future=True)


engine = _build_engine(settings.database_url)
engine.inspect = inspect
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class FlowRecord(Base):
    __tablename__ = 'flows'

    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String(64), index=True)
    direction = Column(String(16))
    volume = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class OHLCVRecord(Base):
    __tablename__ = 'ohlcv'

    id = Column(Integer, primary_key=True, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class OpenInterestRecord(Base):
    __tablename__ = 'openinterest'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(32), index=True, default='XRPUSDT')
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ScoreRecord(Base):
    __tablename__ = 'scores'

    id = Column(Integer, primary_key=True, index=True)
    composite = Column(Float)
    flow_pressure = Column(Float)
    leverage_regime = Column(Float)
    accumulation = Column(Float)
    manipulation = Column(Float)
    anomaly = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class NewsRecord(Base):
    __tablename__ = 'news'

    id = Column(Integer, primary_key=True, index=True)
    headline = Column(String(256))
    source = Column(String(128))
    url = Column(Text)
    tag = Column(String(64))
    published_at = Column(DateTime, default=datetime.utcnow, index=True)
    summary = Column(Text)


class MarketStateSnapshot(Base):
    __tablename__ = 'market_state_snapshots'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    state_vector = Column(Text)
    composite_axes = Column(Text)


class GeometrySnapshotRecord(Base):
    __tablename__ = 'geometry_snapshots'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    coords = Column(Text)
    motif_id = Column(String(64), index=True)
    transition_probs = Column(Text)
    local_vector = Column(Text)


class SwarmSnapshotRecord(Base):
    __tablename__ = 'swarm_snapshots'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    motif_id = Column(String(64), index=True)
    per_horizon = Column(Text)
    agent_breakdown = Column(Text)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
