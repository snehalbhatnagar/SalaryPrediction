"""Stores each prediction so the app can show a history.

Uses SQLite by default. Point DATABASE_URL at MySQL to use that instead, e.g.
    DATABASE_URL=mysql+pymysql://user:password@localhost:3306/salary_prediction
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert, select

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, nullable=False),
    Column("country", String(80), nullable=False),
    Column("education", String(40), nullable=False),
    Column("years_pro", Float, nullable=False),
    Column("role", String(60), nullable=False),
    Column("remote", String(20), nullable=False),
    Column("predicted_usd", Integer, nullable=False),
    Column("low_usd", Integer, nullable=False),
    Column("high_usd", Integer, nullable=False),
)


def get_engine(url: str | None = None):
    engine = create_engine(url or os.getenv("DATABASE_URL", "sqlite:///predictions.db"))
    metadata.create_all(engine)
    return engine


def save_prediction(engine, inputs: dict, predicted: float, low: float, high: float) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(predictions).values(
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                **inputs,
                predicted_usd=round(predicted),
                low_usd=round(low),
                high_usd=round(high),
            )
        )


def recent_predictions(engine, limit: int = 20) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(select(predictions).order_by(predictions.c.id.desc()).limit(limit))
        return [dict(r._mapping) for r in rows]
