from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base

class Project(Base):
    __tablename__ = 'projects'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    frequency: Mapped[str] = mapped_column(String(20), default='D')
    years_of_history: Mapped[int] = mapped_column(Integer, default=5)
    grouped_forecast: Mapped[bool] = mapped_column(Boolean, default=False)
    forecast_scope: Mapped[str] = mapped_column(String(50), default='company_total')
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class Run(Base):
    __tablename__ = 'runs'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[str] = mapped_column(String(200), index=True)
    status: Mapped[str] = mapped_column(String(50), default='draft')
    target_column: Mapped[str | None] = mapped_column(String(100), nullable=True)
    date_column: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category_column: Mapped[str | None] = mapped_column(String(100), nullable=True)
    selected_models: Mapped[str | None] = mapped_column(Text, nullable=True)
    scoring_metric: Mapped[str | None] = mapped_column(String(50), nullable=True)
    forecast_horizon: Mapped[str | None] = mapped_column(String(50), nullable=True)
    output_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
