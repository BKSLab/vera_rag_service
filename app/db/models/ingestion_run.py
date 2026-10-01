from datetime import date, datetime

from sqlalchemy import Date, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base


class IngestionRun(Base):
    """Независимый от реестра и коллекции журнал одной попытки индексации."""

    __tablename__ = 'ingestion_runs'
    __table_args__ = (
        Index('ix_ingestion_runs_document_started', 'document_id', 'started_at'),
        Index('ix_ingestion_runs_status_started', 'status', 'started_at'),
        Index('ix_ingestion_runs_request_id', 'request_id'),
        Index('ix_ingestion_runs_change_id', 'change_id'),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(Text)
    source_title: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(20))
    version: Mapped[str] = mapped_column(Text)
    revision_date: Mapped[date] = mapped_column(Date)
    section_number: Mapped[str | None] = mapped_column(Text)
    collection_name: Mapped[str] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    change_id: Mapped[int | None]
    status: Mapped[str] = mapped_column(String(20))
    current_stage: Mapped[str | None] = mapped_column(String(40))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stages: Mapped[list] = mapped_column(JSONB)
    result: Mapped[dict] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
