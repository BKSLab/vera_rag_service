from datetime import UTC, datetime, timedelta

from app.db.models.ingestion_run import IngestionRun
from app.services.ingestion_journal import STAGE_LABELS, STATUS_LABELS

STALE_AFTER = timedelta(seconds=60)


def run_view(run: IngestionRun) -> dict:
    """Общий снимок истории для админки и защищённого API Legal Sync."""
    values = {column.name: getattr(run, column.name) for column in IngestionRun.__table__.columns}
    now = datetime.now(UTC)
    values['display_status'] = 'stalled' if run.status == 'running' and now - run.updated_at > STALE_AFTER else run.status
    values['duration_seconds'] = round(((run.finished_at or now) - run.started_at).total_seconds(), 1)
    values['stages'] = [{**step, 'label': STAGE_LABELS.get(step['key'], step['key']),
                         'status_label': STATUS_LABELS.get(step['status'], step['status'])} for step in run.stages]
    return values
