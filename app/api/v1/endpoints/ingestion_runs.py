from fastapi import APIRouter, Query, Response
from fastapi.encoders import jsonable_encoder
from sqlalchemy import func, select

from app.db.models.ingestion_run import IngestionRun
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.db_session import DbSessionDep
from app.services.ingestion_history import run_view

router = APIRouter(dependencies=[VerifyApiKeyDep])


@router.get('/ingestion-runs', summary='История обработки документа для сквозной диагностики')
async def ingestion_runs(
    session: DbSessionDep,
    response: Response,
    document_id: str = Query(min_length=1, max_length=255),
    request_id: str | None = Query(None, max_length=255),
    page: int = Query(1, ge=1, le=1_000_000),
    page_size: int = Query(20, ge=1, le=50),
):
    conditions = [IngestionRun.document_id == document_id]
    if request_id:
        conditions.append(IngestionRun.request_id == request_id)
    total = await session.scalar(select(func.count()).select_from(IngestionRun).where(*conditions))
    rows = (await session.scalars(select(IngestionRun).where(*conditions).order_by(
        IngestionRun.started_at.desc(), IngestionRun.id.desc(),
    ).offset((page - 1) * page_size).limit(page_size))).all()
    response.headers['Cache-Control'] = 'no-store'
    return jsonable_encoder({'total': total, 'page': page, 'page_size': page_size, 'items': [run_view(row) for row in rows]})
