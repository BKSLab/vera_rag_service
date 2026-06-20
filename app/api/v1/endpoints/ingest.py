from fastapi import APIRouter, HTTPException, status

from app.core.config_logger import logger
from app.dependencies.services import IngestionServiceDep
from app.exceptions.embedding import EmbeddingApiRequestError
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import DocumentMetadataInput, IngestRequest, IngestResponse

router = APIRouter()


@router.post(
    path='/ingest',
    status_code=status.HTTP_200_OK,
    summary='Индексация документа в базе знаний',
    description=(
        'Запускает ingestion-пайплайн (Этапы 1–4 плана) для одного документа. '
        'Если документ уже проиндексирован под другой версией — новая версия '
        'upsert\'ится первой, старая удаляется только после успешного upsert '
        '(раздел 3 плана, явный workflow обновления документа).'
    ),
    operation_id='ingestDocument',
    response_description='Сводка ingestion: количество чанков и замещённые версии.',
    responses={
        200: {
            'description': 'Документ проиндексирован.',
            'content': {'application/json': {'example': {'document_id': 'fz-181-art21', 'version': '2026-01-01', 'chunks_count': 3, 'replaced_versions': []}}},
        },
        422: {'description': 'Невалидный source_type или пустой текст.'},
        500: {
            'description': 'Ошибка при запросе к LLM-обогащению или Embedding API.',
            'content': {'application/json': {'example': {'detail': 'Ошибка при запросе к LLM API. Подробности: ...'}}},
        },
    },
    response_model=IngestResponse,
)
async def ingest_document(data: IngestRequest, service: IngestionServiceDep) -> IngestResponse:
    """Индексирует документ в базе знаний.

    Args:
        data: Документ и его метаданные.
        service: Сервис ingestion.

    Returns:
        Сводка ingestion.
    """
    logger.info('🚀 Запрос POST /ingest. document_id=%s.', data.document_id)
    try:
        document_metadata = DocumentMetadataInput(
            source_title=data.source_title,
            audience=data.audience,
            topic=data.topic,
            version=data.version,
            effective_date=data.effective_date,
        )
        result = await service.ingest_document(
            document_id=data.document_id,
            raw_text=data.raw_text,
            source_type=data.source_type,
            document_metadata=document_metadata,
        )
        logger.info('✅ Запрос POST /ingest выполнен. document_id=%s.', data.document_id)
        return result
    except (LlmApiRequestError, EmbeddingApiRequestError) as error:
        logger.exception('❌ Ошибка ingestion. document_id=%s. Детали: %s', data.document_id, error)
        raise HTTPException(status_code=error.status_code, detail=error.detail)
