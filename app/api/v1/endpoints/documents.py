from fastapi import APIRouter, HTTPException, Request, status

from app.core.config_logger import logger
from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import DocumentsServiceDep, IngestionServiceDep
from app.models.schemas import DocumentDeletedResponse, SectionUpdateRequest, SectionUpdateResponse

router = APIRouter(dependencies=[VerifyApiKeyDep])


@router.delete(
    path='/document/{document_id}',
    status_code=status.HTTP_200_OK,
    summary='Удаление документа из базы знаний',
    description='Удаляет документ и все его чанки (все версии) из Qdrant.',
    operation_id='deleteDocument',
    response_description='Подтверждение удаления.',
    responses={
        200: {'description': 'Документ удалён (в том числе если его не было).', 'content': {'application/json': {'example': {'document_id': 'fz-181-art21'}}}},
    },
    response_model=DocumentDeletedResponse,
)
@limiter.limit('20/minute')
async def delete_document(request: Request, document_id: str, service: DocumentsServiceDep) -> DocumentDeletedResponse:
    """Удаляет документ и все его чанки из базы знаний.

    Args:
        request: HTTP-запрос (нужен `slowapi` для определения IP клиента, API-2/SEC-2).
        document_id: Идентификатор документа.
        service: Сервис управления документами.

    Returns:
        Подтверждение удаления.
    """
    logger.info('🗑️ Запрос DELETE /document/%s.', document_id)
    await service.delete_document(document_id)
    logger.info('✅ Запрос DELETE /document/%s выполнен.', document_id)
    return DocumentDeletedResponse(document_id=document_id)


@router.put(
    path='/document/{document_id}/sections/{section_number}',
    status_code=status.HTTP_200_OK,
    summary='Гранулярное обновление статьи/пункта нормативного акта',
    description=(
        'Заменяет текущую редакцию одной статьи или пункта (labor_code, federal_law, other_npa) '
        'без переиндексации всего документа. Старые чанки секции не удаляются — '
        'помечаются is_actual=False с effective_until для поддержки будущих запросов "на дату X".'
    ),
    operation_id='updateSection',
    response_model=SectionUpdateResponse,
)
@limiter.limit('10/minute')
async def update_section(
    request: Request,
    document_id: str,
    section_number: str,
    data: SectionUpdateRequest,
    service: IngestionServiceDep,
) -> SectionUpdateResponse:
    """Гранулярное обновление одной статьи/пункта нормативного акта (Этап 13 плана).

    Args:
        request: HTTP-запрос (нужен slowapi для rate limiting).
        document_id: Идентификатор документа.
        section_number: Номер статьи/пункта (например, "128").
        data: Тело запроса с текстом новой редакции и метаданными.
        service: Сервис ingestion.

    Returns:
        Сводка обновления.

    Raises:
        422: Если category не поддерживает гранулярное обновление.
    """
    logger.info(
        '📝 Запрос PUT /document/%s/sections/%s (category=%s, version=%s).',
        document_id, section_number, data.category, data.version,
    )
    try:
        result = await service.ingest_section(document_id, section_number, data)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    logger.info(
        '✅ Секция %s/%s обновлена: %d чанков, %d устарели.',
        document_id, section_number, result.chunks_count, result.superseded_chunks,
    )
    return result
