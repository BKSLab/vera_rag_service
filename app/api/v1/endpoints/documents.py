from fastapi import APIRouter, Request, status

from app.core.config_logger import logger
from app.core.rate_limit import limiter
from app.dependencies.auth import VerifyApiKeyDep
from app.dependencies.services import DocumentsServiceDep
from app.models.schemas import DocumentDeletedResponse

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
