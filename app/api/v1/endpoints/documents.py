from fastapi import APIRouter, status

from app.core.config_logger import logger
from app.dependencies.services import DocumentsServiceDep
from app.models.schemas import DocumentDeletedResponse

router = APIRouter()


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
async def delete_document(document_id: str, service: DocumentsServiceDep) -> DocumentDeletedResponse:
    """Удаляет документ и все его чанки из базы знаний.

    Args:
        document_id: Идентификатор документа.
        service: Сервис управления документами.

    Returns:
        Подтверждение удаления.
    """
    logger.info('🗑️ Запрос DELETE /document/%s.', document_id)
    await service.delete_document(document_id)
    logger.info('✅ Запрос DELETE /document/%s выполнен.', document_id)
    return DocumentDeletedResponse(document_id=document_id)
