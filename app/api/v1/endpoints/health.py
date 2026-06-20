from fastapi import APIRouter, HTTPException, status

from app.core.config_logger import logger
from app.dependencies.services import HealthServiceDep
from app.exceptions.health import DatabaseUnavailableError
from app.schemas.health import HealthSchema

router = APIRouter()


@router.get(
    path='/health',
    status_code=status.HTTP_200_OK,
    summary='Статус сервиса',
    description='Проверяет доступность сервиса и подключения к Postgres.',
    operation_id='getHealth',
    response_description='Агрегированный статус сервиса и его зависимостей.',
    responses={
        200: {'description': 'Сервис доступен.', 'content': {'application/json': {'example': {'status': 'ok', 'database': 'ok'}}}},
        503: {'description': 'Критичная зависимость недоступна.', 'content': {'application/json': {'example': {'detail': 'База данных недоступна.'}}}},
    },
    response_model=HealthSchema,
)
async def get_health(service: HealthServiceDep) -> HealthSchema:
    """Возвращает статус сервиса.

    Args:
        service: Сервис проверки здоровья.

    Returns:
        Агрегированный статус сервиса и его зависимостей.
    """
    logger.info('🚀 Запрос GET /health.')
    try:
        result = await service.check_health()
        logger.info('✅ Запрос GET /health выполнен.')
        return result
    except DatabaseUnavailableError as error:
        logger.exception('❌ Сервис недоступен. Детали: %s', error)
        raise HTTPException(status_code=error.status_code, detail=error.detail)
