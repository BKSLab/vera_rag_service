from pydantic import BaseModel, Field


class HealthSchema(BaseModel):
    """Статус сервиса и его критичных зависимостей."""

    status: str = Field(..., description='Общий статус сервиса.', examples=['ok'])
    database: str = Field(..., description='Статус подключения к Postgres.', examples=['ok'])
