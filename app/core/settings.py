from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsBase(BaseSettings):
    """Базовый класс для всех доменных настроек проекта."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )


class AppSettings(SettingsBase):
    """Общие настройки приложения."""

    app_name: str = 'vera_rag_service'
    logging_config_path: str = 'logging.ini'
    secret_key: SecretStr
    admin_login: str
    admin_password: SecretStr


class DBSettings(SettingsBase):
    """Настройки подключения к Postgres."""

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: SecretStr
    postgres_name: str

    @property
    def url_connect(self) -> str:
        return (
            f'postgresql+asyncpg://{self.postgres_user}:'
            f'{self.postgres_password.get_secret_value()}@'
            f'{self.postgres_host}:{self.postgres_port}/'
            f'{self.postgres_name}'
        )


class QdrantSettings(SettingsBase):
    """Настройки подключения к Qdrant."""

    qdrant_url: str = 'http://localhost:6333'
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = 'vera_kb'


class YandexSettings(SettingsBase):
    """Настройки доступа к Yandex Cloud API (embeddings + LLM-обогащение чанков).

    Значения — заглушки до получения реальных credentials от пользователя
    (раздел 0.1 RAG_SERVICE_PLAN.md). `yandex_llm_api_url` и формат заголовка
    авторизации (`Api-Key` для статичного API-ключа сервисного аккаунта,
    либо `Bearer` для IAM-токена) — требуют проверки на реальном ключе,
    сейчас выставлены по документации Yandex Cloud Foundation Models
    (OpenAI-совместимый Chat Completions gateway) без подтверждения вызовом.
    """

    yandex_api_key: SecretStr = SecretStr('PLACEHOLDER_YANDEX_API_KEY')
    yandex_folder_id: str = 'PLACEHOLDER_YANDEX_FOLDER_ID'
    yandex_embedding_doc_model: str = 'text-embeddings-v2-doc'
    yandex_embedding_query_model: str = 'text-embeddings-v2-query'
    yandex_llm_model: str = 'PLACEHOLDER_YANDEX_LLM_MODEL'
    yandex_llm_api_url: str = 'https://ai.api.cloud.yandex.net/v1/chat/completions'
    yandex_embedding_api_url: str = 'https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding'

    # Замерено реальным вызовом API на Этапе 4 (2026-06-19): и v2-модели,
    # и легаси text-search-doc отдают 256 — расходится с предположением
    # "максимум 768" из раздела 0.1 плана (см. примечание там же).
    yandex_embedding_dim: int = 256

    @property
    def llm_model_uri(self) -> str:
        return f'gpt://{self.yandex_folder_id}/{self.yandex_llm_model}'

    @property
    def embedding_doc_model_uri(self) -> str:
        return f'emb://{self.yandex_folder_id}/{self.yandex_embedding_doc_model}/latest'

    @property
    def embedding_query_model_uri(self) -> str:
        return f'emb://{self.yandex_folder_id}/{self.yandex_embedding_query_model}/latest'


class PolzaSettings(SettingsBase):
    """Настройки доступа к Polza AI — OpenAI-совместимый провайдер-агрегатор,
    используется только для reranker'а (Этап 6), а не для embeddings/enrichment —
    те идут через прямой Yandex Cloud API (раздел 0.1 плана: дороже у
    агрегатора при равном качестве). Для reranking важна не цена модели,
    а доступ к конкретной нейтральной модели (Gemini) без завязки на Yandex.
    """

    polza_api_key: SecretStr = SecretStr('PLACEHOLDER_POLZA_API_KEY')
    polza_llm_model: str = 'google/gemini-3.1-flash-lite-preview'
    polza_llm_api_url: str = 'https://polza.ai/api/v1/chat/completions'


class Settings(BaseSettings):
    """Агрегатор всех доменных настроек проекта."""

    app: AppSettings = Field(default_factory=AppSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    yandex: YandexSettings = Field(default_factory=YandexSettings)
    polza: PolzaSettings = Field(default_factory=PolzaSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
