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
    admin_session_https_only: bool = False
    """Cookie сессии админки только по HTTPS (ADM-7,
    AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md). По умолчанию `False` —
    локальная разработка (`hypercorn` без TLS, `docker-compose` на
    `localhost`) иначе не смогла бы аутентифицироваться в браузере. В
    production за HTTPS-терминирующим reverse-proxy — обязательно `True`.
    """
    api_key: SecretStr
    """Единый ключ доступа к публичному REST API (`/search`, `/ingest`, `DELETE /document/{id}`).

    Один статический ключ, не таблица ключей в БД — единственный документированный
    потребитель сейчас — MCP Tools Server (раздел 5 RAG_SERVICE_PLAN.md). См.
    AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md, ARCH-1/API-1/SEC-1.
    """


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

    # v2-модели поддерживают 128/256(default)/512/768 через параметр `dim`
    # в запросе к Text Embedding API. Раньше клиент ошибочно слал это поле
    # как `vectorDimension` (несуществующее имя, молча игнорировалось API,
    # реально всегда возвращались дефолтные 256 независимо от значения
    # здесь) — обнаружено и исправлено 2026-07-08 (`app/clients/embeddings.py`),
    # подтверждено прямыми вызовами API на всех четырёх значениях.
    yandex_embedding_dim: int = 768

    @property
    def llm_model_uri(self) -> str:
        return f'gpt://{self.yandex_folder_id}/{self.yandex_llm_model}'

    @property
    def embedding_doc_model_uri(self) -> str:
        return f'emb://{self.yandex_folder_id}/{self.yandex_embedding_doc_model}/latest'

    @property
    def embedding_query_model_uri(self) -> str:
        return f'emb://{self.yandex_folder_id}/{self.yandex_embedding_query_model}/latest'


class SearchSettings(SettingsBase):
    """Настройки гибридного поиска (Этап 5/5.1)."""

    dense_top_k_per_category: int = 4
    sparse_top_k_per_category: int = 4
    question_dense_top_k_per_category: int = 2
    question_dense_top_k: int = 10
    """Top-K на каждую category при категорийно-сбалансированном поиске
    (SEARCH-2, AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — выбраны как
    компромисс без эмпирической проверки на реальном корпусе (раздел 5.1
    плана). Вынесены в `Settings`, чтобы их можно было менять и замерять
    без редеплоя кода, когда появится реальный корпус и набор "важных"
    вопросов для recall@k замера.

    `question_dense_*` — отдельные лимиты для dense-lanes по гипотетическим
    вопросам (`question_0..4`). Эти векторы повышают recall по бытовым
    формулировкам запроса, но не должны раздувать candidate pool настолько
    же сильно, как основной `chunk`-вектор.
    """


class PolzaSettings(SettingsBase):
    """Настройки доступа к Polza AI — OpenAI-совместимый провайдер-агрегатор,
    используется для reranker'а (Этап 6) и обогащения чанков (Этап 3,
    перенесено с Yandex 2026-06-21 — см. `app/dependencies/clients.py`).

    Модели заданы отдельными настройками (раздел 8 плана) — reranking
    (hot path поиска) и обогащение (offline ingestion) имеют разные
    требования к latency и могут со временем разойтись по модели; единая
    настройка `polza_llm_model` не позволяла это выразить.
    """

    polza_api_key: SecretStr = SecretStr('PLACEHOLDER_POLZA_API_KEY')
    # Выбор модели (Lite vs Pro vs другие кандидаты) — открытый вопрос,
    # раздел 8 плана: `google/gemini-3.1-flash` (без `-lite`) не существует
    # в каталоге Polza, временно возвращено рабочее значение до
    # сравнительного теста на наших данных.
    polza_enrichment_llm_model: str = 'google/gemini-3.1-flash-lite-preview'
    polza_reranker_llm_model: str = 'google/gemini-3.1-flash-lite-preview'
    polza_query_expansion_llm_model: str = 'google/gemini-3.1-flash-lite-preview'
    polza_enrichment_timeout_seconds: int = 90
    polza_enrichment_retries: int = 3
    polza_query_expansion_timeout_seconds: int = 12
    polza_query_expansion_retries: int = 1
    polza_reranker_timeout_seconds: int = 12
    polza_reranker_retries: int = 1
    """Модель расширения запроса (декомпозиция + переформулировка,
    раздел 8 плана) — отдельная настройка, как и у reranker'а/обогащения:
    в hot path поиска (как reranker), но задача проще (1 короткий запрос
    пользователя, не кандидаты-чанки) — может со временем разойтись по
    модели с reranker'ом.

    Timeout/retry тоже разделены по use-case: enrichment — offline ingestion
    и может ждать дольше; query expansion/reranker — hot path `/search`,
    поэтому при деградации LLM должны быстро уйти в fallback.
    """
    polza_llm_api_url: str = 'https://polza.ai/api/v1/chat/completions'


class ObservabilitySettings(SettingsBase):
    """Минимальный distributed trace поиска без экспорта query/chunk content."""

    phoenix_enabled: bool = True
    phoenix_otlp_endpoint: str = 'http://localhost:6006/v1/traces'
    phoenix_project_name: str = 'vera-local'


class Settings(BaseSettings):
    """Агрегатор всех доменных настроек проекта."""

    app: AppSettings = Field(default_factory=AppSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    yandex: YandexSettings = Field(default_factory=YandexSettings)
    polza: PolzaSettings = Field(default_factory=PolzaSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
