# RAG Service — План реализации

> **Статус реализации** (обновляется по ходу разработки):
> - ✅ Скаффолд проекта — settings (`app/core/settings.py`, заглушки Yandex), логирование, `app/db/session.py` + Alembic (async), `app/main.py` с lifespan-проверкой Postgres, `GET /health`, `requirements.txt`, `docker-compose.yml` (+ Qdrant), `.env`/`.env.example`. Проверено: `/health` отвечает `{"status":"ok","database":"ok"}` на локальной БД.
> - ✅ Этап 1 — Препроцессинг документов (`app/ingestion/preprocess.py`, `app/models/schemas.py::Section`, 10 юнит-тестов зелёные).
> - ✅ Этап 2 — Иерархический чанкинг (`app/ingestion/chunking.py`, `app/models/schemas.py::Chunk`, оценка токенов по эвристике chars/4, overlap 100 токенов, сквозной `chunk_index` по документу, 9 юнит-тестов зелёные).
> - ✅ Этап 3 — Обогащение чанков LLM (`app/clients/llm.py` по образцу `LLM_CLIENT_REFERENCE.md`, `app/ingestion/enrichment.py`, промпт в `app/ingestion/prompts/enrichment.py`, схемы `ChunkEnrichmentResult`/`EnrichedChunk`). DI: `app/dependencies/{http_client,clients}.py`. **Проверено реальными вызовами к Yandex Cloud** (`https://ai.api.cloud.yandex.net/v1/chat/completions`, заголовок `Api-Key`, модель `yandexgpt/rc`) — 5/5 успешных enrichment-вызовов подряд после фикса. Найдено и исправлено на живом API: принудительный `response_format: json_object` у этой модели ломает генерацию списков строк (зацикленные числа, потеря поля) — убран; модель почти всегда оборачивает JSON в markdown code fence и иногда добавляет `_..._`-эмфазис внутрь JSON — обе обёртки снимаются в `LlmClient._extract_validated` перед валидацией схемой (закреплено регрессионными тестами). Тесты: клиент — `httpx.MockTransport` (retry/backoff/невалидная схема/code fence/эмфазис), enrichment — `AsyncMock(spec=LlmClient)`. 31 тест зелёный.
> - ⚠️ **Найдено на Этапе 4, важно для раздела 0.1**: реальный замер показал размерность вектора Yandex Text Embeddings v2 (и легаси text-search-doc/query) — **256**, не 768. Заявленный в плане "максимум 768" не подтверждён действующим API на дату проверки (2026-06-19). Решение по умолчанию: используем фактические 256 (`YandexSettings.yandex_embedding_dim`), приоритет точности на тонких смысловых различиях из раздела 0.1 при этом не достигается в заявленном объёме — стоит обсудить с тем, кто фиксировал решение, не появилась ли отдельная опция увеличения размерности (в Yandex Cloud такое иногда доступно отдельным параметром запроса, не найдено в ответе API на момент проверки).
> - ✅ Этап 4 — Embedding и upsert в Qdrant (`app/clients/embeddings.py` — `EmbeddingClient` с retry/backoff, `app/embeddings/embedder.py`, `app/vectorstore/qdrant_client.py` — `QdrantVectorStore` с одной точкой на чанк и именованными векторами `chunk`+`question_0..4`, поддержка явного workflow обновления документа через `delete_document(document_id, version)`). DI: `app/dependencies/{clients,vectorstore}.py`, `app/main.py` lifespan создаёт коллекцию при старте. Подтверждено реальным сквозным прогоном: preprocess → chunk → enrich (LLM) → embed (Yandex) → upsert в Qdrant → count → delete — всё отработало на реальных API. 12 новых тестов (юнит на моках + 8 интеграционных на реальном локальном Qdrant из `docker-compose.yml`) зелёные, итого 43.
> - ✅ Этап 5 — Hybrid search (`app/search/hybrid.py` — `dense_search` через Qdrant `query_points` по named vector `chunk` с фильтром по metadata до векторного сравнения, `sparse_search` — BM25 (`rank_bm25`) в памяти над кандидатами после фильтра, без нативных sparse-векторов Qdrant — осознанное упрощение для небольшого корпуса, см. docstring; `app/search/fusion.py` — классический RRF по рангам, k=60). Payload точки Qdrant расширен (`app/vectorstore/qdrant_client.py::build_chunk_payload`) текстом/заголовком/вопросами — нужны для BM25 и для финальной выдачи (Этап 7). Бизнес-правило `audience`: фильтр по `seeker`/`employer` включает и сам `both` (раздел 3 плана). 12 новых тестов (6 юнит на fusion + 6 интеграционных на реальном Qdrant: ранжирование, фильтр аудитории, точное совпадение термина BM25, RRF-склейка) зелёные, итого 55.
> - ✅ Этап 6 — Reranker, реализован через LLM, а не cross-encoder (закрывает открытый вопрос раздела 0.1). Провайдер — **Polza AI** (`google/gemini-3.1-flash-lite-preview`, OpenAI-совместимый, `Bearer`-аутентификация), отдельно от Yandex (используется только для reranking, не для embeddings/enrichment — раздел 0.1 не менялся). `app/search/reranker.py::rerank_chunks` — кандидатам присваиваются номера (не chunk_id — длинный UUID в выводе LLM рискует быть искажён на символ), модель возвращает только реально релевантные номера (не обязательно ровно 5), деградация до исходного RRF-порядка при отказе LLM. DI: `get_reranker_llm_client` в `app/dependencies/clients.py`, `PolzaSettings`. **Проверено реальными вызовами** — 3/3 точно выделили единственный релевантный кандидат среди 5 нерелевантных по юридическому тексту; в отличие от YandexGPT (Этап 3), Gemini через Polza не потребовал доработки промпта под капризы JSON-вывода. 6 новых тестов (`AsyncMock(spec=LlmClient)`, включая fallback-сценарии) зелёные, итого 61.
> - ✅ Этап 7 — API (`app/api/v1/endpoints/{search,ingest,documents,health}.py`), роутеры подключены в `app/main.py` с префиксом `/api/v1`. `POST /search` и `POST /ingest` оборачивают `SearchService`/`IngestionService` (`app/services/{search,ingestion}.py` — оркестрация embed_query → hybrid → rerank и preprocess → chunk → enrich → embed → upsert поверх уже готовых компонентов Этапов 1–6), `DELETE /document/{id}` — тонкий `DocumentsService` над `QdrantVectorStore.delete_document`. `IngestionService.ingest_document` реализует явный workflow обновления документа из раздела 3/Этапа 7 плана: старые версии документа находятся до upsert новой и удаляются только после его успешного завершения. DI-фабрики — `app/dependencies/services.py` (`SearchServiceDep`/`IngestionServiceDep`/`DocumentsServiceDep`/`HealthServiceDep`). Pydantic-контракт — `app/models/schemas.py` (`SearchRequest/Response`, `IngestRequest/Response`, `DocumentDeletedResponse`). 12 новых API-тестов (`tests/api/endpoints/`, `app.dependency_overrides` + `httpx.ASGITransport`, по разделу 18 `FASTAPI_PATTERNS.md`: успешные сценарии + маппинг доменных исключений в 422/500/503) зелёные, итого 62.
> - ✅ Этап 8 — Персистентное логирование `/search` в Postgres. **OTel/Arize Phoenix исключены из рамок этого репозитория** (см. примечание ниже, в описании Этапа 8) — единственный механизм наблюдаемости здесь: структурные логи (`logging.ini`) + таблица `search_logs` (журнал событий, без `updated_at` — `app/db/models/search_log.py::SearchLog`, миграция `app/db/alembic/versions/20260620_0807_add_search_logs_table.py`). `app/repositories/search_log.py::SearchLogRepository.save_search_log` — только запись. `app/search/hybrid.py` рефакторен: `hybrid_search()` теперь возвращает `HybridSearchResult` (dense/sparse/fused) вместо одного только fused-списка — нужно для лога; `get_candidate_chunk_ids()` оставлен как тонкая обёртка для обратной совместимости с уже существующими интеграционными тестами. `SearchService.search` (`app/services/search.py`) замеряет latency по стадиям (`embed_query`, `hybrid_search`, `rerank`) через `time.perf_counter()` и пишет запись лога после каждого запроса (включая случай пустого результата); отказ записи лога (`SearchLogRepositoryError`) перехватывается и логируется как предупреждение — не ронял поисковый запрос (деградация при частичном отказе, раздел 9 `FASTAPI_PATTERNS.md`). DI: `app/dependencies/repositories.py::SearchLogRepositoryDep` (первый файл уровня "репозиторий" в проекте). Тесты: 3 unit (`tests/unit/services/test_search_service.py`, моки Qdrant-клиента через `SimpleNamespace` + `AsyncMock(spec=...)`, включая сценарий отказа записи лога) + 1 интеграционный на реальном Postgres через `testcontainers` (`tests/integration/repositories/test_search_log_repository.py`, общие фикстуры — `tests/conftest.py`). Зелёные, итого 77. **Найдено при первом использовании `testcontainers`** (зависимость была в `requirements.txt`, но не использовалась): эталонная фикстура `engine` из раздела 18 `FASTAPI_PATTERNS.md` (session-scoped) ловит `RuntimeError: ...different loop` на pytest-asyncio 0.25 — у каждого теста свой event loop, session-scoped asyncpg-engine, созданный на одном loop, не работает на другом. Исправлено: `engine` сделан function-scoped (контейнер Postgres всё равно один на прогон — `postgres_container` остаётся session-scoped, пересоздаётся только лёгкий `AsyncEngine`).
> - ✅ Админка (`sqladmin`) — добавлена по явному запросу: дизайн, тёмная тема и подход взяты целиком из родственного сервиса `api_work_for_everyone` (общий продукт «Работа для всех», один и тот же `DESIGN_GUIDE.md`). `app/admin/{__init__,auth,views}.py` — `create_admin()` по образцу раздела 14 `FASTAPI_PATTERNS.md`, `AdminLoginAuth` — логин/пароль (`admin_login`/`admin_password` в `Settings`) как отдельная плоскость доступа, без БД (раздел 6 `FASTAPI_PATTERNS.md`, «Двухуровневая авторизация» — здесь нет обычных API-ключей, поэтому только один уровень). `SearchLogAdmin` — единственный `ModelView`, read-only (`can_create=False`, `can_edit=False`), с JSON pretty-print кандидатов/финального ответа и цветным бейджем `audience` в списке/деталях. Статика и шаблон (`app/static/admin-theme.css`, `app/templates/sqladmin/base.html`) скопированы из `api_work_for_everyone` без изменений — единый визуальный язык всех сервисов продукта. Новые настройки: `secret_key`/`admin_login`/`admin_password` в `AppSettings`, добавлены в `.env`/`.env.example`. Новые зависимости: `sqladmin==0.23.0`, `WTForms==3.1.2`, `itsdangerous==2.2.0`. **Проверено реальным запуском** (`hypercorn` локально, Postgres+Qdrant в Docker): `/admin/login` отдаёт форму, успешный логин по `admin_login`/`admin_password`, `/admin/search-log/list` отдаёт 200 с тёмной темой и корректным заголовком, статика `/static/admin-theme.css` раздаётся. Регрессии — полный набор тестов (77) зелёный после добавления.

> - ✅ Этап 5.1 — Категорийно-сбалансированный retrieval (`category` вместо `source_type` сквозь весь pipeline — `app/models/metadata.py`, `schemas.py`, `preprocess.py`, `chunking.py`, `qdrant_client.py`, `services/*`, API, `search_logs`, админка). `app/search/hybrid.py::hybrid_search` — если вызывающий не указал `category` явно, dense+sparse запускаются отдельно на каждую из 5 категорий (top-4+top-4, `DENSE_TOP_K_PER_CATEGORY`/`SPARSE_TOP_K_PER_CATEGORY`), каждая лента — отдельный список в RRF-фьюжне (`app/search/fusion.py`, без изменений — уже умел принимать произвольное число списков). `preprocess.py::_CATEGORY_TO_STRUCTURE` сопоставляет category со стратегией извлечения структуры (`labor_code`/`federal_law`/`other_npa`/`case_law` → парсер по "Статья N", `authorial` → markdown-заголовки); `extract_law_sections` получил fallback на случай отсутствия "Статья N" в тексте (раньше — молчаливая потеря всего документа). `SearchResultChunk.category` — новое поле в ответе `/search`, нужно потребителю (Agent Service) для порядка "база → практика → иные акты → комментарий". Новый интеграционный тест (`tests/integration/search/test_hybrid_search.py::test_hybrid_search_balances_candidates_across_categories`) воспроизводит сценарий риска (раздел 4): `case_law`-чанк, не попадающий в плоский top-20 из-за многочисленных `labor_code`-чанков с более высоким cosine-score, остаётся среди кандидатов при категорийно-сбалансированном поиске. 78 тестов зелёных (64 unit/API + 14 интеграционных, включая новый — проверено на реальном локальном Qdrant из `docker-compose.yml`).
> - ✅ Весь ранее не закоммиченный код (Этапы 1–8, админка) зафиксирован в git и запушен в `origin/main` 2026-06-20.
> - ✅ Этап 11 — Админка: загрузка документов (`/admin/document-upload`) + реестр `documents` в Postgres + интерактивное тестирование поиска (`/admin/search-test`). Подробности — раздел "Этап 11" ниже. Найден и исправлен баг авторизации sqladmin: `Admin.add_base_view()` напрямую (как в официальном примере библиотеки) не защищает страницу логином — нужен `Admin.add_view()`.
> - ✅ Этап 12 — Техническое ревью, независимая верификация и устранение всех 47 находок (2026-06-21). Подробности — раздел "Этап 12" ниже и сводная таблица в разделе 7. Закрыты все 5 production-блокеров (авторизация API, BM25/event-loop, идемпотентность ingestion, секреты в Docker-образе, stored XSS) + 42 находки Medium/Low. 143 теста зелёных (+ 2 нагрузочных, `pytest -m slow`), `ruff check .` чист, Docker-образ собран и проверен реально. Документы самого ревью (`REVIEW_AND_IMPROVEMENT_PLAN.md`, `AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md`, `task_audit.md`, `audit_and_implementation_prompt.md`) консолидированы в этот файл и удалены из репозитория — единый источник правды.

> Источник: исходные проектные документы AGENT_VERA_ARCHITECTURE (раздел "RAG Service — внутреннее устройство") и AGENT_VERA_WBS (п. 3.1, 3.4, 4.1, 5.1–5.6) — на момент составления плана находились в корне проекта, впоследствии удалены из репозитория.
> Период разработки по дорожной карте: Июнь Н3 — Август Н1 (Фаза 3), наполнение БЗ — Август Н2 (Фаза 4), тестирование — Август Н2–Н3 (Фаза 5).
> Роль: Backend. Контент для базы знаний поставляет Expert (нормативные акты, авторские статьи).

---

## 0. Назначение и рамки сервиса

RAG Service — самостоятельный, переиспользуемый сервис семантического поиска по базе знаний (нормативные акты + авторские статьи о правах людей с инвалидностью в сфере трудоустройства). Потребитель — MCP Tools Server (синхронный HTTP hot path через тул `kb_search`). Сервис не знает об Agent Service и пользователях напрямую.

**В рамках этого репозитория:**
- ingestion pipeline (препроцессинг → чанкинг → обогащение → embedding → upsert в Qdrant)
- search pipeline (hybrid search → RRF → reranker)
- HTTP API (`/search`, `/ingest`, `/document/{id}`, `/health`)
- персистентное логирование поисковых запросов (Postgres, для админ-аналитики качества поиска)
- юнит-тесты, Docker-образ

**Вне рамок:** UI, RabbitMQ, Agent Service, MCP Tools Server (только клиентский контракт), сбор/верификация нормативных текстов (готовит Expert).

---

## 0.1. Зафиксированные технические решения

| Решение | Выбор | Причина |
|---|---|---|
| Embedding-модель | **Yandex Text Embeddings v2** — пара `text-embeddings-v2-doc` (индексация) / `text-embeddings-v2-query` (поиск), облачный API | Нет ресурсов для self-hosted embedding-модели (CPU/RAM на сервере). Раздельные doc/query модели обучены совместно — корректное асимметричное кодирование. Низкая цена, поддержка русского, данные обрабатываются в юрисдикции РФ (важно для 152-ФЗ, т.к. тема — права людей с инвалидностью) |
| Размерность вектора | **768** (план) → **256 факт.** ⚠️ см. примечание в «Статус реализации» | Максимальная доступная в v2 — по документации на момент планирования. Реальный вызов API 2026-06-19 вернул 256 и для v2, и для легаси text-search-doc/query — расхождение не объяснено, требует уточнения у Yandex/повторной проверки документации |
| Способ доступа к Yandex API | **Прямой Yandex Cloud API** (Yandex AI Studio SDK / REST), без провайдера-агрегатора | Официальный тариф Yandex Cloud на эмбеддинги — 0,0101 ₽/1000 токенов (10,1 ₽/1М), у агрегатора та же модель стоила 13,01 ₽/1М — переплата ≈29% без дополнительной ценности. То же решение по умолчанию применяется к LLM для обогащения чанков (см. ниже) |
| LLM для обогащения чанков (Этап 3) | Yandex Cloud Model Gallery, конкретная модель не финализирована (кандидаты: `YandexGPT Pro 5.1` для качества или `YandexGPT Lite`/`gpt-oss-20b` для экономии) | Офлайн-шаг, не в hot path — приоритет качество понимания юридического текста, не скорость. Прямой Yandex Cloud API по той же причине, что и эмбеддинги |
| Reranker | **Решено** — LLM-reranker через Polza AI (`google/gemini-3.1-flash-lite-preview`), не cross-encoder | Избегает CPU/RAM-ресурсов self-hosted `bge-reranker-base` и подписки на Cohere/Voyage. Hot path — но один вызов укладывается в SLA на практике (см. Этап 6). Не Yandex: нужна стабильность JSON-вывода без доработки промпта под конкретного провайдера (см. находку в Этапе 3) |
| Категоризация источников и покрытие при retrieval | **Решено 2026-06-20** — поле `category` (5 значений: `labor_code`, `case_law`, `federal_law`, `other_npa`, `authorial`) заменяет грубый `source_type` (`law`/`article`); hybrid search (Этап 5) становится категорийно-сбалансированным — гарантированный пул кандидатов на каждую категорию, а не общий плоский top-20. Одна база/один сервис, не пять отдельных RAG (это обсуждалось и отклонено — см. Этап 5.1) | ТК РФ — самый большой корпус и лексически ближайший к типичной формулировке вопроса пользователя, поэтому при плоском ранжировании по сходству он систематически вымывает из топ-20 более редкие, но юридически значимые источники (разъяснения Пленумов ВС РФ, подзаконные акты/иные ФЗ, авторские комментарии). Решение воспроизводит реальную методологию юридического анализа (база — ТК РФ → разъяснения высших судов → иные акты/ФЗ → авторские комментарии, по аналогии с системой КонсультантПлюс): без гарантированного покрытия каждой категории итоговые консультации Веры по важным вопросам останутся юридически неполными |
| Sparse-поиск (BM25) | **Решено 2026-06-21 (Этап 12, SEARCH-1/QD-3)** — нативные sparse-векторы Qdrant с IDF-модификатором (`app/vectorstore/sparse.py`), не `rank_bm25` в памяти (изначальное упрощение Этапа 5) | Клиентская реализация выгружала весь корпус (`scroll`) и пересчитывала BM25-индекс с нуля на каждый запрос, ×5 при категорийной балансировке — O(N) от размера корпуса, блокировало единственный event loop сервиса. Нативный sparse-вектор — обычный индексный запрос, не масштабируется хуже dense-поиска |
| Авторизация публичного API | **Решено 2026-06-21 (Этап 12, ARCH-1/API-1/SEC-1)** — единый `X-API-Key`, сверяется `hmac.compare_digest` с одним значением из `Settings`, не таблица ключей в БД | Изначально не было авторизации вообще — приемлемо только за закрытым периметром с одним доверенным клиентом (MCP Tools Server); единственный известный потребитель делает полную таблицу ключей избыточной на этом этапе, см. Этап 12 |

---

## 1. Структура проекта

> Обновлено на Этапе 12 (2026-06-21) — отражает фактическую структуру после ревью, не первоначальный план. Расхождения с более ранней версией этого раздела: `app/config.py` → `app/core/settings.py` (+ `config_logger.py`, `rate_limit.py`, `circuit_breaker.py`, `request_context.py`); `app/api/*` → `app/api/v1/endpoints/*`; `app/logging/` не существовал — логирование поисковых запросов живёт в `app/repositories/search_log.py` + `app/db/models/search_log.py`; появились слои `admin/`, `dependencies/`, `repositories/`, `exceptions/`, `clients/`, которых не было в исходном плане.

```
vera_rag_service/
├── app/
│   ├── main.py                    # FastAPI app, middleware (rate limit, request_id), lifespan, /metrics
│   ├── core/
│   │   ├── settings.py             # Pydantic Settings (app/db/qdrant/yandex/polza/search)
│   │   ├── config_logger.py        # logging.config.fileConfig + RequestIdLogFilter
│   │   ├── request_context.py      # contextvars: request_id сквозной через HTTP-слой и логи
│   │   ├── rate_limit.py           # slowapi Limiter (общий для API и /admin/login)
│   │   └── circuit_breaker.py      # CircuitBreaker — module-level singleton на провайдера
│   ├── api/v1/endpoints/
│   │   ├── search.py                # POST /search (X-API-Key, rate limit)
│   │   ├── ingest.py                # POST /ingest (X-API-Key, rate limit, лимиты размера)
│   │   ├── documents.py             # DELETE /document/{id}
│   │   └── health.py                # GET /health (без авторизации)
│   ├── dependencies/                # Depends-фабрики: auth, clients, http_client, vectorstore, repositories, services, db_session
│   ├── ingestion/
│   │   ├── preprocess.py           # очистка текста, извлечение структуры (статьи/пункты/секции)
│   │   ├── chunking.py             # иерархический чанкинг, overlap, детерминированный chunk_id (uuid5)
│   │   ├── enrichment.py           # синтетические заголовки + гипотетические вопросы (LLM)
│   │   ├── extract.py              # извлечение текста из PDF/DOCX/MD/TXT, лимиты размера/страниц
│   │   └── prompts/enrichment.py
│   ├── search/
│   │   ├── hybrid.py               # dense + sparse (нативные Qdrant sparse-векторы, IDF) + категорийная балансировка
│   │   ├── fusion.py               # RRF (Reciprocal Rank Fusion)
│   │   ├── reranker.py             # LLM-reranker (Polza/Gemini), не cross-encoder
│   │   └── prompts/reranker.py
│   ├── embeddings/
│   │   └── embedder.py             # обёртка над Yandex Embedding API
│   ├── clients/
│   │   ├── llm.py                  # LlmClient — retry/backoff, circuit breaker, strip_markdown_artifacts
│   │   ├── embeddings.py           # EmbeddingClient — то же
│   │   └── http_client.py          # общий module-level httpx.AsyncClient (не per-request)
│   ├── vectorstore/
│   │   ├── qdrant_client.py        # коллекция: named vectors + sparse + квантизация + payload-индексы
│   │   ├── sparse.py               # text → sparse-вектор (term-frequency, для нативного BM25 Qdrant)
│   │   └── client.py               # module-level singleton AsyncQdrantClient
│   ├── repositories/
│   │   ├── search_log.py           # запись search_logs (Postgres)
│   │   └── document.py             # реестр documents + advisory lock + delete
│   ├── services/
│   │   ├── search.py               # embed_query → hybrid → rerank → лог
│   │   ├── ingestion.py            # preprocess → chunk → enrich → embed → upsert → реестр
│   │   └── documents.py            # удаление документа (Qdrant + реестр, единая точка для API и админки)
│   ├── admin/                      # sqladmin: views, auth, csrf, dashboard, reconciliation, services (DI вручную)
│   ├── models/
│   │   ├── schemas.py              # Pydantic: SearchRequest/Response, IngestRequest/Response, Chunk, ...
│   │   └── metadata.py             # ChunkMetadata, Category, Audience
│   ├── exceptions/                 # доменные исключения по слоям (llm, embedding, document, ingestion, search_log, health)
│   └── db/
│       ├── session.py               # module-level singleton AsyncEngine
│       ├── models/                  # SearchLog, Document
│       └── alembic/                 # async-миграции
├── tests/
│   ├── unit/                      # клиенты, чанкинг, препроцессинг, reranker, fusion, sparse, circuit breaker, admin auth/csrf
│   ├── api/endpoints/              # httpx.ASGITransport + dependency_overrides: auth, rate_limit, admin views, request_id, metrics
│   ├── integration/                # реальные Qdrant (docker-compose) + Postgres (testcontainers): vectorstore, search, services, repositories, admin
│   └── performance/                 # @pytest.mark.slow — синтетический корпус 5000 чанков, не входит в обычный прогон
├── .github/workflows/ci.yml        # lint (ruff) + тесты на каждый push/PR
├── Dockerfile                       # multi-stage, non-root, HEALTHCHECK
├── docker-compose.yml               # rag_service + qdrant (memory limit) + db
├── entrypoint.sh                    # RUN_MIGRATIONS_ON_START, HYPERCORN_WORKERS
├── pyproject.toml                   # ruff
├── requirements.txt / requirements-dev.txt
└── README.md                        # + чеклист перед production-развёртыванием
```

---

## 2. Этапы реализации

### Этап 1 — Препроцессинг документов ✅ Выполнено
- Загрузка исходников (PDF / MD / TXT) от Expert
- Очистка текста: переносы строк, артефакты форматирования, колонтитулы
- Извлечение структуры:
  - нормативные акты → номера статей/пунктов как метаданные
  - авторские статьи → заголовки секций как метаданные
- Выход: список секций с текстом + структурными метаданными

### Этап 2 — Иерархический чанкинг ✅ Выполнено
- Документ → секция → чанк (300–500 токенов, overlap 50–100 токенов)
- Секция сохраняется как метаданные у каждого чанка (контекст шире чанка для LLM на этапе генерации)
- Присвоение `chunk_id`, `chunk_index`
- Юнит-тесты: границы чанков, overlap, сохранение метаданных секции

### Этап 3 — Обогащение чанков (LLM, ingestion-time) ✅ Выполнено, проверено на реальном Yandex Cloud API
- Синтетический заголовок чанка (LLM суммирует суть) — добавляется в текст перед эмбеддингом
- Генерация 3–5 гипотетических вопросов на чанк — индексируются как дополнительные векторы
- Цель: компенсировать слабость векторного поиска на юридических текстах (см. исходный архитектурный документ, раздел "Особенности поиска по юридическим текстам")

### Этап 4 — Embedding и upsert в Qdrant ✅ Выполнено, проверено на реальном Yandex API + локальном Qdrant
- Embedding-модель: **Yandex Text Embeddings v2**, `text-embeddings-v2-doc` при индексации / `text-embeddings-v2-query` при поиске (см. раздел 0.1), размерность вектора — 768
- Генерация эмбеддингов через Yandex AI Studio API: чанк (заголовок+текст) + гипотетические вопросы
- Создание коллекции(й) в Qdrant с поддержкой dense (768) + sparse векторов (BM25 — реализуется отдельно, модель не даёт sparse "из коробки")
- Upsert: вектор + полная схема метаданных (см. раздел 3)
- Скрипт пакетной загрузки (`scripts/ingest_corpus.py`) для Фазы 4
- Учесть в `config.py`: `YANDEX_FOLDER_ID`, `YANDEX_API_KEY`, retry/backoff на вызовы внешнего API (сетевая зависимость в ingestion и в query-time hot path)

### Этап 5 — Hybrid search ✅ Выполнено, проверено на реальном Qdrant
- Dense search (cosine) — top-20
- Sparse search (BM25) — top-20, закрывает точные термины ("статья 21", "квота 2%"). **Изменено на Этапе 12**: изначально реализовано как `rank_bm25` в памяти над кандидатами после `scroll` (осознанное упрощение для небольшого корпуса) — на Этапе 12 (SEARCH-1/QD-3) мигрировано на нативные sparse-векторы Qdrant с IDF-модификатором (`app/vectorstore/sparse.py`, `app/search/hybrid.py::sparse_search`) — обычный индексный запрос вместо полной выгрузки коллекции на каждый запрос. См. раздел "Этап 12".
- RRF fusion — объединение ранжирований в единый список кандидатов
- Фильтрация по метаданным до векторного сравнения (`audience`, `category`, `topic`)

### Этап 5.1 — Категорийно-сбалансированный retrieval (покрытие источников) ✅ Выполнено
> Зафиксировано и реализовано 2026-06-20 по итогам обсуждения с тем, кто 15 лет проработал юристом до разработки (см. раздел 0.1). Расширяет Этап 5, не отдельный сервис — три варианта (5 независимых RAG-сервисов / 5 коллекций с ручным фьюжном агентом / один сервис с category-aware retrieval) обсуждены, выбран последний: 5 отдельных RAG означали бы 5× LLM-вызовов reranker'а на каждый запрос пользователя (латентность = по самому медленному из пяти + агенту нужен ещё один LLM-шаг синтеза) и неизбежный шум — нерелевантные категории (например, судебная практика на вопрос, целиком закрываемый ТК РФ) всё равно обязаны вернуть что-то в топ-5.

**Проблема:** реальная методология юридического анализа (источник: 15 лет практики в профессии) — любой ответ строится в порядке (1) базовые нормы ТК РФ → (2) разъяснения высших судов (Пленумы ВС РФ, значимая практика) → (3) иные федеральные законы и подзаконные акты (постановления Правительства и т.п.) → (4) авторские разъяснения порядка применения норм (аналог авторских материалов в КонсультантПлюс). Плоский hybrid search (текущий Этап 5) ранжирует кандидатов по сходству с запросом без учёта этой иерархии — ТК РФ как самый большой и лексически близкий к формулировке вопроса корпус систематически вымывает из top-20 малочисленные, но юридически важные источники (пункт 2–4 методологии). Расчёт на "общий индекс сам разберётся" не подтверждён и не должен приниматься без проверки на реальном корпусе.

**Решение:**
- Поле `category` (заменяет `source_type` в `app/models/metadata.py`, `app/models/schemas.py`, payload Qdrant `app/vectorstore/qdrant_client.py`) — `Literal['labor_code', 'case_law', 'federal_law', 'other_npa', 'authorial']`. `topic` (тема, например `quota`) остаётся отдельным полем — `category` про тип/иерархию источника, не про предмет вопроса.
- `app/search/hybrid.py`: вместо одного плоского top-20 (`DENSE_TOP_K`/`SPARSE_TOP_K`) — параллельный dense+sparse поиск **на каждую категорию отдельно** (фильтр по `category`, по аналогии с уже существующим фильтром по `audience`/`topic`), top-4 dense + top-4 sparse на категорию (`DENSE_TOP_K_PER_CATEGORY`/`SPARSE_TOP_K_PER_CATEGORY`), чтобы суммарный объём кандидатов на входе RRF/reranker остался того же порядка, что и сейчас (5 категорий × 4+4 ≈ текущие 20+20). Категория без кандидатов после фильтра просто не даёт вклада — без ошибки, по аналогии с деградацией reranker'а при отказе LLM (Этап 6). Если вызывающий указал `category` явно — балансировать нечего, обычный плоский top-20 внутри запрошенной категории.
- RRF (Этап 5, `app/search/fusion.py`) фьюзит уже объединённый из всех категорий пул — гарантия представленности, не гарантия попадания в финальный топ-5: финальное решение, насколько разъяснение Пленума или авторский комментарий реально релевантны конкретному вопросу, остаётся за LLM-reranker'ом (Этап 6), как и сейчас.
- `SearchResultChunk` (`app/models/schemas.py`) сейчас не возвращает ни `source_type`, ни `category` — добавить `category` в ответ `/search`, чтобы потребитель (Agent Service) мог выстроить финальный ответ Веры в том же порядке (база → практика → иные акты → комментарий), не теряя эту структуру на своей стороне. Сама генерация ответа агентом остаётся вне рамок RAG Service (раздел 0).
- Затронуты тесты: `tests/unit/test_chunking.py`/`test_preprocess.py` (если используют `source_type` в фикстурах), `tests/integration/search/test_hybrid_search.py`, `tests/integration/vectorstore/test_qdrant_client.py`, `tests/api/endpoints/*`, `tests/unit/services/test_search_service.py` — везде, где сейчас фигурирует `SourceType`/`source_type`.
- Админка (`app/admin/views.py`, Этап 11.1 ниже) — форма загрузки документа должна предлагать выбор `category` из 5 значений, не `law`/`article`.
- **Открытый вопрос:** нужна ли эмпирическая проверка top-k на категорию (8+8 — первая оценка, не измерено) после загрузки реального корпуса (ТК РФ + хотя бы несколько Постановлений Пленума) — см. риски, раздел 4.

### Этап 6 — Reranker ✅ Выполнено (LLM-reranker через Polza/Gemini, проверено на реальных вызовах)
- Не cross-encoder (`BAAI/bge-reranker-base`), как изначально планировалось в разделе 0.1 — решение пересмотрено в пользу LLM-reranker'а (Polza AI/Gemini), см. раздел 0.1 и подробности реализации в статус-шапке выше
- Переоценка кандидатов по паре (запрос, чанк), кандидатам присваиваются номера для устойчивости к искажению UUID в выводе LLM
- Возврат до 5 финальных результатов с метаданными и `source_title`; деградация до исходного RRF-порядка при отказе LLM
- **Этап 12 (LLM-3/SEC-5, SEARCH-3):** промпт обёрнут в XML-теги-разделители (`<user_query>`/`<candidate id="N">`) с явной анти-injection инструкцией; текст каждого кандидата урезается до 600 символов, чтобы суммарная длина промпта не росла неограниченно с числом категорий

### Этап 7 — API ✅ Выполнено
| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/search` | query + фильтры (`audience`, `topic`) → топ-5 чанков с метаданными |
| `POST` | `/ingest` | добавление документа в БЗ (запуск ingestion pipeline) |
| `DELETE` | `/document/{id}` | удаление документа и всех его чанков из Qdrant |
| `GET` | `/health` | статус сервиса и подключения к Qdrant |

Pydantic-схемы запросов/ответов фиксируются в `app/models/schemas.py` — это и есть контракт для MCP Tools Server (см. WBS 2.9.2, 3.4.6).

**Обновление документа (новая редакция нормативного акта):**
- Явный workflow, а не произвольная комбинация `DELETE`+`POST /ingest`: `POST /ingest` для новой редакции создаёт чанки с новым `version`/`effective_date`; только после успешного upsert новой версии — удаление чанков старой версии по `document_id`+старому `version`
- Гарантирует отсутствие окна недоступности источника между удалением старой и загрузкой новой редакции
- Аудит "какая редакция была проиндексирована" — на уровне **документа**, не отдельного чанка: реестр `documents` в Postgres (`is_active: bool`, Этап 11.1) хранит историю версий, старые чанки в Qdrant физически удаляются (не помечаются неактивными). **Изменено на Этапе 12 (ING-6):** поле `is_active` было и в `ChunkMetadata`/Qdrant payload, но всегда оставалось `True` (мёртвое поле — удаление чанков делается физическим `delete`, не сменой флага) — убрано из схемы метаданных чанка, см. раздел 3

### Этап 8 — Персистентное логирование поисковых запросов ✅ Выполнено
> Решение: OTel/Arize Phoenix исключены из рамок этого репозитория (см. ниже) — единственный механизм наблюдаемости здесь — структурные логи (`logging.ini`, раздел 4 `FASTAPI_PATTERNS.md`) плюс эта таблица.

- Хранилище: таблица в Postgres — не Qdrant, чтобы не смешивать операционные данные поиска с векторным хранилищем
- Что пишем на каждый `/search`:
  - вход: `query`, `audience`, `topic`, `request_id`, `timestamp`
  - промежуточные данные: результаты dense top-20, sparse/BM25 top-20 (до fusion), результат RRF fusion, результат reranker (топ-5 с финальными скорами)
  - выход: финальный список чанков, отданный MCP Tools Server, latency по стадиям (`embed_query`, `hybrid_search`, `rerank`)
- Назначение: оценка релевантности (WBS 5.4, 5.6), отладка некорректных результатов поиска, накопление датасета для улучшения чанкинга/обогащения
- **Граница ответственности:** RAG Service логирует только сам факт поискового запроса и его результат. Полный диалог "вопрос пользователя → финальный ответ Веры" (после LLM-генерации) — зона Agent Service, не этого репозитория. Если нужна сводная админ-панель для контроля качества ответов на уровне всего продукта — это отдельная задача вне рамок RAG Service (возможно, в Agent Service или отдельном компоненте), зафиксировано как открытый вопрос
- **Почему не OTel/Arize Phoenix:** изначально план предполагал OTel-спаны (`embed_query`, `vector_search`) с экспортом в Arize Phoenix. Отказались — RAG Service не видит финальный ответ агента (см. границу ответственности выше), поэтому трейс на стороне этого сервиса всегда обрублен на retrieval-шагах и не даёт той картины "вопрос → ответ", ради которой Phoenix обычно и подключают. Распределённый трейсинг через весь продукт (RAG-спаны вложены в трейс Agent Service) — решение уровня всего продукта, не этого репозитория; если оно будет принято, тема возвращается отдельным пунктом. До тех пор инженерная отладка latency/ошибок этого сервиса закрывается обычными структурными логами

### Этап 9 — Тестирование ✅ Выполнено по факту (накоплено по ходу Этапов 1–12, формализовано CI на Этапе 12)
- Юнит-тесты ingestion pipeline (чанкинг, заполнение метаданных) — WBS 3.1.11 ✅
- Юнит-тесты hybrid search и reranker на синтетических запросах — WBS 3.1.12 ✅
- Фикстуры — синтетические документы/чанки во всех unit/integration-тестах (не отдельная заранее подготовленная директория `tests/fixtures/`, как планировалось изначально — фикстуры строятся в каждом тестовом модуле по месту)
- Интеграционная проверка после получения реального корпуса (Фаза 4, п. 4.1.3–4.1.4) — **остаётся открытым пунктом**: 143 теста зелёных на синтетических данных + 2 нагрузочных (`tests/performance`, 5000 синтетических чанков), но реального корпуса от Expert ещё нет — recall/качество на нём не проверены (см. SEARCH-2, открытый вопрос)
- **Добавлено на Этапе 12, не было в исходном плане:** CI (`.github/workflows/ci.yml`, ARCH-7), `ruff` с конфигурацией (`pyproject.toml`), тесты на саму авторизацию admin/API (TEST-4), regression-тест на гонки/идемпотентность ingestion (TEST-2), нагрузочный тест на масштаб (TEST-3, `@pytest.mark.slow`)

### Этап 10 — Деплой ✅ Выполнено
- Dockerfile сервиса — multi-stage, non-root, `HEALTHCHECK` (см. Этап 12, ARCH-8)
- docker-compose с Qdrant (self-hosted) для тестового окружения — + явный memory limit для Qdrant (Этап 12, QD-2)
- Переменные окружения — см. `.env.example`: `QDRANT_URL` и весь остальной конфиг через `app/core/settings.py` (не отдельные `EMBEDDING_MODEL`/`RERANKER_MODEL`, как абстрактно планировалось — конкретные провайдер-специфичные настройки, см. раздел 0.1)
- Деплой в тестовое окружение (WBS 3.1.14) — локально проверено (`docker-compose up`, `hypercorn`); реальное тестовое окружение продукта «Работа для всех» — вне этого репозитория

### Этап 11 — Админка: управление документами и интерактивное тестирование поиска ✅ Выполнено
> Зафиксировано 2026-06-20 по итогам ревью Этапа 8 (админка с журналом `search_logs`) — найден разрыв между тем, как документы реально готовит Expert, и тем, что умеет принимать сервис. Реализовано 2026-06-20 (после Этапа 5.1, до Этапов 9–10 — план явно допускал такой порядок при наличии ресурсов).

**11.1. Загрузка и управление документами через `/admin`** — реализовано
- **Открытый вопрос решён:** выбран вариант (а) — новая таблица `documents` в Postgres (`app/db/models/document.py::Document`, миграция `app/db/alembic/versions/20260620_2200_add_documents_table.py`, depends_on `search_logs`). Одна строка — одна версия документа: `document_id`, `version`, `category`, `source_title`, `audience`, `topic`, `effective_date`, `is_active`, `created_at`. Пишется/обновляется из `IngestionService` (`app/repositories/document.py::DocumentRepository.save_document`/`mark_versions_inactive`) — отказ записи сюда не должен ронять ingestion (Qdrant к этому моменту уже успешен), перехватывается и логируется как предупреждение, по аналогии с `search_logs` (Этап 8).
- Извлечение текста из PDF/MD/TXT — `app/ingestion/extract.py::extract_text_from_upload`, новая зависимость `pdfplumber==0.11.10`. `python-multipart` не добавлялся отдельно — уже была транзитивной зависимостью `sqladmin`.
- **Добавлено позже (после Этапа 12):** поддержка `.docx` — `extract_text_from_docx` в том же модуле, парсер (параграфы + гиперссылки + таблицы + колонтитулы, в исходном порядке документа) перенесён из родственного внутреннего проекта `FileTextParser` (`app/text_extraction/docx_format_handler.py`) **без** извлечения текста с встроенных изображений (там это делалось через LLM Vision OCR — осознанно не переносилось). Важное отличие от исходной реализации: переносы строк между параграфами/ячейками **сохраняются** (не схлопываются в одну строку через `re.sub(r'\s+', ' ', text)`, как в источнике) — иначе `preprocess_document::LAW_ARTICLE_PATTERN` (ищет `^Статья\s+N` в начале строки) не нашёл бы статьи в тексте из .docx. Лимит `MAX_CHARACTERS=350000` из исходного проекта не переносился — это решение по бизнес-логике другого проекта, у нас уже есть свои лимиты (`MAX_UPLOAD_SIZE_BYTES`, `MAX_CHUNKS_PER_DOCUMENT`). Новая зависимость — `python-docx==1.1.2` (тащит `lxml` транзитивно). `.doc`/`.rtf` — не поддерживаются.
- Форма загрузки — `app/admin/views.py::DocumentUploadView` (`BaseView` + `@expose('/document-upload', methods=['GET','POST'])`), шаблон `app/templates/document_upload.html`: файл + `document_id`, `category` (5 значений, Этап 5.1), `source_title`, `audience`, `topic`, `version`, `effective_date` → вызывает `IngestionService.ingest_document` напрямую.
- Список/удаление — `app/admin/views.py::DocumentAdmin` (`ModelView` над `Document`, read+delete только). Удаление в админке — удаление документа из БЗ целиком, не только строки реестра: `delete_model` переопределён, чтобы сначала удалить чанки из Qdrant (`QdrantVectorStore.delete_document`), источник правды о содержимом БЗ, и только потом саму строку.
- **Находка, важная для следующих BaseView-страниц:** sqladmin BaseView-страницы не проходят через FastAPI `Depends()` — сервисы (`IngestionService`/`SearchService`) собираются вручную из тех же чистых функций-зависимостей (`app/admin/services.py::build_ingestion_service`/`build_search_service`), не дублируя логику, только способ вызова.
- **Баг, найденный и исправленный при ручной проверке:** `Admin.add_base_view()` (вызванная напрямую, как показано в официальном примере sqladmin) **не выставляет `_admin_ref`** на класс view — из-за этого `login_required` (декоратор sqladmin, применяемый через `@expose`) тихо пропускает проверку авторизации, и страница отвечает `200` без логина вместо редиректа на `/admin/login`. Подтверждено и исправлено: регистрировать BaseView через `Admin.add_view()` (как и `ModelView`), а не `add_base_view()` напрямую — она сначала проставляет `_admin_ref`, затем делегирует. Проверено `curl` до/после фикса на живом сервере.

**11.2. Интерактивное тестирование поиска через `/admin`** — реализовано
- `app/services/search.py::SearchService.search_with_diagnostics` — `search()` стал тонкой обёрткой над ним (без дублирования retrieval-логики), возвращает `SearchDiagnostics` (dense/sparse/fused до фьюжна, порядок reranker'а, финальные `results`).
- `app/admin/views.py::SearchTestView` (`BaseView` + `@expose('/search-test', ...)`), шаблон `app/templates/search_test.html`: форма (вопрос + опциональные `audience`/`topic`/`category`) → `search_with_diagnostics` → рендер по стадиям. Каждый прогон автоматически попадает в `search_logs` (Этап 8) — `search_with_diagnostics` пишет лог так же, как `search`.

**Проверено сквозным прогоном на реальном сервере** (`hypercorn` локально, Postgres+Qdrant в Docker): логин → загрузка `.txt` через форму → реальный ingestion (Yandex embedding + enrichment) → запись появилась в `/admin/document/list` и в Qdrant (`points/count`) → поиск через `/admin/search-test` нашёл проиндексированный чанк с синтетическим заголовком от LLM → удаление через админку убрало чанк из Qdrant **и** строку из `documents`. 83 теста зелёных (+ 6 новых: `test_extract.py`, `test_document_repository.py`).

**11.3. Расширение: просмотр содержимого чанков + дашборд мониторинга** — реализовано 2026-06-20
> Зафиксировано по итогам обратной связи: админка должна быть полноценной консолью управления сервисом (отслеживание работы + управление БЗ), не только журналом логов — `DocumentAdmin` показывал только метаданные реестра в Postgres, не сам индексированный текст, и не было сводной картины состояния сервиса.
- `QdrantVectorStore.list_chunks(document_id, version=None)` — payload всех чанков документа (текст, синтетический заголовок, гипотетические вопросы, метаданные), отсортированные по `chunk_index`.
- `app/admin/views.py::DocumentChunksView` (`/admin/document-chunks`, форма по `document_id`+опционально `version`) — `document_id` в списке `DocumentAdmin` стал ссылкой сюда (`_document_id_link`, экранирование через `markupsafe.escape`).
- `app/admin/dashboard.py::get_dashboard_stats` + `app/admin/views.py::DashboardView` (`/admin/dashboard`) — сводка: статус Postgres/Qdrant, количество документов (всего/уникальных/активных), количество чанков в Qdrant, количество поисковых запросов, средняя latency по стадиям, время последнего поиска. Postgres и Qdrant опрашиваются независимо — недоступность одного не скрывает статистику другого (деградация по аналогии с разделом 9 `FASTAPI_PATTERNS.md`).
- **Баг, найденный при проверке (не sqladmin, наш):** при правке Этапа 5.1 миграция `search_logs` (`source_type`→`category`) была отредактирована "на месте" в предположении, что ещё не применялась к рабочей БД — на самом деле уже была применена раньше в этой же сессии. Alembic не переигрывает изменённые старые миграции, поэтому живая колонка осталась `source_type`, а ORM-модель указывала `category` — `UndefinedColumnError` при первой реальной записи `search_logs` через `/admin/search-test`. Исправлено настоящей миграцией `ALTER TABLE` (`20260620_2210_rename_search_logs_source_type_to_category.py`), не повторной правкой старого файла. **Урок:** редактировать уже как-либо применённую миграцию "на месте" нельзя даже в личном/несданном проекте — нужно проверять `alembic_version` в целевой БД, не только git-историю.
- Проверено сквозным прогоном: загрузка документа → `/admin/document-chunks` показал реальный текст и синтетический заголовок → `/admin/dashboard` отразил счётчики до/после загрузки и поиска, включая latency по стадиям после реального поиска. 85 тестов зелёных (+ 2 новых в `test_qdrant_client.py` для `list_chunks`).

### Этап 12 — Техническое ревью, независимая верификация и устранение находок ✅ Выполнено
> Зафиксировано и реализовано 2026-06-21. Внешнее критичное техническое ревью кодовой базы (заказано отдельным заданием, не часть исходной дорожной карты выше) дало 47 находок по архитектуре, безопасности, масштабируемости и поддерживаемости. Каждая находка была независимо перепроверена против реального кода (не принята на веру) — статус Confirmed/Partially Confirmed для всех 47, ни одна не оказалась ложной. После верификации — реализация и тесты на каждую находку. Документы самого процесса ревью (исходное задание на ревью, само ревью, задание на верификацию+имплементацию, верификация) консолидированы в этот файл и удалены из репозитория — этот раздел и таблица в разделе 7 — единственный источник правды о том, что было найдено и что сделано.

**Главные блокеры для публичного production-развёртывания (закрыты):**
1. **Авторизация API** (ARCH-1/API-1/SEC-1) — `/search`, `/ingest`, `DELETE /document/{id}` не имели вообще никакой проверки. `app/dependencies/auth.py::verify_api_key` — `X-API-Key` против `Settings.app.api_key` через `hmac.compare_digest`, подключено на уровне роутера (`APIRouter(dependencies=[VerifyApiKeyDep])`) во всех трёх модулях; `/health` осознанно без ключа.
2. **BM25 блокировал event loop** (SEARCH-1/QD-3) — клиентский `rank_bm25` выгружал весь корпус и пересчитывал индекс на каждый запрос (×5 при категорийной балансировке), O(N) от размера корпуса на единственном worker'е. Мигрировано на нативные sparse-векторы Qdrant с IDF (`app/vectorstore/sparse.py::text_to_sparse_vector`, term-frequency + `zlib.crc32` для индексов токенов — без построения отдельного словаря). Заодно — payload-индексы (QD-1: `category`/`audience`/`document_id`/`version`/`topic`) и int8-квантизация `chunk`-вектора (QD-2) в той же миграции схемы коллекции.
3. **Ingestion не идемпотентен** (ING-1/2/3) — `chunk_id = uuid4()` при каждом вызове плодил дубликаты при повторном/параллельном ingestion. `chunk_document(sections, version)` теперь вычисляет `chunk_id = uuid5(NAMESPACE_URL, f"vera-rag-service:{document_id}:{version}:{chunk_index}")` — повтор того же `document_id`+`version` перезаписывает те же точки Qdrant, не плодит новые. `DocumentRepository.acquire_document_lock`/`release_document_lock` — сессионный `pg_advisory_lock` вокруг `ingest_document` серилизует конкурентные вызовы. Цикл удаления старых версий обёрнут в try/except (не падает на середине, `IngestResponse.not_removed_versions` явно сообщает о неудалённых).
4. **`.env` в Docker-образе** (SEC-6) — `.dockerignore` не исключал `.env`/`.env.*`, секреты потенциально утекали в слои образа.
5. **Stored XSS в админке** (ADM-1/SEC-3) — `_fmt_json` вставлял JSON в HTML через `Markup()` без экранирования; контент происходил из реальных документов/LLM-вывода. Исправлено `escape(pretty)`, тот же паттерн, что уже использовался в `_document_id_link`.

**Остальные находки (42 шт., Critical/High/Medium/Low) — по категориям, подробности и точные файлы см. таблицу в разделе 7:**
- **API/безопасность** — rate limiting (`slowapi`, по IP, отдельный строгий лимит на `/ingest` и `/admin/login`), лимит размера `raw_text`/числа чанков, унификация удаления документа (API и админка — один код в `DocumentsService`), CSRF (synchronizer token в сессии), `hmac.compare_digest` для логина админки, лимит размера файла/страниц PDF, `https_only` для сессии админки (конфигурируемо, по умолчанию `False` для локальной разработки).
- **LLM/prompt injection** — обходные пути под капризы YandexGPT (`strip_markdown_artifacts`) изолированы от Polza/Gemini через флаг конструктора; circuit breaker (`CircuitBreaker`, module-level singleton на провайдера+use-case — не на клиент, который создаётся per-request); XML-теги-разделители + анти-injection инструкция в промптах reranker'а и enrichment'а; урезание текста кандидата в промпте reranker'а.
- **Логирование/наблюдаемость** — структурированные (JSON) логи вместо текста с эмодзи (`python-json-logger`); сквозной `request_id` через `contextvars` и middleware (виден и в структурных логах, и в `search_logs.request_id`, и в заголовке ответа `X-Request-ID`); текст поискового запроса убран из stdout-логов (152-ФЗ — остаётся только в защищённой авторизацией `search_logs`); окно 24 часа для расчёта средней latency на дашборде (не по всей истории таблицы); `/metrics` (Prometheus, без авторизации — ограничивать на уровне сети в production).
- **Инфраструктура** — `pyproject.toml` (`ruff`) + `.github/workflows/ci.yml` (lint+тесты на каждый push/PR); multi-stage `Dockerfile` (non-root, `HEALTHCHECK`, без build-тулчейна и dev-зависимостей в runtime, `requirements-dev.txt` отдельно); общий module-level `httpx.AsyncClient` для Yandex/Polza вместо пересоздания на каждый запрос; `--workers` 1→2 (после фикса блокировки event loop); `RUN_MIGRATIONS_ON_START` — переключатель в `entrypoint.sh` для будущего перехода на несколько реплик.
- **Прочее** — сверка реестра `documents` (Postgres) и содержимого Qdrant по запросу на дашборде (ING-5, с ограничением по числу активных документов); удалено мёртвое поле `is_active` из `ChunkMetadata`/Qdrant payload (ING-6); top-K на категорию вынесен в `Settings` (SEARCH-2); `IngestionService.ingest_document` разбит на именованные шаги без новой pipeline-абстракции (ARCH-2, YAGNI); тесты на регрессию авторизации admin BaseView (TEST-4) и на идемпотентность/гонки ingestion (TEST-2); нагрузочный тест на синтетическом корпусе 5000 чанков, `@pytest.mark.slow`, не в обычном прогоне (TEST-3).
- **Операционные заметки (не код, см. `README.md`, раздел "Чеклист перед production-развёртыванием")** — не использовать `/rc`-канал YandexGPT в production (LLM-4); настроить периодические снапшоты Qdrant с выгрузкой вне `qdrant_data`-volume (QD-5); явный memory limit для контейнера Qdrant в `docker-compose.yml` (раньше лимита не было вообще, не только "недостаточный" — QD-2).

**Проверено:** 143 теста зелёных (юнит + API + интеграционные на реальных Qdrant/Postgres) + 2 нагрузочных (`pytest -m slow tests/performance`, синтетический корпус 5000 чанков, ~11с — подтверждает отсутствие регрессии SEARCH-1/QD-3). `ruff check .` — чисто. Docker-образ собран и проверен реально (`docker build`/`docker run`): non-root пользователь, `gcc`/`pytest` отсутствуют в финальном слое, runtime-зависимости импортируются без ошибок.

**Не закрыто, осознанно (зафиксировано как технический долг, не забытая деталь):**
- Полное партиционирование/retention-политика для `search_logs` (LOG-3/LOG-4) — инфраструктурное решение (куда архивировать, на каком расписании), не принимается в одностороннем порядке правкой кода; сделано то, что устранимо кодом (окно 24ч для дашборда, текст запроса не в stdout).
- `ING-4` (`asyncio.gather` без `return_exceptions=True`) — улучшена диагностика (видно, какие именно `chunk_index` не прошли), но сохранена политика "весь документ или ничего" — частичный успех для нормативных документов означал бы юридический риск (документ неполон в БЗ без явного сигнала), сочли более серьёзным, чем повторная оплата LLM/embedding при retry.
- Эмпирическая проверка top-K на категорию (SEARCH-2) и ценности отдельных `question_N`-векторов (QD-2) — требуют реального корпуса от Expert, физически невозможны до его получения.
- `ARCH-3` (категории — захардкоженный `Literal`) и `ARCH-5` (связанность с Qdrant) — осознанно без изменений (YAGNI), решение зафиксировано как обоснованное в самом ревью.

---

## 3. Схема метаданных чанка

```python
{
    "chunk_id": "uuid",
    "document_id": "fz-181-art21",
    "category": "labor_code" | "case_law" | "federal_law" | "other_npa" | "authorial",
    "source_title": "ФЗ-181, Статья 21",
    "audience": "seeker" | "employer" | "both",
    "topic": "quota" | "rights" | "dismissal" | "workplace" | ...,
    "date_added": "2026-06-01",
    "chunk_index": 3,
    "version": "2026-01-01",        # дата редакции нормативного акта (для законов) или ревизии статьи
    "effective_date": "2026-01-01"  # дата вступления редакции в силу
}
```

`audience` — ключевое поле для фильтрации до векторного сравнения (вопрос работодателя исключает чанки только для соискателей).

`category` — **изменено 2026-06-20, было `source_type: "law" | "article"`** (см. раздел 0.1 и Этап 5.1). Пять значений отражают реальную иерархию юридического анализа, а не просто "закон/статья":
- `labor_code` — ТК РФ, базовые нормы;
- `case_law` — значимая судебная практика, разъяснения Пленумов ВС РФ и иных высших судов;
- `federal_law` — иные федеральные законы (не ТК РФ), например ФЗ-181;
- `other_npa` — подзаконные нормативные акты (постановления Правительства и т.п.);
- `authorial` — авторские статьи и систематизации (готовит Expert), аналог авторских разъяснений в КонсультантПлюс.

Используется не только для фильтрации по явному запросу, но и как основа категорийно-сбалансированного retrieval (Этап 5.1) — каждая категория получает гарантированный пул кандидатов независимо от объёма корпуса, чтобы малочисленные, но юридически значимые источники (`case_law`, `authorial`) не вымывались из топ-20 крупным `labor_code`.

`version` / `effective_date` — нужны для аудита: если ответ агента дан со ссылкой на чанк, должно быть восстанавливаемо, какая именно редакция документа была проиндексирована на момент ответа.

---

## 3.1. Оценка стоимости эмбеддингов (бюджет)

Тариф Yandex Cloud (прямой API): **0,0101 ₽ / 1000 токенов**.

**Эталонный замер — ТК РФ** (статистика Word: 839 201 знаков с пробелами, 102 422 слова):
- Оценка объёма в токенах: ~170 000–200 000 (русский текст — в среднем 1,4 токена/слово или 1 токен на ~4 символа)
- Стоимость индексации только текста: ~2 ₽
- С учётом overlap при чанкинге (+10–15%) и эмбеддинга синтетических заголовков + 3-5 гипотетических вопросов на чанк (+20–30%): **~3–4 ₽** за полную индексацию документа

**С буфером +20% на непредвиденный рост объёма (доп. документы, повторные переиндексации, корректировки):**
- Один документ масштаба ТК РФ: **~4–5 ₽**
- Весь корпус (ТК РФ + ФЗ-181 + подзаконные акты + авторские статьи, оценочно 4–5× объёма ТК РФ): **~18–30 ₽** одноразово на полную индексацию

Query-time эмбеддинг (запрос пользователя, 10–30 токенов) — стоимость на уровне тысячных долей копейки за запрос, в бюджет не закладывается отдельной строкой.

**Важно:** это оценка только эмбеддингов. В бюджет ingestion дополнительно входит LLM для обогащения чанков (Этап 3) — тарифицируется отдельно по ценам Model Gallery (см. раздел 0.1), не входит в расчёт выше.

---

## 4. Зависимости и риски

| Риск | Влияние | Митигация |
|---|---|---|
| Корпус документов (нормативка + статьи) готовит Expert, может задержаться | Блокирует Фазу 4 (наполнение БЗ) и реальное тестирование качества поиска | Разработка и тесты на синтетических документах/запросах не блокируются; интеграция корпуса — отдельный шаг |
| Reranker-модель не финализирован | Влияет на качество поиска и latency | Зафиксировать в `config.py` как параметр, чтобы можно было заменить без переписывания pipeline |
| Embedding через внешний API (Yandex) — сетевая зависимость в hot path поиска | Риск задержки/недоступности влияет на latency `embed_query` и целевой SLA "первый токен ≤5 сек" (WBS 5.3) | Таймауты + retry с backoff, мониторинг latency через структурные логи и таблицу `search_log` (Этап 8) |
| LLM-обогащение чанков (заголовки, гипотетические вопросы) увеличивает время и стоимость ingestion | Замедляет Фазу 4 | Ingestion — офлайн процесс, не в hot path; допустимо дольше |
| Контракт `/search` должен быть согласован с MCP Tools Server до интеграции (Фаза 3.4) | Риск рассинхронизации форматов | Зафиксировать Pydantic-схемы как источник правды, согласовать с MCP Tools Server разработчиком до этапа 3.4 |
| Без версионирования документов невозможно восстановить, какая редакция закона была проиндексирована на момент конкретного ответа агента | Юридический/репутационный риск при проверке корректности старых ответов | Поля `version`/`effective_date` в метаданных чанка (раздел 3) + явный workflow обновления документа (раздел "Обновление документа", Этап 7) |
| Полная Q&A-админка (вопрос пользователя → финальный ответ Веры) не входит в зону RAG Service | PM/QA/Expert не имеют единого инструмента контроля качества ответов на уровне продукта | Зафиксировано как открытый вопрос вне рамок этого репозитория — решение и владение компонентом за Agent Service или отдельным сервисом |
| До Этапа 11 у Expert/контент-мейкера нет способа загрузить документ без ручной сборки JSON с текстом внутри строки (только `POST /ingest` напрямую) | Блокирует реальное наполнение БЗ силами не-разработчика (Фаза 4) | Загрузка файла через `/admin` — см. Этап 11.1; до реализации документ временно может загружать разработчик через `POST /ingest` |
| Плоский hybrid search ранжирует только по сходству с запросом — крупный корпус ТК РФ (`category=labor_code`) лексически ближе к типичной формулировке вопроса, чем малочисленные `case_law`/`authorial`, и может вымывать их из top-20 даже когда они юридически важны | Консультации по важным вопросам окажутся неполными — отсутствует разъяснение судебной практики или авторский комментарий о порядке применения нормы, хотя они есть в БЗ | Категорийно-сбалансированный retrieval — гарантированный пул кандидатов на каждую `category` (Этап 5.1); top-k на категорию (первая оценка — 8+8) нужно проверить эмпирически после загрузки реального корпуса с реальными "важными" вопросами |

---

## 5. Связь с другими сервисами (контракт)

```
MCP Tools Server  --HTTP POST /search-->  RAG Service
                  { query, audience?, top_k }
                  <-- { chunks: [{ text, source_title, audience, topic, score }] }
```

Контракт фиксируется в WBS п. 2.9.2 и 3.4.6 — итоговая версия документируется по факту реализации Этапа 7.

---

## 6. Соответствие WBS

| Этап плана | Пункт WBS |
|---|---|
| 1–4 (ingestion) | 3.1.1–3.1.4 |
| 5–6 (search) | 3.1.5–3.1.6 |
| 7 (API) | 3.1.7 |
| 8 (логирование запросов) | 3.1.10 |
| 9 (тесты) | 3.1.11–3.1.12, 3.1.13 (документация) |
| 10 (деплой) | 3.1.14 |
| Интеграция с MCP/Agent | 3.4.1–3.4.6 |
| Наполнение БЗ | 4.1.1–4.1.5 |
| Тестирование качества | 5.1–5.6 |

---

## 7. Доработки по итогам технического ревью (Этап 12)

> Эта таблица — не часть исходной дорожной карты (разделы 0–6 выше, Этапы 1–11). Это отдельный, дополнительно проведённый проход — внешнее техническое ревью кодовой базы по запросу, не плановый этап. Перечисляет все 47 находок ревью, их статус верификации (все Confirmed/Partially Confirmed против реального кода — ни одна не была принята на веру) и что именно сделано. ID — из исходного ревью, для трассируемости. Сгруппировано по приоритету, как в самом ревью.

### Critical

| ID | Находка | Что сделано | Файлы |
|---|---|---|---|
| ARCH-1 / API-1 / SEC-1 | Публичный API без авторизации | `X-API-Key` через `hmac.compare_digest`, на уровне роутера для `/search`,`/ingest`,`/document`; `/health` без ключа | `app/dependencies/auth.py`, `app/api/v1/endpoints/*.py` |
| SEARCH-1 / QD-3 | Клиентский BM25 блокировал event loop, O(N) от размера корпуса | Нативные sparse-векторы Qdrant с IDF вместо `rank_bm25` | `app/vectorstore/sparse.py`, `app/vectorstore/qdrant_client.py`, `app/search/hybrid.py` |
| ING-1 | Ingestion не идемпотентен (`chunk_id=uuid4()` каждый раз) | Детерминированный `chunk_id = uuid5(...)` от document_id+version+index | `app/ingestion/chunking.py` |
| SEC-6 | `.env` не исключён из `.dockerignore` | Добавлены `.env`/`.env.*`, исключение `!.env.example` | `.dockerignore` |
| TEST-2 | Нет тестов на идемпотентность/гонки ingestion | 3 интеграционных теста на реальном Qdrant+Postgres | `tests/integration/services/test_ingestion_service.py` |

### High

| ID | Находка | Что сделано | Файлы |
|---|---|---|---|
| ADM-1 / SEC-3 | Stored XSS в админке (`_fmt_json` без экранирования) | `escape(pretty)` перед вставкой в `Markup()` | `app/admin/views.py` |
| ING-2 | Нет защиты от конкурентного ingestion одного документа | Сессионный `pg_advisory_lock` вокруг `ingest_document` | `app/repositories/document.py`, `app/services/ingestion.py` |
| ING-3 | Частичный отказ удаления старых версий не компенсировался | try/except в цикле удаления + `not_removed_versions` в ответе | `app/services/ingestion.py`, `app/models/schemas.py` |
| API-4 | `/ingest` не идемпотентен на уровне контракта | Следствие ING-1, без отдельного диффа | — |
| QD-1 | Нет payload-индексов на фильтруемых полях | Индексы для `category`/`audience`/`document_id`/`version`/`topic` | `app/vectorstore/qdrant_client.py` |
| QD-2 | 6 векторов на чанк без квантизации; не было memory limit у Qdrant | int8-квантизация `chunk`-вектора + явный `deploy.resources.limits.memory` для Qdrant | `app/vectorstore/qdrant_client.py`, `docker-compose.yml` |
| TEST-1 | Интеграционные тесты репозиториев не изолированы | Очистка таблиц после `rollback()` в фикстуре | `tests/conftest.py` |
| API-2 / SEC-2 | Нет rate limiting | `slowapi`, лимиты по IP на всех публичных эндпоинтах + `/admin/login` | `app/core/rate_limit.py`, `app/main.py` |
| API-3 | `raw_text` без `max_length` | `max_length` (Pydantic) + проверка в сервисе (покрывает админку) + лимит числа чанков | `app/models/schemas.py`, `app/services/ingestion.py`, `app/exceptions/ingestion.py` |
| ARCH-4 | Два пути удаления документа дают разный результат | `DocumentsService` сам чистит реестр; админка вызывает тот же сервис | `app/services/documents.py`, `app/admin/services.py` |
| ADM-6 / ING-7 / SEC-4 | Загрузка файлов без лимита размера | Лимит размера файла (20MB) + лимит страниц PDF (2000) + проверка `Content-Length` | `app/ingestion/extract.py`, `app/admin/views.py` |
| LLM-3 / SEC-5 | Prompt injection в reranker/enrichment | XML-теги-границы + анти-injection инструкция в промптах | `app/search/reranker.py`, `app/ingestion/enrichment.py`, `app/search/prompts/reranker.py`, `app/ingestion/prompts/enrichment.py` |
| LOG-1 | Неструктурированные логи с эмодзи | JSON-форматтер (`python-json-logger`) | `logging.ini` |
| LOG-3 | `search_logs` растёт неограниченно | Окно 24ч для latency на дашборде (частично — без полного партиционирования) | `app/admin/dashboard.py` |
| LOG-4 / SEC-8 | Текст запроса логировался без redaction (152-ФЗ) | Текст запроса убран из stdout-логов, остаётся только в защищённой `search_logs` | `app/api/v1/endpoints/search.py` |
| LOG-5 | Нет метрик/алертинга | `/metrics` (Prometheus) | `app/main.py` |
| ARCH-7 | Нет CI/CD, нет конфигурации линтера | `pyproject.toml` (ruff) + GitHub Actions (lint+тесты) | `pyproject.toml`, `.github/workflows/ci.yml` |
| TEST-4 | Admin BaseView без автотестов | 8 тестов: регрессия на баг авторизации + happy-path логина | `tests/api/endpoints/test_admin_views.py` |

### Medium

| ID | Находка | Что сделано | Файлы |
|---|---|---|---|
| ING-4 | `asyncio.gather` без `return_exceptions=True` | `return_exceptions=True` + точная диагностика chunk_index; политика "всё или ничего" сохранена осознанно | `app/ingestion/enrichment.py`, `app/embeddings/embedder.py` |
| SEARCH-2 | Top-K на категорию (4+4) не валидирован | Вынесен в `Settings`; эмпирический замер — открытый пункт (нужен реальный корпус) | `app/core/settings.py`, `app/search/hybrid.py` |
| ADM-3 | Нет CSRF-токенов в формах админки | Synchronizer token pattern (токен в сессии) | `app/admin/csrf.py` |
| ADM-7 | Сессия админки без явного `https_only` | `admin_session_https_only` настройка (default `False` для локальной разработки) | `app/admin/auth.py`, `app/core/settings.py` |
| LLM-1 | Yandex-специфичные text-workaround'ы применялись ко всем провайдерам | `strip_markdown_artifacts` флаг, включён только для Yandex-клиента | `app/clients/llm.py`, `app/dependencies/clients.py` |
| LLM-2 | Нет circuit breaker между независимыми запросами | Самописный `CircuitBreaker`, module-level singleton на провайдера+use-case (3 шт.) | `app/core/circuit_breaker.py`, `app/dependencies/clients.py` |
| LLM-4 | Модель зафиксирована на нестабильном `/rc`-канале | Операционная заметка (не код) — чеклист в README | `README.md` |
| ADM-2 | Единый логин/пароль, нет аудита по актору | IP клиента логируется при удалении документа через админку | `app/admin/views.py` |
| ARCH-2 | `SearchService`/`IngestionService` — тенденция к god-service | `ingest_document` разбит на именованные шаги (без новой pipeline-абстракции — YAGNI) | `app/services/ingestion.py` |
| ARCH-6 | `httpx.AsyncClient` пересоздавался на каждый запрос | Module-level singleton, закрывается в lifespan | `app/clients/http_client.py`, `app/main.py` |
| ARCH-8 | Dockerfile: одностадийный, dev-зависимости, root, нет HEALTHCHECK | Multi-stage, non-root, HEALTHCHECK, `requirements-dev.txt` — собран и проверен реально | `Dockerfile`, `requirements.txt`, `requirements-dev.txt` |
| ARCH-9 | `--workers 1` | Поднято до 2 (после фикса SEARCH-1), конфигурируемо | `entrypoint.sh` |
| ARCH-10 | Миграции применяются на каждом старте контейнера | `RUN_MIGRATIONS_ON_START` переключатель для будущего multi-replica | `entrypoint.sh` |
| LOG-2 | Нет сквозного `request_id` | Middleware + `contextvars`, единый id в логах, `search_logs`, заголовке ответа | `app/core/request_context.py`, `app/main.py`, `app/services/search.py` |
| TEST-3 | Нет тестов производительности/масштаба | 5000-чанковый корпус, `@pytest.mark.slow`, запущен реально (~11с) | `tests/performance/test_search_scale.py` |
| ING-5 | Нет сверки Postgres-реестра и содержимого Qdrant | Сверка по запросу на дашборде (не cron — нет деплой-инфраструктуры в репо) | `app/admin/reconciliation.py`, `app/admin/dashboard.py` |
| QD-5 | Нет снапшотов/бэкапа Qdrant | Операционная заметка (не код) — чеклист в README | `README.md` |
| ADM-4 / ADM-5 | `==`-сравнение пароля, нет защиты от брутфорса | `hmac.compare_digest` + `5/minute` лимит на `/admin/login` | `app/admin/auth.py` |

### Low

| ID | Находка | Что сделано | Файлы |
|---|---|---|---|
| ING-6 | `is_active` в Qdrant — мёртвое поле | Удалено из `ChunkMetadata`/Qdrant payload (документный аудит уже есть в Postgres) | `app/models/metadata.py`, `app/vectorstore/qdrant_client.py` |
| SEARCH-3 | Промпт reranker'а без ограничения суммарной длины | Текст кандидата урезается до 600 символов | `app/search/reranker.py` |
| ARCH-3 | `Category` — захардкоженный `Literal` | Без изменений — осознанно (YAGNI), решение зафиксировано как обоснованное | — |

### Информационные (без диффа кода)

| ID | Находка | Статус |
|---|---|---|
| ARCH-5 | Замена Qdrant/провайдеров: оценка связанности | Без изменений — осознанно (YAGNI) |
| QD-4 | Сводный ответ о масштабируемости (100K/1M/10M чанков) | Покрыт фиксами SEARCH-1/QD-1/QD-2/QD-3 выше, отдельного действия не требовал |

**Итог:** 44 находки полностью реализованы, 2 — частично осознанно (LOG-3/LOG-4 — без полного партиционирования; ING-4 — без частичного успеха батча), 1 уточнена в формулировке (LLM-4 — операционная, не код), 2 информационные (без действия), 2 — операционные заметки в README, не код. Полный прогон тестов после каждого изменения — 143 обычных + 2 нагрузочных, `ruff check .` чисто.
