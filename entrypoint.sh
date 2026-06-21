#!/bin/bash

set -e

# Добавляем в PYTHONPATH и корневую директорию проекта (/app),
# и директорию с исходным кодом (/app/app).
# Это позволяет и команде запуска найти app.main, и самому приложению
# найти свои внутренние модули (api, core и т.д.).
export PYTHONPATH="/app:/app/app:${PYTHONPATH}"

# ARCH-10 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — при единственном
# экземпляре (текущая конфигурация docker-compose.yml, без deploy.replicas)
# применять миграции на каждом старте контейнера безопасно. При переходе на
# несколько реплик одновременный запуск `alembic upgrade head` несколькими
# контейнерами против одной БД не гарантированно безопасен для всех типов
# миграций — тогда нужно вынести этот шаг в отдельный, однократно
# запускаемый шаг деплоя (CI/CD-пайплайн, init-container, ручной
# `docker-compose run --rm rag_service alembic upgrade head` перед
# `--scale`) и установить RUN_MIGRATIONS_ON_START=false здесь.
if [ "${RUN_MIGRATIONS_ON_START:-true}" = "true" ]; then
    echo "application of migrations"
    alembic upgrade head
else
    echo "RUN_MIGRATIONS_ON_START=false — миграции пропущены (применяются отдельным шагом деплоя)"
fi

# ARCH-9 — поднято с 1 после устранения SEARCH-1 (синхронный BM25,
# блокировавший единственный воркер на каждый поисковый запрос без явного
# category-фильтра, теперь — нативные sparse-векторы Qdrant, обычный
# индексный запрос). Дальнейшее увеличение/переход на несколько реплик —
# после нагрузочного замера (TEST-3), не наугад.
echo "Starting in production mode..."
exec hypercorn app.main:app --bind 0.0.0.0:8000 --workers "${HYPERCORN_WORKERS:-2}"