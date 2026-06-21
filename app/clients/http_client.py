import httpx

# ARCH-6 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — общий
# `httpx.AsyncClient` для внешних API (Yandex/Polza), не пересоздаётся на
# каждый запрос. Каждый `/search` делает минимум 2 внешних HTTPS-вызова
# (embedding + reranker) — пересоздание клиента означало бы новое TCP+TLS
# соединение на каждый из них, систематическая надбавка к latency hot path.
# Жизненный цикл — как у `qdrant_client`/`engine` (app/vectorstore/client.py,
# app/db/session.py): модуль-level singleton, закрывается в lifespan.
external_api_http_client = httpx.AsyncClient()
