FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# gcc/libpq-dev нужны только для компиляции зависимостей (asyncpg и т.п.)
# на этапе pip install — не нужны в runtime-образе (ARCH-8,
# AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md).
RUN apt-get update \
    && apt-get -y install --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Только runtime-зависимости (tzdata, curl для HEALTHCHECK) — без
# build-тулчейна (gcc/libpq-dev) и без dev-зависимостей (pytest/ruff/
# testcontainers, см. requirements-dev.txt) — меньше образ, меньше
# поверхность атаки при компрометации процесса.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata curl && \
    ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    echo "Europe/Moscow" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Непривилегированный пользователь — нарушение принципа наименьших
# привилегий устранено (ARCH-8): при компрометации процесса атакующий не
# получает root внутри контейнера.
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
