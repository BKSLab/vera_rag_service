"""rename_search_logs_source_type_to_category

Revision ID: 3f6a1c9e2b7d
Revises: 8d5e42d7f9bb
Create Date: 2026-06-20 22:10:00.000000

Колонка `search_logs.source_type` была переименована в `category` в коде
(`app/db/models/search_log.py`) при Этапе 5.1 правкой уже применённой
миграции `4888a7c516e0` "на месте" — ошибочно, так как та миграция уже
была выполнена на рабочей БД до правки. Alembic не переигрывает старые
миграции, поэтому реальная колонка осталась `source_type` — обнаружено
по живой ошибке `UndefinedColumnError` при записи `search_logs` через
admin (`/admin/search-test`). Эта миграция приводит фактическую схему
в соответствие модели через настоящий `ALTER TABLE`, а не правку файла.
"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '3f6a1c9e2b7d'
down_revision: str | None = '8d5e42d7f9bb'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column('search_logs', 'source_type', new_column_name='category')


def downgrade() -> None:
    op.alter_column('search_logs', 'category', new_column_name='source_type')
