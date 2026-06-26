"""add_query_expansion_to_search_logs

Revision ID: 9a1d4e8c5f02
Revises: 3f6a1c9e2b7d
Create Date: 2026-06-23 12:00:00.000000

Расширение запроса перед поиском (декомпозиция составного вопроса на
подвопросы + переформулировка каждого ближе к терминологии трудового
права, раздел 8 плана) — добавляет `query_variants` (тексты вариантов
запроса, каждый прошёл свой hybrid_search) и `latency_query_expansion_ms`
(латентность этой новой стадии) в журнал `search_logs`.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '9a1d4e8c5f02'
down_revision: str | None = '3f6a1c9e2b7d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'search_logs',
        sa.Column(
            'query_variants', JSONB(), nullable=False, server_default='[]',
            comment='Подвопросы/переформулировки после расширения запроса: список текстов.',
        ),
    )
    op.alter_column('search_logs', 'query_variants', server_default=None)
    op.add_column(
        'search_logs',
        sa.Column(
            'latency_query_expansion_ms', sa.Float(), nullable=False, server_default='0',
            comment='Латентность стадии расширения запроса, мс.',
        ),
    )
    op.alter_column('search_logs', 'latency_query_expansion_ms', server_default=None)


def downgrade() -> None:
    op.drop_column('search_logs', 'latency_query_expansion_ms')
    op.drop_column('search_logs', 'query_variants')
