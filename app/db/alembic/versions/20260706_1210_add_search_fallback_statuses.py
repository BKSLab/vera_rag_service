"""add_search_fallback_statuses

Revision ID: 20260706_1210
Revises: 20260706_1200
Create Date: 2026-07-06 12:10:00.000000

Статусы fallback-режимов hot path поиска: отдельно фиксируем, был ли
query expansion выполнен штатно и как завершился reranker. Это позволяет
отличить обычный пустой результат от технической деградации LLM.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20260706_1210'
down_revision: str | None = '20260706_1200'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'search_logs',
        sa.Column(
            'query_expansion_status',
            sa.String(length=40),
            nullable=False,
            server_default='ok',
            comment='Статус расширения запроса: ok или fallback_unavailable.',
        ),
    )
    op.alter_column('search_logs', 'query_expansion_status', server_default=None)
    op.add_column(
        'search_logs',
        sa.Column(
            'reranker_status',
            sa.String(length=40),
            nullable=False,
            server_default='ok',
            comment='Статус reranker: ok/no_candidates/no_relevant/fallback_unavailable/fallback_invalid_output.',
        ),
    )
    op.alter_column('search_logs', 'reranker_status', server_default=None)


def downgrade() -> None:
    op.drop_column('search_logs', 'reranker_status')
    op.drop_column('search_logs', 'query_expansion_status')
