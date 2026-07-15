"""add_search_logs_table

Revision ID: 4888a7c516e0
Revises: 
Create Date: 2026-06-20 08:07:57.429626

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4888a7c516e0'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'search_logs',
        sa.Column('id', sa.Integer(), nullable=False, comment='Уникальный идентификатор записи журнала.'),
        sa.Column('request_id', sa.String(length=36), nullable=False, comment='UUID конкретного запроса — для сопоставления со структурными логами приложения.'),
        sa.Column('query', sa.Text(), nullable=False, comment='Исходный текст поискового запроса.'),
        sa.Column('audience', sa.String(length=20), nullable=True, comment='Значение фильтра audience, если был задан.'),
        sa.Column('topic', sa.String(length=100), nullable=True, comment='Значение фильтра topic, если был задан.'),
        sa.Column('source_type', sa.String(length=20), nullable=True, comment='Значение фильтра source_type, если был задан.'),
        sa.Column('dense_candidates', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Top-20 кандидатов dense-поиска до фьюжна: [[chunk_id, score], ...].'),
        sa.Column('sparse_candidates', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Top-20 кандидатов sparse/BM25-поиска до фьюжна: [[chunk_id, score], ...].'),
        sa.Column('rrf_candidates', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Результат RRF fusion: [[chunk_id, score], ...].'),
        sa.Column('reranked_chunk_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='chunk_id в порядке, который вернул LLM-reranker.'),
        sa.Column('final_response', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Финальный список чанков, отданный клиенту /search.'),
        sa.Column('latency_embed_query_ms', sa.Float(), nullable=False, comment='Латентность стадии embed_query, мс.'),
        sa.Column('latency_hybrid_search_ms', sa.Float(), nullable=False, comment='Латентность стадии hybrid_search (dense+sparse+RRF), мс.'),
        sa.Column('latency_rerank_ms', sa.Float(), nullable=False, comment='Латентность стадии rerank, мс.'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент выполнения поискового запроса.'),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('search_logs')
