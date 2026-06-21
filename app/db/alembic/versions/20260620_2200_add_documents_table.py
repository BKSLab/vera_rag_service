"""add_documents_table

Revision ID: 8d5e42d7f9bb
Revises: 4888a7c516e0
Create Date: 2026-06-20 22:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '8d5e42d7f9bb'
down_revision: str | None = '4888a7c516e0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False, comment='Уникальный идентификатор записи реестра.'),
        sa.Column('document_id', sa.String(length=255), nullable=False, comment='Идентификатор документа-источника.'),
        sa.Column('version', sa.String(length=20), nullable=False, comment='Версия документа, под которой проиндексирован.'),
        sa.Column('category', sa.String(length=20), nullable=False, comment='Категория источника (раздел 3 плана).'),
        sa.Column('source_title', sa.String(length=255), nullable=False, comment='Человекочитаемое название источника.'),
        sa.Column('audience', sa.String(length=20), nullable=False, comment='Целевая аудитория документа.'),
        sa.Column('topic', sa.String(length=100), nullable=False, comment='Тема документа.'),
        sa.Column('effective_date', sa.Date(), nullable=False, comment='Дата вступления редакции в силу.'),
        sa.Column('is_active', sa.Boolean(), nullable=False, comment='Признак активной (актуальной) версии — неактивные хранятся для аудита.'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент успешного ingestion этой версии.'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_documents_document_id'), 'documents', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_documents_document_id'), table_name='documents')
    op.drop_table('documents')
