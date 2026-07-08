"""documents_topic_to_topics_array

Revision ID: 20260708_1510
Revises: 20260708_1500
Create Date: 2026-07-08 15:10:00.000000

`topic` был одной строкой на весь документ — для широких кодексов/законов
(labor_code/federal_law) это не имело смысла (документ регулирует десятки
разных тем), а для узких категорий (other_npa/case_law/authorial) не
позволял указать больше одной темы сразу (обсуждение с пользователем
2026-07-08). Заменено на массив названий тем — снимок на момент загрузки:
переименование темы в справочнике `topics` задним числом не отражается на
уже загруженных документах, как и остальные метаданные версии
(version/effective_date).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260708_1510'
down_revision: str | None = '20260708_1500'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column(
            'topics', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]',
            comment='Темы документа (раздел 3 плана) — пусто для labor_code/federal_law.',
        ),
    )
    op.execute("UPDATE documents SET topics = to_jsonb(ARRAY[topic]) WHERE topic IS NOT NULL AND topic != ''")
    op.alter_column('documents', 'topics', server_default=None)
    op.drop_column('documents', 'topic')


def downgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('topic', sa.String(length=100), nullable=False, server_default='', comment='Тема документа.'),
    )
    op.execute("UPDATE documents SET topic = COALESCE(topics->>0, '')")
    op.alter_column('documents', 'topic', server_default=None)
    op.drop_column('documents', 'topics')
