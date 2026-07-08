"""add_topics_table

Revision ID: 20260708_1500
Revises: 20260706_1210
Create Date: 2026-07-08 15:00:00.000000

Справочник тем (раздел 3 плана) — отдельная таблица вместо хардкода в
Python (как Category/Audience), потому что список тем должен пополняться
контент-менеджером через админку без деплоя кода. Осмысленны только для
other_npa/case_law/authorial (см. TopicsNotAllowedForCategoryError) —
начальный набор отражает домен сервиса (трудовые права людей с
инвалидностью), задан пользователем 2026-07-08.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20260708_1500'
down_revision: str | None = '20260706_1210'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INITIAL_TOPICS = [
    'квотирование',
    'трудовые права',
    'приём на работу',
    'увольнение',
    'трудоустройство',
    'рабочее место инвалидов',
    'охрана труда',
    'дискриминация',
]


def upgrade() -> None:
    topics_table = op.create_table(
        'topics',
        sa.Column('id', sa.Integer(), nullable=False, comment='Уникальный идентификатор темы.'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='Название темы (например, "квотирование").'),
        sa.Column('comment', sa.String(length=255), nullable=True, comment='Пояснение для админки — не выводится в API/поиске.'),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False,
            comment='Момент добавления темы в справочник.',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    op.bulk_insert(topics_table, [{'name': name} for name in _INITIAL_TOPICS])


def downgrade() -> None:
    op.drop_table('topics')
