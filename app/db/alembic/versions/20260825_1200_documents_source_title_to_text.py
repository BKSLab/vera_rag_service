"""documents_source_title_to_text

Revision ID: 20260825_1200
Revises: 20260708_1510
Create Date: 2026-08-25 12:00:00.000000

`source_title` изначально задумывался как короткое обозначение источника
(плейсхолдер формы загрузки — «ФЗ-181, Статья 21»), и `String(255)` для этого
хватало. По решению владельца (2026-08-25) поле означает другое: полное точное
наименование документа — вид акта, дата принятия, номер и официальное название
целиком. Именно эта строка уходит в ответ `/search`, то есть агенту и конечному
пользователю как ссылка на источник, поэтому она обязана быть юридически
корректной, а не удобной по длине. У федеральных законов полное название
превышает 255 символов, поэтому ограничение снимается.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20260825_1200'
down_revision: str | None = '20260708_1510'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        'documents',
        'source_title',
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=False,
        comment='Полное официальное наименование документа: вид акта, дата, номер и название.',
        existing_comment='Человекочитаемое название источника.',
    )


def downgrade() -> None:
    # Названия длиннее 255 символов при откате не помещаются в String(255) —
    # обрезаем явно, иначе ALTER упадёт на реальных данных.
    op.execute('UPDATE documents SET source_title = left(source_title, 255)')
    op.alter_column(
        'documents',
        'source_title',
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=False,
        comment='Человекочитаемое название источника.',
        existing_comment='Полное официальное наименование документа: вид акта, дата, номер и название.',
    )
