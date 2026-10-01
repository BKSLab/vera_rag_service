"""act_requisites

Revision ID: 20260907_1300
Revises: 20260907_1200
Create Date: 2026-09-07 13:00:00.000000

Наименование документа переезжает со свободной строки на разобранные
реквизиты: вид акта, номер, дата, название, принявший орган. Собрать строку
из реквизитов можно всегда, разобрать обратно — нет, поэтому `source_title`
теперь выводится из них, а не вводится руками.

`documents.effective_date` переименована в `revision_date`: поле означает
«действующая редакция акта от», а не дату вступления в силу отдельной статьи.

По решению владельца база переиндексируется целиком, поэтому старые строки
реестра не переносятся: они описывают документы наименованием одной строкой,
и представить их в реквизитах можно было бы только выдумав дату подписания
акта. Строки удаляются, реестр наполняется заново при переиндексации — это
позволяет объявить `act_date` и `act_title` NOT NULL честно, без
`server_default`, который иначе остался бы на колонке навсегда и позволял бы
молча записать пустое наименование.

Необязательные реквизиты (`act_type`, `act_number`, `act_authority`,
`revision_date`) остаются nullable: у авторских материалов их нет, у судебной
практики нет редакции. Обязательность по категориям обеспечивается на уровне
API (`ActRequisitesMixin`), а не ограничениями БД.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20260907_1300'
down_revision: str | None = '20260907_1200'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Реестр наполняется заново переиндексацией: старые строки не описать
    # реквизитами, не выдумав их. Чанки Qdrant пересоздаются тем же прогоном.
    op.execute('DELETE FROM documents')

    op.add_column(
        'documents',
        sa.Column('act_type', sa.String(length=120), nullable=True, comment='Вид акта; отсутствует у авторских материалов.'),
    )
    op.add_column(
        'documents',
        sa.Column('act_number', sa.String(length=100), nullable=True, comment='Номер акта; отсутствует у авторских материалов.'),
    )
    op.add_column(
        'documents',
        sa.Column(
            'act_date',
            sa.Date(),
            nullable=False,
            comment='Дата акта: подписания для правовых актов, публикации для авторских.',
        ),
    )
    op.add_column(
        'documents',
        sa.Column('act_title', sa.Text(), nullable=False, comment='Наименование акта без реквизитов.'),
    )
    op.add_column(
        'documents',
        sa.Column('act_authority', sa.String(length=255), nullable=True, comment='Принявший (подписавший) орган.'),
    )
    op.alter_column(
        'documents',
        'effective_date',
        new_column_name='revision_date',
        existing_type=sa.Date(),
        nullable=True,
        comment=(
            'Дата действующей редакции акта на момент этой загрузки. Постатейные обновления '
            'её не двигают: строка реестра описывает состоявшуюся загрузку, а что менялось '
            'потом — отвечает document_change_log. Не применяется к судебной практике и '
            'авторским материалам.'
        ),
    )


def downgrade() -> None:
    op.alter_column(
        'documents',
        'revision_date',
        new_column_name='effective_date',
        existing_type=sa.Date(),
        nullable=False,
        comment='Дата вступления редакции в силу.',
    )
    op.drop_column('documents', 'act_authority')
    op.drop_column('documents', 'act_title')
    op.drop_column('documents', 'act_date')
    op.drop_column('documents', 'act_number')
    op.drop_column('documents', 'act_type')
