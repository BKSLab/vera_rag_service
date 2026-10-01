"""add_document_change_log

Revision ID: 20260907_1200
Revises: 20260825_1200
Create Date: 2026-09-07 12:00:00.000000

Журнал изменений документов базы знаний. Таблица `documents` отвечает на
вопрос «какие версии документа есть сейчас», но не на вопрос «что и когда в
нём поменялось». Для юридической БЗ второй вопрос основной: по нему видно,
какие статьи были переиндексированы, какой редакцией и с какой даты она
действует.

Записи добавляются при каждой попытке гранулярного обновления статьи через
`PUT /document/{id}/sections/{number}` — в том числе автоматической из Legal
Sync Service. Отклонённые и упавшие попытки пишутся наравне с применёнными:
журнал, показывающий только удачи, не даёт контроля — пустая строка в нём
неотличима от «сервис синхронизации ничего не присылал».
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20260907_1200'
down_revision: str | None = '20260825_1200'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'document_change_log',
        sa.Column('id', sa.Integer(), nullable=False, comment='Уникальный идентификатор записи журнала.'),
        sa.Column('document_id', sa.String(length=255), nullable=False, comment='Идентификатор изменённого документа.'),
        sa.Column('section_number', sa.String(length=100), nullable=False, comment='Номер переиндексированной статьи или пункта.'),
        sa.Column('section_title', sa.Text(), nullable=True, comment='Заголовок статьи на момент изменения.'),
        sa.Column('category', sa.String(length=20), nullable=False, comment='Категория источника на момент изменения.'),
        sa.Column('version', sa.String(length=20), nullable=False, comment='Версия, под которой проиндексирована новая редакция.'),
        sa.Column('revision_date', sa.Date(), nullable=False, comment='Дата редакции, из которой взят новый текст статьи.'),
        sa.Column(
            'status',
            sa.String(length=20),
            nullable=False,
            comment='Исход попытки: applied | rejected | failed.',
        ),
        sa.Column('error', sa.Text(), nullable=True, comment='Причина отказа; пусто для применённых изменений.'),
        sa.Column('amending_act_type', sa.String(length=120), nullable=True, comment='Вид акта, которым внесены изменения.'),
        sa.Column('amending_act_number', sa.String(length=100), nullable=True, comment='Номер акта, которым внесены изменения.'),
        sa.Column('amending_act_date', sa.Date(), nullable=True, comment='Дата акта, которым внесены изменения.'),
        sa.Column('chunks_count', sa.Integer(), nullable=False, server_default='0', comment='Сколько чанков создано для новой редакции.'),
        sa.Column('superseded_chunks', sa.Integer(), nullable=False, server_default='0', comment='Сколько чанков прошлой редакции помечено неактуальными.'),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
            comment='Момент, когда изменение было применено к базе знаний.',
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_document_change_log_document_id', 'document_change_log', ['document_id'])
    op.create_index(
        'ix_document_change_log_document_id_created_at',
        'document_change_log',
        ['document_id', 'created_at'],
    )
    # Дашборд считает отказы за сутки — без этого индекса счётчик означал бы
    # seq scan по всему журналу при каждом открытии страницы.
    op.create_index(
        'ix_document_change_log_status_created_at',
        'document_change_log',
        ['status', 'created_at'],
    )


def downgrade() -> None:
    op.drop_index('ix_document_change_log_status_created_at', table_name='document_change_log')
    op.drop_index('ix_document_change_log_document_id_created_at', table_name='document_change_log')
    op.drop_index('ix_document_change_log_document_id', table_name='document_change_log')
    op.drop_table('document_change_log')
