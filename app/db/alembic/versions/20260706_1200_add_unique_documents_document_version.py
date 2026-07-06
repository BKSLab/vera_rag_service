"""add_unique_documents_document_version

Revision ID: 20260706_1200
Revises: 9a1d4e8c5f02
Create Date: 2026-07-06 12:00:00.000000

Идемпотентный реестр документов: одна строка на пару
`(document_id, version)`. Qdrant уже перезаписывает чанки той же версии
по deterministic `chunk_id`; Postgres-реестр должен вести себя так же,
а не плодить дубли при повторной загрузке.
"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20260706_1200'
down_revision: str | None = '9a1d4e8c5f02'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM documents d
        USING documents newer
        WHERE d.document_id = newer.document_id
          AND d.version = newer.version
          AND d.id < newer.id
        """
    )
    op.create_unique_constraint(
        'uq_documents_document_id_version',
        'documents',
        ['document_id', 'version'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_documents_document_id_version', 'documents', type_='unique')
