from datetime import date

from sqlalchemy import select

from app.db.models.document import Document
from app.repositories.document import DocumentRepository


async def test_save_document_persists_all_fields(db_session):
    repository = DocumentRepository(db_session)
    document = Document(
        document_id='fz-181-art21',
        version='2026-01-01',
        category='federal_law',
        source_title='ФЗ-181, Статья 21',
        audience='both',
        topic='quota',
        effective_date=date(2026, 1, 1),
        is_active=True,
    )

    await repository.save_document(document)

    saved = (
        await db_session.execute(select(Document).where(Document.document_id == 'fz-181-art21'))
    ).scalar_one()
    assert saved.version == '2026-01-01'
    assert saved.category == 'federal_law'
    assert saved.is_active is True
    assert saved.created_at is not None


async def test_mark_versions_inactive_updates_only_given_versions(db_session):
    repository = DocumentRepository(db_session)
    old_version = Document(
        document_id='fz-181-art21', version='2025-01-01', category='federal_law',
        source_title='ФЗ-181', audience='both', topic='quota', effective_date=date(2025, 1, 1), is_active=True,
    )
    new_version = Document(
        document_id='fz-181-art21', version='2026-01-01', category='federal_law',
        source_title='ФЗ-181', audience='both', topic='quota', effective_date=date(2026, 1, 1), is_active=True,
    )
    await repository.save_document(old_version)
    await repository.save_document(new_version)

    await repository.mark_versions_inactive('fz-181-art21', ['2025-01-01'])

    rows = (
        await db_session.execute(select(Document).where(Document.document_id == 'fz-181-art21'))
    ).scalars().all()
    is_active_by_version = {row.version: row.is_active for row in rows}
    assert is_active_by_version == {'2025-01-01': False, '2026-01-01': True}
