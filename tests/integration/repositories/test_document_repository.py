from datetime import date

from sqlalchemy import func, select

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
        topics=['quota'],
        act_type='Федеральный закон',
        act_number='181-ФЗ',
        act_date=date(1995, 11, 24),
        act_title='О социальной защите инвалидов в Российской Федерации',
        act_authority='Президент Российской Федерации',
        revision_date=date(2026, 1, 1),
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
        source_title='ФЗ-181', audience='both', topics=['quota'],
        act_type='Федеральный закон', act_number='181-ФЗ', act_date=date(1995, 11, 24),
        act_title='О социальной защите инвалидов', act_authority='Президент Российской Федерации',
        revision_date=date(2025, 1, 1), is_active=True,
    )
    new_version = Document(
        document_id='fz-181-art21', version='2026-01-01', category='federal_law',
        source_title='ФЗ-181', audience='both', topics=['quota'],
        act_type='Федеральный закон', act_number='181-ФЗ', act_date=date(1995, 11, 24),
        act_title='О социальной защите инвалидов', act_authority='Президент Российской Федерации',
        revision_date=date(2026, 1, 1), is_active=True,
    )
    await repository.save_document(old_version)
    await repository.save_document(new_version)

    await repository.mark_versions_inactive('fz-181-art21', ['2025-01-01'])

    rows = (
        await db_session.execute(select(Document).where(Document.document_id == 'fz-181-art21'))
    ).scalars().all()
    is_active_by_version = {row.version: row.is_active for row in rows}
    assert is_active_by_version == {'2025-01-01': False, '2026-01-01': True}


async def test_save_document_upserts_same_document_version(db_session):
    repository = DocumentRepository(db_session)
    await repository.save_document(
        Document(
            document_id='fz-181-art21',
            version='2026-01-01',
            category='federal_law',
            source_title='Старое название',
            audience='both',
            topics=['quota'],
            act_type='Федеральный закон',
            act_number='181-ФЗ',
            act_date=date(1995, 11, 24),
            act_title='Старое название',
            act_authority='Президент Российской Федерации',
            revision_date=date(2026, 1, 1),
            is_active=False,
        )
    )

    await repository.save_document(
        Document(
            document_id='fz-181-art21',
            version='2026-01-01',
            category='labor_code',
            source_title='Новое название',
            audience='employer',
            topics=['workplace'],
            act_type='Кодекс Российской Федерации',
            act_number='197-ФЗ',
            act_date=date(2001, 12, 30),
            act_title='Новое название',
            act_authority='Президент Российской Федерации',
            revision_date=date(2026, 2, 1),
            is_active=True,
        )
    )

    rows_count = (
        await db_session.execute(
            select(func.count()).select_from(Document).where(Document.document_id == 'fz-181-art21')
        )
    ).scalar_one()
    saved = (
        await db_session.execute(select(Document).where(Document.document_id == 'fz-181-art21'))
    ).scalar_one()

    assert rows_count == 1
    assert saved.category == 'labor_code'
    assert saved.source_title == 'Новое название'
    assert saved.act_title == 'Новое название'
    assert saved.act_number == '197-ФЗ'
    assert saved.audience == 'employer'
    assert saved.topics == ['workplace']
    assert saved.revision_date == date(2026, 2, 1)
    assert saved.is_active is True
