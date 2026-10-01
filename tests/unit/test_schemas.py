from datetime import date

import pytest
from pydantic import ValidationError

from app.models.schemas import DocumentMetadataInput, IngestRequest, SectionUpdateRequest


def _document_metadata(version: str) -> DocumentMetadataInput:
    return DocumentMetadataInput(
        source_title='Источник',
        audience='both',
        topics=[],
        version=version,
        effective_date=date(2026, 7, 18),
    )


def _ingest_request(revision_date: date) -> IngestRequest:
    return IngestRequest(
        category='labor_code',
        act_type='Кодекс Российской Федерации',
        act_number='197-ФЗ',
        act_date=date(2001, 12, 30),
        act_title='Трудовой кодекс Российской Федерации',
        source_title='Трудовой кодекс Российской Федерации',
        revision_date=revision_date,
        raw_text='Текст документа.',
        audience='both',
        topics=[],
    )


def _section_update_request(revision_date: date) -> SectionUpdateRequest:
    return SectionUpdateRequest(
        category='labor_code',
        raw_text='Текст секции.',
        section_title='Статья 1',
        revision_date=revision_date,
        source_title='Источник',
        audience='both',
        topics=[],
    )


def test_document_metadata_version_rejects_invalid_iso_date():
    with pytest.raises(ValidationError):
        _document_metadata('2026-077-18')


def test_document_metadata_version_normalizes_compact_iso_date():
    assert _document_metadata('20260718').version == '2026-07-18'


@pytest.mark.parametrize('factory', [_ingest_request, _section_update_request])
def test_version_is_derived_from_revision_date(factory):
    """Версия больше не вводится: она выводится из даты редакции."""

    model = factory(date(2026, 7, 18))

    assert model.version == '2026-07-18'
