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


def _ingest_request(version: str) -> IngestRequest:
    return IngestRequest(
        document_id='doc-1',
        category='labor_code',
        raw_text='Текст документа.',
        source_title='Источник',
        audience='both',
        topics=[],
        version=version,
        effective_date=date(2026, 7, 18),
    )


def _section_update_request(version: str) -> SectionUpdateRequest:
    return SectionUpdateRequest(
        category='labor_code',
        raw_text='Текст секции.',
        section_title='Статья 1',
        version=version,
        effective_date=date(2026, 7, 18),
        source_title='Источник',
        audience='both',
        topics=[],
    )


@pytest.mark.parametrize(
    'factory',
    [_document_metadata, _ingest_request, _section_update_request],
)
def test_version_rejects_invalid_iso_date(factory):
    with pytest.raises(ValidationError):
        factory('2026-077-18')


@pytest.mark.parametrize(
    'factory',
    [_document_metadata, _ingest_request, _section_update_request],
)
def test_version_normalizes_compact_iso_date(factory):
    model = factory('20260718')

    assert model.version == '2026-07-18'
