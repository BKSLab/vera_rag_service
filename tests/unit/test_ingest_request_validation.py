from datetime import date

import pytest
from pydantic import ValidationError

from app.models.schemas import IngestRequest


def _request(**overrides) -> IngestRequest:
    payload = {
        'category': 'federal_law',
        'act_type': 'Федеральный закон',
        'act_number': '181-ФЗ',
        'act_date': date(1995, 11, 24),
        'act_title': 'О социальной защите инвалидов в Российской Федерации',
        'source_title': 'Федеральный закон «О социальной защите инвалидов в Российской Федерации» от 24 ноября 1995 N 181-ФЗ',
        'revision_date': date(2026, 5, 25),
        'raw_text': 'Статья 1. Текст.',
        'audience': 'both',
        'topics': [],
    }
    payload.update(overrides)
    return IngestRequest(**payload)


# Блок выводимых полей


def test_identifier_is_derived_from_requisites_and_title_is_kept_as_given():
    """Идентификатор выводится, наименование — нет.

    Наименование задаёт оператор и оно уходит в промпт обогащения, то есть в
    эмбеддинги: собранная из реквизитов строка отличалась бы от принятого
    написания (кавычки, дата словами, «N») и меняла бы вход пайплайна,
    настроенного бенчмарками.
    """
    request = _request()

    assert request.document_id == 'fz-181-1995'
    assert request.source_title == (
        'Федеральный закон «О социальной защите инвалидов в Российской Федерации» '
        'от 24 ноября 1995 N 181-ФЗ'
    )
    assert request.version == '2026-05-25'


def test_authority_is_filled_from_act_type():
    assert _request().act_authority == 'Президент Российской Федерации'


def test_explicit_authority_is_kept():
    assert _request(act_authority='Минтруд России').act_authority == 'Минтруд России'


def test_content_date_falls_back_to_act_date_without_revision():
    """У судебной практики редакций не бывает — датируем самим документом."""

    request = _request(
        category='case_law',
        act_type='Постановление Пленума Верховного Суда Российской Федерации',
        act_number='15',
        act_date=date(2018, 5, 29),
        act_title='О применении судами законодательства о труде',
        revision_date=None,
        topics=['трудовые споры'],
    )

    assert request.content_date == date(2018, 5, 29)
    assert request.version == '2018-05-29'


# Блок валидации категории и реквизитов


def test_number_without_fz_suffix_is_rejected_for_federal_law():
    """Простая проверка, которая ловит перепутанную категорию на вводе."""

    with pytest.raises(ValidationError, match='ФЗ'):
        _request(act_number='845')


def test_government_decree_number_is_not_constrained():
    request = _request(
        category='other_npa',
        act_type='Постановление Правительства Российской Федерации',
        act_number='845',
        act_date=date(2026, 7, 4),
        act_title='О некоторых вопросах',
        revision_date=date(2026, 7, 4),
    )

    assert request.document_id == 'npa-845-2026'


def test_act_type_must_match_category():
    with pytest.raises(ValidationError, match='недопустим для категории'):
        _request(category='other_npa', act_type='Федеральный закон', act_number='845')


def test_missing_act_type_is_rejected_for_npa():
    with pytest.raises(ValidationError, match='обязателен вид акта'):
        _request(act_type=None)


def test_missing_number_is_rejected_for_npa():
    with pytest.raises(ValidationError, match='обязателен номер акта'):
        _request(act_number=None)


# Блок даты редакции


def test_revision_date_is_required_for_npa():
    with pytest.raises(ValidationError, match='дата действующей редакции'):
        _request(revision_date=None)


def test_revision_date_cannot_precede_the_act_itself():
    with pytest.raises(ValidationError, match='раньше даты самого акта'):
        _request(revision_date=date(1990, 1, 1))


def test_revision_date_is_rejected_where_it_does_not_apply():
    """Постановление Пленума не изменяется, а заменяется новым."""

    with pytest.raises(ValidationError, match='не применяется'):
        _request(
            category='case_law',
            act_type='Обзор судебной практики',
            act_number='1',
            revision_date=date(2026, 5, 25),
        )


# Блок авторских материалов


def test_authorial_material_needs_explicit_identifier():
    with pytest.raises(ValidationError, match='задайте document_id явно'):
        _request(
            category='authorial',
            act_type=None,
            act_number=None,
            revision_date=None,
            act_title='Как оформить отпуск',
        )


def test_authorial_material_accepts_explicit_identifier():
    request = _request(
        category='authorial',
        act_type=None,
        act_number=None,
        revision_date=None,
        act_title='Как оформить отпуск',
        source_title='Как оформить отпуск',
        document_id='art-otpusk-2026',
        topics=['отпуска'],
    )

    assert request.document_id == 'art-otpusk-2026'
    assert request.source_title == 'Как оформить отпуск'
