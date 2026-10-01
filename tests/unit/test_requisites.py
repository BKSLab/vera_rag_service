from datetime import date

import pytest

from app.models.requisites import (
    allowed_act_types,
    build_act_reference,
    build_document_id,
    default_authority,
    requisites_required,
    revision_applicable,
    transliterate_number,
)


def test_act_reference_has_no_title():
    assert build_act_reference(
        act_type='Федеральный закон',
        act_number='246-ФЗ',
        act_date=date(2026, 7, 26),
    ) == 'Федеральный закон от 26.07.2026 № 246-ФЗ'


# Блок идентификатора документа


@pytest.mark.parametrize(
    ('act_number', 'expected'),
    [
        ('181-ФЗ', '181-fz'),
        ('197-ФЗ', '197-fz'),
        ('845', '845'),
        ('1234-р', '1234-r'),
        ('33н', '33n'),
        ('1-ФКЗ', '1-fkz'),
    ],
)
def test_number_is_transliterated_for_url(act_number, expected):
    assert transliterate_number(act_number) == expected


def test_document_id_is_composed_from_category_number_and_year():
    assert build_document_id('federal_law', '181-ФЗ', date(1995, 11, 24)) == 'fz-181-1995'
    assert build_document_id('labor_code', '197-ФЗ', date(2001, 12, 30)) == 'tk-197-2001'
    assert build_document_id('other_npa', '845', date(2026, 7, 4)) == 'npa-845-2026'


def test_constitutional_law_suffix_is_kept_as_distinguishing():
    """«-ФКЗ» отличает конституционный закон от обычного и убираться не должен."""

    assert build_document_id('federal_law', '1-ФКЗ', date(2020, 3, 14)) == 'fz-1-fkz-2020'
    assert build_document_id('federal_law', '1-ФЗ', date(2020, 3, 14)) == 'fz-1-2020'


def test_same_number_in_different_years_gives_different_ids():
    """Номер акта не уникален: 51-ФЗ — это и ГК РФ 1994 года, и ФЗ 2001 года."""

    civil_code = build_document_id('labor_code', '51-ФЗ', date(1994, 11, 30))
    federal_law = build_document_id('federal_law', '51-ФЗ', date(2001, 5, 14))

    assert civil_code != federal_law


def test_document_id_is_stable_for_same_requisites():
    """На document_id завязан реестр Legal Sync — он обязан быть воспроизводимым."""

    first = build_document_id('federal_law', '181-ФЗ', date(1995, 11, 24))
    second = build_document_id('federal_law', '181-ФЗ', date(1995, 11, 24))

    assert first == second


# Блок правил применимости


def test_labor_code_has_single_act_type():
    assert allowed_act_types('labor_code') == ('Кодекс Российской Федерации',)


def test_authorial_has_no_act_types():
    assert allowed_act_types('authorial') == ()


def test_requisites_required_for_npa_and_case_law():
    assert requisites_required('labor_code') is True
    assert requisites_required('federal_law') is True
    assert requisites_required('other_npa') is True
    assert requisites_required('case_law') is True
    assert requisites_required('authorial') is False


def test_revision_applies_only_to_npa():
    """Постановление Пленума не меняется актом той же силы, оно заменяется новым."""

    assert revision_applicable('labor_code') is True
    assert revision_applicable('other_npa') is True
    assert revision_applicable('case_law') is False
    assert revision_applicable('authorial') is False


def test_authority_defaults_follow_portal_wording():
    assert default_authority('Федеральный закон') == 'Президент Российской Федерации'
    assert default_authority('Постановление Правительства Российской Федерации') == (
        'Правительство Российской Федерации'
    )


def test_order_authority_has_no_default():
    """Приказ издаёт конкретное министерство — угадывать его нельзя."""

    assert default_authority('Приказ') is None
