from types import SimpleNamespace

from starlette.datastructures import FormData

from app.admin.views import _fmt_json, build_ingest_request


def test_fmt_json_escapes_html_in_value():
    """Регрессия на stored XSS (ADM-1/SEC-3) — содержимое из реальных
    документов/LLM-вывода может содержать `<script>` и т.п., и не должно
    попадать в HTML-страницу админки неэкранированным."""
    model = SimpleNamespace(final_response=[{'text': '<script>alert(1)</script>'}])

    rendered = _fmt_json(model, 'final_response')

    assert '<script>' not in str(rendered)
    assert '&lt;script&gt;' in str(rendered)


def _submitted_form(**fields: str | list[str]) -> FormData:
    """Собирает тело формы так, как его отправляет браузер.

    Скрытые для выбранной категории поля карточка выключает, а выключенные
    контролы браузер в запрос не кладёт вовсе — поэтому в теле нет пустых
    значений, есть отсутствующие ключи. Именно на этом ломались загрузки:
    сервер получал `None` там, где реквизит обязателен.
    """
    items: list[tuple[str, str]] = []
    for name, value in fields.items():
        if isinstance(value, list):
            items.extend((name, item) for item in value)
        else:
            items.append((name, value))
    return FormData(items)


def test_labor_code_upload_carries_its_single_act_type():
    """Регрессия: вид акта у кодекса единственный, но отправлен быть обязан.

    Карточка не должна выключать `select` с одним вариантом — иначе поле не
    доходит до сервера и загрузка ТК РФ падает на «обязателен вид акта».
    """
    form = _submitted_form(
        category='labor_code',
        act_type='Кодекс Российской Федерации',
        act_number='197-ФЗ',
        act_date='2001-12-30',
        act_title='Трудовой кодекс Российской Федерации',
        source_title='Трудовой кодекс Российской Федерации',
        act_authority='',
        revision_date='2026-05-25',
        audience='both',
    )

    request = build_ingest_request(form, 'labor_code', raw_text='Статья 1. Цели.')

    assert request.document_id == 'tk-197-2001'
    assert request.act_type == 'Кодекс Российской Федерации'
    assert request.act_authority == 'Президент Российской Федерации'
    assert request.version == '2026-05-25'


def test_authorial_material_upload_carries_explicit_identifier():
    """Регрессия: у авторского материала нет номера акта, из которого
    выводится идентификатор, поэтому карточка обязана дать его задать."""
    form = _submitted_form(
        category='authorial',
        act_date='2026-09-01',
        act_title='Как оформить отпуск без сохранения заработной платы',
        source_title='Как оформить отпуск без сохранения заработной платы',
        document_id='art-otpusk-2026',
        audience='seeker',
        topics=['Отпуска'],
    )

    request = build_ingest_request(form, 'authorial', raw_text='Текст материала.')

    assert request.document_id == 'art-otpusk-2026'
    assert request.act_type is None
    assert request.act_number is None
    assert request.revision_date is None
    assert request.version == '2026-09-01'


def test_case_law_upload_goes_without_revision_date():
    """Постановление Пленума не изменяется редакциями: карточка это поле
    прячет, и пустое значение не должно превращаться в ошибку типа."""
    form = _submitted_form(
        category='case_law',
        act_type='Постановление Пленума Верховного Суда Российской Федерации',
        act_number='15',
        act_date='2018-05-29',
        act_title='О применении судами законодательства, регулирующего труд работников',
        source_title='Постановление Пленума Верховного Суда Российской Федерации от 29 мая 2018 N 15',
        act_authority='',
        audience='both',
    )

    request = build_ingest_request(form, 'case_law', raw_text='Пункт 1.')

    assert request.document_id == 'sud-15-2018'
    assert request.revision_date is None
    assert request.act_authority == 'Верховный Суд Российской Федерации'


def test_government_decree_upload_keeps_topics():
    """Постановление Правительства — единственная категория НПА, где темы
    осмысленны, и они должны доезжать до запроса."""
    form = _submitted_form(
        category='other_npa',
        act_type='Постановление Правительства Российской Федерации',
        act_number='1268',
        act_date='2024-09-20',
        act_title='Об утверждении Правил квотирования рабочих мест',
        source_title='Постановление Правительства Российской Федерации от 20 сентября 2024 N 1268',
        act_authority='',
        revision_date='2026-01-15',
        audience='employer',
        topics=['Квотирование', 'Инвалидность'],
    )

    request = build_ingest_request(form, 'other_npa', raw_text='Пункт 1.')

    assert request.document_id == 'npa-1268-2024'
    assert request.topics == ['Квотирование', 'Инвалидность']
    assert request.act_authority == 'Правительство Российской Федерации'
