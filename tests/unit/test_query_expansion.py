from unittest.mock import AsyncMock

from app.clients.llm import LlmClient
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import QueryExpansionResult, QueryVariant
from app.search.prompts.query_expansion import QUERY_EXPANSION_PROMPT
from app.search.query_expansion import expand_query, expand_query_with_status


async def test_expand_query_falls_back_to_original_when_llm_unavailable():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.side_effect = LlmApiRequestError(error_details='boom', request_url='https://x')

    result = await expand_query(llm_client, 'какая квота на инвалидов')

    assert result == ['какая квота на инвалидов']


async def test_expand_query_with_status_reports_fallback_when_llm_unavailable():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.side_effect = LlmApiRequestError(error_details='boom', request_url='https://x')

    result = await expand_query_with_status(llm_client, 'какая квота на инвалидов')

    assert result.queries == ['какая квота на инвалидов']
    assert result.status == 'fallback_unavailable'


async def test_expand_query_returns_single_variant_with_rephrasing():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[
            QueryVariant(
                sub_question='какая квота на инвалидов',
                rephrasings=['квота на трудоустройство инвалидов в процентах'],
            )
        ]
    )

    result = await expand_query(llm_client, 'какая квота на инвалидов')

    assert result == ['какая квота на инвалидов', 'квота на трудоустройство инвалидов в процентах']


async def test_expand_query_decomposes_compound_question_into_sub_questions():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[
            QueryVariant(sub_question='сколько дней отпуск у инвалида', rephrasings=['продолжительность отпуска инвалида']),
            QueryVariant(sub_question='как оформить квоту', rephrasings=[]),
        ]
    )

    result = await expand_query(llm_client, 'сколько дней отпуск у инвалида и как оформить квоту')

    assert result == [
        'сколько дней отпуск у инвалида и как оформить квоту',
        'сколько дней отпуск у инвалида',
        'продолжительность отпуска инвалида',
        'как оформить квоту',
    ]


async def test_expand_query_keeps_original_first_when_llm_omits_it():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[
            QueryVariant(
                sub_question='увольнение по инициативе работодателя',
                rephrasings=['основания увольнения работника'],
            )
        ]
    )

    result = await expand_query(llm_client, 'п. 2 ч. 1 ст. 81 ТК РФ')

    assert result == [
        'п. 2 ч. 1 ст. 81 ТК РФ',
        'увольнение по инициативе работодателя',
        'основания увольнения работника',
    ]


async def test_expand_query_dedupes_original_and_repeated_model_texts():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[QueryVariant(sub_question='вопрос', rephrasings=['вопрос'])]
    )

    result = await expand_query(llm_client, 'вопрос')

    assert result == ['вопрос']


async def test_expand_query_dedupes_case_and_whitespace_variants_preserving_first_text():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[
            QueryVariant(
                sub_question='статья 21 федерального закона от 24.11.1995 № 181-фз',
                rephrasings=['СТАТЬЯ   21 ФЕДЕРАЛЬНОГО ЗАКОНА ОТ 24.11.1995 № 181-ФЗ'],
            )
        ]
    )
    original = 'Статья 21 Федерального закона от 24.11.1995 № 181-ФЗ'

    result = await expand_query(llm_client, original)

    assert result == [original]


def test_query_expansion_result_caps_variants_and_rephrasings_above_limit():
    """Расширение запроса ограничено MAX_SUB_QUESTIONS×MAX_REPHRASINGS_PER_SUB_QUESTION
    (раздел 8 плана) — LLM, вернувший больше, не должен взорвать веер
    параллельных hybrid_search."""
    result = QueryExpansionResult(
        variants=[
            QueryVariant(sub_question=f'подвопрос {i}', rephrasings=[f'перефраз {i} a', f'перефраз {i} b'])
            for i in range(5)
        ]
    )

    assert len(result.variants) == 3
    assert all(len(variant.rephrasings) == 1 for variant in result.variants)


def test_query_expansion_prompt_preserves_reference_only_query_verbatim():
    lowered = QUERY_EXPANSION_PROMPT.lower()

    assert 'приоритетное правило для запроса, состоящего только из одной ссылки' in lowered
    assert 'символ в символ повторять исходную ссылку' in lowered
    assert '"rephrasings" должен быть пустым списком' in lowered
    assert 'не расшифровывай содержание нормы' in lowered
    assert 'п. 2 ч. 1 ст. 81 тк рф' in lowered
    assert 'статья 21 федерального закона от 24.11.1995 № 181-фз' in lowered


def test_query_expansion_prompt_preserves_reference_inside_question():
    lowered = QUERY_EXPANSION_PROMPT.lower()

    assert 'если ссылка на норму является частью содержательного вопроса' in lowered
    assert 'сохрани весь исходный вопрос без изменений в "sub_question"' in lowered
    assert 'дословно сохранять все реквизиты ссылки' in lowered
    assert 'не заменяй ссылку её предполагаемым содержанием' in lowered
    assert 'не добавляй ответ на вопрос до поиска' in lowered


def test_query_expansion_prompt_forbids_new_user_facts():
    lowered = QUERY_EXPANSION_PROMPT.lower()

    assert 'не добавляет новых фактов, ролей, обстоятельств' in lowered
    assert 'не означает, что пользователь является инвалидом' in lowered
    assert 'только сведения, прямо присутствующие в <user_query>' in lowered


def test_query_expansion_prompt_preserves_yes_no_uncertainty():
    lowered = QUERY_EXPANSION_PROMPT.lower()

    assert 'сохраняй неопределённость' in lowered
    assert 'не превращай вопрос о возможности в утверждение о наличии запрета' in lowered
    assert 'правовое регулирование увольнения в период временной нетрудоспособности' in lowered
    assert 'недопустимо `запрет на увольнение работника' in lowered
