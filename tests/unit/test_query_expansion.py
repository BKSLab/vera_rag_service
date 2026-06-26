from unittest.mock import AsyncMock

from app.clients.llm import LlmClient
from app.exceptions.llm import LlmApiRequestError
from app.models.schemas import QueryExpansionResult, QueryVariant
from app.search.query_expansion import expand_query


async def test_expand_query_falls_back_to_original_when_llm_unavailable():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.side_effect = LlmApiRequestError(error_details='boom', request_url='https://x')

    result = await expand_query(llm_client, 'какая квота на инвалидов')

    assert result == ['какая квота на инвалидов']


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
        'сколько дней отпуск у инвалида',
        'продолжительность отпуска инвалида',
        'как оформить квоту',
    ]


async def test_expand_query_dedupes_repeated_texts():
    llm_client = AsyncMock(spec=LlmClient)
    llm_client.get_llm_response.return_value = QueryExpansionResult(
        variants=[QueryVariant(sub_question='вопрос', rephrasings=['вопрос'])]
    )

    result = await expand_query(llm_client, 'вопрос')

    assert result == ['вопрос']


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
