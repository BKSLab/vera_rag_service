import json
from datetime import date

import httpx
import pytest
from pydantic import SecretStr

from app.clients.legal_sync import LegalSyncClient
from app.core.settings import LegalSyncSettings
from app.exceptions.legal_sync import LegalSyncClientError, LegalSyncRequisitesError


def _build_client(handler) -> LegalSyncClient:
    settings = LegalSyncSettings(
        legal_sync_enabled=True,
        legal_sync_base_url='http://legal-sync.test',
        legal_sync_api_key=SecretStr('sync-key'),
    )
    return LegalSyncClient(
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        settings=settings,
    )


async def _register(client: LegalSyncClient, **overrides):
    payload = {
        'document_id': 'fz-181-1995',
        'category': 'federal_law',
        'act_type': 'Федеральный закон',
        'act_number': '181-ФЗ',
        'act_date': date(1995, 11, 24),
        'act_title': 'О социальной защите инвалидов в Российской Федерации',
        'source_title': (
            'Федеральный закон "О социальной защите инвалидов в Российской Федерации" '
            'от 24.11.1995 № 181-ФЗ'
        ),
        'audience': 'both',
        'topics': [],
    }
    payload.update(overrides)
    return await client.register_tracked_document(**payload)


async def test_requisites_are_sent_structurally():
    """Реквизиты передаются полями, а не вычисляются из наименования."""

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content)
        captured['api_key'] = request.headers.get('X-API-Key')
        return httpx.Response(201, json={'id': 1, 'document_id': 'fz-181-1995'})

    client = _build_client(handler)
    await _register(client)

    assert captured['body']['document_number'] == '181-ФЗ'
    assert captured['body']['adoption_date'] == '1995-11-24'
    assert captured['body']['document_id'] == 'fz-181-1995'
    assert captured['api_key'] == 'sync-key'


async def test_search_hint_carries_act_type_not_full_title():
    """Legal Sync ищет акт по словам наименования — реквизиты в них только мешают."""

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content)
        return httpx.Response(201, json={})

    client = _build_client(handler)
    await _register(client)

    assert captured['body']['short_name'] == 'Федеральный закон'
    assert captured['body']['full_title'] == 'О социальной защите инвалидов в Российской Федерации'


async def test_government_decree_goes_to_government_block():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content)
        return httpx.Response(201, json={})

    client = _build_client(handler)
    await _register(
        client,
        category='other_npa',
        act_type='Постановление Правительства Российской Федерации',
        act_number='845',
        act_date=date(2026, 7, 4),
        act_title='О чём-то',
    )

    assert captured['body']['publication_block'] == 'government'
    assert captured['body']['document_number'] == '845'


async def test_document_without_number_cannot_be_tracked():
    """Без номера акт в банке правовых актов не найти, а угадывать нельзя."""

    client = _build_client(lambda request: httpx.Response(201, json={}))

    with pytest.raises(LegalSyncRequisitesError):
        await _register(client, act_number=None)


async def test_already_tracked_document_is_not_an_error():
    client = _build_client(lambda request: httpx.Response(409, text='уже существует'))

    result = await _register(client)

    assert result['already_tracked'] is True


async def test_service_failure_raises_client_error():
    client = _build_client(lambda request: httpx.Response(500, text='упал'))

    with pytest.raises(LegalSyncClientError):
        await _register(client)


async def test_missing_api_key_fails_before_request():
    settings = LegalSyncSettings(legal_sync_enabled=True, legal_sync_api_key=None)
    client = LegalSyncClient(httpx_client=httpx.AsyncClient(), settings=settings)

    with pytest.raises(LegalSyncClientError):
        await _register(client)
