from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.admin import create_admin
from app.api.v1.endpoints.ingestion_runs import router
from app.core.settings import get_settings
from app.dependencies.db_session import get_db_session
from app.repositories.ingestion_run import IngestionRunRepository


@pytest.fixture
async def trace_client(db_session):
    app = FastAPI()
    create_admin(app, db_session.bind)
    app.include_router(router, prefix='/api/v1')
    app.dependency_overrides[get_db_session] = lambda: db_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        yield client


async def seed(db_session, **overrides):
    now = datetime.now(UTC)
    values = dict(
        id=str(uuid4()), document_id='tk-197-2001', source_title='<script>alert(1)</script>',
        kind='section', source='legal_sync', version='2026-09-21', revision_date=now.date(), section_number='59',
        collection_name='test_collection', request_id='attempt-42', change_id=42,
        status='succeeded', current_stage='integrity', started_at=now, updated_at=now, finished_at=now,
        stages=[{'key': 'integrity', 'status': 'succeeded', 'details': {'identifiers_match': True}}],
        result={'input_sha256': 'abc'}, error=None,
    )
    values.update(overrides)
    await IngestionRunRepository(async_sessionmaker(db_session.bind, expire_on_commit=False)).save(values)
    return values['id']


async def login(client):
    settings = get_settings().app
    response = await client.post('/admin/login', data={
        'username': settings.admin_login, 'password': settings.admin_password.get_secret_value(),
    })
    assert response.status_code == 302


async def test_admin_history_filters_details_export_and_auth(trace_client, db_session):
    run_id = await seed(db_session)
    for path in ('/admin/ingestion-log', f'/admin/ingestion-run/{run_id}', f'/admin/ingestion-run/{run_id}?format=json'):
        assert (await trace_client.get(path)).status_code == 302
    await login(trace_client)
    response = await trace_client.get('/admin/ingestion-log?request_id=attempt-42&change_id=42')
    assert response.status_code == 200 and run_id in response.text
    assert run_id not in (await trace_client.get('/admin/ingestion-log?request_id=other')).text
    assert (await trace_client.get('/admin/ingestion-log?status=made-up')).status_code == 400
    assert (await trace_client.get('/admin/ingestion-log?page=0')).status_code == 400
    assert (await trace_client.get('/admin/ingestion-run/missing')).status_code == 404
    details = await trace_client.get(f'/admin/ingestion-run/{run_id}')
    assert details.status_code == 200
    assert '<script>alert(1)</script>' not in details.text
    assert '&lt;script&gt;' in details.text
    exported = await trace_client.get(f'/admin/ingestion-run/{run_id}?format=json')
    assert exported.json()['id'] == run_id and exported.headers['cache-control'] == 'no-store'


async def test_stalled_running_filter_and_pagination(trace_client, db_session):
    stale = await seed(db_session, status='running', finished_at=None, updated_at=datetime.now(UTC) - timedelta(minutes=3))
    for _ in range(26):
        await seed(db_session)
    await login(trace_client)
    response = await trace_client.get('/admin/ingestion-log?status=stalled')
    assert stale in response.text and 'Нет сигнала' in response.text
    assert stale not in (await trace_client.get('/admin/ingestion-log?status=running')).text
    page = await trace_client.get('/admin/ingestion-log?page=2')
    assert page.status_code == 200 and stale in page.text


async def test_cross_service_read_api_requires_key_and_preserves_correlation(trace_client, db_session):
    run_id = await seed(db_session)
    path = '/api/v1/ingestion-runs?document_id=tk-197-2001&request_id=attempt-42'
    assert (await trace_client.get(path)).status_code != 200
    response = await trace_client.get(path, headers={'X-API-Key': get_settings().app.api_key.get_secret_value()})
    assert response.status_code == 200
    data = response.json()
    assert data['total'] == 1 and data['items'][0]['id'] == run_id
    assert data['items'][0]['request_id'] == 'attempt-42'
