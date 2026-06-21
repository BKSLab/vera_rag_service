from httpx import AsyncClient


async def test_metrics_endpoint_exposes_prometheus_format(async_client: AsyncClient):
    """LOG-5 — базовые метрики доступны для скрейпинга Prometheus."""
    response = await async_client.get('/metrics')

    assert response.status_code == 200
    assert 'text/plain' in response.headers['content-type']
