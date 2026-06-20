from qdrant_client import AsyncQdrantClient

from app.core.settings import get_settings

_settings = get_settings().qdrant

qdrant_client = AsyncQdrantClient(
    url=_settings.qdrant_url,
    api_key=_settings.qdrant_api_key.get_secret_value() if _settings.qdrant_api_key else None,
)
