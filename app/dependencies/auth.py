import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.settings import get_settings


async def verify_api_key(x_api_key: Annotated[str, Header()]) -> None:
    """Проверяет `X-API-Key` против единого ключа доступа из `Settings`.

    Единственный документированный потребитель публичного API сейчас — MCP
    Tools Server (раздел 5 RAG_SERVICE_PLAN.md), поэтому один статический
    ключ достаточен — без таблицы ключей в БД (раздел 6 FASTAPI_PATTERNS.md,
    вариант "X-Master-Key"). См. AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md,
    ARCH-1/API-1/SEC-1.
    """
    expected = get_settings().app.api_key.get_secret_value()
    if not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Невалидный API-ключ.')


VerifyApiKeyDep = Depends(verify_api_key)
