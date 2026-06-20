from collections.abc import AsyncGenerator
from typing import Annotated

import httpx
from fastapi import Depends


async def get_http_session() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient() as client:
        yield client


HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_session)]
