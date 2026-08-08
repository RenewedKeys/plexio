import asyncio
from typing import Annotated, Any

import aiohttp
from aiohttp import ClientSession
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, status

from plexio.dependencies import get_http_client
from plexio.settings import settings

router = APIRouter(prefix='/api/v1')

PLEX_API_URL = 'https://plex.tv/api/v2'
PLEX_HEADERS = {
    'Accept': 'application/json',
    'X-Plex-Product': 'Plexio',
    'X-Plex-Version': '1.0.0',
}


async def _plex_request(
    http: ClientSession,
    method: str,
    path: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int] | None = None,
) -> Any:
    try:
        async with http.request(
            method,
            f'{PLEX_API_URL}{path}',
            headers={**PLEX_HEADERS, **headers},
            params=params,
            timeout=settings.plex_requests_timeout,
        ) as response:
            if response.status >= 400:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f'Plex API returned HTTP {response.status}',
                )
            return await response.json()
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail='Unable to reach the Plex API',
        ) from exc


@router.post('/plex-pin')
async def create_plex_pin(
    http: Annotated[ClientSession, Depends(get_http_client)],
    client_identifier: str = Header(
        ...,
        alias='X-Plex-Client-Identifier',
        min_length=1,
        max_length=255,
    ),
):
    return await _plex_request(
        http,
        'POST',
        '/pins',
        headers={'X-Plex-Client-Identifier': client_identifier},
        params={'strong': 'true'},
    )


@router.get('/plex-token/{pin_id}')
async def get_plex_token(
    http: Annotated[ClientSession, Depends(get_http_client)],
    pin_id: int = Path(..., gt=0),
    client_identifier: str = Header(
        ...,
        alias='X-Plex-Client-Identifier',
        min_length=1,
        max_length=255,
    ),
    code: str = Query(..., min_length=1, max_length=255),
):
    return await _plex_request(
        http,
        'GET',
        f'/pins/{pin_id}',
        headers={'X-Plex-Client-Identifier': client_identifier},
        params={'code': code},
    )


@router.get('/plex-resources')
async def get_plex_resources(
    http: Annotated[ClientSession, Depends(get_http_client)],
    client_identifier: str = Header(
        ...,
        alias='X-Plex-Client-Identifier',
        min_length=1,
        max_length=255,
    ),
    token: str = Header(
        ...,
        alias='X-Plex-Token',
        min_length=1,
        max_length=4096,
    ),
    include_https: int = Query(1, alias='includeHttps', ge=0, le=1),
    include_relay: int = Query(1, alias='includeRelay', ge=0, le=1),
):
    return await _plex_request(
        http,
        'GET',
        '/resources',
        headers={
            'X-Plex-Client-Identifier': client_identifier,
            'X-Plex-Token': token,
        },
        params={
            'includeHttps': include_https,
            'includeRelay': include_relay,
        },
    )
