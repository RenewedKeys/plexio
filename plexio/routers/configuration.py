from typing import Annotated

from aiohttp import ClientSession
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from yarl import URL

from plexio.dependencies import get_http_client, verify_admin_key
from plexio.models.addon import AddonConfiguration
from plexio.plex.media_server_api import check_server_connection
from plexio.settings import settings

router = APIRouter(prefix='/api/v1')


@router.get('/test-connection')
async def test_connection(
    http: Annotated[ClientSession, Depends(get_http_client)],
    url: str,
    token: str,
):
    success = await check_server_connection(
        client=http,
        url=URL(url),
        token=token,
    )
    return {'success': success}

@router.get('/public-config')
async def public_config():
    """
    Return runtime configuration safe to expose to the frontend.

    Currently surfaces base_url so the configure UI can generate install URLs
    using the operator-specified public origin instead of window.location.origin.
    Empty string when unset -- frontend falls back to window.location.origin.
    """
    return {'base_url': settings.base_url or ''}


@router.post('/sessions')
async def create_session(
    request: Request,
    config: dict = Body(...),
    label: str | None = None,
):
    """Create a server-side session that stores the addon configuration.

    The request body is the addon configuration as camelCase JSON -- the
    same shape legacy base64 install URLs carry. Optional `label` query
    param tags the install for later identification. Returns the session
    id to embed in the install URL: /{session_id}/manifest.json
    """
    store = getattr(request.state, 'sessions', None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Sessions are not enabled',
        )
    try:
        AddonConfiguration(**config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid configuration: {exc}',
        )
    session_id = await store.create(config, label=label)
    return {'session_id': session_id}


@router.get('/sessions', dependencies=[Depends(verify_admin_key)])
async def list_sessions(request: Request):
    """List stored sessions (metadata only, never tokens). Requires X-Admin-Key."""
    store = getattr(request.state, 'sessions', None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Sessions are not enabled',
        )
    return {'sessions': await store.list()}


@router.delete('/sessions/{session_id}', dependencies=[Depends(verify_admin_key)])
async def delete_session(request: Request, session_id: str):
    """Revoke (delete) a stored session by id. Requires X-Admin-Key."""
    store = getattr(request.state, 'sessions', None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Sessions are not enabled',
        )
    if not await store.delete(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Session not found',
        )
    return {'deleted': session_id}
