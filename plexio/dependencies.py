import base64
import json

from aiohttp import ClientSession
from fastapi import HTTPException, Request, status
from sentry_sdk import set_user

from plexio.models.addon import AddonConfiguration


def get_http_client(request: Request) -> ClientSession:
    return request.state.plex_client


def get_cache(request: Request):
    return request.state.cache


async def get_addon_configuration(
    request: Request,
    base64_cfg: str | None = None,
    session_id: str | None = None,
) -> AddonConfiguration | None:
    # Legacy form: the full config is base64-encoded in the URL itself.
    if base64_cfg is not None:
        decoded = base64.b64decode(base64_cfg)
        return AddonConfiguration(**json.loads(decoded))
    # Session form: config is stored server-side, keyed by session id.
    if session_id is not None:
        store = getattr(request.state, 'sessions', None)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Sessions are not enabled',
            )
        configuration = await store.get_config(session_id)
        if configuration is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Session not found',
            )
        return configuration
    # Unconfigured: bare manifest with no install context.
    return None


def set_sentry_user(
    installation_id: str | None = None,
    session_id: str | None = None,
) -> None:
    identifier = installation_id or session_id
    if identifier:
        set_user({'id': identifier})
