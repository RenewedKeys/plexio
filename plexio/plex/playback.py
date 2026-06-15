"""Playback proxy: stream media through Plexio so progress can be reported to Plex.

Active only when a configuration has report_playback enabled. The /stream handler
emits a token-free /{cfg}/play/... URL pointing here; we proxy the bytes from the
Plex part URL (token held server-side), forward Range requests, and report the
playback position to Plex via /:/timeline. We do NOT scrobble -- Plex's own
"video played" watched threshold decides watched from the reported position, so a
partial view resumes at the exact point and a full playthrough marks watched.
"""
import base64
import time

import aiohttp
from fastapi.responses import StreamingResponse

PLEX_PRODUCT = 'Plexio'
PING_INTERVAL = 10.0      # seconds between timeline updates
CHUNK = 1 << 16           # 64 KiB


def b64decode_path(token: str) -> str:
    token += '=' * (-len(token) % 4)
    return base64.urlsafe_b64decode(token).decode()


def _client_id(identifier: str) -> str:
    return f'plexio-{identifier}'


async def _timeline(client, *, url, token, rating_key, state, time_ms,
                    duration_ms, identifier):
    try:
        await client.get(
            url / ':/timeline' % {
                'ratingKey': rating_key,
                'key': f'/library/metadata/{rating_key}',
                'state': state,
                'time': max(time_ms, 0),
                'duration': max(duration_ms, 0),
                'X-Plex-Token': token,
            },
            headers={
                'X-Plex-Client-Identifier': _client_id(identifier),
                'X-Plex-Product': PLEX_PRODUCT,
                'X-Plex-Device-Name': PLEX_PRODUCT,
            },
            timeout=aiohttp.ClientTimeout(total=5),
        )
    except Exception:
        pass


def _total_and_start(resp):
    """Full file size + start offset, from Content-Range (206) or Content-Length (200)."""
    cr = resp.headers.get('Content-Range')
    if cr and '/' in cr:
        try:
            rng, total = cr.split(' ', 1)[1].split('/')
            return int(total), int(rng.split('-')[0])
        except (ValueError, IndexError):
            pass
    try:
        return (int(resp.headers.get('Content-Length', 0)) or None), 0
    except ValueError:
        return None, 0


async def proxy_playback(request, *, client, configuration, rating_key,
                         duration_ms, part_key, identifier):
    upstream = (
        configuration.streaming_url
        / part_key[1:]
        % {'X-Plex-Token': configuration.access_token}
    )
    fwd = {}
    rng = request.headers.get('range')
    if rng:
        fwd['Range'] = rng

    resp = await client.get(
        upstream,
        headers=fwd,
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=60),
    )
    total, start = _total_and_start(resp)
    passthrough = {
        h: resp.headers[h]
        for h in ('Content-Length', 'Content-Range', 'Accept-Ranges')
        if h in resp.headers
    }

    async def streamer():
        streamed = 0
        last_ping = 0.0
        started = False
        try:
            async for chunk in resp.content.iter_chunked(CHUNK):
                yield chunk
                streamed += len(chunk)
                if not total or not duration_ms:
                    continue
                now = time.monotonic()
                if not started or now - last_ping >= PING_INTERVAL:
                    started = True
                    last_ping = now
                    await _timeline(
                        client, url=configuration.discovery_url,
                        token=configuration.access_token, rating_key=rating_key,
                        state='playing',
                        time_ms=int((start + streamed) / total * duration_ms),
                        duration_ms=duration_ms, identifier=identifier,
                    )
        finally:
            resp.close()
            if started and total and duration_ms:
                await _timeline(
                    client, url=configuration.discovery_url,
                    token=configuration.access_token, rating_key=rating_key,
                    state='stopped',
                    time_ms=int((start + streamed) / total * duration_ms),
                    duration_ms=duration_ms, identifier=identifier,
                )

    return StreamingResponse(
        streamer(),
        status_code=resp.status,
        headers=passthrough,
        media_type=resp.headers.get('Content-Type'),
    )
