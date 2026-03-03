"""
Session manager that keeps Plex transcode sessions alive by sending
periodic /:/timeline heartbeat updates.

Plexio is stateless — it returns a URL and Stremio takes over. Without
heartbeats, Plex kills transcode sessions after ~180s of idle time.
This module starts heartbeating when a transcode URL is generated and
continues for the media's full duration plus a grace period.
"""

import asyncio
import logging
import ssl
import uuid
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class PlexSession:
    """Represents an active Plex playback session."""

    def __init__(
        self,
        session_id: str,
        server_url: str,
        access_token: str,
        rating_key: str,
        duration_ms: int,
        media_key: str,
    ):
        self.session_id = session_id
        self.server_url = server_url
        self.access_token = access_token
        self.rating_key = rating_key
        self.duration_ms = duration_ms
        self.media_key = media_key
        self.start_time = datetime.utcnow()
        self.current_time_ms = 0
        self.state = 'playing'
        self._task: Optional[asyncio.Task] = None

    @property
    def elapsed_ms(self) -> int:
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        return int(min(elapsed, self.duration_ms))


class SessionManager:
    """
    Manages active Plex playback sessions with periodic heartbeats.

    Sessions are keyed by (access_token, rating_key) to avoid duplicates
    when the same content is requested multiple times.
    """

    HEARTBEAT_INTERVAL = 15  # seconds between timeline updates
    GRACE_PERIOD = timedelta(minutes=30)
    MAX_SESSIONS = 10

    def __init__(self):
        self._sessions: dict[str, PlexSession] = {}
        self._session_keys: dict[tuple[str, str], str] = {}
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self._http_client: Optional[aiohttp.ClientSession] = None

    def _get_http_client(self) -> aiohttp.ClientSession:
        if self._http_client is None or self._http_client.closed:
            self._http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(ssl=self._ssl_context),
            )
        return self._http_client

    async def start_session(
        self,
        server_url: str,
        access_token: str,
        rating_key: str,
        duration_ms: int,
        media_key: str,
    ) -> str:
        """Start a new heartbeat session for a transcode stream."""
        content_key = (access_token, rating_key)

        # Replace existing session for the same content
        if content_key in self._session_keys:
            old_session_id = self._session_keys[content_key]
            await self.stop_session(old_session_id)

        # Enforce session limit
        if len(self._sessions) >= self.MAX_SESSIONS:
            oldest_id = next(iter(self._sessions))
            await self.stop_session(oldest_id)

        session_id = str(uuid.uuid4())
        client_id = f'plexio-{session_id[:8]}'

        session = PlexSession(
            session_id=session_id,
            server_url=server_url.rstrip('/'),
            access_token=access_token,
            rating_key=rating_key,
            duration_ms=duration_ms,
            media_key=media_key,
        )

        self._sessions[session_id] = session
        self._session_keys[content_key] = session_id
        session._task = asyncio.create_task(
            self._heartbeat_loop(session, client_id),
        )

        logger.info(
            f'Started session {session_id} for ratingKey={rating_key} '
            f'(duration={duration_ms}ms)',
        )
        return session_id

    async def stop_session(self, session_id: str):
        """Stop a session and send a 'stopped' timeline update."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return

        # Clean up the content key mapping
        content_key = (session.access_token, session.rating_key)
        if self._session_keys.get(content_key) == session_id:
            del self._session_keys[content_key]

        if session._task and not session._task.done():
            session._task.cancel()

        session.state = 'stopped'
        await self._send_timeline(session, f'plexio-{session_id[:8]}')
        logger.info(f'Stopped session {session_id}')

    async def _heartbeat_loop(self, session: PlexSession, client_id: str):
        """Send periodic timeline updates until the session expires."""
        try:
            max_duration = (
                timedelta(
                    milliseconds=session.duration_ms,
                )
                + self.GRACE_PERIOD
            )

            while True:
                elapsed = datetime.utcnow() - session.start_time
                if elapsed > max_duration:
                    logger.info(
                        f'Session {session.session_id} exceeded max duration, '
                        f'stopping',
                    )
                    break

                session.current_time_ms = session.elapsed_ms
                await self._send_timeline(session, client_id)
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)

        except asyncio.CancelledError:
            logger.debug(f'Session {session.session_id} heartbeat cancelled')
        except Exception as e:
            logger.error(
                f'Session {session.session_id} heartbeat error: {e}',
            )
        finally:
            self._sessions.pop(session.session_id, None)
            content_key = (session.access_token, session.rating_key)
            if self._session_keys.get(content_key) == session.session_id:
                self._session_keys.pop(content_key, None)
            session.state = 'stopped'
            try:
                await self._send_timeline(session, client_id)
            except Exception:
                pass

    async def _send_timeline(self, session: PlexSession, client_id: str):
        """Send a single /:/timeline update to the Plex server."""
        params = {
            'ratingKey': session.rating_key,
            'key': session.media_key,
            'state': session.state,
            'time': str(session.current_time_ms),
            'duration': str(session.duration_ms),
            'X-Plex-Token': session.access_token,
            'X-Plex-Client-Identifier': client_id,
            'X-Plex-Product': 'Plexio',
            'X-Plex-Version': '0.2.0',
            'X-Plex-Platform': 'Stremio',
            'X-Plex-Device': 'Plexio Addon',
            'X-Plex-Device-Name': 'Plexio',
        }

        url = f'{session.server_url}/:/timeline'

        try:
            client = self._get_http_client()
            async with client.get(url, params=params) as response:
                if response.status == 200:
                    logger.debug(
                        f'Timeline update sent for session {session.session_id} '
                        f'(time={session.current_time_ms}ms, '
                        f'state={session.state})',
                    )
                else:
                    logger.warning(
                        f'Timeline update failed for session '
                        f'{session.session_id}: {response.status}',
                    )
        except Exception as e:
            logger.warning(
                f'Timeline update error for session ' f'{session.session_id}: {e}',
            )

    async def cleanup(self):
        """Stop all sessions and close the HTTP client."""
        for session_id in list(self._sessions.keys()):
            await self.stop_session(session_id)
        if self._http_client and not self._http_client.closed:
            await self._http_client.close()
