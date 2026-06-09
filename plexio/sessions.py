import json
import os
import uuid
from datetime import datetime, timezone

import aiosqlite
from cryptography.fernet import Fernet, InvalidToken

from plexio.models.addon import AddonConfiguration

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    config_json  TEXT NOT NULL,
    label        TEXT,
    server_name  TEXT,
    created_at   TEXT NOT NULL,
    last_used_at TEXT
)
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_fernet(settings) -> Fernet:
    """Resolve the Fernet key. An operator-provided SESSION_ENCRYPTION_KEY takes
    precedence; otherwise use (or create) a persistent key file next to the DB so
    tokens are encrypted at rest by default with no configuration required."""
    key = settings.session_encryption_key
    if key:
        return Fernet(key if isinstance(key, bytes) else key.encode())
    parent = os.path.dirname(settings.session_db_path) or '.'
    key_path = os.path.join(parent, 'session.key')
    if os.path.exists(key_path):
        with open(key_path, 'rb') as f:
            return Fernet(f.read().strip())
    new_key = Fernet.generate_key()
    with open(key_path, 'wb') as f:
        f.write(new_key)
    os.chmod(key_path, 0o600)
    return Fernet(new_key)


async def init_sessions(settings):
    """Open the SQLite session store and ensure the schema exists.

    Returns None when sessions are disabled, so callers can treat the
    feature as absent without special-casing elsewhere.
    """
    if not settings.enable_sessions:
        return None
    parent = os.path.dirname(settings.session_db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fernet = _init_fernet(settings)
    db = await aiosqlite.connect(settings.session_db_path)
    await db.execute(_CREATE_TABLE)
    await db.commit()
    return SessionStore(db, fernet)


class SessionStore:
    """Durable, server-side store mapping a session id to an addon config.

    The config is stored as the same camelCase JSON shape that legacy
    base64 install URLs carry, so it round-trips through AddonConfiguration
    exactly as the legacy decode path does.
    """

    def __init__(self, db: aiosqlite.Connection, fernet: Fernet):
        self._db = db
        self._fernet = fernet

    async def create(self, config: dict, label: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        server_name = config.get('serverName') or config.get('server_name')
        now = _utcnow()
        payload = self._fernet.encrypt(json.dumps(config).encode()).decode()
        await self._db.execute(
            'INSERT INTO sessions '
            '(session_id, config_json, label, server_name, created_at, last_used_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, payload, label, server_name, now, now),
        )
        await self._db.commit()
        return session_id

    async def get_config(self, session_id: str) -> AddonConfiguration | None:
        async with self._db.execute(
            'SELECT config_json FROM sessions WHERE session_id = ?',
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        stored = row[0]
        try:
            plain = self._fernet.decrypt(stored.encode()).decode()
            reencrypted = None
        except InvalidToken:
            # Legacy plaintext row (pre-0.4.1): read as-is, then migrate it.
            plain = stored
            reencrypted = self._fernet.encrypt(plain.encode()).decode()
        now = _utcnow()
        if reencrypted is not None:
            await self._db.execute(
                'UPDATE sessions SET config_json = ?, last_used_at = ? '
                'WHERE session_id = ?',
                (reencrypted, now, session_id),
            )
        else:
            await self._db.execute(
                'UPDATE sessions SET last_used_at = ? WHERE session_id = ?',
                (now, session_id),
            )
        await self._db.commit()
        return AddonConfiguration(**json.loads(plain))

    async def list(self) -> list[dict]:
        async with self._db.execute(
            'SELECT session_id, label, server_name, created_at, last_used_at '
            'FROM sessions ORDER BY created_at DESC'
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                'session_id': r[0],
                'label': r[1],
                'server_name': r[2],
                'created_at': r[3],
                'last_used_at': r[4],
            }
            for r in rows
        ]

    async def delete(self, session_id: str) -> bool:
        cur = await self._db.execute(
            'DELETE FROM sessions WHERE session_id = ?',
            (session_id,),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def close(self):
        await self._db.close()
