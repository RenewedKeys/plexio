# Plexio (natedogg058 fork)

> **This is a maintained fork of [vanchaxy/plexio](https://github.com/vanchaxy/plexio).**
> Upstream is dormant (last release May 2025). This fork adds fixes and improvements for self-hosted deployments.

## What's different from upstream

- **`behaviorHints.filename` on stream objects** — populates the Stremio-standard field used by clients for release fingerprinting (IntroDB skip intro, Trakt scrobbling, OpenSubtitles hash lookup). Closes a gap vs AIOStreams and other Stremio-standard addons. ([upstream PR #69](https://github.com/vanchaxy/plexio/pull/69))
- **Wider default CORS regex** — covers localhost on any port, private LAN ranges (`192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`), Tailscale tailnet domains (`*.ts.net`), and `app.strem.io`. Reduces friction for self-hosted deployments behind reverse proxies or on Tailscale. `CORS_ORIGIN_REGEX` env var override is preserved.
- **`behaviorHints.videoSize` on stream objects** — exposes each version's file size so clients can display or choose by size. (0.3.1)
- **`BASE_URL` env var** — sets the public origin used for install-URL generation behind a reverse proxy / Tailscale Funnel, instead of relying on `window.location.origin`. (0.3.0)
- **Server-side sessions (optional)** — install URLs can reference a stored session id (`/{session_id}/...`) instead of embedding the full config (including the Plex token) as base64. Config is persisted in SQLite under `/data`; legacy base64 URLs continue to work unchanged. Requires a writable `/data` volume (see Installation). (0.4.0)
- **Encrypted sessions + revocation** — stored session config is Fernet-encrypted at rest (`SESSION_ENCRYPTION_KEY`, or an auto-generated `session.key` next to the DB). Operators can list and revoke sessions via admin-gated `GET` / `DELETE /api/v1/sessions` (`ADMIN_KEY`). (0.4.1)

## Installation

Pull the published image:
```bash
docker run -d -p 7777:80 -v plexio-data:/data ghcr.io/natedogg058/plexio:latest
```
Or build from source with `docker build -t plexio-fork .`.

**Persistent storage (sessions):** the optional server-side session store keeps a SQLite DB at `/data/sessions.db`. The image creates `/data` owned by the `unit` app user (uid 999), so a Docker **named volume** (as above) inherits writable ownership automatically. If you bind-mount a host directory instead, `chown 999:999` it first. Disable the store entirely with `ENABLE_SESSIONS=false`, in which case no `/data` access is needed.

**Session env vars:** `ADMIN_KEY` enables and protects the list/revoke endpoints (unset = those endpoints return 403). `SESSION_ENCRYPTION_KEY` sets the Fernet key for encryption at rest; if unset, a key file is created automatically alongside the database.

## Roadmap

See [ISSUES](https://github.com/natedogg058/plexio/issues) for open work. Planned fork-specific additions:
- `BASE_URL` env var for reverse proxy / Tailscale Funnel deployments
- Documentation expansion for self-hosting behind reverse proxies
- Investigation of upstream toggle-default behaviour

---

*Original upstream README below.*

---
# Plexio: Plex Interaction for Stremio

⚠️ Plexio is an independent project and is not in any way affiliated with Plex or Stremio. ⚠️

Plexio is an addon that bridges the gap between Plex and Stremio, enabling seamless 
integration of your Plex media within the Stremio interface. With Plexio, you can discover 
and stream your Plex content directly in Stremio.

### Features
* offers both direct and transcoded streams;
* stream locally or from remote devices;
* allows searching through your Plex library;
* works with Cinemeta and other IMDB-based addons;
* handles media without IMDB matching;
* uses OAuth for safe login without sharing passwords;
* fully open-source with self-hosting support.


## Self-Hosting
If you'd prefer to self-host Plexio, you can do so easily using Docker. Follow these steps:

1. Use the following command to start a Plexio instance:
   ```bash
   docker run -d -p 7777:80 ghcr.io/vanchaxy/plexio
   ```
2. Plexio addon will be available at http://localhost:7777/.

### Optional Configuration with Environment Variables
* *CORS_ORIGIN_REGEX*: A regex pattern to define allowed CORS origins 
(default: `https?:\/\/localhost:\d+|.*plexio.stream|.*strem.io|.*stremio.com`).
* *PLEX_REQUESTS_TIMEOUT*: Timeout for Plex server requests in seconds (default: `20`).
* *CACHE_TYPE*: Defines the cache type to use `memory`/`redis` (default: `memory`).
* *REDIS_URL*: URL for a Redis instance if you use `redis` cache (default: `redis://redis:6399/0`).
* *PLEX_MATCHING_TOKEN*: Auth token for Plex media matching (default: `None`).
* *SENTRY_DSN*: DSN for error tracking with Sentry (default: `None`).

### Using addon with shared Plex server
If you are using Plexio with a Plex server that you do not own (you will see a "shared" badge 
next to the server name), you must provide the `PLEX_MATCHING_TOKEN` environment variable. 
This token is an access token from a Plex server you own, which will be used to
query the Plex API and resolve the Plex GUID using IMDB IDs.

To find your Plex authentication token, open any media on a Plex server you own.
Look for the XML data for the media and find the `X-Plex-Token` in the URL. 
Copy the token from the URL.

You can learn more about finding your authentication token in the official Plex article 
["Finding an authentication token"](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

## Local Development
1. Fork the Repository.
2. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/plexio.git
   ```
3. Create a `.env` file and configure the required environment variables.
4. Run doker-compose:
   ```bash
   docker-compose up --build
   ```

## Contacts

For bug reports, feature requests, or general questions, join our
[Discord support forum](https://discord.gg/8RWUkebmDs).

Alternatively, you can open an issue directly in this repository.

