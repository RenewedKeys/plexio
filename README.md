# Plexio (natedogg058 fork)

> **This is a updated fork of [natedogg058/plexio](https://github.com/natedogg058/plexio).**
> Upstream is dormant (last release June 2026). This fork adds fixes and improvements for self-hosted deployments.

## What's different from upstream

- **Server-side proxy Plex auth fix** — the configure UI's PIN-based Plex sign-in (creating the auth pin, polling for the token, and listing servers) now routes through new backend endpoints (`/api/v1/plex-pin`, `/api/v1/plex-token/{id}`, `/api/v1/plex-resources`) that call `plex.tv` server-side, instead of the browser calling `plex.tv` directly. (0.7.0)
- **Completion-only playback reporting** — the byte-position playback proxy no longer sends periodic progress pings estimated from bytes served, since that estimate is wrong under VBR and races ahead when a client bulk-buffers, producing bogus resume points. Instead it reports a single "stopped" state once at least 90% of the file (with an 8 MiB floor to ignore probes) has streamed, so Plex's own threshold marks the item watched. (0.7.1)
- **`behaviorHints.filename` distinguishes Direct Play from Transcoded streams** — Direct Play keeps the real, unmodified filename (needed for hash/fingerprint-based lookups like OpenSubtitles, IntroDB, and Trakt scrobbling), while every Transcode stream now gets a `[Transcode]` / `[Transcode-720p]`-style tag appended before the extension. Tools like AIOStreams parse `behaviorHints.filename` (not the addon's `name`/`description`) to derive quality/resolution and expose it as `{stream.filename}` in custom formatters, so previously Direct Play and Transcode variants were indistinguishable downstream. (0.7.2)
- **Skip transcode stream options for 4K/UHD source media** — some Plex-hosting providers block transcoding of 4K content outright at the server level (resource protection), so only Direct Play works for that content there. Plexio now detects 4K sources (by pixel width and/or Plex's `videoResolution` label) and stops offering "transcode original" and "transcode down" streams for them, leaving Direct Play unaffected and always offered. (0.7.3)
- **Removed hardcoded `X-Plex-Platform=Chrome` on transcode streams** — every transcode stream URL forced Plex to build its HLS output profiled specifically for a Chrome browser client, regardless of what was actually requesting it (Stremio doesn't pass real client/platform info through to addons). Stricter players like Infuse could fail to even start playback against a manifest shaped for a platform they weren't running as, even though the same URL played fine in more permissive players like VLC. Plexio no longer asserts any client platform, so Plex falls back to its own generic transcode profile instead of one tuned for a browser. (0.7.4)

---

*Original upstream README below.*

---
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
- **Configure page uses sessions by default** — the configure UI now creates a server-side session on install and generates the short `/{session_id}/manifest.json` URL (Plex token never in the URL), automatically falling back to the legacy base64 URL if the session store is disabled or unreachable. (0.4.2)
- **Idempotent session creation** — submitting an identical config returns the existing session instead of minting a duplicate, keeping the admin session list clean (e.g. clicking clipboard then Install no longer creates two). (0.4.3)
- **Health endpoints** — `GET /api/v1/health` is a dependency-aware liveness probe (app + session store; 503 if the store is down), and `GET /api/v1/health/{session_id}` deep-checks whether that session's Plex backend is actually reachable (reachability only, never the token), so an uptime monitor can catch backend outages rather than just web-server outages. (0.5.0)
- **Continue Watching & Recently Added catalogs** — adds discovery rows to the Stremio board: "Continue Watching" (Plex On Deck — in-progress movies plus next-up/in-progress episodes, the latter surfaced as their parent series, deduped) and "Recently Added", each split into Movies / Shows and shown only for the library types you've configured. Catalog items resolve through the normal meta/stream flow (imdb-matched where Plex has the id). Discovery rows only — these don't feed Stremio's native Continue Watching bar, and a series row opens the show page rather than resuming the exact episode. (0.6.0)

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
- Documentation expansion for self-hosting behind reverse proxies
- Investigation of upstream toggle-default behaviour
