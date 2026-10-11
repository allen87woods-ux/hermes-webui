"""
Hermes Web UI -- Gateway session watcher.

Background daemon thread that polls state.db every 5 seconds for changes
to gateway sessions (telegram, discord, slack, etc.). When changes are
detected, it pushes notifications to all subscribed SSE clients.

This enables real-time session list updates in the sidebar without
requiring any changes to hermes-agent.
"""
import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

from api.config import HOME
from api.agent_sessions import open_state_db_readonly, read_importable_agent_session_rows

logger = logging.getLogger(__name__)


# ── State hash tracking ─────────────────────────────────────────────────────

def _snapshot_hash(sessions: list) -> str:
    """Create a lightweight hash of session IDs and timestamps for change detection."""
    key = '|'.join(
        f"{s['session_id']}:{s.get('updated_at', 0)}:{s.get('message_count', 0)}"
        for s in sorted(sessions, key=lambda x: x['session_id'])
    )
    return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()


# Sources excluded from the WebUI sidebar projection. Must match the default
# ``exclude_sources`` used by ``read_importable_agent_session_rows`` so the
# cheap change-detection scan below sees exactly the same row set as the
# expensive projection (otherwise cron message churn would defeat the gate).
_WATCHER_EXCLUDED_SOURCES = ("cron", "webui")


def _cheap_change_fingerprint(db_path: Path) -> str | None:
    """Compute a cheap fingerprint with an index-covered message aggregate.

    The expensive projection (``read_importable_agent_session_rows``) runs a CTE
    plus a per-session ``MAX(messages.timestamp)`` aggregation over an oversampled
    candidate set every poll. On a large ``state.db`` (hundreds of sessions, tens
    of thousands of messages) that is ~10x the cost of a single ``sessions``-table
    scan, and the watcher runs it forever on a 5s timer even when nothing changed
    (issue #3506).

    This hashes every sessions-table column the projection uses, plus a
    per-session ``COUNT`` / ``MAX(messages.timestamp)`` aggregate scoped to the
    same non-cron/webui rows. The message aggregate stays on the agent's existing
    ``(session_id, timestamp)`` covering index, avoiding a table-page lookup for
    every historical message.

    ``role`` is intentionally absent because it is not in that index. A bounded
    periodic full projection in ``GatewayWatcher._poll_once`` covers rare
    role-only visibility mutations without restoring the five-second table scan.

    Returns the fingerprint string, or ``None`` on any error / a pre-source
    schema so the caller falls back to running the expensive projection rather
    than risk skipping a change.
    """
    # Columns the projection reads from the ``sessions`` table. ``id``/``source``
    # are always present (``source`` is required for the projection to run at
    # all); the rest are optional on older agent schemas and filtered below.
    _PROJECTION_SESSION_COLS = (
        'id', 'source', 'session_source', 'title', 'model', 'message_count',
        'started_at', 'ended_at', 'end_reason', 'parent_session_id', 'archived',
        'user_id', 'chat_id', 'chat_type', 'thread_id', 'session_key',
        'origin_chat_id', 'origin_user_id', 'platform',
    )
    try:
        with closing(open_state_db_readonly(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(sessions)")
            cols = {row[1] for row in cur.fetchall()}
            if 'source' not in cols:
                return None
            selectable = [c for c in _PROJECTION_SESSION_COLS if c in cols]
            placeholders = ", ".join("?" for _ in _WATCHER_EXCLUDED_SOURCES)
            cur.execute(
                f"SELECT {', '.join(selectable)} FROM sessions "
                f"WHERE source IS NOT NULL AND source NOT IN ({placeholders}) "
                f"ORDER BY id",
                list(_WATCHER_EXCLUDED_SOURCES),
            )
            h = hashlib.md5(usedforsecurity=False)
            for row in cur.fetchall():
                h.update(repr(row).encode('utf-8', 'replace'))
                h.update(b'\x1e')
            # A same-count transcript rewrite (SessionDB.replace_messages used by
            # /retry, /undo, /compress) deletes + reinserts messages with new
            # timestamps but can leave sessions.message_count unchanged — so the
            # sessions-only scan above would miss it and the watcher would skip a
            # projection whose last_activity (MAX(messages.timestamp)) actually
            # moved. Fold in a PER-SESSION COUNT/MAX aggregate, scoped to the same
            # non-excluded sessions as the projection. COUNT preserves drift
            # detection; MAX catches same-count rewrites because replacement rows
            # receive fresh timestamps. Do not read ``role`` here: the normal
            # (session_id, timestamp) index can then cover this five-second scan
            # instead of forcing a table-page lookup for every historical row.
            if 'messages' in {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
                try:
                    msg_rows = conn.execute(
                        "SELECT s.id, COUNT(m.id), "
                        "COALESCE(MAX(m.timestamp), 0) "
                        "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
                        f"WHERE s.source IS NOT NULL AND s.source NOT IN ({placeholders}) "
                        "GROUP BY s.id ORDER BY s.id",
                        list(_WATCHER_EXCLUDED_SOURCES),
                    ).fetchall()
                    for mrow in msg_rows:
                        h.update(repr(mrow).encode('utf-8', 'replace'))
                        h.update(b'\x1e')
                except sqlite3.Error:
                    # messages table shape unknown → don't trust the fingerprint;
                    # signal the caller to run the full projection.
                    return None
            return h.hexdigest()
    except Exception:
        return None


# ── DB resolution (shared pattern with state_sync.py) ──────────────────────

def _get_state_db_path(hermes_home: Path | None = None) -> Path:
    """Resolve state.db path for the active profile."""
    if hermes_home is not None:
        return Path(hermes_home).expanduser().resolve() / 'state.db'
    try:
        from api.profiles import get_active_hermes_home
        hermes_home = Path(get_active_hermes_home()).expanduser().resolve()
    except Exception:
        hermes_home = Path(os.getenv('HERMES_HOME', str(HOME / '.hermes'))).expanduser().resolve()
    return hermes_home / 'state.db'


def _get_agent_sessions_from_db(db_path: Path | None = None) -> list | None:
    """Read all non-webui sessions from state.db.

    Returns a list of session dicts (including an empty list for a successful
    empty projection), or ``None`` when the projection fails.
    """
    db_path = Path(db_path) if db_path is not None else _get_state_db_path()
    if not db_path.exists():
        return []

    try:
        sessions = []
        for row in read_importable_agent_session_rows(db_path, limit=200, log=logger):
            sessions.append({
                'session_id': row['id'],
                'title': row['title'] or 'Agent Session',
                'model': row['model'] or None,
                'message_count': row['message_count'] or row['actual_message_count'] or 0,
                'created_at': row['started_at'],
                'updated_at': row['last_activity'] or row['started_at'],
                'source': row['source'] or 'cli',
                'raw_source': row.get('raw_source'),
                'session_source': row.get('session_source'),
                'source_label': row.get('source_label'),
            })
        return sessions
    except Exception:
        return None


# ── GatewayWatcher ──────────────────────────────────────────────────────────

class GatewayWatcher:
    """Background thread that polls state.db for agent session changes.

    Usage:
        watcher = GatewayWatcher()
        watcher.start()
        q = watcher.subscribe()
        # ... receive change events via q.get() ...
        watcher.unsubscribe(q)
        watcher.stop()
    """

    POLL_INTERVAL = 5  # seconds between polls
    # ``messages.role`` is not present in the agent's covering
    # ``(session_id, timestamp)`` index, but the full projection uses it for CLI
    # visibility. Keep the hot poll index-only and bound detection of rare
    # role-only mutations with a periodic parity projection.
    PROJECTION_PARITY_INTERVAL = 60.0
    SUBSCRIBER_TIMEOUT = 30  # seconds before sending keepalive comment

    def __init__(
        self,
        *,
        hermes_home: Path | None = None,
        profile_name: str | None = None,
        state_db_path: Path | None = None,
    ):
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._hermes_home = Path(hermes_home).expanduser().resolve() if hermes_home else None
        self._state_db_path = (
            Path(state_db_path).expanduser().resolve()
            if state_db_path is not None
            else _get_state_db_path(self._hermes_home) if self._hermes_home is not None else _get_state_db_path()
        )
        self.profile_name = profile_name or ""
        self._last_hash: str = ''
        self._last_sessions: list = []
        # Cheap sessions-only fingerprint from the previous poll. When it is
        # unchanged we skip the expensive messages-JOIN projection entirely
        # (issue #3506). Empty string forces the first poll to run the full read.
        self._last_cheap_fp: str = ''
        self._last_full_projection_at: float | None = None
        # Local patch 2026-10-11 (#19): last-seen ``{id: (pinned, archived)}`` from
        # state.db. Inbound flag sync acts only on TRANSITIONS (patches #18's mirror
        # is one-way), so this baseline is what makes "who changed it" answerable
        # without timestamps: a change here that the WebUI never made is external.
        self._db_flag_snapshot: dict[str, tuple[bool, bool]] = {}
        # Local patch 2026-10-11 (#20): same idea for titles -- ``{id: (title,
        # title_source)}``. A title changed outside the WebUI (Hermes Desktop's
        # rename, ``hermes sessions`` / CLI /title, the TUI) is adopted into the
        # sidecar so both surfaces list the same name.
        self._db_title_snapshot: dict[str, tuple] = {}
        # A sidecar written less than this long ago may still be mid-toggle (patch
        # #18 writes state.db after the sidecar), so never adopt over a fresh one.
        self.SIDECAR_SETTLE_SECS = 10.0

    def start(self):
        """Start the watcher daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name='gateway-watcher')
        self._thread.start()

    def is_alive(self) -> bool:
        """Return True when the poll thread is running.

        Public accessor used by ``/api/sessions/gateway/stream`` probe mode and
        the live SSE handler to detect a watcher instance whose poll thread
        died silently (e.g. uncaught exception in ``_poll_loop``).  Callers
        use this to decide whether to return 503 and trigger the client-side
        polling fallback, instead of handing out an SSE connection that would
        never emit events.
        """
        t = self._thread
        return t is not None and t.is_alive()

    def stop(self):
        """Stop the watcher thread."""
        self._stop_event.set()
        # Wake up any subscribers
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put(None)  # sentinel
                except Exception:
                    logger.debug("Failed to send sentinel to subscriber")
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def subscribe(self) -> queue.Queue:
        """Subscribe to change events. Returns a queue.Queue.
        Events are dicts: {'type': 'sessions_changed', 'sessions': [...]}
        A None sentinel means the watcher is stopping.
        """
        q = queue.Queue(maxsize=10)
        with self._sub_lock:
            self._subscribers.append(q)
            # Stop-race safety: if stop() already ran (set _stop_event and drained
            # the then-current subscriber list) before we appended, this queue would
            # never receive the sentinel and the SSE loop would hang open with
            # keepalives but no events. Enqueue the sentinel ourselves so the handler
            # closes and reconnects to the live registry watcher. (#3629 / Codex gate)
            if self._stop_event.is_set():
                try:
                    q.put_nowait(None)
                except Exception:
                    logger.debug("Failed to send stop sentinel to late subscriber")
        return q

    def unsubscribe(self, q: queue.Queue):
        """Remove a subscriber queue."""
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _notify_subscribers(self, sessions: list):
        """Push change event to all subscribers."""
        event = {
            'type': 'sessions_changed',
            'sessions': sessions,
        }
        with self._sub_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)  # remove slow consumers
                except Exception:
                    dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass
                # Send a None sentinel so the SSE handler unblocks, closes,
                # and lets the browser's EventSource auto-reconnect.
                try:
                    q.put_nowait(None)
                except Exception:
                    logger.debug("Failed to send sentinel to dead subscriber")

    def _poll_once(self, *, now: float | None = None) -> bool:
        """Run one change-detection pass and report whether projection ran.

        Most passes stay on the covering fingerprint. A bounded parity pass
        protects projection fields (notably role-derived CLI visibility) that
        the agent's existing index cannot see.
        """
        db_path = self._state_db_path
        # A watcher may start before the agent has created state.db. Publishing an
        # empty first snapshot would make an already-rendered sidebar disappear;
        # wait for the first real database instead. If a previously observed DB
        # disappears, the normal projection path still publishes that change.
        if (
            not db_path.exists()
            and self._last_full_projection_at is None
            and not self._last_hash
        ):
            return False

        cheap_fp = _cheap_change_fingerprint(db_path) if db_path.exists() else ''
        current_time = time.monotonic() if now is None else now
        fingerprint_changed = cheap_fp is None or cheap_fp != self._last_cheap_fp
        parity_due = (
            self._last_full_projection_at is None
            or current_time - self._last_full_projection_at
            >= self.PROJECTION_PARITY_INTERVAL
        )
        if not fingerprint_changed and not parity_due:
            return False

        sessions = _get_agent_sessions_from_db(db_path)
        if sessions is None:
            return False
        current_hash = _snapshot_hash(sessions)
        if cheap_fp is not None:
            self._last_cheap_fp = cheap_fp
        self._last_full_projection_at = current_time

        if current_hash != self._last_hash:
            self._last_hash = current_hash
            self._last_sessions = sessions
            self._notify_subscribers(sessions)
        return True

    # ── Local patch 2026-10-11 (#19): inbound flag sync (Desktop -> WebUI) ──
    #
    # Patch #18 mirrors a WebUI pin/archive toggle INTO state.db. Nothing mirrored
    # the other way, so a pin or archive made in Hermes Desktop (PATCH
    # /api/sessions/{id} over ``hermes serve``), by ``hermes sessions archive``, or
    # by the gateway left the WebUI sidecar stale and the two libraries disagreed.
    #
    # This pass adopts an externally-changed state.db flag into the sidecar that
    # backs it. It runs on its own cadence: the cheap fingerprint explicitly
    # excludes source='webui' rows (_WATCHER_EXCLUDED_SOURCES), so a flag change on
    # a WebUI-owned session is invisible to the change gate above and would never
    # reach the projection at all.
    def _read_db_flags(self) -> dict | None:
        """``{session_id: (pinned, archived)}`` for every listable state.db row."""
        db_path = self._state_db_path
        if not db_path.exists():
            return None
        try:
            with closing(open_state_db_readonly(db_path)) as conn:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
                if not {'id', 'pinned', 'archived'} <= cols:
                    return None
                rows = conn.execute(
                    "SELECT id, pinned, archived FROM sessions "
                    "WHERE source IS NOT NULL AND source != 'subagent'").fetchall()
            return {str(r[0]): (bool(r[1]), bool(r[2])) for r in rows}
        except Exception:
            logger.debug("flag-sync: state.db flag read failed", exc_info=True)
            return None

    def _read_db_flags_for(self, session_id: str) -> tuple[bool, bool] | None:
        """Re-read one row's flags (used under the session lock, post-transition)."""
        db_path = self._state_db_path
        if not db_path.exists():
            return None
        try:
            with closing(open_state_db_readonly(db_path)) as conn:
                row = conn.execute(
                    "SELECT pinned, archived FROM sessions WHERE id = ?",
                    (session_id,)).fetchone()
            return (bool(row[0]), bool(row[1])) if row else None
        except Exception:
            return None

    def _adopt_db_flags(self) -> list:
        """Adopt externally-changed pin/archive flags into WebUI sidecars.

        Returns the adopted session ids.

        Two modes. The FIRST pass after boot sweeps every sidecar-backed row (a
        change made while the WebUI was down has no transition to observe, so
        transitions alone would miss it). Later passes act on TRANSITIONS only: a
        row whose state.db value did not move is never touched, which is what keeps
        a failed patch-#18 mirror from being read as an external change and
        reverted.
        """
        flags = self._read_db_flags()
        if flags is None:
            return []
        prev = self._db_flag_snapshot
        self._db_flag_snapshot = flags
        boot_sweep = not prev
        candidates = flags.items() if boot_sweep else (
            (sid, value) for sid, value in flags.items() if prev.get(sid) != value)
        adopted = []
        for sid, value in candidates:
            try:
                if self._adopt_one(sid, *value):
                    adopted.append(sid)
            except Exception:
                logger.debug("flag-sync: adopt failed for %s", sid, exc_info=True)
        if adopted:
            logger.info("flag-sync: %s adopted %d flag(s) changed outside the WebUI: %s",
                        "boot sweep" if boot_sweep else "transition",
                        len(adopted), ", ".join(adopted[:5]))
        return adopted

    def _adopt_one(self, session_id: str, pinned: bool, archived: bool) -> bool:
        """Write one externally-changed flag pair into its sidecar, if one exists."""
        try:
            from api.config import SESSION_DIR, _get_session_agent_lock
            from api.models import get_session
            from api.session_events import publish_session_list_changed
        except Exception:
            return False
        sidecar = Path(SESSION_DIR) / f"{session_id}.json"
        if not sidecar.exists():
            # Projection rows (telegram/cli/cron) read these flags straight out of
            # state.db, so they already agree. Materializing a sidecar here would
            # create store entries the WebUI never asked for.
            return False
        # Cheap pre-check on the sidecar JSON itself: the boot sweep walks every
        # row, and loading a Session (messages and all) just to discard it would be
        # the expensive part of that sweep.
        try:
            on_disk = json.loads(sidecar.read_text())
            if (bool(on_disk.get('pinned')) == pinned
                    and bool(on_disk.get('archived')) == archived):
                return False
        except Exception:
            return False
        try:
            if time.time() - sidecar.stat().st_mtime < self.SIDECAR_SETTLE_SECS:
                # Written moments ago: patch #18 writes state.db right after the
                # sidecar, so this may be our own in-flight toggle, not theirs.
                return False
        except OSError:
            return False
        try:
            session = get_session(session_id)
        except Exception:
            return False
        if session is None:
            return False
        try:
            with _get_session_agent_lock(session_id):
                fresh = self._read_db_flags_for(session_id)
                if fresh is None:
                    return False
                pinned, archived = fresh
                if (bool(getattr(session, 'pinned', False)) == pinned
                        and bool(getattr(session, 'archived', False)) == archived):
                    return False
                session.pinned = pinned
                session.archived = archived
                session.save(touch_updated_at=False)
        except Exception:
            logger.warning("flag-sync: sidecar write failed for %s", session_id, exc_info=True)
            return False
        try:
            publish_session_list_changed("session_flag_sync", session_id=session_id)
        except Exception:
            logger.debug("flag-sync: publish failed for %s", session_id, exc_info=True)
        return True

    # ── Local patch 2026-10-11 (#20): inbound title sync (Desktop -> WebUI) ──
    #
    # Same shape as #19, one column family over. state.db's title is what the
    # Desktop and `hermes sessions list` display; the WebUI displays its sidecar
    # title. Renaming in either place used to move only one of them, so the two
    # surfaces listed the same conversation under two different names.
    def _read_db_titles(self) -> dict | None:
        """``{session_id: (title, title_source)}`` for every listable state.db row."""
        db_path = self._state_db_path
        if not db_path.exists():
            return None
        try:
            with closing(open_state_db_readonly(db_path)) as conn:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
                if not {'id', 'title', 'title_source'} <= cols:
                    return None
                rows = conn.execute(
                    "SELECT id, title, title_source FROM sessions "
                    "WHERE source IS NOT NULL AND source != 'subagent'").fetchall()
            return {str(r[0]): (r[1], r[2]) for r in rows}
        except Exception:
            logger.debug("flag-sync: state.db title read failed", exc_info=True)
            return None

    def _read_db_title_for(self, session_id: str) -> tuple | None:
        db_path = self._state_db_path
        if not db_path.exists():
            return None
        try:
            with closing(open_state_db_readonly(db_path)) as conn:
                row = conn.execute(
                    "SELECT title, title_source FROM sessions WHERE id = ?",
                    (session_id,)).fetchone()
            return (row[0], row[1]) if row else None
        except Exception:
            return None

    def _adopt_titles(self) -> list:
        """Adopt a HUMAN title set outside the WebUI into the sidecar.

        Only ``title_source='user'`` rows qualify. An automatic title in state.db
        must never overwrite what the WebUI is showing: the WebUI titles its own
        sessions (patch #20's outbound half pushes that down), so letting an ``llm``
        title flow back would make the two surfaces trade names forever. A user
        title, by contrast, only ever comes from a person renaming something --
        Hermes Desktop's rename dialog writes exactly that.
        """
        titles = self._read_db_titles()
        if titles is None:
            return []
        prev = self._db_title_snapshot
        self._db_title_snapshot = titles
        boot_sweep = not prev
        candidates = titles.items() if boot_sweep else (
            (sid, value) for sid, value in titles.items() if prev.get(sid) != value)
        adopted = []
        for sid, (title, source) in candidates:
            if source != 'user' or not title:
                continue
            try:
                if self._adopt_title_one(sid, title):
                    adopted.append(sid)
            except Exception:
                logger.debug("flag-sync: title adopt failed for %s", sid, exc_info=True)
        if adopted:
            logger.info("flag-sync: %s adopted %d renamed session(s): %s",
                        "boot sweep" if boot_sweep else "transition",
                        len(adopted), ", ".join(adopted[:5]))
        return adopted

    def _adopt_title_one(self, session_id: str, title: str) -> bool:
        try:
            from api.config import SESSION_DIR, _get_session_agent_lock
            from api.models import get_session
            from api.session_events import publish_session_list_changed
        except Exception:
            return False
        sidecar = Path(SESSION_DIR) / f"{session_id}.json"
        if not sidecar.exists():
            return False
        try:
            on_disk = json.loads(sidecar.read_text())
        except Exception:
            return False
        if str(on_disk.get('title') or '').strip() == str(title).strip():
            return False
        try:
            if time.time() - sidecar.stat().st_mtime < self.SIDECAR_SETTLE_SECS:
                return False
        except OSError:
            return False
        try:
            session = get_session(session_id)
        except Exception:
            return False
        if session is None:
            return False
        try:
            with _get_session_agent_lock(session_id):
                fresh = self._read_db_title_for(session_id)
                if not fresh or fresh[1] != 'user':
                    return False
                title = str(fresh[0] or '').strip()
                if not title or str(getattr(session, 'title', '') or '').strip() == title:
                    return False
                session.title = title
                # It IS a human title now, so protect it from the WebUI's own
                # adaptive re-titling the way a WebUI rename would.
                session.manual_title = True
                session.llm_title_generated = False
                session.save(touch_updated_at=False)
        except Exception:
            logger.warning("flag-sync: sidecar title write failed for %s", session_id, exc_info=True)
            return False
        try:
            publish_session_list_changed("session_title_sync", session_id=session_id)
        except Exception:
            logger.debug("flag-sync: title publish failed for %s", session_id, exc_info=True)
        return True

    def _poll_loop(self):
        """Main polling loop. Runs in a daemon thread."""
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.debug("Error in gateway watcher poll loop", exc_info=True)
            # Local patch 2026-10-11 (#19): inbound flag sync runs every tick, NOT
            # behind the fingerprint gate -- the gate excludes source='webui' rows
            # precisely because their content churn is not what it watches.
            try:
                self._adopt_db_flags()
            except Exception:
                logger.debug("Error in flag-sync pass", exc_info=True)
            # Local patch 2026-10-11 (#20): inbound title sync, same reasoning.
            try:
                self._adopt_titles()
            except Exception:
                logger.debug("Error in title-sync pass", exc_info=True)

            # Sleep in small increments so we can stop promptly
            for _ in range(self.POLL_INTERVAL * 10):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)


# ── Module-level watcher registry ──────────────────────────────────────────

_watchers: dict[str, GatewayWatcher] = {}
_watcher_lock = threading.Lock()

def _resolve_watcher_target(
    *,
    profile_name: str | None = None,
    hermes_home: Path | None = None,
) -> tuple[str, Path | None]:
    """Resolve the watcher profile/home pair for the current request context."""
    resolved_profile = str(profile_name or "").strip()
    resolved_home = Path(hermes_home).expanduser().resolve() if hermes_home is not None else None

    try:
        from api.profiles import get_active_profile_name, get_hermes_home_for_profile

        if not resolved_profile:
            resolved_profile = get_active_profile_name() or "default"
        if resolved_home is None and resolved_profile:
            resolved_home = Path(get_hermes_home_for_profile(resolved_profile)).expanduser().resolve()
    except Exception:
        if resolved_home is None:
            try:
                resolved_home = _get_state_db_path().parent.resolve()
            except Exception:
                resolved_home = None

    return resolved_profile, resolved_home


def _watcher_registry_key(profile_name: str | None = None, hermes_home: Path | None = None) -> str:
    """Return the stable registry key for a watcher target."""
    if hermes_home is not None:
        return str(Path(hermes_home).expanduser().resolve())
    return str(profile_name or "").strip() or "__default__"


def _watcher_has_subscribers(watcher: GatewayWatcher) -> bool:
    subscribers = getattr(watcher, "_subscribers", None)
    sub_lock = getattr(watcher, "_sub_lock", None)
    if subscribers is None or sub_lock is None:
        return False
    with sub_lock:
        return bool(subscribers)


def _pop_idle_watchers_locked(*, exclude_key: str) -> list[GatewayWatcher]:
    stale: list[GatewayWatcher] = []
    for key, watcher in list(_watchers.items()):
        if key == exclude_key or _watcher_has_subscribers(watcher):
            continue
        if _watchers.get(key) is watcher:
            stale.append(_watchers.pop(key))
    return stale


def start_watcher(*, profile_name: str | None = None, hermes_home: Path | None = None):
    """Start the watcher for the resolved profile home (idempotent)."""
    resolved_profile, resolved_home = _resolve_watcher_target(
        profile_name=profile_name,
        hermes_home=hermes_home,
    )
    key = _watcher_registry_key(resolved_profile, resolved_home)
    with _watcher_lock:
        watcher = _watchers.get(key)
        if watcher is None or not watcher.is_alive():
            if watcher is not None:
                watcher.stop()
            watcher = GatewayWatcher(profile_name=resolved_profile, hermes_home=resolved_home)
            watcher.start()
            _watchers[key] = watcher
        return watcher


def stop_watcher(*, profile_name: str | None = None, hermes_home: Path | None = None):
    """Stop either one profile watcher or the entire registry."""
    with _watcher_lock:
        if profile_name is None and hermes_home is None:
            watchers = list(_watchers.values())
            _watchers.clear()
        else:
            resolved_profile, resolved_home = _resolve_watcher_target(
                profile_name=profile_name,
                hermes_home=hermes_home,
            )
            key = _watcher_registry_key(resolved_profile, resolved_home)
            watcher = _watchers.pop(key, None)
            watchers = [watcher] if watcher is not None else []
    for watcher in watchers:
        watcher.stop()


def restart_watcher_for_profile(name: str):
    """Restart only the watcher pinned to the target profile home."""
    from api.profiles import get_hermes_home_for_profile

    hermes_home = Path(get_hermes_home_for_profile(name)).expanduser().resolve()
    key = _watcher_registry_key(name, hermes_home)
    watcher = GatewayWatcher(profile_name=name, hermes_home=hermes_home)
    watcher.start()
    with _watcher_lock:
        existing = _watchers.pop(key, None)
        stale_watchers = [] if existing is not None else _pop_idle_watchers_locked(exclude_key=key)
        _watchers[key] = watcher
    for old_watcher in ([existing] if existing is not None else stale_watchers):
        old_watcher.stop()
    return watcher


def get_watcher(*, profile_name: str | None = None, hermes_home: Path | None = None) -> GatewayWatcher | None:
    """Get or lazily start the watcher for the resolved request profile."""
    resolved_profile, resolved_home = _resolve_watcher_target(
        profile_name=profile_name,
        hermes_home=hermes_home,
    )
    key = _watcher_registry_key(resolved_profile, resolved_home)
    with _watcher_lock:
        watcher = _watchers.get(key)
    if watcher is None or not watcher.is_alive():
        watcher = start_watcher(profile_name=resolved_profile, hermes_home=resolved_home)
    return watcher
