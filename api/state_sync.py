"""
Hermes Web UI -- state.db sync bridge.

Mirrors WebUI session metadata (token usage, title, model) into the
hermes-agent state.db so that /insights, session lists, and cost
tracking include WebUI activity.

Usage/title mirroring is opt-in via the 'sync_to_insights' setting
(default: off). ``sync_session_cwd`` is not: it only fills the workspace
into the row the Agent already created, so clients that group sessions by
``cwd`` (Hermes Desktop) place WebUI sessions under their workspace.
All operations are wrapped in try/except -- if state.db is unavailable,
locked, or the schema doesn't match, the WebUI continues normally.

The bridge uses absolute token counts (not deltas) because the WebUI
Session object already accumulates totals across turns. This avoids
any double-counting risk.
"""
import logging
import ntpath
import os
import threading
import time
from pathlib import Path
from typing import Optional

# ── Local patch #23 (2026-10-11): push attribution for the flag bridge ─────────
# The inbound watcher (api/gateway_watcher.py) adopts state.db changes into the
# sidecars, but a difference it never SEES cannot be attributed: two opposite
# edits to the same session inside one watcher interval (5s) cancel out, so
# state.db looks unchanged and the WebUI side keeps its value -- the Desktop's
# edit was silently lost. Recording what we successfully pushed lets the watcher
# read the asymmetry: if state.db no longer holds the value WE wrote, someone
# else changed it afterwards, so the db is newer and wins.
PUSH_LEDGER: dict = {}
PUSH_LEDGER_MAX_AGE = 600.0
_PUSH_LEDGER_MAX_ENTRIES = 200


def _record_push(session_id: str, attr: str, value: bool) -> None:
    entry = PUSH_LEDGER.setdefault(str(session_id), {})
    entry[attr] = (bool(value), time.time())
    if len(PUSH_LEDGER) > _PUSH_LEDGER_MAX_ENTRIES:
        # drop the oldest sessions, never the newest pushes
        order = sorted(PUSH_LEDGER, key=lambda k: max(t for _, t in PUSH_LEDGER[k].values()))
        for key in order[: len(PUSH_LEDGER) - _PUSH_LEDGER_MAX_ENTRIES]:
            PUSH_LEDGER.pop(key, None)


def pushed_recently(session_id: str, max_age: float = PUSH_LEDGER_MAX_AGE) -> dict:
    """``{attr: (value, age_seconds)}`` for pushes that actually landed."""
    entry = PUSH_LEDGER.get(str(session_id))
    if not entry:
        return {}
    now = time.time()
    return {attr: (value, round(now - at, 1))
            for attr, (value, at) in entry.items() if now - at <= max_age}


def clear_pushes(session_id: str) -> None:
    PUSH_LEDGER.pop(str(session_id), None)

logger = logging.getLogger(__name__)


def _get_state_db(profile: Optional[str] = None):
    """Get a SessionDB instance for a profile's state.db.

    When ``profile`` is provided the function resolves *that* profile's
    home directory directly (via ``_resolve_profile_home_for_name``).
    If resolution fails (unknown profile name, IO error, etc.) the
    function returns ``None`` rather than silently falling back to
    ``HERMES_HOME`` — silently routing the write to the wrong DB
    would defeat the point of the explicit-profile path (#2762).

    When ``profile`` is None it falls back to the TLS-based
    ``get_active_hermes_home()`` lookup for backward compatibility,
    with a final ``HERMES_HOME`` fallback only on that path. TLS may be
    unset in background/worker threads, in which case the lookup falls
    through to the process-global active profile and can write to the
    wrong DB. Callers that know the session's profile (e.g.
    ``sync_session_usage`` after a stream completes on a background
    thread) should pass it explicitly to avoid that race.

    Returns None if hermes_state is not importable, the explicit
    profile cannot be resolved, or the DB is unavailable. Each caller
    is responsible for calling db.close() when done.
    """
    try:
        from hermes_state import SessionDB
    except ImportError:
        return None

    if profile is not None:
        # Explicit-profile path — a resolution failure here MUST NOT
        # silently fall back to HERMES_HOME or the caller's "write to
        # the named profile" contract is broken (the original #2762
        # symptom: writes leaking into the wrong profile's state.db).
        #
        # Defense-in-depth (per #2827 maintainer review): validate the
        # name shape BEFORE handing it to ``_resolve_profile_home_for_name``.
        # The resolver itself rarely raises — for an invalid-but-non-
        # malicious name (e.g. one that fails ``_PROFILE_ID_RE``) it
        # quietly returns ``_DEFAULT_HERMES_HOME``, which is the exact
        # leak we're trying to prevent on the explicit-profile path.
        # Validating up-front turns that quiet leak into an explicit
        # "refuse + log + return None" so the contract is "write to
        # the EXACT named profile, or write nowhere."
        try:
            from api.profiles import (
                _resolve_profile_home_for_name,
                _PROFILE_ID_RE,
                _is_root_profile,
            )
            if not (_is_root_profile(profile) or _PROFILE_ID_RE.fullmatch(profile)):
                logger.warning(
                    "state_sync: refusing invalid profile name %r — skipping "
                    "write rather than leaking to the default state.db (#2762).",
                    profile,
                )
                return None
            hermes_home = Path(_resolve_profile_home_for_name(profile)).expanduser().resolve()
        except Exception:
            logger.warning(
                "state_sync: could not resolve profile %r — skipping write rather "
                "than leaking to the active profile (#2762).", profile,
            )
            return None
    else:
        # Implicit / TLS-fallback path — preserves pre-#2762 behavior
        # for any caller that doesn't pass profile= explicitly.
        try:
            from api.profiles import get_active_hermes_home
            hermes_home = Path(get_active_hermes_home()).expanduser().resolve()
        except Exception:
            logger.debug("Failed to resolve hermes home, using default")
            hermes_home = Path(os.getenv('HERMES_HOME', str(Path.home() / '.hermes')))

    db_path = hermes_home / 'state.db'
    if not db_path.exists():
        return None

    try:
        return SessionDB(db_path)
    except Exception:
        logger.debug("Failed to open state.db")
        return None


def sync_session_start(session_id: str, model=None, profile: Optional[str] = None) -> None:
    """Register a WebUI session in state.db (idempotent).
    Called when a session's first message is sent.

    ``profile`` lets the caller name the target state.db explicitly,
    avoiding the TLS-vs-background-thread mismatch in #2762. When
    omitted, the active profile is resolved from TLS (then process
    globals) as before.
    """
    db = _get_state_db(profile=profile)
    if not db:
        return
    try:
        db.ensure_session(
            session_id=session_id,
            source='webui',
            model=model,
        )
    except Exception:
        logger.debug("Failed to sync session start to state.db")
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Failed to close state.db")


def _normalize_session_cwd(workspace) -> str:
    """Canonical text form of a WebUI workspace for ``sessions.cwd``.

    Trailing ``/`` and ``\\`` separators are stripped (``/a/b/`` and ``/a/b``
    are one workspace) so equality checks and prefix grouping stay stable.
    Whitespace is never trimmed: it can be part of a real name. A path
    that is only an anchor is returned untouched: ``/``, a drive root such
    as ``C:\\`` (``C:`` would be drive-relative, a different path) and a UNC
    share root such as ``\\\\host\\share\\``.
    """
    text = str(workspace or "")
    # ``strip()`` only detects blank input: surrounding whitespace is part of a
    # valid directory name (``/x/acme `` is not ``/x/acme``) and is kept.
    if not text.strip():
        return ""
    _drive, rest = ntpath.splitdrive(text)
    if not rest.strip("/\\"):
        return text
    return text.rstrip("/\\")


def sync_session_cwd(session_id: str, workspace, profile: Optional[str] = None, db=None) -> bool:
    """Mirror a WebUI session's workspace into ``sessions.cwd`` in state.db.

    The agent creates the state.db row lazily on the first turn but only
    stamps ``cwd`` for CLI-family sources, so WebUI rows were left with an
    empty ``cwd`` and clients that group sessions by working directory
    (Hermes Desktop) filed them under "Home" instead of their workspace.

    Only an EXISTING row whose ``source`` is ``webui`` is updated; this never
    creates one, so sessions that
    never sent a message stay out of state.db exactly as before. A row that
    already records the same ``cwd`` is left untouched, so the per-row Git
    metadata generation is only bumped on a real workspace change (where the
    stale ``git_branch``/``git_repo_root`` are cleared by
    ``update_session_cwd``). Not gated by ``sync_to_insights``: the row is
    written by the agent regardless of that setting.

    ``db`` lets the streaming path reuse the agent's own SessionDB (already
    bound to the session's profile); it is not closed here. Otherwise the
    profile's state.db is opened via ``_get_state_db(profile=...)`` (#2762).
    Returns True when a row was updated.
    """
    cwd = _normalize_session_cwd(workspace)
    if not session_id or not cwd:
        return False
    owns_db = db is None
    if owns_db:
        # A legacy ``profile=None`` session lives in the root home. Without an
        # explicit name ``_get_state_db`` would fall back to the process-active
        # profile, which this background write must never read.
        db = _get_state_db(profile=profile or "default")
    if not db:
        return False
    try:
        if not hasattr(db, "update_session_cwd"):
            return False
        row = db.get_session(session_id)
        if not row:
            return False
        # Only rows the WebUI owns: an imported or foreign session (CLI, TUI,
        # Desktop, gateway) keeps its own working directory and git metadata.
        if row.get("source") != "webui":
            return False
        if _normalize_session_cwd(row.get("cwd")) == cwd:
            return False
        return db.update_session_cwd(session_id, cwd) is not None
    except Exception:
        logger.debug("Failed to sync session cwd to state.db for %s", session_id)
        return False
    finally:
        if owns_db:
            try:
                db.close()
            except Exception:
                logger.debug("Failed to close state.db")


_CWD_SYNC_LOCK = threading.Lock()
_CWD_SYNC_THREADS: set = set()
_CWD_SYNC_THREADS_LOCK = threading.Lock()


def sync_session_cwd_background(resolve) -> threading.Thread:
    """Run :func:`sync_session_cwd` off the caller's thread.

    The cwd mirror is optional metadata, but ``SessionDB`` writes retry for up
    to ~20 s under contention. Stream cleanup and the workspace-update response
    must not wait for that, so the write runs on a daemon thread.

    ``resolve`` returns ``(session_id, workspace, profile)`` or ``None`` and is
    called *when the write runs*, under a module lock that serialises these
    writes. It must look the session up by id at that moment rather than close
    over a ``Session`` object: that object can be replaced (LRU eviction, disk
    reload) between scheduling and running, and a stale one would write an
    older workspace. Resolved that way, overlapping syncs converge on the
    latest value whatever order they run in. Failures are
    logged at debug level and never reach the caller.
    """
    def _worker():
        try:
            with _CWD_SYNC_LOCK:
                target = resolve()
                if target:
                    session_id, workspace, profile = target
                    sync_session_cwd(session_id, workspace, profile=profile)
        except Exception:
            logger.debug("Background session cwd sync failed", exc_info=True)
        finally:
            with _CWD_SYNC_THREADS_LOCK:
                _CWD_SYNC_THREADS.discard(threading.current_thread())

    thread = threading.Thread(target=_worker, name="webui-cwd-sync", daemon=True)
    with _CWD_SYNC_THREADS_LOCK:
        _CWD_SYNC_THREADS.add(thread)
    thread.start()
    return thread


def drain_cwd_syncs(timeout: float = 5.0) -> bool:
    """Wait for in-flight background cwd syncs (tests, orderly shutdown)."""
    with _CWD_SYNC_THREADS_LOCK:
        pending = list(_CWD_SYNC_THREADS)
    for thread in pending:
        thread.join(timeout)
    return not any(thread.is_alive() for thread in pending)


def sync_session_usage(session_id: str, input_tokens: int=0, output_tokens: int=0,
                       estimated_cost=None, model=None, title: Optional[str] = None,
                       message_count: Optional[int] = None, profile: Optional[str] = None,
                       cache_read_tokens: int = 0, cache_write_tokens: int = 0,
                       api_call_count: Optional[int] = None) -> None:
    """Update token usage and title for a WebUI session in state.db.
    Called after each turn completes. Uses absolute=True to set totals
    (the WebUI Session already accumulates across turns).

    ``profile`` lets the caller name the target state.db explicitly,
    which is what fixes #2762: this function is invoked from the
    agent streaming worker thread, where the request-thread's TLS
    profile context has not been propagated. Without an explicit
    profile, the TLS lookup falls back to the process-global active
    profile and writes the session's usage to the wrong state.db
    (e.g. ``hiyuki``'s instead of the cookie-switched ``maiko``'s).
    """
    db = _get_state_db(profile=profile)
    if not db:
        return
    try:
        # Ensure session exists first (idempotent)
        db.ensure_session(session_id=session_id, source='webui', model=model)
        # Set absolute token counts. WebUI's sidecar already accumulates
        # input/output/cache totals across turns, so mirror the same absolute
        # values into state.db. Omitting cache counters makes insights/reporting
        # show false 0% hit rates even when the live stream and sidecar saw warm
        # prefix reads.
        db.update_token_counts(
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            api_call_count=api_call_count,
            estimated_cost_usd=estimated_cost,
            model=model,
            absolute=True,
        )
        # Update title if we have one, using the public API
        if title:
            try:
                db.set_session_title(session_id, title)
            except Exception:
                logger.debug("Failed to sync session title to state.db")
        # Update message count
        if message_count is not None:
            try:
                def _set_msg_count(conn):
                    conn.execute(
                        "UPDATE sessions SET message_count = ? WHERE id = ?",
                        (message_count, session_id),
                    )
                db._execute_write(_set_msg_count)
            except Exception:
                logger.debug("Failed to sync message count to state.db")
    except Exception:
        logger.debug("Failed to sync session usage to state.db")
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Failed to close state.db")


def _push_title(db, session_id: str, title: str, *, user: bool) -> bool:
    """Write one title into state.db, deriving a lineage variant if the UNIQUE
    ``idx_sessions_title_unique`` rejects it (two conversations may legitimately
    want the same name; the second becomes "<title> #2")."""
    def _write(text: str) -> bool:
        if user:
            return bool(db.set_session_title(session_id, text))
        return bool(db.set_auto_title(session_id, text, source=db.TITLE_SOURCE_LLM))

    try:
        return _write(title)
    except ValueError:
        try:
            alt = db.get_next_title_in_lineage(title)
        except Exception:
            return False
        return bool(alt and alt != title and _write(alt))


def sync_session_title(session_id: str, title: str, profile: Optional[str] = None) -> None:
    """Sync an auto-generated title to state.db (not gated by sync_to_insights).

    Background title generation writes the title to the WebUI sidecar JSON but
    not to hermes-agent's state.db, so ``hermes sessions list`` (and the Desktop
    app's session list) shows a different name for the same conversation.

    Local patch #20 (2026-10-11) — this used to be ``set_auto_title_if_empty``,
    which fills a NULL title and then stops. The observable effect was that the
    WebUI's title and the Desktop's title drifted apart permanently for any
    conversation the agent had already auto-titled (measured: 322 of 524 WebUI
    sessions), so the two surfaces listed the same chat under two names.

    Now: fill when untitled, and when state.db already holds an automatic title
    that differs, upgrade it to the WebUI's text. Provenance is respected in both
    directions — a ``user`` title (a human named it via CLI/TUI/gateway/Desktop)
    is NEVER overwritten, and an automatic title is written as ``llm`` so it keeps
    its honest rank and stays replaceable.

    On a title collision (two sessions with the same auto-title), the title is
    de-duplicated via ``get_next_title_in_lineage`` (e.g. "My Session" ->
    "My Session #2") and retried, so the second session is never left blank.
    """
    if not title:
        return
    db = _get_state_db(profile=profile)
    if not db:
        return
    try:
        # Ensure the session row exists (idempotent) so the UPDATE has a target.
        db.ensure_session(session_id=session_id, source='webui')
        # hermes-agent's SessionDB.set_auto_title_if_empty was renamed to
        # set_auto_title(session_id, title, *, source) in the state-module
        # split (agent commit 53db597201, released v2026.9.7). set_auto_title
        # preserves the same "only populate NULL / never clobber a manual
        # rename" semantics (returns False, untouched, when a higher-authority
        # title holds the row) and requires an explicit auto source.
        _llm_source = getattr(db, "TITLE_SOURCE_LLM", "llm")
        try:
            current = db.get_session_title(session_id)
            source = db.get_session_title_source(session_id)
        except Exception:
            current, source = None, None
        if source == db.TITLE_SOURCE_USER:
            # A human named this conversation elsewhere; the sidecar is the stale
            # side, not state.db.
            return
        if current == title:
            return
        if source is not None:
            # An automatic title (llm/derived) already holds the row and the ranks
            # are equal, so a plain re-write would be a no-op by design. Drop the
            # provenance one rank so the WebUI's newer title can land as ``llm``.
            try:
                db.set_session_title_source(session_id, db.TITLE_SOURCE_DERIVED)
            except Exception:
                return
        _push_title(db, session_id, title, user=False)
    except Exception:
        logger.debug("Failed to sync session title to state.db for %s", session_id)
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Failed to close state.db")


def sync_session_title_manual(session_id: str, title: str, profile: Optional[str] = None) -> None:
    """Mirror a MANUAL WebUI rename into state.db under ``user`` authority.

    Local patch #20 (2026-10-11). ``/api/session/rename`` wrote only the sidecar,
    so a session renamed in the WebUI kept its old auto-title in state.db and the
    Desktop app listed it under that old name (live case: a session renamed
    "Pyrowave" showed up in the Desktop as "Household high speed internet
    benefits" — i.e. looked like it was missing).

    ``TITLE_SOURCE_USER`` outranks every automatic title, so this always lands
    (``_set_session_title`` allows a user write over any provenance), and it also
    protects the name from later automatic re-titling on either surface.

    Best effort: never raises into the request path.
    """
    if not title:
        return
    db = _get_state_db(profile=profile)
    if not db:
        return
    try:
        db.ensure_session(session_id=session_id, source='webui')
        current = db.get_session_title(session_id)
        if current == title and db.get_session_title_source(session_id) == db.TITLE_SOURCE_USER:
            return
        _push_title(db, session_id, title, user=True)
    except Exception:
        logger.debug("Failed to sync manual title to state.db for %s", session_id)
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Failed to close state.db")


def _ensure_state_db_row(db, session_id: str, profile: Optional[str] = None) -> bool:
    """Create the state.db row for a WebUI session that does not have one yet.

    Local patch #24 (2026-10-11, Allen: "make them visible just in case"): a chat
    created in the WebUI exists only as a sidecar until the agent actually runs a
    turn in it. state.db -- and therefore Hermes Desktop -- cannot see it, and a
    pin/archive/rename made here has nowhere to mirror to. The mirror is the moment
    the Desktop needs the row, so mint it there.

    Uses the agent's own ``ensure_session()`` upsert, which never overwrites a field
    an earlier writer set: if the agent already materialized the session this is a
    no-op. ``cwd`` is deliberately left unset -- every existing webui row has NULL
    cwd (513/513 measured) and the Desktop lists them all fine.
    """
    try:
        db.ensure_session(session_id, "webui", profile_name=profile or None)
        return True
    except Exception:
        logger.warning("state_sync: could not materialize %s for the Desktop",
                       session_id, exc_info=True)
        return False


def _write_flag(db, session_id: str, attr: str, value: bool,
                profile: Optional[str] = None) -> bool:
    """Write one flag, materializing the row first if the session has none."""
    setter = db.set_session_pinned if attr == "pinned" else db.set_session_archived
    for attempt in ("write", "materialize+retry"):
        if setter(session_id, value):
            _record_push(session_id, attr, bool(value))
            return True
        if attempt == "materialize+retry" or not _ensure_state_db_row(db, session_id, profile):
            break
    logger.warning("state_sync: no state.db row matched %s=%s for %s", attr, value, session_id)
    return False


def sync_session_flags(session_id: str, *, pinned: Optional[bool] = None,
                       archived: Optional[bool] = None,
                       profile: Optional[str] = None) -> None:
    """Mirror a WebUI pin/archive toggle into the agent state.db.

    The WebUI keeps pin/archive in its own sidecar JSON, but the Desktop app and
    every ``hermes serve`` backend read those two flags from state.db
    (``hermes_cli/web_routers/sessions.py``: GET ``/api/sessions``,
    PATCH ``/api/sessions/{id}``). Without this bridge a pin or archive made on
    the phone is invisible on the Desktop -- and vice versa for a pin made there.

    Both flags ride the agent's own lineage-wide UPDATE (a compression lineage is
    ONE conversation, so continuation siblings and subagent children follow the
    root). Idempotent and best effort: never raises into the request path.
    """
    if pinned is None and archived is None:
        return
    db = _get_state_db(profile=profile)
    if not db:
        return
    try:
        if pinned is not None:
            _write_flag(db, session_id, "pinned", bool(pinned), profile)
        if archived is not None:
            _write_flag(db, session_id, "archived", bool(archived), profile)
    except Exception:
        logger.debug("Failed to sync session flags to state.db for %s", session_id)
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Failed to close state.db")
