"""Fail-closed guard for in-process Hermes Agent source revisions.

Hermes WebUI currently imports ``run_agent.AIAgent`` into its long-lived server
process. If the Agent checkout changes while that process is alive, Python may
combine already-cached modules with newly-read source. Refuse to reuse that
mixed runtime.

The guard stays fail-closed. What changed (local patch 2026-09-18) is only the
recovery: a detected change arms an in-place re-exec through
``api.updates._schedule_restart`` (same PID, no systemd service restart, waits
for in-flight work with no deadline), so the 409 cannot become a
manual-restart dead end.
"""

from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import stat
import sys
import subprocess
import threading
import time

# Retain the discovered path as a diagnostic/test-visible compatibility value;
# runtime identity is deliberately captured from the loaded module below.
from api.config import (
    PYTHON_EXE,
    _AGENT_DIR,  # noqa: F401
    _DEFAULT_STATE_HOME,
)
from api.subprocess_utils import windows_hide_flags

_RESTART_REQUIRED_MESSAGE = (
    "Hermes Agent was updated while Hermes WebUI was running. "
    "WebUI cannot verify that the Agent update completed safely. "
    "Check the Agent update outcome and environment first. "
    "Restart Hermes WebUI manually before retrying this action."
)
_AGENT_UPDATE_MARKER = ".hermes-update-in-progress"
_AGENT_RECOVERY_MARKERS = (".update-incomplete", ".lazy-refresh-incomplete")
_AGENT_UPDATE_MAX_AGE_SECONDS = 20 * 60
# Slow local Git reads must not look like source drift after just two seconds.
# Share one total budget across all reads; unreadable revisions fail closed.
_GIT_REVISION_TIMEOUT_SECONDS = 10
# The update marker holds a PID and a start timestamp (two short numeric lines).
# Anything larger is not a legitimate marker; cap the read so a huge or growing
# regular file can never exhaust memory on the stale-runtime request path.
_AGENT_UPDATE_MARKER_MAX_BYTES = 64 * 1024
# O_NOFOLLOW is POSIX; on platforms that lack it the fast os.open() path is not
# taken at all (see _MARKER_SAFE_OPEN_AVAILABLE below).
# The marker read hardening relies on two POSIX-only open flags to stay both
# non-blocking (never hang on a FIFO/device) and symlink-safe. O_NONBLOCK is
# Unix-only and O_NOFOLLOW is absent on some platforms; accessing them
# unconditionally raises AttributeError on native Windows. Resolve them safely
# and only take the os.open() fast path when BOTH are genuinely available —
# otherwise the read cannot prove non-blocking + no-follow and must fall back to
# an lstat-only classification (see _read_live_agent_update).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_MARKER_SAFE_OPEN_AVAILABLE = bool(getattr(os, "O_NOFOLLOW", 0)) and bool(
    getattr(os, "O_NONBLOCK", 0)
)
# Local patch 2026-09-18: recovery messages for the in-place reload path.
_RESTART_MESSAGE = (
    "Hermes Agent source changed while Hermes WebUI was running. "
    "Hermes WebUI is reloading onto the new revision automatically; "
    "retry in a few seconds (no restart needed)."
)

# Used when the reload is already queued but a running turn owns the process:
# the restart waits for that turn on purpose, so this is a wait, not a failure.
_RESTART_MESSAGE_BUSY = (
    "Hermes Agent source changed while Hermes WebUI was running. "
    "Reloading onto the new revision as soon as the running turn finishes; "
    "retry this action after it completes."
)
_HERMES_HOME = Path(_DEFAULT_STATE_HOME)
_AGENT_PYTHON = Path(PYTHON_EXE).expanduser() if PYTHON_EXE else None


# Local patch 2026-09-24 (TSK-143 / webui patch #15): the WebUI process builds
# its agents in-process, and upstream never calls
# agent.shell_hooks.register_from_config() for it, so a ``hooks:`` block in
# ~/.hermes/config.yaml (e.g. the pre_tool_call risk guard) was silently inert
# on every WebUI surface.  Register the configured shell hooks at boot —
# same pattern as gateway/run_startup.py.  Idempotent (the shell_hooks module
# keeps its own registered-key set), fail-open on any error: a hook wiring
# problem must never break WebUI startup.
_HOOKS_REGISTERED = False


def register_configured_shell_hooks() -> None:
    """Wire the config's ``hooks:`` block into this process (once)."""
    global _HOOKS_REGISTERED
    if _HOOKS_REGISTERED:
        return
    _HOOKS_REGISTERED = True
    try:
        from hermes_cli.config import load_config
        from agent.shell_hooks import register_from_config

        register_from_config(load_config())
    except Exception as exc:  # never break WebUI startup over hook wiring
        print(f"[!!] shell-hook registration failed (non-fatal): {exc}", flush=True)


def _read_agent_revision(
    agent_dir: Path | None,
    *,
    module_path: Path | None = None,
) -> str | None:
    """Return the tracked Agent HEAD within one total Git-read budget."""
    deadline = time.monotonic() + _GIT_REVISION_TIMEOUT_SECONDS

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("git", _GIT_REVISION_TIMEOUT_SECONDS)
        return remaining

    if agent_dir is None:
        return None

    if module_path is None:
        module = sys.modules.get("run_agent")
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return None
        try:
            module_path = Path(module_file).resolve()
        except (OSError, RuntimeError, TypeError):
            return None

    try:
        worktree_result = subprocess.run(
            ["git", "-C", str(agent_dir), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining_timeout(),
            creationflags=windows_hide_flags(),
        )
        if worktree_result.returncode != 0:
            return None
        worktree = Path(worktree_result.stdout.strip()).resolve()
        relative_module = module_path.relative_to(worktree).as_posix()
        tracked_result = subprocess.run(
            [
                "git",
                "--literal-pathspecs",
                "-C",
                str(worktree),
                "ls-files",
                "--error-unmatch",
                "--",
                relative_module,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining_timeout(),
            creationflags=windows_hide_flags(),
        )
        if tracked_result.returncode != 0:
            return None
        revision_result = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining_timeout(),
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError):
        return None

    # Scheduling/process-creation overhead can exceed a subprocess timeout.
    if time.monotonic() >= deadline:
        return None
    revision = revision_result.stdout.strip()
    return revision if revision_result.returncode == 0 and revision else None


def _read_checkout_head(agent_dir: Path | None) -> str | None:
    """Read HEAD without requiring ``run_agent`` (boot banner only)."""
    if agent_dir is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(agent_dir), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    revision = (result.stdout or "").strip()
    return revision if result.returncode == 0 and revision else None


def loaded_agent_revision_banner() -> str:
    """One-line boot record of the Agent checkout this image will load.

    Replaces the retired polling watcher: this is the boot-side signal that a
    fresh process image came up (used by the acceptance test to detect the
    in-place re-exec), with no polling thread and no timer.
    """
    if _AGENT_DIR is None:
        return "no agent dir; revision guard disabled"
    revision = _read_checkout_head(Path(_AGENT_DIR))
    if revision is None:
        return f"{_AGENT_DIR} (not a git work tree; revision guard disabled)"
    return f"{_AGENT_DIR} @ {revision[:12]} (in-place reload armed on demand)"


_AGENT_SOURCE_DIR: Path | None = None
_AGENT_MODULE_PATH: Path | None = None
_AGENT_REVISION: str | None = None
_AIAgent = None
_RUNTIME_LOCK = threading.Lock()

# Local patch 2026-09-18 (agent-revision guard): state for the in-place restart
# armed when a stale revision is detected.  One arm per stale episode, with a
# re-arm window so a restart that never landed cannot brick the recovery.
_RESTART_ARM_LOCK = threading.Lock()
_RESTART_ARMED_AT: float | None = None
_RESTART_REARM_AFTER_SECONDS = 300.0


class AgentRuntimeChangedError(RuntimeError):
    """Raised when the loaded Agent runtime no longer matches its source tree."""

    def __init__(
        self,
        message: str,
        *,
        agent_update_state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_update_state = agent_update_state


def agent_runtime_stale_payload(exc: AgentRuntimeChangedError) -> dict:
    """Return the shared retry response for every stale-runtime entry point."""
    payload = {
        "error": str(exc),
        "type": "agent_runtime_stale",
        "retryable": True,
        "restart_scheduled": False,
    }
    if exc.agent_update_state is not None:
        payload["agent_update_state"] = exc.agent_update_state
    return payload


def _pid_is_alive(pid: int) -> bool | None:
    """Return PID liveness, or ``None`` when the platform cannot confirm it."""
    if pid <= 0:
        return False
    if pid.bit_length() > 32:
        return None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            )
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                error = ctypes.get_last_error()
                if error == 5:  # ERROR_ACCESS_DENIED still proves the PID exists.
                    return True
                if error == 87:  # ERROR_INVALID_PARAMETER for a missing PID.
                    return False
                return None
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return None
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, ValueError):
        return None
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return None
    return True


def _read_live_agent_update(marker: Path) -> str:
    """Classify the shared Agent update marker without changing Agent state.

    The marker is attacker-adjacent shared state (any process that can write the
    Agent home can create it), so the read is hardened: never follow a symlink,
    never block on a FIFO/device, and never read an unbounded regular file.
    Anything that is not a small regular file is classified ``unknown`` rather
    than allowed to hang or exhaust memory on a stale-runtime request path.
    """
    if not _MARKER_SAFE_OPEN_AVAILABLE:
        # Without both O_NONBLOCK and O_NOFOLLOW we cannot prove the read is
        # non-blocking and symlink-safe (e.g. native Windows), so never open the
        # marker: an unverifiable marker fails closed to ``unknown``, and only a
        # genuinely missing path is ``absent``.
        try:
            marker.lstat()
        except FileNotFoundError:
            return "absent"
        except (OSError, ValueError, TypeError):
            return "unknown"
        return "unknown"
    try:
        fd = os.open(marker, os.O_RDONLY | _O_NONBLOCK | _O_NOFOLLOW)
    except FileNotFoundError:
        try:
            marker.lstat()
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unknown"
        # Path exists to lstat (e.g. a dangling/looping symlink) but O_NOFOLLOW
        # refused to open it — treat as an unverifiable marker.
        return "unknown"
    except (OSError, ValueError, TypeError):
        # ELOOP (symlink under O_NOFOLLOW), ENXIO/EWOULDBLOCK (FIFO with no
        # writer under O_NONBLOCK), a non-path marker object, or any other open
        # failure — fail closed: an unreadable marker is never proof of safety.
        return "unknown"

    try:
        try:
            st = os.fstat(fd)
        except OSError:
            return "unknown"
        if not stat.S_ISREG(st.st_mode):
            # FIFO, device, directory, socket — never a legitimate marker.
            return "unknown"
        if st.st_size > _AGENT_UPDATE_MARKER_MAX_BYTES:
            return "unknown"
        try:
            # Read one byte past the cap so an oversized file that lied about
            # st_size (or grew mid-read) is still rejected rather than truncated.
            data = os.read(fd, _AGENT_UPDATE_MARKER_MAX_BYTES + 1)
        except (OSError, BlockingIOError):
            return "unknown"
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    if len(data) > _AGENT_UPDATE_MARKER_MAX_BYTES:
        return "unknown"
    try:
        raw = data.decode("utf-8")
    except UnicodeError:
        return "unknown"

    lines = raw.splitlines()
    try:
        pid = int(lines[0].strip())
        started_at = float(lines[1].strip())
    except (IndexError, TypeError, ValueError):
        return "unknown"
    if pid <= 0 or not math.isfinite(started_at):
        return "unknown"

    age_seconds = time.time() - started_at
    if age_seconds < 0:
        return "unknown"
    if age_seconds > _AGENT_UPDATE_MAX_AGE_SECONDS:
        return "stale"
    alive = _pid_is_alive(pid)
    if alive is None:
        return "unknown"
    return "active" if alive else "stale"


def _marker_presence(marker: Path) -> str:
    """Return ``present``, ``absent``, or ``unknown`` for a recovery marker."""
    try:
        marker.lstat()
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unknown"
    return "present"


def _agent_install_roots() -> tuple[Path, ...]:
    """Return portable roots that can own the Agent's venv recovery markers."""
    candidates: list[Path] = []
    if _AGENT_SOURCE_DIR is not None:
        candidates.append(_AGENT_SOURCE_DIR)
    if _AGENT_PYTHON is not None:
        # A venv Python is commonly a symlink to a shared interpreter. Keep the
        # configured venv path so its installation's recovery markers are read.
        python_path = _AGENT_PYTHON
        if python_path.parent.name.lower() in {"bin", "scripts"}:
            venv_dir = python_path.parent.parent
            if venv_dir.name.lower() in {"venv", ".venv"}:
                candidates.append(venv_dir.parent)

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key not in seen:
            seen.add(key)
            roots.append(candidate)
    return tuple(roots)


def _agent_update_transaction_state() -> str:
    """Report marker diagnostics, never proof of successful completion.

    The Agent removes its active marker on failed/interrupted exits too. Neither
    its absence nor a stale PID proves the checkout or environment is healthy.
    """
    live_state = _read_live_agent_update(_HERMES_HOME / _AGENT_UPDATE_MARKER)
    if live_state == "unknown":
        return "unknown"

    recovery_present = False
    for root in _agent_install_roots():
        for marker_name in _AGENT_RECOVERY_MARKERS:
            presence = _marker_presence(root / marker_name)
            if presence == "unknown":
                return "unknown"
            recovery_present = recovery_present or presence == "present"
    if recovery_present:
        return "incomplete"
    return "unverified" if live_state == "absent" else live_state


def _loaded_agent_source_identity() -> tuple[Path, Path] | None:
    """Return the source directory and file that supplied ``run_agent``."""
    module = sys.modules.get("run_agent")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    try:
        module_path = Path(module_file).resolve()
        return module_path.parent, module_path
    except (OSError, RuntimeError, TypeError):
        return None


def _capture_loaded_agent_revision() -> None:
    """Bind the guard to the checkout that supplied the loaded Agent module."""
    global _AGENT_SOURCE_DIR, _AGENT_MODULE_PATH, _AGENT_REVISION

    if _AGENT_REVISION is not None:
        ensure_agent_runtime_current()
        return

    identity = _loaded_agent_source_identity()
    if identity is None:
        return
    source_dir, module_path = identity
    current_revision = _read_agent_revision(source_dir, module_path=module_path)
    _AGENT_SOURCE_DIR = source_dir
    _AGENT_MODULE_PATH = module_path
    _AGENT_REVISION = current_revision


def _restart_is_blocked() -> bool:
    """True while a running turn/stream would hold the armed reload back."""
    try:
        from api.updates import _restart_blocker_snapshot  # noqa: PLC0415

        return bool(_restart_blocker_snapshot().get("restart_blocked"))
    except Exception:  # noqa: BLE001
        # Unknown state is not "long-running work"; the reload itself refuses to
        # proceed while it cannot prove the process is idle.
        return False


def _arm_in_place_restart() -> None:
    """Arm an in-place re-exec so the next action loads the current source.

    ``api.updates._schedule_restart`` purges stale bytecode and ``os.execv``s
    the process image (same PID, no systemd service restart, on-disk sessions
    untouched), and it waits for in-flight chat work first.  ``max_wait_seconds
    =None`` keeps that wait unbounded on purpose: no caller is blocked on this
    restart, so a running turn must never be preempted.

    Deliberately best-effort.  The caller still raises
    ``AgentRuntimeChangedError``, so any failure here degrades to the previous
    manual-restart behaviour and can never serve a mixed runtime.  Never blocks
    and never takes ``_RUNTIME_LOCK`` (callers may already hold it).
    """
    global _RESTART_ARMED_AT

    now = time.monotonic()
    with _RESTART_ARM_LOCK:
        if (
            _RESTART_ARMED_AT is not None
            and now - _RESTART_ARMED_AT < _RESTART_REARM_AFTER_SECONDS
        ):
            return
        _RESTART_ARMED_AT = now

    try:
        from api.updates import _schedule_restart  # noqa: PLC0415

        _schedule_restart(2.0, max_wait_seconds=None)
        print(
            "[ok] agent-runtime guard: stale agent revision armed an in-place "
            "restart; no manual restart needed",
            flush=True,
        )
    except Exception as exc:  # never mask the stale-runtime error below
        print(
            f"[!!] agent-runtime guard: could not arm in-place restart: {exc}",
            flush=True,
        )


def ensure_agent_runtime_current() -> None:
    """Reject a known Git checkout change instead of mixing Python modules."""
    if _AGENT_REVISION is None:
        return
    fresh_revision = None
    try:
        fresh_revision = _read_agent_revision(
            _AGENT_SOURCE_DIR, module_path=_AGENT_MODULE_PATH
        )
    except Exception:
        # An unreadable revision is indistinguishable from a changed one, so a
        # failed read must fail CLOSED like every other identity-loss shape
        # (deleted, permission-denied, empty, corrupt, removed directory).
        # Letting the raw exception escape would surface as an HTTP 500 instead
        # of the typed stale-runtime response the barrier is meant to produce.
        fresh_revision = None
    if fresh_revision == _AGENT_REVISION:
        return

    # Local patch 2026-09-18: recover automatically instead of dead-ending on
    # the 409. The guard itself stays fail-closed. The message tells the truth
    # about which of the two states this is: reloading now, or reloading once a
    # long-running turn releases the process.
    _arm_in_place_restart()
    raise AgentRuntimeChangedError(
        _RESTART_MESSAGE_BUSY if _restart_is_blocked() else _RESTART_MESSAGE,
        agent_update_state=_agent_update_transaction_state(),
    )


def require_ai_agent_class():
    """Import ``AIAgent`` after proving the loaded source revision is current."""
    ensure_agent_runtime_current()
    from run_agent import AIAgent  # noqa: PLC0415

    _capture_loaded_agent_revision()
    return AIAgent


def get_ai_agent_class():
    """Return ``AIAgent`` while preserving the existing lazy-import retry."""
    global _AIAgent, _AGENT_REVISION

    with _RUNTIME_LOCK:
        ensure_agent_runtime_current()
        if _AIAgent is None:
            try:
                agent_class = require_ai_agent_class()
            except ImportError:
                return None
            _AIAgent = agent_class
        return _AIAgent
