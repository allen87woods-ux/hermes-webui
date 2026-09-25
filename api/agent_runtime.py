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

from pathlib import Path
import sys
import subprocess
import threading
import time

# Retain the discovered path as a diagnostic/test-visible compatibility value;
# runtime identity is deliberately captured from the loaded module below.
from api.config import _AGENT_DIR  # noqa: F401
from api.subprocess_utils import windows_hide_flags

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
    """Return the loaded Agent checkout HEAD, or ``None`` if it is not tracked."""
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
            timeout=2,
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
            timeout=2,
            creationflags=windows_hide_flags(),
        )
        if tracked_result.returncode != 0:
            return None
        revision_result = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError):
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
    if (
        _read_agent_revision(_AGENT_SOURCE_DIR, module_path=_AGENT_MODULE_PATH)
        != _AGENT_REVISION
    ):
        # Local patch 2026-09-18: recover automatically instead of dead-ending
        # on the 409.  The guard itself stays fail-closed.  The message tells
        # the truth about which of the two states this is: reloading now, or
        # reloading once a long running turn releases the process.
        _arm_in_place_restart()
        raise AgentRuntimeChangedError(
            _RESTART_MESSAGE_BUSY if _restart_is_blocked() else _RESTART_MESSAGE
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
