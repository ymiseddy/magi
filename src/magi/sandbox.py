"""Bubblewrap sandbox used by the shell tools.

Two consumers:

* :func:`magi.tools.bash_readonly` always runs inside the sandbox when
  bubblewrap is available. Its allowlist in :mod:`magi.safe_shell` is the
  primary gate; the read-only mount namespace is the backstop, so a mistake in
  the allowlist still cannot modify the filesystem.
* :func:`magi.tools.bash` can opt in through the ``sandbox`` section of
  ``magi.yaml``. When it does, the project directory is bound read-write and
  everything else is read-only.

The sandbox is best-effort by default (``mode: auto``): if ``bwrap`` is
missing or cannot create user namespaces, callers fall back to running the
command unsandboxed. ``mode: require`` turns that fallback into an error.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MODE_AUTO = "auto"
MODE_OFF = "off"
MODE_REQUIRE = "require"
MODES: tuple[str, ...] = (MODE_AUTO, MODE_OFF, MODE_REQUIRE)

#: Fixed PATH for sandboxed read-only commands. Resolving binaries against a
#: known directory list stops a model from shadowing `cat` in the project.
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"

#: Writable tmpfs used as HOME when the real home directory is not bound.
TMP_HOME = "/tmp"


class SandboxUnavailable(RuntimeError):
    """Raised when the sandbox is required but cannot be used."""


@dataclass(frozen=True)
class SandboxSettings:
    """The ``sandbox`` section of ``magi.yaml``."""

    mode: str = MODE_AUTO
    network: bool = True
    writable_paths: tuple[Path, ...] = ()

    def describe(self) -> str:
        state = "available" if available() else "unavailable"
        return (
            f"mode={self.mode}, network={self.network}, "
            f"extra_writable_paths={len(self.writable_paths)}, bwrap={state}"
        )


_settings = SandboxSettings()


def configure(settings: SandboxSettings) -> None:
    """Install the settings used by :func:`wrap`."""
    global _settings
    _settings = settings


def settings() -> SandboxSettings:
    """Return the currently installed settings."""
    return _settings


@lru_cache(maxsize=1)
def bwrap_path() -> str | None:
    """Return the path to a working ``bwrap`` binary, or None.

    The probe is cached: a binary that cannot create a namespace on this host
    will not start working mid-session, and probing on every call would add a
    subprocess to each tool invocation.
    """
    binary = shutil.which("bwrap")
    if binary is None:
        return None

    probe: list[str] = [
        binary,
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--",
        "/bin/true",
    ]
    try:
        completed = subprocess.run(
            probe,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return binary if completed.returncode == 0 else None


def available() -> bool:
    """Return True when shell commands can be sandboxed on this host."""
    return bwrap_path() is not None


def clear_probe_cache() -> None:
    """Forget a previous probe result (used by tests)."""
    bwrap_path.cache_clear()


def _resolved_existing(paths: Sequence[Path]) -> list[Path]:
    """Resolve and de-duplicate bind targets, dropping anything missing."""
    found: list[Path] = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in found:
            found.append(resolved)
    return found


def _home_for(writable: Sequence[Path]) -> str:
    """Pick the HOME to expose inside the sandbox.

    The real home directory is only used when the caller explicitly bound it
    read-write; otherwise HOME points at the writable tmpfs so tools that
    insist on writing dotfiles have somewhere to go.
    """
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        return TMP_HOME
    for path in writable:
        if path == home or path in home.parents:
            return str(home)
    return TMP_HOME


def wrap(
    argv: Sequence[str],
    *,
    cwd: Path,
    writable: Sequence[Path] = (),
    network: bool | None = None,
    clean_env: bool = False,
) -> list[str] | None:
    """Wrap ``argv`` in a bubblewrap sandbox.

    Args:
        argv: The program and arguments to run. It is executed directly, never
            through a shell.
        cwd: Directory to enter inside the sandbox.
        writable: Extra paths to bind read-write. Everything else is read-only.
        network: Override the configured network access. ``None`` uses the
            configured value.
        clean_env: Start from an empty environment with a fixed ``PATH``.
            Read-only commands use this so no API keys or tokens are visible
            to the child process.

    Returns:
        The wrapped argv, or ``None`` when the command should run unsandboxed.

    Raises:
        SandboxUnavailable: If ``sandbox.mode`` is ``require`` and bubblewrap
            is missing or unusable.
    """
    if _settings.mode == MODE_OFF:
        return None

    binary = bwrap_path()
    if binary is None:
        if _settings.mode == MODE_REQUIRE:
            raise SandboxUnavailable(
                "sandbox.mode is 'require' but bubblewrap is missing or unusable"
            )
        return None

    try:
        workdir = cwd.resolve()
    except OSError:
        return None
    if not workdir.is_dir():
        return None

    bind_rw = _resolved_existing(writable)
    allow_network = _settings.network if network is None else network

    args: list[str] = [binary, "--unshare-all", "--die-with-parent", "--new-session"]
    if allow_network:
        # --unshare-all includes --unshare-net; put the network back.
        args.append("--share-net")
    if clean_env:
        args += ["--clearenv", "--setenv", "PATH", DEFAULT_PATH, "--setenv", "LANG", "C.UTF-8"]
    args += [
        "--setenv",
        "HOME",
        _home_for(bind_rw),
        # Keep git from refreshing the index lock in a read-only tree, and stop
        # anything from shelling out to an interactive pager.
        "--setenv",
        "GIT_OPTIONAL_LOCKS",
        "0",
        "--setenv",
        "GIT_PAGER",
        "cat",
        "--setenv",
        "PAGER",
        "cat",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]
    # Binds come after the /tmp tmpfs so a writable path under /tmp wins.
    for path in bind_rw:
        args += ["--bind", str(path), str(path)]
    if not any(workdir == path or path in workdir.parents for path in bind_rw):
        # The /tmp tmpfs would otherwise shadow a working directory that lives
        # under /tmp, so put it back read-only.
        args += ["--ro-bind", str(workdir), str(workdir)]
    args += ["--chdir", str(workdir), "--", *argv]
    return args
