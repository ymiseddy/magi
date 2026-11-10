"""Validation of read-only shell commands, without ever invoking a shell.

The ``bash`` tool hands its input to ``sh -c``, so any check on that string is
only a guess about how a *second* parser will read it. This module refuses to
guess: the command is tokenised here, validated against an allowlist of
read-only programs, and then executed as an argv list with ``shell=False``.
What runs is exactly what was checked, and no shell ever sees the original
string.

The policy is deny-by-default. Anything the tokeniser or a per-command checker
does not explicitly accept raises :class:`RejectedCommand`, including:

* quoting and escaping tricks, expansions, subshells and redirections --
  ``shlex.split("echo hi; rm -rf /")`` yields ``["echo", "hi;", ...]``, so a
  naive argv[0] check approves a destructive command; ``"c\\at f"`` becomes
  ``["cat", "f"]`` and spoofs the program name outright,
* pipelines longer than :data:`MAX_STAGES`,
* programs outside the allowlist (``rm``, ``mv``, ``sed -i``, ``xargs``,
  ``curl``, ``python``, ``bash``, ...),
* flags that turn a read-only program into a writer or an exec wrapper
  (``sort -o``, ``uniq in out``, ``find -delete``, ``rg --pre``, ``git -c``,
  ``tail -f``, ``file -C``, ``date -s``, ...),
* paths that resolve outside the project directory, so ``cat`` cannot read
  ``~/.ssh/id_rsa`` and ``find`` cannot walk ``/etc``.

Everything here is pure: no I/O beyond path resolution, so it is cheap to
unit test against a corpus of nasty commands.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Longest command string accepted, in characters.
MAX_COMMAND_LENGTH = 4096

#: Most ``|`` stages accepted in one call.
MAX_STAGES = 5

#: Directories searched for programs. Resolving against a fixed list keeps a
#: planted ``cat`` in the project or an inherited PATH from being executed.
SAFE_BIN_DIRS: tuple[str, ...] = (
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    "/opt/homebrew/bin",
    "/opt/local/bin",
    "/run/current-system/sw/bin",
)

#: Token produced for an unquoted ``|``; every other metacharacter is refused
#: by the tokeniser instead of being given a meaning.
PIPE = "|"

_FORBIDDEN: dict[str, str] = {
    ";": "command chaining with ';' is not supported",
    "&": "'&' (background jobs) is not supported",
    "<": "input redirection is not supported",
    ">": "output redirection is not supported",
    "(": "subshells and grouping are not supported",
    ")": "subshells and grouping are not supported",
    "{": "brace expansion is not supported",
    "}": "brace expansion is not supported",
    "[": "globbing is not supported; quote the pattern so the program globs it",
    "]": "globbing is not supported; quote the pattern so the program globs it",
    "*": "globbing is not supported; quote the pattern so the program globs it",
    "?": "globbing is not supported; quote the pattern so the program globs it",
    "#": "comments are not supported",
    "!": "history expansion is not supported",
    "$": "variable and command substitution are not supported",
    "`": "command substitution is not supported",
    "\\": "backslash escapes are not supported; use quotes",
}


_TILDE_MESSAGE = "home directory expansion is not supported; use a project-relative path"


class RejectedCommand(ValueError):
    """Raised when a command is not provably read-only.

    The message is written for the model, not the user: it explains what to do
    instead, so the next attempt succeeds.
    """


@dataclass(frozen=True)
class _Command:
    """Per-program rules.

    ``denied_flags`` is matched against both the whole flag (``--output``) and
    its clustered letters (``-o`` in ``-ov``), so attached values cannot slip
    past. ``allowed_flags``, when set, is a whitelist matched against whole
    flags only; anything else is refused.
    """

    denied_flags: frozenset[str] = frozenset()
    allowed_flags: frozenset[str] | None = None
    min_operands: int = 0
    max_operands: int | None = None


_FIND_ALLOWED_FLAGS: frozenset[str] = frozenset(
    {
        # Options that do not take a path or change output framing.
        "-H",
        "-L",
        "-P",
        "-E",
        "-d",
        "-s",
        "-depth",
        "-maxdepth",
        "-mindepth",
        "-noleaf",
        "-follow",
        "-ignore_readdir_race",
        "-nowarn",
        "-warn",
        "-xdev",
        "-mount",
        # Tests.
        "-name",
        "-iname",
        "-path",
        "-ipath",
        "-wholename",
        "-iwholename",
        "-regex",
        "-iregex",
        "-lname",
        "-ilname",
        "-type",
        "-xtype",
        "-size",
        "-mtime",
        "-mmin",
        "-atime",
        "-amin",
        "-ctime",
        "-cmin",
        "-newer",
        "-anewer",
        "-cnewer",
        "-newermt",
        "-used",
        "-empty",
        "-readable",
        "-writable",
        "-executable",
        "-perm",
        "-user",
        "-group",
        "-nouser",
        "-nogroup",
        "-links",
        "-inum",
        "-samefile",
        "-true",
        "-false",
        "-not",
        "-a",
        "-and",
        "-o",
        "-or",
        # Actions that only write to stdout.
        "-print",
        "-print0",
        "-printf",
        "-ls",
        "-prune",
        "-quit",
        "--help",
        "--version",
    }
)

_GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "annotate",
        "blame",
        "cat-file",
        "count-objects",
        "describe",
        "diff",
        "for-each-ref",
        "grep",
        "log",
        "ls-files",
        "ls-tree",
        "merge-base",
        "name-rev",
        "rev-list",
        "rev-parse",
        "shortlog",
        "show",
        "show-ref",
        "status",
        "var",
        "whatchanged",
    }
)

_GIT_DENIED_FLAGS: frozenset[str] = frozenset(
    {
        # Could point git at another repository or another config.
        "-c",
        "--config",
        "--config-env",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--work-tree",
        # Could make git run an arbitrary program.
        "--ext-diff",
        "--textconv",
        "--upload-pack",
        "--receive-pack",
        "-O",
        "--open-files-in-pager",
        # Could write a file.
        "-o",
        "--output",
        "--paginate",
    }
)


_READERS: dict[str, _Command] = {
    "cat": _Command(min_operands=1),
    "tac": _Command(min_operands=1),
    "head": _Command(denied_flags=frozenset({"-f", "--follow", "--pid"}), min_operands=1),
    "tail": _Command(
        denied_flags=frozenset({"-f", "-F", "--follow", "--retry", "--pid"}),
        min_operands=1,
    ),
    "wc": _Command(min_operands=1),
    "ls": _Command(),
    "stat": _Command(min_operands=1),
    "file": _Command(denied_flags=frozenset({"-C", "--compile"}), min_operands=1),
    "du": _Command(),
    "df": _Command(),
    "pwd": _Command(),
    "echo": _Command(),
    "printf": _Command(),
    "basename": _Command(min_operands=1),
    "dirname": _Command(min_operands=1),
    "realpath": _Command(min_operands=1),
    "readlink": _Command(min_operands=1),
    "nl": _Command(),
    "cut": _Command(),
    "tr": _Command(min_operands=1),
    "od": _Command(),
    "hexdump": _Command(),
    "strings": _Command(min_operands=1),
    "column": _Command(),
    "fold": _Command(),
    "expand": _Command(),
    "unexpand": _Command(),
    "rev": _Command(),
    "paste": _Command(min_operands=1),
    "join": _Command(min_operands=2),
    "comm": _Command(min_operands=2),
    "diff": _Command(min_operands=1),
    "md5sum": _Command(min_operands=1),
    "sha1sum": _Command(min_operands=1),
    "sha256sum": _Command(min_operands=1),
    "jq": _Command(min_operands=1),
    "date": _Command(denied_flags=frozenset({"-s", "--set"})),
    "uname": _Command(),
    "hostname": _Command(),
    "id": _Command(),
    "whoami": _Command(),
    "printenv": _Command(),
    "which": _Command(min_operands=1),
    "numfmt": _Command(),
    "tree": _Command(denied_flags=frozenset({"-o", "--output"})),
    # sort -o / --output would write; --compress-program would exec.
    "sort": _Command(
        denied_flags=frozenset({"-o", "--output", "--compress-program"})
    ),
    # uniq FILE OUT writes OUT; only the single-operand form is read-only.
    "uniq": _Command(max_operands=1),
    "grep": _Command(),
    "egrep": _Command(),
    "fgrep": _Command(),
    # rg --pre / -z run an external program to produce input.
    "rg": _Command(denied_flags=frozenset({"--pre", "--pre-glob", "--hostname-bin", "-z", "--search-zip"})),
    "find": _Command(allowed_flags=_FIND_ALLOWED_FLAGS, min_operands=1),
}

#: Every program :func:`validate_command` accepts, sorted for display.
READ_ONLY_COMMANDS: tuple[str, ...] = tuple(sorted((*_READERS, "git")))


def _flag_names(token: str) -> list[str]:
    """Return the forms a flag token can be matched under.

    ``--output=x`` yields ``["--output"]``; ``-ov`` yields
    ``["-ov", "-o", "-v"]`` so a denied short flag is found inside a cluster.
    """
    if token.startswith("--"):
        return [token.split("=", 1)[0]]
    if token.startswith("-") and token != "-":
        return [token, *(f"-{char}" for char in token[1:])]
    return []


def _whole_flag(token: str) -> str:
    """Return the flag without any attached ``=value``."""
    if token.startswith("--"):
        return token.split("=", 1)[0]
    return token


def tokenize(command: str) -> list[str]:
    """Split a command into tokens, refusing everything a shell could interpret.

    Quotes are honoured only to group words: everything inside them is taken
    literally, so ``rg -g '*.py' foo`` passes a glob to ripgrep without this
    module ever treating it as one. Expansions are refused even inside double
    quotes, where a shell would still expand them.

    Args:
        command: The raw command string.

    Returns:
        Tokens, with unquoted ``|`` preserved as :data:`PIPE`.

    Raises:
        RejectedCommand: If the command uses anything outside the accepted
            subset.
    """
    tokens: list[str] = []
    buffer: list[str] = []
    started = False
    index = 0
    length = len(command)

    def flush() -> None:
        nonlocal started
        if started:
            tokens.append("".join(buffer))
            buffer.clear()
            started = False

    while index < length:
        char = command[index]

        if char == "'":
            end = command.find("'", index + 1)
            if end == -1:
                raise RejectedCommand("unbalanced single quote")
            buffer.append(command[index + 1 : end])
            started = True
            index = end + 1
            continue

        if char == '"':
            end = command.find('"', index + 1)
            if end == -1:
                raise RejectedCommand("unbalanced double quote")
            inner = command[index + 1 : end]
            for bad, reason in (
                ("$", "variable expansion inside double quotes"),
                ("`", "command substitution inside double quotes"),
                ("\\", "backslash escapes inside double quotes"),
            ):
                if bad in inner:
                    raise RejectedCommand(f"{reason} is not supported")
            buffer.append(inner)
            started = True
            index = end + 1
            continue

        if char in " \t":
            flush()
            index += 1
            continue

        if char == PIPE:
            flush()
            tokens.append(PIPE)
            index += 1
            continue

        if char == "~":
            # A shell only expands '~' at the start of a word (or after '=' or
            # ':' in an assignment), so `HEAD~1` and `foo~bar` stay literal.
            if not started or "".join(buffer[-1:]) in {"=", ":"}:
                raise RejectedCommand(_TILDE_MESSAGE)
            buffer.append(char)
            index += 1
            continue

        if char in _FORBIDDEN:
            raise RejectedCommand(_FORBIDDEN[char])

        if char in "\n\r":
            raise RejectedCommand("one command per call please; newlines are not supported")

        if ord(char) < 0x20 or ord(char) == 0x7F:
            raise RejectedCommand("control characters are not supported")

        buffer.append(char)
        started = True
        index += 1

    flush()
    return tokens


def _split_stages(tokens: Sequence[str]) -> list[list[str]]:
    """Split tokens on :data:`PIPE` and reject empty or oversized pipelines."""
    stages: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token == PIPE:
            if not current:
                raise RejectedCommand("empty command in pipeline")
            stages.append(current)
            current = []
        else:
            current.append(token)
    if not current:
        raise RejectedCommand("empty command in pipeline")
    stages.append(current)

    if len(stages) > MAX_STAGES:
        raise RejectedCommand(
            f"pipelines are limited to {MAX_STAGES} stages; split the command up"
        )
    return stages


def _split_flags(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split argv[1:] into flag tokens and operands, honouring ``--``."""
    flags: list[str] = []
    operands: list[str] = []
    only_operands = False
    for token in argv[1:]:
        if only_operands:
            operands.append(token)
        elif token == "--":
            only_operands = True
        elif token.startswith("-") and token != "-":
            flags.append(token)
        else:
            operands.append(token)
    return flags, operands


def _check_flag_denylist(name: str, spec: _Command, flags: Sequence[str]) -> None:
    """Refuse flags that turn a read-only program into a writer or exec wrapper."""
    for flag in flags:
        for form in _flag_names(flag):
            if form in spec.denied_flags:
                raise RejectedCommand(
                    f"'{name} {form}' is not read-only; drop it or use bash (with approval)"
                )
        if spec.allowed_flags is not None and _whole_flag(flag) not in spec.allowed_flags:
            raise RejectedCommand(
                f"'{name} {_whole_flag(flag)}' is not on the read-only allowlist for {name}"
            )


def _check_operand_count(
    name: str,
    spec: _Command,
    operands: Sequence[str],
    stdin_available: bool,
) -> None:
    """Enforce per-program operand counts (e.g. ``uniq FILE OUT`` writes)."""
    if len(operands) < spec.min_operands and not stdin_available:
        raise RejectedCommand(
            f"'{name}' needs at least {spec.min_operands} file argument(s) when not piped"
        )
    if spec.max_operands is not None and len(operands) > spec.max_operands:
        raise RejectedCommand(
            f"'{name}' takes at most {spec.max_operands} argument(s); more would write a file"
        )


def _path_candidates(token: str) -> list[str]:
    """Return the parts of a token that could name a path."""
    candidates = [token]
    if "=" in token:
        candidates.append(token.split("=", 1)[1])
    return candidates


def _check_contained(token: str, root: Path) -> None:
    """Refuse tokens that name a path outside ``root``.

    Only tokens that look like paths are resolved, so ordinary words and
    patterns pass through untouched. Symlinks are resolved first, so a link
    inside the project cannot point out of it.
    """
    for candidate in _path_candidates(token):
        if not candidate or candidate == "-":
            continue
        if candidate.startswith("~"):
            raise RejectedCommand("home directory paths are not supported; stay inside the project")
        looks_like_path = os.path.isabs(candidate) or "/" in candidate or candidate in {".", ".."}
        if not looks_like_path:
            continue
        try:
            resolved = Path(candidate).resolve() if os.path.isabs(candidate) else (root / candidate).resolve()
        except OSError:
            raise RejectedCommand(f"could not resolve path {candidate!r}") from None
        if not resolved.is_relative_to(root):
            raise RejectedCommand(
                f"path {candidate!r} resolves outside the project directory ({root})"
            )


def _check_git(argv: Sequence[str]) -> None:
    """Validate a ``git`` invocation: read-only subcommand, no config escapes."""
    tokens = list(argv[1:])
    subcommand: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            continue
        if token == "-C":
            if index + 1 >= len(tokens):
                raise RejectedCommand("'git -C' needs a directory")
            index += 2
            continue
        forms = _flag_names(token)
        if forms:
            for form in forms:
                if form in _GIT_DENIED_FLAGS:
                    raise RejectedCommand(
                        f"'git {form}' is not read-only; drop it or use bash (with approval)"
                    )
            index += 1
            continue
        if subcommand is None:
            subcommand = token
            if subcommand not in _GIT_READONLY_SUBCOMMANDS:
                allowed = ", ".join(sorted(_GIT_READONLY_SUBCOMMANDS))
                raise RejectedCommand(f"'git {subcommand}' writes; read-only: {allowed}")
        index += 1

    if subcommand is None:
        raise RejectedCommand("'git' needs a subcommand")


def _check_stage(stage: Sequence[str], root: Path, stdin_available: bool = False) -> str:
    """Validate one pipeline stage and return the resolved program path.

    Args:
        stage: The argv to check.
        root: Directory that paths must stay inside.
        stdin_available: True for stages after the first, where the previous
            stage supplies input and file operands become optional.
    """
    if not stage:
        raise RejectedCommand("empty command")

    name = stage[0]
    if "=" in name:
        raise RejectedCommand("environment prefixes (VAR=value cmd) are not supported")

    if name == "git":
        _check_git(stage)
    else:
        spec = _READERS.get(name)
        if spec is None:
            allowed = ", ".join(READ_ONLY_COMMANDS)
            raise RejectedCommand(f"'{name}' is not on the read-only allowlist; allowed: {allowed}")
        flags, operands = _split_flags(stage)
        _check_flag_denylist(name, spec, flags)
        _check_operand_count(name, spec, operands, stdin_available)

    for token in stage[1:]:
        _check_contained(token, root)

    return _resolve_binary(name)


def _resolve_binary(name: str) -> str:
    """Resolve a program name to a path inside :data:`SAFE_BIN_DIRS`."""
    if "/" in name or "\\" in name:
        raise RejectedCommand("program names may not contain a path separator")
    for directory in SAFE_BIN_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RejectedCommand(f"'{name}' is not installed in {', '.join(SAFE_BIN_DIRS)}")


def validate_command(command: str, root: Path) -> list[list[str]]:
    """Validate a command and split it into executable pipeline stages.

    Args:
        command: The raw command string from the model.
        root: Directory that all paths must stay inside. Normally the project
            root, which is also the working directory for execution.

    Returns:
        One argv list per pipeline stage, with program names resolved to
        absolute paths. Run them with ``shell=False``; the stages are meant to
        be chained with pipes.

    Raises:
        RejectedCommand: If any part of the command is not provably read-only.
    """
    if not command or not command.strip():
        raise RejectedCommand("empty command")

    if len(command) > MAX_COMMAND_LENGTH:
        raise RejectedCommand(
            f"command is {len(command)} characters; the limit is {MAX_COMMAND_LENGTH}"
        )

    try:
        root = root.resolve()
    except OSError:
        raise RejectedCommand(f"could not resolve project directory {root}") from None

    tokens = tokenize(command)
    stages = _split_stages(tokens)

    resolved: list[list[str]] = []
    for index, stage in enumerate(stages):
        program = _check_stage(stage, root, stdin_available=index > 0)
        resolved.append([program, *stage[1:]])
    return resolved
