"""Tool implementations for the Magi CLI."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import IO
from urllib.parse import urlsplit

import httpx
from pydantic_ai import RunContext

from .safe_shell import RejectedCommand, validate_command
from .sandbox import (
    SandboxUnavailable,
    settings as sandbox_settings,
    wrap as sandbox_wrap,
)

DEFAULT_MAX_READ = 20_000

#: Seconds any single shell tool call may run for.
COMMAND_TIMEOUT = 60

#: Exit code used for a timed-out pipeline (same convention as coreutils).
_TIMEOUT_EXIT = 124

#: Programs whose exit code 1 means "nothing matched", not "it broke".
_NO_MATCH_PROGRAMS = frozenset({"rg", "grep", "egrep", "fgrep"})

# Some models (e.g. Qwen) do not reproduce the marker lines byte-for-byte, so we
# accept any run of three or more marker characters. One or two characters are
# rejected on purpose: lines like "=" or "==" show up in real file contents
# (e.g. setext/rST heading underlines) and would be ambiguous with a divider.
#
# Optional labels and indentation are tolerated as well, so all of these parse
# the same way:
#
#     <<<<<<< SEARCH            <<<<<<<< SEARCH          <<<<<< search
#     =======                   =========                ======
#     >>>>>>> REPLACE           >>>>>>>> Replace         >>>>>>>>>
#
# Anything trailing the label is ignored too, since models sometimes repeat the
# file name (e.g. "<<<<<<< SEARCH src/app.py").
#
_SEARCH_START_RE = re.compile(
    r"^<{3,}[ \t]*(?:(?:SEARCH|ORIGINAL|BEFORE)\b.*)?$", re.IGNORECASE
)
_DIVIDER_RE = re.compile(r"^={3,}[ \t]*$")
_SEARCH_END_RE = re.compile(
    r"^>{3,}[ \t]*(?:(?:REPLACE|UPDATED|AFTER|NEW|END)\b.*)?$", re.IGNORECASE
)

def _format_output(output: str, limit: int = DEFAULT_MAX_READ) -> str:
    """Trim trailing blank lines and truncate very long output."""
    text = output.rstrip("\n")
    if not text.strip():
        return "(no output)"
    if len(text) > limit:
        return f"[truncated to {limit} chars]\n{text[:limit]}"
    return text


def _run_pipeline(stages: Sequence[Sequence[str]], cwd: Path) -> tuple[int, str]:
    """Run validated pipeline stages, piping each stdout into the next stdin.

    Only the last stage's stderr is captured (merged into its stdout) so that
    one command's diagnostics cannot be mistaken for another's data.

    Args:
        stages: One argv list per stage, already validated and sandbox-wrapped.
        cwd: Working directory for every stage.

    Returns:
        ``(exit code, output)`` where a timeout is reported as
        :data:`_TIMEOUT_EXIT`.
    """
    processes: list[subprocess.Popen[str]] = []
    previous: IO[str] | None = None
    try:
        for index, argv in enumerate(stages):
            last = index == len(stages) - 1
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                stdin=previous if previous is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if last else subprocess.DEVNULL,
                text=True,
                errors="replace",
            )
            if previous is not None:
                previous.close()
            previous = process.stdout
            processes.append(process)

        try:
            output, _ = processes[-1].communicate(timeout=COMMAND_TIMEOUT)
        except subprocess.TimeoutExpired:
            for process in processes:
                _ = process.kill()
            for process in processes:
                _ = process.wait()
            return _TIMEOUT_EXIT, ""
        return processes[-1].returncode, output or ""
    finally:
        for process in processes:
            if process.poll() is None:
                _ = process.kill()
                _ = process.wait()
        if previous is not None and not previous.closed:
            previous.close()


async def bash(_: RunContext, command: str) -> str:
    """Run a bash command after getting user approval.

    The command is passed to `/bin/sh -c`, so anything the shell supports is
    available: pipes, redirects, variables, command substitution. When
    `sandbox.mode` in `magi.yaml` allows it, the command runs inside a
    bubblewrap sandbox with the project directory writable and the rest of the
    filesystem read-only.

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        command: The bash command to execute.

    Returns:
        The output of the command, or a message if denied/error.
    """
    cwd = Path.cwd()
    inner = ["/bin/sh", "-c", command]
    try:
        sandboxed = sandbox_wrap(
            inner,
            cwd=cwd,
            writable=(cwd, *sandbox_settings().writable_paths),
        )
    except SandboxUnavailable as exc:
        return f"Error: {exc}"

    try:
        if sandboxed is None:
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                cwd=cwd,
                # Never hand the terminal's stdin to the command. An inherited
                # stdin lets an approved command (cat, head, rg with no path,
                # git commit opening an editor, ...) read the user's next
                # answer to the approval prompt, so the prompt appears to need
                # several `y<enter>` presses. It also makes such a command hang
                # until COMMAND_TIMEOUT instead of failing fast.
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=COMMAND_TIMEOUT,
            )
        else:
            result = await asyncio.to_thread(
                subprocess.run,
                sandboxed,
                cwd=cwd,
                # See above: bwrap does not redirect fd 0 for us.
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=COMMAND_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {COMMAND_TIMEOUT} seconds."
    except Exception as exc:  # pragma: no cover - pass unexpected failure upstream
        return f"Error running command: {exc}"

    if result.returncode != 0:
        return f"Error (exit code {result.returncode}):\n{result.stdout or ''}"
    return _format_output(result.stdout or "")


async def bash_readonly(_: RunContext, command: str) -> str:
    """Run a read-only command and return its output. No approval needed.

    This tool does not need approval because it cannot change anything: there is
    no shell, no redirects, no writes, and no network. Prefer it over `bash` for
    inspecting a project, and reach for `bash` (which asks the user) only when
    something must actually be modified, installed or executed.

    Supported programs include: cat, head, tail, wc, ls, tree, stat, file, du,
    df, nl, cut, tr, sort, uniq, grep, rg, find, diff, jq, the checksums,
    od, hexdump, strings, column, date, uname, which, pwd, echo, printf and
    git log/show/diff/status/blame/grep/ls-files/rev-parse/describe.

    Syntax limits (violating any of them returns a "Refused" message):
      - One pipeline of up to 5 stages joined by `|`; no `;`, `&&`, `||`.
      - No redirects (`>`, `>>`, `<`), no `$(...)` or backticks, no `$VAR`.
      - No unquoted globs or `~`: quote patterns for the program itself, e.g.
        `rg -g '*.py' needle`, and stay inside the project directory.
      - Paths must resolve inside the project, so no `/etc`, no `~/.ssh`.
      - `git` only reads: no checkout/reset/commit/push/clean/stash/apply.

    Runs inside a read-only sandbox when bubblewrap is available.

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        command: The read-only command to execute.

    Returns:
        The output of the command, a "Refused: ..." explanation when the
        command is not provably read-only, or a message on error.
    """
    root = Path.cwd()
    try:
        stages = validate_command(command, root)
    except RejectedCommand as exc:
        return (
            f"Refused: {exc}\n"
            "Nothing was run. Adjust the command, or use bash (which requires "
            "user approval) if you really need it."
        )

    wrapped: list[list[str]] = []
    for argv in stages:
        try:
            sandboxed = sandbox_wrap(argv, cwd=root, network=False, clean_env=True)
        except SandboxUnavailable as exc:
            return f"Error: {exc}"
        wrapped.append(sandboxed if sandboxed is not None else argv)

    try:
        returncode, output = await asyncio.to_thread(_run_pipeline, wrapped, root)
    except Exception as exc:  # pragma: no cover - surface unexpected failures
        return f"Error running command: {exc}"

    if returncode == _TIMEOUT_EXIT:
        return f"Command timed out after {COMMAND_TIMEOUT} seconds."

    program = os.path.basename(stages[-1][0])
    if returncode == 1 and program in _NO_MATCH_PROGRAMS:
        return _format_output(output) if output.strip() else "(no matches)"
    if returncode == 1 and program == "diff":
        # diff exits 1 when the inputs differ, which is a normal answer.
        return _format_output(output)
    if returncode != 0:
        return f"Error (exit code {returncode}):\n{_format_output(output)}"
    return _format_output(output)

async def edit_file(_: RunContext, relative_path: str, diff: str) -> str:
    """Apply a SEARCH/REPLACE style patch to a file within the project directory.

    The diff must be composed of one or more blocks using the following format:

    <<<<<<< SEARCH
    existing text (leave blank to insert without matching)
    =======
    replacement text
    >>>>>>> REPLACE

    The marker lines are parsed leniently: any run of three or more marker
    characters is accepted, the SEARCH/REPLACE labels are optional and
    case-insensitive (anything after the label, such as a file name, is
    ignored), and leading indentation is ignored. So "<<<<<<< SEARCH",
    "<<<<<<<< search", "<<<<<", "=======", "====", ">>>>>>> REPLACE" and
    ">>>>>>>>>" are all valid markers.

    Note:
        A line inside the search section that consists solely of "=" characters
        is treated as the separator, so avoid matching such a line as the first
        line of a search block (e.g. setext/Markdown heading underlines).

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        relative_path: File path relative to the current working directory.
        diff: Patch text built from SEARCH/REPLACE blocks.

    Returns:
        Result of applying the patch, or an error message.

    Notes:
        - If the target file does not exist, it will be created.
        - Empty SEARCH sections insert at the current cursor position (start of file initially).
    """
    base_dir = Path.cwd()
    path = Path(relative_path)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        return "Error: Path must be relative and cannot traverse outside the project."

    target_path = (base_dir / path).resolve()
    try:
        _relative_target = target_path.relative_to(base_dir)
    except ValueError:
        return "Error: Target path must be within the project directory."

    file_exists = target_path.exists()
    if file_exists and not target_path.is_file():
        return "Error: Path is not a file."
    if not file_exists:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return f"Error creating directories for {relative_path}: {exc}"

    def _parse_blocks(spec: str) -> list[tuple[str, str]]:
        normalized = spec.replace("\r\n", "\n")
        lines = normalized.splitlines(keepends=True)
        if not normalized.strip():
            raise ValueError("Diff is empty.")

        def _classify(line: str) -> str | None:
            """Return the kind of marker this line is, or None for content."""
            stripped = line.strip()
            if not stripped:
                return None
            if _SEARCH_START_RE.match(stripped):
                return "start"
            if _DIVIDER_RE.match(stripped):
                return "divider"
            if _SEARCH_END_RE.match(stripped):
                return "end"
            return None

        state = "await"
        search_buf: list[str] = []
        replace_buf: list[str] = []
        blocks: list[tuple[str, str]] = []
        block_start = 0

        for line_no, line in enumerate(lines, start=1):
            marker = _classify(line)
            if state == "await":
                if marker == "start":
                    state = "search"
                    search_buf = []
                    block_start = line_no
                elif marker in {"divider", "end"}:
                    raise ValueError(
                        f"Line {line_no}: unexpected {marker} marker before any block."
                    )
                elif line.strip():
                    raise ValueError(
                        f"Line {line_no} has content before the first patch block: {line.strip()!r}"
                    )
                # Blank lines between blocks are ignored.
            elif state == "search":
                if marker == "divider":
                    state = "replace"
                    replace_buf = []
                elif marker == "end":
                    raise ValueError(
                        f"Block at line {block_start}: missing '=======' separator (line {line_no})."
                    )
                else:
                    search_buf.append(line)
            else:  # state == "replace"
                if marker == "end":
                    blocks.append(("".join(search_buf), "".join(replace_buf)))
                    state = "await"
                elif marker == "start":
                    raise ValueError(
                        f"Block at line {block_start}: missing closing marker (line {line_no})."
                    )
                else:
                    # Extra '=======' lines are treated as literal content so
                    # that replacements touching heading underlines still work.
                    replace_buf.append(line)

        if state != "await":
            raise ValueError(
                f"Block at line {block_start} is incomplete; missing a marker."
            )
        if not blocks:
            raise ValueError("Diff did not contain any SEARCH/REPLACE blocks.")

        return blocks

    try:
        replacements = _parse_blocks(diff)
    except ValueError as exc:
        return f"Error parsing diff: {exc}"

    def _read() -> str:
        with target_path.open("r", encoding="utf-8") as f:
            return f.read()

    if file_exists:
        try:
            original_text = await asyncio.to_thread(_read)
        except Exception as exc:  # pragma: no cover - surface unexpected read issues
            return f"Error reading file: {exc}"
    else:
        original_text = ""

    updated_text = original_text
    cursor = 0
    for idx, (search_text, replace_text) in enumerate(replacements, start=1):
        if search_text:
            match_index = updated_text.find(search_text, cursor)
            if match_index == -1:
                return f"Error: Could not find SEARCH block {idx} in {relative_path}."
        else:
            match_index = cursor
        updated_text = (
            updated_text[:match_index]
            + replace_text
            + updated_text[match_index + len(search_text):]
        )
        cursor = match_index + len(replace_text)

    def _write(content: str) -> None:
        with target_path.open("w", encoding="utf-8") as f:
            _ = f.write(content)

    try:
        await asyncio.to_thread(_write, updated_text)
    except Exception as exc:  # pragma: no cover - surface unexpected write issues
        return f"Error writing file: {exc}"

    return f"Applied {len(replacements)} replacement(s) successfully."


async def read_file(_: RunContext, relative_path: str, max_chars: int | None = None) -> str:
    """Read a text file within the project directory (with optional character limit).

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        relative_path: File path relative to the current working directory.
        max_chars: Maximum number of characters to return (defaults to 20k).

    Returns:
        File content (possibly truncated) or an error message.
    """
    base_dir = Path.cwd()
    path = Path(relative_path)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        return "Error: Path must be relative and cannot traverse outside the project."

    target_path = (base_dir / path).resolve()
    try:
        _relative_target = target_path.relative_to(base_dir)
    except ValueError:
        return "Error: Target path must be within the project directory."

    if not target_path.exists():
        return "Error: File does not exist."
    if not target_path.is_file():
        return "Error: Path is not a file."

    limit = DEFAULT_MAX_READ if max_chars is None else max_chars
    if limit <= 0:
        return "Error: max_chars must be positive."

    def _read_text() -> str:
        with target_path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(limit + 1)

    try:
        content = await asyncio.to_thread(_read_text)
    except Exception as exc:
        return f"Error reading file: {exc}"

    if len(content) > limit:
        return f"[truncated to {limit} chars]\n{content[:limit]}"
    return content





async def deepseek_search(_: RunContext, query: str, max_results: int = 10) -> str:
    """Search the web using the DeepSeek Search API.

    This tool calls the DeepSeek `/v1/tools/search` endpoint and returns
    the search results (URLs, titles, snippets, and optional full content).

    Requires the `DEEPSEEK_API_KEY` environment variable to be set.

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        query: The search query string.
        max_results: Maximum number of search results to return (default 10).

    Returns:
        A formatted string containing a numbered list of search results
        (title, URL, snippet, and optional content), or an error message
        if the request fails.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY environment variable is not set."

    payload: dict[str, object] = {
        "query": query,
        "max_results": max_results,
        "include_content": True,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/tools/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        return f"Error calling DeepSeek Search API: {exc}"

    try:
        data = response.json()
    except ValueError:
        return "Error: Could not parse DeepSeek Search API response as JSON."

    if response.status_code != 200:
        # Error format: { "error": { "code": "...", "message": "..." } }
        error_info = data.get("error", {})
        code = error_info.get("code", "unknown")
        message = error_info.get("message", "")
        return f"Error: DeepSeek Search API returned status {response.status_code} ({code}): {message}"

    results = data.get("results", [])
    if not results:
        return "(no search results returned)"

    lines: list[str] = []
    for i, result in enumerate(results[:max_results], 1):
        title = result.get("title", "(no title)")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        entry = f"{i}. {title}\n   URL: {url}\n   {snippet}"
        if result.get("content"):
            entry += f"\n   Content: {result['content']}"
        lines.append(entry)

    return "\n".join(lines)


async def deepseek_fetch(
    _: RunContext, url: str, max_chars: int | None = None
) -> str:
    """Fetch a single web page via the DeepSeek Fetch API.

    This tool calls the DeepSeek `/v1/tools/fetch` endpoint and returns the
    page content converted to Markdown.

    Requires the `DEEPSEEK_API_KEY` environment variable to be set.

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        url: The page URL to fetch. Must be an http(s) URL with a host and
            must not contain a username or password.
        max_chars: Maximum number of characters of content to return
            (defaults to 20k). Pass a negative value for no limit.

    Returns:
        The page title followed by its Markdown content, or an error message
        if the request fails.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY environment variable is not set."

    if not isinstance(url, str) or not url.strip():
        return "Error: url must be a non-empty string."
    url = url.strip()
    if len(url) > 8192:
        return "Error: url must be at most 8192 characters."

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return "Error: url must use the http or https scheme."
    if not parsed.hostname:
        return "Error: url must include a host."
    if parsed.username is not None or parsed.password is not None:
        return "Error: url must not contain a username or password."

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/tools/fetch",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": url},
            )
    except httpx.HTTPError as exc:
        return f"Error calling DeepSeek Fetch API: {exc}"

    try:
        data = response.json()
    except ValueError:
        return "Error: Could not parse DeepSeek Fetch API response as JSON."

    if response.status_code != 200:
        # Error format: { "error": { "code": "...", "message": "..." } }
        error_info = data.get("error", {}) if isinstance(data, dict) else {}
        code = error_info.get("code", "unknown")
        message = error_info.get("message", "")
        return (
            f"Error: DeepSeek Fetch API returned status "
            f"{response.status_code} ({code}): {message}"
        )

    if not isinstance(data, dict):
        return "Error: Unexpected DeepSeek Fetch API response shape."

    title = data.get("title") or "(no title)"
    content = data.get("content") or ""
    if not content:
        return f"{title}\n\n(no content could be extracted from the page)"

    limit = DEFAULT_MAX_READ if max_chars is None else max_chars
    if limit <= 0:
        return f"{title}\n\n{content}"

    if len(content) > limit:
        return (
            f"Title: {title}\n\n"
            f"[truncated to {limit} chars]\n{content[:limit]}"
        )
    return f"Title: {title}\n\n{content}"
