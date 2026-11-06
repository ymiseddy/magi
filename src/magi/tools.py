"""Tool implementations for the Magi CLI."""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from pydantic_ai import RunContext

DEFAULT_MAX_READ = 20_000

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

async def bash(_: RunContext, command: str) -> str:
    """Run a bash command after getting user approval.

    Args:
        ctx: Run context provided by pydantic-ai (unused, but required).
        command: The bash command to execute.

    Returns:
        The output of the command, or a message if denied/error.
    """

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as exc:  # pragma: no cover - pass unexpected failure upstream
        return f"Error running command: {exc}"

    if result.returncode != 0:
        return f"Error (exit code {result.returncode}): {result.stderr}"
    return result.stdout or "(no output)"

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

