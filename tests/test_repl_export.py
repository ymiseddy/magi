# pyright: reportPrivateUsage=false
"""Tests for the `/export` slash command and the stored last result.

These exercise the command in isolation: no agent or model is involved, so the
tests drive the slash-command dispatcher with a small recording IO and a no-op
session manager.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from typing_extensions import override

from pydantic_ai import Agent, ModelMessage

from magi.io import OTYPE_ERROR, OTYPE_PROMPT, ReaderWriter
from magi.repl import MagiRepl
from magi.session import NoOpSessionManager


class RecordingIO:
    """Minimal ReaderWriter that records everything written."""

    def __init__(self) -> None:
        self.outputs: list[tuple[str, str]] = []

    def read(self) -> str | None:
        return None

    def readapproval(self) -> bool:
        return True

    def write(self, otype: str, data: str) -> None:
        self.outputs.append((otype, data))

    def writeln(self, otype: str, data: str) -> None:
        self.outputs.append((otype, data + "\n"))

    def endresult(self) -> None:
        pass

    def output_types(self) -> list[str]:
        return [otype for otype, _ in self.outputs]


def _make_repl(io: RecordingIO) -> MagiRepl:
    return MagiRepl(
        agent=cast(Agent, cast(object, None)),
        io=cast(ReaderWriter, cast(object, io)),
        session_manager=NoOpSessionManager(),
    )


class ExportCommandTests(unittest.TestCase):
    _tmp: tempfile.TemporaryDirectory[str]
    root: Path
    _previous_cwd: str
    io: RecordingIO
    repl: MagiRepl

    @override
    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self._previous_cwd = os.getcwd()
        self.io = RecordingIO()
        self.repl = _make_repl(self.io)

    @override
    def setUp(self) -> None:
        # /export resolves relative paths against the process working
        # directory, so run the command from inside the temp dir.
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._previous_cwd)

    def _run(self, command: str) -> None:
        handled, _history = self.repl._try_slash_command(command, [])
        self.assertTrue(handled)

    def test_registered_and_listed_in_help(self) -> None:
        self._run("/help")
        help_text = "".join(text for _, text in self.io.outputs)
        self.assertIn("/export", help_text)

    def test_export_without_result_reports_and_writes_nothing(self) -> None:
        self._run("/export")
        self.assertIn(OTYPE_PROMPT, self.io.output_types())
        self.assertFalse((self.root / "response.md").exists())

    def test_export_writes_last_result_to_default_file(self) -> None:
        self.repl._last_result = "# Hello\n\nWorld."
        self._run("/export")
        target = self.root / "response.md"
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "# Hello\n\nWorld.\n")

    def test_export_accepts_explicit_path_and_adds_md(self) -> None:
        self.repl._last_result = "body"
        self._run("/export notes")
        target = self.root / "notes.md"
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "body\n")

    def test_export_keeps_existing_extension(self) -> None:
        self.repl._last_result = "body"
        self._run("/export out.markdown")
        self.assertTrue((self.root / "out.markdown").exists())
        self.assertFalse((self.root / "out.markdown.md").exists())

    def test_export_reports_write_errors(self) -> None:
        self.repl._last_result = "body"
        self._run("/export missing_dir/file.md")
        self.assertIn(OTYPE_ERROR, self.io.output_types())

    def test_export_does_not_touch_session_history(self) -> None:
        history: list[ModelMessage] = []
        self.repl._last_result = "body"
        handled, result_history = self.repl._try_slash_command("/export", history)
        self.assertTrue(handled)
        self.assertIs(result_history, history)


if __name__ == "__main__":
    _ = unittest.main()
