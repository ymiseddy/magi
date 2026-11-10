"""Tests for the console reader and approver, especially EOF handling.

stdin can close at any point: Ctrl-D at a prompt, a terminal that went away, or
a pipe whose writer exited. Neither reader may let ``EOFError`` escape, because
it would surface as a traceback from inside the agent loop instead of a clean
exit -- and an approval that can never be answered must be denied, not raised.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, TypeVar
from unittest.mock import patch

from magi.io import ConsoleApprover, ConsoleInputter

T = TypeVar("T")


def _call_with_inputs(func: Callable[[], T], *inputs: str | BaseException) -> tuple[T, str, str]:
    """Run ``func`` with ``builtins.input`` fed by ``inputs``.

    Entries that are exceptions are raised by the fake ``input`` instead of
    being returned, which is how a real EOF reaches the caller.

    Returns ``(result, stdout, stderr)``.
    """
    pending = list(inputs)

    def fake_input(*_args: object, **_kwargs: object) -> str:
        value = pending.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    out, err = io.StringIO(), io.StringIO()
    with patch("builtins.input", fake_input), redirect_stdout(out), redirect_stderr(err):
        result = func()
    return result, out.getvalue(), err.getvalue()


class ConsoleApproverTests(unittest.TestCase):
    def _readapproval(self, *inputs: str | BaseException) -> tuple[bool, str, str]:
        return _call_with_inputs(ConsoleApprover().readapproval, *inputs)

    def test_accepts_yes_spellings(self) -> None:
        for answer in ("y", "Y", "yes", "YES", " y "):
            approved, _out, _err = self._readapproval(answer)
            self.assertTrue(approved, answer)

    def test_rejects_no_spellings(self) -> None:
        for answer in ("n", "N", "no", "NO", " n "):
            approved, _out, _err = self._readapproval(answer)
            self.assertFalse(approved, answer)

    def test_invalid_answer_reprompts(self) -> None:
        approved, out, _err = self._readapproval("maybe", "y")
        self.assertTrue(approved)
        self.assertIn("Invalid input", out)

    def test_eof_denies_instead_of_raising(self) -> None:
        approved, _out, err = self._readapproval(EOFError())
        self.assertFalse(approved)
        self.assertIn("denied", err)

    def test_eof_after_invalid_answer_denies(self) -> None:
        approved, _out, err = self._readapproval("maybe", EOFError())
        self.assertFalse(approved)
        self.assertIn("denied", err)


class ConsoleInputterTests(unittest.TestCase):
    def _read(self, *inputs: str | BaseException) -> tuple[str | None, str, str]:
        return _call_with_inputs(ConsoleInputter().read, *inputs)

    def test_returns_the_typed_line(self) -> None:
        result, _out, _err = self._read("hello")
        self.assertEqual(result, "hello")

    def test_empty_line_means_end_of_input(self) -> None:
        result, _out, _err = self._read("")
        self.assertIsNone(result)

    def test_eof_behaves_like_an_empty_line(self) -> None:
        result, _out, _err = self._read(EOFError())
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
