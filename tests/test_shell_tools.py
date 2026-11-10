"""Tests for the shell tools, the sandbox and the approval policy.

These cover the wiring rather than the grammar (see `test_safe_shell.py`): does
a refused command actually not run, does the sandbox backstop hold, and is
`bash_readonly` exempt from approval unless the config says otherwise.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, cast
from typing_extensions import override
from unittest import mock

from magi.agent_builder import SAFE_BY_DESIGN_TOOLS, ToolApprovalPolicy
from magi.sandbox import (
    MODE_OFF,
    MODE_REQUIRE,
    MODE_AUTO,
    SandboxSettings,
    SandboxUnavailable,
    available,
    clear_probe_cache,
    configure,
    wrap,
)
from magi.tools import bash_readonly

if TYPE_CHECKING:
    from pydantic_ai import RunContext

# The tools ignore their context, so tests pass a placeholder.
_CTX = cast("RunContext[None]", cast(object, None))


class ApprovalPolicyTests(unittest.TestCase):
    def test_bash_readonly_is_safe_by_design(self) -> None:
        self.assertIn("bash_readonly", SAFE_BY_DESIGN_TOOLS)
        self.assertFalse(ToolApprovalPolicy().requires_approval("bash_readonly"))
        # Everything else still prompts by default.
        self.assertTrue(ToolApprovalPolicy().requires_approval("bash"))
        self.assertTrue(ToolApprovalPolicy().requires_approval("edit_file"))

    def test_explicit_require_approval_still_wins(self) -> None:
        policy = ToolApprovalPolicy(require_approval_tools=frozenset({"bash_readonly"}))
        self.assertTrue(policy.requires_approval("bash_readonly"))

    def test_auto_approve_all_still_prompts_for_required_tools(self) -> None:
        policy = ToolApprovalPolicy.from_config(
            {"tool_approval": {"auto_approve": True, "require_approval": ["bash"]}}
        )
        self.assertFalse(policy.requires_approval("edit_file"))
        self.assertTrue(policy.requires_approval("bash"))

    def test_approve_everything_matches_the_cli_flag(self) -> None:
        policy = ToolApprovalPolicy().approve_everything()
        for tool_name in ("bash", "bash_readonly", "edit_file"):
            with self.subTest(tool=tool_name):
                self.assertFalse(policy.requires_approval(tool_name))

    def test_shipped_config_keeps_bash_gated(self) -> None:
        policy = ToolApprovalPolicy.from_config(
            {
                "tool_approval": {
                    "auto_approve": False,
                    "approved_tools": ["read_file"],
                    "require_approval": ["bash", "edit_file"],
                }
            }
        )
        self.assertTrue(policy.requires_approval("bash"))
        self.assertFalse(policy.requires_approval("read_file"))
        self.assertFalse(policy.requires_approval("bash_readonly"))


class SandboxTests(unittest.TestCase):
    root: Path
    _tmp: tempfile.TemporaryDirectory[str]

    @override
    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        configure(SandboxSettings())

    @override
    def tearDown(self) -> None:
        configure(SandboxSettings())

    def test_mode_off_never_wraps(self) -> None:
        configure(SandboxSettings(mode=MODE_OFF))
        self.assertIsNone(wrap(["/bin/true"], cwd=self.root))

    def test_require_mode_raises_when_bwrap_is_missing(self) -> None:
        configure(SandboxSettings(mode=MODE_REQUIRE))
        clear_probe_cache()
        self.addCleanup(clear_probe_cache)
        with mock.patch("magi.sandbox.shutil.which", return_value=None):
            with self.assertRaises(SandboxUnavailable):
                _ = wrap(["/bin/true"], cwd=self.root)

    @unittest.skipUnless(available(), "bubblewrap is not available")
    def test_wrapped_command_runs(self) -> None:
        wrapped = wrap(["/bin/echo", "hello"], cwd=self.root, clean_env=True)
        self.assertIsNotNone(wrapped)
        assert wrapped is not None
        completed = subprocess.run(wrapped, capture_output=True, text=True, check=False)
        self.assertEqual(completed.stdout.strip(), "hello")

    @unittest.skipUnless(available(), "bubblewrap is not available")
    def test_sandbox_blocks_writes_outside_the_project(self) -> None:
        # The backstop behind the allowlist: a command that got past validation
        # still cannot write anywhere without a read-write bind.
        script = f"echo x > {self.root}/probe.txt; echo x > /usr/local/magi-probe"
        wrapped = wrap(["/bin/sh", "-c", script], cwd=self.root)
        self.assertIsNotNone(wrapped)
        assert wrapped is not None
        completed = subprocess.run(
            wrapped, capture_output=True, text=True, check=False, timeout=30
        )
        self.assertFalse((self.root / "probe.txt").exists())
        self.assertIn("Read-only file system", completed.stdout + completed.stderr)

    @unittest.skipUnless(available(), "bubblewrap is not available")
    def test_writable_path_is_bound_read_write(self) -> None:
        wrapped = wrap(
            ["/bin/sh", "-c", "echo x > writable.txt"],
            cwd=self.root,
            writable=(self.root,),
        )
        self.assertIsNotNone(wrapped)
        assert wrapped is not None
        _ = subprocess.run(wrapped, capture_output=True, text=True, check=False, timeout=30)
        self.assertTrue((self.root / "writable.txt").exists())

    @unittest.skipUnless(available(), "bubblewrap is not available")
    def test_read_only_wrapper_has_no_network(self) -> None:
        wrapped = wrap(
            ["/bin/sh", "-c", "getent hosts example.com || echo no-net"],
            cwd=self.root,
            network=False,
            clean_env=True,
        )
        self.assertIsNotNone(wrapped)
        assert wrapped is not None
        completed = subprocess.run(
            wrapped, capture_output=True, text=True, check=False, timeout=30
        )
        self.assertIn("no-net", completed.stdout)

    @unittest.skipUnless(available(), "bubblewrap is not available")
    def test_working_directory_under_tmp_is_not_shadowed(self) -> None:
        # The sandbox mounts a tmpfs over /tmp; the cwd has to survive that.
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp).resolve()
            _ = (workdir / "notes.txt").write_text("alpha", encoding="utf-8")
            wrapped = wrap(["/usr/bin/cat", "notes.txt"], cwd=workdir, clean_env=True)
            self.assertIsNotNone(wrapped)
            assert wrapped is not None
            completed = subprocess.run(
                wrapped, capture_output=True, text=True, check=False, timeout=30
            )
            self.assertEqual(completed.stdout.strip(), "alpha")


class BashReadonlyToolTests(unittest.IsolatedAsyncioTestCase):
    root: Path
    _tmp: tempfile.TemporaryDirectory[str]

    @override
    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        _ = (self.root / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    @override
    def setUp(self) -> None:
        patcher = mock.patch.object(Path, "cwd", return_value=self.root)
        _ = patcher.start()
        self.addCleanup(patcher.stop)
        configure(SandboxSettings(mode=MODE_AUTO))

    @override
    def tearDown(self) -> None:
        configure(SandboxSettings())

    async def test_runs_an_allowed_command(self) -> None:
        output = await bash_readonly(_CTX, "cat notes.txt")
        self.assertIn("alpha", output)

    async def test_pipeline_output(self) -> None:
        output = await bash_readonly(_CTX, "cat notes.txt | head -1")
        self.assertEqual(output.strip(), "alpha")

    async def test_refuses_without_running(self) -> None:
        output = await bash_readonly(_CTX, "rm -rf notes.txt")
        self.assertTrue(output.startswith("Refused:"))
        self.assertTrue((self.root / "notes.txt").exists())

    async def test_refuses_path_escape(self) -> None:
        output = await bash_readonly(_CTX, "cat ../../etc/hostname")
        self.assertTrue(output.startswith("Refused:"))

    async def test_missing_match_is_not_an_error(self) -> None:
        output = await bash_readonly(_CTX, "grep zzzz notes.txt")
        self.assertEqual(output, "(no matches)")


if __name__ == "__main__":
    _ = unittest.main()
