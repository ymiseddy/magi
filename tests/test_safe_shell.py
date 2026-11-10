"""Tests for the read-only command validator.

The corpus of rejected commands is the point of this file: every entry either
writes, executes something else, or reads outside the project while looking
harmless. Adding a bypass here whenever one is found is what keeps the
allowlist honest.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing_extensions import override

from magi.safe_shell import (
    MAX_STAGES,
    READ_ONLY_COMMANDS,
    SAFE_BIN_DIRS,
    RejectedCommand,
    tokenize,
    validate_command,
)


class SafeShellTestCase(unittest.TestCase):
    """Shared fixture: a throwaway project root."""

    root: Path
    _tmp: tempfile.TemporaryDirectory[str]

    @override
    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        _ = (self.root / "src").mkdir()
        _ = (self.root / "README.md").write_text("hello", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def validate(self, command: str) -> list[list[str]]:
        return validate_command(command, self.root)

    def rejection(self, command: str) -> str:
        """Return the refusal message for a command that must be rejected."""
        with self.assertRaises(RejectedCommand) as ctx:
            _ = self.validate(command)
        return str(ctx.exception)

    def assert_rejected(self, command: str) -> None:
        _ = self.rejection(command)


class AcceptsReadOnlyCommands(SafeShellTestCase):
    def test_common_inspection_commands(self) -> None:
        commands = [
            "ls -la",
            "ls src",
            "cat README.md",
            "cat README.md | head -20",
            "head -n 5 README.md",
            "tail -20 README.md",
            "wc -l README.md",
            "pwd",
            "stat README.md",
            "file README.md",
            "sort -u README.md | head -2",
            "uniq README.md",
            "diff README.md README.md",
            "echo hello",
            "find . -maxdepth 2 -name '*.py' -type f",
            "find . -name '*.py' -print0",
            "grep -rn hello src/",
            "grep -c hello README.md | wc -l",
            "git status",
            "git log --oneline -5",
            "git diff --stat",
            "git diff HEAD~1",
            "git show HEAD:README.md",
            "git ls-files",
            "git blame README.md",
            "git rev-parse HEAD",
        ]
        for command in commands:
            with self.subTest(command=command):
                stages = self.validate(command)
                self.assertTrue(stages)
                self.assertTrue(all(stage for stage in stages))

    def test_quoted_glob_is_passed_through_to_the_program(self) -> None:
        stages = self.validate("rg -g '*.py' needle")
        self.assertEqual(stages[0][1:], ["-g", "*.py", "needle"])

    def test_quoted_metacharacters_are_literal_arguments(self) -> None:
        # No shell ever sees this, so ';' and '|' inside quotes are just text.
        stages = self.validate("echo 'a;b|c'")
        self.assertEqual(stages[0][1:], ["a;b|c"])

    def test_pipeline_stages_are_split(self) -> None:
        stages = self.validate("cat README.md | head -1")
        self.assertEqual(len(stages), 2)
        self.assertTrue(stages[1][0].endswith("/head"))

    def test_programs_resolve_into_trusted_directories(self) -> None:
        for command in ("ls", "cat README.md", "git status"):
            with self.subTest(command=command):
                program = self.validate(command)[0][0]
                self.assertTrue(
                    any(program.startswith(f"{directory}/") for directory in SAFE_BIN_DIRS),
                    f"{program} is not in a trusted bin directory",
                )

    def test_longest_pipeline_is_accepted(self) -> None:
        command = " | ".join(["cat README.md"] + ["head -1"] * (MAX_STAGES - 1))
        self.assertEqual(len(self.validate(command)), MAX_STAGES)

    def test_allowlist_is_not_empty(self) -> None:
        self.assertIn("cat", READ_ONLY_COMMANDS)
        self.assertIn("git", READ_ONLY_COMMANDS)


class RejectsShellSyntax(SafeShellTestCase):
    """Anything a shell would interpret must never reach a shell."""

    def test_command_chaining(self) -> None:
        for command in (
            "echo hi; rm -rf /",
            "ls && rm -rf /",
            "ls || rm -rf /",
            "ls & rm -rf /",
            "ls | rm -rf /",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_expansions_and_substitution(self) -> None:
        for command in (
            "cat $(rm -rf /)",
            "cat `rm -rf /`",
            "echo $HOME",
            "echo ${HOME}",
            'cat "$(id)"',
            "ls ~",
            "ls ~/",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_redirection(self) -> None:
        for command in (
            "ls > out.txt",
            "ls >> out.txt",
            "cat < README.md",
            "ls 2> /dev/null",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_program_name_spoofing(self) -> None:
        # shlex.split("c\\at f") == ["cat", "f"], so an argv[0] check alone
        # would approve this.
        for command in (
            "c\\at README.md",
            "./cat README.md",
            "/bin/cat README.md",
            "src/cat README.md",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_environment_assignment_prefix(self) -> None:
        self.assert_rejected("FOO=bar ls")

    def test_multiple_commands_on_separate_lines(self) -> None:
        self.assert_rejected("ls\nrm -rf /")

    def test_unbalanced_quotes(self) -> None:
        for command in ("cat 'unbalanced", 'cat "unbalanced', "cat a'b"):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_empty_command(self) -> None:
        for command in ("", "   ", "|", "ls |", "| ls", "ls | | wc"):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_pipeline_too_long(self) -> None:
        command = " | ".join(["cat README.md"] + ["head -1"] * MAX_STAGES)
        self.assertIn(str(MAX_STAGES), self.rejection(command))

    def test_overlong_command(self) -> None:
        self.assert_rejected("ls " + "a" * 5000)


class RejectsDangerousPrograms(SafeShellTestCase):
    def test_writing_programs(self) -> None:
        for command in (
            "rm -rf /",
            "mv README.md src/",
            "cp README.md /tmp/",
            "touch newfile",
            "mkdir newdir",
            "chmod 777 README.md",
            "ln -s README.md link",
            "tee out.txt",
            "dd if=README.md of=out.txt",
            "install README.md /tmp/",
            "truncate -s0 README.md",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_shells_and_interpreters(self) -> None:
        for command in (
            "bash -c 'rm -rf /'",
            "sh -c 'rm -rf /'",
            "python -c 'print(1)'",
            "perl -e 'unlink 1'",
            "ruby -e 'puts 1'",
            "node -e '1'",
            "awk 'BEGIN{system(\"id\")}'",
            "sed -i 's/a/b/' README.md",
            "xargs rm",
            "env rm -rf /",
            "eval rm -rf /",
            "command rm -rf /",
            "exec rm -rf /",
            "nohup rm -rf /",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_network_programs(self) -> None:
        for command in ("curl http://example.com", "wget http://example.com", "nc -z host 80"):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_write_flags_on_allowed_programs(self) -> None:
        for command in (
            "sort -o out.txt README.md",
            "sort --output=out.txt README.md",
            "sort --compress-program=/bin/sh README.md",
            "find . -delete",
            "find . -exec rm {} +",
            "find . -execdir rm {} +",
            "find . -ok rm {} +",
            "find . -fls out.txt",
            "find . -fprint out.txt",
            "find . -fprintf out.txt x",
            "rg --pre ./evil needle",
            "rg --pre-glob '*.md' --pre ./evil needle",
            "rg -z needle",
            "rg --search-zip needle",
            "file -C -m magic",
            "date -s '2020-01-01'",
            "tail -f README.md",
            "tail -F README.md",
            "tree -o out.txt",
            "uniq README.md out.txt",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_git_writes_and_config_escapes(self) -> None:
        for command in (
            "git commit -m x",
            "git checkout main",
            "git reset --hard",
            "git clean -xfd",
            "git push",
            "git pull",
            "git rm README.md",
            "git mv README.md src/",
            "git stash",
            "git apply patch.diff",
            "git rebase main",
            "git config core.pager sh",
            "git -c core.pager=sh log",
            "git --exec-path=/tmp log",
            "git log --output=out.txt",
            "git diff --ext-diff",
            "git grep --open-files-in-pager=sh needle",
            "git upload-pack",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_find_whitelist_is_exhaustive(self) -> None:
        # Unlisted primaries fail closed, so new find features stay blocked
        # until someone deliberately allows them.
        self.assert_rejected("find . -newerct '2020-01-01'")


class RejectsPathsOutsideProject(SafeShellTestCase):
    def test_absolute_paths(self) -> None:
        for command in ("cat /etc/passwd", "ls /root", "grep -r password /etc"):
            with self.subTest(command=command):
                self.assertIn("outside the project", self.rejection(command))

    def test_parent_traversal(self) -> None:
        for command in (
            "cat ../secret",
            "cat ../../etc/passwd",
            "ls src/../..",
            "grep -r x ../",
        ):
            with self.subTest(command=command):
                self.assert_rejected(command)

    def test_home_paths(self) -> None:
        self.assert_rejected("cat ~/.ssh/id_rsa")
        self.assert_rejected("cat $HOME/.ssh/id_rsa")

    def test_symlink_out_of_project(self) -> None:
        link = self.root / "escape"
        try:
            link.symlink_to("/etc")
        except OSError:
            self.skipTest("cannot create symlinks here")
        self.assertIn("outside the project", self.rejection("cat escape/passwd"))

    def test_paths_inside_project_are_fine(self) -> None:
        _ = self.validate("cat src/../README.md")
        _ = self.validate("ls .")


class TokenizerBehaviour(unittest.TestCase):
    def test_splits_on_whitespace(self) -> None:
        self.assertEqual(tokenize("ls  -l\t-a"), ["ls", "-l", "-a"])

    def test_preserves_pipe_tokens(self) -> None:
        self.assertEqual(tokenize("cat a | head"), ["cat", "a", "|", "head"])

    def test_single_quotes_are_literal(self) -> None:
        self.assertEqual(tokenize("rg 'a b'"), ["rg", "a b"])

    def test_tilde_only_expands_at_word_start(self) -> None:
        self.assertEqual(tokenize("git show HEAD~1"), ["git", "show", "HEAD~1"])

    def test_rejects_dollar_in_double_quotes(self) -> None:
        with self.assertRaises(RejectedCommand):
            _ = tokenize('echo "$HOME"')


@unittest.skipUnless(shutil.which("git"), "git is required")
class GitSubcommandChecks(unittest.TestCase):
    def test_allows_readonly_subcommand_shapes(self) -> None:
        root = Path.cwd().resolve()
        for command in ("git log", "git show --stat", "git diff HEAD~1", "git status"):
            with self.subTest(command=command):
                self.assertTrue(validate_command(command, root))


if __name__ == "__main__":
    _ = unittest.main()
