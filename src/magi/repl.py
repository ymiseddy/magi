from dataclasses import dataclass
import asyncio
import os
from pathlib import Path
import subprocess
import shlex
from collections.abc import AsyncIterator
from typing import Callable, TypeVar, cast, override

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, DeferredToolRequests, DeferredToolResults, ModelMessage, PartStartEvent, PartDeltaEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta
from watchdog.events import DirCreatedEvent, DirDeletedEvent, DirModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent, FileMovedEvent
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from magi.io import OTYPE_PROMPT, OTYPE_RESULT, OTYPE_THINKING, ReaderWriter
from magi.session import SessionManager

SlashCommandHandler = Callable[[list[str], list[ModelMessage]], tuple[bool, list[ModelMessage]]]


@dataclass(frozen=True)
class SlashCommandDefinition:
    name: str
    description: str

F = TypeVar("F", bound=Callable[..., object])

def slashcommand(name: str, description: str) -> Callable[[F], F]:
    """Attach slash-command metadata to a handler method."""

    def decorator(func: F) -> F:
        setattr(func, "_slash_command_definition", SlashCommandDefinition(name=name, description=description))
        return func

    return decorator


class MagiRepl:
    def __init__(
        self,
        agent: Agent,
        io: ReaderWriter,
        session_manager: SessionManager,
        available_models: list[str] | None = None,
        available_system_prompts: list[str] | None = None,
        default_system_prompt: str | None = None,
    ) -> None:
        self.agent: Agent = agent
        self.io: ReaderWriter = io
        self.session_manager: SessionManager = session_manager
        self.available_models: list[str] = available_models or []
        self.available_system_prompts: list[str] = available_system_prompts or []
        self.default_system_prompt: str | None = default_system_prompt
        self._command_handlers: dict[str, SlashCommandHandler] = {}
        self._command_descriptions: dict[str, str] = {}
        self._register_decorated_commands()

    def _register_decorated_commands(self) -> None:
        """Register slash command handlers declared via decorator metadata."""
        for attribute_name in dir(self):
            handler = cast(object, getattr(self, attribute_name))  # type: ignore[no-untyped-call]
            if not callable(handler):
                continue
            definition: object = getattr(handler, "_slash_command_definition", None)  # type: ignore[no-untyped-call]
            if not isinstance(definition, SlashCommandDefinition):
                continue
            # At this point we know handler is a callable with the slash command signature
            typed_handler = cast(SlashCommandHandler, handler)
            self.register_slash_command(definition.name, typed_handler, definition.description)

    @slashcommand("clear", "Clear current session history.")
    def _slash_clear(self, _args: list[str], _session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        new_history: list[ModelMessage] = []
        self.io.writeln(OTYPE_RESULT, "Session history cleared.")
        return True, new_history

    @slashcommand("save", "Save session history to disk.")
    def _slash_save(self, _args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        self.session_manager.save(session_history)
        self.io.writeln(OTYPE_RESULT, "Session saved.")
        return True, session_history

    @slashcommand("load", "Reload session history from disk.")
    def _slash_load(self, _args: list[str], _session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        new_history = self.session_manager.load()
        self.io.writeln(OTYPE_RESULT, "Session history reloaded.")
        return True, new_history


    @slashcommand("history", "Show current session history.")
    def _slash_history(self, _args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        if not session_history:
            self.io.writeln(OTYPE_RESULT, "Session history is empty.")
            return True, session_history
        history_str = "\n".join(
            f"{i+1}. [{msg}]" for i, msg in enumerate(session_history)
        )
        self.io.writeln(OTYPE_RESULT, f"Current session history:\n{history_str}")
        return True, session_history

    @slashcommand("help", "Show this help.")
    def _slash_help(self, _: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        commands = "\n".join(
            f"  /{name:<7} - {description}"
            for name, description in sorted(self._command_descriptions.items())
        )
        self.io.writeln(OTYPE_RESULT, f"Available slash commands:\n{commands}")
        return True, session_history

    @slashcommand("models", "List available models.")
    def _slash_models(self, _args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        if not self.available_models:
            self.io.writeln(OTYPE_RESULT, "No models are configured.")
            return True, session_history

        models = "\n".join(f"  {name}" for name in self.available_models)
        self.io.writeln(OTYPE_RESULT, f"Available models:\n{models}")
        return True, session_history

    @slashcommand("prompts", "List available system prompts.")
    def _slash_prompts(self, _args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        if not self.available_system_prompts:
            self.io.writeln(OTYPE_RESULT, "No system prompts are configured.")
            return True, session_history

        prompts = "\n".join(
            f"  {name}{' (default)' if name == self.default_system_prompt else ''}"
            for name in self.available_system_prompts
        )
        self.io.writeln(OTYPE_RESULT, f"Available system prompts:\n{prompts}")
        return True, session_history


    def _get_approvals(self, requests: DeferredToolRequests) -> DeferredToolResults:
        approval_results = DeferredToolResults()
        for approval in requests.approvals:
            # Now what?
            prompt = f"\n\nAgent is requesting approval to run `{approval.tool_name}` with arguments `{approval.args}`. Approve? (y/n)"
            self.io.write(OTYPE_PROMPT, f"{prompt}")
            approved = self.io.readapproval()
            approval_results.approvals[approval.tool_call_id] = approved
            self.io.write(OTYPE_PROMPT, f"\n\n")
        return approval_results

    def _process_slash_command(self, cmd: str, args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        """Process slash command. Returns (handled, new_history)."""
        handler = self._command_handlers.get(cmd)
        if handler is not None:
            return handler(args, session_history)
        # Unknown command, treat as not handled (maybe pass to agent)
        self.io.writeln(OTYPE_RESULT, f"Unknown slash command '{cmd}'. Treating as regular prompt.")
        return False, session_history

    def register_slash_command(self, name: str, handler: SlashCommandHandler, description: str = "") -> None:
        """Register a custom slash command handler."""
        self._command_handlers[name] = handler
        self._command_descriptions[name] = description

    def _try_slash_command(self, prompt: str, session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        """
        Handle slash command if applicable.
        Returns (handled, new_history). If not a slash command or not handled, returns (False, session_history).
        """
        if not prompt.startswith('/'):
            return False, session_history
        try:
            parts = shlex.split(prompt.strip(), posix=True)
        except ValueError:
            # Fallback to simple splitting if quotes are mismatched
            parts = prompt.strip().split()
        if not parts:
            return False, session_history
        cmd = parts[0][1:]  # remove leading slash
        args = parts[1:] if len(parts) > 1 else []
        handled, new_history = self._process_slash_command(cmd, args, session_history)
        return handled, new_history


    async def _run_prompt(self, prompt: str, session_history: list[ModelMessage] ) -> list[ModelMessage]:
        complete = False
        approval_results: DeferredToolResults | None = None
        event_stream: AsyncIterator[AgentStreamEvent | AgentRunResultEvent[str | DeferredToolRequests]]
        while not complete:
            complete = True

            # Handle slash commands
            handled, new_history = self._try_slash_command(prompt, session_history)
            if handled:
                # Slash command handled; return updated history without agent interaction
                return new_history
            # If not handled, fall through to normal agent processing

            if approval_results is not None:
                event_stream = self.agent.run_stream_events(
                    message_history=session_history,
                    deferred_tool_results=approval_results,
                    output_type=[str, DeferredToolRequests]
                )
            else:
                event_stream = self.agent.run_stream_events(
                    prompt, 
                    message_history=session_history,
                    output_type=[str, DeferredToolRequests]
                )
            mode: str | None = None
            async for event in event_stream:
                match event:
                    case PartStartEvent(part=ThinkingPart(content=content)):
                        if mode != "thinking":
                            self.io.write(OTYPE_THINKING, "\n\n## Thinking...\n\n")
                            mode = "thinking"
                        self.io.write(OTYPE_THINKING, content)
                    case PartStartEvent(part=TextPart(content=content)):
                        if mode != "text":
                            self.io.write(OTYPE_RESULT, "\n\n## Text Response:\n\n")
                            mode = "text"
                        self.io.write(OTYPE_RESULT, content)
                    case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=content_delta)):
                        if mode != "thinking":
                            self.io.write(OTYPE_THINKING, "\n\n## Thinking...\n\n")
                            mode = "thinking"
                        if content_delta is not None:
                            self.io.write(OTYPE_THINKING, content_delta)
                    case PartDeltaEvent(delta=TextPartDelta(content_delta=content_delta)):
                        if mode != "text":
                            self.io.write(OTYPE_RESULT, "\n\n## Text Response:\n\n")
                            mode = "text"
                        self.io.write(OTYPE_RESULT, content_delta)
                    case AgentRunResultEvent() as agent_run_result:
                        result = agent_run_result.result
                        output = result.output
                        if isinstance(output, DeferredToolRequests):
                            complete = False
                            approval_results = self._get_approvals(output)

                        session_history = list(result.all_messages())
                    case _:
                        pass

            self.io.write(OTYPE_PROMPT, f"\n")
        return session_history


    async def run(self, prompt: str | None = None, _isatty: bool=False) -> None:
        """Run the session in either interactive or non-interactive mode."""

        if prompt is not None:
            await self.run_non_interactive(prompt)
        else:
            await self.run_interactive()

    async def run_interactive(self) -> None:
        """Run an interactive session, prompting the user for input until they exit."""
        session_history = self.session_manager.load()

        while True:
            self.io.write(OTYPE_PROMPT, "> ")
            prompt = self.io.read()
            if not prompt:
                break

            session_history = await self._run_prompt(prompt, session_history)
            self.io.write(OTYPE_PROMPT, f"\n")


        self.session_manager.save(session_history)

    async def run_non_interactive(self, prompt: str) -> None:
        """Run a single prompt without interactive input, then exit."""
        session_history = self.session_manager.load()
        session_history = await self._run_prompt(prompt, session_history)
        self.session_manager.save(session_history)

    def _is_git_ignored(self, path: Path) -> bool:
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", str(path)],
                cwd=Path.cwd(),
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            return False

        if result.returncode == 0:
            return True
        if result.returncode == 128:
            return False
        return False

    def _line_requests_ai(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped.endswith("AI!"):
            return False

        comment_prefixes = ("#", "//", "--", ";", "%", "/*", "*", "<!--")
        return stripped.startswith(comment_prefixes)

    def _file_requests_ai(self, path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as f:
                return any(self._line_requests_ai(line) for line in f)
        except (OSError, UnicodeDecodeError):
            return False

    def _build_watch_prompt(self, path: Path, contents: str) -> str:
        return (
            f"File: {path}\n\n"
            "A watched file changed and contains one or more comment lines ending with `AI!`.\n"
            "Implement the requested change in this file.\n\n"
            "```text\n"
            f"{contents}\n"
            "```"
        )

    def _is_watchable_path(self, path: Path) -> bool:
        root = Path.cwd()

        try:
            relpath = path.relative_to(root)
        except ValueError:
            return False

        if any(part.startswith(".") for part in relpath.parts):
            return False

        return not self._is_git_ignored(relpath)

    def _iter_watch_files(self) -> list[Path]:
        root = Path.cwd()
        files: list[Path] = []

        for current_root, dirnames, filenames in root.walk(top_down=True):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not dirname.startswith(".")
                and not self._is_git_ignored(Path(current_root, dirname).relative_to(root))
            ]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                path = Path(current_root, filename)
                relpath = path.relative_to(root)
                if self._is_git_ignored(relpath):
                    continue

                if path.is_file():
                    files.append(path)

        return files

    def _start_watchdog_observer(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[Path | tuple[str, Path]],
    ) -> BaseObserver:
        root = Path.cwd()
        repl = self

        class ChangeHandler(FileSystemEventHandler):
            def _enqueue(self, item: Path | tuple[str, Path]) -> None:
                _ = loop.call_soon_threadsafe(queue.put_nowait, item)

            def _event_path(self, raw_path: bytes | str) -> Path:
                return Path(os.fsdecode(raw_path))

            def _handle_file_event(self, event: FileSystemEvent) -> None:
                if event.is_directory:
                    return

                path = self._event_path(event.src_path)
                if repl._is_watchable_path(path):
                    self._enqueue(path)

            @override
            def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
                self._handle_file_event(event)

            @override
            def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
                self._handle_file_event(event)

            @override
            def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
                if event.is_directory:
                    return

                path = self._event_path(event.src_path)
                if repl._is_watchable_path(path):
                    self._enqueue(("deleted", path))

            @override
            def on_moved(self, event: FileMovedEvent | FileSystemMovedEvent) -> None:
                if event.is_directory:
                    return

                src_path = self._event_path(event.src_path)
                dest_path = self._event_path(event.dest_path)

                if repl._is_watchable_path(src_path):
                    self._enqueue(("deleted", src_path))
                if repl._is_watchable_path(dest_path):
                    self._enqueue(dest_path)

        observer = Observer()
        _ = observer.schedule(ChangeHandler(), str(root), recursive=True)
        observer.start()
        return observer

    async def watch(self) -> None:
        """Watch mode: watch for changes to files in the current directory and subdirectories,
           if the file contains a comment line ending with `AI!`, send the file contents as a 
            prompt to the agent. Do not monitor files starting with a dot (e.g. .sessions/).
            Also, respect .gitignore if present, and do not monitor ignored files.
            """
        session_history = self.session_manager.load()
        known_mtimes: dict[Path, int] = {}

        for path in self._iter_watch_files():
            try:
                known_mtimes[path] = path.stat().st_mtime_ns
            except OSError:
                continue

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Path | tuple[str, Path]] = asyncio.Queue()
        observer = self._start_watchdog_observer(loop, queue)

        self.io.writeln(OTYPE_RESULT, "Watching for AI! comments...")

        try:
            while True:
                event = await queue.get()
                if isinstance(event, tuple):
                    action, path = event
                    if action == "deleted":
                        _ = known_mtimes.pop(path, None)
                    continue

                path = event
                try:
                    stat = path.stat()
                except OSError:
                    _ = known_mtimes.pop(path, None)
                    continue

                mtime_ns = stat.st_mtime_ns
                previous_mtime = known_mtimes.get(path)
                known_mtimes[path] = mtime_ns

                if previous_mtime is None or previous_mtime == mtime_ns:
                    continue

                if not self._file_requests_ai(path):
                    continue

                try:
                    contents = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                self.io.writeln(OTYPE_RESULT, f"Detected AI! request in {path}")
                prompt = self._build_watch_prompt(path.relative_to(Path.cwd()), contents)
                session_history = await self._run_prompt(prompt, session_history)
                self.session_manager.save(session_history)
        finally:
            observer.stop()
            await asyncio.to_thread(observer.join)
