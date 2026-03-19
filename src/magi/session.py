

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shlex
from typing import Any, override, Callable
from collections.abc import AsyncIterator

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, DeferredToolRequests, DeferredToolResults, ModelMessage, ModelMessagesTypeAdapter, PartStartEvent, PartDeltaEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta

from magi.io import OTYPE_PROMPT, OTYPE_RESULT, OTYPE_THINKING, ReaderWriter
 
import asyncio


SESSION_DIRECTORY = ".sessions"


SlashCommandHandler = Callable[[list[str], list[ModelMessage]], tuple[bool, list[ModelMessage]]]


@dataclass(frozen=True)
class SlashCommandDefinition:
    name: str
    description: str


def slashcommand(name: str, description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach slash-command metadata to a handler method."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_slash_command_definition", SlashCommandDefinition(name=name, description=description))
        return func

    return decorator

class SessionManager(ABC):

    @abstractmethod
    def load(self) -> list[ModelMessage]:
        pass


    @abstractmethod
    def save(self, history: list[ModelMessage]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

class NoOpSessionManager(SessionManager):

    @override
    def load(self) -> list[ModelMessage]:
        return []

    @override
    def save(self, history: list[ModelMessage]) -> None:
        pass

    @override
    def clear(self) -> None:
        pass


class FileSessionManager(SessionManager):

    def __init__(self, session_name: str) -> None:
        self.session_name: str = session_name

    def _session_history_path(self) -> str:
        """Return the session-history path for the given model selection."""

        return os.path.join(
            SESSION_DIRECTORY,
            f"{self.session_name}.json",
        )


    @override
    def load(self) -> list[ModelMessage]:
        """Load session history from disk, falling back to a legacy location."""

        path = self._session_history_path()
        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                history_json = f.read()
            return ModelMessagesTypeAdapter.validate_json(history_json)
        except Exception:
            raise


    @override
    def save(self, history: list[ModelMessage]) -> None:
        """Persist a session history to disk."""
        history_blob = ModelMessagesTypeAdapter.dump_json(history)
        path = self._session_history_path()

        # Create the session directory if it doesn't exist
        os.makedirs(SESSION_DIRECTORY, exist_ok=True)
        with open(path, "wb") as f:
            _ = f.write(history_blob)


    @override
    def clear(self) -> None:
        """Delete stored session history files.

        Returns a tuple of (deleted_any, errors).
        """
        path = self._session_history_path()
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except OSError as _exc:
            raise

class MagiSession:
    def __init__(self, agent: Agent, io: ReaderWriter, session_manager: SessionManager) -> None:
        self.agent: Agent = agent
        self.io: ReaderWriter = io
        self.session_manager: SessionManager = session_manager
        self._command_handlers: dict[str, SlashCommandHandler] = {}
        self._command_descriptions: dict[str, str] = {}
        self._register_decorated_commands()

    def _register_decorated_commands(self) -> None:
        """Register slash command handlers declared via decorator metadata."""
        for attribute_name in dir(self):
            handler = getattr(self, attribute_name)
            definition = getattr(handler, "_slash_command_definition", None)
            if definition is None:
                continue
            self.register_slash_command(definition.name, handler, definition.description)

    @slashcommand("clear", "Clear current session history.")
    def _slash_clear(self, args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        new_history: list[ModelMessage] = []
        self.io.writeln(OTYPE_RESULT, "Session history cleared.")
        return True, new_history

    @slashcommand("save", "Save session history to disk.")
    def _slash_save(self, args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        self.session_manager.save(session_history)
        self.io.writeln(OTYPE_RESULT, "Session saved.")
        return True, session_history

    @slashcommand("load", "Reload session history from disk.")
    def _slash_load(self, args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        new_history = self.session_manager.load()
        self.io.writeln(OTYPE_RESULT, "Session history reloaded.")
        return True, new_history


    @slashcommand("history", "Show current session history.")
    def _slash_history(self, args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        if not session_history:
            self.io.writeln(OTYPE_RESULT, "Session history is empty.")
            return True, session_history
        history_str = "\n".join(
            f"{i+1}. [{msg}]" for i, msg in enumerate(session_history)
        )
        self.io.writeln(OTYPE_RESULT, f"Current session history:\n{history_str}")
        return True, session_history

    @slashcommand("help", "Show this help.")
    def _slash_help(self, args: list[str], session_history: list[ModelMessage]) -> tuple[bool, list[ModelMessage]]:
        commands = "\n".join(
            f"  /{name:<7} - {description}"
            for name, description in sorted(self._command_descriptions.items())
        )
        self.io.writeln(OTYPE_RESULT, f"Available slash commands:\n{commands}")
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


    async def run(self, prompt: str | None = None, isatty: bool=False) -> None:
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
        await asyncio.sleep(.5)

    async def run_non_interactive(self, prompt: str) -> None:
        """Run a single prompt without interactive input, then exit."""
        session_history = self.session_manager.load()
        session_history = await self._run_prompt(prompt, session_history)
        self.session_manager.save(session_history)
