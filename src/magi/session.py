

from abc import ABC, abstractmethod
import os
import shlex
from typing import override
from collections.abc import AsyncIterator

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, DeferredToolRequests, DeferredToolResults, ModelMessage, ModelMessagesTypeAdapter, PartStartEvent, PartDeltaEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta

from magi.io import OTYPE_PROMPT, OTYPE_RESULT, OTYPE_THINKING, ReaderWriter
 
import asyncio


SESSION_DIRECTORY = ".sessions"

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

    async def run_non_interactive(self, _prompt: str) -> None:
        pass


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

        # Instead of just hardcoding the commands, we should have a more flexible system. Perhapks use a decorator to register slash commands on methods? We should search for any methods decorated (be mindful if this is a subclass of MagiSession) and build a command registry. AI!
        handled = True
        new_history = session_history
        if cmd == "clear":
            new_history = []
            self.io.writeln(OTYPE_RESULT, "Session history cleared.")
        elif cmd == "save":
            self.session_manager.save(session_history)
            self.io.writeln(OTYPE_RESULT, "Session saved.")
        elif cmd == "load":
            new_history = self.session_manager.load()
            self.io.writeln(OTYPE_RESULT, "Session history reloaded.")
        elif cmd == "help":
            self.io.writeln(OTYPE_RESULT, "Available slash commands:\n"
                                          "  /clear   - Clear current session history.\n"
                                          "  /save    - Save session history to disk.\n"
                                          "  /load    - Reload session history from disk.\n"
                                          "  /help    - Show this help.")
        else:
            # Unknown command, treat as not handled (maybe pass to agent)
            self.io.writeln(OTYPE_RESULT, f"Unknown slash command '{cmd}'. Treating as regular prompt.")
            handled = False
        return handled, new_history

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


    async def run_interactive(self) -> None:
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

