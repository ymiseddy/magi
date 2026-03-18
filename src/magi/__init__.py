# pyright: reportImportCycles=false

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import cast
import dotenv

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, DeferredToolRequests, DeferredToolResults, ModelMessage, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta

from magi.session import FileSessionManager, NoOpSessionManager, SessionManager
from .io import ReaderWriter, OTYPE_RESULT, OTYPE_THINKING, OTYPE_PROMPT
from .arguments import CommandArguments
from . import config


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can perform tasks for the user."
    " You have access to tools and can use them to accomplish tasks."
    " Always try to use tools when appropriate, and be sure to follow"
    " the instructions provided by the user."
)


def _string_dict(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    result: dict[str, str] = {}
    for key, item in cast(dict[object, object], value).items():
        if isinstance(key, str) and isinstance(item, str):
            result[key] = item
    return result


def _object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if isinstance(key, str):
            result[key] = item
    return result


def _require_str(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Model config field '{key}' must be a string.")
    return value


def _resolve_system_prompt(args: CommandArguments, config: dict[str, object]) -> str:
    """
    Determine which system prompt to use based on CLI arguments and configuration.
    Prefers prompts defined under system_prompts, falls back to legacy system_prompt
    string, and ultimately uses DEFAULT_SYSTEM_PROMPT.
    """
    prompts_cfg = _string_dict(config.get("system_prompts"))
    requested_key = args.system_prompt
    if prompts_cfg:
        default_key_obj = config.get("default_system_prompt")
        default_key = default_key_obj if isinstance(default_key_obj, str) else None
        prompt_key = requested_key or default_key

        if prompt_key is None and len(prompts_cfg) == 1:
            prompt_key = next(iter(prompts_cfg.keys()))

        if prompt_key is not None:
            prompt_value = prompts_cfg.get(prompt_key)
            if prompt_value is not None:
                return prompt_value
            print(f"Warning: system prompt '{prompt_key}' not found; using available default.", file=sys.stderr)

        if default_key and default_key in prompts_cfg:
            return prompts_cfg[default_key]

        # Fall back to the first prompt defined in the map.
        return next(iter(prompts_cfg.values()))

    legacy_prompt = config.get("system_prompt")
    if isinstance(legacy_prompt, str):
        if requested_key:
            print(
                f"Warning: system prompt key '{requested_key}' provided but system_prompts are not configured; using legacy system_prompt value.",
                file=sys.stderr,
            )
        return legacy_prompt

    return DEFAULT_SYSTEM_PROMPT


def build_agent(args: CommandArguments, config: dict[str, object]) -> Agent:
    default_model = config.get("default_model")
    model_name = args.model or (default_model if isinstance(default_model, str) else "deepseek-reasoner")
    models_cfg = _object_dict(config.get("models"))
    if models_cfg is None:
        raise ValueError("Config must define a 'models' mapping.")

    raw_model = _object_dict(models_cfg.get(model_name))
    if raw_model is None:
        raise ValueError(f"Model '{model_name}' not found in config.")

    system_prompt = _resolve_system_prompt(args, config)
    from .agent_builder import OpenAIAgentBuilder

    agent = OpenAIAgentBuilder() \
        .with_url(_require_str(raw_model, "base_url")) \
        .with_api_key(_require_str(raw_model, "api_key")) \
        .with_system_prompt(system_prompt) \
        .using_model(_require_str(raw_model, "model")) \
        .with_tools() \
        .maybe_with_skills() \
        .build()
    return agent


class MagiSession:
    def __init__(self, args: CommandArguments, config: dict[str, object]) -> None:
        self.agent: Agent = build_agent(args, config)
        self.io: ReaderWriter = ReaderWriter.console()
        self.session_manager: SessionManager

        if args.no_session:
            self.session_manager = NoOpSessionManager()
        else:
            session_name = args.session or "default"
            self.session_manager = FileSessionManager(session_name)

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


    async def _run_prompt(self, prompt: str, session_history: list[ModelMessage] ) -> list[ModelMessage]:
        complete = False
        approval_results: DeferredToolResults | None = None
        event_stream: AsyncIterator[AgentStreamEvent | AgentRunResultEvent[str | DeferredToolRequests]]
        while not complete:
            complete = True

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


def main() -> None:
    _ = dotenv.load_dotenv()
    args = CommandArguments(sys.argv[1:]) 
    cfg = config.load_config()

    producer = MagiSession(args, cfg)
    asyncio.run(producer.run_interactive())
