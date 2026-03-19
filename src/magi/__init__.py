# pyright: reportImportCycles=false

import asyncio
import sys
from typing import cast
import dotenv
from pydantic_ai import Agent


from .repl import MagiRepl
from .session import FileSessionManager, NoOpSessionManager, SessionManager
from .io import ReaderWriter
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


def _available_model_names(config: dict[str, object]) -> list[str]:
    models_cfg = _object_dict(config.get("models"))
    if models_cfg is None:
        return []

    return sorted(models_cfg.keys())


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
    requested_key = args.system_prompt or ("watch" if args.watch else None)
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

    builder = OpenAIAgentBuilder() \
        .with_url(_require_str(raw_model, "base_url")) \
        .with_api_key(_require_str(raw_model, "api_key")) \
        .with_system_prompt(system_prompt) \
        .using_model(_require_str(raw_model, "model")) \
        .with_tools() \
        .maybe_with_skills()

    if args.auto_approve:
        builder = builder.without_tool_approval() \
                    .without_skill_approval()


    agent = builder.build()
    return agent


class Dependencies:
    def __init__(self, config: dict[str, object], args: CommandArguments, isatty: bool=False) -> None:
        self._config: dict[str, object] = config
        self._args: CommandArguments = args

        self._agent: Agent | None = None
        self._io: ReaderWriter | None = None
        self._session_manager: SessionManager | None = None
        self._isatty: bool = isatty

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = build_agent(self._args, self._config)

        return self._agent

    @property
    def io(self) -> ReaderWriter:
        if self._io is None:
            if not self._isatty:
                self._io = ReaderWriter.non_interactive()
            else:
                self._io = ReaderWriter.console()

        return self._io

    @property
    def session_manager(self) -> SessionManager:
        if self._session_manager is None:
            if self._args.no_session:
                self._session_manager = NoOpSessionManager()
            else:
                session_name = self._args.session or "default"
                self._session_manager = FileSessionManager(session_name)

        return self._session_manager

    @property
    def magi_session(self) -> MagiRepl:
        return MagiRepl(
            agent=self.agent,
            io=self.io,
            session_manager=self.session_manager,
            available_models=_available_model_names(self._config),
        )

def main() -> None:
    prompt = None
    isatty = sys.stdin.isatty() and sys.stdout.isatty()
    if not isatty:
        prompt = sys.stdin.read()

    _ = dotenv.load_dotenv()
    args = CommandArguments(sys.argv[1:]) 
    if args.query:
        prompt = args.query
    cfg = config.load_config()

    deps = Dependencies(cfg, args, isatty)
    magi_session = deps.magi_session
    if args.watch:
        asyncio.run(magi_session.watch())
        return

    asyncio.run(magi_session.run(prompt, isatty))
