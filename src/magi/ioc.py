import sys
from typing import TYPE_CHECKING, cast
from pydantic_ai import Agent


from .repl import MagiRepl
from .session import FileSessionManager, NoOpSessionManager, SessionManager
from .io import ReaderWriter
from .arguments import CommandArguments

if TYPE_CHECKING:
    from .agent_builder import ToolApprovalPolicy

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


def _available_system_prompt_names(config: dict[str, object]) -> list[str]:
    prompts_cfg = _string_dict(config.get("system_prompts"))
    if prompts_cfg is None:
        return []

    return sorted(prompts_cfg.keys())


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




TOOL_APPROVAL_KEYS = frozenset(
    {"auto_approve", "approved_tools", "require_approval", "auto_approve_skills"}
)


def _resolve_tool_approval_policy(config: dict[str, object]) -> "ToolApprovalPolicy":
    """
    Build the per-tool approval policy from the `tool_approval` config section,
    falling back to the legacy top level `auto_approve` flag. Malformed or
    misspelled entries are reported so a typo does not silently change nothing.
    """
    from .agent_builder import ToolApprovalPolicy

    section = config.get("tool_approval")
    if section is None:
        return ToolApprovalPolicy.from_config(config)

    if not isinstance(section, dict):
        print(
            "Warning: 'tool_approval' must be a mapping; ignoring it.",
            file=sys.stderr,
        )
        return ToolApprovalPolicy(auto_approve_all=config.get("auto_approve") is True)

    unknown_keys = sorted(
        key
        for key in cast(dict[object, object], section)
        if isinstance(key, str) and key not in TOOL_APPROVAL_KEYS
    )
    if unknown_keys:
        message = (
            f"Warning: unknown tool_approval key(s) {', '.join(unknown_keys)}; "
            f"expected one of {', '.join(sorted(TOOL_APPROVAL_KEYS))}."
        )
        print(message, file=sys.stderr)

    return ToolApprovalPolicy.from_config(config)


class Dependencies:
    def __init__(self, config: dict[str, object], args: CommandArguments, isatty: bool=False) -> None:
        self._config: dict[str, object] = config
        self._args: CommandArguments = args

        self._agent: Agent | None = None
        self._io: ReaderWriter | None = None
        self._session_manager: SessionManager | None = None
        self._isatty: bool = isatty


    def __build_agent(self) -> Agent:
        
        default_model = self._config.get("default_model")
        model_name = self._args.model or (default_model if isinstance(default_model, str) else "deepseek-reasoner")
        models_cfg = _object_dict(self._config.get("models"))
        if models_cfg is None:
            raise ValueError("Config must define a 'models' mapping.")

        raw_model = _object_dict(models_cfg.get(model_name))
        if raw_model is None:
            raise ValueError(f"Model '{model_name}' not found in config.")

        system_prompt = _resolve_system_prompt(self._args, self._config)
        from .agent_builder import OpenAIAgentBuilder

        approval_policy = _resolve_tool_approval_policy(self._config)
        if self._args.auto_approve:
            # -y on the command line wins over anything configured in magi.yaml.
            approval_policy = approval_policy.approve_everything()

        builder = OpenAIAgentBuilder() \
            .with_url(_require_str(raw_model, "base_url")) \
            .with_api_key(_require_str(raw_model, "api_key")) \
            .with_system_prompt(system_prompt) \
            .using_model(_require_str(raw_model, "model")) \
            .with_tool_approval(approval_policy) \
            .with_tools() \
            .maybe_with_skills()

        agent = builder.build()
        return agent

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = self.__build_agent()

        return self._agent

    @property
    def io(self) -> ReaderWriter:
        if self._io is None:
            if not self._isatty:
                self._io = ReaderWriter.non_interactive()
            else:
                self._io = ReaderWriter.rich()

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
            available_system_prompts=_available_system_prompt_names(self._config),
            default_system_prompt=(
                cast(str, self._config.get("default_system_prompt"))
                if isinstance(self._config.get("default_system_prompt"), str)
                else None
            ),
        )

