from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_skills import SkillsDirectory, SkillsToolset

from .tools import bash, deepseek_fetch, deepseek_search, edit_file, read_file


def _string_list(value: object) -> list[str]:
    """Normalize a YAML list of tool names, ignoring non-string entries."""
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [
        item
        for item in cast(Iterable[object], value)
        if isinstance(item, str) and item.strip()
    ]


@dataclass(frozen=True)
class ToolApprovalPolicy:
    """Decides, per tool, whether a user approval prompt is required.

    Resolution order for a single tool, highest precedence first:

    1. the tool is listed in ``require_approval`` -> approval required
    2. the tool is listed in ``approved_tools`` -> approval not required
    3. ``auto_approve`` is true -> approval not required
    4. otherwise -> approval required
    """

    auto_approve_all: bool = False
    approved_tools: frozenset[str] = field(default_factory=frozenset)
    require_approval_tools: frozenset[str] = field(default_factory=frozenset)
    auto_approve_skills: bool = False

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ToolApprovalPolicy":
        """Build a policy from the ``tool_approval`` section of ``magi.yaml``.

        The legacy top level ``auto_approve`` flag is honored when the
        ``tool_approval`` section does not set ``auto_approve`` itself.
        """
        section = config.get("tool_approval")
        section_dict: dict[str, object] = (
            cast(dict[str, object], section) if isinstance(section, dict) else {}
        )
        if "auto_approve" in section_dict:
            auto_approve_value: object = section_dict["auto_approve"]
        else:
            auto_approve_value = config.get("auto_approve")

        return cls(
            auto_approve_all=auto_approve_value is True,
            approved_tools=frozenset(_string_list(section_dict.get("approved_tools"))),
            require_approval_tools=frozenset(
                _string_list(section_dict.get("require_approval"))
            ),
            auto_approve_skills=section_dict.get("auto_approve_skills") is True,
        )

    def requires_approval(self, tool_name: str) -> bool:
        """Return True when ``tool_name`` must be confirmed by the user."""
        if tool_name in self.require_approval_tools:
            return True
        if tool_name in self.approved_tools:
            return False
        return not self.auto_approve_all

    def with_approved(self, *tool_names: str) -> "ToolApprovalPolicy":
        """Return a copy that auto-approves the given tools."""
        return ToolApprovalPolicy(
            auto_approve_all=self.auto_approve_all,
            approved_tools=self.approved_tools.union(tool_names),
            require_approval_tools=self.require_approval_tools.difference(tool_names),
            auto_approve_skills=self.auto_approve_skills,
        )

    def with_required_approval(self, *tool_names: str) -> "ToolApprovalPolicy":
        """Return a copy that always prompts for the given tools."""
        return ToolApprovalPolicy(
            auto_approve_all=self.auto_approve_all,
            approved_tools=self.approved_tools.difference(tool_names),
            require_approval_tools=self.require_approval_tools.union(tool_names),
            auto_approve_skills=self.auto_approve_skills,
        )

    def approve_everything(self) -> "ToolApprovalPolicy":
        """Return a copy that auto-approves every tool and skill call."""
        return ToolApprovalPolicy(
            auto_approve_all=True,
            approved_tools=self.approved_tools,
            require_approval_tools=frozenset(),
            auto_approve_skills=True,
        )


class OpenAIAgentBuilder:
    def __init__(self) -> None:
        self._base_url: str|None = None
        self._model: str|None = None
        self._tools: list[Callable[..., object]] = []
        self._approval_policy: ToolApprovalPolicy = ToolApprovalPolicy()
        self._api_key: str|None = None
        self._system_prompt: str|None = None
        self._skill_toolset: SkillsToolset|None = None

    def with_tool_approval(self, policy: ToolApprovalPolicy) -> "OpenAIAgentBuilder":
        """
        This method sets the approval policy used to decide, per tool, whether a
        call must be approved by the user before it runs. See
        `ToolApprovalPolicy` for the resolution rules.
        """
        self._approval_policy = policy
        return self

    def without_skill_approval(self) -> "OpenAIAgentBuilder":
        """
        This method configures the builder to not require approval for using skills. By default, the builder is set to require approval for both tools and skills. Calling this method will set the internal flag to indicate that skill usage does not require approval.
        """
        current = self._approval_policy
        self._approval_policy = ToolApprovalPolicy(
            auto_approve_all=current.auto_approve_all,
            approved_tools=current.approved_tools,
            require_approval_tools=current.require_approval_tools,
            auto_approve_skills=True,
        )
        return self

    def without_tool_approval(self, *tool_names: str) -> "OpenAIAgentBuilder":
        """
        This method configures the builder to not require approval for using tools.

        With no arguments every tool call is auto-approved. When one or more tool
        names are given, only those tools are auto-approved and any configured
        approval requirement for them is dropped.
        """
        if tool_names:
            self._approval_policy = self._approval_policy.with_approved(*tool_names)
        else:
            self._approval_policy = self._approval_policy.approve_everything()
        return self

    def with_tool_approval_required(self, *tool_names: str) -> "OpenAIAgentBuilder":
        """
        This method configures the builder to always require approval for the
        named tools, even when other tools are auto-approved. Calling it with no
        arguments leaves the current policy untouched.
        """
        if tool_names:
            self._approval_policy = self._approval_policy.with_required_approval(*tool_names)
        return self

    def with_url(self, url: str) -> "OpenAIAgentBuilder":
        """
        This method sets the base URL for the OpenAI API. The `url` parameter
        should be a string representing the base URL of the API endpoint you
        want to use. If not set, it will default to the standard OpenAI API
        endpoint.
        """
        self._base_url = url
        return self

    def with_api_key(self, api_key: str) -> "OpenAIAgentBuilder":
        """
        This method sets the API key to be used for authentication with the
        OpenAI API. The `api_key` parameter should be a string containing your
        OpenAI API key.
        """
        self._api_key = api_key
        return self

    def using_model(self, model: str) -> "OpenAIAgentBuilder":
        """
        This method sets the language model to be used by the agent. The
        `model` parameter should be a string representing the name of the
        model, such as "gpt-3.5-turbo" or "gpt-4".
        """
        self._model = model
        return self

    def with_tools(self) -> "OpenAIAgentBuilder":
        """
        This method adds a predefined set of tools to the agent. The tools
        included are:
        - `bash`: A tool for executing bash commands.
        - `deepseek_search`: A tool for web search using the DeepSeek Search API.
        - `deepseek_fetch`: A tool for fetching a web page as Markdown using the
          DeepSeek Fetch API.
        - `edit_file`: A tool for editing files.
        - `read_file`: A tool for reading files.
        """

        self._tools = [bash, deepseek_search, deepseek_fetch, edit_file, read_file]
        return self

    def maybe_with_skills(self, directories: list[str|Path|SkillsDirectory]|None=None) -> "OpenAIAgentBuilder":
        """
        This method conditionally adds skills to the agent if any skill
        directories are found. If the `directories` parameter is not provided,
        it will automatically search for ".skills" directories in the current
        working directory, the user's home directory, and the directory where
        this script is located. If any skill directories are found, it will
        call the `with_skills()` method to add them to the agent. If no skill
        directories are found, it will simply return the builder instance
        without adding any skills.
        """
        if directories is None:
            directories = self.__locate_skill_directories()

        if directories: 
            return self.with_skills(directories)
        return self

    def with_skills(self, directories: list[str|Path|SkillsDirectory]|None) -> "OpenAIAgentBuilder":
        """
        This method allows you to specify a list of directories containing
        skills to be used by the agent. If no directories are provided, it will
        automatically search for ".skills" directories in the current working
        directory, the user's home directory, and the directory where this
        script is located.
        """
        if directories is None:
            directories = self.__locate_skill_directories()

        if len(directories) == 0:
            raise ValueError("No skill directories found. Please provide at least one directory containing skills.")

        self._skill_toolset = SkillsToolset(directories=directories)
        return self


    def __locate_skill_directories(self) -> list[str|Path|SkillsDirectory]:
        """
        This method looks for ".skills" directories in the current working
        directory, the user's home directory, and the directory where this
        script is located. It returns a list of paths to any found ".skills"
        directories.
        """
        directories: list[str | Path | SkillsDirectory] = []
        # Check to see if there is a ".skills" directory in the current working directory, if so, add it to the directories list - just do this one task and stop.
        cwd_skills = Path.cwd() / ".skills"
        if cwd_skills.exists() and cwd_skills.is_dir():
            directories.append(str(cwd_skills))


        home_skills = Path.home() / ".skills"
        if home_skills.exists() and home_skills.is_dir():
            directories.append(str(home_skills))

        script_skills = Path(__file__).resolve().parent / ".skills"
        if script_skills.exists() and script_skills.is_dir():
            directories.append(str(script_skills))

        return directories

    def with_system_prompt(self, system_prompt: str) -> "OpenAIAgentBuilder":

        """
        This method sets the system prompt for the agent. The `system_prompt`
        parameter should be a string containing the instructions or context that
        you want to provide to the agent before it starts processing user input.
        This can help guide the agent's behavior and responses.
        """
        self._system_prompt = system_prompt
        return self

    def build(self) -> Agent:
        """
        Builds and returns an Agent instance based on the provided configuration.
        """
        provider = OpenAIProvider(
            api_key=self._api_key,
            base_url=self._base_url
        )

        if self._model is None:
            raise ValueError("Model name must be specified using the using_model() method.")



        model = OpenAIChatModel(model_name=self._model, provider=provider)

        toolsets: list[AbstractToolset[None]] = []


        if self._skill_toolset is not None:
            skill_toolset = self._skill_toolset
            if not self._approval_policy.auto_approve_skills:
                skill_toolset = self._skill_toolset.approval_required()

            toolsets.append(skill_toolset)

        if self._tools:
            builtin_toolset = FunctionToolset()
            for tool in self._tools:
                tool_name = getattr(tool, "__name__", "")
                builtin_toolset.add_function(
                    tool,
                    requires_approval=self._approval_policy.requires_approval(tool_name),
                )
            toolsets.append(builtin_toolset)


        if self._system_prompt is None:
            self._system_prompt = "You are a helpful assistant. Always try to use tools when appropriate, and be sure to follow the instructions provided by the user."

        agent = Agent(model, toolsets=toolsets, system_prompt=self._system_prompt)
        return agent
