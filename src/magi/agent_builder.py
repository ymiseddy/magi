from pathlib import Path
from pydantic_ai import Agent, FunctionToolset, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_skills import SkillsDirectory, SkillsToolset

from magi import tools

class OpenAIAgentBuilder:
    def __init__(self) -> None:
        self._base_url: str|None = None
        self._model: str|None = None
        self._tools: list[Tool] = []
        self._api_key: str|None = None
        self._system_prompt: str|None = None
        self._skill_toolset: SkillsToolset|None = None


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
        - `tools.bash`: A tool for executing bash commands.
        - `tools.edit_file`: A tool for editing files.
        - `tools.read_file`: A tool for reading files.
        """

        self._tools = [tools.bash, tools.edit_file, tools.read_file]  # pyright: ignore[reportAttributeAccessIssue]
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
        directories: list[str | Path | SkillsDirectory] | None = []
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
        toolsets = [self._skill_toolset] if self._skill_toolset is not None else []

        if self._tools:
            builtin_toolset = FunctionToolset()
            for tool in self._tools:
                builtin_toolset.add_function(tool, requires_approval=True)
            toolsets.append(builtin_toolset)


        if self._system_prompt is None:
            self._system_prompt = "You are a helpful assistant. Always try to use tools when appropriate, and be sure to follow the instructions provided by the user."

        agent = Agent(model, toolsets=toolsets, system_prompt=self._system_prompt)  
        return agent


