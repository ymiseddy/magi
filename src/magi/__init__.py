from abc import ABC, abstractmethod
from ast import Dict
import asyncio
import os
import sys
from typing import override
import dotenv

from pydantic_ai import Agent, AgentRunResultEvent, ModelMessage, ModelMessagesTypeAdapter, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta

from magi.session import FileSessionManager, NoOpSessionManager, SessionManager
from .io import ReaderWriter, OTYPE_RESULT, OTYPE_ERROR, OTYPE_THINKING, OTYPE_PROMPT
from .arguments import CommandArguments
from . import config
from . import agent_builder

def build_agent(args: CommandArguments, config: Dict) -> Agent:
    model_name: str = args.model or config.get("default_model", "deepseek-reasoner")
    model = config["models"][model_name]

    system_prompt = "You are a helpful assistant that can perform tasks for the user." + \
        " You have access to tools and can use them to accomplish tasks." + \
        " Always try to use tools when appropriate, and be sure to follow" + \
        " the instructions provided by the user."

    agent = agent_builder.OpenAIAgentBuilder() \
        .with_url(model["base_url"]) \
        .with_api_key(model["api_key"]) \
        .with_system_prompt(system_prompt) \
        .using_model(model["model"])  \
        .build()
    return agent


class MagiProducer:
    def __init__(self, args: CommandArguments, config: Dict) -> None:
        self.args: CommandArguments = args
        self.config: Dict = config
        self.agent: Agent = build_agent(args, config)
        self.io: ReaderWriter = ReaderWriter.console()
        self.session_manager: SessionManager

        if args.no_session:
            self.session_manager = NoOpSessionManager()
        else:
            session_name = args.session or "default"
            self.session_manager = FileSessionManager(session_name)


    async def run(self) -> None:

        session_history = self.session_manager.load()


        while True:
            self.io.write(OTYPE_PROMPT, "> ")
            prompt = self.io.read()
            if prompt is None:
                break

            self.io.write(OTYPE_PROMPT, f"\n")

            event_stream = self.agent.run_stream_events(prompt, message_history=session_history)
            mode = None
            async for event in event_stream:
                if isinstance(event, PartStartEvent):
                    if hasattr(event.part, 'content'):
                        if isinstance(event.part, ThinkingPart):
                            if mode != "thinking":
                                self.io.write(OTYPE_THINKING, "\n\n## Thinking...\n\n")
                                mode = "thinking"
                            self.io.write(OTYPE_THINKING, event.part.content)
                        elif isinstance(event.part, TextPart):
                            if mode != "text":
                                self.io.write(OTYPE_RESULT, "\n\n## Text Response:\n\n")
                                mode = "text"
                            self.io.write(OTYPE_RESULT, event.part.content)
                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, ThinkingPartDelta):
                        if mode != "thinking":
                            self.io.write(OTYPE_THINKING, "\n\n## Thinking...\n\n")
                            mode = "thinking"
                        self.io.write(OTYPE_THINKING, event.delta.content_delta)
                    elif isinstance(event.delta, TextPartDelta):
                        if mode != "text":
                            self.io.write(OTYPE_RESULT, "\n\n## Text Response:\n\n")
                            mode = "text"
                        self.io.write(OTYPE_RESULT, event.delta.content_delta)
                    else:
                        self.io.write(OTYPE_RESULT, f"delta={event.delta}")
                        self.io.write(OTYPE_RESULT, f"delta type={type(event.delta)}")
                elif isinstance(event, AgentRunResultEvent):
                    session_history = list(event.result.all_messages())
                else:
                    pass
            self.io.write(OTYPE_PROMPT, f"\n")

        self.session_manager.save(session_history)
        await asyncio.sleep(.5)


def main() -> None:
    _ = dotenv.load_dotenv()
    args = CommandArguments(sys.argv[1:]) 
    cfg = config.load_config()

    producer = MagiProducer(args, cfg)
    asyncio.run(producer.run())


