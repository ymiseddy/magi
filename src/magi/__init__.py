import asyncio
import sys
from collections.abc import Coroutine
import dotenv


from .ioc import Dependencies
from .arguments import CommandArguments
from . import config


def _run_async(coro: Coroutine[object, object, None], isatty: bool) -> None:
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        if isatty:
            print(file=sys.stderr)


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
        _run_async(magi_session.watch(), isatty)
        return

    _run_async(magi_session.run(prompt, isatty), isatty)
