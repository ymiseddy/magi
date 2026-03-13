import argparse

Arguments = list[str] | None

class CommandArguments(argparse.Namespace):
    """Command-line arguments."""
    session: str="main"
    watch: bool=False
    auto_approve: bool=False
    no_session: bool=False
    clear_session: bool=False
    filename: str|None=None
    query: str|None=None
    model: str|None=None

    def __init__(self, Arguments:Arguments=None):
        super().__init__()
        parser = argparse.ArgumentParser(
            description="Magi LLM agent.",
        )
        _ = parser.add_argument(
            "-S",
            "--session",
            default="main",
            help="Name of the session to use.",
        )

        _ = parser.add_argument(
            "-W",
            "--watch",
            action="store_true",
            help="Enable watch mode.",
        )
        _ = parser.add_argument(
            "-y",
            "--auto-approve",
            action="store_true",
            help="Automatically approve tool usage without prompting.",
        )
        _ = parser.add_argument(
            "--no-session",
            action="store_true",
            help="Disable loading and saving session history.",
        )
        _ = parser.add_argument(
            "--clear-session",
            action="store_true",
            help="Delete the stored history for the selected session/model and exit.",
        )

        _ = parser.add_argument(
            "-q",
            "--query",
            type=str,
            default=None,
            help="Query string (if not using filename).",
        )
        _ = parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Name of the model defined in models.yml (defaults to the file's default_model).",
        )

        _ = parser.add_argument(
            "filename",
            nargs="?",
            type=str,
            help="Filename containing query.",
        )

        _ = parser.parse_args(Arguments, namespace=self)


