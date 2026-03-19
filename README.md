# Magi

Magi is a small CLI for running a configurable LLM agent against your local project. It supports:

- interactive chat in the terminal
- one-shot prompts from `-q` or stdin
- persisted session history
- approval-gated local tools
- optional skill loading from `.skills/`
- a watch mode that reacts to `AI!` comments in files

## Requirements

- Python 3.12+
- `uv`
- at least one configured model API key

## Install

```bash
uv sync
```

This project defines a console script named `magi`. You can run it through `uv`, through the checked-in launcher script, or through `just`.

## Configuration

Magi loads environment variables from `.env` and YAML config from `magi.yaml`.

The default repo config already defines these model entries:

- `deepseek-reasoner` via `DEEPSEEK_API_KEY`
- `deepseek-chat` via `DEEPSEEK_API_KEY`
- `glm-5` via `OPENROUTER_API_KEY`

Example `.env`:

```dotenv
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
```

YAML values can reference environment variables with `${VAR_NAME}`.

Config discovery order:

1. `./magi.yaml`
2. `~/.config/magi/magi.yaml`
3. bundled `magi.yaml` files found with the package

Later files override earlier keys.

Key config fields:

- `default_model`: model alias used when `-m/--model` is omitted
- `default_system_prompt`: default key inside `system_prompts`
- `system_prompts`: named prompts selectable with `-p/--system-prompt`
- `session`: default session name stored under `.sessions/`
- `auto_approve`: present in config, but current CLI behavior is driven by `-y/--auto-approve`
- `models`: map of model aliases to `base_url`, `api_key`, and upstream `model`

## Usage

Show help:

```bash
uv run magi --help
```

Run interactively:

```bash
uv run magi
```

Run a one-shot prompt:

```bash
uv run magi -q "Summarize the architecture of this repository."
```

Pipe a prompt over stdin:

```bash
echo "Review this project structure." | uv run magi
```

Use a specific model and system prompt:

```bash
uv run magi -m deepseek-chat -p code_review -q "Review the latest changes."
```

Run without loading or saving session history:

```bash
uv run magi -S -q "Ephemeral run"
```

Auto-approve tool and skill calls:

```bash
uv run magi -y
```

Alternative entrypoints:

```bash
./magi -q "Hello"
just run -- -q "Hello"
```

## Sessions

Session history is stored in `.sessions/<session>.json`.

Use a named session:

```bash
uv run magi -s main
uv run magi -s code-review
```

Available in-chat slash commands:

- `/help`
- `/models`
- `/prompts`
- `/history`
- `/save`
- `/load`
- `/clear`

## Watch Mode

Watch mode scans the current project for non-hidden, non-ignored files. When a changed file contains a comment line ending with `AI!`, Magi sends that file to the agent using the `watch` system prompt.

Start watch mode:

```bash
uv run magi -W
```

Example trigger:

```python
# Refactor this function to avoid duplicate API calls AI!
```

Notes:

- files under hidden directories like `.sessions/` are skipped
- `.gitignore` is respected
- the watcher polls once per second

## Built-In Tools

The agent is wired with three built-in tools:

- `bash`: run a shell command
- `read_file`: read a file from the current project
- `edit_file`: apply search/replace style edits to a file

By default, tool calls require approval. Pass `-y` to disable approval prompts for the current run.

If a `.skills/` directory exists in the project root, Magi also loads skills from there. It also checks `~/.skills`.

## Development

Run the type checker:

```bash
uv run basedpyright
```

Convenience commands:

```bash
just run -- -q "Hello"
just lint
```

## Project Layout

```text
src/magi/__init__.py       CLI entrypoint and dependency wiring
src/magi/arguments.py      argument parsing
src/magi/config.py         YAML config loading and env expansion
src/magi/repl.py           interactive loop, slash commands, watch mode
src/magi/session.py        session persistence
src/magi/tools.py          built-in tool implementations
src/magi/agent_builder.py  model/tool/skill assembly
```

## Current Limitations

- The CLI parser exposes a positional `filename` argument, but the current entrypoint does not read prompt text from that file. Use `-q` or stdin instead.
- `--clear-session` appears in `--help`, but it is not currently handled in `main()`.
