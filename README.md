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
- `tool_approval`: per-tool approval rules (see [Tool Approval](#tool-approval))
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

## Tool Approval

Every tool and skill call is confirmed by the user before it runs. The
`tool_approval` section of `magi.yaml` lets you auto-approve individual tools so
harmless ones (reads, searches) run without prompting while destructive ones
(`bash`, `edit_file`) stay gated:

```yaml
tool_approval:
  # Approve every tool call without prompting.
  auto_approve: false

  # Tools that never require approval.
  approved_tools:
    - read_file
    - deepseek_search
    - deepseek_fetch

  # Tools that always prompt, even when `auto_approve` is true.
  require_approval:
    - bash
    - edit_file

  # Load skills without prompting (skills are approval-gated by default).
  auto_approve_skills: false
```

Precedence for a single tool, highest first:

1. listed under `require_approval` — always prompt
2. listed under `approved_tools` — never prompt
3. safe by construction — never prompt (`bash_readonly`, see
   [Read-Only Shell](#read-only-shell))
4. `auto_approve: true` — never prompt, for all tools
5. otherwise — prompt

Tool names match the names shown by `/tools`: `bash`, `bash_readonly`,
`read_file`, `edit_file`, `deepseek_search`, `deepseek_fetch`, plus any skill
tool such as `load_skill` or `run_skill_script`.

Two things override the config:

- `-y/--auto-approve` on the command line approves every tool and skill call
  for that run, ignoring `require_approval`.
- A legacy top-level `auto_approve: true` key behaves like
  `tool_approval.auto_approve: true`. Prefer the `tool_approval` section.

## Read-Only Shell

`bash_readonly` is for inspection: reading files, searching, listing, and
read-only `git`. It never asks for approval because it cannot change anything.

The catch with "is this bash command safe?" is that any check on the string is
a guess about how a *second* parser (`sh -c`) will read it. `shlex.split`
turns `echo hi; rm -rf /` into `["echo", "hi;", ...]` and `c\\at f` into
`["cat", "f"]`, so an `argv[0]` check approves exactly the wrong things. So
`bash_readonly` does not parse-and-hope: it tokenises the command itself,
validates it, and then executes an argv list with `shell=False`. No shell ever
sees the input, so what runs is what was checked.

Accepted:

- One pipeline of up to 5 stages joined with `|`.
- Read-only programs: `cat`, `head`, `tail`, `wc`, `ls`, `tree`, `stat`,
  `file`, `du`, `df`, `nl`, `cut`, `tr`, `sort`, `uniq`, `grep`, `rg`, `find`,
  `diff`, `jq`, checksums, `od`, `strings`, `column`, `date`, `uname`,
  `which`, `pwd`, `echo`, and read-only `git` (`log`, `show`, `diff`,
  `status`, `blame`, `grep`, `ls-files`, `rev-parse`, `describe`, ...).
- Paths inside the project. Absolute paths, `..` escapes, `~` and symlinks
  pointing out of the project are refused, so a "harmless" `cat` cannot read
  `~/.ssh/id_rsa`.

Refused, with a message explaining the alternative: `;`, `&&`, `||`,
redirects, `$(...)`, backticks, `$VAR`, unquoted globs, `cd`, environment
prefixes (`FOO=bar cmd`), anything not on the allowlist (`rm`, `sed -i`,
`xargs`, `curl`, `python`, `bash`, ...), flag forms that write or exec (`sort
-o`, `find -delete`/`-exec`, `rg --pre`, `uniq FILE OUT`, `tail -f`,
`git -c`), and `git` subcommands that change the repository.

Quotes are literal and passed straight to the program, which is what makes
`rg -g '*.py' needle` work: the glob reaches ripgrep untouched.

## Sandbox

`bash` can additionally run inside [bubblewrap](https://github.com/containers/bubblewrap):
the project directory is writable, everything else is read-only, and `$HOME`
points at a throwaway directory. `bash_readonly` always uses the sandbox when
it is available, without network access, as a second line of defence behind
the allowlist.

```yaml
sandbox:
  mode: auto      # auto | require | off
  network: true   # network for `bash`; `bash_readonly` never gets it
  writable_paths: []   # extra directories to bind read-write
```

`auto` sandboxes when `bwrap` is present and runs unsandboxed otherwise,
`require` refuses to run `bash` at all without it, and `off` disables the
sandbox everywhere (the `bash_readonly` allowlist still applies).

The sandbox limits blast radius; it is not a permission system. Approved
commands still execute for real, so `mode: off` restores the previous
behaviour exactly.

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
- `/tools`
- `/history`
- `/save`
- `/load`
- `/export [path]`
- `/clear`

`/export` writes the text of the most recent agent response to a Markdown
file. Pass a path as the argument (`.md` is appended when no extension is
given); it defaults to `response.md` in the current directory.

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

The agent is wired with these built-in tools:

- `bash`: run a shell command
- `bash_readonly`: run an allowlisted read-only command without a shell and
  without approval (see [Read-Only Shell](#read-only-shell))
- `read_file`: read a file from the current project
- `edit_file`: apply search/replace style edits to a file. Patch markers are parsed
  leniently, so any run of three or more `<`, `=` or `>` characters works (labels are
  optional and case-insensitive), which keeps smaller models like Qwen from failing
  on an off-by-one marker width
- `deepseek_search`: web search through the DeepSeek Search API
- `deepseek_fetch`: fetch a web page as Markdown through the DeepSeek Fetch API

By default, tool calls require approval. Auto-approve individual tools with the
`tool_approval` section of `magi.yaml` (see [Tool Approval](#tool-approval)), or pass
`-y` to disable approval prompts for the current run.

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
