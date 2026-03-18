
set positional-arguments

run *args:
	uv run magi "$@"

lint:
	uv run basedpyright
