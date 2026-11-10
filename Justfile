
set positional-arguments

run *args:
	uv run magi "$@"

lint:
	uv run basedpyright

test:
	uv run python -m unittest discover -s tests -t . -v
