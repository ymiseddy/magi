import os
from pathlib import Path
import re
from typing import cast

import yaml
from yaml import SafeLoader
from yaml.nodes import ScalarNode

# Regex to find placeholders like ${MY_VAR} or ${MY_VAR:-default}
pattern = re.compile(r'.*?\$\{([\w]+)(?::-(.*?))?\}.*?')

def env_var_constructor(loader: SafeLoader, node: ScalarNode) -> str:
    value = loader.construct_scalar(node)
    # Using expandvars is the easiest way to handle ${VAR} syntax
    return os.path.expandvars(value)

# Register the constructor with a tag like !env
yaml.SafeLoader.add_constructor('!env', env_var_constructor)

# Optional: Add a resolver so you don't have to use the !env tag manually
yaml.SafeLoader.add_implicit_resolver('!env', pattern, None)  # pyright: ignore[reportUnknownMemberType]


def _discover_config_files() -> list[str]:
    """
    This function searches for YAML configuration files in the current working
    directory, the user's home directory, and the directory where this script is
    located. It returns a list of paths to any found YAML configuration files.
    """

    base_name = "magi.yaml"
    config_files: list[str] = []
    # Check for YAML files in the current working directory
    cwd = Path.cwd()
    for file in cwd.glob(base_name):
        config_files.append(str(file))

    # Check for YAML file in ~/.config/magi/magi.yaml
    config_dir = Path.home() / ".config" / "magi"
    if config_dir.exists() and config_dir.is_dir():
        for file in config_dir.glob(base_name):
            config_files.append(str(file))

    # Check for YAML files in the script's directory
    script_dir = Path(__file__).resolve().parent
    for file in script_dir.glob(base_name):
        config_files.append(str(file))
   
    return config_files


def _merge_configs(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_configs(
                {k: v for k, v in cast(dict[object, object], current).items() if isinstance(k, str)},
                {k: v for k, v in cast(dict[object, object], value).items() if isinstance(k, str)},
            )
        else:
            merged[key] = value
    return merged


def load_config(files: list[str] | None = None) -> dict[str, object]:
    if files is None:
        files = _discover_config_files()
    config: dict[str, object] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = cast(object, yaml.safe_load(f))
                if isinstance(data, dict):
                    typed_data = cast(dict[object, object], data)
                    config = _merge_configs(
                        config,
                        {key: value for key, value in typed_data.items() if isinstance(key, str)},
                    )
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {file}: {e}")
    return config
