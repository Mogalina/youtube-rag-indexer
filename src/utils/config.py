import yaml
from pathlib import Path
from functools import lru_cache


# Project root directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load the configuration file from the project root.
    Uses lru_cache so it is only loaded from disk once per process.

    Args:
        config_path: Path to config file relative to `_PROJECT_ROOT`

    Returns:
        Parsed configuration dictionary
    """
    config_file = _PROJECT_ROOT / config_path
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
