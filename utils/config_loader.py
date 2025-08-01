# Standard Library
from pathlib import Path
from typing import Any, Dict, Union

# Third-Party Library
from yaml import YAMLError

from utils.config_manager import get_config_manager

# SynThesisAI Modules
from utils.exceptions import ConfigError


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file using the centralized ConfigManager.

    This function is a wrapper around the ConfigManager to maintain
    backward compatibility while providing centralized configuration management.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Dictionary containing the loaded configuration

    Raises:
        ConfigError: If the configuration file is not found or invalid.
    """
    path = Path(config_path)
    config_manager = get_config_manager()
    try:
        config_manager.load_config(path)
    except FileNotFoundError as e:
        raise ConfigError("Configuration file not found", str(path)) from e
    except YAMLError as e:
        raise ConfigError("Invalid YAML configuration", str(path)) from e
    except Exception as e:
        raise ConfigError(str(e), str(path)) from e

    return config_manager.get_all()
