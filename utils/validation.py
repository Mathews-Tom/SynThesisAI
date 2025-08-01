from typing import Any, Mapping

from utils.exceptions import ValidationError


def assert_valid_model_config(role: str, config: Mapping[str, Any]) -> None:
    """
    Ensure that model config dict contains required fields.

    :param role: The role name for the config.
    :param config: Model configuration mapping.
    :raises ValidationError: If config is not a dict or missing required keys.
    """
    if not isinstance(config, dict):
        raise ValidationError(f"{role.capitalize()} config must be a dictionary.")

    required_fields = ("provider", "model_name")
    for field in required_fields:
        if not config.get(field):
            raise ValidationError(f"Missing '{field}' in {role} model config.", field=field)
