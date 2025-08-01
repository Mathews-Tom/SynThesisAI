from typing import Optional


def get_input(prompt: str, default: Optional[str] = None) -> Optional[str]:
    """Prompt the user for input, optionally using a default value."""
    prompt_display = f"{prompt} [{default}]" if default is not None else prompt
    user_input = input(f"{prompt_display}: ").strip()
    if user_input:
        return user_input
    return default


def format_duration(seconds: float) -> str:
    """
    Convert duration in seconds to human-readable format (e.g., '2h 34m 43.25s').

    Example:
        3661.23 -> '1h 1m 1.23s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s"
    if minutes > 0:
        return f"{minutes}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"
