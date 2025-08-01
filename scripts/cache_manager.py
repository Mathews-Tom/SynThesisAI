#!/usr/bin/env python3
"""Cache Manager Script.

Convenience script to run the cache and performance management CLI.
"""

# Standard Library
import sys
from pathlib import Path

# SynthesisAI Modules
from core.cli.cache_manager_cli import main


def _bootstrap() -> int:
    """Bootstrap the cache manager CLI.

    Returns:
        Exit code from the cache manager CLI.
    """
    project_root: Path = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    return main()


if __name__ == "__main__":
    sys.exit(_bootstrap())
