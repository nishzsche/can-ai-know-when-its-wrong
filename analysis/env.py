"""Load API keys from a git-ignored `.env` file in the repo root, without overriding real env vars.

`.env` holds lines like `OPENROUTER_API_KEY=sk-or-v1-...`. It's listed in .gitignore; never commit
it, and never print the values.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path = ENV_FILE) -> list[str]:
    """Set variables from `path` that aren't already set. Returns the names it set (not values)."""
    if not path.exists():
        return []
    set_names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            set_names.append(key)
    return set_names
