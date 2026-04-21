"""
Configuration loader for SherpaConnect.

The application reads a JSON file (``config.json`` by default) that maps
language codes to model configurations for STT, TTS, and translation.

See ``config.example.json`` for a fully annotated example.
"""

from __future__ import annotations

import json
import os

# Minimal default configuration – works if the user runs with --no-tts
# and passes --input-lang / --output-lang that are the same (no translation).
_DEFAULTS: dict = {
    "stt": {},
    "tts": {},
    "translation": {
        "engine": "argos",
    },
}


def load_config(path: str) -> dict:
    """
    Load and return the configuration dict from *path*.

    If *path* does not exist, the built-in defaults are returned so that
    the application can still be used for testing without a config file.
    """
    if not os.path.isfile(path):
        return dict(_DEFAULTS)

    with open(path, "r", encoding="utf-8") as fh:
        user_config: dict = json.load(fh)

    # Shallow-merge defaults so that missing top-level keys are filled in
    merged = dict(_DEFAULTS)
    merged.update(user_config)

    # Deep-merge translation sub-config
    if "translation" in user_config:
        merged["translation"] = {**_DEFAULTS["translation"], **user_config["translation"]}

    return merged
