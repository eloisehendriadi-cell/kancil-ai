# llm_config.py - Centralized LLM Configuration
# Use environment variables to select which LLM provider to use
from __future__ import annotations

import os
from typing import Literal

# Load .env file if it exists (no external dependencies needed)
def _load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    # Parse KEY=VALUE
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # Only set if not already set via environment
                        if key and not os.getenv(key):
                            os.environ[key] = value
        except Exception as e:
            print(f"[LLM Config] Warning: Could not load .env file: {e}")

_load_env_file()

# Determine which LLM provider to use (default to groq if not set)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()

# Common settings
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

# Default model names per provider
DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openrouter": "meta-llama/llama-3.1-8b-instruct",
    "ollama": "llama2",
    "openai": "gpt-3.5-turbo",
}

# Default base URLs per provider
DEFAULT_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://127.0.0.1:11434",
    "openai": "https://api.openai.com/v1",
}

# Validate provider exists (warn if not recognized)
if LLM_PROVIDER not in DEFAULT_MODELS:
    # Don't crash - just use groq as fallback and warn
    print(f"[LLM Config] WARNING: Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Using 'groq' as default.")
    LLM_PROVIDER = "groq"

# Set final values
MODEL = LLM_MODEL or DEFAULT_MODELS.get(LLM_PROVIDER, "")
BASE_URL = LLM_BASE_URL or DEFAULT_URLS.get(LLM_PROVIDER, "")
API_KEY = LLM_API_KEY


def get_llm_config() -> dict:
    """Return the complete LLM configuration."""
    return {
        "provider": LLM_PROVIDER,
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "model": MODEL,
    }


def validate_config() -> bool:
    """
    Validate configuration before making LLM calls.
    Returns True if valid, False otherwise.
    """
    if not LLM_PROVIDER:
        return False
    if LLM_PROVIDER != "ollama" and not API_KEY:
        return False
    if not MODEL:
        return False
    return True


# Display configuration on import (for debugging)
if os.getenv("DEBUG"):
    config = get_llm_config()
    print(f"[LLM Config] Provider: {config['provider']}, Model: {config['model']}, URL: {config['base_url']}")
