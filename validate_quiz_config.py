#!/usr/bin/env python3
"""
Quick validation script to verify quiz API configuration.
Run this before starting the app to catch configuration issues early.
"""

import os
import sys

def validate_config():
    """Validate LLM configuration."""
    print("=" * 60)
    print("🔍 Quiz API Configuration Validator")
    print("=" * 60)
    
    # Import after checking for .env
    from llm_config import LLM_PROVIDER, API_KEY, BASE_URL, MODEL
    
    checks = [
        ("LLM Provider", LLM_PROVIDER, lambda x: x in ("groq", "openrouter", "openai", "ollama")),
        ("API Key Present", bool(API_KEY) or LLM_PROVIDER == "ollama", lambda x: x),
        ("Model Configured", bool(MODEL), lambda x: x),
        ("Base URL Configured", bool(BASE_URL), lambda x: x),
    ]
    
    all_ok = True
    for name, value, validator in checks:
        status = "✅" if validator(value) else "❌"
        display_value = "***" if name == "API Key Present" and API_KEY else value
        print(f"{status} {name:20s}: {display_value}")
        if not validator(value):
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("✨ Configuration looks good!")
        print("\nYou can now start the app:")
        print("  python3 app.py")
        print("\nThen visit: http://127.0.0.1:5050/workspace")
    else:
        print("⚠️  Configuration incomplete. Please check:")
        print("  1. .env file exists in the app directory")
        print("  2. LLM_PROVIDER is set to a valid option")
        print("  3. LLM_API_KEY is set (unless using ollama)")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        validate_config()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
