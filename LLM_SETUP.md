# LLM Configuration Guide

## Overview

The application now uses a **unified LLM configuration system** that supports multiple providers. You control everything through environment variables.

## Setup

1. **Copy the example configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and set your LLM provider:**
   ```bash
   LLM_PROVIDER=groq
   LLM_API_KEY=your-api-key-here
   ```

3. **That's it!** The application will automatically:
   - Detect your provider
   - Use the correct API endpoint
   - Apply appropriate model defaults
   - Configure all features (chat, quiz, notes, podcast, flashcards)

## Supported Providers

### Groq (Recommended - Free tier available)
```bash
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...  # Get from https://console.groq.com
LLM_MODEL=llama-3.1-8b-instant  # Optional - this is the default
```

### OpenRouter (Access to many models)
```bash
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-...  # Get from https://openrouter.ai
LLM_MODEL=meta-llama/llama-3.1-8b-instruct  # Optional - this is the default
```

### OpenAI (ChatGPT)
```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-...  # Get from https://platform.openai.com
LLM_MODEL=gpt-3.5-turbo  # Optional - this is the default
```

### Ollama (Local - Free, no API key needed)
```bash
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434  # Optional - this is the default
LLM_MODEL=llama2  # Optional - this is the default
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | - | Provider: `groq`, `openrouter`, `openai`, or `ollama` |
| `LLM_API_KEY` | If not Ollama | - | API key for your provider |
| `LLM_BASE_URL` | No | Provider-specific | Override the API endpoint URL |
| `LLM_MODEL` | No | Provider-specific | Override the model name |
| `FLASK_SECRET_KEY` | No | `dev-secret` | Flask session encryption key |
| `DEBUG` | No | `false` | Enable debug mode |
| `PORT` | No | `5050` | Server port |
| `SITE_URL` | No | `https://example.com` | Your site URL (for OpenRouter) |
| `SITE_NAME` | No | `Kancil AI` | Your app name (for OpenRouter) |

## What Gets Unified

All these features now use the same LLM provider and model:

- **Chat** - Chatbot conversations
- **Quiz** - Multiple choice question generation
- **Notes** - Study note summarization
- **Podcast** - Educational podcast scripts
- **Flashcards** - Question/answer pair generation

## Migration from Old Config

If you were previously using multiple providers with fallbacks, the old environment variables are **no longer used**:

| Old Variable | New Equivalent |
|--------------|----------------|
| `GROQ_API_KEY` | `LLM_API_KEY` |
| `GROQ_BASE` | `LLM_BASE_URL` |
| `GROQ_MODEL` | `LLM_MODEL` |
| `OPENROUTER_API_KEY` | Use `LLM_API_KEY` with `LLM_PROVIDER=openrouter` |
| `OPENROUTER_BASE` | Use `LLM_BASE_URL` |
| `OLLAMA_HOST` | Use `LLM_BASE_URL` with `LLM_PROVIDER=ollama` |
| `MODEL_NAME` | `LLM_MODEL` |

## Testing Your Setup

```bash
# This will validate your configuration on startup
python app.py
```

If configured correctly, you'll see:
```
[LLM Config] Provider: groq, Model: llama-3.1-8b-instant, URL: https://api.groq.com/openai/v1
```

## Troubleshooting

**Error: "Invalid LLM_PROVIDER"**
- Make sure `LLM_PROVIDER` is one of: `groq`, `openrouter`, `openai`, `ollama`

**Error: "LLM_API_KEY environment variable is required"**
- You're using a provider that needs an API key (not Ollama)
- Set `LLM_API_KEY` in your `.env` file

**Model requests failing**
- Check that your `LLM_API_KEY` is valid
- Verify `LLM_PROVIDER` matches your API key's provider
- For Ollama, ensure it's running on `LLM_BASE_URL`

## Code Changes

The configuration is now centralized in `llm_config.py`. Key modules updated:

- `llm_config.py` - New central configuration module
- `llm_client.py` - Uses `llm_config`, removed Groq hardcoding
- `workspace.py` - Uses `llm_config`, simplified `_llm_complete()`
- `quiz_app.py` - Uses `llm_config`, removed provider fallbacks
