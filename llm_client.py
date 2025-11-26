# llm_client.py
from __future__ import annotations
import os
import requests
from typing import List, Dict, Optional
# Chat history management
import uuid
from chat_history import save_chat, load_chat, rename_chat, list_chats
from llm_config import LLM_PROVIDER, API_KEY, BASE_URL, MODEL

def chat_text(
    prompt_or_messages: str | List[Dict[str, str]],
    *,
    chat_id: Optional[str] = None,
    chat_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
    timeout: int = 60,
) -> str:
    """
    Return a single assistant message string using the configured LLM.
    Also saves the chat history for persistence.
    """
    if not API_KEY and LLM_PROVIDER != "ollama":
        return ""
    
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in (prompt_or_messages or [])
            if (m.get("content") or "").strip()
        ]
        if not messages:
            return ""
    
    model_to_use = model or MODEL
    
    # Make request based on provider
    if LLM_PROVIDER == "groq":
        return _call_groq(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    elif LLM_PROVIDER == "openrouter":
        return _call_openrouter(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    elif LLM_PROVIDER == "openai":
        return _call_openai(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    else:
        return ""


def _call_groq(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Call Groq API (OpenAI compatible)."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        assistant_message = (data["choices"][0]["message"]["content"] or "").strip()
        _save_chat_history(messages, assistant_message, chat_id, chat_name)
        return assistant_message
    except Exception:
        return ""


def _call_openrouter(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Call OpenRouter API (OpenAI compatible)."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": os.getenv("SITE_URL", "https://example.com"),
        "X-Title": os.getenv("SITE_NAME", "Kancil AI"),
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        assistant_message = (data["choices"][0]["message"]["content"] or "").strip()
        _save_chat_history(messages, assistant_message, chat_id, chat_name)
        return assistant_message
    except Exception:
        return ""


def _call_openai(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Call OpenAI API (OpenAI compatible)."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        assistant_message = (data["choices"][0]["message"]["content"] or "").strip()
        _save_chat_history(messages, assistant_message, chat_id, chat_name)
        return assistant_message
    except Exception:
        return ""


def _call_ollama(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Call Ollama API."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    try:
        r = requests.post(f"{BASE_URL}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        assistant_message = (data.get("message") or {}).get("content", "").strip()
        _save_chat_history(messages, assistant_message, chat_id, chat_name)
        return assistant_message
    except Exception:
        return ""


def _save_chat_history(messages, assistant_message, chat_id, chat_name):
    """Save chat history."""
    if chat_id is None:
        chat_id = str(uuid.uuid4())
    if chat_name is None:
        chat_name = chat_id
    messages.append({"role": "assistant", "content": assistant_message})
    save_chat(chat_id, chat_name, messages)

# Utility functions for chat sessions
def get_chat_sessions():
    """Return all chat sessions."""
    return list_chats()

def get_chat(chat_id: str):
    """Return a specific chat session."""
    return load_chat(chat_id)

def rename_chat_session(chat_id: str, new_name: str):
    """Rename a chat session."""
    rename_chat(chat_id, new_name)

