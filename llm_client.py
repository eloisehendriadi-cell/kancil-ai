# llm_client.py
from __future__ import annotations
import os
import requests
from typing import List, Dict, Optional
# Chat history management
import uuid
from chat_history import save_chat, load_chat, rename_chat, list_chats

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE = os.getenv("GROQ_BASE", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

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
    Return a single assistant message string using Groq Chat Completions.
    Also saves the chat history for persistence.
    """
    if not GROQ_API_KEY:
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
    body = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(f"{GROQ_BASE}/chat/completions", json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        assistant_message = (data["choices"][0]["message"]["content"] or "").strip()
        # Save chat history
        if chat_id is None:
            chat_id = str(uuid.uuid4())
        if chat_name is None:
            chat_name = chat_id
        # Append assistant message
        messages.append({"role": "assistant", "content": assistant_message})
        save_chat(chat_id, chat_name, messages)
        return assistant_message
    except Exception:
        return ""

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

