# llm_client.py
from __future__ import annotations
import os
import requests
from typing import List, Dict, Optional, Iterator
# Chat history management
import uuid
from chat_history import save_chat, load_chat, rename_chat, list_chats
from llm_config import LLM_PROVIDER, API_KEY, BASE_URL, MODEL
from openai import OpenAI

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
    """Call OpenAI API using the OpenAI package."""
    try:
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout=timeout,
        )
        assistant_message = (response.choices[0].message.content or "").strip()
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


def chat_text_stream(
    prompt_or_messages: str | List[Dict[str, str]],
    *,
    chat_id: Optional[str] = None,
    chat_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
    timeout: int = 60,
) -> Iterator[str]:
    """
    Stream assistant response tokens using the configured LLM.
    Yields tokens as they arrive. Saves complete message at the end.
    """
    if not API_KEY and LLM_PROVIDER != "ollama":
        yield ""
        return
    
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in (prompt_or_messages or [])
            if (m.get("content") or "").strip()
        ]
        if not messages:
            yield ""
            return
    
    model_to_use = model or MODEL
    
    # Stream based on provider
    if LLM_PROVIDER == "openai":
        yield from _stream_openai(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    elif LLM_PROVIDER in ["groq", "openrouter"]:
        yield from _stream_openai_compatible(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    elif LLM_PROVIDER == "ollama":
        yield from _stream_ollama(messages, model_to_use, temperature, max_tokens, timeout, chat_id, chat_name)
    else:
        yield ""


def _stream_openai(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Stream from OpenAI API."""
    try:
        client = OpenAI(api_key=API_KEY)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout=timeout,
            stream=True,
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token
        
        # Save after streaming completes
        _save_chat_history(messages, full_response, chat_id, chat_name)
    except Exception as e:
        yield f"[Error: {str(e)}]"


def _stream_openai_compatible(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Stream from OpenAI-compatible APIs (Groq, OpenRouter)."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    if LLM_PROVIDER == "openrouter":
        headers["HTTP-Referer"] = os.getenv("SITE_URL", "https://example.com")
        headers["X-Title"] = os.getenv("SITE_NAME", "Kancil AI")
    
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout,
            stream=True
        )
        r.raise_for_status()
        
        full_response = ""
        for line in r.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                token = delta['content']
                                full_response += token
                                yield token
                    except:
                        continue
        
        # Save after streaming completes
        _save_chat_history(messages, full_response, chat_id, chat_name)
    except Exception as e:
        yield f"[Error: {str(e)}]"


def _stream_ollama(messages, model, temperature, max_tokens, timeout, chat_id, chat_name):
    """Stream from Ollama API."""
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    
    try:
        r = requests.post(
            f"{BASE_URL}/api/chat",
            json=body,
            timeout=timeout,
            stream=True
        )
        r.raise_for_status()
        
        full_response = ""
        for line in r.iter_lines():
            if line:
                try:
                    import json
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        token = data['message']['content']
                        full_response += token
                        yield token
                except:
                    continue
        
        # Save after streaming completes
        _save_chat_history(messages, full_response, chat_id, chat_name)
    except Exception as e:
        yield f"[Error: {str(e)}]"

