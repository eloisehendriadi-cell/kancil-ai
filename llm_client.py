# the_ai_tutor/llm_client.py
from __future__ import annotations
import os, time, requests

def chat_complete(prompt: str, *, model: str | None = None, temperature: float = 0.2, num_predict: int = 600, timeout: int = 60) -> str:
    """
    If OPENROUTER_API_KEY is set -> use OpenRouter (hosted, public-ready)
    Else -> fall back to local Ollama (OLLAMA_HOST).
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        base = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
        use_model = model or os.getenv("QUIZ_MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("SITE_URL", "https://example.com"),
            "X-Title": os.getenv("SITE_NAME", "Kancil AI"),
            "Content-Type": "application/json",
        }
        body = {
            "model": use_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        r = requests.post(f"{base}/chat/completions", json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    # ---- Fallback to Ollama if no OPENROUTER_API_KEY present ----
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    use_model = model or os.getenv("MODEL_NAME", "llama3.2:3b")
    opts = {"temperature": temperature, "num_predict": num_predict}

    # try /api/chat then /api/generate
    try:
        r = requests.post(f"{host}/api/chat",
            json={"model": use_model, "messages":[{"role":"user","content":prompt}], "options": opts, "stream": False},
            timeout=timeout)
        if r.status_code != 404:
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content","").strip()
    except Exception:
        pass

    tries, backoff = 0, 1.6
    while tries < 3:
        try:
            r2 = requests.post(f"{host}/api/generate",
                json={"model": use_model, "prompt": prompt, "options": opts, "stream": False},
                timeout=timeout)
            r2.raise_for_status()
            return (r2.json().get("response") or "").strip()
        except Exception:
            time.sleep(backoff); backoff *= 1.5; tries += 1
    return ""

