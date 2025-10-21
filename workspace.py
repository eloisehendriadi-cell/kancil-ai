# the_ai_tutor/workspace.py
from __future__ import annotations

import os, json, re, shutil, uuid, hashlib, time, requests
from typing import Any, Dict, List
from flask import Blueprint, current_app, jsonify, render_template, request, session

# ======== Optional: shared quiz/notes imports ========
try:
    from .quiz_app import generate_quiz_items
except Exception:
    from quiz_app import generate_quiz_items  # type: ignore

try:
    from .notes_app import summarize_text
except Exception:
    from notes_app import summarize_text  # type: ignore

# ======== Blueprint ========
workspace_bp = Blueprint("workspace", __name__, url_prefix="/workspace")

# ======== Disk store ========
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WS_STORE = os.path.join(BASE_DIR, "workspace_store")
os.makedirs(WS_STORE, exist_ok=True)


def _ws_key() -> str:
    return (session.get("ws_key") or "").strip()


def _ws_dir() -> str:
    k = _ws_key()
    if not k:
        return ""
    d = os.path.join(WS_STORE, k)
    os.makedirs(d, exist_ok=True)
    return d


def _write_text(name: str, text: str):
    d = _ws_dir()
    if not d:
        return
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write(text or "")


def _read_text(name: str) -> str:
    d = _ws_dir()
    if not d:
        return ""
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return ""
    try:
        return open(p, "r", encoding="utf-8").read()
    except Exception:
        return ""


def _write_json(name: str, obj: Any):
    _write_text(name, json.dumps(obj, ensure_ascii=False, indent=2))


def _read_json(name: str):
    raw = _read_text(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _reset_session():
    key = _ws_key()
    if key:
        shutil.rmtree(os.path.join(WS_STORE, key), ignore_errors=True)
    for k in ("ws_key", "shared_source_title", "ws_cache_key"):
        session.pop(k, None)


def _get_source_text() -> str:
    return _read_text("source.txt")


def _set_source(title: str, text: str):
    key = uuid.uuid4().hex
    session["ws_key"] = key
    session["shared_source_title"] = (title or "Untitled").strip()
    session["ws_cache_key"] = _hash(text or "")
    os.makedirs(_ws_dir(), exist_ok=True)
    _write_text("source.txt", text or "")
    for fname in ("notes.html", "podcast.txt", "quiz.json", "flash.json"):
        fp = os.path.join(_ws_dir(), fname)
        if os.path.exists(fp):
            os.remove(fp)


# ======== Hosted LLM wrapper (OpenRouter or Ollama) ========
def _chat_complete(prompt: str, *, model=None, temperature=0.2, num_predict=600, timeout=90) -> str:
    """
    Uses OpenRouter if OPENROUTER_API_KEY is present, else Ollama locally.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        base = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
        use_model = model or os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("SITE_URL", "https://kancil.ai"),
            "X-Title": os.getenv("SITE_NAME", "Kancil AI"),
            "Content-Type": "application/json",
        }
        body = {
            "model": use_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        try:
            r = requests.post(f"{base}/chat/completions", json=body, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            print("OpenRouter error:", e)
            return ""

    # Fallback to local Ollama
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = model or os.getenv("MODEL_NAME", "llama3.2:3b")
    opts = {"temperature": temperature, "num_predict": num_predict}
    try:
        r = requests.post(f"{host}/api/chat",
                          json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                "options": opts, "stream": False}, timeout=timeout)
        if r.status_code != 404:
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "").strip()
    except Exception:
        pass
    return ""


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ======== Routes ========

@workspace_bp.get("/", endpoint="open_workspace")
def open_workspace():
    return render_template("workspace.html", title="Workspace")


@workspace_bp.post("/reset", endpoint="reset_workspace")
def reset_workspace():
    _reset_session()
    return jsonify({"ok": True})


@workspace_bp.post("/use_generated", endpoint="use_generated")
def use_generated():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No source text provided"})
    _reset_session()
    _set_source(title, text)
    return jsonify({"ok": True})


@workspace_bp.post("/use_saved", endpoint="use_saved")
def use_saved():
    notes_file = current_app.config.get("NOTES_FILE", "saved_notes.json")
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Missing title."})

    try:
        arr = json.load(open(notes_file, "r", encoding="utf-8"))
    except Exception:
        arr = []

    for n in arr:
        if (n.get("title") or "").strip().lower() == title.lower():
            html = (n.get("content") or "").strip()
            faux_source = f"[SAVED NOTE] {title}\n(length={len(html)})"
            _reset_session()
            _set_source(title, faux_source)
            _write_text("notes.html", html)
            return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Saved note not found"})


@workspace_bp.post("/seed", endpoint="seed_workspace")
def seed_workspace():
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "Untitled").strip()
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No source text provided"})

    _reset_session()
    _set_source(title, text)
    try:
        _write_text("notes.html", summarize_text(text))
    except Exception:
        pass
    return jsonify({"ok": True, "title": title})


# ======== API Endpoints ========

@workspace_bp.post("/api/title")
def api_update_title():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    session["shared_source_title"] = title
    return jsonify({"ok": True})


@workspace_bp.post("/api/notes")
def api_generate_notes():
    force = bool((request.get_json(silent=True) or {}).get("force"))
    if not force:
        cached = _read_text("notes.html")
        if cached:
            return jsonify({"ok": True, "html": cached})

    src = _get_source_text()
    if not src:
        return jsonify({"ok": False, "error": "No source in session"})
    html = summarize_text(src)
    _write_text("notes.html", html)
    return jsonify({"ok": True, "html": html})


@workspace_bp.post("/api/podcast")
def api_generate_podcast():
    force = bool((request.get_json(silent=True) or {}).get("force"))
    if not force:
        cached = _read_text("podcast.txt")
        if cached:
            return jsonify({"ok": True, "script": cached})

    notes_html = _read_text("notes.html")
    src = _get_source_text()
    plain = _strip_html_to_text(notes_html or src)
    title = session.get("shared_source_title", "Untitled")
    prompt = f"""Turn this study note into a clear 3–4 minute educational podcast dialogue.
Tone: friendly teacher + student.
Two speakers: Host: and Guest:
Keep strictly to facts from the note.

Title: {title}
Content:
{plain}"""
    script = _chat_complete(prompt, temperature=0.24, num_predict=700)
    if not script:
        return jsonify({"ok": False, "error": "Empty script returned"})
    _write_text("podcast.txt", script)
    return jsonify({"ok": True, "script": script})


@workspace_bp.post("/api/quiz")
def api_generate_quiz():
    force = bool((request.get_json(silent=True) or {}).get("force"))
    if not force:
        cached = _read_json("quiz.json")
        if cached:
            return jsonify({"ok": True, "items": cached})

    src = (_get_source_text() or "").strip()
    if not src:
        return jsonify({"ok": False, "error": "No source in session"})
    title = session.get("shared_source_title", "Topic")

    try:
        items = generate_quiz_items(src, title, 12)
        _write_json("quiz.json", items)
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Quiz generation failed: {e}"})


@workspace_bp.post("/api/flashcards")
def api_generate_flashcards():
    force = bool((request.get_json(silent=True) or {}).get("force"))
    if not force:
        cached = _read_json("flash.json")
        if cached:
            return jsonify({"ok": True, "cards": cached})

    src = _get_source_text()
    if not src:
        return jsonify({"ok": False, "error": "No source in session"})

    prompt = f"""Create 16 concise Q/A flashcards as JSON:
[{{"front": str, "back": str}}]
Use short, factual pairs derived only from the following study notes:
{src}
"""
    raw = _chat_complete(prompt, temperature=0.2, num_predict=500)
    m = re.search(r"\[\s*\{.*\}\s*\]", raw, flags=re.S)
    try:
        cards = json.loads(m.group(0) if m else raw)
    except Exception:
        cards = []
    _write_json("flash.json", cards)
    return jsonify({"ok": True, "cards": cards})

