# the_ai_tutor/workspace.py
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    session,
)

# --------------------------------------------------------------------------------------
# Optional helpers from sibling modules (works whether package or flat layout)
# --------------------------------------------------------------------------------------
_make_quiz_items = None
try:
    # packaged
    from .quiz_app import generate_quiz_items as _make_quiz_items  # type: ignore
except Exception:
    try:
        # flat
        from quiz_app import generate_quiz_items as _make_quiz_items  # type: ignore
    except Exception:
        _make_quiz_items = None

try:
    from .notes_app import summarize_text  # type: ignore
except Exception:
    try:
        from notes_app import summarize_text  # type: ignore
    except Exception:
        # super simple fallback
        def summarize_text(txt: str) -> str:  # type: ignore
            plain = re.sub(r"<[^>]+>", " ", txt or "")
            plain = re.sub(r"\s+", " ", plain).strip()
            return f"<h2>Summary</h2><p>{plain[:1200]}</p>"

# --------------------------------------------------------------------------------------
# Blueprint
# --------------------------------------------------------------------------------------
workspace_bp = Blueprint(
    "workspace",
    __name__,
    template_folder="templates",
    url_prefix="/workspace",
)

# --------------------------------------------------------------------------------------
# Writable store (Render-safe: falls back to /tmp)
# --------------------------------------------------------------------------------------
def _writable_base(default: Optional[str] = None) -> str:
    base = os.getenv("WRITE_BASE", default or os.getcwd())
    return base if os.access(base, os.W_OK) else "/tmp"

WS_STORE = os.path.join(_writable_base(), "workspace_store")
os.makedirs(WS_STORE, exist_ok=True)


def _ws_key() -> str:
    return (session.get("ws_key") or "").strip()


def _ws_dir() -> str:
    key = _ws_key()
    if not key:
        return ""
    d = os.path.join(WS_STORE, key)
    os.makedirs(d, exist_ok=True)
    return d


def _write_text(name: str, text: str) -> None:
    d = _ws_dir()
    if not d:
        return
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text or "")


def _read_text(name: str) -> str:
    d = _ws_dir()
    if not d:
        return ""
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _write_json(name: str, obj: Any) -> None:
    _write_text(name, json.dumps(obj, ensure_ascii=False, indent=2))


def _read_json(name: str) -> Any:
    raw = _read_text(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _reset_session() -> None:
    key = _ws_key()
    if key:
        shutil.rmtree(os.path.join(WS_STORE, key), ignore_errors=True)
    for k in ("ws_key", "shared_source_title", "ws_cache_key"):
        session.pop(k, None)


def _get_source_text() -> str:
    return _read_text("source.txt")


def _set_source(title: str, text: str) -> None:
    key = uuid.uuid4().hex
    session["ws_key"] = key
    session["shared_source_title"] = (title or "Untitled").strip()
    session["ws_cache_key"] = _hash(text or "")
    os.makedirs(_ws_dir(), exist_ok=True)
    _write_text("source.txt", text or "")
    # wipe stale artifacts
    for fname in ("notes.html", "podcast.txt", "quiz.json", "flash.json"):
        fp = os.path.join(_ws_dir(), fname)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

# --------------------------------------------------------------------------------------
# LLM provider: Unified config
# --------------------------------------------------------------------------------------
from llm_config import LLM_PROVIDER, API_KEY, BASE_URL, MODEL


def _llm_complete(
    prompt_or_messages: str | List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.22,
    max_tokens: int = 700,
    timeout: int = 60,
) -> str:
    """Call the configured LLM provider."""
    if not API_KEY and LLM_PROVIDER != "ollama":
        return ""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    messages = (
        [{"role": "user", "content": prompt_or_messages}]
        if isinstance(prompt_or_messages, str)
        else [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in (prompt_or_messages or [])]
    )
    if not messages:
        return ""

    model_to_use = model or MODEL
    
    body = {
        "model": model_to_use,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    
    if LLM_PROVIDER in ("groq", "openrouter", "openai"):
        body["stream"] = False
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""
    
    elif LLM_PROVIDER == "ollama":
        body = {
            "model": model_to_use,
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
            return (data.get("message") or {}).get("content", "").strip()
        except Exception:
            return ""
    
    return ""

# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
@workspace_bp.get("/", endpoint="open_workspace")
def open_workspace():
    return render_template("workspace.html", title="Workspace")

# --------------------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------------------
@workspace_bp.post("/reset", endpoint="reset_workspace")
def reset_workspace():
    _reset_session()
    return jsonify({"ok": True})

# --------------------------------------------------------------------------------------
# Wiring from dashboard (seed from generated or saved)
# --------------------------------------------------------------------------------------
@workspace_bp.post("/use_generated", endpoint="use_generated")
def use_generated():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No source text provided."}), 400
    _reset_session()
    _set_source(title, text)
    return jsonify({"ok": True})


@workspace_bp.post("/use_saved", endpoint="use_saved")
def use_saved():
    """
    Open a saved note WITHOUT regenerating; serve its HTML directly.
    This sets a faux source so podcast/quiz/flashcards still work.
    """
    notes_file = current_app.config.get("NOTES_FILE", "saved_notes.json")
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Missing title."}), 400

    try:
        arr = json.load(open(notes_file, "r", encoding="utf-8"))
    except Exception:
        arr = []

    match = next((n for n in arr if (n.get("title", "").strip().lower() == title.lower())), None)
    if not match:
        return jsonify({"ok": False, "error": "Saved note not found"}), 404

    html = (match.get("content") or "").strip()
    faux_source = f"[SAVED NOTE] {title}\n(length={len(_strip_html_to_text(html))} chars)"
    _reset_session()
    _set_source(title, faux_source)
    _write_text("notes.html", html)
    return jsonify({"ok": True})


@workspace_bp.post("/seed", endpoint="seed_workspace")
def seed_workspace():
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "Untitled").strip()
    text = (body.get("text") or "").strip()
    notes_html = (body.get("notes_html") or "").strip()
    prebuild_quiz = bool(body.get("prebuild_quiz"))

    if not text:
        return jsonify({"ok": False, "error": "No source text provided."}), 400

    _reset_session()
    _set_source(title, text)

    if notes_html:
        _write_text("notes.html", notes_html)
    else:
        try:
            _write_text("notes.html", summarize_text(text))
        except Exception:
            pass

    if prebuild_quiz and _make_quiz_items:
        try:
            items = _make_quiz_items(text, title or "Topic", 10)
            _write_json("quiz.json", items)
        except Exception:
            pass

    return jsonify({"ok": True, "title": title})

# --------------------------------------------------------------------------------------
# APIs with caching
# --------------------------------------------------------------------------------------
@workspace_bp.post("/api/title", endpoint="api_update_title")
def api_update_title():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    session["shared_source_title"] = title
    return jsonify({"ok": True, "title": title})


@workspace_bp.post("/api/notes", endpoint="api_generate_notes")
def api_generate_notes():
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    if not force:
        cached = _read_text("notes.html")
        if cached:
            return jsonify({"ok": True, "html": cached})

    src = _get_source_text()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400

    html = summarize_text(src)
    _write_text("notes.html", html)
    session["ws_cache_key"] = _hash(src)
    return jsonify({"ok": True, "html": html})


@workspace_bp.post("/api/podcast", endpoint="api_generate_podcast")
def api_generate_podcast():
    """Generate or return cached podcast script (hosted LLM by default)."""
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    if not force:
        cached = _read_text("podcast.txt")
        if cached:
            return jsonify({"ok": True, "script": cached})

    notes_html = _read_text("notes.html")
    src = _get_source_text()
    if not (notes_html or src):
        return jsonify({"ok": False, "error": "No source in session."}), 400

    title = session.get("shared_source_title", "Untitled")
    plain = _strip_html_to_text(notes_html) if notes_html else _strip_html_to_text(src)
    prompt = f"""Turn this study note into a clear 3–4 minute podcast script.
- Conversational, friendly teacher tone
- Two speakers: Host: and Guest: (label each turn)
- Include light signposting and a brief recap
- No hallucinations; stick strictly to the note content

TITLE: {title}
NOTES (plain text):
{plain}
"""
    script = _llm_complete(prompt, temperature=0.24, max_tokens=700)
    if not script:
        return jsonify({"ok": False, "error": "Empty script returned from model."}), 502

    _write_text("podcast.txt", script)
    session["ws_cache_key"] = _hash(src or plain)
    return jsonify({"ok": True, "script": script})

# --------------------------------------------------------------------------------------
# QUIZ API (delegates to quiz_app.generate_quiz_items)
# --------------------------------------------------------------------------------------
@workspace_bp.post("/api/quiz", endpoint="api_generate_quiz")
def api_generate_quiz():
    """
    Thin wrapper: call quiz_app.generate_quiz_items().
    Supports 'avoid' list and simple on-disk caching.
    """
    body = request.get_json(silent=True) or {}
    request_count = int(body.get("count", 12))
    avoid = set((q or "").strip().lower() for q in (body.get("avoid") or []))
    force = bool(body.get("force", True))

    if not force:
        cached = _read_json("quiz.json")
        if isinstance(cached, list) and cached:
            if avoid:
                fresh = [it for it in cached if (it.get("question", "").strip().lower() not in avoid)]
                return jsonify({"ok": True, "items": fresh[:request_count]})
            return jsonify({"ok": True, "items": cached[:request_count]})

    src = (_get_source_text() or "").strip()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400

    title = (session.get("shared_source_title") or "Topic").strip()
    if _make_quiz_items is None:
        return jsonify({"ok": False, "error": "Quiz helper unavailable."}), 500

    try:
        # Request more than needed to account for questions that might be filtered out
        # But be more conservative for small batches
        generate_count = max(request_count + 2, int(request_count * 1.5))
        print(f"[WORKSPACE] Generating {generate_count} quiz items (requested: {request_count}) from {len(src)} chars of source text...")
        
        # Retry once if first attempt returns empty
        batch = _make_quiz_items(src, title, generate_count)
        if not batch:
            print(f"[WORKSPACE] First attempt returned empty, retrying...")
            batch = _make_quiz_items(src, title, generate_count)
        
        print(f"[WORKSPACE] Quiz generation returned {len(batch or [])} items")
    except Exception as e:
        error_msg = f"Quiz generation failed: {type(e).__name__}: {e}"
        print(f"[WORKSPACE] {error_msg}")
        return jsonify({"ok": False, "error": error_msg}), 502

    # Check if we got any items at all
    if not batch or len(batch) == 0:
        error_msg = "Quiz generation returned no items (LLM may be unavailable or timing out)"
        print(f"[WORKSPACE] {error_msg}")
        return jsonify({"ok": False, "error": error_msg}), 502

    out, seen = [], set()
    skipped_duplicates = 0
    for it in (batch or []):
        q = (it.get("question") or "").strip()
        if not q:
            print(f"[WORKSPACE] Skipping item with empty question")
            continue
        k = q.lower()
        if k in avoid:
            skipped_duplicates += 1
            continue
        if k in seen:
            skipped_duplicates += 1
            continue
        out.append(it)
        seen.add(k)
        if len(out) >= request_count:
            break

    print(f"[WORKSPACE] After filtering: {len(out)} valid items from {len(batch or [])} generated (skipped {skipped_duplicates} duplicates/avoided)")

    if not out:
        # Don't fail - just return what we have or a helpful error
        if len(batch or []) > 0:
            error_msg = f"No NEW quiz items produced - {len(batch)} generated but all were duplicates or avoided."
            print(f"[WORKSPACE] {error_msg}")
            # Return the ones we have anyway, even if they're duplicates
            out = batch[:request_count] if batch else []
        
        if not out:
            error_msg = "Quiz generation failed - no items produced. Try regenerating."
            print(f"[WORKSPACE] {error_msg}")
            return jsonify({"ok": False, "error": error_msg}), 502

    # merge cache (dedupe by question)
    cached = _read_json("quiz.json") or []
    merged = cached + out
    uniq, seen_all = [], set()
    for it in merged:
        qk = (it.get("question") or "").strip().lower()
        if qk and qk not in seen_all:
            uniq.append(it)
            seen_all.add(qk)
    _write_json("quiz.json", uniq[:200])

    session["ws_cache_key"] = _hash(src)
    return jsonify({"ok": True, "items": out})

# --------------------------------------------------------------------------------------
# FLASHCARDS API
# --------------------------------------------------------------------------------------
@workspace_bp.post("/api/flashcards", endpoint="api_generate_flashcards")
def api_generate_flashcards():
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    if not force:
        cached = _read_json("flash.json")
        if cached is not None:
            return jsonify({"ok": True, "cards": cached})

    src = _get_source_text()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400

    prompt = """Create 16 concise Q/A flashcards as JSON:
[{"front": str, "back": str}]
Use crisp definitions or cause→effect pairs. Stay faithful to the source ONLY.

SOURCE:
""" + src

    try:
        raw = _llm_complete(prompt, temperature=0.20, max_tokens=700)
        m = re.search(r"\[\s*\{.*?\}\s*\]", raw or "", flags=re.S)
        payload = m.group(0) if m else (raw or "[]")
        cards = json.loads(payload)
        if not isinstance(cards, list):
            raise ValueError("Model did not return a JSON list.")
    except Exception as e:
        return jsonify({"ok": False, "error": f"Flashcards parse/generation failed: {e}"}), 502

    _write_json("flash.json", cards)
    session["ws_cache_key"] = _hash(src)
    return jsonify({"ok": True, "cards": cards})

