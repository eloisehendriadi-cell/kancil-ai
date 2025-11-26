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
    send_file,
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
    """Fully reset workspace session and clear all cached artifacts."""
    key = _ws_key()
    if key:
        ws_path = os.path.join(WS_STORE, key)
        if os.path.exists(ws_path):
            shutil.rmtree(ws_path, ignore_errors=True)
    # Clear all session keys related to workspace
    for k in ("ws_key", "shared_source_title", "ws_cache_key"):
        session.pop(k, None)


def _get_source_text() -> str:
    return _read_text("source.txt")


def _set_source(title: str, text: str) -> None:
    """Set new source material and clear all cached artifacts."""
    # Generate new unique key for this workspace session
    key = uuid.uuid4().hex
    session["ws_key"] = key
    session["shared_source_title"] = (title or "Untitled").strip()
    # IMPORTANT: Set cache key to empty initially to force fresh generation
    # It will be set to the actual hash after first successful generation
    session["ws_cache_key"] = ""
    
    # Create workspace directory (will be fresh since key is new)
    ws_dir = _ws_dir()
    os.makedirs(ws_dir, exist_ok=True)
    
    print(f"[WORKSPACE] Created new workspace: key={key}, dir={ws_dir}")
    
    # Write source text
    _write_text("source.txt", text or "")
    
    # Ensure all cached artifacts are removed (defensive cleanup)
    # This shouldn't be necessary with a new key, but better safe than sorry
    for fname in ("notes.html", "podcast.txt", "quiz.json", "flash.json"):
        fp = os.path.join(ws_dir, fname)
        if os.path.exists(fp):
            print(f"[WORKSPACE] Warning: Found unexpected {fname} in new workspace, removing it")
            try:
                os.remove(fp)
            except Exception as e:
                print(f"[WORKSPACE] Error removing {fname}: {e}")


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


def _generate_podcast_audio(script: str) -> Optional[str]:
    """
    Generate audio from podcast script using OpenAI TTS API.
    Returns the URL path to the audio file, or None if generation fails.
    Uses simple MP3 concatenation without external dependencies.
    """
    if not API_KEY or LLM_PROVIDER != "openai":
        print("[WORKSPACE] Audio generation requires OpenAI API key and provider")
        return None
    
    # Parse script into speaker segments
    lines = script.split('\n')
    segments = []
    current_speaker = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect speaker labels - handle both plain and markdown formats
        # Patterns: "Host:", "**Host:**", "Host **", etc.
        host_match = re.match(r'^\*{0,2}Host\*{0,2}:\s*(.*)', line, re.IGNORECASE)
        guest_match = re.match(r'^\*{0,2}Guest\*{0,2}:\s*(.*)', line, re.IGNORECASE)
        
        if host_match:
            if current_speaker and current_text:
                segments.append((current_speaker, ' '.join(current_text)))
            current_speaker = "host"
            text_after_label = host_match.group(1).strip()
            current_text = [text_after_label] if text_after_label else []
        elif guest_match:
            if current_speaker and current_text:
                segments.append((current_speaker, ' '.join(current_text)))
            current_speaker = "guest"
            text_after_label = guest_match.group(1).strip()
            current_text = [text_after_label] if text_after_label else []
        elif current_speaker:
            current_text.append(line)
    
    # Don't forget the last segment
    if current_speaker and current_text:
        segments.append((current_speaker, ' '.join(current_text)))
    
    if not segments:
        print("[WORKSPACE] No speaker segments found in script, generating single audio from full text")
        # Generate audio from the full script without speaker separation
        segments = [("host", script)]  # Use single voice for entire script
    
    # Voice mapping: Use distinctly different voices for Host and Guest
    # OpenAI TTS voices: alloy (neutral), echo (male), fable (British), onyx (deep male), nova (female), shimmer (soft female)
    voices = {
        "host": "echo",     # Male voice - clear and authoritative
        "guest": "shimmer"  # Female voice - warm and friendly
    }
    
    ws_dir = _ws_dir()
    if not ws_dir:
        print("[WORKSPACE] No workspace directory available")
        return None
    
    print(f"[WORKSPACE] Generating audio for {len(segments)} speaker segments")
    
    try:
        # Collect all MP3 segments
        audio_segments = []
        
        for i, (speaker, text) in enumerate(segments):
            if not text.strip():
                continue
            
            voice = voices.get(speaker, "echo")
            print(f"[WORKSPACE] Generating segment {i+1}/{len(segments)} with voice '{voice}' for {speaker}")
            
            # Call OpenAI TTS API
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            tts_body = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "mp3"
            }
            
            r = requests.post(
                f"{BASE_URL}/audio/speech",
                json=tts_body,
                headers=headers,
                timeout=60
            )
            r.raise_for_status()
            
            # Store the MP3 data
            audio_segments.append(r.content)
        
        # Concatenate all MP3 segments (simple binary concatenation works for MP3)
        combined_audio = b''.join(audio_segments)
        
        # Save combined audio
        audio_path = os.path.join(ws_dir, "podcast_audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(combined_audio)
        
        print(f"[WORKSPACE] Successfully generated podcast audio with {len(segments)} segments")
        
        # Return URL path
        return f"/workspace/podcast_audio/{_ws_key()}"
        
    except Exception as e:
        print(f"[WORKSPACE] TTS API error: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    """Use generated content in workspace, properly clearing previous state."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No source text provided."}), 400
    
    # IMPORTANT: Completely reset previous workspace state
    _reset_session()
    
    # Set new source (creates new workspace key)
    _set_source(title, text)
    
    return jsonify({"ok": True})


@workspace_bp.post("/use_saved", endpoint="use_saved")
def use_saved():
    """
    Open a saved note and restore ALL workspace artifacts (notes, quiz, flashcards, podcast).
    This allows users to continue where they left off.
    Properly resets previous workspace state before loading.
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
    
    # Get saved source text (preferred) or extract from HTML
    text_source = (match.get("source") or "").strip()
    if not text_source or len(text_source.strip()) < 50:
        # Fallback: extract actual text content from HTML
        text_source = _strip_html_to_text(html)
    
    if not text_source or len(text_source.strip()) < 50:
        # Final fallback to minimal source if extraction fails
        text_source = f"[SAVED NOTE] {title}\n(length={len(html)} chars)"
    
    # IMPORTANT: Reset previous workspace state completely
    _reset_session()
    
    # Set new source with actual text content (creates new workspace key)
    _set_source(title, text_source)
    
    # Write the saved notes HTML
    _write_text("notes.html", html)
    
    # Restore all saved artifacts if they exist
    quiz_data = match.get("quiz")
    if quiz_data is not None:
        _write_json("quiz.json", quiz_data)
        print(f"[WORKSPACE] Restored {len(quiz_data) if isinstance(quiz_data, list) else 0} quiz questions")
    
    flash_data = match.get("flashcards")
    if flash_data is not None:
        _write_json("flash.json", flash_data)
        print(f"[WORKSPACE] Restored {len(flash_data) if isinstance(flash_data, list) else 0} flashcards")
    
    podcast_text = match.get("podcast")
    if podcast_text:
        _write_text("podcast.txt", podcast_text)
        print(f"[WORKSPACE] Restored podcast script ({len(podcast_text)} chars)")
    
    # Mark cache as valid since we've loaded saved content
    session["ws_cache_key"] = _hash(text_source)
    
    return jsonify({"ok": True})


@workspace_bp.post("/seed", endpoint="seed_workspace")
def seed_workspace():
    """Seed workspace with new source material, properly clearing previous state."""
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "Untitled").strip()
    text = (body.get("text") or "").strip()
    notes_html = (body.get("notes_html") or "").strip()
    prebuild_quiz = bool(body.get("prebuild_quiz"))

    if not text:
        return jsonify({"ok": False, "error": "No source text provided."}), 400

    # IMPORTANT: Completely reset previous workspace state
    _reset_session()
    
    # Set new source (creates new workspace key and directory)
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
    
    src = _get_source_text()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400
    
    # Validate cache is for current source
    current_cache_key = _hash(src)
    stored_cache_key = session.get("ws_cache_key", "")
    cache_is_stale = (current_cache_key != stored_cache_key)
    
    if not force and not cache_is_stale:
        cached = _read_text("notes.html")
        if cached:
            return jsonify({"ok": True, "html": cached})

    html = summarize_text(src)
    _write_text("notes.html", html)
    session["ws_cache_key"] = _hash(src)
    return jsonify({"ok": True, "html": html})


@workspace_bp.post("/api/podcast", endpoint="api_generate_podcast")
def api_generate_podcast():
    """Generate podcast script and audio with different voices for Host and Guest."""
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    
    notes_html = _read_text("notes.html")
    src = _get_source_text()
    if not (notes_html or src):
        return jsonify({"ok": False, "error": "No source in session."}), 400
    
    # Validate cache is for current source
    current_cache_key = _hash(src or notes_html)
    stored_cache_key = session.get("ws_cache_key", "")
    cache_is_stale = (current_cache_key != stored_cache_key)
    
    if not force and not cache_is_stale:
        cached_script = _read_text("podcast.txt")
        cached_audio = _read_text("podcast_audio.mp3")
        if cached_script:
            audio_url = f"/workspace/podcast_audio/{_ws_key()}" if cached_audio else None
            return jsonify({"ok": True, "script": cached_script, "audio_url": audio_url})
        # If force=false and no cache, return empty to indicate "not generated yet"
        if not force:
            return jsonify({"ok": True, "script": "", "audio_url": None, "message": "No podcast generated yet. Generate to create podcast script."})

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
    
    # Generate audio from script with different voices
    audio_url = None
    try:
        audio_url = _generate_podcast_audio(script)
    except Exception as e:
        print(f"[WORKSPACE] Audio generation failed: {e}")
        # Continue without audio - script is still saved
    
    session["ws_cache_key"] = _hash(src or plain)
    return jsonify({"ok": True, "script": script, "audio_url": audio_url})

# --------------------------------------------------------------------------------------
# QUIZ API (delegates to quiz_app.generate_quiz_items)
# --------------------------------------------------------------------------------------
@workspace_bp.post("/api/quiz", endpoint="api_generate_quiz")
def api_generate_quiz():
    """
    Thin wrapper: call quiz_app.generate_quiz_items().
    Supports 'avoid' list and simple on-disk caching.
    Validates workspace session before using cache.
    """
    body = request.get_json(silent=True) or {}
    request_count = int(body.get("count", 12))
    avoid = set((q or "").strip().lower() for q in (body.get("avoid") or []))
    force = bool(body.get("force", True))

    # IMPORTANT: Validate we have a workspace session before using cache
    src = (_get_source_text() or "").strip()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400

    # Check if cache is valid for current session
    current_cache_key = _hash(src)
    stored_cache_key = session.get("ws_cache_key", "")
    cache_is_stale = (current_cache_key != stored_cache_key)
    
    if not force and not cache_is_stale:
        cached = _read_json("quiz.json")
        if isinstance(cached, list) and cached:
            if avoid:
                fresh = [it for it in cached if (it.get("question", "").strip().lower() not in avoid)]
                return jsonify({"ok": True, "items": fresh[:request_count]})
            return jsonify({"ok": True, "items": cached[:request_count]})
        # If force=false and no cache, return empty to indicate "not generated yet"
        if not force:
            return jsonify({"ok": True, "items": [], "message": "No quiz generated yet. Generate to create questions."})

    src = (_get_source_text() or "").strip()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400

    title = (session.get("shared_source_title") or "Topic").strip()
    if _make_quiz_items is None:
        return jsonify({"ok": False, "error": "Quiz helper unavailable."}), 500

    # Log source info for debugging (especially helpful for YouTube transcripts)
    src_length = len(src)
    ws_dir = _ws_dir()
    ws_key = _ws_key()
    print(f"[WORKSPACE] Quiz request: count={request_count}, source_length={src_length}, title='{title}'")
    print(f"[WORKSPACE] Workspace: key={ws_key}, dir={ws_dir}, cache_stale={cache_is_stale}")
    
    # If cache is stale, clear it completely to start fresh
    if cache_is_stale:
        print(f"[WORKSPACE] Cache is STALE (source changed), clearing quiz cache and starting fresh")
        cached_quiz = []
        cached_count = 0
        offset = 0
        # Delete the old cache file if it somehow exists
        try:
            fp = os.path.join(ws_dir, "quiz.json")
            if os.path.exists(fp):
                print(f"[WORKSPACE] Warning: Found old quiz.json in new workspace, removing it")
                os.remove(fp)
        except Exception as e:
            print(f"[WORKSPACE] Error removing old quiz.json: {e}")
    else:
        # Cache is valid, read it for incremental generation
        print(f"[WORKSPACE] Cache is VALID, reading for incremental generation")
        cached_quiz = _read_json("quiz.json") or []
        cached_count = len(cached_quiz)
        offset = cached_count // 7  # Every ~7 questions, move to next content window
    
    print(f"[WORKSPACE] Cached questions: {cached_count}, using offset: {offset}")
    
    try:
        # Request more than needed to account for questions that might be filtered out
        # But be more conservative for small batches
        generate_count = max(request_count + 2, int(request_count * 1.5))
        print(f"[WORKSPACE] Generating {generate_count} quiz items (requested: {request_count}) from {src_length} chars of source text...")
        
        # For very long sources, warn user
        if src_length > 10000:
            print(f"[WORKSPACE] Warning: Very long source ({src_length} chars) may cause slower generation")
        
        # Pass offset to enable sliding window for long sources
        # Try with offset first, fall back to no offset for compatibility
        try:
            batch = _make_quiz_items(src, title, generate_count, offset)
        except TypeError:
            # Fallback for old signature without offset
            print(f"[WORKSPACE] Quiz function doesn't support offset, using legacy mode")
            batch = _make_quiz_items(src, title, generate_count)
        
        if not batch:
            print(f"[WORKSPACE] First attempt returned empty, retrying...")
            time.sleep(1)  # Brief pause before retry
            try:
                batch = _make_quiz_items(src, title, generate_count, offset)
            except TypeError:
                batch = _make_quiz_items(src, title, generate_count)
        
        print(f"[WORKSPACE] Quiz generation returned {len(batch or [])} items")
    except Exception as e:
        import traceback as _tb
        tb = _tb.format_exc()[-800:]
        error_msg = f"Quiz generation failed: {type(e).__name__}: {e}"
        print(f"[WORKSPACE] {error_msg}\\n{tb}")
        
        # Provide more helpful error messages
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            return jsonify({
                "ok": False, 
                "error": "Quiz generation timed out. The source might be too long. Try with a shorter text or wait and try again.",
                "trace": tb
            }), 502
        elif "connection" in str(e).lower() or "ConnectTimeout" in str(e):
            return jsonify({
                "ok": False,
                "error": "Cannot connect to the LLM service. Please check your internet connection and API configuration.",
                "trace": tb
            }), 502
        else:
            return jsonify({"ok": False, "error": error_msg, "trace": tb}), 502

    # Check if we got any items at all
    if not batch or len(batch) == 0:
        # Check if source is too short
        if src_length < 200:
            error_msg = "Source text is too short to generate quiz questions. Please provide at least 200 characters of content."
        # Check if it's a long transcript that might have timed out
        elif src_length > 10000:
            error_msg = "Quiz generation returned no items. The source might be too long (long YouTube transcripts can cause timeouts). Try refreshing and generating again."
        else:
            error_msg = "Quiz generation returned no items (LLM may be unavailable or timing out). Please try again in a moment."
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

    # Merge with cache only if cache is valid (not stale)
    # If cache is stale, we already cleared cached_quiz above, so just use new questions
    if not cache_is_stale and cached_quiz:
        print(f"[WORKSPACE] Merging {len(out)} new questions with {len(cached_quiz)} cached questions")
        merged = cached_quiz + out
    else:
        print(f"[WORKSPACE] Starting fresh with {len(out)} new questions (no merge)")
        merged = out
    
    # Deduplicate by question text
    uniq, seen_all = [], set()
    for it in merged:
        qk = (it.get("question") or "").strip().lower()
        if qk and qk not in seen_all:
            uniq.append(it)
            seen_all.add(qk)
    
    print(f"[WORKSPACE] Saving {len(uniq)} total unique questions to cache")
    _write_json("quiz.json", uniq[:200])

    # Update cache key to mark cache as valid
    session["ws_cache_key"] = _hash(src)
    return jsonify({"ok": True, "items": out})

# --------------------------------------------------------------------------------------
# FLASHCARDS API
# --------------------------------------------------------------------------------------
@workspace_bp.post("/api/flashcards", endpoint="api_generate_flashcards")
def api_generate_flashcards():
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    
    src = _get_source_text()
    if not src:
        return jsonify({"ok": False, "error": "No source in session."}), 400
    
    # Validate cache is for current source
    current_cache_key = _hash(src)
    stored_cache_key = session.get("ws_cache_key", "")
    cache_is_stale = (current_cache_key != stored_cache_key)
    
    if not force and not cache_is_stale:
        cached = _read_json("flash.json")
        if cached is not None and len(cached) > 0:
            return jsonify({"ok": True, "cards": cached})
        # If force=false and no cache, return empty to indicate "not generated yet"
        if not force:
            return jsonify({"ok": True, "cards": [], "message": "No flashcards generated yet. Generate to create flashcards."})

    prompt = """Create 16 concise Q/A flashcards as JSON:
[{"front": str, "back": str}]
Use crisp definitions or cause→effect pairs. Stay faithful to the source ONLY.

SOURCE:
""" + src

    try:
        raw = _llm_complete(prompt, temperature=0.20, max_tokens=700)
        # Try to extract a JSON array from model output
        m = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", raw or "", flags=re.S)
        payload = m.group(0) if m else (raw or "[]")
        try:
            cards = json.loads(payload)
        except Exception:
            # Attempt to recover by extracting individual JSON objects
            objs = re.findall(r"\{[\s\S]*?\}", payload)
            parsed = []
            for o in objs:
                try:
                    parsed.append(json.loads(o))
                except Exception:
                    continue
            if parsed:
                cards = parsed
            else:
                raise ValueError("Model did not return a parseable JSON list.")
        if not isinstance(cards, list):
            raise ValueError("Model did not return a JSON list.")
    except Exception as e:
        import traceback as _tb
        tb = _tb.format_exc()[-800:]
        print(f"[WORKSPACE] Flashcards generation error: {type(e).__name__}: {e}\n{tb}")
        # Return helpful error with truncated model output for debugging
        snippet = (raw or "")[:800]
        return jsonify({"ok": False, "error": f"Flashcards parse/generation failed: {type(e).__name__}: {e}", "model_output_snippet": snippet}), 502

    _write_json("flash.json", cards)
    session["ws_cache_key"] = _hash(src)
    return jsonify({"ok": True, "cards": cards})


# SAVE WORKSPACE API
@workspace_bp.post("/api/save", endpoint="api_save_workspace")
def api_save_workspace():
    """Save current workspace to saved notes in dashboard, including ALL artifacts."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    content = (data.get("content") or "").strip()
    
    if not content:
        return jsonify({"ok": False, "error": "No content to save"}), 400
    
    notes_file = current_app.config.get("NOTES_FILE", "saved_notes.json")
    
    try:
        # Load existing notes
        try:
            with open(notes_file, "r", encoding="utf-8") as f:
                notes = json.load(f)
        except Exception:
            notes = []
        
        # Check if note with this title already exists
        existing_idx = None
        for i, note in enumerate(notes):
            if note.get("title", "").strip().lower() == title.lower():
                existing_idx = i
                break
        
        # Get source text from current workspace
        source_text = _get_source_text()
        
        # Collect all workspace artifacts
        quiz_data = _read_json("quiz.json")
        flash_data = _read_json("flash.json")
        podcast_text = _read_text("podcast.txt")
        
        # Prepare note data with all artifacts
        note_data = {
            "title": title,
            "content": content,
            "source": source_text,
            "quiz": quiz_data,
            "flashcards": flash_data,
            "podcast": podcast_text,
            "timestamp": time.time()
        }
        
        # Update or append
        if existing_idx is not None:
            notes[existing_idx] = note_data
        else:
            notes.append(note_data)
        
        # Save back to file
        with open(notes_file, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        return jsonify({"ok": True, "message": "Workspace saved successfully"})
    
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# SERVE PODCAST AUDIO
@workspace_bp.get("/podcast_audio/<ws_key>", endpoint="serve_podcast_audio")
def serve_podcast_audio(ws_key: str):
    """Serve the generated podcast audio file."""
    # Validate workspace key
    ws_path = os.path.join(WS_STORE, ws_key)
    if not os.path.exists(ws_path):
        return jsonify({"error": "Workspace not found"}), 404
    
    audio_path = os.path.join(ws_path, "podcast_audio.mp3")
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 404
    
    return send_file(audio_path, mimetype="audio/mpeg")

