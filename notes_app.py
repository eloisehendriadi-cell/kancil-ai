# the_ai_tutor/notes_app.py
from __future__ import annotations

import os
import re
import io
import json
import traceback
from typing import List

import markdown
import requests
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

try:
    # Prefer pypdf if present; fall back to PyPDF2
    import pypdf as _Pdf
except Exception:  # pragma: no cover
    try:
        import PyPDF2 as _Pdf
    except Exception:
        _Pdf = None  # handled gracefully later

# -----------------------------
# Blueprint
# -----------------------------
notes_bp = Blueprint("notes", __name__, template_folder="templates")

# -----------------------------
# Ollama config
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma:2b")

DEFAULT_OPTIONS = {
    "temperature": 0.12,
    "top_p": 0.9,
    "repeat_penalty": 1.15,
    "num_ctx": 3072,
    "num_predict": 1200,
    "num_thread": os.cpu_count() or 4,
}
FAST_OPTIONS = {**DEFAULT_OPTIONS, "num_predict": 750}

UPLOAD_FOLDER = "uploads"
NOTES_FILE = "saved_notes.json"
ALLOWED_EXT = {"txt", "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(NOTES_FILE):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# Hard caps to avoid long-running parses
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "80"))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", "25000000"))  # 25MB
MIN_PAGE_CHARS = 80

# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Lightweight, defensive PDF text extraction.
    Never raises; returns best-effort text.
    """
    if _Pdf is None:
        return ""
    try:
        with open(pdf_path, "rb") as fh:
            data = fh.read()
        if len(data) > MAX_PDF_BYTES:
            return ""
        reader = _Pdf.PdfReader(io.BytesIO(data)) if hasattr(_Pdf, "PdfReader") else _Pdf.PdfFileReader(io.BytesIO(data))
        pages = []
        total = len(reader.pages) if hasattr(reader, "pages") else reader.getNumPages()
        n = min(total, MAX_PDF_PAGES)
        for i in range(n):
            try:
                page = reader.pages[i] if hasattr(reader, "pages") else reader.getPage(i)
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if len(txt.strip()) >= MIN_PAGE_CHARS:
                pages.append(txt)
        return "\n".join(pages)
    except Exception:
        return ""


def query_ollama_chat(messages, stream: bool = False, fast: bool = True, timeout: int = 120) -> str:
    options = FAST_OPTIONS if fast else DEFAULT_OPTIONS
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": stream,
                "options": options,
                "keep_alive": "10m",
            },
            timeout=None if stream else timeout,
            stream=stream,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()
    except Exception as e:
        return f"Error contacting Ollama: {e}"


def query_ollama(prompt: str, **kwargs) -> str:
    return query_ollama_chat([{"role": "user", "content": prompt}], **kwargs)

# -----------------------------
# Styling & highlight
# -----------------------------
HI_OPEN, HI_CLOSE = "<<hi>>", "<</hi>>"

def _to_html_with_highlights(md_text: str) -> str:
    def hi_repl(m):
        inner = m.group(1)
        return f'<mark class="hi-imp">{inner}</mark>'

    md_text = re.sub(
        re.escape(HI_OPEN) + r"(.*?)" + re.escape(HI_CLOSE),
        hi_repl,
        md_text,
        flags=re.S,
    )

    def clarify_bubbles(text: str) -> str:
        # clarify “bubbles” → “oxygen bubbles” when not already nearby
        return re.sub(r'(?i)\b(bubbles)\b(?![^<]{0,20}oxygen)', r'oxygen \1', text)

    md_text = clarify_bubbles(md_text)

    html = markdown.markdown(
        md_text,
        extensions=["extra", "sane_lists", "tables", "nl2br"],
        output_format="xhtml",
    )
    style = """
<style>
  .hi-imp { background: #fff3a6; padding: 0 .15em; border-radius: .25rem }
  table { border-collapse: collapse; width: 100%; background: #fff }
  th, td { border: 1px solid #e5e7eb; padding: 8px 10px; vertical-align: top }
  thead th { background: #f8fafc; }
  h1, h2, h3 { margin-top: 1.1rem; }
</style>
"""
    return style + html

# -----------------------------
# Cleaning & guardrails
# -----------------------------
NOISE_LINES = [
    r"^©\s*\d{4}\b",
    r"cambridge assessment", r"marshall cavendish",
    r"student'?s book", r"powerpoint slides",
    r"this (presentation|content) .* (been through|endorsement)",
    r"^sections?:\s*$", r"^section\s+\d+",
    r"^table\s*:?$", r"^figure\s*\d+",
    r"^\s*•\s*$", r"^\s*-\s*$",
]

STRIP_INLINE = [r"\s*•\s*", r"\s+–\s+"]
QUESTIONY = [
    r"(?i)\bquestion\s*\d+\b",
    r"(?i)\bwrite\b.*\bequation\b",
    r"(?i)\bidentify\b.*\b(label|layers?)\b",
    r"(?i)\bexplain how\b",
]
STOPWORDS = set("""
a an and are as at be but by for from has have if in into is it its of on or such that the their there this to was were will with you your
""".split())

def sanitize_source_text(text: str) -> str:
    lines, keep = text.splitlines(), []
    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        if any(re.search(p, raw, flags=re.I) for p in NOISE_LINES):
            continue
        keep.append(raw)
    cleaned = "\n".join(keep)
    for pat in STRIP_INLINE:
        cleaned = re.sub(pat, " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"(?i)\bmind\s*map\b", "concept overview", cleaned)
    return cleaned.strip()

# ---------- Title inference ----------
def top_keywords(text: str, k: int = 16) -> List[str]:
    counts = {}
    for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower()):
        if w in STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[:k]]


def infer_topic_title(text: str, fallback: str = "Study Notes") -> str:
    BAD = re.compile(r"(?i)(cambridge|marshall|copyright|slides|sections|figure|table)")
    candidates = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t or BAD.search(t) or len(t) < 6:
            continue
        score = 0
        if t.isupper():
            score += 3
        if t.istitle():
            score += 2
        if len(t) <= 80:
            score += 1
        candidates.append((score, t))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        title = candidates[0][1]
        return re.sub(r"^[IVX]+\.\s*", "", title)[:120]
    kws = top_keywords(text, 6)
    if kws:
        return " ".join(w.capitalize() for w in kws[:3])
    return fallback


def output_drifted(md: str, kws: List[str], min_hits: int = 3) -> bool:
    if not kws:
        return False
    text = (md or "").lower()
    hits = sum(1 for w in kws[:8] if w in text)
    return hits < min_hits


def looks_like_meta_or_exam(md: str) -> bool:
    return any(re.search(p, md or "") for p in QUESTIONY) or bool(re.search(r"(?i)\bthis (presentation|document) covers\b", md or ""))


def violates_topic(md: str, topic: str, kws: List[str]) -> bool:
    text = (md or "").lower()
    must = set(w.lower() for w in kws[:6])
    hits = sum(1 for w in must if w in text)
    head = topic.split(":")[0].split("-")[0].strip().lower()
    head_word = re.split(r"\W+", head)[0] if head else ""
    need_topic = bool(head_word) and head_word not in text
    return hits < 2 or need_topic


def ensure_min_highlights(md: str, min_count: int = 6) -> str:
    count = len(re.findall(re.escape(HI_OPEN), md or ""))
    if count >= min_count:
        return md
    used = [0]

    def wrap(m):
        used[0] += 1
        if used[0] <= (min_count - count):
            return f"{HI_OPEN}{m.group(1)}{HI_CLOSE}"
        return m.group(0)

    return re.sub(r"\*\*(.+?)\*\*", wrap, md or "")


def enforce_emoji_headers(md: str) -> str:
    def repl_h1(m): return f"# 🌿 {m.group(1)}"
    md = re.sub(r"(?im)^\s*#\s*(.+)$", repl_h1, md, count=1)
    upgrades = [
        (r"(?im)^\s*##\s*Topic Overview",           "🧠 Topic Overview"),
        (r"(?im)^\s*##\s*Key Concepts",             "🔑 Key Concepts"),
        (r"(?im)^\s*##\s*Processes / Mechanisms",   "🧬 Processes / Mechanisms"),
        (r"(?im)^\s*##\s*Comparisons & Tables",     "📊 Comparisons & Tables"),
        (r"(?im)^\s*##\s*Applications / Examples",  "💡 Applications / Examples"),
        (r"(?im)^\s*##\s*Misconceptions & Pitfalls","⚠️ Misconceptions & Pitfalls"),
        (r"(?im)^\s*##\s*Mini-Quiz.*",              "✍️ Mini-Quiz"),
    ]
    for pat, repl in upgrades:
        md = re.sub(pat, f"## {repl}", md)
    return md

# -----------------------------
# Prompting
# -----------------------------
STYLEGUIDE_SYSTEM = f"""
You create **exam-ready, text-only study notes** from the user's source.
Hard rules:
- NEVER refuse; always produce notes from the given text.
- Use ONLY facts from the source. If a detail is missing, write *Not stated in source*.
- Use emoji section headers (>=5 distinct emojis across the doc).
- Write **220–400 words** of bullets before the first table.
- Include at least one markdown table AND a **Cause → Effect (with evidence)** table when relevant.
- Use {HI_OPEN}…{HI_CLOSE} on 6–10 critical terms (verbatim).
- Do not output “Sections:”, “This presentation covers…”, or “Question 1”.
Elaboration rubric (apply only when consistent with the source):
- Describe **how** each factor changes the outcome (direction + brief reason).
- In plant photosynthesis experiments, “bubbles” refers to **oxygen bubbles** unless contradicted by the text.
""".strip()


def build_messages(topic: str, clean_source: str, kw_line: str):
    user_prompt = f"""
TITLE (do not say 'mind map'): {topic}

Write **concise, faithful notes** that stay STRICTLY within this source.
If the source includes multiple chapters, summarise ONLY the part that matches
the TITLE and the TOPIC-LOCK KEYWORDS below.

TOPIC-LOCK KEYWORDS (anchor to these terms; avoid unrelated themes):
{kw_line}

# 🧠 Topic Overview
- 5–8 bullets summarising the big picture (verbatim facts; crisp).

## 🔑 Key Concepts
- Bold important **terms** with one-line meanings.

## 🧬 Processes / Mechanisms
1. Step-by-step sequence as described.
2. If experiments are included, mention the **measurement** only (no extra speculation).

## 📊 Comparisons & Tables
- Include at least one term-definition table.
- If affecting/limiting factors are present, also include:
| Factor | Cause → Effect (how outcome changes) | Evidence / Example |
| --- | --- | --- |

## 💡 Applications / Examples
- Short source-backed bullets only.

## ⚠️ Misconceptions & Pitfalls
- Clarify typical confusions. If none, write “Not stated in source”.

## ✍️ Mini-Quiz (3 quick checks)
- Q: …
  - Answer: …

SOURCE (cleaned; summarise faithfully; do not hallucinate):
{clean_source}
""".strip()

    return [
        {"role": "system", "content": STYLEGUIDE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

# -----------------------------
# Summariser
# -----------------------------
def summarize_text(text: str) -> str:
    clean = sanitize_source_text(text)
    topic = infer_topic_title(clean, "Study Notes")

    if len(clean) < 220:
        minimal = f"""
# 🧠 {topic}
- Source text is brief; notes below reflect only the explicit content provided.
- For fuller notes, provide more text.

## 🔑 Key Concepts
- {HI_OPEN}Key ideas{HI_CLOSE}: Minimal, based on provided lines.

## 📊 Comparisons & Tables
| Item | What the source states |
| --- | --- |
| Coverage | Limited; extend source for details |
"""
        return _to_html_with_highlights(minimal)

    kws = top_keywords(clean, k=16)
    kw_line = ", ".join(kws[:14])

    msgs = build_messages(topic, clean, kw_line)
    md = query_ollama_chat(msgs, fast=True)

    if looks_like_meta_or_exam(md) or output_drifted(md, kws, 3) or len(md or "") < 200:
        msgs.append({"role": "user", "content":
            "Regenerate **strictly ON-TOPIC**. Follow the TITLE and TOPIC-LOCK KEYWORDS. "
            "Ensure 220–400 words before the first table and include one table."
        })
        md = query_ollama_chat(msgs, fast=True)

    if violates_topic(md, topic, kws):
        anchor = ", ".join(kws[:8])
        msgs = build_messages(f"{topic} — focus strictly on: {anchor}", clean, kw_line)
        md = query_ollama_chat(msgs, fast=True)

    md = ensure_min_highlights(md or "", min_count=6)
    md = enforce_emoji_headers(md or "")
    md = re.sub(r"(?i)\bmind\s*map\b", "overview", md or "")

    return _to_html_with_highlights(md or "")

# -----------------------------
# Routes (classic page)
# -----------------------------
@notes_bp.route("/notes", methods=["GET", "POST"])
def notes():
    # (kept same behavior)
    output = None
    if request.method == "POST":
        method = request.form.get("method")
        text = (request.form.get("note_input") or "").strip()
        file = request.files.get("note_pdf")

        if method == "text" and text:
            output = summarize_text(text)

        elif method == "pdf" and file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            pdf_text = extract_text_from_pdf(path)

            if len(pdf_text.strip()) < 220:
                output = _to_html_with_highlights("""
# 📄 Couldn’t read enough text from the PDF
- The extracted text is very short/noisy.
- Try a **text-based PDF** (not scanned) or paste the text directly.
""")
            else:
                output = summarize_text(pdf_text)

        elif method == "youtube":
            yt = (request.form.get("note_youtube") or "").strip()
            if yt:
                msgs = [
                    {"role": "system", "content": STYLEGUIDE_SYSTEM},
                    {"role": "user", "content": f"""
Summarise this YouTube transcript into exam-ready, text-only notes.
- Emoji headers (5+), 220–400 words before first table,
- At least one table + **Cause → Effect (with evidence)** table when relevant,
- {HI_OPEN}…{HI_CLOSE} highlights (6–10),
- No meta “Sections:”/“This presentation covers…”, no “Question 1…”.

URL: {yt}
(Assume transcript text is available; be conservative and avoid speculation.)
""".strip()}
                ]
                md = query_ollama_chat(msgs, fast=True)
                if looks_like_meta_or_exam(md) or len(md or "") < 200:
                    msgs.append({"role": "user", "content": "Regenerate strictly on-topic with one table."})
                    md = query_ollama_chat(msgs, fast=True)
                md = ensure_min_highlights(md or "", min_count=6)
                md = enforce_emoji_headers(md or "")
                output = _to_html_with_highlights(md or "")

    return render_template("note_generator.html", output=output)

# -----------------------------
# FAST EXTRACT (JSON)
# -----------------------------
@notes_bp.route("/notes/extract_pdf", methods=["POST"])
def notes_extract_pdf():
    """
    Quickly extract raw text from an uploaded PDF and return a title.
    Returns JSON: { ok, source, title, warning? } — always 200.
    """
    try:
        f = request.files.get("note_pdf")
        if not f or not f.filename.lower().endswith(".pdf"):
            return jsonify({"ok": False, "error": "Please choose a .pdf file."}), 200

        safe_name = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, safe_name)
        f.save(path)

        source = extract_text_from_pdf(path) or ""
        title = os.path.splitext(safe_name)[0] or "Study Notes"
        if not source.strip():
            return jsonify({"ok": False, "title": title, "source": "", "warning": "No extractable text (scanned or image-only PDF)."}), 200

        return jsonify({"ok": True, "source": source, "title": title}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "Extract failed", "detail": str(e), "trace": traceback.format_exc()[-400:]}), 200

# -----------------------------
# XHR endpoint for inline dashboard generation
# -----------------------------
@notes_bp.route("/notes/xhr", methods=["POST"])
def notes_xhr():
    """
    Generate notes without leaving the Dashboard (AJAX).
    Returns JSON: { ok, html?, source?, title?, error?, warning? } — always 200.
    """
    try:
        method = (request.form.get("method") or "").strip().lower()
        if method not in {"text", "pdf", "youtube"}:
            return jsonify({"ok": False, "error": "Unsupported method"}), 200

        # From text
        if method == "text":
            source = (request.form.get("note_input") or "").strip()
            if not source:
                return jsonify({"ok": False, "error": "Please paste some text."}), 200
            title = infer_topic_title(sanitize_source_text(source), "Study Notes")
            html = summarize_text(source)
            return jsonify({"ok": True, "html": html, "source": source, "title": title}), 200

        # From PDF — return PDF filename as title
        if method == "pdf":
            f = request.files.get("note_pdf")
            if not f or not f.filename.lower().endswith(".pdf"):
                return jsonify({"ok": False, "error": "Please choose a .pdf file."}), 200

            safe_name = secure_filename(f.filename)
            path = os.path.join(UPLOAD_FOLDER, safe_name)
            f.save(path)

            source = extract_text_from_pdf(path) or ""
            title_from_filename = os.path.splitext(safe_name)[0] or "Study Notes"

            if len(source.strip()) < 220:
                html = _to_html_with_highlights("""
# 📄 Couldn’t read enough text from the PDF
- The extracted text is very short/noisy.
- Try a **text-based PDF** (not scanned) or paste the text directly.
""")
                return jsonify({
                    "ok": True,
                    "html": html,
                    "source": source,
                    "title": title_from_filename
                }), 200

            html = summarize_text(source)
            return jsonify({
                "ok": True,
                "html": html,
                "source": source,
                "title": title_from_filename
            }), 200

        # From YouTube
        if method == "youtube":
            yt = (request.form.get("note_youtube") or "").strip()
            if not yt:
                return jsonify({"ok": False, "error": "Paste a YouTube URL."}), 200

            msgs = [
                {"role": "system", "content": STYLEGUIDE_SYSTEM},
                {"role": "user", "content": f"""
Summarise this YouTube transcript into exam-ready, text-only notes.
- Emoji headers (5+), 220–400 words before first table,
- At least one table + **Cause → Effect (with evidence)** table when relevant,
- {HI_OPEN}…{HI_CLOSE} highlights (6–10),
- No meta “Sections:”/“This presentation covers…”, no “Question 1…”.

URL: {yt}
(Assume transcript text is available; be conservative and avoid speculation.)
""".strip()}
            ]
            md = query_ollama_chat(msgs, fast=True)
            if looks_like_meta_or_exam(md) or len(md or "") < 200:
                msgs.append({"role": "user", "content": "Regenerate strictly on-topic with one table."})
                md = query_ollama_chat(msgs, fast=True)

            md = ensure_min_highlights(md or "", min_count=6)
            md = enforce_emoji_headers(md or "")
            html = _to_html_with_highlights(md or "")
            title = "YouTube Notes"
            source = f"URL: {yt}"
            return jsonify({"ok": True, "html": html, "source": source, "title": title}), 200

        return jsonify({"ok": False, "error": "Invalid request."}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": "Generation failed", "detail": str(e), "trace": traceback.format_exc()[-400:]}), 200

# -----------------------------
# Save note (append-only, kept)
# -----------------------------
@notes_bp.route("/save_note", methods=["POST"])
def save_note():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()
    if not title or not content:
        return jsonify({"error": "Missing title or content"}), 400

    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            notes = json.load(f)
    except Exception:
        notes = []

    notes.append({"title": title, "content": content})
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True})

# ==============================
# Saved Notes CRUD (NEW)
# ==============================
def _load_notes():
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []

def _write_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def _find_idx_by_title(notes, title: str) -> int:
    tnorm = (title or "").strip().lower()
    return next((i for i, n in enumerate(notes)
                 if (n.get("title", "").strip().lower() == tnorm)), -1)

@notes_bp.route("/notes/list", methods=["GET"])
def list_notes():
    notes = _load_notes()
    return jsonify({
        "ok": True,
        "notes": [
            {"title": (n.get("title") or "Untitled"), "size": len(n.get("content") or "")}
            for n in notes
        ]
    })

@notes_bp.route("/notes/read", methods=["GET"])
def read_note():
    title = (request.args.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Missing title"}), 400
    notes = _load_notes()
    i = _find_idx_by_title(notes, title)
    if i == -1:
        return jsonify({"ok": False, "error": "Note not found"}), 404
    return jsonify({"ok": True, "title": notes[i].get("title") or "Untitled",
                    "content": notes[i].get("content") or ""})

@notes_bp.route("/notes/rename", methods=["POST"])
def rename_note():
    data = request.get_json(silent=True) or {}
    old_title = (data.get("old_title") or "").strip()
    new_title = (data.get("new_title") or "").strip()
    if not old_title or not new_title:
        return jsonify({"ok": False, "error": "Missing old_title or new_title"}), 400

    notes = _load_notes()
    i = _find_idx_by_title(notes, old_title)
    if i == -1:
        return jsonify({"ok": False, "error": "Note not found"}), 404

    if _find_idx_by_title(notes, new_title) != -1:
        return jsonify({"ok": False, "error": "A note with that title already exists"}), 400

    notes[i]["title"] = new_title
    _write_notes(notes)
    return jsonify({"ok": True})

@notes_bp.route("/notes/delete", methods=["POST"])
def delete_note():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Missing title"}), 400

    notes = _load_notes()
    i = _find_idx_by_title(notes, title)
    if i == -1:
        return jsonify({"ok": False, "error": "Note not found"}), 404

    notes.pop(i)
    _write_notes(notes)
    return jsonify({"ok": True})

@notes_bp.route("/notes/save_or_update", methods=["POST"])
def save_or_update_note():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip()
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"ok": False, "error": "Missing content"}), 400

    notes = _load_notes()
    i = _find_idx_by_title(notes, title)
    if i == -1:
        notes.append({"title": title, "content": content})
    else:
        notes[i]["content"] = content
    _write_notes(notes)
    return jsonify({"ok": True})

