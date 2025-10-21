from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import os
from werkzeug.utils import secure_filename

# Reuse your existing helpers
from notes_app import summarize_text, extract_text_from_pdf, allowed_file
from quiz_app import _generate as generate_quiz_items  # internal generator already returns items

dashboard_bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

# Where to stash PDF uploads (temporary; we only keep text in session)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Helpers -------------------------------------------------
def _ensure_source():
    """Return (text, error) from session."""
    txt = (session.get("shared_source_text") or "").strip()
    return (txt, None) if txt else ("", "No source selected yet. Upload a PDF or paste text in Dashboard.")

# ---- Pages ---------------------------------------------------
@dashboard_bp.route("/")
def home():
    return render_template("home.html")

@dashboard_bp.route("/past-papers")
def past_papers():
    # Placeholder content per your request
    return render_template("past_papers.html")

@dashboard_bp.route("/chatbot")
def chatbot():
    # Simple placeholder; hook your bot here later
    return render_template("chatbot_placeholder.html")

@dashboard_bp.route("/dashboard")
def dashboard():
    # “Note dashboard” chooser page
    return render_template("note_dashboard.html")

@dashboard_bp.route("/workspace")
def workspace():
    # Left tabs (Notes | Podcast | Quiz | Flashcards)
    return render_template("workspace.html")

# ---- Source setters (shared for all tools) -------------------
@dashboard_bp.route("/api/set_source_text", methods=["POST"])
def set_source_text():
    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Text is empty."}), 400
    session["shared_source_text"] = text
    session["shared_source_title"] = (request.form.get("title") or "Untitled source").strip()
    return jsonify({"ok": True})

@dashboard_bp.route("/api/set_source_youtube", methods=["POST"])
def set_source_youtube():
    url = (request.form.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "YouTube URL is empty."}), 400
    # We don’t fetch transcripts here; we store the URL and ask notes generator to assume transcript (like your notes_app does)
    session["shared_source_text"] = f"[YOUTUBE_URL]\n{url}"
    session["shared_source_title"] = "YouTube Source"
    return jsonify({"ok": True})

@dashboard_bp.route("/api/set_source_pdf", methods=["POST"])
def set_source_pdf():
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file received."}), 400
    if not allowed_file(f.filename):
        return jsonify({"ok": False, "error": "Unsupported file type."}), 400
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)
    try:
        text = extract_text_from_pdf(path)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to read PDF: {e}"}), 500
    text = (text or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Couldn’t extract text. Try a text-based (not scanned) PDF."}), 400
    session["shared_source_text"] = text
    session["shared_source_title"] = os.path.splitext(os.path.basename(path))[0]
    return jsonify({"ok": True})

# ---- Generators (use shared source) --------------------------
@dashboard_bp.route("/api/generate/notes", methods=["POST"])
def api_generate_notes():
    src, err = _ensure_source()
    if err:
        return jsonify({"ok": False, "error": err}), 400

    # if source was YouTube URL, call the YouTube path in notes_app behavior
    if src.startswith("[YOUTUBE_URL]"):
        # Reuse the YouTube logic from notes_app (system prompt there assumes transcript available).
        # Quick shim: just send the URL string into the same summarize_text (which handles guardrails).
        # Optional: you could call the yt-special handler you already wrote.
        url = src.splitlines()[-1].strip()
        # Light wrapper prompt so summarize_text treats it as plain text context
        html = summarize_text(f"YouTube URL: {url}\n(Assume transcript available; conservative details.)")
    else:
        html = summarize_text(src)

    return jsonify({"ok": True, "html": html})

@dashboard_bp.route("/api/generate/quiz", methods=["POST"])
def api_generate_quiz():
    src, err = _ensure_source()
    if err:
        return jsonify({"ok": False, "error": err}), 400

    title = (session.get("shared_source_title") or "Key Concepts").strip()
    items = generate_quiz_items(title, src, target=12)
    if not items:
        return jsonify({"ok": False, "error": "Unable to generate quiz items."}), 500
    return jsonify({"ok": True, "items": items})

@dashboard_bp.route("/api/generate/podcast", methods=["POST"])
def api_generate_podcast():
    # Simple, solid “podcast script” from notes text (no audio). You can swap in your TTS later.
    from notes_app import query_ollama_chat  # reuse your Ollama caller
    src, err = _ensure_source()
    if err:
        return jsonify({"ok": False, "error": err}), 400

    system = "You are a friendly science host. Write a short, engaging 3–4 minute podcast script from the user notes. No hallucinations."
    user = (
        "Turn the following notes into a podcast script with a host and short sections. "
        "Keep it accurate, crisp, and teen-friendly. Use playful emojis for section headers.\n\n"
        f"NOTES:\n{src}"
    )
    script = query_ollama_chat([{"role": "system", "content": system},
                                {"role": "user", "content": user}], fast=True)
    return jsonify({"ok": True, "script": script})

@dashboard_bp.route("/api/generate/flashcards", methods=["POST"])
def api_generate_flashcards():
    # Simple deterministic flashcards from the same mined facts as quiz
    from quiz_app import _mine_facts  # internal but perfect for flashcards
    src, err = _ensure_source()
    if err:
        return jsonify({"ok": False, "error": err}), 400

    facts = _mine_facts(src)
    cards = []
    for f in facts[:24]:
        q = f"{f['term']}"
        a = f"{f['fact']}"
        cards.append({"front": q, "back": a})
    if not cards:
        cards = [{"front": "Photosynthesis", "back": "Process making glucose using light, CO₂, and water."}]
    return jsonify({"ok": True, "cards": cards})

