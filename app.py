# app.py
from __future__ import annotations

import os
import json
import importlib
from flask import Flask, render_template, send_from_directory, redirect

# ---------- writable paths (Render-safe) ----------
def _writable_base() -> str:
    """
    Use the repo directory locally; fall back to /tmp on hosts where
    the repo FS is read-only (e.g., Render). You can override with WRITE_BASE.
    """
    return "." if os.access(".", os.W_OK) else "/tmp"

BASE = os.getenv("WRITE_BASE", _writable_base())

UPLOAD_FOLDER = os.path.join(BASE, "uploads")
NOTES_FILE    = os.path.join(BASE, "saved_notes.json")
CONVOS_DIR    = os.path.join(BASE, "convos")
INDEX_FILE    = os.path.join(CONVOS_DIR, "index.json")


def ensure_dirs() -> None:
    """Create required folders/files if missing (under the writable BASE)."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(CONVOS_DIR, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, "podcasts"), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, "images"), exist_ok=True)

    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def _try_import(*candidates):
    """
    Try importing the first available dotted module + attribute.
    candidates: tuples like ("the_ai_tutor.notes_app", "notes_bp")
    Returns the attribute or None.
    """
    for mod_path, attr in candidates:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr, None)
            if obj is not None:
                return obj
        except Exception:
            continue
    return None


def create_app() -> Flask:
    # Templates live under the package folder
    app = Flask(
        __name__,
        template_folder="the_ai_tutor/templates",
        static_folder="static",
    )
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    app.url_map.strict_slashes = False

    ensure_dirs()

    # Make paths available to blueprints
    app.config.update(
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        NOTES_FILE=NOTES_FILE,
        CONVOS_DIR=CONVOS_DIR,
    )

    # ---------- Home + Dashboard ----------
    @app.route("/", endpoint="index")
    def index():
        return render_template("index.html")

    @app.route("/dashboard", endpoint="dashboard")
    def dashboard():
        try:
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                notes = json.load(f)
        except Exception:
            notes = []
        return render_template("dashboard.html", notes=notes, title="Dashboard")

    @app.route("/past-papers", endpoint="past_papers")
    def past_papers():
        return render_template("past_papers.html", title="Past Papers")

    @app.route("/uploads/<path:filename>", endpoint="uploaded_file")
    def uploaded_file(filename: str):
        # works for nested paths like "podcasts/xxx.mp3" or "images/foo.png"
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    # ---------- Register feature blueprints ----------
    # Chat / Chatbot
    chat_bp = _try_import(
        ("the_ai_tutor.chat_app", "chat_bp"),
        ("chat_app", "chat_bp"),
        ("the_ai_tutor.chatbot_app", "chatbot_bp"),  # some modules name it chatbot_bp
        ("chatbot_app", "chatbot_bp"),
        ("the_ai_tutor.chatbot_app", "chat_bp"),
        ("chatbot_app", "chat_bp"),
    )

    # Notes
    notes_bp = _try_import(
        ("the_ai_tutor.notes_app", "notes_bp"),
        ("notes_app", "notes_bp"),
    )

    # Quiz
    quiz_bp = _try_import(
        ("the_ai_tutor.quiz_app", "quiz_bp"),
        ("quiz_app", "quiz_bp"),
    )

    # Podcast
    podcast_bp = _try_import(
        ("the_ai_tutor.podcast_app", "podcast_bp"),
        ("podcast_app", "podcast_bp"),
    )

    # Workspace
    workspace_bp = _try_import(
        ("the_ai_tutor.workspace", "workspace_bp"),
        ("workspace", "workspace_bp"),
    )

    # Register whichever exist (print clear warnings)
    if notes_bp:
        app.register_blueprint(notes_bp)
    else:
        print("[WARN] Notes blueprint not found (notes_app).")

    if quiz_bp:
        app.register_blueprint(quiz_bp)
    else:
        print("[WARN] Quiz blueprint not found (quiz_app).")

    if podcast_bp:
        app.register_blueprint(podcast_bp)
    else:
        print("[WARN] Podcast blueprint not found (podcast_app).")

    if workspace_bp:
        # explicit prefix to keep URLs tidy
        app.register_blueprint(workspace_bp, url_prefix="/workspace")
    else:
        print("[WARN] Workspace blueprint not found. Endpoints under /workspace/* will 404.")

    if chat_bp:
        # Canonical path for chatbot UI + APIs
        app.register_blueprint(chat_bp, url_prefix="/chatbot")

        # ---- Compatibility redirects: /chat → /chatbot ----
        @app.route("/chat")
        def _chat_root_redirect():
            return redirect("/chatbot", code=301)

        @app.route("/chat/<path:rest>")
        def _chat_wildcard_redirect(rest: str):
            # Preserve any subpath the frontend tries to call
            return redirect(f"/chatbot/{rest}", code=307)
        # ---------------------------------------------------
    else:
        print("[WARN] Chat blueprint not found (chat_app/chatbot_app).")

    # --- DEBUG: list routes once on boot ---
    with app.app_context():
        print("\n=== ROUTES ===")
        for r in sorted(app.url_map.iter_rules(), key=lambda x: str(x)):
            print(f"{r.endpoint:35s} -> {r}")
        print("==============\n")

    return app


if __name__ == "__main__":
    app = create_app()
    # Move off 5000 on macOS; keep single PID
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5050)), debug=True)

