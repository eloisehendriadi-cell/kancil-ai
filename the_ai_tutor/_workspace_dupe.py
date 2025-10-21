# the_ai_tutor/workspace.py
from flask import Blueprint, request, jsonify, session, render_template
import json, os

workspace_bp = Blueprint("workspace", __name__, url_prefix="/workspace")

NOTES_FILE = "saved_notes.json"

@workspace_bp.route("/", methods=["GET"])
def open_workspace():
    # Renders the shared Workspace UI
    return render_template("workspace.html", title="Workspace")

@workspace_bp.route("/use_saved", methods=["POST"])
def use_saved():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Missing note title"}), 400

    # Load notes
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            notes = json.load(f)
    except Exception:
        notes = []

    # Find note by title (case-insensitive)
    match = next((n for n in notes if (n.get("title","").strip().lower() == title.lower())), None)
    if not match:
        return jsonify({"ok": False, "error": "Note not found"}), 404

    # Arm the Workspace shared source
    session["shared_source_type"] = "saved_note"
    session["shared_source_title"] = match.get("title", title)
    session["shared_source_payload"] = match.get("content", "")

    return jsonify({"ok": True})

