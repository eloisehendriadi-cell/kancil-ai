# the_ai_tutor/chat_app.py
from __future__ import annotations
from typing import List, Dict
from flask import Blueprint, render_template, request, jsonify, session

from llm_client import chat_text, get_chat_sessions, get_chat, rename_chat_session

chat_bp = Blueprint("chat", __name__, template_folder="templates")

@chat_bp.get("/", endpoint="chat_ui")
def chat_ui():
    return render_template("chat.html", title="Chatbot")

def _chat(messages: List[Dict[str, str]]) -> str:
    # one call; llm_client handles model/env
    out = chat_text(messages, temperature=0.25, max_tokens=700)
    return out or "Sorry—no model provider is configured or reachable right now."

@chat_bp.post("/send", endpoint="chat_send")
def chat_send():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    history = data.get("history") or []
    chat_id = data.get("chat_id")
    chat_name = data.get("chat_name")

    if not text:
        return jsonify({"ok": False, "error": "empty message"}), 400

    msgs: List[Dict[str, str]] = []
    for m in history[-12:]:
        r, c = (m.get("role") or "user"), (m.get("content") or "")
        if r in ("user", "assistant") and c:
            msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": text})

    reply = chat_text(msgs, chat_id=chat_id, chat_name=chat_name)
    # chat_text returns assistant message and saves history
    # Get updated chat_id and chat_name from history
    if not chat_id:
        # If new, get the latest session
        sessions = get_chat_sessions()
        if sessions:
            chat_id = sessions[-1]["id"]
            chat_name = sessions[-1]["name"]
    return jsonify({"ok": True, "reply": reply, "chat_id": chat_id, "chat_name": chat_name})

# List all chat sessions
@chat_bp.get("/sessions")
def chat_sessions():
    sessions = get_chat_sessions()
    return jsonify(sessions)

# Load a specific chat session
@chat_bp.get("/session/<chat_id>")
def chat_session(chat_id):
    chat = get_chat(chat_id)
    return jsonify({"name": chat.get("name", chat_id), "messages": chat.get("messages", [])})

# Rename a chat session
@chat_bp.post("/session/<chat_id>/rename")
def chat_rename(chat_id):
    data = request.get_json(silent=True) or {}
    new_name = (data.get("new_name") or "").strip()
    if not new_name:
        return jsonify({"ok": False, "error": "No new name provided"}), 400
    rename_chat_session(chat_id, new_name)
    return jsonify({"ok": True})

