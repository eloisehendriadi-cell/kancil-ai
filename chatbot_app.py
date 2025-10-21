# the_ai_tutor/chatbot_app.py
from __future__ import annotations

import os
import re
import io
import json
import uuid
import datetime as dt
from typing import Dict, List, Generator, Any

import requests
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    Response,
    stream_with_context,
    send_file,
)

# -----------------------------------------------------------------------------
# Blueprint (mounted as /chatbot by app.py)
# -----------------------------------------------------------------------------
chat_bp = Blueprint("chat", __name__, template_folder="templates", url_prefix="/chatbot")

# -----------------------------------------------------------------------------
# Ollama config
# -----------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")  # small & quick; override via env

DEFAULT_OPTIONS = {
    "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.2")),
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": int(os.getenv("CHAT_NUM_CTX", "2048")),
    "num_predict": int(os.getenv("CHAT_NUM_PREDICT", "320")),
    "num_thread": os.cpu_count() or 4,
}
FAST_OPTIONS = {**DEFAULT_OPTIONS, "num_predict": min(192, DEFAULT_OPTIONS["num_predict"])}

# -----------------------------------------------------------------------------
# Local storage (JSON files)
# -----------------------------------------------------------------------------
CONVOS_DIR = "convos"
INDEX_FILE = os.path.join(CONVOS_DIR, "index.json")
os.makedirs(CONVOS_DIR, exist_ok=True)
if not os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _cpath(cid: str) -> str:
    return os.path.join(CONVOS_DIR, f"{cid}.json")


def _index_load() -> List[Dict[str, Any]]:
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _index_save(items: List[Dict[str, Any]]) -> None:
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _convo_load(cid: str) -> Dict[str, Any] | None:
    p = _cpath(cid)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _convo_save(convo: Dict[str, Any]) -> None:
    # write convo
    with open(_cpath(convo["id"]), "w", encoding="utf-8") as f:
        json.dump(convo, f, ensure_ascii=False, indent=2)

    # update index
    last = ""
    for m in reversed(convo.get("messages", [])):
        if m.get("role") == "user":
            last = (m.get("content") or "")[:120]
            break

    idx = _index_load()
    found = False
    for it in idx:
        if it["id"] == convo["id"]:
            it.update(
                {
                    "title": convo.get("title", "Chat"),
                    "updated_at": _now_iso(),
                    "last_message": last,
                }
            )
            found = True
            break
    if not found:
        idx.append(
            {
                "id": convo["id"],
                "title": convo.get("title", "Chat"),
                "created_at": convo.get("created_at", _now_iso()),
                "updated_at": _now_iso(),
                "last_message": last,
            }
        )
    _index_save(idx)


def _create_convo(title: str = "New chat") -> Dict[str, Any]:
    cid = uuid.uuid4().hex[:10]
    convo = {
        "id": cid,
        "title": title,
        "created_at": _now_iso(),
        "messages": [],
        "memory": {"facts": [], "summary": ""},
    }
    _convo_save(convo)
    return convo


def _get_default_convo() -> Dict[str, Any]:
    idx = _index_load()
    if idx:
        idx = sorted(idx, key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
        c = _convo_load(idx[0]["id"])
        if c:
            return c
    return _create_convo("Default chat")

# -----------------------------------------------------------------------------
# Mini memory helpers (very lightweight)
# -----------------------------------------------------------------------------
def _facts_from(text: str) -> List[str]:
    text = text or ""
    facts: List[str] = []
    m = re.search(r"(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][\w\s'-]{0,40})", text, re.I)
    if m:
        facts.append(f"My name is {m.group(1).strip()}.")
    m = re.search(r"\bi\s+live\s+in\s+([A-Za-z][\w\s'-]{0,60})", text, re.I)
    if m:
        facts.append(f"I live in {m.group(1).strip()}.")
    m = re.search(r"\bmy\s+favorite\s+color\s+is\s+([A-Za-z]+)", text, re.I)
    if m:
        facts.append(f"My favorite color is {m.group(1).strip()}.")
    return facts


def _merge_facts(old: List[str], new: List[str], cap: int = 24) -> List[str]:
    seen, out = set(), []
    for f in (old or []) + (new or []):
        s = (f or "").strip()
        if not s:
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _recall_name(memory: Dict[str, Any]) -> str | None:
    for f in memory.get("facts", []):
        m = re.search(r"\bmy\s+name\s+is\s+(.+?)\.", f, re.I)
        if m:
            return m.group(1).strip()
    return None


def _system_prompt(memory: Dict[str, Any]) -> str:
    facts = memory.get("facts") or []
    facts_txt = "\n".join(f"- {f}" for f in facts) if facts else "(none)"
    return (
        "You are a concise, helpful assistant.\n"
        "Use the remembered user facts when relevant; do not invent details.\n"
        "If you are unsure, ask a brief clarifying question.\n\n"
        f"Remembered facts:\n{facts_txt}\n"
    )


def _shortcuts(user_msg: str, memory: Dict[str, Any]) -> str | None:
    txt = (user_msg or "").lower().strip()
    if re.search(r"\bwhat('?s| is)\s*my\s*name\b", txt) or "do you remember my name" in txt or "who am i" in txt:
        name = _recall_name(memory)
        return f"Your name is {name}." if name else "I don't have your name saved yet."
    if re.search(r"\bwhat\s+do\s+you\s+remember\b", txt):
        facts = memory.get("facts") or []
        return "Here’s what I remember:\n- " + "\n- ".join(facts) if facts else "I don't have any saved facts yet."
    return None

# -----------------------------------------------------------------------------
# Ollama calls (chat first; fallback to generate)
# -----------------------------------------------------------------------------
def _ollama_chat(messages: List[Dict[str, str]], options: Dict[str, Any], stream: bool = False):
    payload = {"model": MODEL_NAME, "messages": messages, "stream": stream, "options": options}
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=stream, timeout=None if stream else 120)
        if r.status_code == 404:
            raise requests.HTTPError("404 /api/chat")
        r.raise_for_status()
        return r
    except Exception:
        # fallback to /api/generate
        prompt_parts = []
        for m in messages:
            prompt_parts.append(f"{m['role'].upper()}:\n{m['content']}")
        prompt_parts.append("ASSISTANT:")
        gen_payload = {"model": MODEL_NAME, "prompt": "\n\n".join(prompt_parts), "stream": stream, "options": options}
        r2 = requests.post(f"{OLLAMA_HOST}/api/generate", json=gen_payload, stream=stream, timeout=None if stream else 120)
        r2.raise_for_status()
        return r2

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@chat_bp.route("/", methods=["GET"])
def chat_ui():
    return render_template("chatbot.html")

# List/create conversations
@chat_bp.route("/convos", methods=["GET", "POST"])
def convos():
    if request.method == "GET":
        return jsonify(_index_load())
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New chat").strip()
    c = _create_convo(title)
    return jsonify({"id": c["id"], "title": c["title"]})

# Get/patch/delete one conversation
@chat_bp.route("/convos/<cid>", methods=["GET", "PATCH", "DELETE"])
def convos_item(cid):
    convo = _convo_load(cid)
    if convo is None:
        return jsonify({"error": "Conversation not found"}), 404

    if request.method == "GET":
        return jsonify(convo)

    if request.method == "PATCH":
        data = request.get_json(silent=True) or {}
        if "title" in data:
            t = (data["title"] or "").strip()
            if t:
                convo["title"] = t
        if data.get("clear_memory"):
            convo["memory"] = {"facts": [], "summary": ""}
        if data.get("clear_messages"):
            convo["messages"] = []
        _convo_save(convo)
        return jsonify({"ok": True})

    # DELETE
    try:
        os.remove(_cpath(cid))
    except Exception:
        pass
    idx = [it for it in _index_load() if it.get("id") != cid]
    _index_save(idx)
    return jsonify({"ok": True})

# Export conversation
@chat_bp.route("/convos/<cid>/export.<fmt>", methods=["GET"])
def export_convo(cid, fmt):
    convo = _convo_load(cid)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404

    safe_title = re.sub(r"[^A-Za-z0-9 _.-]+", "", (convo.get("title") or "chat")).strip() or "chat"

    if fmt == "json":
        data = json.dumps(convo, ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(io.BytesIO(data), mimetype="application/json", as_attachment=True, download_name=f"{safe_title}.json")

    # markdown
    lines = [f"# {safe_title}", "", f"_Created: {convo.get('created_at','')}_", ""]
    for m in convo.get("messages", []):
        who = (m.get("role") or "user").capitalize()
        ts = m.get("ts", "")
        content = (m.get("content") or "").replace("\r\n", "\n")
        lines += [f"**{who} {f'({ts})' if ts else ''}**", "", content, ""]
    md = "\n".join(lines).encode("utf-8")
    return send_file(io.BytesIO(md), mimetype="text/markdown", as_attachment=True, download_name=f"{safe_title}.md")

# Non-stream send
@chat_bp.route("/conv/<cid>/send", methods=["POST"])
def chat_send(cid):
    payload = request.get_json(silent=True) or {}
    user_msg = (payload.get("message") or request.form.get("message") or "").strip()
    fast = bool(payload.get("fast"))

    if not user_msg:
        return jsonify({"error": "No message provided."}), 400

    convo = _convo_load(cid) or _create_convo(f"Chat {cid}")

    # add facts if present
    pre = _facts_from(user_msg)
    if pre:
        convo["memory"]["facts"] = _merge_facts(convo.get("memory", {}).get("facts", []), pre)
        _convo_save(convo)

    memory = convo.get("memory") or {}
    special = _shortcuts(user_msg, memory)
    if special:
        ts = _now_iso()
        convo["messages"].append({"role": "user", "content": user_msg, "ts": ts})
        convo["messages"].append({"role": "assistant", "content": special, "ts": ts})
        _convo_save(convo)
        return jsonify({"reply": special, "cid": convo["id"]})

    # build short history
    history = convo.get("messages", [])
    recent = history[-12:] if len(history) > 12 else history
    msgs = [{"role": "system", "content": _system_prompt(convo.get("memory") or {})}]
    msgs.extend({"role": m["role"], "content": m["content"]} for m in recent)
    msgs.append({"role": "user", "content": user_msg})

    r = _ollama_chat(msgs, options=FAST_OPTIONS if fast else DEFAULT_OPTIONS, stream=False)
    content = ""
    try:
        j = r.json()
        content = (j.get("message") or {}).get("content", "") or j.get("response", "") or ""
    except Exception:
        content = r.text or ""

    ts = _now_iso()
    convo["messages"].append({"role": "user", "content": user_msg, "ts": ts})
    convo["messages"].append({"role": "assistant", "content": content, "ts": ts})
    convo["memory"]["facts"] = _merge_facts(convo["memory"].get("facts", []), _facts_from(content))
    _convo_save(convo)

    return jsonify({"reply": content, "cid": convo["id"]})

# Stream send (accept GET *and* POST to avoid 405s from clients)
@chat_bp.route("/conv/<cid>/stream", methods=["GET", "POST"])
def chat_stream(cid):
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        user_msg = (data.get("message") or "").strip()
        fast = bool(data.get("fast"))
    else:
        user_msg = (request.args.get("message") or "").strip()
        fast = request.args.get("fast", "false").lower() in ("1", "true", "yes")

    if not user_msg:
        return Response("No message provided", status=400, mimetype="text/plain")

    convo = _convo_load(cid) or _create_convo(f"Chat {cid}")

    # pre-facts
    pre = _facts_from(user_msg)
    if pre:
        convo["memory"]["facts"] = _merge_facts(convo.get("memory", {}).get("facts", []), pre)
        _convo_save(convo)

    memory = convo.get("memory") or {}
    special = _shortcuts(user_msg, memory)
    if special:
        def single():
            yield special
            ts = _now_iso()
            convo["messages"].append({"role": "user", "content": user_msg, "ts": ts})
            convo["messages"].append({"role": "assistant", "content": special, "ts": ts})
            _convo_save(convo)
        return Response(stream_with_context(single()), mimetype="text/plain")

    history = convo.get("messages", [])
    recent = history[-12:] if len(history) > 12 else history
    msgs = [{"role": "system", "content": _system_prompt(memory)}]
    msgs.extend({"role": m["role"], "content": m["content"]} for m in recent)
    msgs.append({"role": "user", "content": user_msg})

    def generate_and_persist() -> Generator[str, None, None]:
        chunks: List[str] = []
        try:
            with _ollama_chat(msgs, options=FAST_OPTIONS if fast else DEFAULT_OPTIONS, stream=True) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore")
                    # /api/chat streams JSON lines (often with 'data:' prefix)
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        piece = (obj.get("message") or {}).get("content", "")
                        if not piece:
                            piece = obj.get("response", "")
                        if piece:
                            chunks.append(piece)
                            yield piece
                        if obj.get("done") is True:
                            break
                    except Exception:
                        # raw text piece (fallback)
                        chunks.append(line)
                        yield line
        except Exception as e:
            yield f"[stream error: {e}]"

        # persist once completed
        try:
            reply = "".join(chunks).strip()
            ts = _now_iso()
            convo["messages"].append({"role": "user", "content": user_msg, "ts": ts})
            convo["messages"].append({"role": "assistant", "content": reply, "ts": ts})
            convo["memory"]["facts"] = _merge_facts(convo["memory"].get("facts", []), _facts_from(reply))
            _convo_save(convo)
        except Exception:
            pass

    resp = Response(stream_with_context(generate_and_persist()), mimetype="text/plain")
    # Helpful headers for proxies
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"] = "no-cache"
    return resp

