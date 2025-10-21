# the_ai_tutor/quiz_app.py
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from html import unescape
from typing import List, Dict, Any, Optional, Tuple

import requests
from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
    session,
    url_for,
)

# -------------------------------------------------------------------
# Blueprint
# -------------------------------------------------------------------
quiz_bp = Blueprint("quiz", __name__, template_folder="templates")

# -------------------------------------------------------------------
# Config (env overrides keep your handover defaults)
# -------------------------------------------------------------------
DEFAULT_NOTES_FILE = "saved_notes.json"

# Primary default: small + fast via Ollama (can be overridden)
QUIZ_MODEL_NAME = os.getenv("QUIZ_MODEL_NAME", "llama3.2:1b").strip()
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# Switch to hosted model transparently if OPENROUTER_API_KEY is set
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
HOSTED_DEFAULT     = os.getenv("QUIZ_HOSTED_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()

# Knobs (good on laptops; from your handover pack)
DEFAULT_OPTIONS = {
    "temperature": float(os.getenv("QUIZ_TEMPERATURE", "0.12")),
    "num_ctx": int(os.getenv("QUIZ_NUM_CTX", "2048")),
    "num_predict": int(os.getenv("QUIZ_NUM_PREDICT", "300")),
}
BATCH_SIZE = int(os.getenv("QUIZ_BATCH_SIZE", "3"))
MAX_TRIES  = int(os.getenv("QUIZ_MAX_TRIES", "5"))

# Validation rules
BAD_OPTIONS = {
    "all of the above", "none of the above", "true", "false",
    "both a and b", "both b and c", "all", "none",
}
YES_NO_STEM = re.compile(r"^(?:is|are|can|does|do|did|will|should|has|have)\b.*\?$", re.I)
ONLY_PUNCT  = re.compile(r"^\W+$")
CODE_FENCE  = re.compile(r"```(?:json)?|```", re.I)
WORDTOK     = re.compile(r"[A-Za-z0-9']+")

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _notes_path() -> str:
    """Path to saved notes (used only by optional /quiz endpoints)."""
    return DEFAULT_NOTES_FILE

def _strip_html(html: str) -> str:
    if not html:
        return ""
    s = unescape(html)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()

def _sentences(text: str) -> List[str]:
    raw = re.split(r"(?:\n+|[•\u2022]+|\.\s+|;\s+)", text or "")
    out: List[str] = []
    for r in raw:
        t = re.sub(r"\s+", " ", (r or "").strip())
        if len(t) >= 8:
            out.append(t)
    # de-dup while preserving order
    seen, uniq = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def _word_count(s: str) -> int:
    return len(WORDTOK.findall(s or ""))

def _near_dup(a: str, b: str) -> bool:
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    if a == b:
        return True
    if not a or not b:
        return False
    sa = set(re.findall(r"[a-z0-9]+", a))
    sb = set(re.findall(r"[a-z0-9]+", b))
    if not sa or not sb:
        return False
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union) >= 0.8

def _options_ok(opts: List[str]) -> bool:
    if len(opts) != 4:
        return False
    norm = [o.strip() for o in opts]
    if any((not o) or ONLY_PUNCT.match(o) for o in norm):
        return False
    # 1–6 words; ≤32 chars
    if any(_word_count(o) < 1 or _word_count(o) > 6 or len(o) > 32 for o in norm):
        return False
    # no duplicates or near-dups
    if len(set(o.lower() for o in norm)) < 4:
        return False
    for i in range(4):
        for j in range(i + 1, 4):
            if _near_dup(norm[i], norm[j]):
                return False
    # ban generic answers
    if any(o.lower() in BAD_OPTIONS for o in norm):
        return False
    # limit punctuation noise
    if any(len(re.sub(r"[A-Za-z0-9 \-]", "", o)) > 3 for o in norm):
        return False
    return True

def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9\- _]", "", s).strip().lower()
    return re.sub(r"[\s_]+", "_", s) or "untitled"

# -------------------------------------------------------------------
# LLM client (OpenRouter if available, else Ollama)
# -------------------------------------------------------------------
def _chat_complete(prompt: str, *, model: Optional[str] = None, temperature: float = 0.2, num_predict: int = 300, timeout: int = 90) -> str:
    if OPENROUTER_API_KEY:
        use_model = model or HOSTED_DEFAULT
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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
        try:
            r = requests.post(f"{OPENROUTER_BASE}/chat/completions", json=body, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""
    # Ollama fallback
    use_model = model or QUIZ_MODEL_NAME
    opts = {"temperature": temperature, "num_predict": num_predict, "num_ctx": DEFAULT_OPTIONS["num_ctx"]}
    # Try /api/chat
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat",
                          json={"model": use_model, "messages": [{"role": "user", "content": prompt}],
                                "options": opts, "stream": False},
                          timeout=timeout)
        if r.status_code != 404:
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "").strip()
    except Exception:
        pass
    # Fallback /api/generate
    try:
        r2 = requests.post(f"{OLLAMA_HOST}/api/generate",
                           json={"model": use_model, "prompt": prompt, "options": opts, "stream": False},
                           timeout=timeout)
        r2.raise_for_status()
        return (r2.json().get("response") or "").strip()
    except Exception:
        return ""

# -------------------------------------------------------------------
# Prompting + parsing
# -------------------------------------------------------------------
PROMPT_SYSTEM = (
    "You are a meticulous exam writer. Create MCQs ONLY from the provided notes."
    " Each item has a clear, factual answer present in the notes."
    " Avoid yes/no stems and avoid 'All/None of the above'."
    " Return ONLY a JSON array; no commentary, no code fences."
    " Each item fields: question, options[4], answer_index (0-3), explanation."
    " The explanation must cite the specific fact (quote or paraphrase) from the notes and include the correct option text."
)

ONE_SHOT = {
    "question": "Which process causes water to move from a region of lower solute concentration to higher solute concentration across a partially permeable membrane?",
    "options": ["Osmosis", "Diffusion", "Active transport", "Transpiration"],
    "answer_index": 0,
    "explanation": "Osmosis is correct because the notes state that osmosis is the net movement of water across a partially permeable membrane from dilute to concentrated solution."
}

def _build_prompt(topic: str, notes_plain: str, n: int) -> str:
    example = json.dumps(ONE_SHOT, ensure_ascii=False)
    return (
        f"{PROMPT_SYSTEM}\n\n"
        f"TOPIC: {topic or 'Topic'}\n"
        f"COUNT: {n}\n\n"
        f"Return a JSON array of {n} items. Example of ONE item:\n{example}\n\n"
        f"NOTES (source of truth):\n{notes_plain}"
    )

def _repair_to_array(text: str) -> str:
    t = CODE_FENCE.sub("", text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return f"[{t}]"
    # strip list markers before objects
    t = re.sub(r"(?m)^\s*(?:\d+[\).\]]\s*|-|\*)\s*(\{)", r"\1", t)
    # if multiple small objects, wrap
    objs = re.findall(r"\{[^{}]+\}", t)
    if len(objs) >= 2:
        return "[" + ",".join(objs) + "]"
    # silly trailing commas
    t = re.sub(r",\s*([\]\}])", r"\1", t)
    return t

def _parse_items(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    t = _repair_to_array(text)
    # try array
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            j = json.loads(m.group(0))
            return j if isinstance(j, list) else []
        except Exception:
            pass
    # try single object with key
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            j = json.loads(m.group(0))
            if isinstance(j, dict):
                if isinstance(j.get("questions"), list):
                    return j["questions"]
        except Exception:
            pass
    # fall back: collect object-like snippets
    out = []
    for mm in re.findall(r"\{[^{}]+\}", t):
        try:
            out.append(json.loads(mm))
        except Exception:
            continue
    return out

# -------------------------------------------------------------------
# Normalization + scoring
# -------------------------------------------------------------------
def _normalize_item(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = (obj.get("question") or "").strip()
    opts = obj.get("options") or []
    ai = obj.get("answer_index")
    exp = (obj.get("explanation") or "").strip()

    if not q or not isinstance(opts, list) or len(opts) != 4:
        return None
    if YES_NO_STEM.match(q):
        return None
    opts = [str(o or "").strip() for o in opts]
    if not _options_ok(opts):
        return None
    try:
        ai = int(ai)
    except Exception:
        return None
    if ai < 0 or ai > 3:
        return None
    # mask if stem reveals answer verbatim
    answer = opts[ai].strip()
    if answer and re.search(rf"\b{re.escape(answer)}\b", q, flags=re.I):
        q = re.sub(rf"\b{re.escape(answer)}\b", "___", q, flags=re.I)

    # Explanation rule: must contain correct option text
    if answer.lower() not in (exp or "").lower():
        if exp:
            exp = f"{answer} — {exp}"
        else:
            exp = f"{answer} is correct."

    return {
        "question": q,
        "options": opts,
        "answer_index": ai,
        "explanation": exp,
    }

def _option_length_score(opts: List[str]) -> float:
    lens = [len(o) for o in opts]
    med  = sorted(lens)[len(lens)//2]
    diffs = [abs(l - med) for l in lens]
    return 1.0 - (sum(diffs) / (len(diffs) * max(1, med)))

def _score_item(item: Dict[str, Any]) -> float:
    score = 0.6 * _option_length_score(item["options"])
    if "___" in item["question"] and "?" not in item["question"]:
        score += 0.25
    if "which of the following" in item["question"].lower() and "about" not in item["question"].lower():
        score -= 0.15
    return score

# -------------------------------------------------------------------
# Grounding to notes: find supporting sentence + rewrite explanation
# -------------------------------------------------------------------
def _best_support_sentence(sentences: List[str], answer: str, question: str) -> Optional[str]:
    a = (answer or "").lower().strip()
    q = (question or "").lower().strip()
    if not sentences:
        return None
    best, best_score = None, 0.0
    a_tokens = set(re.findall(r"[a-z0-9]+", a))
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    for s in sentences:
        s_low = s.lower()
        direct = 1.0 if a and a in s_low else 0.0
        s_tokens = set(re.findall(r"[a-z0-9]+", s_low))
        overlap_a = len(a_tokens & s_tokens) / max(1, len(a_tokens))
        overlap_q = len(q_tokens & s_tokens) / max(1, len(q_tokens))
        score = direct * 1.2 + overlap_a * 0.7 + overlap_q * 0.3
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 0.35 else None

def _ground_explanation(item: Dict[str, Any], support: str) -> str:
    ans = item["options"][item["answer_index"]]
    return f"{ans} is correct because the notes state: “{support}”"

# -------------------------------------------------------------------
# LLM generation loop (batched, ranked, validated, grounded)
# -------------------------------------------------------------------
def _generate_from_notes(topic: str, notes_plain: str, target: int) -> List[Dict[str, Any]]:
    target = max(10, min(20, int(target)))
    accepted: List[Dict[str, Any]] = []
    seen = set()
    tries = 0

    while len(accepted) < target and tries < MAX_TRIES:
        need = min(BATCH_SIZE, target - len(accepted))
        prompt = _build_prompt(topic, notes_plain, need)
        raw = _chat_complete(
            prompt,
            model=QUIZ_MODEL_NAME if not OPENROUTER_API_KEY else HOSTED_DEFAULT,
            temperature=DEFAULT_OPTIONS["temperature"],
            num_predict=DEFAULT_OPTIONS["num_predict"],
        )
        parsed = _parse_items(raw)
        normals: List[Dict[str, Any]] = []
        for obj in parsed or []:
            n = _normalize_item(obj)
            if n:
                normals.append(n)

        # Grounding step
        sentences = _sentences(notes_plain)
        grounded: List[Dict[str, Any]] = []
        for it in normals:
            ans = it["options"][it["answer_index"]]
            support = _best_support_sentence(sentences, ans, it["question"])
            if not support:
                # reject items we cannot ground to the notes
                continue
            # explanation must include fact + answer text
            exp = it.get("explanation", "")
            if len(exp) < 24 or re.search(r"\b(according to|as (?:stated|noted)|from the notes)\b", exp, flags=re.I):
                exp = _ground_explanation(it, support)
            if ans.lower() not in exp.lower():
                exp = f"{ans} — {exp}"
            it["explanation"] = exp
            it["supporting_sentence"] = support
            grounded.append(it)

        # rank, dedupe by question
        ranked = sorted(grounded, key=_score_item, reverse=True)
        for it in ranked:
            qk = it["question"].strip().lower()
            if qk in seen:
                continue
            seen.add(qk)
            accepted.append(it)
            if len(accepted) >= target:
                break

        tries += 1
        # small temperature bump if we failed to accept anything
        if not grounded and DEFAULT_OPTIONS["temperature"] <= 0.18:
            DEFAULT_OPTIONS["temperature"] = round(DEFAULT_OPTIONS["temperature"] + 0.02, 2)

    return accepted[:target]

# -------------------------------------------------------------------
# Public helper (as used by workspace.py)
# -------------------------------------------------------------------
def generate_quiz_items(source_text: str, topic: str = "Topic", count: int = 12) -> List[Dict[str, Any]]:
    """
    Build N validated MCQs from the provided source text/notes (HTML or plain).
    Returns a list of items with keys:
      - question (str)
      - options (list[str] of length 4)
      - answer_index (int 0..3)
      - explanation (str)  # grounded + includes correct option text
      - supporting_sentence (str)  # verbatim from notes
    """
    text = source_text or ""
    notes_plain = _strip_html(text) if ("<" in text and ">" in text) else text
    target = max(10, min(15, int(count or 12)))
    items = _generate_from_notes(topic or "Topic", notes_plain, target)
    if not items:
        # Always return a list (workspace wrapper expects that)
        return []
    return items

# Legacy alias (some older code may import this)
make_quiz_items = generate_quiz_items

# -------------------------------------------------------------------
# Minimal routes to support your embedded player
# -------------------------------------------------------------------
@quiz_bp.route("/player")
def quiz_player():
    topic = (request.args.get("topic") or session.get("shared_source_title") or "Topic").strip()
    api_url = request.args.get("api") or url_for("workspace.api_generate_quiz")
    embed = request.args.get("embed") == "1"
    return render_template("quiz.html", topic=topic, api_url=api_url, embed=embed)

# Optional: simple menu/page (safe no-op if templates are absent)
@quiz_bp.route("/quiz")
def quiz_menu():
    return render_template("quiz_menu.html")

