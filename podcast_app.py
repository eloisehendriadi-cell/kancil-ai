# the_ai_tutor/podcast_app.py
from flask import Blueprint, request, jsonify, current_app, render_template_string
import os, re, uuid, subprocess, tempfile, math, wave, random, shutil, json
import numpy as np
import requests

podcast_bp = Blueprint("podcast", __name__, template_folder="templates")

# ------------------------------
# Config (Ollama + TTS)
# ------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME",  "gemma:2b")

# Preferred voices (used only if installed)
PREF_HOST_VOICE  = os.getenv("PODCAST_HOST_VOICE",  "Alex")       # macOS voice
PREF_GUEST_VOICE = os.getenv("PODCAST_GUEST_VOICE", "Samantha")   # macOS voice

SAY_BIN = shutil.which("say") or "/usr/bin/say"
AFC_BIN = shutil.which("afconvert") or "/usr/bin/afconvert"
SAMPLE_RATE = 16000  # 16 kHz mono

# ------------------------------
# LLM helper
# ------------------------------
DEFAULT_OPTIONS = {
    "temperature": 0.35,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 2048,
    "num_predict": 700,  # allow longer scripts (~3–4 min spoken)
}

def query_ollama_chat(messages, stream=False, timeout=120):
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": stream,
            "options": DEFAULT_OPTIONS,
            "keep_alive": "10m",
        },
        timeout=None if stream else timeout,
        stream=stream,
    )
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()

def query_ollama(prompt: str):
    return query_ollama_chat([{"role": "user", "content": prompt}], stream=False)

# ------------------------------
# Script cleaning & parsing
# ------------------------------
# Leading label like: **Host:**, Host:, Guest-, Dr. Sarah:, Sarah Johnson:
LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*([A-Za-z][A-Za-z .'\-]{0,40})\s*[:\-]\s*(?:\*\*)?\s*",
    re.I,
)

PUNCT_WORDS = [
    "asterisk","asterisks","comma","period","slash","backslash","semicolon","semicolon(s)",
    "colon","hyphen","dash","underscore","ampersand","ellipsis","ellipses",
    "hashtag","hash","at sign","question mark","exclamation mark"
]

def strip_html(s: str) -> str:
    if not s: return ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def _speaker_from_label(lbl: str) -> str:
    """Map any label to host/guest; anything not containing 'host' becomes guest."""
    return "host" if "host" in (lbl or "").lower() else "guest"

def clean_for_tts(text: str) -> str:
    """Remove markdown, labels, stage directions, and spoken punctuation words."""
    if not text: return ""
    t = text
    # strip a leading NAME: in this segment
    t = re.sub(r"^\s*(?:\*\*)?\s*[A-Za-z][A-Za-z .'\-]{0,40}\s*:\s*(?:\*\*)?\s*", "", t)
    # remove markdown ornaments and headings
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)           # **bold**
    t = re.sub(r"^#{1,6}\s+.*$", "", t, flags=re.M)  # # Headings
    t = t.replace("—", "-").replace("•", " ")
    # remove stage directions like [music], (laughs), *applause*
    t = re.sub(r"\[.*?\]", "", t)
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"[*_`~]", "", t)
    # remove literal words that shouldn't be spoken
    for w in PUNCT_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", "", t, flags=re.I)
    # whitespace tidy
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _prepare_for_say_prosody(text: str) -> str:
    """
    Add gentle rate and micro-pauses for macOS 'say' to sound less robotic.
    """
    s = clean_for_tts(text)
    if not s:
        return s
    # Insert small silences between sentences
    parts = re.split(r"([.!?])\s+", s)
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if seg:
            out.append(seg + (punct or ""))
            out.append("[[slnc 120]]")
    s = " ".join(out).strip()
    # Slightly brisk rate, normal pitch/volume (190 is ~conversational)
    return f"[[rate 190]] {s}"

def parse_dialogue(script: str):
    """
    Returns [{'speaker': 'host'|'guest', 'text': '...'}]
    Understands labels like 'Host:' and arbitrary names like 'Dr. Johnson:'.
    If no labels, alternate by paragraph.
    """
    lines = [ln.strip() for ln in script.splitlines() if ln.strip()]
    segs, current = [], None
    for ln in lines:
        m = LABEL_RE.match(ln)
        if m:
            speaker = _speaker_from_label(m.group(1))
            spoken = clean_for_tts(LABEL_RE.sub("", ln, count=1))
            if spoken:
                segs.append({"speaker": speaker, "text": spoken})
            current = speaker
        else:
            spoken = clean_for_tts(ln)
            if not spoken:
                continue
            if current is None:
                current = "host" if not segs else ("guest" if segs[-1]["speaker"] == "host" else "host")
            segs.append({"speaker": current, "text": spoken})

    if not segs:
        cleaned = clean_for_tts(script)
        if cleaned:
            segs = [{"speaker": "host", "text": cleaned}]
    return segs

# ------------------------------
# Voice discovery & robust `say`
# ------------------------------
def _list_mac_voices():
    """Return list of (name, lang) tuples from `say -v ?`."""
    if not SAY_BIN or not os.path.exists(SAY_BIN):
        return []
    try:
        out = subprocess.run([SAY_BIN, "-v", "?"], capture_output=True, text=True, check=False)
    except Exception:
        return []
    voices = []
    for line in (out.stdout or "").splitlines():
        if not line.strip():
            continue
        name = line.split()[0]
        m = re.search(r"\b([a-z]{2}[_-][A-Za-z]{2})\b", line)
        lang = m.group(1) if m else ""
        voices.append((name, lang))
    return voices

def _choose_voices(preferences=None):
    """Pick two distinct installed English voices; prefer given names if present."""
    voices = _list_mac_voices()
    english = [n for n, lang in voices if "en" in (lang or "").lower()]

    chosen = []
    if preferences:
        for p in preferences:
            if p in english and p not in chosen:
                chosen.append(p)
            if len(chosen) == 2:
                break

    leftovers = [v for v in english if v not in chosen]
    random.shuffle(leftovers)
    chosen += leftovers[: 2 - len(chosen)]

    if len(chosen) < 2:
        others = [n for n, _ in voices if n not in chosen]
        random.shuffle(others)
        chosen += others[: 2 - len(chosen)]

    if len(chosen) == 1:
        chosen.append(chosen[0])
    elif not chosen:
        chosen = [None, None]
    return chosen[0], chosen[1]

def _say_to_wav(text: str, voice: str, out_wav: str, rate=SAMPLE_RATE):
    """
    Uses macOS 'say' to AIFF, then 'afconvert' to 16 kHz wav.
    Retries without -v if the requested voice isn’t available.
    """
    if not SAY_BIN or not os.path.exists(SAY_BIN):
        raise RuntimeError("macOS 'say' is not available on this system.")
    if not AFC_BIN or not os.path.exists(AFC_BIN):
        raise RuntimeError("macOS 'afconvert' is not available on this system.")

    with tempfile.TemporaryDirectory() as td:
        aiff_path = os.path.join(td, "seg.aiff")

        cmd = [SAY_BIN, "-o", aiff_path]
        if voice:
            cmd += ["-v", voice]
        cmd += [text]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and voice:
            # Retry with default voice
            proc = subprocess.run([SAY_BIN, "-o", aiff_path, text], capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "macOS `say` failed")

        proc = subprocess.run(
            [AFC_BIN, aiff_path, out_wav, "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1"],
            capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "afconvert failed")

def _read_wav_int16(path: str, expect_rate=SAMPLE_RATE):
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels()
        fr    = wf.getframerate()
        nfrm  = wf.getnframes()
        raw   = wf.readframes(nfrm)
    data = np.frombuffer(raw, dtype=np.int16)
    if nchan > 1:
        data = data.reshape(-1, nchan).mean(axis=1).astype(np.int16)
    if fr != expect_rate:
        ratio = expect_rate / fr
        x = np.arange(len(data))
        xi = np.linspace(0, len(data)-1, int(len(data)*ratio))
        data = np.interp(xi, x, data).astype(np.int16)
    return data

def _write_wav_int16(path: str, data: np.ndarray, rate=SAMPLE_RATE):
    data = np.asarray(data, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())

# ------------------------------
# Background music generator (random pad)
# ------------------------------
def _bgm_pad(duration_s: float, rate=SAMPLE_RATE, seed=None) -> np.ndarray:
    """
    Simple gentle pad: random 4-chord loop in C major. Very quiet (~ -26 dBFS).
    """
    rng = random.Random(seed)
    length = int(duration_s * rate)

    chords = [
        [261.63, 329.63, 392.00],   # C major: C E G
        [196.00, 246.94, 392.00],   # G major: G B G
        [220.00, 261.63, 329.63],   # A minor: A C E
        [174.61, 220.00, 261.63],   # F major: F A C
    ]
    prog = [rng.choice(chords) for _ in range(4)]
    seg_len = max(1, length // 4)

    out = np.zeros(length, dtype=np.float32)
    idx = 0
    for i, chord in enumerate(prog):
        end = idx + seg_len if i < 3 else length
        n   = max(1, end - idx)
        tt  = np.linspace(0, n / rate, n, endpoint=False)

        lfo = 0.5 + 0.5*np.sin(2*np.pi*0.2*tt)
        chord_wave = np.zeros_like(tt, dtype=np.float32)
        for f in chord:
            chord_wave += np.sin(2*np.pi*f*tt + rng.uniform(0, math.pi))

        env = np.sin(np.linspace(0, math.pi, n)) ** 1.5
        seg = (chord_wave / 3.0) * env * lfo
        out[idx:end] = seg
        idx = end

    out = out / (np.max(np.abs(out)) + 1e-9)
    out *= 0.05  # -26 dBFS-ish
    return (out * 32767.0).astype(np.int16)

# ------------------------------
# Script generation + audio build
# ------------------------------
def _generate_script_from_notes(title: str, plain_text: str) -> str:
    """
    Target ~3–4 minutes spoken (≈ 475–650 words at ~150–170 wpm).
    Two speakers with clear labels for parsing.
    """
    prompt = (
        "Write an engaging two-speaker educational podcast script with a Host and a Guest.\n"
        "- Length target: 475–650 words (~3–4 minutes spoken).\n"
        "- Use ONLY the provided notes (no extra facts, no hallucinations).\n"
        "- Keep labels like 'Host:' or 'Guest:' at the start of lines for every turn.\n"
        "- Start with a 1–2 sentence teaser, then cover the core ideas step-by-step,\n"
        "  include one quick real-world connection, and finish with a crisp recap.\n"
        "- Keep sentences natural and conversational; avoid long lists.\n"
        f"Topic: {title}\n\n"
        "NOTES:\n" + (plain_text or "No notes.") + "\n\n"
        "Return ONLY the script text with 'Host:' and 'Guest:' labels."
    )
    return query_ollama(prompt)

def _segments_from_script(script_text: str):
    # If labels exist, parse them; else alternate by paragraph
    if any(LABEL_RE.match(ln) for ln in script_text.splitlines()):
        return parse_dialogue(script_text)
    paras = [p.strip() for p in script_text.split("\n\n") if p.strip()]
    segs, speaker = [], "host"
    for p in paras:
        segs.append({"speaker": speaker, "text": clean_for_tts(p)})
        speaker = "guest" if speaker == "host" else "host"
    return segs

def _synthesize_segments(segs, out_dir, host_voice, guest_voice):
    arrays = []
    for i, seg in enumerate(segs):
        voice = host_voice if seg["speaker"] == "host" else guest_voice
        tmp_wav = os.path.join(out_dir, f"seg_{i:03d}.wav")
        text_for_tts = _prepare_for_say_prosody(seg["text"])
        _say_to_wav(text_for_tts, voice, tmp_wav, rate=SAMPLE_RATE)
        arr = _read_wav_int16(tmp_wav, expect_rate=SAMPLE_RATE)
        arrays.append(arr)
    return arrays

def _concat_with_gaps(arrays, gap_ms=220):
    gap = np.zeros(int(SAMPLE_RATE * (gap_ms/1000.0)), dtype=np.int16)
    if not arrays:
        return np.zeros(0, dtype=np.int16)
    out = arrays[0]
    for a in arrays[1:]:
        out = np.concatenate([out, gap, a])
    return out

def _mix_with_bgm(voice_pcm, bgm_pcm, bgm_gain=0.18):
    # pad/trim bgm
    if len(bgm_pcm) < len(voice_pcm):
        reps = int(math.ceil(len(voice_pcm)/len(bgm_pcm))) or 1
        bgm_pcm = np.tile(bgm_pcm, reps)[:len(voice_pcm)]
    else:
        bgm_pcm = bgm_pcm[:len(voice_pcm)]
    # lower bgm to keep voice clear
    bgm_scaled = (bgm_pcm.astype(np.int32) * bgm_gain).astype(np.int32)
    mixed = voice_pcm.astype(np.int32) + bgm_scaled
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    return mixed

def _ensure_dirs():
    root = current_app.config.get("UPLOAD_FOLDER", "uploads")
    pod_dir = os.path.join(root, "podcasts")
    os.makedirs(pod_dir, exist_ok=True)
    return pod_dir

def _build_audio(title: str, html: str):
    notes_text = strip_html(html)
    if not notes_text:
        return None, "No note content provided."

    # 1) Script
    script = _generate_script_from_notes(title, notes_text)
    if not script or script.lower().startswith("error contacting ollama"):
        return None, "Could not generate script."

    # 2) Segments
    segments = _segments_from_script(script)

    # 3) Voices (auto-pick installed English voices; prefer env hints)
    host_voice, guest_voice = _choose_voices([PREF_HOST_VOICE, PREF_GUEST_VOICE])

    # 4) TTS
    pod_dir = _ensure_dirs()
    tmp_dir = tempfile.mkdtemp(prefix="pod_", dir=pod_dir)
    try:
        arrays = _synthesize_segments(segments, tmp_dir, host_voice, guest_voice)
        voice_track = _concat_with_gaps(arrays, gap_ms=220)

        # 5) BGM + mix
        dur_s = max(1.0, len(voice_track) / float(SAMPLE_RATE))
        bgm   = _bgm_pad(dur_s, rate=SAMPLE_RATE, seed=random.randint(1, 99999))
        final = _mix_with_bgm(voice_track, bgm, bgm_gain=0.18)

        # 6) Write file
        fname = f"pod_{uuid.uuid4().hex[:10]}.wav"
        out_path = os.path.join(pod_dir, fname)
        _write_wav_int16(out_path, final, rate=SAMPLE_RATE)
    finally:
        # best-effort cleanup
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return (fname, script), None

# ------------------------------
# Public routes
# ------------------------------
@podcast_bp.route("/podcasts", methods=["GET"])
def podcasts():
    return render_template_string("""
    {% extends 'base.html' %}
    {% block title %}Podcasts • Kancil AI{% endblock %}
    {% block content %}
      <section class="card">
        <h2 style="margin:0 0 .5rem 0">🎙️ Podcasts</h2>
        <p class="muted">Generate a podcast from any saved note via the Dashboard or Workspace.</p>
        <p class="muted">Audio files will appear under <code>/uploads/podcasts/</code>.</p>
      </section>
    {% endblock %}
    """)

def _get_body():
    """Allow both form and JSON bodies."""
    if request.is_json:
        return request.get_json(silent=True) or {}
    return {k: v for k, v in request.form.items()}

@podcast_bp.route("/podcast/script", methods=["POST"])
def podcast_script_only():
    """
    JSON/form: { title, html } or { title, text }
    Returns: { ok, script }  (no audio synthesis)
    """
    try:
        data  = _get_body()
        title = (data.get("title") or "Untitled").strip()
        html  = (data.get("html")  or data.get("text") or "").strip()
        notes_text = strip_html(html)
        if not notes_text:
            return jsonify({"ok": False, "error": "No note content provided."}), 400
        script = _generate_script_from_notes(title, notes_text)
        if not script:
            return jsonify({"ok": False, "error": "LLM returned empty script."}), 500
        return jsonify({"ok": True, "script": script})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Script error: {e}"}), 500

@podcast_bp.route("/podcast/generate", methods=["POST"])
@podcast_bp.route("/notes/podcast", methods=["POST"])
def generate_podcast():
    """
    JSON/form: { title, html } or { title, text }
    Returns: { audio_url, script, filename } on success
    """
    try:
        data  = _get_body()
        title = (data.get("title") or "Untitled").strip()
        html  = (data.get("html")  or data.get("text") or "").strip()
        result, err = _build_audio(title, html)
        if err:
            return jsonify({"error": err}), 400
        fname, script = result
        audio_url = f"/uploads/podcasts/{fname}"
        return jsonify({"audio_url": audio_url, "script": script, "filename": fname})
    except RuntimeError as e:
        # Typically missing macOS tools
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"TTS error: {e}"}), 500

