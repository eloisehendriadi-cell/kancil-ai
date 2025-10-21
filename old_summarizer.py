from transformers import pipeline

# Load model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if False else -1,
)

CHUNK_WORDS = 400
PART_MAX = 150
PART_MIN = 50
FINAL_MAX = 180
FINAL_MIN = 60

def _split_into_chunks(text: str, chunk_size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def _summarize_chunk(chunk: str) -> str:
    out = summarizer(
        chunk,
        max_length=PART_MAX,
        min_length=PART_MIN,
        do_sample=False,
    )[0]["summary_text"]
    return out.strip()

def summarize_text(long_text: str) -> str:
    if not long_text.strip():
        return "⚠️ No text provided."

    chunks = _split_into_chunks(long_text)
    partials = [_summarize_chunk(c) for c in chunks]

    if len(partials) == 1:
        return partials[0]

    final_summary = summarizer(
        " ".join(partials),
        max_length=FINAL_MAX,
        min_length=FINAL_MIN,
        do_sample=False,
    )[0]["summary_text"]
    return final_summary.strip()

