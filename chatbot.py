import ollama

def stream_response(prompt):
    try:
        stream = ollama.chat(
            model="gemma:2b",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

