import ollama

def summarize_text(prompt):
    refined_prompt = (
        "Summarize the following text in 3-5 concise sentences. "
        "Do not explain. Only summarize the core points clearly:\n\n"
        f"{prompt}"
    )

    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": refined_prompt}]
    )

    return response['message']['content']

