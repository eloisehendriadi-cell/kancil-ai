# chatbot_logic.py

from transformers import pipeline

# You can replace this with any small conversational model
chatbot = pipeline("text-generation", model="gpt2")

def get_chatbot_response(user_input):
    response = chatbot(user_input, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]

