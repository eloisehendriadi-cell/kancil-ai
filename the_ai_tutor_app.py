from flask import Flask, request, render_template
from the_summarizer import summarize_text
from the_pdf_reader import extract_text_from_pdf
from the_yt_transcriber import download_audio_from_youtube, transcribe_audio
from chatbot import get_chatbot_response
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize-text", methods=["POST"])
def summarize_text_route():
    text = request.form.get("text", "")
    if not text:
        return render_template("index.html", result="⚠️ Please enter some text.")
    summary = summarize_text(text)
    return render_template("index.html", result=summary)

@app.route("/summarize-pdf", methods=["POST"])
def summarize_pdf():
    pdf = request.files.get("pdf")
    if not pdf:
        return render_template("index.html", result="⚠️ Please upload a PDF file.")
    path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(path)
    extracted_text = extract_text_from_pdf(path)
    summary = summarize_text(extracted_text)
    return render_template("index.html", result=summary)

@app.route("/summarize-youtube", methods=["POST"])
def summarize_youtube():
    url = request.form.get("youtube_url")
    if not url:
        return render_template("index.html", result="⚠️ Please paste a YouTube link.")
    audio_file = download_audio_from_youtube(url)
    transcript = transcribe_audio(audio_file)
    summary = summarize_text(transcript)
    return render_template("index.html", result=summary)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_input = ""
    bot_response = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        bot_response = get_chatbot_response(user_input)
    return render_template("chat.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)

