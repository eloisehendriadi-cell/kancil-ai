from the_yt_transcriber import download_audio_from_youtube, transcribe_audio
from the_summarizer import summarize_text

# Replace with any short and clear YouTube video
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Download & transcribe
audio_file = download_audio_from_youtube(url)
transcript = transcribe_audio(audio_file)

# Print raw transcript
print("🎤 Transcript:\n")
print(transcript[:1000])

# Optional: summarize the transcript
print("\n📝 Summary:\n")
print(summarize_text(transcript))

