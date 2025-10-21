from youtube_transcript_api import YouTubeTranscriptApi
from summarizer import summarize_text

def extract_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return None

def summarize_youtube(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return summarize_text(text)
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"

