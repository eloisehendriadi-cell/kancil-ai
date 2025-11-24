"""
YouTube transcript extraction utility.
Attempts to extract subtitles/captions from YouTube videos.
"""

import re
import json
from urllib.parse import urlparse, parse_qs
from typing import Optional


def _extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.
    Supports: youtube.com, youtu.be, youtube-nocookie.com
    """
    if not url:
        return None
    
    url = url.strip()
    
    # youtu.be format: https://youtu.be/dQw4w9WgXcQ
    match = re.search(r'youtu\.be/([^/?&]+)', url)
    if match:
        return match.group(1)
    
    # youtube.com format: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    match = re.search(r'youtube\.com/.*[?&]v=([^&]+)', url)
    if match:
        return match.group(1)
    
    # youtube-nocookie format
    match = re.search(r'youtube-nocookie\.com/embed/([^/?&]+)', url)
    if match:
        return match.group(1)
    
    # Direct video ID (fallback)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    
    return None


def _get_transcript_with_api(video_id: str) -> Optional[str]:
    """
    Try to get transcript using youtube-transcript-api.
    This is the primary method.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
        
        api = YouTubeTranscriptApi()
        
        try:
            # Use the fetch method
            transcript = api.fetch(video_id, languages=['en'])
        except (TranscriptsDisabled, NoTranscriptFound):
            # Try any available language
            try:
                transcript_list = api.list(video_id)
                # Try first available transcript
                if hasattr(transcript_list, 'manually_created_transcripts') and transcript_list.manually_created_transcripts:
                    transcript = transcript_list.manually_created_transcripts[0].fetch()
                elif hasattr(transcript_list, 'generated_transcripts') and transcript_list.generated_transcripts:
                    transcript = transcript_list.generated_transcripts[0].fetch()
                else:
                    return None
            except Exception:
                return None
        
        if transcript:
            # Combine all text segments from FetchedTranscript
            texts = []
            for item in transcript:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
            text = ' '.join(texts)
            return text.strip() if text.strip() else None
    except ImportError:
        pass
    except Exception as e:
        print(f"[YOUTUBE] API error: {type(e).__name__}: {e}")
    
    return None


def _get_transcript_with_yt_dlp(video_id: str) -> Optional[str]:
    """
    Fallback: Try using yt-dlp to extract subtitles.
    """
    try:
        import subprocess
        import tempfile
        import os
        import glob
        
        # Check if yt-dlp is available
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return None
        
        # Use a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, 'video')
            
            cmd = [
                'yt-dlp',
                '--write-auto-subs',
                '--sub-format', 'json',
                '--skip-download',
                '-o', output_template,
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0:
                # Look for subtitle files
                sub_files = glob.glob(os.path.join(temp_dir, '*.json'))
                if sub_files:
                    try:
                        with open(sub_files[0], 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Extract text from JSON subtitle format
                        if isinstance(data, dict) and 'events' in data:
                            texts = []
                            for event in data['events']:
                                if 'segs' in event:
                                    for seg in event['segs']:
                                        if 'utf8' in seg:
                                            texts.append(seg['utf8'])
                            text = ' '.join(texts)
                            return text.strip() if text.strip() else None
                    except Exception as e:
                        print(f"[YOUTUBE] yt-dlp JSON parse error: {e}")
                        
                # Try VTT format as fallback
                vtt_files = glob.glob(os.path.join(temp_dir, '*.vtt'))
                if vtt_files:
                    try:
                        with open(vtt_files[0], 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Parse VTT format
                        lines = content.split('\n')
                        text_lines = [
                            line.strip() for line in lines 
                            if line.strip() 
                            and not line.startswith('WEBVTT') 
                            and not line.startswith('NOTE') 
                            and '-->' not in line
                            and not re.match(r'^\d{2}:\d{2}:\d{2}', line)
                        ]
                        text = ' '.join(text_lines)
                        return text.strip() if text.strip() else None
                    except Exception as e:
                        print(f"[YOUTUBE] yt-dlp VTT parse error: {e}")
    except Exception as e:
        print(f"[YOUTUBE] yt-dlp fallback error: {type(e).__name__}: {e}")
    
    return None


def _get_transcript_web_scrape(video_id: str) -> Optional[str]:
    """
    Last resort: Try to extract caption JSON from initial data.
    """
    try:
        import requests
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Look for captions data in the HTML
        match = re.search(r'"captions":\{"playerCaptionsTracklistRenderer":\{(.+?)\}\}', response.text)
        if match:
            # Found captions - try to extract caption track
            captions_section = match.group(1)
            caption_tracks = re.findall(r'"baseUrl":"([^"]+)"', captions_section)
            
            if caption_tracks:
                # Get the first caption track (usually English)
                caption_url = caption_tracks[0].replace('\\u0026', '&')
                caption_response = requests.get(caption_url, headers=headers, timeout=10)
                caption_response.raise_for_status()
                
                # Parse VTT captions
                lines = caption_response.text.split('\n')
                text_lines = [
                    line.strip() for line in lines
                    if line.strip()
                    and not line.startswith('WEBVTT')
                    and not line.startswith('NOTE')
                    and '-->' not in line
                    and not re.match(r'^\d{2}:\d{2}:\d{2}', line)
                ]
                text = ' '.join(text_lines)
                return text.strip() if text.strip() else None
    except Exception as e:
        print(f"[YOUTUBE] Web scrape error: {type(e).__name__}: {e}")
    
    return None


def get_youtube_transcript(url: str) -> Optional[str]:
    """
    Attempt to extract transcript from a YouTube video.
    Tries multiple methods in order:
    1. youtube-transcript-api (primary)
    2. yt-dlp (fallback)
    3. Web scraping (last resort)
    
    Args:
        url: YouTube video URL or video ID
        
    Returns:
        Transcript text if successful, None otherwise
    """
    video_id = _extract_video_id(url)
    if not video_id:
        print(f"[YOUTUBE] Could not extract video ID from: {url}")
        return None
    
    print(f"[YOUTUBE] Attempting to extract transcript for video: {video_id}")
    
    # Try method 1: youtube-transcript-api
    print(f"[YOUTUBE] Trying youtube-transcript-api...")
    transcript = _get_transcript_with_api(video_id)
    if transcript:
        print(f"[YOUTUBE] Success with youtube-transcript-api ({len(transcript)} chars)")
        return transcript
    
    # Try method 2: yt-dlp
    print(f"[YOUTUBE] Trying yt-dlp...")
    transcript = _get_transcript_with_yt_dlp(video_id)
    if transcript:
        print(f"[YOUTUBE] Success with yt-dlp ({len(transcript)} chars)")
        return transcript
    
    # Try method 3: Web scraping
    print(f"[YOUTUBE] Trying web scraping...")
    transcript = _get_transcript_web_scrape(video_id)
    if transcript:
        print(f"[YOUTUBE] Success with web scraping ({len(transcript)} chars)")
        return transcript
    
    print(f"[YOUTUBE] All methods failed for video: {video_id}")
    return None


def is_youtube_url(text: str) -> bool:
    """Check if text looks like a YouTube URL or video ID."""
    if not text:
        return False
    return bool(_extract_video_id(text.strip()))
