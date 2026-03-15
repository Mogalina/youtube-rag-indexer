from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def get_video_id(url: str) -> str:
    """
    Get the video identifier from a YouTube video URL.

    Args:
        url: The YouTube video URL
    
    Returns:
        The video identifier
    """
    query = parse_qs(urlparse(url).query)
    return query["v"][0]


def get_transcript(url: str, language: str = "en") -> str:
    """
    Get the full transcript of a YouTube video.

    Args:
        url: The YouTube video URL
        language: Optional language of the transcript
    
    Returns:
        The full transcript of the video
    """
    video_id = get_video_id(url)

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)

    try:
        transcript = transcript_list.find_transcript([language])
    except:
        transcript = transcript_list.find_transcript(['en']).translate(language)

    # Fetch the transcript segments
    segments = transcript.fetch()

    # Join the segments into a single string
    full_text = " ".join(segment.text for segment in segments)

    return full_text
