from typing import Optional

class HSUAIClipError(Exception):
    """Base exception class for HSUAIClip application."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

class ExecutableNotFoundError(HSUAIClipError):
    """Raised when a required external executable (e.g., ffmpeg, node) is not found."""
    pass

class MediaDownloadError(HSUAIClipError):
    """Raised when media download or metadata extraction fails."""
    pass

class RateLimitError(MediaDownloadError):
    """Raised when download fails due to rate limiting (HTTP 429)."""
    pass

class VideoProcessingError(HSUAIClipError):
    """Raised when video processing (FFmpeg/OpenCV) fails."""
    pass

class AnalysisError(HSUAIClipError):
    """Raised when AI content analysis fails."""
    pass

class AuthenticationError(AnalysisError):
    """Raised when API Key is invalid or expired."""
    pass

class QuotaExceededError(AnalysisError):
    """Raised when API quota/limits are exceeded."""
    pass

class ContentPolicyViolationError(AnalysisError):
    """Raised when content is blocked by safety filters."""
    pass

class TranscriptionError(HSUAIClipError):
    """Raised when audio transcription fails."""
    pass