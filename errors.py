"""
Custom exceptions for Voice Assistant.
Provides structured error handling across the application.
"""


class VoiceAssistantError(Exception):
    """Base exception for all voice assistant errors."""
    
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert error to dictionary for JSON response."""
        return {
            "type": "error",
            "code": self.code,
            "message": self.message
        }


class ConfigurationError(VoiceAssistantError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR")


class ServiceConnectionError(VoiceAssistantError):
    """Raised when connection to external service fails."""
    
    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(f"{service}: {message}", f"{service.upper()}_CONNECTION_ERROR")


class STTError(VoiceAssistantError):
    """Raised when Speech-to-Text service encounters an error."""
    
    def __init__(self, message: str):
        super().__init__(message, "STT_ERROR")


class LLMError(VoiceAssistantError):
    """Raised when Language Model service encounters an error."""
    
    def __init__(self, message: str):
        super().__init__(message, "LLM_ERROR")


class TTSError(VoiceAssistantError):
    """Raised when Text-to-Speech service encounters an error."""
    
    def __init__(self, message: str):
        super().__init__(message, "TTS_ERROR")


class SessionError(VoiceAssistantError):
    """Raised when voice session encounters an error."""
    
    def __init__(self, message: str):
        super().__init__(message, "SESSION_ERROR")


class AudioProcessingError(VoiceAssistantError):
    """Raised when audio processing fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "AUDIO_PROCESSING_ERROR")


class RateLimitError(VoiceAssistantError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, service: str, retry_after: int = None):
        message = f"Rate limit exceeded for {service}"
        if retry_after:
            message += f". Retry after {retry_after} seconds."
        self.service = service
        self.retry_after = retry_after
        super().__init__(message, "RATE_LIMIT_ERROR")
