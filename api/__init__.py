"""
API package for the CZSU Multi-Agent Text-to-SQL application.

This package contains the modular structure for the API server,
refactored from the original api_server.py file.
"""

__version__ = "1.0.0"

# Import commonly used items for easier access
try:
    # Main FastAPI app
    from .main import app
    
    # Authentication
    from .dependencies.auth import get_current_user
    from .auth.jwt_auth import verify_google_jwt
    
    # Models
    from .models.requests import AnalyzeRequest, FeedbackRequest, SentimentRequest
    from .models.responses import ChatMessage, ChatThreadResponse, PaginatedChatThreadsResponse
    
    # Configuration
    from .config.settings import GLOBAL_CHECKPOINTER, GC_MEMORY_THRESHOLD, start_time
    
    __all__ = [
        'app',
        'get_current_user',
        'verify_google_jwt',
        'AnalyzeRequest',
        'FeedbackRequest', 
        'SentimentRequest',
        'ChatMessage',
        'ChatThreadResponse',
        'PaginatedChatThreadsResponse',
        'GLOBAL_CHECKPOINTER',
        'GC_MEMORY_THRESHOLD',
        'start_time'
    ]
    
except ImportError as e:
    # If there are import issues during development, don't break the package
    print(f"Warning: Some API imports failed during package initialization: {e}")
    __all__ = [] 