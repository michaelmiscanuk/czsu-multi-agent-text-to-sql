"""
Data models package for the API server.

This package contains Pydantic models for request/response validation
and data structures for the CZSU Multi-Agent Text-to-SQL application.
"""

# Import request models
from .requests import (
    AnalyzeRequest,
    FeedbackRequest,
    SentimentRequest
)

# Import response models
from .responses import (
    ChatThreadResponse,
    PaginatedChatThreadsResponse,
    ChatMessage
)

# Export all models for easier access
__all__ = [
    # Request models
    'AnalyzeRequest',
    'FeedbackRequest',
    'SentimentRequest',
    
    # Response models
    'ChatThreadResponse',
    'PaginatedChatThreadsResponse',
    'ChatMessage'
] 