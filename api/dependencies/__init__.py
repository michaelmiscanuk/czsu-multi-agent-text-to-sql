"""
Dependencies package for the API server.

This package contains FastAPI dependencies for authentication,
validation, and other dependency injection for the CZSU Multi-Agent Text-to-SQL application.
"""

# Import authentication dependencies
from .auth import get_current_user

# Export all dependencies for easier access
__all__ = [
    'get_current_user'
] 