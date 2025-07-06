"""
Authentication package for the API server.

This package contains JWT authentication, Google OAuth verification,
and other authentication mechanisms for the CZSU Multi-Agent Text-to-SQL application.
"""

# Import JWT authentication functions
from .jwt_auth import verify_google_jwt

# Export all authentication functions for easier access
__all__ = [
    'verify_google_jwt'
] 