"""
API package for the CZSU Multi-Agent Text-to-SQL application.

This package contains the modular structure for the API server,
refactored from the original api_server.py file.
"""

__version__ = "1.0.0"

# Don't import anything during package initialization to avoid import errors
# that could prevent the API server from starting.
# Individual modules will import what they need when they need it.

__all__ = []
