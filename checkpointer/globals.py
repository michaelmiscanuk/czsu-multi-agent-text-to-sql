"""Global state management and type definitions for PostgreSQL checkpointer.

This module manages global state variables and provides type definitions
used throughout the checkpointer system.
"""

from __future__ import annotations

# This file will contain:
# - _GLOBAL_CHECKPOINTER global variable
# - _CONNECTION_STRING_CACHE global variable
# - _CHECKPOINTER_INIT_LOCK global variable
# - TypeVar definitions
# - BASE_DIR calculation

# Global checkpointer instance (AsyncPostgresSaver or None)
_GLOBAL_CHECKPOINTER = None

# Connection string cache to avoid recalculating
_CONNECTION_STRING_CACHE = None

# Async lock for checkpointer initialization
_CHECKPOINTER_INIT_LOCK = None
