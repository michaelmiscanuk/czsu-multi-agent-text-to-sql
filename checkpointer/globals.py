"""Global state management and type definitions for PostgreSQL checkpointer.

This module manages global state variables and provides type definitions
used throughout the checkpointer system.
"""

# This file will contain:
# - _GLOBAL_CHECKPOINTER global variable
# - _CONNECTION_STRING_CACHE global variable
# - _CHECKPOINTER_INIT_LOCK global variable
# - TypeVar definitions
# - BASE_DIR calculation
