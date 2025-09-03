"""Retry logic and error recovery decorators.

This module provides retry decorators and error recovery mechanisms
for robust error handling in the checkpointer system.
"""

from __future__ import annotations

import functools
import traceback
from typing import Callable, Awaitable

from api.utils.debug import print__checkpointers_debug
from checkpointer.error_handling.prepared_statements import (
    is_prepared_statement_error,
    clear_prepared_statements,
)
from checkpointer.config import DEFAULT_MAX_RETRIES, T
from checkpointer.globals import _GLOBAL_CHECKPOINTER


# This file will contain:
# - retry_on_prepared_statement_error() decorator
def retry_on_prepared_statement_error(max_retries: int = DEFAULT_MAX_RETRIES):
    """Decorator factory for automatic retry logic on prepared statement errors.

    This decorator provides robust error recovery for functions that may encounter
    prepared statement conflicts. It automatically detects such errors, performs
    cleanup operations, and retries the function with fresh connections.

    Args:
        max_retries (int): Maximum number of retry attempts before giving up

    Returns:
        Callable: Decorator function that wraps target functions with retry logic

    Retry Strategy:
        1. Execute the decorated function normally
        2. If prepared statement error detected, perform cleanup:
           - Clear existing prepared statements
           - Recreate global checkpointer if needed
           - Reset connection state
        3. Retry the function with fresh connection state
        4. Continue until success or max retries exceeded

    Recovery Actions:
        - clear_prepared_statements(): Remove conflicting prepared statements
        - Checkpointer recreation: Fresh AsyncPostgresSaver instance
        - Global state reset: Clean slate for retry attempts

    Note:
        - Only retries on confirmed prepared statement errors
        - Maintains original exception for non-recoverable errors
        - Includes detailed debug logging for troubleshooting
        - Graceful handling of cleanup failures during retry
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            print__checkpointers_debug(
                f"202 - RETRY WRAPPER START: Starting {func.__name__} with max_retries={max_retries}"
            )
            last_error = None

            for attempt in range(max_retries + 1):
                print__checkpointers_debug(
                    f"203 - RETRY ATTEMPT: Attempt {attempt + 1}/{max_retries + 1} for {func.__name__}"
                )
                try:
                    result = await func(*args, **kwargs)
                    print__checkpointers_debug(
                        f"204 - RETRY SUCCESS: {func.__name__} succeeded on attempt {attempt + 1}"
                    )
                    return result
                except Exception as e:
                    last_error = e
                    print__checkpointers_debug(
                        f"205 - RETRY ERROR: {func.__name__} failed on attempt {attempt + 1}: {str(e)}"
                    )

                    # Add full traceback for debugging
                    full_traceback = traceback.format_exc()
                    print__checkpointers_debug(
                        f"205.1 - RETRY TRACEBACK: {full_traceback}"
                    )

                    if is_prepared_statement_error(e):
                        print__checkpointers_debug(
                            f"206 - PREPARED STATEMENT ERROR: Detected prepared statement error in {func.__name__}"
                        )

                        if attempt < max_retries:
                            print__checkpointers_debug(
                                f"207 - RETRY CLEANUP: Clearing prepared statements before retry {attempt + 2}"
                            )
                            try:
                                await clear_prepared_statements()
                                # Clear the global checkpointer to force recreation on next access
                                global _GLOBAL_CHECKPOINTER
                                if _GLOBAL_CHECKPOINTER:
                                    print__checkpointers_debug(
                                        "208 - CHECKPOINTER RECREATION: Clearing checkpointer due to prepared statement error"
                                    )
                                    # Close the existing checkpointer
                                    try:
                                        if (
                                            hasattr(_GLOBAL_CHECKPOINTER, "pool")
                                            and _GLOBAL_CHECKPOINTER.pool
                                        ):
                                            await _GLOBAL_CHECKPOINTER.pool.close()
                                    except Exception as close_error:
                                        print__checkpointers_debug(
                                            f"209 - CLOSE ERROR: Error closing checkpointer pool: {close_error}"
                                        )

                                    # Clear the global reference to force recreation
                                    _GLOBAL_CHECKPOINTER = None
                                    print__checkpointers_debug(
                                        "210 - CHECKPOINTER CLEARED: Global checkpointer cleared for recreation"
                                    )
                            except Exception as cleanup_error:
                                print__checkpointers_debug(
                                    f"211 - CLEANUP ERROR: Error during cleanup: {cleanup_error}"
                                )
                            continue

                    # If it's not a prepared statement error, or we've exhausted retries, re-raise
                    print__checkpointers_debug(
                        f"212 - RETRY EXHAUSTED: No more retries for {func.__name__}, re-raising error"
                    )
                    raise

            # This should never be reached, but just in case
            print__checkpointers_debug(
                f"213 - RETRY FALLBACK: Fallback error re-raise for {func.__name__}"
            )
            raise last_error

        return wrapper

    return decorator
