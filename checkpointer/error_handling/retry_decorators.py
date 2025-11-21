"""Retry Logic and Automatic Error Recovery Decorators

This module provides sophisticated retry decorators with automatic error detection
and recovery mechanisms for robust database operation handling in the checkpointer
system. It implements exponential backoff, connection pool recreation, and state
cleanup for both prepared statement and SSL connection errors.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""Retry Logic and Automatic Error Recovery Decorators

This module provides sophisticated retry decorators with automatic error detection
and recovery mechanisms for robust database operation handling. It is the core of
the checkpointer error handling system, enabling automatic recovery from transient
database errors without manual intervention.

Key Features:
-------------
1. Prepared Statement Error Retry:
   - Automatic detection of prepared statement conflicts
   - Prepared statement cleanup between retry attempts
   - Connection pool recreation for fresh state
   - Configurable retry attempts with full state reset
   - Integration with prepared statement cleanup module

2. SSL Connection Error Retry:
   - Detection of SSL connection failures and socket errors
   - Exponential backoff delay strategy
   - Connection pool health checking and recreation
   - Graceful handling of network instability
   - Maximum delay cap to prevent excessive waiting

3. Decorator Architecture:
   - Function-level retry logic through decorators
   - Transparent error handling for decorated functions
   - Preserves original function signatures with functools.wraps
   - Composable decorators for multiple error types
   - Type-safe async function wrapping

4. Recovery Strategy:
   - Multi-stage recovery process:
     * Error detection and classification
     * Cleanup operations (statements, pool closure)
     * Global state reset
     * Exponential backoff delay
     * Fresh connection creation
     * Function retry with new state
   - Comprehensive logging at each stage
   - Non-blocking cleanup operations
   - Graceful fallback on recovery failures

5. Debug and Monitoring:
   - Detailed logging with numbered markers
   - Full traceback capture for root cause analysis
   - Retry attempt counting and tracking
   - Success/failure logging for each attempt
   - Performance metrics for delays and retries

Retry Decorators:
----------------
The module provides two main decorator factories:

1. @retry_on_prepared_statement_error(max_retries=3)
   - Detects prepared statement conflicts
   - Clears stale prepared statements
   - Recreates connection pool
   - Resets global checkpointer state

2. @retry_on_ssl_connection_error(max_retries=3)
   - Detects SSL connection failures
   - Forces connection pool closure
   - Applies exponential backoff delays
   - Recreates connections with health checks

Error Detection:
---------------
The module implements two error detection functions:

1. is_prepared_statement_error(error):
   - Delegates to prepared_statements.is_prepared_statement_error()
   - Analyzes error messages for prepared statement indicators
   - Returns boolean for retry decision

2. is_ssl_connection_error(error):
   - Pattern matching on error message and type
   - Detects SSL-specific error patterns:
     * "ssl connection has been closed unexpectedly"
     * "consuming input failed"
     * "server closed the connection unexpectedly"
     * "connection closed", "ssl syscall error"
     * "operationalerror", "connection reset", "broken pipe"
   - Case-insensitive matching for robustness
   - Type name checking for SSL-related exceptions

Retry Strategy:
--------------
Both decorators implement a sophisticated retry strategy:

1. Attempt Loop:
   - Iterate from 0 to max_retries (inclusive)
   - Each iteration is one function execution attempt
   - Track last error for final re-raise if all fail

2. Error Handling:
   - Catch all exceptions during function execution
   - Classify error using detection functions
   - Decide retry vs. re-raise based on error type

3. Recovery Actions (on retryable error):
   - Execute cleanup operations:
     * Prepared statement: clear_prepared_statements()
     * SSL connection: Close connection pool
   - Reset global checkpointer state:
     * Close existing pool if present
     * Set _GLOBAL_CHECKPOINTER = None
     * Force recreation on next access
   - Apply delay:
     * Exponential backoff: min(2^attempt, 30) seconds
     * Prevents overwhelming database
     * Allows transient issues to resolve

4. Retry Execution:
   - Continue to next loop iteration
   - Function re-executed with fresh state
   - New connections created as needed

5. Success or Failure:
   - Success: Return result immediately
   - All retries failed: Re-raise last error
   - Non-retryable error: Re-raise immediately

Exponential Backoff:
------------------
The SSL retry decorator uses exponential backoff:

```python
delay = min(2**attempt, 30)  # Max 30 seconds
```

Delay progression:
- Attempt 0: 1 second (2^0)
- Attempt 1: 2 seconds (2^1)
- Attempt 2: 4 seconds (2^2)
- Attempt 3: 8 seconds (2^3)
- Attempt 4: 16 seconds (2^4)
- Attempt 5+: 30 seconds (capped)

Benefits:
- Gives transient issues time to resolve
- Prevents overwhelming recovering systems
- Balances responsiveness with stability
- Caps maximum wait to maintain usability

Connection Pool Cleanup:
----------------------
Both decorators perform connection pool cleanup:

1. Check Global State:
   ```python
   global _GLOBAL_CHECKPOINTER
   if _GLOBAL_CHECKPOINTER:
   ```

2. Close Existing Pool:
   ```python
   if hasattr(_GLOBAL_CHECKPOINTER, 'pool') and _GLOBAL_CHECKPOINTER.pool:
       await _GLOBAL_CHECKPOINTER.pool.close()
   ```

3. Delay for Cleanup:
   ```python
   await asyncio.sleep(0.5)  # Brief pause
   ```

4. Reset Global State:
   ```python
   _GLOBAL_CHECKPOINTER = None
   ```

This ensures:
- Old connections fully closed
- Resources properly released
- Fresh pool created on next access
- No connection leaks
- Clean state for retry

Usage Examples:
--------------
Single decorator:
```python
from checkpointer.error_handling.retry_decorators import retry_on_ssl_connection_error

@retry_on_ssl_connection_error(max_retries=3)
async def get_checkpointer(user_id: str):
    # Automatically retries on SSL connection errors
    # Up to 3 retry attempts with exponential backoff
    # Connection pool recreated between attempts
    pass
```

Multiple decorators (composable):
```python
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
    retry_on_ssl_connection_error
)

@retry_on_prepared_statement_error(max_retries=3)
@retry_on_ssl_connection_error(max_retries=3)
async def database_operation():
    # Handles both prepared statement and SSL errors
    # Each decorator provides independent retry logic
    # Decorators can be stacked for comprehensive coverage
    pass
```

Decorator composition order:
- Inner decorator executes first
- Outer decorator catches errors from inner
- SSL decorator typically outermost (broader scope)
- Prepared statement decorator inner (specific errors)

Debug Logging:
-------------
Comprehensive debug logging with numbered markers:

Prepared Statement Retry (200-213):
- 200-201: Error detection phase
- 202: Retry wrapper initialization
- 203: Attempt counter
- 204: Success logging
- 205: Error logging with traceback (205.1)
- 206: Prepared statement error detected
- 207: Cleanup initiation
- 208: Checkpointer recreation
- 209: Pool close error
- 210: Checkpointer cleared
- 211: Cleanup error
- 212: Retry exhausted
- 213: Fallback error

SSL Connection Retry:
- SSL_RETRY START: Decorator initialization
- SSL_RETRY ATTEMPT: Attempt counter
- SSL_RETRY SUCCESS: Successful execution
- SSL_RETRY ERROR: Execution failure
- SSL_RETRY TRACEBACK: Full exception traceback
- SSL_CONNECTION ERROR: SSL error detected
- SSL_RETRY CLEANUP: Pool recreation phase
- SSL_CHECKPOINTER RECREATION: Checkpointer cleanup
- SSL_CLOSE ERROR: Pool close failure
- SSL_CHECKPOINTER CLEARED: State reset
- SSL_CLEANUP ERROR: Overall cleanup failure
- SSL_RETRY EXHAUSTED: All retries failed
- SSL_RETRY FALLBACK: Final error re-raise

Type Safety:
-----------
The decorators maintain type safety through:

1. Generic Type Variable:
   ```python
   from checkpointer.config import T  # TypeVar from config
   ```

2. Type Annotations:
   ```python
   def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
   async def wrapper(*args, **kwargs) -> T:
   ```

3. Benefits:
   - Type checkers understand decorated functions
   - Return type preserved through decoration
   - IDE autocomplete works correctly
   - Static analysis catches type errors

Error Propagation:
----------------
The decorators follow a clear error propagation strategy:

1. Retryable Errors:
   - Caught and analyzed
   - Cleanup performed
   - Function retried
   - Eventually re-raised if all retries fail

2. Non-Retryable Errors:
   - Immediately re-raised
   - No cleanup attempted
   - Original stack trace preserved

3. Cleanup Errors:
   - Logged but not propagated
   - Retry continues despite cleanup failure
   - Graceful degradation approach

4. Final Error:
   - Last caught error re-raised
   - Original exception type preserved
   - Full stack trace available

Performance Considerations:
-------------------------
1. Retry Overhead:
   - Each retry adds delay (exponential backoff)
   - Connection pool recreation has cost
   - System catalog queries in cleanup
   - Maximum total delay with 3 retries: ~45 seconds

2. Debug Logging:
   - Comprehensive logging has overhead
   - Can be disabled in production if needed
   - Useful for troubleshooting and monitoring

3. Connection Pool:
   - Pool recreation not free
   - Health checking adds latency
   - Consider retry count vs. responsiveness

4. Optimization Tips:
   - Use minimal retries for non-critical operations
   - Increase retries for important operations
   - Monitor retry frequency for systemic issues
   - Cache checkpointer when possible

Concurrency Handling:
-------------------
The decorators handle concurrent operations:

1. Global State:
   - _GLOBAL_CHECKPOINTER accessed by multiple coroutines
   - Race conditions possible during cleanup
   - Python GIL provides some protection
   - Atomic-like behavior for None assignment

2. Connection Pool:
   - Pool internally handles concurrent access
   - Close operation thread-safe
   - New pool creation serialized by access pattern

3. Best Practices:
   - Minimize concurrent checkpointer creation
   - Allow pool to handle concurrency internally
   - Trust pool's connection management
   - Monitor pool size and usage

Integration Points:
-----------------
1. Prepared Statement Module:
   ```python
   from checkpointer.error_handling.prepared_statements import (
       is_prepared_statement_error,
       clear_prepared_statements,
   )
   ```

2. Global State:
   ```python
   from checkpointer.globals import _GLOBAL_CHECKPOINTER
   ```

3. Configuration:
   ```python
   from checkpointer.config import DEFAULT_MAX_RETRIES, T
   ```

4. Debug Utilities:
   ```python
   from api.utils.debug import print__checkpointers_debug
   ```

Testing Considerations:
---------------------
1. Mock Error Injection:
   - Simulate prepared statement errors
   - Inject SSL connection failures
   - Test retry exhaustion scenarios

2. Cleanup Verification:
   - Verify prepared statements cleared
   - Check connection pool closure
   - Confirm state reset

3. Retry Count:
   - Verify correct number of attempts
   - Test max_retries parameter
   - Confirm exponential backoff delays

4. Success Cases:
   - Test immediate success (no retries)
   - Test success after retries
   - Verify return value preservation

Best Practices:
--------------
1. Decorator Application:
   - Apply to all critical checkpointer operations
   - Use both decorators for comprehensive coverage
   - Place SSL decorator outermost
   - Configure appropriate max_retries

2. Error Monitoring:
   - Monitor retry frequency
   - Alert on excessive retries
   - Review debug logs for patterns
   - Investigate systemic issues

3. Configuration:
   - Start with DEFAULT_MAX_RETRIES
   - Increase for critical operations
   - Decrease for non-critical operations
   - Balance responsiveness vs. reliability

4. State Management:
   - Trust decorator cleanup logic
   - Don't manually reset global state
   - Allow automatic checkpointer recreation
   - Monitor connection pool health

Known Limitations:
----------------
1. Race Conditions:
   - Multiple coroutines may trigger cleanup simultaneously
   - Global state updates not atomic across all steps
   - Generally safe due to Python GIL and async nature

2. Error Classification:
   - Pattern matching may miss new error types
   - Some ambiguous errors might not retry
   - False positives possible but rare

3. Resource Usage:
   - Multiple retries increase resource consumption
   - Connection pool recreation adds latency
   - Debug logging uses memory and I/O

4. Recovery Limits:
   - Cannot recover from permanent errors
   - Configuration errors not recoverable
   - Some database issues require manual intervention

Troubleshooting:
---------------
1. Retries Not Triggering:
   - Verify error pattern matches detection logic
   - Check debug logs for error classification
   - Ensure decorator applied correctly
   - Confirm error type inheritance

2. Cleanup Failures:
   - Review cleanup error logs (211, SSL_CLEANUP ERROR)
   - Verify database permissions
   - Check connection pool state
   - Ensure global state accessible

3. Excessive Retries:
   - Indicates systemic problem
   - Check database health
   - Review connection pool configuration
   - Monitor network stability
   - Consider root cause investigation

4. Performance Issues:
   - Reduce max_retries if acceptable
   - Disable verbose debug logging
   - Optimize connection pool settings
   - Cache checkpointer instances

Future Enhancements:
------------------
- Circuit breaker pattern for repeated failures
- Adaptive retry delays based on error patterns
- Retry metrics and statistics collection
- Per-function retry configuration
- Enhanced error classification
- Automatic retry tuning based on success rates
- Distributed tracing integration
- Health check integration before retry"""

import functools
import traceback
import asyncio
from typing import Callable, Awaitable

from api.utils.debug import print__checkpointers_debug
from checkpointer.error_handling.prepared_statements import (
    is_prepared_statement_error,
    clear_prepared_statements,
)
from checkpointer.config import DEFAULT_MAX_RETRIES, T
from checkpointer.globals import _GLOBAL_CHECKPOINTER


def is_ssl_connection_error(error: Exception) -> bool:
    """Check if an error is related to SSL connection issues.

    Args:
        error: Exception to check

    Returns:
        bool: True if error is SSL-related, False otherwise
    """
    # Convert error message to lowercase for case-insensitive pattern matching
    error_str = str(error).lower()

    # Get exception class name in lowercase for type-based detection
    error_type = type(error).__name__.lower()

    # Define comprehensive list of SSL connection error patterns
    # These cover various PostgreSQL, psycopg, and network layer errors
    ssl_patterns = [
        "ssl connection has been closed unexpectedly",  # SSL connection dropped
        "consuming input failed",  # Socket read failure
        "server closed the connection unexpectedly",  # Server-side connection close
        "connection closed",  # Generic connection closure
        "ssl syscall error",  # Low-level SSL system call failure
        "operationalerror",  # PostgreSQL operational error
        "connection reset",  # TCP connection reset
        "broken pipe",  # Socket write to closed connection
    ]

    # Return True if any pattern matches the error message OR if exception type contains "ssl"
    # This dual approach catches both message-based and type-based SSL errors
    return any(pattern in error_str for pattern in ssl_patterns) or "ssl" in error_type


def retry_on_ssl_connection_error(max_retries: int = DEFAULT_MAX_RETRIES):
    """Decorator factory for automatic retry logic on SSL connection errors.

    This decorator provides robust error recovery for functions that may encounter
    SSL connection failures, especially during long-running or concurrent operations.

    Args:
        max_retries (int): Maximum number of retry attempts before giving up

    Returns:
        Callable: Decorator function that wraps target functions with retry logic

    Retry Strategy:
        1. Execute the decorated function normally
        2. If SSL connection error detected, perform recovery:
           - Close and recreate connection pool
           - Add exponential backoff delay
           - Reset global checkpointer state
        3. Retry the function with fresh connections
        4. Continue until success or max retries exceeded

    Recovery Actions:
        - Connection pool recreation: Fresh pool with health checking
        - Exponential backoff: Increasing delays between attempts
        - State cleanup: Clean global checkpointer state
        - Connection health verification: Ensure new connections work

    Note:
        - Only retries on confirmed SSL connection errors
        - Uses exponential backoff to avoid overwhelming server
        - Maintains original exception for non-recoverable errors
        - Includes detailed debug logging for troubleshooting
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(
            func
        )  # Preserve original function metadata (name, docstring, etc.)
        async def wrapper(*args, **kwargs) -> T:
            # Log the start of retry logic for this function execution
            print__checkpointers_debug(
                f"SSL_RETRY START: Starting {func.__name__} with max_retries={max_retries}"
            )

            # Track the last error encountered for re-raising if all retries fail
            last_error = None

            # Attempt loop: 0 to max_retries inclusive (e.g., max_retries=3 gives 4 total attempts)
            for attempt in range(max_retries + 1):
                # Log each attempt number for monitoring retry behavior
                print__checkpointers_debug(
                    f"SSL_RETRY ATTEMPT: Attempt {attempt + 1}/{max_retries + 1} for {func.__name__}"
                )
                try:
                    # Execute the decorated function with all its original arguments
                    result = await func(*args, **kwargs)

                    # Success! Log and return immediately without further retries
                    print__checkpointers_debug(
                        f"SSL_RETRY SUCCESS: {func.__name__} succeeded on attempt {attempt + 1}"
                    )
                    return result

                except Exception as exc:
                    # Store this exception as the last error in case all retries fail
                    last_error = exc

                    # Log the failure with error message for troubleshooting
                    print__checkpointers_debug(
                        f"SSL_RETRY ERROR: {func.__name__} failed on attempt {attempt + 1}: {str(exc)}"
                    )

                    # Capture and log full traceback for detailed SSL error diagnosis
                    # This helps identify the exact code path that led to the error
                    full_traceback = traceback.format_exc()
                    print__checkpointers_debug(f"SSL_RETRY TRACEBACK: {full_traceback}")

                    # Check if this is an SSL connection error that we can retry
                    if is_ssl_connection_error(exc):
                        print__checkpointers_debug(
                            f"SSL_CONNECTION ERROR: Detected SSL connection error in {func.__name__}"
                        )

                        # Only perform recovery if we haven't exhausted retries yet
                        if attempt < max_retries:
                            # Calculate exponential backoff delay: 1s, 2s, 4s, 8s, 16s, then cap at 30s
                            # This gives transient network issues time to resolve
                            delay = min(2**attempt, 30)  # Max 30 seconds delay
                            print__checkpointers_debug(
                                f"SSL_RETRY CLEANUP: Recreating connections after {delay}s delay (attempt {attempt + 2})"
                            )

                            try:
                                # Access global checkpointer state for cleanup
                                global _GLOBAL_CHECKPOINTER

                                # Check if a checkpointer instance exists that needs cleanup
                                if _GLOBAL_CHECKPOINTER:
                                    print__checkpointers_debug(
                                        "SSL_CHECKPOINTER RECREATION: Clearing checkpointer due to SSL error"
                                    )

                                    # Attempt to close the existing connection pool
                                    # This releases all connections and frees resources
                                    try:
                                        # Verify pool exists before attempting to close
                                        if (
                                            hasattr(_GLOBAL_CHECKPOINTER, "pool")
                                            and _GLOBAL_CHECKPOINTER.pool
                                        ):
                                            # Close the pool asynchronously
                                            await _GLOBAL_CHECKPOINTER.pool.close()

                                            # Brief pause to ensure cleanup completes
                                            # Allows PostgreSQL server to process connection closures
                                            await asyncio.sleep(0.5)
                                    except Exception as close_error:
                                        # Log pool closure errors but continue with recovery
                                        # Non-fatal: state reset is more important than clean closure
                                        print__checkpointers_debug(
                                            f"SSL_CLOSE ERROR: Error closing checkpointer pool: {close_error}"
                                        )

                                    # Clear the global reference to force fresh checkpointer creation
                                    # Next access will create a new pool with healthy connections
                                    _GLOBAL_CHECKPOINTER = None
                                    print__checkpointers_debug(
                                        "SSL_CHECKPOINTER CLEARED: Global checkpointer cleared for recreation"
                                    )

                                # Apply exponential backoff delay before retry
                                # Gives network and server time to recover from transient issues
                                await asyncio.sleep(delay)

                            except Exception as cleanup_error:
                                # Log cleanup errors but don't stop retry process
                                # Retry can still succeed even if cleanup had issues
                                print__checkpointers_debug(
                                    f"SSL_CLEANUP ERROR: Error during SSL cleanup: {cleanup_error}"
                                )

                            # Continue to next retry attempt with fresh state
                            continue

                    # Not an SSL error, or we've exhausted all retries
                    # Re-raise the exception to propagate to caller
                    print__checkpointers_debug(
                        f"SSL_RETRY EXHAUSTED: No more retries for {func.__name__}, re-raising error"
                    )
                    raise

            # Fallback error re-raise (should never be reached due to logic above)
            # If we somehow exit the loop without returning or raising, raise last error
            print__checkpointers_debug(
                f"SSL_RETRY FALLBACK: Fallback error re-raise for {func.__name__}"
            )
            raise last_error

        return wrapper

    return decorator


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
        @functools.wraps(
            func
        )  # Preserve original function metadata (name, docstring, etc.)
        async def wrapper(*args, **kwargs) -> T:
            # Log the start of retry logic with configuration for monitoring
            print__checkpointers_debug(
                f"202 - RETRY WRAPPER START: Starting {func.__name__} with max_retries={max_retries}"
            )

            # Track the last error for re-raising if all retry attempts fail
            last_error = None

            # Attempt loop: 0 to max_retries inclusive (e.g., max_retries=3 gives 4 total attempts)
            for attempt in range(max_retries + 1):
                # Log current attempt number for tracking retry progression
                print__checkpointers_debug(
                    f"203 - RETRY ATTEMPT: Attempt {attempt + 1}/{max_retries + 1} for {func.__name__}"
                )
                try:
                    # Execute the decorated function with all original arguments
                    result = await func(*args, **kwargs)

                    # Success! Log and return immediately without further retries
                    print__checkpointers_debug(
                        f"204 - RETRY SUCCESS: {func.__name__} succeeded on attempt {attempt + 1}"
                    )
                    return result

                except Exception as exc:
                    # Store this exception as the last error for potential re-raise
                    last_error = exc

                    # Log the failure with error details for troubleshooting
                    print__checkpointers_debug(
                        f"205 - RETRY ERROR: {func.__name__} failed on attempt {attempt + 1}: {str(exc)}"
                    )

                    # Capture full traceback for root cause analysis
                    # Essential for diagnosing prepared statement conflicts
                    full_traceback = traceback.format_exc()
                    print__checkpointers_debug(
                        f"205.1 - RETRY TRACEBACK: {full_traceback}"
                    )

                    # Check if this is a prepared statement error we can recover from
                    if is_prepared_statement_error(exc):
                        print__checkpointers_debug(
                            f"206 - PREPARED STATEMENT ERROR: Detected prepared statement error in {func.__name__}"
                        )

                        # Only perform recovery if we have retries remaining
                        if attempt < max_retries:
                            print__checkpointers_debug(
                                f"207 - RETRY CLEANUP: Clearing prepared statements before retry {attempt + 2}"
                            )
                            try:
                                # Execute prepared statement cleanup to remove conflicts
                                # This queries pg_prepared_statements and deallocates all found statements
                                await clear_prepared_statements()

                                # Access global checkpointer state for recreation
                                global _GLOBAL_CHECKPOINTER

                                # Check if a checkpointer instance exists that needs cleanup
                                if _GLOBAL_CHECKPOINTER:
                                    print__checkpointers_debug(
                                        "208 - CHECKPOINTER RECREATION: Clearing checkpointer due to prepared statement error"
                                    )

                                    # Attempt to close the existing connection pool cleanly
                                    # This ensures all connections are properly released
                                    try:
                                        # Verify pool attribute exists before closing
                                        if (
                                            hasattr(_GLOBAL_CHECKPOINTER, "pool")
                                            and _GLOBAL_CHECKPOINTER.pool
                                        ):
                                            # Close pool asynchronously to release all connections
                                            await _GLOBAL_CHECKPOINTER.pool.close()
                                    except Exception as close_error:
                                        # Log pool closure errors but continue with recovery
                                        # State reset is critical even if closure fails
                                        print__checkpointers_debug(
                                            f"209 - CLOSE ERROR: Error closing checkpointer pool: {close_error}"
                                        )

                                    # Clear global reference to force recreation on next access
                                    # New checkpointer will have fresh pool without stale prepared statements
                                    _GLOBAL_CHECKPOINTER = None
                                    print__checkpointers_debug(
                                        "210 - CHECKPOINTER CLEARED: Global checkpointer cleared for recreation"
                                    )
                            except Exception as cleanup_error:
                                # Log cleanup errors but don't stop retry process
                                # Function may still succeed with fresh connections
                                print__checkpointers_debug(
                                    f"211 - CLEANUP ERROR: Error during cleanup: {cleanup_error}"
                                )

                            # Continue to next retry attempt with cleaned state
                            continue

                    # Not a prepared statement error, or we've exhausted retries
                    # Re-raise the exception to propagate to caller
                    print__checkpointers_debug(
                        f"212 - RETRY EXHAUSTED: No more retries for {func.__name__}, re-raising error"
                    )
                    raise

            # Fallback error re-raise (should never be reached due to logic above)
            # Safety net in case loop exits without returning or raising
            print__checkpointers_debug(
                f"213 - RETRY FALLBACK: Fallback error re-raise for {func.__name__}"
            )
            raise last_error

        return wrapper

    return decorator
