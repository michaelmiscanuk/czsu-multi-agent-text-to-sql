"""
Retry utilities for handling rate limiting and transient errors in evaluations.

This module provides robust retry mechanisms with exponential backoff for handling
rate limiting errors from various LLM providers (Mistral, OpenAI, Anthropic, etc.).
"""

import asyncio
import time
import random
from functools import wraps
from typing import TypeVar, Callable, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function signatures
T = TypeVar("T")

# Rate limit error classes from various providers
RATE_LIMIT_ERRORS = (
    # OpenAI/Azure OpenAI
    Exception,  # Catch all for now - will be refined below
)

# Try to import specific rate limit errors
try:
    from openai import RateLimitError as OpenAIRateLimitError

    RATE_LIMIT_ERRORS += (OpenAIRateLimitError,)
except ImportError:
    pass

try:
    from anthropic import RateLimitError as AnthropicRateLimitError

    RATE_LIMIT_ERRORS += (AnthropicRateLimitError,)
except ImportError:
    pass

try:
    from mistralai.exceptions import MistralAPIException

    RATE_LIMIT_ERRORS += (MistralAPIException,)
except ImportError:
    pass

try:
    from google.api_core.exceptions import ResourceExhausted

    RATE_LIMIT_ERRORS += (ResourceExhausted,)
except ImportError:
    pass


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is a rate limit error.

    Args:
        error: Exception to check

    Returns:
        True if error is a rate limit error, False otherwise
    """
    # Check by exception type
    if isinstance(error, RATE_LIMIT_ERRORS):
        return True

    # Check by error message (fallback for unknown providers)
    error_msg = str(error).lower()
    rate_limit_keywords = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "quota_exceeded",
        "resource exhausted",
        "throttled",
        "throttling",
    ]
    return any(keyword in error_msg for keyword in rate_limit_keywords)


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 300.0 = 5 minutes)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)

    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = min(base_delay * (exponential_base**attempt), max_delay)

    # Add jitter (random factor between 0.5 and 1.5)
    if jitter:
        jitter_factor = 0.5 + random.random()  # Random between 0.5 and 1.5
        delay *= jitter_factor

    return delay


def retry_with_exponential_backoff(
    max_attempts: int = 30,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    log_retries: bool = True,
):
    """
    Decorator for retrying async functions with exponential backoff.

    This decorator will retry the decorated function up to max_attempts times,
    with exponentially increasing delays between attempts. It specifically handles
    rate limiting errors from various LLM providers.

    Args:
        max_attempts: Maximum number of attempts (default: 30)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 300.0 = 5 minutes)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        log_retries: Log retry attempts (default: True)

    Example:
        ```python
        @retry_with_exponential_backoff(max_attempts=30)
        async def call_llm():
            return await llm.ainvoke("Hello")
        ```
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if it's a rate limit error
                    if is_rate_limit_error(e):
                        if (
                            attempt < max_attempts - 1
                        ):  # Don't wait after the last attempt
                            delay = calculate_backoff_delay(
                                attempt, base_delay, max_delay, exponential_base, jitter
                            )

                            if log_retries:
                                logger.warning(
                                    f"Rate limit error in {func.__name__} (attempt {attempt + 1}/{max_attempts}). "
                                    f"Retrying in {delay:.2f}s... Error: {str(e)[:100]}"
                                )

                            await asyncio.sleep(delay)
                            continue
                        else:
                            if log_retries:
                                logger.error(
                                    f"Rate limit error in {func.__name__} - all {max_attempts} attempts exhausted. "
                                    f"Error: {str(e)[:100]}"
                                )
                    else:
                        # Not a rate limit error - re-raise immediately
                        if log_retries:
                            logger.error(
                                f"Non-rate-limit error in {func.__name__}: {type(e).__name__}: {str(e)[:100]}"
                            )
                        raise

            # If we get here, all attempts failed
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if it's a rate limit error
                    if is_rate_limit_error(e):
                        if (
                            attempt < max_attempts - 1
                        ):  # Don't wait after the last attempt
                            delay = calculate_backoff_delay(
                                attempt, base_delay, max_delay, exponential_base, jitter
                            )

                            if log_retries:
                                logger.warning(
                                    f"Rate limit error in {func.__name__} (attempt {attempt + 1}/{max_attempts}). "
                                    f"Retrying in {delay:.2f}s... Error: {str(e)[:100]}"
                                )

                            time.sleep(delay)
                            continue
                        else:
                            if log_retries:
                                logger.error(
                                    f"Rate limit error in {func.__name__} - all {max_attempts} attempts exhausted. "
                                    f"Error: {str(e)[:100]}"
                                )
                    else:
                        # Not a rate limit error - re-raise immediately
                        if log_retries:
                            logger.error(
                                f"Non-rate-limit error in {func.__name__}: {type(e).__name__}: {str(e)[:100]}"
                            )
                        raise

            # If we get here, all attempts failed
            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience function for manual retry logic
async def retry_async_call(
    func: Callable[..., Any],
    *args,
    max_attempts: int = 30,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    log_retries: bool = True,
    **kwargs,
) -> Any:
    """
    Retry an async function call with exponential backoff.

    This is a convenience function for retrying calls without using a decorator.

    Args:
        func: Async function to call
        *args: Positional arguments to pass to func
        max_attempts: Maximum number of attempts (default: 30)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 300.0 = 5 minutes)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter (default: True)
        log_retries: Log retry attempts (default: True)
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result from successful function call

    Example:
        ```python
        result = await retry_async_call(
            llm.ainvoke,
            "Hello",
            max_attempts=30
        )
        ```
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            if is_rate_limit_error(e):
                if attempt < max_attempts - 1:
                    delay = calculate_backoff_delay(
                        attempt, base_delay, max_delay, exponential_base, jitter
                    )

                    if log_retries:
                        logger.warning(
                            f"Rate limit error (attempt {attempt + 1}/{max_attempts}). "
                            f"Retrying in {delay:.2f}s... Error: {str(e)[:100]}"
                        )

                    await asyncio.sleep(delay)
                    continue
                else:
                    if log_retries:
                        logger.error(
                            f"Rate limit error - all {max_attempts} attempts exhausted. "
                            f"Error: {str(e)[:100]}"
                        )
            else:
                if log_retries:
                    logger.error(
                        f"Non-rate-limit error: {type(e).__name__}: {str(e)[:100]}"
                    )
                raise

    raise last_exception
