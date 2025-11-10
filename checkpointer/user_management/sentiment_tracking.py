"""Sentiment and metadata operations for user feedback tracking.

This module handles user feedback and sentiment tracking
for the PostgreSQL checkpointer system.
"""

from __future__ import annotations

from typing import Dict

from api.utils.debug import print__checkpointers_debug
from checkpointer.database.connection import get_direct_connection
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
)
from checkpointer.config import DEFAULT_MAX_RETRIES


# This file will contain:
# - update_thread_run_sentiment() function
# - get_thread_run_sentiments() function
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def update_thread_run_sentiment(run_id: str, sentiment: bool) -> bool:
    """Update sentiment for a thread run by run_id with retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(f"Updating sentiment for run {run_id}: {sentiment}")
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s
                """,
                    (sentiment, run_id),
                )
                updated = cur.rowcount
        print__checkpointers_debug(f"Updated sentiment for {updated} entries")
        return int(updated) > 0
    except Exception as exc:
        print__checkpointers_debug(f"Failed to update sentiment: {exc}")
        return False


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get all sentiments for a thread with retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(f"Getting sentiments for thread {thread_id}")
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT run_id, sentiment 
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s AND sentiment IS NOT NULL
                """,
                    (email, thread_id),
                )
                rows = await cur.fetchall()
        sentiments = {row[0]: row[1] for row in rows}
        print__checkpointers_debug(f"Retrieved {len(sentiments)} sentiments")
        return sentiments
    except Exception as exc:
        print__checkpointers_debug(f"Failed to get sentiments: {exc}")
        return {}
