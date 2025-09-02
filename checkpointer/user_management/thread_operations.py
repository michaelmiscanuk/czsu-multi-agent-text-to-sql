"""Thread creation, retrieval, and management operations.

This module handles user thread operations and management
for the PostgreSQL checkpointer system.
"""
from __future__ import annotations

import traceback
import uuid
from typing import List, Dict, Any

from api.utils.debug import print__checkpointers_debug
from checkpointer.database.connection import get_direct_connection
from checkpointer.error_handling.retry_decorators import retry_on_prepared_statement_error
from checkpointer.config import DEFAULT_MAX_RETRIES, THREAD_TITLE_MAX_LENGTH, THREAD_TITLE_SUFFIX_LENGTH


# This file will contain:
# - create_thread_run_entry() function
# - get_user_chat_threads() function
# - get_user_chat_threads_count() function
# - delete_user_thread_entries() function
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def create_thread_run_entry(
    email: str, thread_id: str, prompt: str = None, run_id: str = None
) -> str:
    """Create a new thread run entry in the database with retry logic for prepared statement errors."""
    print__checkpointers_debug(
        f"286 - CREATE THREAD ENTRY START: Creating thread run entry for user={email}, thread={thread_id}"
    )
    try:
        if not run_id:
            run_id = str(uuid.uuid4())
            print__checkpointers_debug(
                f"287 - GENERATE RUN ID: Generated new run_id: {run_id}"
            )

        print__checkpointers_debug(
            f"288 - DATABASE INSERT: Inserting thread run entry with run_id={run_id}"
        )

        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO users_threads_runs (email, thread_id, run_id, prompt)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        thread_id = EXCLUDED.thread_id,
                        prompt = EXCLUDED.prompt,
                        timestamp = CURRENT_TIMESTAMP
                """,
                    (email, thread_id, run_id, prompt),
                )

        print__checkpointers_debug(
            f"289 - CREATE THREAD ENTRY SUCCESS: Thread run entry created successfully: {run_id}"
        )
        return run_id
    except Exception as e:
        print__checkpointers_debug(
            f"290 - CREATE THREAD ENTRY ERROR: Failed to create thread run entry: {e}"
        )
        # Return the run_id even if database storage fails
        if not run_id:
            run_id = str(uuid.uuid4())
        print__checkpointers_debug(
            f"291 - CREATE THREAD ENTRY FALLBACK: Returning run_id despite database error: {run_id}"
        )
        return run_id


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_user_chat_threads(
    email: str, limit: int = None, offset: int = 0
) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination and retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(
            f"Getting chat threads for user: {email} (limit: {limit}, offset: {offset})"
        )

        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                base_query = """
                    SELECT 
                        thread_id,
                        MAX(timestamp) as latest_timestamp,
                        COUNT(*) as run_count,
                        (SELECT prompt FROM users_threads_runs utr2 
                         WHERE utr2.email = %s AND utr2.thread_id = utr.thread_id 
                         ORDER BY timestamp ASC LIMIT 1) as first_prompt
                    FROM users_threads_runs utr
                    WHERE email = %s
                    GROUP BY thread_id
                    ORDER BY latest_timestamp DESC
                """

                params = [email, email]

                if limit is not None:
                    base_query += " LIMIT %s OFFSET %s"
                    params.extend([limit, offset])

                await cur.execute(base_query, params)
                rows = await cur.fetchall()

                threads = []
                for row in rows:
                    thread_id = row[0]
                    latest_timestamp = row[1]
                    run_count = row[2]
                    first_prompt = row[3]

                    title = (
                        (first_prompt[:THREAD_TITLE_MAX_LENGTH] + "...")
                        if first_prompt
                        and len(first_prompt)
                        > THREAD_TITLE_MAX_LENGTH + THREAD_TITLE_SUFFIX_LENGTH
                        else (first_prompt or "Untitled Conversation")
                    )

                    threads.append(
                        {
                            "thread_id": thread_id,
                            "latest_timestamp": latest_timestamp,
                            "run_count": run_count,
                            "title": title,
                            "full_prompt": first_prompt or "",
                        }
                    )

                print__checkpointers_debug(
                    f"Retrieved {len(threads)} threads for user {email}"
                )
                return threads

    except Exception as e:
        print__checkpointers_debug(f"Failed to get chat threads for user {email}: {e}")
        # Return empty list instead of raising exception to prevent API crashes
        print__checkpointers_debug("Returning empty threads list due to error")
        return []


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_user_chat_threads_count(email: str) -> int:
    """Get total count of chat threads for a user with retry logic for prepared statement errors."""
    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(DISTINCT thread_id) as total_threads
                    FROM users_threads_runs
                    WHERE email = %s
                """,
                    (email,),
                )

                result = await cur.fetchone()
                total_count = result[0] if result else 0

            return total_count or 0

    except Exception as e:
        print__checkpointers_debug(
            f"Failed to get chat threads count for user {email}: {e}"
        )
        # Return 0 instead of raising exception to prevent API crashes
        print__checkpointers_debug("Returning 0 thread count due to error")
        return 0


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def delete_user_thread_entries(email: str, thread_id: str) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table with retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(
            f"Deleting thread entries for user: {email}, thread: {thread_id}"
        )

        async with get_direct_connection() as conn:
            # First, count the entries to be deleted
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """,
                    (email, thread_id),
                )
                result = await cur.fetchone()
                entries_to_delete = result[0] if result else 0

            print__checkpointers_debug(f"Found {entries_to_delete} entries to delete")

            if entries_to_delete == 0:
                return {
                    "deleted_count": 0,
                    "message": "No entries found to delete",
                    "thread_id": thread_id,
                    "user_email": email,
                }

            # Delete the entries
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """,
                    (email, thread_id),
                )
                deleted_count = cur.rowcount

            print__checkpointers_debug(
                f"Deleted {deleted_count} entries for user {email}, thread {thread_id}"
            )

            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email,
            }

    except Exception as e:
        print__checkpointers_debug(
            f"Failed to delete thread entries for user {email}, thread {thread_id}: {e}"
        )
        print__checkpointers_debug(f"Full traceback: {traceback.format_exc()}")
        raise
