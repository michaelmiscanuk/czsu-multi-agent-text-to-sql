# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Import configuration and globals
from api.config.settings import (
    BULK_CACHE_TIMEOUT,
    _bulk_loading_cache,
    _bulk_loading_locks,
)

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import traceback helper
from api.helpers import traceback_json_response

# Import models
from api.models.responses import ChatMessage

# Import the reusable function from chat.py
from api.routes.chat import get_thread_messages_with_metadata

# Import debug functions
from api.utils.debug import print__chat_all_messages_debug

# Import memory utilities
from api.utils.memory import log_memory_usage

# Import database connection functions
from my_agent.utils.postgres_checkpointer import (
    get_direct_connection,
    get_healthy_checkpointer,
)

# Load environment variables
load_dotenv()

# Create router for bulk operations endpoints
router = APIRouter()

# ENVIRONMENT VARIABLES
MAX_CONCURRENT_BULK_THREADS = int(
    os.environ.get("MAX_CONCURRENT_BULK_THREADS", "3")
)  # Read from .env with fallback to 3


@router.get("/chat/all-messages-for-all-threads")
async def get_all_chat_messages(user=Depends(get_current_user)) -> Dict:
    """Get all chat messages for the authenticated user using bulk loading with improved caching."""

    print__chat_all_messages_debug("🔍 CHAT_ALL_MESSAGES ENDPOINT - ENTRY POINT")

    user_email = user["email"]
    print__chat_all_messages_debug(f"🔍 User email extracted: {user_email}")
    print__chat_all_messages_debug(
        f"📥 BULK REQUEST: Loading ALL chat messages for user: {user_email}"
    )

    # Check if we have a recent cached result
    cache_key = f"bulk_messages_{user_email}"
    current_time = time.time()
    print__chat_all_messages_debug(f"🔍 Cache key: {cache_key}")
    print__chat_all_messages_debug(f"🔍 Current time: {current_time}")

    if cache_key in _bulk_loading_cache:
        print__chat_all_messages_debug("🔍 Cache entry found for user")
        cached_data, cache_time = _bulk_loading_cache[cache_key]
        cache_age = current_time - cache_time
        print__chat_all_messages_debug(
            f"🔍 Cache age: {cache_age:.1f}s (timeout: {BULK_CACHE_TIMEOUT}s)"
        )

        if cache_age < BULK_CACHE_TIMEOUT:
            print__chat_all_messages_debug(
                f"✅ CACHE HIT: Returning cached bulk data for {user_email} (age: {cache_age:.1f}s)"
            )

            # Return cached data with appropriate headers
            response = JSONResponse(content=cached_data)
            response.headers["Cache-Control"] = (
                f"public, max-age={int(BULK_CACHE_TIMEOUT - cache_age)}"
            )
            response.headers["ETag"] = f"bulk-{user_email}-{int(cache_time)}"
            print__chat_all_messages_debug(
                "🔍 CHAT_ALL_MESSAGES ENDPOINT - CACHE HIT EXIT"
            )
            return response
        else:
            print__chat_all_messages_debug(
                f"⏰ CACHE EXPIRED: Cached data too old ({cache_age:.1f}s), will refresh"
            )
            del _bulk_loading_cache[cache_key]
            print__chat_all_messages_debug("🔍 Expired cache entry deleted")
    else:
        print__chat_all_messages_debug("🔍 No cache entry found for user")

    # Use a lock to prevent multiple simultaneous requests from the same user
    print__chat_all_messages_debug(
        f"🔍 Attempting to acquire lock for user: {user_email}"
    )
    async with _bulk_loading_locks[user_email]:
        print__chat_all_messages_debug(f"🔒 Lock acquired for user: {user_email}")

        # Double-check cache after acquiring lock (another request might have completed)
        if cache_key in _bulk_loading_cache:
            print__chat_all_messages_debug(
                "🔍 Double-checking cache after lock acquisition"
            )
            cached_data, cache_time = _bulk_loading_cache[cache_key]
            cache_age = current_time - cache_time
            if cache_age < BULK_CACHE_TIMEOUT:
                print__chat_all_messages_debug(
                    f"✅ CACHE HIT (after lock): Returning cached bulk data for {user_email}"
                )
                print__chat_all_messages_debug(
                    "🔍 CHAT_ALL_MESSAGES ENDPOINT - CACHE HIT AFTER LOCK EXIT"
                )
                return cached_data
            else:
                print__chat_all_messages_debug(
                    "🔍 Cache still expired after lock, proceeding with fresh request"
                )

        print__chat_all_messages_debug(
            f"🔄 CACHE MISS: Processing fresh bulk request for {user_email}"
        )

        # Simple memory check before starting
        print__chat_all_messages_debug("🔍 Starting memory check")
        log_memory_usage("bulk_start")
        print__chat_all_messages_debug("🔍 Memory check completed")

        try:
            print__chat_all_messages_debug("🔍 Getting healthy checkpointer")
            checkpointer = await get_healthy_checkpointer()
            print__chat_all_messages_debug(
                f"🔍 Checkpointer obtained: {type(checkpointer).__name__}"
            )

            # STEP 1: Get all user threads, run-ids, and sentiments in ONE query
            print__chat_all_messages_debug(
                "🔍 BULK QUERY: Getting all user threads, run-ids, and sentiments"
            )
            user_thread_ids = []
            all_run_ids = {}
            all_sentiments = {}

            # FIXED: Use our working get_direct_connection() function instead of checkpointer.conn
            print__chat_all_messages_debug("🔍 Importing get_direct_connection")
            print__chat_all_messages_debug("🔍 Getting direct connection")

            # FIXED: Use get_direct_connection() as async context manager
            print__chat_all_messages_debug("🔍 Using direct connection context manager")
            async with get_direct_connection() as conn:
                print__chat_all_messages_debug(
                    f"🔍 Connection obtained: {type(conn).__name__}"
                )
                async with conn.cursor() as cur:
                    print__chat_all_messages_debug(
                        "🔍 Cursor created, executing bulk query"
                    )
                    # Single query for all threads, run-ids, and sentiments
                    # FIXED: Use psycopg format (%s) instead of asyncpg format ($1)
                    await cur.execute(
                        """
                        SELECT 
                            thread_id, 
                            run_id, 
                            prompt, 
                            timestamp,
                            sentiment
                        FROM users_threads_runs 
                        WHERE email = %s
                        ORDER BY thread_id, timestamp ASC
                    """,
                        (user_email,),
                    )

                    print__chat_all_messages_debug(
                        "🔍 Bulk query executed, fetching results"
                    )
                    rows = await cur.fetchall()
                    print__chat_all_messages_debug(
                        f"🔍 Retrieved {len(rows)} rows from database"
                    )

                for i, row in enumerate(rows):
                    print__chat_all_messages_debug(
                        f"🔍 Processing row {i+1}/{len(rows)}"
                    )
                    # FIXED: Use index-based access instead of dict-based for psycopg
                    thread_id = row[0]  # thread_id
                    run_id = row[1]  # run_id
                    prompt = row[2]  # prompt
                    timestamp = row[3]  # timestamp
                    sentiment = row[4]  # sentiment

                    print__chat_all_messages_debug(
                        f"🔍 Row data: thread_id={thread_id}, run_id={run_id}, prompt_length={len(prompt) if prompt else 0}"
                    )

                    # Track unique thread IDs
                    if thread_id not in user_thread_ids:
                        user_thread_ids.append(thread_id)
                        print__chat_all_messages_debug(
                            f"🔍 New thread discovered: {thread_id}"
                        )

                    # Build run-ids dictionary
                    if thread_id not in all_run_ids:
                        all_run_ids[thread_id] = []
                        print__chat_all_messages_debug(
                            f"🔍 Initializing run_ids list for thread: {thread_id}"
                        )
                    all_run_ids[thread_id].append(
                        {
                            "run_id": run_id,
                            "prompt": prompt,
                            "timestamp": timestamp.isoformat(),
                        }
                    )

                    # Build sentiments dictionary
                    if sentiment is not None:
                        if thread_id not in all_sentiments:
                            all_sentiments[thread_id] = {}
                            print__chat_all_messages_debug(
                                f"🔍 Initializing sentiments dict for thread: {thread_id}"
                            )
                        all_sentiments[thread_id][run_id] = sentiment
                        print__chat_all_messages_debug(
                            f"🔍 Added sentiment for run_id {run_id}: {sentiment}"
                        )

            print__chat_all_messages_debug(
                f"📊 BULK: Found {len(user_thread_ids)} threads"
            )
            print__chat_all_messages_debug(
                f"📊 BULK: Found {len(all_run_ids)} thread run_ids"
            )
            print__chat_all_messages_debug(
                f"📊 BULK: Found {len(all_sentiments)} thread sentiments"
            )

            if not user_thread_ids:
                print__chat_all_messages_debug(
                    "⚠ No threads found for user - returning empty dictionary"
                )
                empty_result = {"messages": {}, "runIds": {}, "sentiments": {}}
                _bulk_loading_cache[cache_key] = (empty_result, current_time)
                print__chat_all_messages_debug(
                    "🔍 CHAT_ALL_MESSAGES ENDPOINT - EMPTY RESULT EXIT"
                )
                return empty_result

            # STEP 2: Process threads with limited concurrency (max 3 concurrent)
            print__chat_all_messages_debug(
                f"🔄 Processing {len(user_thread_ids)} threads with limited concurrency"
            )

            async def process_single_thread(thread_id: str):
                """Process a single thread using the proven working functions."""
                try:
                    print__chat_all_messages_debug(f"🔄 Processing thread {thread_id}")

                    # Use the new reusable function from chat.py
                    chat_messages = await get_thread_messages_with_metadata(
                        checkpointer, thread_id, user_email, "cached_bulk_processing"
                    )

                    print__chat_all_messages_debug(
                        f"✅ Processed {len(chat_messages)} messages for thread {thread_id}"
                    )
                    return thread_id, chat_messages

                except Exception as e:
                    print__chat_all_messages_debug(
                        f"❌ Error processing thread {thread_id}: {e}"
                    )
                    print__chat_all_messages_debug(
                        f"🔍 Thread processing error type: {type(e).__name__}"
                    )
                    print__chat_all_messages_debug(
                        f"🔍 Thread processing error traceback: {traceback.format_exc()}"
                    )
                    return thread_id, []

            MAX_CONCURRENT_BULK_THREADS = 3
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_BULK_THREADS)
            print__chat_all_messages_debug(
                f"🔍 Semaphore created with {MAX_CONCURRENT_BULK_THREADS} slots"
            )

            async def process_single_thread_with_limit(thread_id: str):
                """Process a single thread with concurrency limiting."""
                print__chat_all_messages_debug(
                    f"🔍 Waiting for semaphore slot for thread: {thread_id}"
                )
                async with semaphore:
                    print__chat_all_messages_debug(
                        f"🔍 Semaphore acquired for thread: {thread_id}"
                    )
                    result = await process_single_thread(thread_id)
                    print__chat_all_messages_debug(
                        f"🔍 Semaphore released for thread: {thread_id}"
                    )
                    return result

            print__chat_all_messages_debug(
                f"🔒 Processing with max {MAX_CONCURRENT_BULK_THREADS} concurrent operations"
            )

            # Use asyncio.gather with limited concurrency
            print__chat_all_messages_debug(
                f"🔍 Starting asyncio.gather for {len(user_thread_ids)} threads"
            )
            thread_results = await asyncio.gather(
                *[
                    process_single_thread_with_limit(thread_id)
                    for thread_id in user_thread_ids
                ],
                return_exceptions=True,
            )
            print__chat_all_messages_debug(
                "🔍 asyncio.gather completed, processing results"
            )

            # Collect results
            all_messages = {}
            total_messages = 0

            for i, result in enumerate(thread_results):
                print__chat_all_messages_debug(
                    f"🔍 Processing thread result {i+1}/{len(thread_results)}"
                )
                if isinstance(result, Exception):
                    print__chat_all_messages_debug(
                        f"⚠ Exception in thread processing: {result}"
                    )
                    print__chat_all_messages_debug(
                        f"🔍 Exception type: {type(result).__name__}"
                    )
                    print__chat_all_messages_debug(
                        f"🔍 Exception traceback: {traceback.format_exc()}"
                    )
                    continue

                thread_id, chat_messages = result
                all_messages[thread_id] = chat_messages
                total_messages += len(chat_messages)
                print__chat_all_messages_debug(
                    f"🔍 Added {len(chat_messages)} messages for thread {thread_id}"
                )

            print__chat_all_messages_debug(
                f"✅ BULK LOADING COMPLETE: {len(all_messages)} threads, {total_messages} total messages"
            )

            # Simple memory check after completion
            print__chat_all_messages_debug("🔍 Starting post-completion memory check")
            log_memory_usage("bulk_complete")
            print__chat_all_messages_debug("🔍 Post-completion memory check completed")

            # Convert all ChatMessage objects to dicts for JSON serialization
            for thread_id in all_messages:
                all_messages[thread_id] = [
                    msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()
                    for msg in all_messages[thread_id]
                ]

            result = {
                "messages": all_messages,
                "runIds": all_run_ids,
                "sentiments": all_sentiments,
            }
            print__chat_all_messages_debug(
                f"🔍 Result dictionary created with {len(result)} keys"
            )

            # Cache the result
            _bulk_loading_cache[cache_key] = (result, current_time)
            print__chat_all_messages_debug(
                f"💾 CACHED: Bulk result for {user_email} (expires in {BULK_CACHE_TIMEOUT}s)"
            )

            # Return with cache headers
            response = JSONResponse(content=result)
            response.headers["Cache-Control"] = f"public, max-age={BULK_CACHE_TIMEOUT}"
            response.headers["ETag"] = f"bulk-{user_email}-{int(current_time)}"
            print__chat_all_messages_debug("🔍 JSONResponse created with cache headers")
            print__chat_all_messages_debug(
                "🔍 CHAT_ALL_MESSAGES ENDPOINT - SUCCESSFUL EXIT"
            )
            return response

        except Exception as e:
            print__chat_all_messages_debug(
                f"❌ BULK ERROR: Failed to process bulk request for {user_email}: {e}"
            )
            print__chat_all_messages_debug(
                f"🔍 Main exception type: {type(e).__name__}"
            )
            print__chat_all_messages_debug(
                f"Full error traceback: {traceback.format_exc()}"
            )

            # Return empty result but cache it briefly to prevent error loops
            empty_result = {"messages": {}, "runIds": {}, "sentiments": {}}
            _bulk_loading_cache[cache_key] = (empty_result, current_time)
            print__chat_all_messages_debug("🔍 Cached empty result due to error")

            resp = traceback_json_response(e)
            if resp:
                return resp

            response = JSONResponse(content=empty_result, status_code=500)
            response.headers["Cache-Control"] = (
                "no-cache, no-store"  # Don't cache errors
            )
            print__chat_all_messages_debug("🔍 CHAT_ALL_MESSAGES ENDPOINT - ERROR EXIT")
            return response
