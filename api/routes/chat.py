# Load environment variables early
import os

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import time

# Standard imports
import traceback
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]


# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import models
from api.models.requests import FeedbackRequest, SentimentRequest
from api.models.responses import (
    ChatMessage,
    ChatThreadResponse,
    PaginatedChatThreadsResponse,
)

# Import debug functions
from api.utils.debug import (
    print__chat_all_messages_one_thread_debug,
    print__chat_sentiments_debug,
    print__chat_threads_debug,
    print__delete_chat_debug,
    print__sentiment_flow,
)

# Import utility functions
from api.utils.memory import log_memory_usage, perform_deletion_operations

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
# Import global variables from api.config.settings
from api.config.settings import (
    BULK_CACHE_TIMEOUT,
    MAX_CONCURRENT_ANALYSES,
    _bulk_loading_cache,
    _bulk_loading_locks,
)
from api.helpers import traceback_json_response
from api.utils.debug import print__chat_all_messages_debug
from checkpointer.user_management.thread_operations import (
    get_user_chat_threads,
    get_user_chat_threads_count,
)
from checkpointer.user_management.sentiment_tracking import get_thread_run_sentiments
from checkpointer.database.connection import get_direct_connection
from checkpointer.checkpointer.factory import get_global_checkpointer

# Load environment variables
load_dotenv()

# Create router for chat endpoints
router = APIRouter()


async def get_thread_messages_with_metadata(
    checkpointer, thread_id: str, user_email: str, source_context: str = "general"
) -> List[ChatMessage]:
    """
    Extract and process all messages for a single thread with metadata.

    This function consolidates checkpoint processing to extract all metadata in one pass:
    - User prompts and AI responses
    - Queries and results
    - Datasets used
    - SQL queries
    - PDF chunks

    Args:
        checkpointer: The database checkpointer instance
        thread_id: The thread ID to process
        user_email: The user's email
        source_context: Context for metadata (e.g., "single_thread", "bulk_processing")

    Returns:
        List of ChatMessage objects for the thread
    """

    print__chat_all_messages_debug(
        f"🔄 Processing thread {thread_id} for user {user_email}"
    )

    try:
        # Security check: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__chat_all_messages_debug(
                f"🔍 SECURITY CHECK: Verifying thread ownership for user: {user_email}"
            )

            try:
                async with get_direct_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT COUNT(*) FROM users_threads_runs 
                            WHERE email = %s AND thread_id = %s
                        """,
                            (user_email, thread_id),
                        )
                        result = await cur.fetchone()
                        thread_entries_count = result[0] if result else 0

                    if thread_entries_count == 0:
                        print__chat_all_messages_debug(
                            f"🔍 SECURITY DENIED: User {user_email} does not own thread {thread_id} - access denied"
                        )
                        return []

                    print__chat_all_messages_debug(
                        f"🔍 SECURITY GRANTED: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted"
                    )
            except Exception as e:
                print__chat_all_messages_debug(
                    f"🔍 SECURITY ERROR: Could not verify thread ownership: {e}"
                )
                return []

        config = {"configurable": {"thread_id": thread_id}}

        # Get checkpoint tuples using alist() with fallback to aget_tuple()
        checkpoint_tuples = []
        try:
            print__chat_all_messages_debug(
                "🔍 ALIST METHOD: Using official AsyncPostgresSaver.alist() method"
            )

            # Get all checkpoints to capture complete conversation and metadata
            async for checkpoint_tuple in checkpointer.alist(config, limit=200):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__chat_all_messages_debug(
                f"🔍 ALIST ERROR: Error using alist(): {alist_error}"
            )

            # Fallback: use aget_tuple() to get the latest checkpoint only
            if not checkpoint_tuples:
                print__chat_all_messages_debug(
                    "🔍 FALLBACK METHOD: Trying fallback method using aget_tuple()"
                )
                try:
                    state_snapshot = await checkpointer.aget_tuple(config)
                    if state_snapshot:
                        checkpoint_tuples = [state_snapshot]
                        print__chat_all_messages_debug(
                            "🔍 FALLBACK SUCCESS: Using fallback method - got latest checkpoint only"
                        )
                except Exception as fallback_error:
                    print__chat_all_messages_debug(
                        f"🔍 FALLBACK ERROR: Fallback method also failed: {fallback_error}"
                    )
                    return []

        if not checkpoint_tuples:
            print__chat_all_messages_debug(
                f"🔍 NO CHECKPOINTS: No checkpoints found for thread: {thread_id}"
            )
            return []

        print__chat_all_messages_debug(
            f"🔍 CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for verified thread"
        )

        # Sort checkpoints by step number (chronological order)
        checkpoint_tuples.sort(
            key=lambda x: x.metadata.get("step", 0) if x.metadata else 0
        )

        # Extract all interactions in one pass through the checkpoints
        interactions = []

        print__chat_all_messages_debug(
            f"🔍 INTERACTION EXTRACTION: Extracting complete interactions from {len(checkpoint_tuples)} checkpoints"
        )

        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            metadata = checkpoint_tuple.metadata or {}
            step = metadata.get("step", 0)
            writes = metadata.get("writes", {})

            interaction = {
                "step": step,
                "checkpoint_index": checkpoint_index,
            }

            # Extract user prompt from metadata.writes.__start__.prompt
            if isinstance(writes, dict) and "__start__" in writes:
                start_data = writes["__start__"]
                if isinstance(start_data, dict) and "prompt" in start_data:
                    prompt = start_data["prompt"]
                    if prompt and prompt.strip():
                        interaction["prompt"] = prompt.strip()
                        print__chat_all_messages_debug(
                            f"🔍 USER PROMPT FOUND: Step {step}: {prompt[:50]}..."
                        )

            # Extract AI answer and all metadata from metadata.writes.submit_final_answer
            if isinstance(writes, dict) and "submit_final_answer" in writes:
                submit_data = writes["submit_final_answer"]
                if isinstance(submit_data, dict):
                    # Extract final answer
                    if "final_answer" in submit_data:
                        final_answer = submit_data["final_answer"]
                        if final_answer and final_answer.strip():
                            interaction["final_answer"] = final_answer.strip()
                            print__chat_all_messages_debug(
                                f"🔍 AI ANSWER FOUND: Step {step}: {final_answer[:50]}..."
                            )

                    # Extract follow-up prompts (only for this interaction)
                    if "followup_prompts" in submit_data:
                        followup_prompts = submit_data["followup_prompts"]
                        if followup_prompts:
                            interaction["followup_prompts"] = followup_prompts
                            print__chat_all_messages_debug(
                                f"🔍 FOLLOWUP PROMPTS FOUND: Step {step}: Found {len(followup_prompts)} follow-up prompts"
                            )

                    # Extract queries and results (only for this interaction)
                    if "queries_and_results" in submit_data:
                        queries_and_results = submit_data["queries_and_results"]
                        if queries_and_results:
                            interaction["queries_and_results"] = queries_and_results
                            print__chat_all_messages_debug(
                                f"🔍 QUERIES FOUND: Step {step}: Found queries and results for this interaction"
                            )

                    # Extract datasets used from top_selection_codes (only for this interaction)
                    if "top_selection_codes" in submit_data:
                        top_selection_codes = submit_data["top_selection_codes"]
                        if top_selection_codes:
                            interaction["datasets_used"] = top_selection_codes
                            print__chat_all_messages_debug(
                                f"🔍 DATASETS FOUND: Step {step}: Found {len(top_selection_codes)} datasets for this interaction"
                            )

                    # Extract top chunks (only for this interaction)
                    if "top_chunks" in submit_data:
                        top_chunks_raw = submit_data["top_chunks"]
                        if top_chunks_raw:
                            chunks_processed = []
                            for chunk in top_chunks_raw:
                                try:
                                    # Extract page_content and metadata
                                    chunk_data = {
                                        "page_content": (
                                            chunk.page_content
                                            if hasattr(chunk, "page_content")
                                            else chunk.get("page_content", "")
                                        )
                                    }

                                    # Extract metadata (source_file, page_number, etc.)
                                    metadata = (
                                        chunk.metadata
                                        if hasattr(chunk, "metadata")
                                        else chunk.get("metadata", {})
                                    )

                                    if metadata:
                                        if "source_file" in metadata:
                                            chunk_data["source_file"] = metadata[
                                                "source_file"
                                            ]
                                        if "page_number" in metadata:
                                            chunk_data["page_number"] = metadata[
                                                "page_number"
                                            ]
                                        # Add any other metadata fields that might be useful
                                        chunk_data["metadata"] = metadata

                                    chunks_processed.append(chunk_data)

                                except Exception as chunk_error:
                                    print__chat_all_messages_debug(
                                        f"🔍 Error processing chunk: {chunk_error}"
                                    )
                                    continue

                            if chunks_processed:
                                interaction["top_chunks"] = chunks_processed
                                print__chat_all_messages_debug(
                                    f"🔍 CHUNKS FOUND: Step {step}: Found {len(chunks_processed)} chunks for this interaction"
                                )

                    # Extract other metadata that might be in submit_final_answer
                    # (This is where we could extract other per-interaction metadata)

            # Only add interaction if it has either prompt or final_answer
            if "prompt" in interaction or "final_answer" in interaction:
                interactions.append(interaction)

        # Sort interactions by step number (chronological order)
        interactions.sort(key=lambda x: x["step"])

        print__chat_all_messages_debug(
            f"🔍 INTERACTION SUCCESS: Created {len(interactions)} complete interactions"
        )

        if not interactions:
            print__chat_all_messages_debug(
                f"⚠ No interactions found for thread {thread_id}"
            )
            return []

        # Convert interactions to ChatMessage objects
        chat_messages = []
        message_counter = 0

        print__chat_all_messages_debug(
            f"🔍 Converting {len(interactions)} interactions to ChatMessage objects"
        )

        for i, interaction in enumerate(interactions):
            print__chat_all_messages_debug(
                f"🔍 Processing interaction {i+1}/{len(interactions)}: Step {interaction['step']}"
            )

            # Create one message per interaction with both prompt and final_answer
            message_counter += 1
            chat_message = ChatMessage(
                id=f"msg_{message_counter}",
                threadId=thread_id,
                user=user_email,
                createdAt=int(
                    datetime.fromtimestamp(
                        1700000000 + message_counter * 1000
                    ).timestamp()
                    * 1000
                ),
                prompt=interaction.get("prompt"),
                final_answer=interaction.get("final_answer"),
                queries_and_results=interaction.get("queries_and_results"),
                datasets_used=interaction.get("datasets_used"),
                top_chunks=interaction.get("top_chunks"),
                sql_query=None,
                error=None,
                isLoading=False,
                startedAt=None,
                isError=False,
                followup_prompts=interaction.get("followup_prompts"),
            )

            # Extract SQL query from queries_and_results if available
            if (
                chat_message.queries_and_results
                and len(chat_message.queries_and_results) > 0
            ):
                try:
                    chat_message.sql_query = (
                        chat_message.queries_and_results[0][0]
                        if chat_message.queries_and_results[0]
                        else None
                    )
                except (IndexError, TypeError):
                    chat_message.sql_query = None

            chat_messages.append(chat_message)
            print__chat_all_messages_debug(
                f"🔍 ADDED MESSAGE: Step {interaction['step']}: prompt={interaction.get('prompt', '')[:50]} final_answer={interaction.get('final_answer', '')[:50]}..."
            )

        print__chat_all_messages_debug(
            f"✅ Processed {len(chat_messages)} messages for thread {thread_id}"
        )
        return chat_messages

    except Exception as e:
        print__chat_all_messages_debug(f"❌ Error processing thread {thread_id}: {e}")
        print__chat_all_messages_debug(
            f"🔍 Thread processing error type: {type(e).__name__}"
        )
        print__chat_all_messages_debug(
            f"🔍 Thread processing error traceback: {traceback.format_exc()}"
        )
        return []


@router.get("/chat/{thread_id}/sentiments")
async def get_thread_sentiments(thread_id: str, user=Depends(get_current_user)):
    """Get sentiment values for all messages in a thread."""

    print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - ENTRY POINT")
    print__chat_sentiments_debug(f"🔍 Request received: thread_id={thread_id}")

    user_email = user.get("email")
    print__chat_sentiments_debug(f"🔍 User email extracted: {user_email}")

    if not user_email:
        print__chat_sentiments_debug("🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    try:
        print__chat_sentiments_debug(
            f"🔍 Getting sentiments for thread {thread_id}, user: {user_email}"
        )
        print__sentiment_flow(
            f"📥 Getting sentiments for thread {thread_id}, user: {user_email}"
        )
        sentiments = await get_thread_run_sentiments(user_email, thread_id)
        print__chat_sentiments_debug(f"🔍 Retrieved {len(sentiments)} sentiment values")

        print__sentiment_flow(f"✅ Retrieved {len(sentiments)} sentiment values")
        print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - SUCCESSFUL EXIT")
        return sentiments

    except Exception as e:
        print__chat_sentiments_debug(
            f"🚨 Exception in chat sentiments processing: {type(e).__name__}: {str(e)}"
        )
        print__chat_sentiments_debug(
            f"🚨 Chat sentiments processing traceback: {traceback.format_exc()}"
        )
        print__sentiment_flow(
            f"❌ Failed to get sentiments for thread {thread_id}: {e}"
        )
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise HTTPException(
            status_code=500, detail=f"Failed to get sentiments: {e}"
        ) from e


@router.get(
    "/chat-threads",
    summary="Get user's chat threads",
    description="""
    **Retrieve a paginated list of chat threads for the authenticated user.**
    
    Threads are ordered by most recent activity first. Each thread includes:
    - Thread ID for referencing in other endpoints
    - Latest timestamp of activity
    - Number of query runs in the thread
    - Title derived from the first query
    - Full first prompt for context
    
    **Pagination:** Use `page` and `limit` parameters to control results.
    """,
    response_model=PaginatedChatThreadsResponse,
    response_description="Paginated list of chat threads with metadata",
    responses={200: {"description": "Successfully retrieved chat threads"}},
)
async def get_chat_threads(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=50, description="Number of threads per page"),
    user=Depends(get_current_user),
) -> PaginatedChatThreadsResponse:
    """Get paginated chat threads for the authenticated user."""

    print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ENTRY POINT")
    print__chat_threads_debug(f"🔍 Request parameters: page={page}, limit={limit}")

    try:
        user_email = user["email"]
        print__chat_threads_debug(f"🔍 User email extracted: {user_email}")
        print__chat_threads_debug(
            f"Loading chat threads for user: {user_email} (page: {page}, limit: {limit})"
        )

        print__chat_threads_debug("🔍 Starting simplified approach")
        print__chat_threads_debug("Getting chat threads with simplified approach")

        # Get total count first
        print__chat_threads_debug("🔍 Getting total threads count")
        print__chat_threads_debug(f"Getting chat threads count for user: {user_email}")
        total_count = await get_user_chat_threads_count(user_email)
        print__chat_threads_debug(f"🔍 Total count retrieved: {total_count}")
        print__chat_threads_debug(
            f"Total threads count for user {user_email}: {total_count}"
        )

        # Calculate offset for pagination
        offset = (page - 1) * limit
        print__chat_threads_debug(f"🔍 Calculated offset: {offset}")

        # Get threads for this page
        print__chat_threads_debug(
            f"🔍 Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})"
        )
        print__chat_threads_debug(
            f"Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})"
        )
        threads = await get_user_chat_threads(user_email, limit=limit, offset=offset)
        print__chat_threads_debug(f"🔍 Retrieved threads: {threads}")
        if threads is None:
            print__chat_threads_debug(
                "get_user_chat_threads returned None! Setting to empty list."
            )
            threads = []
        print__chat_threads_debug(f"🔍 Retrieved {len(threads)} threads from database")
        print__chat_threads_debug(
            f"Retrieved {len(threads)} threads for user {user_email}"
        )

        # Try/except around the for-loop to catch and print any errors
        try:
            chat_thread_responses = []
            for thread in threads:
                print__chat_threads_debug(f"Processing thread dict: {thread}")
                chat_thread_response = ChatThreadResponse(
                    thread_id=thread["thread_id"],
                    latest_timestamp=thread["latest_timestamp"],
                    run_count=thread["run_count"],
                    title=thread["title"],
                    full_prompt=thread["full_prompt"],
                )
                chat_thread_responses.append(chat_thread_response)
        except Exception as e:
            print__chat_threads_debug(f"Exception in /chat-threads for-loop: {e}")
            print__chat_threads_debug(f"Traceback: {traceback.format_exc()}")
            # Return empty result on error
            return PaginatedChatThreadsResponse(
                threads=[], total_count=0, page=page, limit=limit, has_more=False
            )

        # Convert to response format
        print__chat_threads_debug("🔍 Converting threads to response format")
        chat_thread_responses = []
        for thread in threads:
            chat_thread_response = ChatThreadResponse(
                thread_id=thread["thread_id"],
                latest_timestamp=thread["latest_timestamp"],
                run_count=thread["run_count"],
                title=thread["title"],
                full_prompt=thread["full_prompt"],
            )
            chat_thread_responses.append(chat_thread_response)

        # Calculate pagination info
        has_more = (offset + len(chat_thread_responses)) < total_count
        print__chat_threads_debug(f"🔍 Pagination calculated: has_more={has_more}")

        print__chat_threads_debug(
            f"Retrieved {len(threads)} threads for user {user_email} (total: {total_count})"
        )
        print__chat_threads_debug(
            f"Returning {len(chat_thread_responses)} threads to frontend (page {page}/{(total_count + limit - 1) // limit})"
        )

        result = PaginatedChatThreadsResponse(
            threads=chat_thread_responses,
            total_count=total_count,
            page=page,
            limit=limit,
            has_more=has_more,
        )
        print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - SUCCESSFUL EXIT")
        return result

    except Exception as e:
        print__chat_threads_debug(
            f"🚨 Exception in chat threads processing: {type(e).__name__}: {str(e)}"
        )
        print__chat_threads_debug(
            f"🚨 Chat threads processing traceback: {traceback.format_exc()}"
        )
        print__chat_threads_debug(f"❌ Error getting chat threads: {e}")
        print__chat_threads_debug(f"Full traceback: {traceback.format_exc()}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        # Return error response
        result = PaginatedChatThreadsResponse(
            threads=[], total_count=0, page=page, limit=limit, has_more=False
        )
        print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ERROR EXIT")
        return result


@router.delete("/chat/{thread_id}")
async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Delete all PostgreSQL checkpoint records and thread entries for a specific thread_id."""

    print__delete_chat_debug("🔍 DELETE_CHAT ENDPOINT - ENTRY POINT")
    print__delete_chat_debug(f"🔍 Request received: thread_id={thread_id}")

    user_email = user.get("email")
    print__delete_chat_debug(f"🔍 User email extracted: {user_email}")

    if not user_email:
        print__delete_chat_debug("🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__delete_chat_debug(
        f"🗑️ Deleting chat thread {thread_id} for user {user_email}"
    )

    try:
        # Use the same connection method as get_user_chat_threads for consistency
        async with get_direct_connection() as conn:
            result_data = await perform_deletion_operations(conn, user_email, thread_id)
            return result_data

    except Exception as e:
        error_msg = str(e)
        print__delete_chat_debug(
            f"❌ Failed to delete checkpoint records for thread {thread_id}: {e}"
        )
        print__delete_chat_debug(f"🔧 DEBUG: Main exception type: {type(e).__name__}")
        print__delete_chat_debug(
            f"🔧 DEBUG: Main exception traceback: {traceback.format_exc()}"
        )
        if any(
            keyword in error_msg.lower()
            for keyword in [
                "ssl error",
                "connection",
                "timeout",
                "operational error",
                "server closed",
                "bad connection",
                "consuming input failed",
            ]
        ):
            print__delete_chat_debug(
                "⚠️ PostgreSQL connection unavailable - no records to delete"
            )
            return {
                "message": "PostgreSQL connection unavailable - no records to delete",
                "thread_id": thread_id,
                "user_email": user_email,
                "warning": "Database connection issues",
            }
        else:
            resp = traceback_json_response(e)
            if resp:
                return resp
            raise HTTPException(
                status_code=500, detail=f"Failed to delete checkpoint records: {e}"
            ) from e


@router.get("/chat/all-messages-for-one-thread/{thread_id}")
async def get_all_chat_messages_for_one_thread(
    thread_id: str, user=Depends(get_current_user)
) -> Dict:
    """Get all chat messages for a specific thread, packaged in a dictionary with metadata."""

    print__chat_all_messages_one_thread_debug(
        "🔍 CHAT_SINGLE_THREAD ENDPOINT - ENTRY POINT"
    )

    user_email = user["email"]
    print__chat_all_messages_one_thread_debug(
        f"📥 SINGLE THREAD REQUEST: Loading chat messages for thread: {thread_id}, user: {user_email}"
    )

    try:
        checkpointer = await get_global_checkpointer()

        # Get all messages and their per-interaction metadata
        chat_messages = await get_thread_messages_with_metadata(
            checkpointer, thread_id, user_email, "single_thread_processing"
        )

        # Get run-ids and sentiments for the entire thread
        thread_run_ids = []
        thread_sentiments = {}
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT run_id, prompt, timestamp, sentiment
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                    ORDER BY timestamp ASC
                """,
                    (user_email, thread_id),
                )
                rows = await cur.fetchall()

            for row in rows:
                run_id, prompt, timestamp, sentiment = row
                thread_run_ids.append(
                    {
                        "run_id": run_id,
                        "prompt": prompt,
                        "timestamp": timestamp.isoformat(),
                    }
                )
                if sentiment is not None:
                    thread_sentiments[run_id] = sentiment

        # Match run_ids to messages by index (AI messages with final_answer)
        print__chat_all_messages_one_thread_debug(
            f"🔍 MATCHING RUN_IDS: Found {len(thread_run_ids)} run_ids and {len(chat_messages)} messages"
        )

        ai_message_index = 0
        for idx, msg in enumerate(chat_messages):
            print__chat_all_messages_one_thread_debug(
                f"🔍 Message {idx}: has_prompt={bool(msg.prompt)}, has_final_answer={bool(msg.final_answer)}"
            )

            if msg.final_answer and ai_message_index < len(thread_run_ids):
                msg.run_id = thread_run_ids[ai_message_index]["run_id"]
                print__chat_all_messages_one_thread_debug(
                    f"🔍 MATCHED: Message {idx} -> run_id {msg.run_id}"
                )
                ai_message_index += 1
            else:
                if msg.final_answer:
                    print__chat_all_messages_one_thread_debug(
                        f"⚠️ WARNING: Message {idx} has final_answer but no run_id available (ai_message_index={ai_message_index}, run_ids_count={len(thread_run_ids)})"
                    )

        print__chat_all_messages_one_thread_debug(
            f"🔍 MATCHING COMPLETE: Matched {ai_message_index} messages to run_ids"
        )

        # Serialize ChatMessage objects to dictionaries for the final JSON response
        chat_messages_serialized = [
            msg.model_dump(exclude_none=True) for msg in chat_messages
        ]

        # Package everything into the dictionary structure the frontend expects
        result = {
            "messages": chat_messages_serialized,
            "runIds": thread_run_ids,
            "sentiments": thread_sentiments,
        }

        print__chat_all_messages_one_thread_debug(
            f"✅ SINGLE THREAD PROCESSING COMPLETE: Returning {len(chat_messages)} messages and metadata."
        )

        return result

    except Exception as e:
        print__chat_all_messages_one_thread_debug(
            f"❌ SINGLE THREAD ERROR: Failed to process request for thread {thread_id}: {e}"
        )
        print__chat_all_messages_one_thread_debug(
            f"Full error traceback: {traceback.format_exc()}"
        )
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise HTTPException(
            status_code=500, detail="Failed to process chat thread"
        ) from e
