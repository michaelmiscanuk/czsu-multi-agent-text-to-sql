# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Constants
try:
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import uuid
import traceback
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import models
from api.models.requests import AnalyzeRequest, FeedbackRequest, SentimentRequest
from api.models.responses import ChatThreadResponse, PaginatedChatThreadsResponse, ChatMessage

# Import debug functions
from api.utils.debug import (
    print__analyze_debug, print__feedback_debug, print__sentiment_debug,
    print__chat_sentiments_debug, print__chat_threads_debug, 
    print__chat_messages_debug, print__delete_chat_debug,
    print__analysis_tracing_debug, print__feedback_flow, print__sentiment_flow
)

# Import utility functions
from api.utils.memory import perform_deletion_operations

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
from my_agent.utils.postgres_checkpointer import (
    get_healthy_checkpointer,
    get_thread_run_sentiments,
    get_user_chat_threads_count,
    get_user_chat_threads
)

# Create router for chat endpoints
router = APIRouter()

@router.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """
    Main chat analysis endpoint - COMPLEX DEPENDENCIES
    
    This endpoint requires extensive refactoring to fully extract due to:
    - Database connection management (checkpointer, pool connections)
    - Analysis pipeline (analysis_main function)
    - Complex error handling and fallback mechanisms
    - Global state management (semaphores, memory monitoring)
    """
    print__analyze_debug("🔧 REFACTORING NOTE: This endpoint needs extensive dependency extraction")
    print__analysis_tracing_debug("CHAT ROUTES: analyze endpoint called but not fully implemented")
    
    raise HTTPException(
        status_code=501, 
        detail="Chat analyze endpoint requires extensive refactoring to extract from main server. Dependencies: database connections, analysis pipeline, global state."
    )

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
        print__chat_sentiments_debug(f"🔍 Getting sentiments for thread {thread_id}, user: {user_email}")
        print__sentiment_flow(f"📥 Getting sentiments for thread {thread_id}, user: {user_email}")
        sentiments = await get_thread_run_sentiments(user_email, thread_id)
        print__chat_sentiments_debug(f"🔍 Retrieved {len(sentiments)} sentiment values")
        
        print__sentiment_flow(f"✅ Retrieved {len(sentiments)} sentiment values")
        print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - SUCCESSFUL EXIT")
        return sentiments
    
    except Exception as e:
        print__chat_sentiments_debug(f"🚨 Exception in chat sentiments processing: {type(e).__name__}: {str(e)}")
        print__chat_sentiments_debug(f"🚨 Chat sentiments processing traceback: {traceback.format_exc()}")
        print__sentiment_flow(f"❌ Failed to get sentiments for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiments: {e}")

@router.get("/chat-threads")
async def get_chat_threads(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=50, description="Number of threads per page"),
    user=Depends(get_current_user)
) -> PaginatedChatThreadsResponse:
    """Get paginated chat threads for the authenticated user."""
    
    print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ENTRY POINT")
    print__chat_threads_debug(f"🔍 Request parameters: page={page}, limit={limit}")
    
    try:
        user_email = user["email"]
        print__chat_threads_debug(f"🔍 User email extracted: {user_email}")
        print__chat_threads_debug(f"Loading chat threads for user: {user_email} (page: {page}, limit: {limit})")
        
        print__chat_threads_debug("🔍 Starting simplified approach")
        print__chat_threads_debug("Getting chat threads with simplified approach")
        
        # Get total count first
        print__chat_threads_debug("🔍 Getting total threads count")
        print__chat_threads_debug(f"Getting chat threads count for user: {user_email}")
        total_count = await get_user_chat_threads_count(user_email)
        print__chat_threads_debug(f"🔍 Total count retrieved: {total_count}")
        print__chat_threads_debug(f"Total threads count for user {user_email}: {total_count}")
        
        # Calculate offset for pagination
        offset = (page - 1) * limit
        print__chat_threads_debug(f"🔍 Calculated offset: {offset}")
        
        # Get threads for this page
        print__chat_threads_debug(f"🔍 Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})")
        print__chat_threads_debug(f"Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})")
        threads = await get_user_chat_threads(user_email, limit=limit, offset=offset)
        print__chat_threads_debug(f"🔍 Retrieved threads: {threads}")
        if threads is None:
            print__chat_threads_debug("get_user_chat_threads returned None! Setting to empty list.")
            threads = []
        print__chat_threads_debug(f"🔍 Retrieved {len(threads)} threads from database")
        print__chat_threads_debug(f"Retrieved {len(threads)} threads for user {user_email}")
        
        # Try/except around the for-loop to catch and print any errors
        try:
            chat_thread_responses = []
            for thread in threads:
                print("[GENERIC-DEBUG] Processing thread dict:", thread)
                chat_thread_response = ChatThreadResponse(
                    thread_id=thread['thread_id'],
                    latest_timestamp=thread['latest_timestamp'],
                    run_count=thread['run_count'],
                    title=thread['title'],
                    full_prompt=thread['full_prompt']
                )
                chat_thread_responses.append(chat_thread_response)
        except Exception as e:
            print("[GENERIC-ERROR] Exception in /chat-threads for-loop:", e)
            print(traceback.format_exc())
            # Return empty result on error
            return PaginatedChatThreadsResponse(
                threads=[],
                total_count=0,
                page=page,
                limit=limit,
                has_more=False
            )
        
        # Convert to response format
        print__chat_threads_debug("🔍 Converting threads to response format")
        chat_thread_responses = []
        for thread in threads:
            chat_thread_response = ChatThreadResponse(
                thread_id=thread['thread_id'],
                latest_timestamp=thread['latest_timestamp'],
                run_count=thread['run_count'],
                title=thread['title'],
                full_prompt=thread['full_prompt']
            )
            chat_thread_responses.append(chat_thread_response)
        
        # Calculate pagination info
        has_more = (offset + len(chat_thread_responses)) < total_count
        print__chat_threads_debug(f"🔍 Pagination calculated: has_more={has_more}")
        
        print__chat_threads_debug(f"Retrieved {len(threads)} threads for user {user_email} (total: {total_count})")
        print__chat_threads_debug(f"Returning {len(chat_thread_responses)} threads to frontend (page {page}/{(total_count + limit - 1) // limit})")
        
        result = PaginatedChatThreadsResponse(
            threads=chat_thread_responses,
            total_count=total_count,
            page=page,
            limit=limit,
            has_more=has_more
        )
        print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - SUCCESSFUL EXIT")
        return result
        
    except Exception as e:
        print__chat_threads_debug(f"🚨 Exception in chat threads processing: {type(e).__name__}: {str(e)}")
        print__chat_threads_debug(f"🚨 Chat threads processing traceback: {traceback.format_exc()}")
        print__chat_threads_debug(f"❌ Error getting chat threads: {e}")
        print__chat_threads_debug(f"Full traceback: {traceback.format_exc()}")
        
        # Return error response
        result = PaginatedChatThreadsResponse(
            threads=[],
            total_count=0,
            page=page,
            limit=limit,
            has_more=False
        )
        print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ERROR EXIT")
        return result

@router.delete("/chat/{thread_id}")
async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Delete all PostgreSQL checkpoint records and thread entries for a specific thread_id."""
    
    print__delete_chat_debug(f"🔍 DELETE_CHAT ENDPOINT - ENTRY POINT")
    print__delete_chat_debug(f"🔍 Request received: thread_id={thread_id}")
    
    user_email = user.get("email")
    print__delete_chat_debug(f"🔍 User email extracted: {user_email}")
    
    if not user_email:
        print__delete_chat_debug(f"🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__delete_chat_debug(f"🗑️ Deleting chat thread {thread_id} for user {user_email}")
    
    try:
        # Get a healthy checkpointer
        print__delete_chat_debug(f"🔧 DEBUG: Getting healthy checkpointer...")
        checkpointer = await get_healthy_checkpointer()
        print__delete_chat_debug(f"🔧 DEBUG: Checkpointer type: {type(checkpointer).__name__}")
        
        # Check if we have a PostgreSQL checkpointer (not InMemorySaver)
        print__delete_chat_debug(f"🔧 DEBUG: Checking if checkpointer has 'conn' attribute...")
        if not hasattr(checkpointer, 'conn'):
            print__delete_chat_debug(f"⚠️ No PostgreSQL checkpointer available - nothing to delete")
            return {"message": "No PostgreSQL checkpointer available - nothing to delete"}
        
        print__delete_chat_debug(f"🔧 DEBUG: Checkpointer has 'conn' attribute")
        print__delete_chat_debug(f"🔧 DEBUG: checkpointer.conn type: {type(checkpointer.conn).__name__}")
        
        # Access the connection through the conn attribute
        conn_obj = checkpointer.conn
        print__delete_chat_debug(f"🔧 DEBUG: Connection object set, type: {type(conn_obj).__name__}")
        
        # FIXED: Handle both connection pool and single connection cases
        if hasattr(conn_obj, 'connection') and callable(getattr(conn_obj, 'connection', None)):
            # It's a connection pool - use pool.connection()
            print__delete_chat_debug(f"🔧 DEBUG: Using connection pool pattern...")
            async with conn_obj.connection() as conn:
                print__delete_chat_debug(f"🔧 DEBUG: Successfully got connection from pool, type: {type(conn).__name__}")
                result_data = await perform_deletion_operations(conn, user_email, thread_id)
                return result_data
        else:
            # It's a single connection - use it directly
            print__delete_chat_debug(f"🔧 DEBUG: Using single connection pattern...")
            conn = conn_obj
            print__delete_chat_debug(f"🔧 DEBUG: Using direct connection, type: {type(conn).__name__}")
            result_data = await perform_deletion_operations(conn, user_email, thread_id)
            return result_data
            
    except Exception as e:
        error_msg = str(e)
        print__delete_chat_debug(f"❌ Failed to delete checkpoint records for thread {thread_id}: {e}")
        print__delete_chat_debug(f"🔧 DEBUG: Main exception type: {type(e).__name__}")
        print__delete_chat_debug(f"🔧 DEBUG: Main exception traceback: {traceback.format_exc()}")
        
        # If it's a connection error, don't treat it as a failure since it means 
        # there are likely no records to delete anyway
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print__delete_chat_debug(f"⚠️ PostgreSQL connection unavailable - no records to delete")
            return {
                "message": "PostgreSQL connection unavailable - no records to delete", 
                "thread_id": thread_id,
                "user_email": user_email,
                "warning": "Database connection issues"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint records: {e}")

# Placeholder endpoints that still need extensive extraction
@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user=Depends(get_current_user)):
    """
    Submit feedback endpoint - MOVED TO DEDICATED FEEDBACK ROUTES
    
    This endpoint has been moved to api/routes/feedback.py for Phase 8.4
    """
    raise HTTPException(
        status_code=501,
        detail="Feedback endpoint moved to api/routes/feedback.py - use dedicated feedback router"
    )

@router.post("/sentiment")
async def update_sentiment(request: SentimentRequest, user=Depends(get_current_user)):
    """
    Update sentiment endpoint - MOVED TO DEDICATED FEEDBACK ROUTES
    
    This endpoint has been moved to api/routes/feedback.py for Phase 8.4
    """
    raise HTTPException(
        status_code=501,
        detail="Sentiment endpoint moved to api/routes/feedback.py - use dedicated feedback router"
    )

@router.get("/chat/{thread_id}/messages")
async def get_chat_messages(thread_id: str, user=Depends(get_current_user)) -> List[ChatMessage]:
    """
    Get chat messages endpoint - MOVED TO DEDICATED MESSAGE ROUTES
    
    This endpoint has been moved to api/routes/messages.py for Phase 8.6
    """
    raise HTTPException(
        status_code=501,
        detail="Chat messages endpoint moved to api/routes/messages.py - use dedicated message router"
    )

@router.get("/chat/{thread_id}/run-ids")
async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
    """
    Get message run IDs endpoint - MOVED TO DEDICATED MESSAGE ROUTES
    
    This endpoint has been moved to api/routes/messages.py for Phase 8.6
    """
    raise HTTPException(
        status_code=501,
        detail="Message run IDs endpoint moved to api/routes/messages.py - use dedicated message router"
    )

@router.get("/chat/all-messages")
async def get_all_chat_messages(user=Depends(get_current_user)) -> Dict:
    """
    Get all chat messages endpoint - COMPLEX DEPENDENCIES
    
    This endpoint requires extraction of complex database queries and concurrency management.
    """
    print__chat_messages_debug("🔧 REFACTORING NOTE: This endpoint needs complex database and concurrency extraction")
    
    raise HTTPException(
        status_code=501,
        detail="All chat messages endpoint requires extraction of complex database queries and concurrency management."
    ) 