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
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import models
from api.models.responses import ChatMessage

# Import debug functions
from api.utils.debug import (
    print__api_postgresql, print__feedback_flow,
    print__chat_messages_debug
)

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
from my_agent.utils.postgres_checkpointer import (
    get_healthy_checkpointer,
    get_conversation_messages_from_checkpoints,
    get_queries_and_results_from_latest_checkpoint
)

# Create router for message endpoints
router = APIRouter()

@router.get("/chat/{thread_id}/messages")
async def get_chat_messages(thread_id: str, user=Depends(get_current_user)) -> List[ChatMessage]:
    """Load conversation messages from PostgreSQL checkpoint history that preserves original user messages."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__api_postgresql(f"📥 Loading checkpoint messages for thread {thread_id}, user: {user_email}")
    
    try:
        # 🔒 SECURITY CHECK: Verify user owns this thread before retrieving messages
        print__api_postgresql(f"🔒 Verifying thread ownership for user: {user_email}, thread: {thread_id}")
        
        # Check if this user has any entries in users_threads_runs for this thread
        checkpointer = await get_healthy_checkpointer()
        
        if not hasattr(checkpointer, 'conn'):
            print__api_postgresql(f"⚠️ No PostgreSQL checkpointer available - returning empty messages")
            return []
        
        # Verify thread ownership using users_threads_runs table
        async with checkpointer.conn.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
            SELECT COUNT(*) FROM users_threads_runs 
            WHERE email = %s AND thread_id = %s
        """, (user_email, thread_id))
        
                ownership_row = await cur.fetchone()
                thread_entries_count = ownership_row[0] if ownership_row else 0
            
            if thread_entries_count == 0:
                print__api_postgresql(f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - access denied")
                # Return empty instead of error to avoid information disclosure
                return []
            
            print__api_postgresql(f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
        
        # Get conversation messages from checkpoint history
        stored_messages = await get_conversation_messages_from_checkpoints(checkpointer, thread_id, user_email)
        
        if not stored_messages:
            print__api_postgresql(f"⚠ No messages found in checkpoints for thread {thread_id}")
            return []
        
        print__api_postgresql(f"📄 Found {len(stored_messages)} messages from checkpoints")
        
        # Get additional metadata from latest checkpoint (like queries_and_results and top_selection_codes)
        queries_and_results = await get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id)
        
        # Get dataset information and SQL query from latest checkpoint
        datasets_used = []
        sql_query = None
        top_chunks = []
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = await checkpointer.aget_tuple(config)
            
            if state_snapshot and state_snapshot.checkpoint:
                channel_values = state_snapshot.checkpoint.get("channel_values", {})
                top_selection_codes = channel_values.get("top_selection_codes", [])
                
                # Use the datasets directly
                datasets_used = top_selection_codes
                
                # Get PDF chunks from checkpoint state
                checkpoint_top_chunks = channel_values.get("top_chunks", [])
                print__api_postgresql(f"📄 Found {len(checkpoint_top_chunks)} PDF chunks in checkpoint for thread {thread_id}")
                
                # Convert Document objects to serializable format
                if checkpoint_top_chunks:
                    for chunk in checkpoint_top_chunks:
                        chunk_data = {
                            "content": chunk.page_content if hasattr(chunk, 'page_content') else str(chunk),
                            "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                        }
                        top_chunks.append(chunk_data)
                    print__api_postgresql(f"📄 Serialized {len(top_chunks)} PDF chunks for frontend")
                
                # Extract SQL query from queries_and_results for SQL button
                if queries_and_results:
                    # Get the last (most recent) SQL query
                    sql_query = queries_and_results[-1][0] if queries_and_results[-1] else None
            
        except Exception as e:
            print__api_postgresql(f"⚠️ Could not get datasets/SQL/chunks from checkpoint: {e}")
            print__api_postgresql(f"🔧 Using fallback empty values: datasets=[], sql=None, chunks=[]")
        
        # Convert stored messages to frontend format
        chat_messages = []
        
        for i, stored_msg in enumerate(stored_messages):
            # Debug: Log the raw stored message
            print__api_postgresql(f"🔍 Processing stored message {i+1}: is_user={stored_msg.get('is_user')}, content='{stored_msg.get('content', '')[:30]}...'")
            
            # Create meta information for messages
            meta_info = {}
            
            # For AI messages, include queries/results, datasets used, and SQL query
            if not stored_msg["is_user"]:
                if queries_and_results:
                    meta_info["queriesAndResults"] = queries_and_results
                if datasets_used:
                    meta_info["datasetsUsed"] = datasets_used
                if sql_query:
                    meta_info["sqlQuery"] = sql_query
                if top_chunks:
                    meta_info["topChunks"] = top_chunks
                meta_info["source"] = "checkpoint_history"
                print__api_postgresql(f"🔍 Added metadata to AI message: datasets={len(datasets_used)}, sql={'Yes' if sql_query else 'No'}, chunks={len(top_chunks)}")
            
            # Convert queries_and_results for AI messages
            queries_results_for_frontend = None
            if not stored_msg["is_user"] and queries_and_results:
                queries_results_for_frontend = queries_and_results
            
            # Create ChatMessage with explicit debugging
            is_user_flag = stored_msg["is_user"]
            print__api_postgresql(f"🔍 Creating ChatMessage: isUser={is_user_flag}")
            
            chat_message = ChatMessage(
                id=stored_msg["id"],
                threadId=thread_id,
                user=user_email if is_user_flag else "AI",
                content=stored_msg["content"],
                isUser=is_user_flag,  # Explicitly use the flag
                createdAt=int(stored_msg["timestamp"].timestamp() * 1000),
                error=None,
                meta=meta_info if meta_info else None,  # Only add meta if it has content
                queriesAndResults=queries_results_for_frontend,
                isLoading=False,
                startedAt=None,
                isError=False
            )
            
            # Debug: Verify the ChatMessage was created correctly
            print__api_postgresql(f"🔍 ChatMessage created: isUser={chat_message.isUser}, user='{chat_message.user}'")
            
            chat_messages.append(chat_message)
        
        print__api_postgresql(f"✅ Converted {len(chat_messages)} messages to frontend format")
        
        # Log the messages for debugging
        for i, msg in enumerate(chat_messages):
            user_type = "👤 User" if msg.isUser else "🤖 AI"
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            datasets_info = f" (datasets: {msg.meta.get('datasetsUsed', [])})" if msg.meta and msg.meta.get('datasetsUsed') else ""
            sql_info = f" (SQL: {msg.meta.get('sqlQuery', 'None')[:30]}...)" if msg.meta and msg.meta.get('sqlQuery') else ""
            print__api_postgresql(f"{i+1}. {user_type}: {content_preview}{datasets_info}{sql_info}")
        
        return chat_messages
        
    except Exception as e:
        error_msg = str(e)
        print__api_postgresql(f"❌ Failed to load checkpoint messages for thread {thread_id}: {e}")
        
        # Handle specific database connection errors gracefully
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print__api_postgresql(f"⚠ Database connection error - returning empty messages")
            return []
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load checkpoint messages: {e}")

@router.get("/chat/{thread_id}/run-ids")
async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
    """Get run_ids for messages in a thread to enable feedback submission."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__feedback_flow(f"🔍 Fetching run_ids for thread {thread_id}")
    
    try:
        pool = await get_healthy_checkpointer()
        pool = pool.conn if hasattr(pool, 'conn') else None
        
        if not pool:
            print__feedback_flow("⚠ No pool available for run_id lookup")
            return {"run_ids": []}
        
        async with pool.connection() as conn:
            print__feedback_flow(f"📊 Executing SQL query for thread {thread_id}")
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT run_id, prompt, timestamp
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                    ORDER BY timestamp ASC
                """, (user_email, thread_id))
                
                run_id_data = []
                rows = await cur.fetchall()
                for row in rows:
                    print__feedback_flow(f"📝 Processing database row - run_id: {row[0]}, prompt: {row[1][:50]}...")
                    try:
                        run_uuid = str(uuid.UUID(row[0])) if row[0] else None
                        if run_uuid:
                            run_id_data.append({
                                "run_id": run_uuid,
                                "prompt": row[1],
                                "timestamp": row[2].isoformat()
                            })
                            print__feedback_flow(f"✅ Valid UUID found: {run_uuid}")
                        else:
                            print__feedback_flow(f"⚠ Null run_id found for prompt: {row[1][:50]}...")
                    except ValueError:
                        print__feedback_flow(f"❌ Invalid UUID in database: {row[0]}")
                        continue
                
                print__feedback_flow(f"📊 Total valid run_ids found: {len(run_id_data)}")
                return {"run_ids": run_id_data}
            
    except Exception as e:
        print__feedback_flow(f"🚨 Error fetching run_ids: {str(e)}")
        return {"run_ids": []} 