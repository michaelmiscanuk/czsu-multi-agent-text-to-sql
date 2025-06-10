import sys
import asyncio
from contextlib import asynccontextmanager

# Configure asyncio event loop policy for Windows compatibility with psycopg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Query, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import requests
import jwt
import json
import os
from jwt.algorithms import RSAAlgorithm
from langchain_core.messages import BaseMessage

from main import main as analysis_main
from my_agent.utils.postgres_checkpointer import (
    get_postgres_checkpointer,
    create_thread_run_entry,
    get_user_chat_threads,
    delete_user_thread_entries,
    get_conversation_messages_from_checkpoints,
    get_queries_and_results_from_latest_checkpoint
)

# Global shared checkpointer for conversation memory across API requests
# This ensures that conversation state is preserved between frontend requests using PostgreSQL
GLOBAL_CHECKPOINTER = None

async def initialize_checkpointer():
    """Initialize the global PostgreSQL checkpointer on startup."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER is None:
        try:
            print("🔗 Initializing PostgreSQL checkpointer and chat system...")
            GLOBAL_CHECKPOINTER = await get_postgres_checkpointer()
            print("✓ Global PostgreSQL checkpointer initialized successfully")
            print("✓ users_threads_runs table verified/created")
        except Exception as e:
            print(f"✗ Failed to initialize PostgreSQL checkpointer: {e}")
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print("⚠ Falling back to InMemorySaver")

async def cleanup_checkpointer():
    """Clean up resources on app shutdown."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
        try:
            await GLOBAL_CHECKPOINTER.conn.close()
            print("✓ PostgreSQL connection pool closed cleanly")
        except Exception as e:
            print(f"⚠ Error closing connection pool: {e}")

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance, recreating if necessary."""
    global GLOBAL_CHECKPOINTER
    
    # Check if current checkpointer is healthy
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn'):
        try:
            # Quick health check
            async with GLOBAL_CHECKPOINTER.conn.connection() as conn:
                await conn.execute("SELECT 1")
            return GLOBAL_CHECKPOINTER
        except Exception as e:
            print(f"⚠ Checkpointer unhealthy, recreating: {e}")
            # Try to cleanup old pool
            try:
                if GLOBAL_CHECKPOINTER.conn:
                    await GLOBAL_CHECKPOINTER.conn.close()
            except:
                pass
            GLOBAL_CHECKPOINTER = None
    
    # Create new checkpointer
    try:
        GLOBAL_CHECKPOINTER = await get_postgres_checkpointer()
        return GLOBAL_CHECKPOINTER
    except Exception as e:
        print(f"⚠ Failed to recreate checkpointer: {e}")
        from langgraph.checkpoint.memory import InMemorySaver
        GLOBAL_CHECKPOINTER = InMemorySaver()
        return GLOBAL_CHECKPOINTER

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_checkpointer()
    yield
    # Shutdown
    await cleanup_checkpointer()

app = FastAPI(lifespan=lifespan)

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    prompt: str
    thread_id: str

class ChatThreadResponse(BaseModel):
    thread_id: str
    latest_timestamp: str
    run_count: int

class ChatMessage(BaseModel):
    id: str
    threadId: str
    user: str
    content: str
    isUser: bool
    createdAt: int
    error: Optional[str] = None
    meta: Optional[dict] = None
    queriesAndResults: Optional[List[List[str]]] = None
    isLoading: Optional[bool] = None
    startedAt: Optional[int] = None
    isError: Optional[bool] = None

GOOGLE_JWK_URL = "https://www.googleapis.com/oauth2/v3/certs"

# Helper to verify Google JWT
def verify_google_jwt(token: str):
    # Get Google public keys
    jwks = requests.get(GOOGLE_JWK_URL).json()
    unverified_header = jwt.get_unverified_header(token)
    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            public_key = RSAAlgorithm.from_jwk(key)
            try:
                # Debug: print the audience in the token and the expected audience
                unverified_payload = jwt.decode(token, options={"verify_signature": False})
                print("[DEBUG] Token aud:", unverified_payload.get("aud"))
                print("[DEBUG] Backend GOOGLE_CLIENT_ID:", os.getenv("GOOGLE_CLIENT_ID"))
                payload = jwt.decode(token, public_key, algorithms=["RS256"], audience=os.getenv("GOOGLE_CLIENT_ID"))
                return payload
            except Exception as e:
                print("[DEBUG] JWT decode error:", e)
                raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    raise HTTPException(status_code=401, detail="Public key not found")

# Dependency for JWT authentication
def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    return verify_google_jwt(token)

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """Analyze request with robust error handling for checkpointer issues."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-PostgreSQL] 📥 Analysis request - User: {user_email}, Thread: {request.thread_id}")
    
    # Get a healthy checkpointer
    checkpointer = await get_healthy_checkpointer()
    
    try:
        # Create thread run entry before analysis
        print(f"[API-PostgreSQL] 🔄 Creating thread run entry for user {user_email}, thread {request.thread_id}")
        run_id = await create_thread_run_entry(user_email, request.thread_id)
        print(f"[API-PostgreSQL] ✅ Thread run entry created with run_id: {run_id}")
        
        result = await analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=checkpointer)
        
        # Add run_id to result
        result["run_id"] = run_id
        print(f"[API-PostgreSQL] 🎉 Analysis completed successfully for run_id: {run_id}")
        
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"[API-PostgreSQL] ❌ Analysis error for user {user_email}, thread {request.thread_id}: {error_msg}")
        
        # Handle specific database connection errors
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print(f"[API-PostgreSQL] ⚠ Database connection error detected: {e}")
            print("[API-PostgreSQL] ⚠ Attempting to use fresh InMemorySaver...")
            
            # Use a completely fresh InMemorySaver for this request
            from langgraph.checkpoint.memory import InMemorySaver
            fallback_checkpointer = InMemorySaver()
            
            try:
                result = await analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=fallback_checkpointer)
                # Add warning to the result
                if isinstance(result, dict):
                    result["warning"] = "Persistent memory temporarily unavailable - using session-only memory"
                print(f"[API-PostgreSQL] ✅ Fallback analysis completed for thread {request.thread_id}")
                return result
            except Exception as fallback_error:
                print(f"[API-PostgreSQL] ✗ Fallback also failed: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {fallback_error}")
        else:
            # Re-raise non-connection errors
            print(f"[API-PostgreSQL] ✗ Analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.get("/chat-threads")
async def get_chat_threads(user=Depends(get_current_user)) -> List[ChatThreadResponse]:
    """Get all chat threads for the authenticated user from PostgreSQL."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-PostgreSQL] 📥 Loading chat threads for user: {user_email}")
    
    try:
        threads = await get_user_chat_threads(user_email)
        print(f"[API-PostgreSQL] ✅ Retrieved {len(threads)} threads for user {user_email}")
        
        # Convert to response format
        response_threads = []
        for thread in threads:
            response_threads.append(ChatThreadResponse(
                thread_id=thread["thread_id"],
                latest_timestamp=thread["latest_timestamp"].isoformat(),
                run_count=thread["run_count"]
            ))
        
        print(f"[API-PostgreSQL] 📤 Returning {len(response_threads)} threads to frontend")
        return response_threads
        
    except Exception as e:
        print(f"[API-PostgreSQL] ❌ Failed to get chat threads for user {user_email}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat threads: {e}")

@app.delete("/chat/{thread_id}")
async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Delete all PostgreSQL checkpoint records and thread entries for a specific thread_id."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-PostgreSQL] 🗑️ Deleting chat thread {thread_id} for user {user_email}")
    
    # Get a healthy checkpointer
    checkpointer = await get_healthy_checkpointer()
    
    # Check if we have a PostgreSQL checkpointer (not InMemorySaver)
    if not hasattr(checkpointer, 'conn'):
        print(f"[API-PostgreSQL] ⚠ No PostgreSQL checkpointer available - nothing to delete")
        return {"message": "No PostgreSQL checkpointer available - nothing to delete"}
    
    try:
        # Access the connection pool through the conn attribute
        pool = checkpointer.conn
        
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            print(f"[API-PostgreSQL] 🔄 Deleting from checkpoint tables for thread {thread_id}")
            
            # Delete from all checkpoint tables
            tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
            deleted_counts = {}
            
            for table in tables:
                try:
                    # First check if the table exists
                    result = await conn.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table,))
                    
                    table_exists = await result.fetchone()
                    if not table_exists or not table_exists[0]:
                        print(f"[API-PostgreSQL] ⚠ Table {table} does not exist, skipping")
                        deleted_counts[table] = 0
                        continue
                    
                    # Delete records for this thread_id
                    result = await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s",
                        (thread_id,)
                    )
                    
                    deleted_counts[table] = result.rowcount if hasattr(result, 'rowcount') else 0
                    print(f"[API-PostgreSQL] ✅ Deleted {deleted_counts[table]} records from {table} for thread_id: {thread_id}")
                    
                except Exception as table_error:
                    print(f"[API-PostgreSQL] ⚠ Error deleting from table {table}: {table_error}")
                    deleted_counts[table] = f"Error: {str(table_error)}"
            
            # Delete from users_threads_runs table
            print(f"[API-PostgreSQL] 🔄 Deleting thread entries for user {user_email}, thread {thread_id}")
            thread_entries_result = await delete_user_thread_entries(user_email, thread_id)
            print(f"[API-PostgreSQL] ✅ Thread entries deletion result: {thread_entries_result}")
            
            result_data = {
                "message": f"Checkpoint records and thread entries deleted for thread_id: {thread_id}",
                "deleted_counts": deleted_counts,
                "thread_entries_deleted": thread_entries_result,
                "thread_id": thread_id,
                "user_email": user_email
            }
            
            print(f"[API-PostgreSQL] 🎉 Successfully deleted thread {thread_id} for user {user_email}")
            return result_data
            
    except Exception as e:
        error_msg = str(e)
        print(f"[API-PostgreSQL] ❌ Failed to delete checkpoint records for thread {thread_id}: {e}")
        
        # If it's a connection error, don't treat it as a failure since it means 
        # there are likely no records to delete anyway
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print(f"[API-PostgreSQL] ⚠ PostgreSQL connection unavailable - no records to delete")
            return {
                "message": "PostgreSQL connection unavailable - no records to delete", 
                "thread_id": thread_id,
                "user_email": user_email,
                "warning": "Database connection issues"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint records: {e}")

@app.get("/catalog")
def get_catalog(
    page: int = Query(1, ge=1),
    q: Optional[str] = None,
    page_size: int = Query(10, ge=1, le=10000),
    user=Depends(get_current_user)
):
    db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
    offset = (page - 1) * page_size
    where_clause = ""
    params = []
    if q:
        where_clause = "WHERE selection_code LIKE ? OR extended_description LIKE ?"
        like_q = f"%{q}%"
        params.extend([like_q, like_q])
    query = f"""
        SELECT selection_code, extended_description
        FROM selection_descriptions
        {where_clause}
        ORDER BY selection_code
        LIMIT ? OFFSET ?
    """
    params.extend([page_size, offset])
    count_query = f"SELECT COUNT(*) FROM selection_descriptions {where_clause}"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(count_query, params[:-2] if q else [])
        total = cursor.fetchone()[0]
        cursor.execute(query, params)
        rows = cursor.fetchall()
    results = [
        {"selection_code": row[0], "extended_description": row[1]} for row in rows
    ]
    return {"results": results, "total": total, "page": page, "page_size": page_size}

@app.get("/data-tables")
def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
    db_path = "data/czsu_data.db"
    desc_db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
    if q:
        q_lower = q.lower()
        tables = [t for t in tables if q_lower in t.lower()]
    # Fetch short_descriptions from the other DB
    desc_map = {}
    try:
        with sqlite3.connect(desc_db_path) as desc_conn:
            desc_cursor = desc_conn.cursor()
            desc_cursor.execute("SELECT selection_code, short_description FROM selection_descriptions")
            for code, short_desc in desc_cursor.fetchall():
                desc_map[code] = short_desc
    except Exception as e:
        print(f"[DEBUG] Error fetching short_descriptions: {e}")
    # Build result list
    result = [
        {"selection_code": t, "short_description": desc_map.get(t, "")}
        for t in tables
    ]
    return {"tables": result}

@app.get("/data-table")
def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
    db_path = "data/czsu_data.db"
    if not table:
        print("[DEBUG] No table specified")
        return {"columns": [], "rows": []}
    print(f"[DEBUG] Requested table: {table}")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            print(f"[DEBUG] Columns: {columns}, Rows count: {len(rows)}")
        except Exception as e:
            print(f"[DEBUG] Error fetching table '{table}': {e}")
            return {"columns": [], "rows": []}
    return {"columns": columns, "rows": rows}

@app.get("/chat/{thread_id}/messages")
async def get_chat_messages(thread_id: str, user=Depends(get_current_user)) -> List[ChatMessage]:
    """Load conversation messages from PostgreSQL checkpoint history that preserves original user messages."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-PostgreSQL] 📥 Loading checkpoint messages for thread {thread_id}, user: {user_email}")
    
    try:
        # Get a healthy checkpointer
        checkpointer = await get_healthy_checkpointer()
        
        if not hasattr(checkpointer, 'conn'):
            print(f"[API-PostgreSQL] ⚠ No PostgreSQL checkpointer available - returning empty messages")
            return []
        
        # Get conversation messages from checkpoint history
        stored_messages = await get_conversation_messages_from_checkpoints(checkpointer, thread_id)
        
        if not stored_messages:
            print(f"[API-PostgreSQL] ⚠ No messages found in checkpoints for thread {thread_id}")
            return []
        
        print(f"[API-PostgreSQL] 📄 Found {len(stored_messages)} messages from checkpoints")
        
        # Get additional metadata from latest checkpoint (like queries_and_results and top_selection_codes)
        queries_and_results = await get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id)
        
        # Get dataset information and SQL query from latest checkpoint
        datasets_used = []
        sql_query = None
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = await checkpointer.aget_tuple(config)
            
            if state_snapshot and state_snapshot.checkpoint:
                channel_values = state_snapshot.checkpoint.get("channel_values", {})
                top_selection_codes = channel_values.get("top_selection_codes", [])
                
                # Add datasets used information
                if top_selection_codes:
                    datasets_used = top_selection_codes
                    print(f"[API-PostgreSQL] 📊 Found datasets used: {datasets_used}")
                
                # Extract SQL query from queries_and_results for SQL button
                if queries_and_results:
                    # Get the last (most recent) SQL query
                    sql_query = queries_and_results[-1][0] if queries_and_results[-1] else None
                    print(f"[API-PostgreSQL] 🔍 Found SQL query: {sql_query[:50] if sql_query else 'None'}...")
                
        except Exception as e:
            print(f"[API-PostgreSQL] ⚠ Could not get datasets/SQL from checkpoint: {e}")
        
        # Convert stored messages to frontend format
        chat_messages = []
        
        for i, stored_msg in enumerate(stored_messages):
            # Debug: Log the raw stored message
            print(f"[API-PostgreSQL] 🔍 Processing stored message {i+1}: is_user={stored_msg.get('is_user')}, content='{stored_msg.get('content', '')[:30]}...'")
            
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
                meta_info["source"] = "checkpoint_history"
                print(f"[API-PostgreSQL] 🔍 Added metadata to AI message: datasets={len(datasets_used)}, sql={'Yes' if sql_query else 'No'}")
            
            # Convert queries_and_results for AI messages
            queries_results_for_frontend = None
            if not stored_msg["is_user"] and queries_and_results:
                queries_results_for_frontend = queries_and_results
            
            # Create ChatMessage with explicit debugging
            is_user_flag = stored_msg["is_user"]
            print(f"[API-PostgreSQL] 🔍 Creating ChatMessage: isUser={is_user_flag}")
            
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
            print(f"[API-PostgreSQL] 🔍 ChatMessage created: isUser={chat_message.isUser}, user='{chat_message.user}'")
            
            chat_messages.append(chat_message)
        
        print(f"[API-PostgreSQL] ✅ Converted {len(chat_messages)} messages to frontend format")
        
        # Log the messages for debugging
        for i, msg in enumerate(chat_messages):
            user_type = "👤 User" if msg.isUser else "🤖 AI"
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            datasets_info = f" (datasets: {msg.meta.get('datasetsUsed', [])})" if msg.meta and msg.meta.get('datasetsUsed') else ""
            sql_info = f" (SQL: {msg.meta.get('sqlQuery', 'None')[:30]}...)" if msg.meta and msg.meta.get('sqlQuery') else ""
            print(f"[API-PostgreSQL] {i+1}. {user_type}: {content_preview}{datasets_info}{sql_info}")
        
        return chat_messages
        
    except Exception as e:
        error_msg = str(e)
        print(f"[API-PostgreSQL] ❌ Failed to load checkpoint messages for thread {thread_id}: {e}")
        
        # Handle specific database connection errors gracefully
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print(f"[API-PostgreSQL] ⚠ Database connection error - returning empty messages")
            return []
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load checkpoint messages: {e}") 