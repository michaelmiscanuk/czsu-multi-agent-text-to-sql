import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

# Configure asyncio event loop policy for Windows compatibility with psycopg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Query, HTTPException, Header, Depends, Request
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
from langsmith import Client

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
            print(f"🔍 Current global checkpointer state: {GLOBAL_CHECKPOINTER}")
            
            GLOBAL_CHECKPOINTER = await get_postgres_checkpointer()
            
            # Verify the checkpointer is healthy
            if hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
                print(f"✓ Checkpointer has connection pool: closed={GLOBAL_CHECKPOINTER.conn.closed}")
            else:
                print("⚠ Checkpointer does not have connection pool")
            
            print("✓ Global PostgreSQL checkpointer initialized successfully")
            print("✓ users_threads_runs table verified/created")
        except Exception as e:
            print(f"✗ Failed to initialize PostgreSQL checkpointer: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print("⚠ Falling back to InMemorySaver")
    else:
        print("⚠ Global checkpointer already exists - skipping initialization")

async def cleanup_checkpointer():
    """Clean up resources on app shutdown."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
        try:
            # Check if pool is already closed before trying to close it
            if not GLOBAL_CHECKPOINTER.conn.closed:
                await GLOBAL_CHECKPOINTER.conn.close()
                print("✓ PostgreSQL connection pool closed cleanly")
            else:
                print("⚠ PostgreSQL connection pool was already closed")
        except Exception as e:
            print(f"⚠ Error closing connection pool: {e}")
        finally:
            GLOBAL_CHECKPOINTER = None

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance, recreating if necessary."""
    global GLOBAL_CHECKPOINTER
    
    # Check if current checkpointer is healthy
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn'):
        try:
            # Check if the pool is closed (this is the key issue)
            if hasattr(GLOBAL_CHECKPOINTER.conn, 'closed') and GLOBAL_CHECKPOINTER.conn.closed:
                print(f"⚠ Checkpointer pool is closed, recreating...")
                GLOBAL_CHECKPOINTER = None
            else:
                # Quick health check
                async with GLOBAL_CHECKPOINTER.conn.connection() as conn:
                    await conn.execute("SELECT 1")
                return GLOBAL_CHECKPOINTER
        except Exception as e:
            print(f"⚠ Checkpointer unhealthy, recreating: {e}")
            # Try to cleanup old pool
            try:
                if GLOBAL_CHECKPOINTER.conn and not GLOBAL_CHECKPOINTER.conn.closed:
                    await GLOBAL_CHECKPOINTER.conn.close()
            except Exception as cleanup_error:
                print(f"⚠ Error during cleanup: {cleanup_error}")
            finally:
                GLOBAL_CHECKPOINTER = None
    
    # Create new checkpointer
    try:
        print("🔄 Creating fresh checkpointer...")
        GLOBAL_CHECKPOINTER = await get_postgres_checkpointer()
        print("✅ Fresh checkpointer created successfully")
        return GLOBAL_CHECKPOINTER
    except Exception as e:
        print(f"⚠ Failed to recreate checkpointer: {e}")
        from langgraph.checkpoint.memory import InMemorySaver
        GLOBAL_CHECKPOINTER = InMemorySaver()
        print("⚠ Falling back to InMemorySaver")
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

class FeedbackRequest(BaseModel):
    run_id: str
    feedback: int  # 1 for thumbs up, 0 for thumbs down
    comment: Optional[str] = None

class ChatThreadResponse(BaseModel):
    thread_id: str
    latest_timestamp: str
    run_count: int
    title: str  # Now includes the title from first prompt
    full_prompt: str  # Full prompt text for tooltip

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
    
    # Get a healthy checkpointer first (this will create fresh pools if needed)
    checkpointer = await get_healthy_checkpointer()
    
    try:
        # Create thread run entry before analysis - this generates the run_id we'll use for LangSmith
        print(f"[API-PostgreSQL] 🔄 Creating thread run entry for user {user_email}, thread {request.thread_id}, prompt: {request.prompt[:50]}...")
        run_id = await create_thread_run_entry(user_email, request.thread_id, request.prompt)
        print(f"[API-PostgreSQL] ✅ Thread run entry created with run_id: {run_id}")
        
        result = await analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=checkpointer, run_id=run_id)
        
        # Add run_id to result so frontend can use it for feedback
        result["run_id"] = run_id
        print(f"[API-PostgreSQL] 🎉 Analysis completed successfully for run_id: {run_id}")
        
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"[API-PostgreSQL] ❌ Analysis error for user {user_email}, thread {request.thread_id}: {error_msg}")
        
        # Handle specific database connection errors
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed", "pool", "closed"
        ]):
            print(f"[API-PostgreSQL] ⚠ Database connection error detected: {e}")
            print("[API-PostgreSQL] ⚠ Attempting to use fresh InMemorySaver...")
            
            # Use a completely fresh InMemorySaver for this request
            from langgraph.checkpoint.memory import InMemorySaver
            fallback_checkpointer = InMemorySaver()
            
            try:
                # For fallback, still generate a run_id for consistency
                fallback_run_id = str(uuid.uuid4())
                result = await analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=fallback_checkpointer, run_id=fallback_run_id)
                # Add warning to the result
                if isinstance(result, dict):
                    result["warning"] = "Persistent memory temporarily unavailable - using session-only memory"
                    result["run_id"] = fallback_run_id
                print(f"[API-PostgreSQL] ✅ Fallback analysis completed for thread {request.thread_id}")
                return result
            except Exception as fallback_error:
                print(f"[API-PostgreSQL] ✗ Fallback also failed: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {fallback_error}")
        else:
            # Re-raise non-connection errors
            print(f"[API-PostgreSQL] ✗ Analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user=Depends(get_current_user)):
    """Submit feedback for a specific run_id to LangSmith."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-LangSmith] 📥 Feedback request - User: {user_email}, Run ID: {request.run_id}, Feedback: {request.feedback}")
    
    try:
        # Initialize LangSmith client
        client = Client()
        
        # Create feedback with SENTIMENT key as requested
        client.create_feedback(
            request.run_id,
            key="SENTIMENT",
            score=request.feedback,  # 1 for thumbs up, 0 for thumbs down
            comment=request.comment if request.comment else None
        )
        
        print(f"[API-LangSmith] ✅ Feedback submitted successfully for run_id: {request.run_id}")
        return {"message": "Feedback submitted successfully", "run_id": request.run_id}
        
    except Exception as e:
        print(f"[API-LangSmith] ❌ Failed to submit feedback for run_id {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")

@app.get("/chat-threads")
async def get_chat_threads(user=Depends(get_current_user)) -> List[ChatThreadResponse]:
    """Get all chat threads for the authenticated user from PostgreSQL."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-PostgreSQL] 📥 Loading chat threads for user: {user_email}")
    
    try:
        # First try to use the checkpointer's connection pool
        checkpointer = await get_healthy_checkpointer()
        
        if hasattr(checkpointer, 'conn') and checkpointer.conn and not checkpointer.conn.closed:
            print(f"[API-PostgreSQL-Debug] 🔍 Using checkpointer connection pool")
            threads = await get_user_chat_threads(user_email, checkpointer.conn)
        else:
            print(f"[API-PostgreSQL-Debug] ⚠ Checkpointer connection pool not available, using direct connection")
            # Fallback to direct connection (this will create its own healthy pool)
            threads = await get_user_chat_threads(user_email)
        
        print(f"[API-PostgreSQL] ✅ Retrieved {len(threads)} threads for user {user_email}")
        
        if len(threads) == 0:
            print(f"[API-PostgreSQL-Debug] 🔍 No threads found - this might be expected for new users")
            print(f"[API-PostgreSQL-Debug] 🔍 User email: '{user_email}'")
        
        # Convert to response format
        response_threads = []
        for thread in threads:
            print(f"[API-PostgreSQL-Debug] 🔍 Processing thread: {thread}")
            response_threads.append(ChatThreadResponse(
                thread_id=thread["thread_id"],
                latest_timestamp=thread["latest_timestamp"].isoformat(),
                run_count=thread["run_count"],
                title=thread["title"],
                full_prompt=thread["full_prompt"]
            ))
        
        print(f"[API-PostgreSQL] 📤 Returning {len(response_threads)} threads to frontend")
        return response_threads
        
    except Exception as e:
        print(f"[API-PostgreSQL] ❌ Failed to get chat threads for user {user_email}: {e}")
        import traceback
        print(f"[API-PostgreSQL-Debug] 🔍 Full traceback: {traceback.format_exc()}")
        
        # If this is a pool-related error, try one more time with completely fresh connection
        error_msg = str(e)
        if any(keyword in error_msg.lower() for keyword in [
            "pool", "closed", "connection", "timeout", "operational error"
        ]):
            print(f"[API-PostgreSQL] 🔄 Attempting one final retry with fresh connection...")
            try:
                # This should create a completely fresh pool
                threads = await get_user_chat_threads(user_email)
                response_threads = []
                for thread in threads:
                    response_threads.append(ChatThreadResponse(
                        thread_id=thread["thread_id"],
                        latest_timestamp=thread["latest_timestamp"].isoformat(),
                        run_count=thread["run_count"],
                        title=thread["title"],
                        full_prompt=thread["full_prompt"]
                    ))
                print(f"[API-PostgreSQL] ✅ Retry successful - returning {len(response_threads)} threads")
                return response_threads
            except Exception as retry_error:
                print(f"[API-PostgreSQL] ❌ Retry also failed: {retry_error}")
        
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
            
            # Delete from users_threads_runs table directly within the same transaction
            print(f"[API-PostgreSQL] 🔄 Deleting thread entries from users_threads_runs for user {user_email}, thread {thread_id}")
            try:
                result = await conn.execute("""
                    DELETE FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (user_email, thread_id))
                
                users_threads_runs_deleted = result.rowcount if hasattr(result, 'rowcount') else 0
                print(f"[API-PostgreSQL] ✅ Deleted {users_threads_runs_deleted} entries from users_threads_runs for user {user_email}, thread {thread_id}")
                
                deleted_counts["users_threads_runs"] = users_threads_runs_deleted
                
            except Exception as e:
                print(f"[API-PostgreSQL] ❌ Error deleting from users_threads_runs: {e}")
                deleted_counts["users_threads_runs"] = f"Error: {str(e)}"
            
            # Also call the helper function for additional cleanup (backward compatibility)
            print(f"[API-PostgreSQL] 🔄 Additional cleanup via helper function...")
            thread_entries_result = await delete_user_thread_entries(user_email, thread_id, pool)
            print(f"[API-PostgreSQL] ✅ Helper function deletion result: {thread_entries_result}")
            
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
                
                # Filter to only include selection codes actually used in queries (same logic as main.py)
                if top_selection_codes and queries_and_results:
                    # Import the filtering function from main.py
                    import sys
                    from pathlib import Path
                    
                    # Get the filtering logic
                    def extract_table_names_from_sql(sql_query: str):
                        """Extract table names from SQL query FROM clauses."""
                        import re
                        
                        # Remove comments and normalize whitespace
                        sql_clean = re.sub(r'--.*?(?=\n|$)', '', sql_query, flags=re.MULTILINE)
                        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
                        sql_clean = ' '.join(sql_clean.split())
                        
                        # Pattern to match FROM clause with table names
                        from_pattern = r'\bFROM\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1(?:\s*,\s*(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\3)*'
                        
                        table_names = []
                        matches = re.finditer(from_pattern, sql_clean, re.IGNORECASE)
                        
                        for match in matches:
                            # Extract the main table name (group 2)
                            if match.group(2):
                                table_names.append(match.group(2).upper())
                            # Extract additional table names if comma-separated (group 4)
                            if match.group(4):
                                table_names.append(match.group(4).upper())
                        
                        # Also handle JOIN clauses
                        join_pattern = r'\bJOIN\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1'
                        join_matches = re.finditer(join_pattern, sql_clean, re.IGNORECASE)
                        
                        for match in join_matches:
                            if match.group(2):
                                table_names.append(match.group(2).upper())
                        
                        return list(set(table_names))  # Remove duplicates
                    
                    def get_used_selection_codes(queries_and_results, top_selection_codes):
                        """Filter top_selection_codes to only include those actually used in queries."""
                        if not queries_and_results or not top_selection_codes:
                            return []
                        
                        # Extract all table names used in queries
                        used_table_names = set()
                        for query, _ in queries_and_results:
                            if query:
                                table_names = extract_table_names_from_sql(query)
                                used_table_names.update(table_names)
                        
                        # Filter selection codes to only include those that match used table names
                        used_selection_codes = []
                        for selection_code in top_selection_codes:
                            if selection_code.upper() in used_table_names:
                                used_selection_codes.append(selection_code)
                        
                        return used_selection_codes
                    
                    # Apply the filtering
                    used_selection_codes = get_used_selection_codes(queries_and_results, top_selection_codes)
                    
                    # Use filtered codes if available, otherwise fallback to all top_selection_codes
                    datasets_used = used_selection_codes if used_selection_codes else top_selection_codes
                    print(f"[API-PostgreSQL] 📊 Found datasets used (filtered): {datasets_used} (from {len(top_selection_codes)} total)")
                elif top_selection_codes:
                    # If no queries yet, use all top_selection_codes
                    datasets_used = top_selection_codes
                    print(f"[API-PostgreSQL] 📊 Found datasets used (unfiltered): {datasets_used}")
                
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

@app.get("/debug/chat/{thread_id}/checkpoints")
async def debug_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Debug endpoint to inspect raw checkpoint data for a thread."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[DEBUG] 🔍 Inspecting checkpoints for thread: {thread_id}")
    
    try:
        checkpointer = await get_healthy_checkpointer()
        
        if not hasattr(checkpointer, 'conn'):
            return {"error": "No PostgreSQL checkpointer available"}
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints for this thread
        checkpoint_tuples = []
        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoint_tuples.append(checkpoint_tuple)
        
        debug_data = {
            "thread_id": thread_id,
            "total_checkpoints": len(checkpoint_tuples),
            "checkpoints": []
        }
        
        for i, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}
            
            checkpoint_info = {
                "index": i,
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id", "unknown"),
                "has_checkpoint": bool(checkpoint),
                "has_metadata": bool(metadata),
                "metadata_writes": metadata.get("writes", {}),
                "channel_values": {}
            }
            
            if checkpoint and "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                messages = channel_values.get("messages", [])
                
                checkpoint_info["channel_values"] = {
                    "message_count": len(messages),
                    "messages": []
                }
                
                for j, msg in enumerate(messages):
                    msg_info = {
                        "index": j,
                        "type": type(msg).__name__,
                        "id": getattr(msg, 'id', None),
                        "content_preview": getattr(msg, 'content', str(msg))[:200] + "..." if hasattr(msg, 'content') and len(getattr(msg, 'content', '')) > 200 else getattr(msg, 'content', str(msg)),
                        "content_length": len(getattr(msg, 'content', ''))
                    }
                    checkpoint_info["channel_values"]["messages"].append(msg_info)
            
            debug_data["checkpoints"].append(checkpoint_info)
        
        return debug_data
        
    except Exception as e:
        print(f"[DEBUG] ❌ Error inspecting checkpoints: {e}")
        return {"error": str(e)}

@app.get("/debug/pool-status")
async def debug_pool_status():
    """Debug endpoint to check pool and checkpointer status (no auth required)."""
    global GLOBAL_CHECKPOINTER
    
    try:
        status = {
            "global_checkpointer_exists": GLOBAL_CHECKPOINTER is not None,
            "checkpointer_type": type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None,
            "has_connection_pool": False,
            "pool_closed": None,
            "pool_healthy": None,
            "can_query": False,
            "timestamp": datetime.now().isoformat()
        }
        
        if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn'):
            status["has_connection_pool"] = True
            
            if GLOBAL_CHECKPOINTER.conn:
                status["pool_closed"] = GLOBAL_CHECKPOINTER.conn.closed
                
                # Test if we can execute a simple query
                try:
                    async with GLOBAL_CHECKPOINTER.conn.connection() as conn:
                        await conn.execute("SELECT 1")
                    status["can_query"] = True
                    status["pool_healthy"] = True
                except Exception as e:
                    status["can_query"] = False
                    status["pool_healthy"] = False
                    status["query_error"] = str(e)
        
        # Try to get a healthy checkpointer
        try:
            healthy_checkpointer = await get_healthy_checkpointer()
            status["healthy_checkpointer_type"] = type(healthy_checkpointer).__name__
            status["healthy_checkpointer_available"] = True
        except Exception as e:
            status["healthy_checkpointer_available"] = False
            status["healthy_checkpointer_error"] = str(e)
        
        return status
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/chat/{thread_id}/run-ids")
async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
    """Get run_ids for messages in a thread to enable feedback submission."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print(f"[API-LangSmith] 📥 Getting run_ids for thread: {thread_id}, user: {user_email}")
    
    try:
        # Get a healthy pool
        pool = await get_healthy_checkpointer()
        pool = pool.conn if hasattr(pool, 'conn') else None
        
        if not pool:
            return {"run_ids": []}
        
        async with pool.connection() as conn:
            # Get all run_ids for this thread and user, ordered by timestamp
            result = await conn.execute("""
                SELECT run_id, prompt, timestamp
                FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
                ORDER BY timestamp ASC
            """, (user_email, thread_id))
            
            run_id_data = []
            async for row in result:
                run_id_data.append({
                    "run_id": row[0],
                    "prompt": row[1],
                    "timestamp": row[2].isoformat()
                })
            
            print(f"[API-LangSmith] ✅ Found {len(run_id_data)} run_ids for thread {thread_id}")
            return {"run_ids": run_id_data}
            
    except Exception as e:
        print(f"[API-LangSmith] ❌ Failed to get run_ids for thread {thread_id}: {e}")
        return {"run_ids": []} 