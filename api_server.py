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

from main import main as analysis_main
from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer

# Global shared checkpointer for conversation memory across API requests
# This ensures that conversation state is preserved between frontend requests using PostgreSQL
GLOBAL_CHECKPOINTER = None

async def initialize_checkpointer():
    """Initialize the global PostgreSQL checkpointer on startup."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER is None:
        try:
            GLOBAL_CHECKPOINTER = await get_postgres_checkpointer()
            print("✓ Global PostgreSQL checkpointer initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize PostgreSQL checkpointer: {e}")
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print("⚠ Falling back to InMemorySaver")

async def cleanup_checkpointer():
    """Clean up resources on app shutdown."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'pool') and GLOBAL_CHECKPOINTER.pool:
        try:
            await GLOBAL_CHECKPOINTER.pool.close()
            print("✓ PostgreSQL connection pool closed cleanly")
        except Exception as e:
            print(f"⚠ Error closing connection pool: {e}")

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance, recreating if necessary."""
    global GLOBAL_CHECKPOINTER
    
    # Check if current checkpointer is healthy
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'pool'):
        try:
            # Quick health check
            async with GLOBAL_CHECKPOINTER.pool.connection() as conn:
                await conn.execute("SELECT 1")
            return GLOBAL_CHECKPOINTER
        except Exception as e:
            print(f"⚠ Checkpointer unhealthy, recreating: {e}")
            # Try to cleanup old pool
            try:
                if GLOBAL_CHECKPOINTER.pool:
                    await GLOBAL_CHECKPOINTER.pool.close()
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
    
    # Get a healthy checkpointer
    checkpointer = await get_healthy_checkpointer()
    
    try:
        result = await analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=checkpointer)
        return result
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific database connection errors
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print(f"⚠ Database connection error detected: {e}")
            print("⚠ Attempting to use fresh InMemorySaver...")
            
            # Use a completely fresh InMemorySaver for this request
            from langgraph.checkpoint.memory import InMemorySaver
            fallback_checkpointer = InMemorySaver()
            
            try:
                result = await analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=fallback_checkpointer)
                # Add warning to the result
                if isinstance(result, dict):
                    result["warning"] = "Persistent memory temporarily unavailable - using session-only memory"
                return result
            except Exception as fallback_error:
                print(f"✗ Fallback also failed: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {fallback_error}")
        else:
            # Re-raise non-connection errors
            print(f"✗ Analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

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