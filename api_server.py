from fastapi import FastAPI, Request, Query, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from main import main as analysis_main
import sqlite3
from typing import List, Optional
import requests
import jwt
import os
import time
from jwt.algorithms import RSAAlgorithm
from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import json

app = FastAPI()

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
    result = await analysis_main(request.prompt)
    return result 

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
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
    if q:
        q_lower = q.lower()
        tables = [t for t in tables if q_lower in t.lower()]
    return {"tables": tables}

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

@app.get('/chat-sessions')
def get_chat_sessions(user=Depends(get_current_user)):
    user_id = user['sub']
    db_path = 'data/chat_sessions.db'
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS chat_sessions (user_id TEXT, session_id TEXT, data TEXT)')
        cursor.execute('SELECT session_id, data FROM chat_sessions WHERE user_id = ?', (user_id,))
        sessions = [{'id': row[0], 'data': json.loads(row[1])} for row in cursor.fetchall()]
    return sessions

@app.post('/chat-sessions')
def save_chat_session(session: dict, user=Depends(get_current_user)):
    user_id = user['sub']
    session_id = session['id']
    data = json.dumps(session)
    db_path = 'data/chat_sessions.db'
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS chat_sessions (user_id TEXT, session_id TEXT, data TEXT)')
        cursor.execute('REPLACE INTO chat_sessions (user_id, session_id, data) VALUES (?, ?, ?)', (user_id, session_id, data))
        conn.commit()
    return {'status': 'ok'} 