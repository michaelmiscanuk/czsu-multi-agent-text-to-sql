# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

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
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

# ============================================================
# RESPONSE MODELS
# ============================================================


class ChatThreadResponse(BaseModel):
    thread_id: str
    latest_timestamp: datetime  # Changed from str to datetime
    run_count: int
    title: str  # Now includes the title from first prompt
    full_prompt: str  # Full prompt text for tooltip


class PaginatedChatThreadsResponse(BaseModel):
    threads: List[ChatThreadResponse]
    total_count: int
    page: int
    limit: int
    has_more: bool


class ChatMessage(BaseModel):
    id: str
    threadId: str
    user: str
    createdAt: int
    prompt: Optional[str] = None
    final_answer: Optional[str] = None
    queries_and_results: Optional[List[List[str]]] = None
    datasets_used: Optional[List[str]] = None
    top_chunks: Optional[List[dict]] = None
    sql_query: Optional[str] = None
    error: Optional[str] = None
    isLoading: Optional[bool] = None
    startedAt: Optional[int] = None
    isError: Optional[bool] = None
    run_id: Optional[str] = None
