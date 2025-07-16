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

import traceback

# Standard imports
import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import models
from api.models.responses import ChatMessage

# Import debug functions
from api.utils.debug import (
    print__api_postgresql,
    print__chat_messages_debug,
    print__feedback_flow,
)

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
from api.helpers import traceback_json_response
from api.routes.chat import get_thread_messages_with_metadata
from my_agent.utils.postgres_checkpointer import get_healthy_checkpointer

# Create router for message endpoints
router = APIRouter()


@router.get("/chat/{thread_id}/messages")
async def get_chat_messages(
    thread_id: str, user=Depends(get_current_user)
) -> List[ChatMessage]:
    """Load conversation messages from PostgreSQL checkpoint history that preserves original user messages."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__api_postgresql(
        f"📥 Loading checkpoint messages for thread {thread_id}, user: {user_email}"
    )

    try:
        # Get healthy checkpointer
        checkpointer = await get_healthy_checkpointer()

        if not hasattr(checkpointer, "conn"):
            print__api_postgresql(
                f"⚠️ No PostgreSQL checkpointer available - returning empty messages"
            )
            return []

        # Use the consolidated function that handles all checkpoint processing and metadata extraction
        print__api_postgresql(
            f"🔍 Using consolidated get_thread_messages_with_metadata function"
        )
        chat_messages = await get_thread_messages_with_metadata(
            checkpointer, thread_id, user_email, "checkpoint_history"
        )

        if not chat_messages:
            print__api_postgresql(f"⚠ No messages found for thread {thread_id}")
            return []

        print__api_postgresql(
            f"✅ Converted {len(chat_messages)} messages to frontend format"
        )

        # Log the messages for debugging
        for i, msg in enumerate(chat_messages):
            user_type = "👤 User" if msg.isUser else "🤖 AI"
            content_preview = (
                msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            )
            datasets_info = (
                f" (datasets: {msg.meta.get('datasetsUsed', [])})"
                if msg.meta and msg.meta.get("datasetsUsed")
                else ""
            )
            sql_info = (
                f" (SQL: {msg.meta.get('sqlQuery', 'None')[:30]}...)"
                if msg.meta and msg.meta.get("sqlQuery")
                else ""
            )
            print__api_postgresql(
                f"{i+1}. {user_type}: {content_preview}{datasets_info}{sql_info}"
            )

        return chat_messages

    except Exception as e:
        error_msg = str(e)
        print__api_postgresql(
            f"❌ Failed to load checkpoint messages for thread {thread_id}: {e}"
        )

        # Handle specific database connection errors gracefully
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
            print__api_postgresql(
                f"⚠ Database connection error - returning empty messages"
            )
            return []
        else:
            resp = traceback_json_response(e)
            if resp:
                return resp
            raise HTTPException(
                status_code=500, detail=f"Failed to load checkpoint messages: {e}"
            )


@router.get("/chat/{thread_id}/run-ids")
async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
    """Get run_ids for messages in a thread to enable feedback submission."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__feedback_flow(f"🔍 Fetching run_ids for thread {thread_id}")

    try:
        pool = await get_healthy_checkpointer()
        pool = pool.conn if hasattr(pool, "conn") else None

        if not pool:
            print__feedback_flow("⚠ No pool available for run_id lookup")
            return {"run_ids": []}

        async with pool.connection() as conn:
            print__feedback_flow(f"📊 Executing SQL query for thread {thread_id}")
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT run_id, prompt, timestamp
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                    ORDER BY timestamp ASC
                """,
                    (user_email, thread_id),
                )

                run_id_data = []
                rows = await cur.fetchall()
                for row in rows:
                    print__feedback_flow(
                        f"📝 Processing database row - run_id: {row[0]}, prompt: {row[1][:50]}..."
                    )
                    try:
                        run_uuid = str(uuid.UUID(row[0])) if row[0] else None
                        if run_uuid:
                            run_id_data.append(
                                {
                                    "run_id": run_uuid,
                                    "prompt": row[1],
                                    "timestamp": row[2].isoformat(),
                                }
                            )
                            print__feedback_flow(f"✅ Valid UUID found: {run_uuid}")
                        else:
                            print__feedback_flow(
                                f"⚠ Null run_id found for prompt: {row[1][:50]}..."
                            )
                    except ValueError:
                        print__feedback_flow(f"❌ Invalid UUID in database: {row[0]}")
                        continue

                print__feedback_flow(
                    f"📊 Total valid run_ids found: {len(run_id_data)}"
                )
                return {"run_ids": run_id_data}

    except Exception as e:
        print__feedback_flow(f"🚨 Error fetching run_ids: {str(e)}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {"run_ids": []}
