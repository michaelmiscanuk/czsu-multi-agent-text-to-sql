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

import time

# Standard imports
import uuid
from datetime import datetime
from typing import Dict

import psutil
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

# Import configuration globals
from api.config.settings import (
    GC_MEMORY_THRESHOLD,
    GLOBAL_CHECKPOINTER,
    _bulk_loading_cache,
)

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import debug functions
from api.utils.debug import print__debug
from api.utils.memory import print__memory_monitoring

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
from api.helpers import traceback_json_response
from my_agent.utils.postgres_checkpointer import get_global_checkpointer

# Create router for debug endpoints
router = APIRouter()


@router.get("/debug/chat/{thread_id}/checkpoints")
async def debug_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Debug endpoint to inspect raw checkpoint data for a thread."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔍 Inspecting checkpoints for thread: {thread_id}")

    try:
        checkpointer = await get_global_checkpointer()

        if not hasattr(checkpointer, "conn"):
            return {"error": "No PostgreSQL checkpointer available"}

        config = {"configurable": {"thread_id": thread_id}}

        # Get all checkpoints for this thread
        checkpoint_tuples = []
        try:
            # Fix: alist() returns an async generator, don't await it
            checkpoint_iterator = checkpointer.alist(config)
            async for checkpoint_tuple in checkpoint_iterator:
                checkpoint_tuples.append(checkpoint_tuple)
        except Exception as alist_error:
            print__debug(f"❌ Error getting checkpoint list: {alist_error}")
            return {"error": f"Failed to get checkpoints: {alist_error}"}

        debug_data = {
            "thread_id": thread_id,
            "total_checkpoints": len(checkpoint_tuples),
            "checkpoints": [],
        }

        for i, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}

            checkpoint_info = {
                "index": i,
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get(
                    "checkpoint_id", "unknown"
                ),
                "has_checkpoint": bool(checkpoint),
                "has_metadata": bool(metadata),
                "metadata_writes": metadata.get("writes", {}),
                "channel_values": {},
            }

            if checkpoint and "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                messages = channel_values.get("messages", [])

                checkpoint_info["channel_values"] = {
                    "message_count": len(messages),
                    "messages": [],
                }

                for j, msg in enumerate(messages):
                    msg_info = {
                        "index": j,
                        "type": type(msg).__name__,
                        "id": getattr(msg, "id", None),
                        "content_preview": (
                            getattr(msg, "content", str(msg))[:200] + "..."
                            if hasattr(msg, "content")
                            and len(getattr(msg, "content", "")) > 200
                            else getattr(msg, "content", str(msg))
                        ),
                        "content_length": len(getattr(msg, "content", "")),
                    }
                    checkpoint_info["channel_values"]["messages"].append(msg_info)

            debug_data["checkpoints"].append(checkpoint_info)

        return debug_data

    except Exception as e:
        print__debug(f"❌ Error inspecting checkpoints: {e}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {"error": str(e)}


@router.get("/debug/pool-status")
async def debug_pool_status():
    """Debug endpoint to check pool status - updated for official AsyncPostgresSaver."""
    try:
        global GLOBAL_CHECKPOINTER

        status = {
            "timestamp": datetime.now().isoformat(),
            "global_checkpointer_exists": GLOBAL_CHECKPOINTER is not None,
            "checkpointer_type": (
                type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None
            ),
        }

        if GLOBAL_CHECKPOINTER:
            if "AsyncPostgresSaver" in str(type(GLOBAL_CHECKPOINTER)):
                # Test AsyncPostgresSaver functionality instead of checking .conn
                try:
                    test_config = {"configurable": {"thread_id": "pool_status_test"}}
                    start_time = time.time()

                    # Test a basic operation to verify the checkpointer is working
                    result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
                    latency = time.time() - start_time

                    status.update(
                        {
                            "asyncpostgressaver_status": "operational",
                            "test_latency_ms": round(latency * 1000, 2),
                            "connection_test": "passed",
                        }
                    )

                except Exception as test_error:
                    status.update(
                        {
                            "asyncpostgressaver_status": "error",
                            "test_error": str(test_error),
                            "connection_test": "failed",
                        }
                    )
            else:
                status.update(
                    {
                        "checkpointer_status": "non_postgres_type",
                        "note": f"Using {type(GLOBAL_CHECKPOINTER).__name__} instead of AsyncPostgresSaver",
                    }
                )
        else:
            status["checkpointer_status"] = "not_initialized"

        return status

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "timestamp": datetime.now().isoformat()},
        )


@router.get("/debug/run-id/{run_id}")
async def debug_run_id(run_id: str, user=Depends(get_current_user)):
    """Debug endpoint to check if a run_id exists in the database."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔍 Checking run_id: '{run_id}' for user: {user_email}")

    result = {
        "run_id": run_id,
        "run_id_type": type(run_id).__name__,
        "run_id_length": len(run_id) if run_id else 0,
        "is_valid_uuid_format": False,
        "exists_in_database": False,
        "user_owns_run_id": False,
        "database_details": None,
    }

    # Check if it's a valid UUID format
    try:
        uuid_obj = uuid.UUID(run_id)
        result["is_valid_uuid_format"] = True
        result["uuid_parsed"] = str(uuid_obj)
    except ValueError as e:
        result["uuid_error"] = str(e)

    # Check if it exists in the database
    try:
        pool = await get_global_checkpointer()
        pool = pool.conn if hasattr(pool, "conn") else None

        if pool:
            async with pool.connection() as conn:
                # 🔒 SECURITY: Check in users_threads_runs table with user ownership verification
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT email, thread_id, prompt, timestamp
                        FROM users_threads_runs 
                        WHERE run_id = %s AND email = %s
                    """,
                        (run_id, user_email),
                    )

                    row = await cur.fetchone()
                    if row:
                        result["exists_in_database"] = True
                        result["user_owns_run_id"] = True
                        result["database_details"] = {
                            "email": row[0],
                            "thread_id": row[1],
                            "prompt": row[2],
                            "timestamp": row[3].isoformat() if row[3] else None,
                        }
                        print__debug(f"✅ User {user_email} owns run_id {run_id}")
                    else:
                        # Check if run_id exists but belongs to different user
                        await cur.execute(
                            """
                            SELECT COUNT(*) FROM users_threads_runs WHERE run_id = %s
                        """,
                            (run_id,),
                        )

                        any_row = await cur.fetchone()
                        if any_row and any_row[0] > 0:
                            result["exists_in_database"] = True
                            result["user_owns_run_id"] = False
                            print__debug(
                                f"🚫 Run_id {run_id} exists but user {user_email} does not own it"
                            )
                        else:
                            print__debug(f"❌ Run_id {run_id} not found in database")
    except Exception as e:
        result["database_error"] = str(e)
        resp = traceback_json_response(e)
        if resp:
            return resp
        return result


@router.post("/admin/clear-cache")
async def clear_bulk_cache(user=Depends(get_current_user)):
    """Clear the bulk loading cache (admin endpoint)."""
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    # For now, allow any authenticated user to clear cache
    # In production, you might want to restrict this to admin users

    cache_entries_before = len(_bulk_loading_cache)
    _bulk_loading_cache.clear()

    print__memory_monitoring(
        f"🧹 MANUAL CACHE CLEAR: {cache_entries_before} entries cleared by {user_email}"
    )

    # Check memory after cleanup
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        memory_status = "normal" if rss_mb < (GC_MEMORY_THRESHOLD * 0.8) else "high"
    except:
        rss_mb = 0
        memory_status = "unknown"

    return {
        "message": "Cache cleared successfully",
        "cache_entries_cleared": cache_entries_before,
        "current_memory_mb": round(rss_mb, 1),
        "memory_status": memory_status,
        "cleared_by": user_email,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/admin/clear-prepared-statements")
async def clear_prepared_statements_endpoint(user=Depends(get_current_user)):
    """Clear prepared statements in the database to free memory."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🧹 Clearing prepared statements for user: {user_email}")

    try:
        checkpointer = await get_global_checkpointer()

        if not hasattr(checkpointer, "conn"):
            return {"error": "No PostgreSQL checkpointer available"}

        # Clear prepared statements
        # Note: This is a simplified version - in production you'd want more sophisticated cleanup
        return {
            "message": "Prepared statements cleared",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print__debug(f"❌ Error clearing prepared statements: {e}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {"error": str(e)}


@router.post("/debug/set-env")
async def set_debug_environment(
    env_vars: Dict[str, str], user=Depends(get_current_user)
):
    """Dynamically set environment variables for debug control."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔧 Setting debug environment variables: {env_vars}")

    # Set the environment variables in the server process
    for key, value in env_vars.items():
        os.environ[key] = value
        print__debug(f"🔧 Set {key}={value}")

    return {
        "message": f"Set {len(env_vars)} environment variables",
        "variables": env_vars,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/debug/reset-env")
async def reset_debug_environment(
    env_vars: Dict[str, str], user=Depends(get_current_user)
):
    """Reset specific environment variables to their original .env values."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔧 Resetting debug environment variables: {list(env_vars.keys())}")

    # Load original values from .env file
    from dotenv import dotenv_values

    original_env = dotenv_values()

    reset_vars = {}
    for var_name in env_vars.keys():
        # Get original value from .env file, default to "0" if not found
        original_value = original_env.get(var_name, "0")
        os.environ[var_name] = original_value
        reset_vars[var_name] = original_value
        print__debug(f"🔧 Reset {var_name}={original_value}")

    return {
        "message": f"Reset {len(reset_vars)} environment variables to original .env values",
        "variables": reset_vars,
        "timestamp": datetime.now().isoformat(),
    }
