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
import gc
import time
from datetime import datetime

import psutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse

# Import globals and utilities from config/utils
from api.config.settings import (
    BULK_CACHE_TIMEOUT,
    GLOBAL_CHECKPOINTER,
    RATE_LIMIT_BURST,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    _bulk_loading_cache,
    _request_count,
    rate_limit_storage,
    start_time,
)

# Import memory-related variables from memory.py
from api.utils.memory import cleanup_bulk_cache
from api.helpers import traceback_json_response

# Create router for health endpoints
router = APIRouter()


@router.get("/health")
async def health_check():
    """Enhanced health check with memory monitoring and database verification."""
    try:
        # Memory check
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # Database check with proper AsyncPostgresSaver handling
        database_healthy = True
        database_error = None
        checkpointer_type = "Unknown"

        try:
            if GLOBAL_CHECKPOINTER:
                checkpointer_type = type(GLOBAL_CHECKPOINTER).__name__

                if "AsyncPostgresSaver" in checkpointer_type:
                    # Test AsyncPostgresSaver with a simple operation
                    test_config = {"configurable": {"thread_id": "health_check_test"}}

                    # Use aget_tuple() which is a basic read operation
                    result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
                    # If we get here without exception, the database is healthy
                    database_healthy = True
                else:
                    # For other checkpointer types (like MemorySaver)
                    database_healthy = True

        except Exception as e:
            database_healthy = False
            database_error = str(e)

        # Response
        status = "healthy" if database_healthy else "degraded"

        health_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - start_time,
            "memory": {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(memory_percent, 2),
            },
            "database": {
                "healthy": database_healthy,
                "checkpointer_type": checkpointer_type,
                "error": database_error,
            },
            "version": "1.0.0",
        }

        # Run garbage collection
        collected = gc.collect()
        health_data["garbage_collector"] = {
            "objects_collected": collected,
            "gc_run": True,
        }

        if not database_healthy:
            return JSONResponse(status_code=503, content=health_data)

        return health_data

    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.get("/health/database")
async def database_health_check():
    """Detailed database health check."""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "checkpointer_available": GLOBAL_CHECKPOINTER is not None,
            "checkpointer_type": (
                type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None
            ),
        }

        if GLOBAL_CHECKPOINTER and "AsyncPostgresSaver" in str(
            type(GLOBAL_CHECKPOINTER)
        ):
            # Test AsyncPostgresSaver functionality
            try:
                test_config = {"configurable": {"thread_id": "db_health_test"}}

                # Test basic read operation
                start_time_local = time.time()
                result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
                read_latency = time.time() - start_time_local

                health_status.update(
                    {
                        "database_connection": "healthy",
                        "read_latency_ms": round(read_latency * 1000, 2),
                        "read_test": "passed",
                    }
                )

            except Exception as e:
                health_status.update(
                    {
                        "database_connection": "error",
                        "error": str(e),
                        "read_test": "failed",
                    }
                )
                return JSONResponse(status_code=503, content=health_status)
        else:
            health_status.update(
                {
                    "database_connection": "using_memory_fallback",
                    "note": "PostgreSQL checkpointer not available",
                }
            )

        return health_status

    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return JSONResponse(
            status_code=500,
            content={
                "timestamp": datetime.now().isoformat(),
                "database_connection": "error",
                "error": str(e),
            },
        )


@router.get("/health/memory")
async def memory_health_check():
    """Enhanced memory-specific health check with cache information."""
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024

        # Clean up expired cache entries
        cleaned_entries = cleanup_bulk_cache()

        # Get memory threshold from environment
        gc_memory_threshold = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))

        status = "healthy"
        if rss_mb > gc_memory_threshold:
            status = "high_memory"
        elif rss_mb > (gc_memory_threshold * 0.8):
            status = "warning"

        cache_info = {
            "active_cache_entries": len(_bulk_loading_cache),
            "cleaned_expired_entries": cleaned_entries,
            "cache_timeout_seconds": BULK_CACHE_TIMEOUT,
        }

        # Calculate estimated memory per thread for scaling guidance
        thread_count = len(_bulk_loading_cache)
        memory_per_thread = rss_mb / max(thread_count, 1) if thread_count > 0 else 0
        estimated_max_threads = (
            int(gc_memory_threshold / max(memory_per_thread, 38))
            if memory_per_thread > 0
            else 50
        )

        return {
            "status": status,
            "memory_rss_mb": round(rss_mb, 1),
            "memory_threshold_mb": gc_memory_threshold,
            "memory_usage_percent": round((rss_mb / gc_memory_threshold) * 100, 1),
            "over_threshold": rss_mb > gc_memory_threshold,
            "total_requests_processed": _request_count,
            "cache_info": cache_info,
            "scaling_info": {
                "estimated_memory_per_thread_mb": round(memory_per_thread, 1),
                "estimated_max_threads_at_threshold": estimated_max_threads,
                "current_thread_count": thread_count,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/health/rate-limits")
async def rate_limit_health_check():
    """Rate limiting health check."""
    try:
        total_clients = len(rate_limit_storage)
        active_clients = sum(1 for requests in rate_limit_storage.values() if requests)

        return {
            "status": "healthy",
            "total_tracked_clients": total_clients,
            "active_clients": active_clients,
            "rate_limit_window": RATE_LIMIT_WINDOW,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_burst": RATE_LIMIT_BURST,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/health/prepared-statements")
async def prepared_statements_health_check():
    """Health check for prepared statements and database connection status."""
    try:
        from checkpointer.error_handling.prepared_statements import (
            clear_prepared_statements,
        )
        from checkpointer.checkpointer.factory import get_global_checkpointer

        # Check if we can get a checkpointer
        try:
            checkpointer = await get_global_checkpointer()
            checkpointer_status = "healthy" if checkpointer else "unavailable"
        except Exception as e:
            checkpointer_status = f"error: {str(e)}"

        # Check prepared statements in the database
        try:
            import psycopg

            from checkpointer.config import get_db_config
            from checkpointer.database.connection import get_connection_kwargs

            config = get_db_config()
            # Create connection string without prepared statement parameters
            connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require"

            # Get connection kwargs for disabling prepared statements
            connection_kwargs = get_connection_kwargs()

            async with await psycopg.AsyncConnection.connect(
                connection_string, **connection_kwargs
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT COUNT(*) as count, 
                               STRING_AGG(name, ', ') as statement_names
                        FROM pg_prepared_statements 
                        WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                    """
                    )
                    result = await cur.fetchone()

                    # Fix: Handle psycopg Row object properly - check if it exists and has data
                    prepared_count = result[0] if result else 0
                    statement_names = (
                        result[1]
                        if result and len(result) > 1 and result[1]
                        else "none"
                    )

                    return {
                        "status": "healthy",
                        "checkpointer_status": checkpointer_status,
                        "prepared_statements_count": prepared_count,
                        "prepared_statement_names": statement_names,
                        "connection_kwargs": connection_kwargs,
                        "timestamp": datetime.now().isoformat(),
                    }

        except Exception as db_error:
            return {
                "status": "degraded",
                "checkpointer_status": checkpointer_status,
                "database_error": str(db_error),
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
