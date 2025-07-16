# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import asyncio
import os
import sys
import traceback
import uuid

# Load environment variables early
from dotenv import load_dotenv

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

# Import configuration and globals
from api.config.settings import (
    INMEMORY_FALLBACK_ENABLED,
    MAX_CONCURRENT_ANALYSES,
    analysis_semaphore,
)

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import models
from api.models.requests import AnalyzeRequest

# Import debug functions
from api.utils.debug import (
    print__analysis_tracing_debug,
    print__analyze_debug,
    print__feedback_flow,
)

# Import memory utilities
from api.utils.memory import log_memory_usage

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
# NEW: Import for calling the single-thread endpoint
import httpx

from api.helpers import traceback_json_response
from main import main as analysis_main
from my_agent.utils.postgres_checkpointer import (
    create_thread_run_entry,
    get_healthy_checkpointer,
)

# Load environment variables
load_dotenv()

# Create router for analysis endpoints
router = APIRouter()


async def get_thread_metadata_from_single_thread_endpoint(
    thread_id: str, user_email: str
) -> dict:
    """Call the single-thread endpoint to get metadata for a specific thread."""
    try:
        print__analyze_debug(
            f"🔍 Calling single-thread endpoint for thread: {thread_id}"
        )
        print__analysis_tracing_debug(
            f"METADATA EXTRACTION: Calling /chat/all-messages-for-one-thread/{thread_id}"
        )

        # Import the single-thread function directly to avoid HTTP overhead
        from unittest.mock import Mock

        from fastapi import Request

        from api.routes.chat import get_all_chat_messages_for_one_thread

        # Create a mock user object
        mock_user = {"email": user_email}

        # Call the function directly instead of making HTTP request
        print__analyze_debug(f"🔍 Calling single-thread function directly")
        result = await get_all_chat_messages_for_one_thread(thread_id, mock_user)

        if isinstance(result, dict):
            # If it's a direct dict response (from cache hit or direct return)
            response_data = result
        else:
            # If it's a JSONResponse object, extract the content
            if hasattr(result, "body"):
                import json

                response_data = json.loads(result.body.decode())
            else:
                print__analyze_debug(
                    f"⚠️ Unexpected response type from single-thread endpoint: {type(result)}"
                )
                return {}

        print__analyze_debug(
            f"🔍 Single-thread endpoint returned {len(response_data.get('messages', []))} messages"
        )

        # Extract metadata from the latest AI message
        messages = response_data.get("messages", [])
        run_ids = response_data.get("runIds", [])
        sentiments = response_data.get("sentiments", {})

        # Find the latest AI message that has metadata
        latest_ai_message = None
        for message in reversed(messages):
            if not message.get("isUser", True) and message.get("meta"):
                latest_ai_message = message
                break

        metadata = {}
        if latest_ai_message and latest_ai_message.get("meta"):
            meta = latest_ai_message["meta"]
            metadata.update(
                {
                    "top_selection_codes": meta.get("datasetsUsed", []),
                    "datasets_used": meta.get("datasetsUsed", []),
                    "queries_and_results": latest_ai_message.get(
                        "queriesAndResults", []
                    ),
                    "sql": meta.get("sqlQuery"),
                    "dataset_url": meta.get("datasetUrl"),
                    "top_chunks": meta.get("topChunks", []),
                }
            )
            print__analyze_debug(
                f"🔍 Extracted metadata: top_selection_codes={len(metadata.get('top_selection_codes', []))}"
            )
        else:
            print__analyze_debug(
                f"⚠️ No AI message with metadata found in single-thread response"
            )
            # Return empty metadata if no AI message found
            metadata = {
                "top_selection_codes": [],
                "datasets_used": [],
                "queries_and_results": [],
                "sql": None,
                "dataset_url": None,
                "top_chunks": [],
            }

        return metadata

    except Exception as e:
        print__analyze_debug(f"🚨 Error calling single-thread endpoint: {e}")
        print__analysis_tracing_debug(f"METADATA EXTRACTION ERROR: {e}")

        resp = traceback_json_response(e)
        if resp:
            return resp

        # Return empty metadata on error
        return {
            "top_selection_codes": [],
            "datasets_used": [],
            "queries_and_results": [],
            "sql": None,
            "dataset_url": None,
            "top_chunks": [],
        }


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """Analyze request with simplified memory monitoring."""

    print__analysis_tracing_debug("01 - ANALYZE ENDPOINT ENTRY: Request received")
    print__analyze_debug("🔍 ANALYZE ENDPOINT - ENTRY POINT")
    print__analyze_debug(f"🔍 Request received: thread_id={request.thread_id}")
    print__analyze_debug(f"🔍 Request prompt length: {len(request.prompt)}")

    try:
        print__analysis_tracing_debug(
            "02 - USER EXTRACTION: Getting user email from token"
        )
        user_email = user.get("email")
        print__analyze_debug(f"🔍 User extraction: {user_email}")
        if not user_email:
            print__analysis_tracing_debug("03 - ERROR: No user email found in token")
            print__analyze_debug("🚨 No user email found in token")
            raise HTTPException(status_code=401, detail="User email not found in token")

        print__analysis_tracing_debug(
            f"04 - USER VALIDATION SUCCESS: User {user_email} validated"
        )
        print__feedback_flow(
            f"📝 New analysis request - Thread: {request.thread_id}, User: {user_email}"
        )
        print__analyze_debug(
            f"🔍 ANALYZE REQUEST RECEIVED: thread_id={request.thread_id}, user={user_email}"
        )

        print__analysis_tracing_debug("05 - MEMORY MONITORING: Starting memory logging")
        # Simple memory check
        print__analyze_debug("🔍 Starting memory logging")
        log_memory_usage("analysis_start")
        run_id = None

        print__analysis_tracing_debug(
            "06 - SEMAPHORE ACQUISITION: Attempting to acquire analysis semaphore"
        )
        print__analyze_debug("🔍 About to acquire analysis semaphore")
        # Limit concurrent analyses to prevent resource exhaustion
        async with analysis_semaphore:
            print__analysis_tracing_debug(
                "07 - SEMAPHORE ACQUIRED: Analysis semaphore acquired"
            )
            print__feedback_flow("🔒 Acquired analysis semaphore")
            print__analyze_debug("🔍 Semaphore acquired successfully")

            try:
                print__analysis_tracing_debug(
                    "08 - CHECKPOINTER INITIALIZATION: Getting healthy checkpointer"
                )
                print__analyze_debug("🔍 About to get healthy checkpointer")
                print__feedback_flow("🔄 Getting healthy checkpointer")
                checkpointer = await get_healthy_checkpointer()
                print__analysis_tracing_debug(
                    f"09 - CHECKPOINTER SUCCESS: Checkpointer obtained ({type(checkpointer).__name__})"
                )
                print__analyze_debug(
                    f"🔍 Checkpointer obtained: {type(checkpointer).__name__}"
                )

                print__analysis_tracing_debug(
                    "10 - THREAD RUN ENTRY: Creating thread run entry in database"
                )
                print__analyze_debug("🔍 About to create thread run entry")
                print__feedback_flow("🔄 Creating thread run entry")
                run_id = await create_thread_run_entry(
                    user_email, request.thread_id, request.prompt
                )
                print__analysis_tracing_debug(
                    f"11 - THREAD RUN SUCCESS: Thread run entry created with run_id {run_id}"
                )
                print__feedback_flow(f"✅ Generated new run_id: {run_id}")
                print__analyze_debug(
                    f"🔍 Thread run entry created successfully: {run_id}"
                )

                print__analysis_tracing_debug(
                    "12 - ANALYSIS MAIN START: Starting analysis_main function"
                )
                print__analyze_debug("🔍 About to start analysis_main")
                print__feedback_flow("🚀 Starting analysis")
                # 8 minute timeout for platform stability
                result = await asyncio.wait_for(
                    analysis_main(
                        request.prompt,
                        thread_id=request.thread_id,
                        checkpointer=checkpointer,
                        run_id=run_id,
                    ),
                    timeout=480,  # 8 minutes timeout
                )

                print__analysis_tracing_debug(
                    "13 - ANALYSIS MAIN SUCCESS: Analysis completed successfully"
                )
                print__analyze_debug("🔍 Analysis completed successfully")
                print__feedback_flow("✅ Analysis completed successfully")

            except Exception as analysis_error:
                print__analysis_tracing_debug(
                    f"14 - ANALYSIS ERROR: Exception in analysis block - {type(analysis_error).__name__}"
                )
                print__analyze_debug(
                    f"🚨 Exception in analysis block: {type(analysis_error).__name__}: {str(analysis_error)}"
                )
                # If there's a database connection issue, try with InMemorySaver as fallback
                error_msg = str(analysis_error).lower()
                print__analyze_debug(f"🔍 Error message (lowercase): {error_msg}")

                # ENHANCED: Check for prepared statement errors specifically
                is_prepared_stmt_error = any(
                    indicator in error_msg
                    for indicator in [
                        "prepared statement",
                        "does not exist",
                        "_pg3_",
                        "_pg_",
                        "invalidsqlstatementname",
                    ]
                )

                if is_prepared_stmt_error:
                    print__analysis_tracing_debug(
                        "15 - PREPARED STATEMENT ERROR: Prepared statement error detected"
                    )
                    print__analyze_debug(
                        f"🔧 PREPARED STATEMENT ERROR DETECTED: {analysis_error}"
                    )
                    print__feedback_flow(
                        f"🔧 Prepared statement error detected - this should be handled by retry logic: {analysis_error}"
                    )

                    resp = traceback_json_response(analysis_error)
                    if resp:
                        return resp

                    # Re-raise prepared statement errors - they should be handled by the retry decorator
                    raise HTTPException(
                        status_code=500,
                        detail="Database prepared statement error. Please try again. {analysis_error}",
                    ) from analysis_error
                elif any(
                    keyword in error_msg
                    for keyword in [
                        "pool",
                        "connection",
                        "closed",
                        "timeout",
                        "ssl",
                        "postgres",
                    ]
                ):
                    print__analysis_tracing_debug(
                        "16 - DATABASE FALLBACK: Database issue detected, attempting fallback"
                    )
                    print__analyze_debug(
                        "🔍 Database issue detected, attempting fallback"
                    )
                    print__feedback_flow(
                        f"⚠️ Database issue detected, trying with InMemorySaver fallback: {analysis_error}"
                    )

                    # Check if InMemorySaver fallback is enabled
                    if not INMEMORY_FALLBACK_ENABLED:
                        print__analysis_tracing_debug(
                            "17 - FALLBACK DISABLED: InMemorySaver fallback is disabled by configuration"
                        )
                        print__analyze_debug(
                            f"🚫 InMemorySaver fallback is DISABLED by configuration - re-raising database error"
                        )
                        print__feedback_flow(
                            f"🚫 InMemorySaver fallback disabled - propagating database error: {analysis_error}"
                        )
                        resp = traceback_json_response(analysis_error)
                        if resp:
                            return resp

                        raise HTTPException(
                            status_code=500,
                            detail="Database connection error. Please try again. {analysis_error}",
                        ) from analysis_error

                    try:
                        print__analysis_tracing_debug(
                            "17 - FALLBACK INITIALIZATION: Importing InMemorySaver"
                        )
                        print__analyze_debug(f"🔍 Importing InMemorySaver")
                        from langgraph.checkpoint.memory import InMemorySaver

                        fallback_checkpointer = InMemorySaver()
                        print__analysis_tracing_debug(
                            "18 - FALLBACK CHECKPOINTER: InMemorySaver created"
                        )
                        print__analyze_debug(f"🔍 InMemorySaver created")

                        # Generate a fallback run_id since database creation might have failed
                        if run_id is None:
                            run_id = str(uuid.uuid4())
                            print__analysis_tracing_debug(
                                f"19 - FALLBACK RUN ID: Generated fallback run_id {run_id}"
                            )
                            print__feedback_flow(
                                f"✅ Generated fallback run_id: {run_id}"
                            )
                            print__analyze_debug(
                                f"🔍 Generated fallback run_id: {run_id}"
                            )

                        print__analysis_tracing_debug(
                            "20 - FALLBACK ANALYSIS: Starting fallback analysis"
                        )
                        print__analyze_debug(f"🔍 Starting fallback analysis")
                        print__feedback_flow(
                            f"🚀 Starting analysis with InMemorySaver fallback"
                        )
                        result = await asyncio.wait_for(
                            analysis_main(
                                request.prompt,
                                thread_id=request.thread_id,
                                checkpointer=fallback_checkpointer,
                                run_id=run_id,
                            ),
                            timeout=480,  # 8 minutes timeout
                        )

                        print__analysis_tracing_debug(
                            "21 - FALLBACK SUCCESS: Fallback analysis completed successfully"
                        )
                        print__analyze_debug(
                            f"🔍 Fallback analysis completed successfully"
                        )
                        print__feedback_flow(
                            f"✅ Analysis completed successfully with fallback"
                        )

                    except Exception as fallback_error:
                        print__analysis_tracing_debug(
                            f"22 - FALLBACK FAILED: Fallback also failed - {type(fallback_error).__name__}"
                        )
                        print__analyze_debug(
                            f"🚨 Fallback also failed: {type(fallback_error).__name__}: {str(fallback_error)}"
                        )
                        print__feedback_flow(
                            f"🚨 Fallback analysis also failed: {fallback_error}"
                        )
                        resp = traceback_json_response(fallback_error)
                        if resp:
                            return resp

                        raise HTTPException(
                            status_code=500,
                            detail="Sorry, there was an error processing your request. Please try again.",
                        )
                else:
                    # Re-raise non-database errors
                    print__analysis_tracing_debug(
                        f"23 - NON-DATABASE ERROR: Non-database error - {type(analysis_error).__name__}"
                    )
                    print__analyze_debug(
                        f"🚨 Non-database error, re-raising: {type(analysis_error).__name__}: {str(analysis_error)}"
                    )
                    print__feedback_flow(f"🚨 Non-database error: {analysis_error}")

                    resp = traceback_json_response(analysis_error)
                    if resp:
                        return resp

                    raise HTTPException(
                        status_code=500,
                        detail="Sorry, there was an error processing your request. Please try again.",
                    )

            print__analysis_tracing_debug(
                "24 - RESPONSE PREPARATION: Preparing response data"
            )
            print__analyze_debug(f"🔍 About to prepare response data")

            # NEW: Get metadata from single-thread endpoint instead of analysis result
            print__analysis_tracing_debug(
                "24a - METADATA EXTRACTION: Getting metadata from single-thread endpoint"
            )
            print__analyze_debug(
                f"🔍 Getting metadata from single-thread endpoint for thread: {request.thread_id}"
            )
            thread_metadata = await get_thread_metadata_from_single_thread_endpoint(
                request.thread_id, user_email
            )
            print__analyze_debug(
                f"🔍 Retrieved metadata from single-thread endpoint: {list(thread_metadata.keys())}"
            )

            # Simple response preparation with metadata from single-thread endpoint
            response_data = {
                "prompt": request.prompt,
                "result": (
                    result["result"]
                    if isinstance(result, dict) and "result" in result
                    else str(result)
                ),
                "queries_and_results": thread_metadata.get("queries_and_results", []),
                "thread_id": request.thread_id,
                "top_selection_codes": thread_metadata.get("top_selection_codes", []),
                "datasets_used": thread_metadata.get("datasets_used", []),
                "iteration": (
                    result.get("iteration", 0) if isinstance(result, dict) else 0
                ),
                "max_iterations": (
                    result.get("max_iterations", 2) if isinstance(result, dict) else 2
                ),
                "sql": thread_metadata.get("sql", None),
                "datasetUrl": thread_metadata.get("dataset_url", None),
                "run_id": run_id,
                "top_chunks": thread_metadata.get("top_chunks", []),
            }

            # DEBUG: Log what was extracted for metadata from single-thread endpoint
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: datasets_used extracted from single-thread: {response_data['datasets_used']}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: top_selection_codes extracted from single-thread: {response_data['top_selection_codes']}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: queries_and_results count: {len(response_data['queries_and_results'])}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: sql query available: {'Yes' if response_data['sql'] else 'No'}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: top_chunks count: {len(response_data['top_chunks'])}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: datasetUrl: {response_data['datasetUrl']}"
            )

            print__analysis_tracing_debug(
                f"25 - RESPONSE SUCCESS: Response data prepared with {len(response_data.keys())} keys"
            )
            print__analyze_debug(f"🔍 Response data prepared successfully")
            print__analyze_debug(f"🔍 Response data keys: {list(response_data.keys())}")
            print__analyze_debug(f"🔍 ANALYZE ENDPOINT - SUCCESSFUL EXIT")
            return response_data

    except asyncio.TimeoutError:
        error_msg = "Analysis timed out after 8 minutes"
        print__analysis_tracing_debug(
            "26 - TIMEOUT ERROR: Analysis timed out after 8 minutes"
        )
        print__analyze_debug(f"🚨 TIMEOUT ERROR: {error_msg}")
        print__feedback_flow(f"🚨 {error_msg}")

        resp = traceback_json_response(asyncio.TimeoutError)
        if resp:
            return resp

        raise HTTPException(status_code=408, detail=error_msg)

    except HTTPException as http_exc:
        print__analysis_tracing_debug(
            f"27 - HTTP EXCEPTION: HTTP exception {http_exc.status_code}"
        )
        print__analyze_debug(
            f"🚨 HTTP EXCEPTION: {http_exc.status_code} - {http_exc.detail}"
        )
        resp = traceback_json_response(http_exc)
        if resp:
            return resp

        raise http_exc

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print__analysis_tracing_debug(
            f"28 - UNEXPECTED EXCEPTION: Unexpected exception - {type(e).__name__}"
        )
        print__analyze_debug(f"🚨 UNEXPECTED EXCEPTION: {type(e).__name__}: {str(e)}")
        print__analyze_debug(f"🚨 Exception traceback: {traceback.format_exc()}")
        print__feedback_flow(f"🚨 {error_msg}")
        resp = traceback_json_response(e)
        if resp:
            return resp

        raise HTTPException(
            status_code=500,
            detail="Sorry, there was an error processing your request. Please try again.",
        )
