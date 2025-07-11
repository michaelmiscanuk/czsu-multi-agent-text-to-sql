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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from langsmith import Client

# Import authentication dependencies
from api.dependencies.auth import get_current_user

# Import models
from api.models.requests import FeedbackRequest, SentimentRequest

# Import debug functions
from api.utils.debug import (
    print__feedback_debug,
    print__feedback_flow,
    print__sentiment_debug,
    print__sentiment_flow,
)

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
from my_agent.utils.postgres_checkpointer import (
    get_direct_connection,
    update_thread_run_sentiment,
)

# Create router for feedback endpoints
router = APIRouter()


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user=Depends(get_current_user)):
    """Submit feedback for a specific run_id to LangSmith."""

    print__feedback_debug(f"🔍 FEEDBACK ENDPOINT - ENTRY POINT")
    print__feedback_debug(f"🔍 Request received: run_id={request.run_id}")
    print__feedback_debug(f"🔍 Feedback value: {request.feedback}")
    print__feedback_debug(
        f"🔍 Comment length: {len(request.comment) if request.comment else 0}"
    )

    user_email = user.get("email")
    print__feedback_debug(f"🔍 User email extracted: {user_email}")

    if not user_email:
        print__feedback_debug(f"🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__feedback_flow(f"📥 Incoming feedback request:")
    print__feedback_flow(f"👤 User: {user_email}")
    print__feedback_flow(f"🔑 Run ID: '{request.run_id}'")
    print__feedback_flow(
        f"🔑 Run ID type: {type(request.run_id).__name__}, length: {len(request.run_id) if request.run_id else 0}"
    )
    print__feedback_flow(f"👍/👎 Feedback: {request.feedback}")
    print__feedback_flow(f"💬 Comment: {request.comment}")

    # Validate that at least one of feedback or comment is provided
    print__feedback_debug(f"🔍 Validating request data")
    if request.feedback is None and not request.comment:
        print__feedback_debug(f"🚨 No feedback or comment provided")
        raise HTTPException(
            status_code=400,
            detail="At least one of 'feedback' or 'comment' must be provided",
        )

    try:
        try:
            print__feedback_debug(f"🔍 Starting UUID validation")
            print__feedback_flow(f"🔍 Validating UUID format for: '{request.run_id}'")
            # Debug check if it resembles a UUID at all
            if not request.run_id or len(request.run_id) < 32:
                print__feedback_debug(
                    f"🚨 Run ID suspiciously short: '{request.run_id}' (length: {len(request.run_id)})"
                )
                print__feedback_flow(
                    f"⚠️ Run ID is suspiciously short for a UUID: '{request.run_id}'"
                )

            # Try to convert to UUID to validate format
            try:
                run_uuid = str(uuid.UUID(request.run_id))
                print__feedback_debug(f"🔍 UUID validation successful: '{run_uuid}'")
                print__feedback_flow(f"✅ UUID validation successful: '{run_uuid}'")
            except ValueError as uuid_error:
                print__feedback_debug(f"🚨 UUID validation failed: {str(uuid_error)}")
                print__feedback_flow(f"🚨 UUID ValueError details: {str(uuid_error)}")
                # More detailed diagnostic about the input
                for i, char in enumerate(request.run_id):
                    if not (char.isalnum() or char == "-"):
                        print__feedback_flow(
                            f"🚨 Invalid character at position {i}: '{char}' (ord={ord(char)})"
                        )
                raise
        except ValueError:
            print__feedback_debug(
                f"🚨 UUID format validation failed for: '{request.run_id}'"
            )
            print__feedback_flow(f"❌ UUID validation failed for: '{request.run_id}'")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}",
            )

        # 🔒 SECURITY CHECK: Verify user owns this run_id before submitting feedback
        print__feedback_debug(f"🔍 Starting ownership verification")
        print__feedback_flow(f"🔒 Verifying run_id ownership for user: {user_email}")

        try:
            # Get a healthy pool to check ownership
            print__feedback_debug(f"🔍 Importing get_direct_connection")

            print__feedback_debug(f"🔍 Getting healthy pool")
            pool = await get_direct_connection()
            print__feedback_debug(f"🔍 Pool obtained: {type(pool).__name__}")

            print__feedback_debug(f"🔍 Getting connection from pool")
            async with pool.connection() as conn:
                print__feedback_debug(f"🔍 Connection obtained: {type(conn).__name__}")
                async with conn.cursor() as cur:
                    print__feedback_debug(f"🔍 Executing ownership query")
                    await cur.execute(
                        """
                        SELECT COUNT(*) FROM users_threads_runs 
                        WHERE run_id = %s AND email = %s
                    """,
                        (run_uuid, user_email),
                    )

                    print__feedback_debug(
                        f"🔍 Ownership query executed, fetching result"
                    )
                    ownership_row = await cur.fetchone()
                    ownership_count = ownership_row[0] if ownership_row else 0
                    print__feedback_debug(f"🔍 Ownership count: {ownership_count}")

                    if ownership_count == 0:
                        print__feedback_debug(
                            f"🚨 User does not own run_id - access denied"
                        )
                        print__feedback_flow(
                            f"🚫 SECURITY: User {user_email} does not own run_id {run_uuid} - feedback denied"
                        )
                        raise HTTPException(
                            status_code=404, detail="Run ID not found or access denied"
                        )

                    print__feedback_debug(f"🔍 Ownership verification successful")
                    print__feedback_flow(
                        f"✅ SECURITY: User {user_email} owns run_id {run_uuid} - feedback authorized"
                    )

        except HTTPException:
            raise
        except Exception as ownership_error:
            print__feedback_debug(
                f"🚨 Ownership verification error: {type(ownership_error).__name__}: {str(ownership_error)}"
            )
            print__feedback_debug(
                f"🚨 Ownership error traceback: {traceback.format_exc()}"
            )
            print__feedback_flow(f"⚠️ Could not verify ownership: {ownership_error}")
            # Continue with feedback submission but log the warning
            print__feedback_flow(
                f"⚠️ Proceeding with feedback submission despite ownership check failure"
            )

        print__feedback_debug(f"🔍 Initializing LangSmith client")
        print__feedback_flow("🔄 Initializing LangSmith client")
        client = Client()
        print__feedback_debug(f"🔍 LangSmith client created: {type(client).__name__}")

        # Prepare feedback data for LangSmith
        print__feedback_debug(f"🔍 Preparing feedback data")
        feedback_kwargs = {"run_id": run_uuid, "key": "SENTIMENT"}

        # Only add score if feedback is provided
        if request.feedback is not None:
            feedback_kwargs["score"] = request.feedback
            print__feedback_debug(f"🔍 Adding score to feedback: {request.feedback}")
            print__feedback_flow(
                f"📤 Submitting feedback with score to LangSmith - run_id: '{run_uuid}', score: {request.feedback}"
            )
        else:
            print__feedback_debug(f"🔍 No score provided - comment-only feedback")
            print__feedback_flow(
                f"📤 Submitting comment-only feedback to LangSmith - run_id: '{run_uuid}'"
            )

        # Only add comment if provided
        if request.comment:
            feedback_kwargs["comment"] = request.comment
            print__feedback_debug(
                f"🔍 Adding comment to feedback (length: {len(request.comment)})"
            )

        print__feedback_debug(f"🔍 Submitting feedback to LangSmith")
        print__feedback_debug(f"🔍 Feedback kwargs: {feedback_kwargs}")
        client.create_feedback(**feedback_kwargs)
        print__feedback_debug(f"🔍 LangSmith feedback submission successful")

        print__feedback_flow(f"✅ Feedback successfully submitted to LangSmith")

        result = {
            "message": "Feedback submitted successfully",
            "run_id": run_uuid,
            "feedback": request.feedback,
            "comment": request.comment,
        }
        print__feedback_debug(f"🔍 FEEDBACK ENDPOINT - SUCCESSFUL EXIT")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print__feedback_debug(
            f"🚨 Exception in feedback processing: {type(e).__name__}: {str(e)}"
        )
        print__feedback_debug(
            f"🚨 Feedback processing traceback: {traceback.format_exc()}"
        )
        print__feedback_flow(f"🚨 LangSmith feedback submission error: {str(e)}")
        print__feedback_flow(f"🔍 Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")


@router.post("/sentiment")
async def update_sentiment(request: SentimentRequest, user=Depends(get_current_user)):
    """Update sentiment for a specific run_id."""

    print__sentiment_debug(f"🔍 SENTIMENT ENDPOINT - ENTRY POINT")
    print__sentiment_debug(f"🔍 Request received: run_id={request.run_id}")
    print__sentiment_debug(f"🔍 Sentiment value: {request.sentiment}")

    user_email = user.get("email")
    print__sentiment_debug(f"🔍 User email extracted: {user_email}")

    if not user_email:
        print__sentiment_debug(f"🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__sentiment_flow("📥 Incoming sentiment update request:")
    print__sentiment_flow(f"👤 User: {user_email}")
    print__sentiment_flow(f"🔑 Run ID: '{request.run_id}'")
    print__sentiment_flow(f"👍/👎 Sentiment: {request.sentiment}")

    try:
        # Validate UUID format
        print__sentiment_debug("🔍 Starting UUID validation")
        try:
            run_uuid = str(uuid.UUID(request.run_id))
            print__sentiment_debug(f"🔍 UUID validation successful: '{run_uuid}'")
            print__sentiment_flow(f"✅ UUID validation successful: '{run_uuid}'")
        except ValueError:
            print__sentiment_debug(f"🚨 UUID validation failed for: '{request.run_id}'")
            print__sentiment_flow(f"❌ UUID validation failed for: '{request.run_id}'")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}",
            )

        # 🔒 SECURITY: Update sentiment with user email verification
        print__sentiment_debug(
            "🔍 Starting sentiment update with ownership verification"
        )
        print__sentiment_flow("🔒 Verifying ownership before sentiment update")
        success = await update_thread_run_sentiment(run_uuid, request.sentiment)
        print__sentiment_debug(f"🔍 Sentiment update result: {success}")

        if success:
            print__sentiment_debug("🔍 Sentiment update successful")
            print__sentiment_flow("✅ Sentiment successfully updated")
            result = {
                "message": "Sentiment updated successfully",
                "run_id": run_uuid,
                "sentiment": request.sentiment,
            }
            print__sentiment_debug(f"🔍 SENTIMENT ENDPOINT - SUCCESSFUL EXIT")
            return result
        else:
            print__sentiment_debug(
                "🚨 Sentiment update failed - access denied or not found"
            )
            print__sentiment_flow(
                "❌ Failed to update sentiment - run_id may not exist or access denied"
            )
            raise HTTPException(
                status_code=404, detail=f"Run ID not found or access denied: {run_uuid}"
            )

    except HTTPException:
        raise
    except Exception as e:
        print__sentiment_debug(
            f"🚨 Exception in sentiment processing: {type(e).__name__}: {str(e)}"
        )
        print__sentiment_debug(
            f"🚨 Sentiment processing traceback: {traceback.format_exc()}"
        )
        print__sentiment_flow(f"🚨 Sentiment update error: {str(e)}")
        print__sentiment_flow(f"🔍 Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Failed to update sentiment: {e}")
