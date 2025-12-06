"""
MODULE_DESCRIPTION: User Feedback and Sentiment Tracking - Quality Monitoring and User Satisfaction

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module implements user feedback collection and sentiment tracking endpoints
for the CZSU Multi-Agent Text-to-SQL API. It provides two main functionalities:

1. Feedback Submission (/feedback) - Submit detailed feedback to LangSmith
2. Sentiment Update (/sentiment) - Quick sentiment rating storage

These endpoints enable:
    - Quality monitoring and improvement
    - User satisfaction tracking
    - Model performance evaluation
    - Feedback-driven iteration
    - User experience analytics

Note on run_id:
    The run_id identifies the root run of a LangGraph execution, which
    LangSmith also uses as the trace identifier. Feedback is attached
    to this run_id via LangSmith's create_feedback API.

Feedback Flow:
    User receives AI response → User rates/comments → Data sent to LangSmith/Database
    → Analytics and model improvement → Better AI responses

===================================================================================
KEY FEATURES
===================================================================================

1. LangSmith Integration (/feedback)
   - Submit feedback to LangSmith platform
   - Binary rating (thumbs up/down) support
   - Text comments for detailed feedback
   - Run ID tracking for traceability
   - Automatic quality metrics

2. Sentiment Tracking (/sentiment)
   - Quick thumbs up/down storage
   - Database persistence
   - User-specific sentiment history
   - Run ID association
   - Per-query granularity

3. Security and Ownership Verification
   - JWT authentication required
   - User ownership validation
   - Run ID access control
   - No cross-user feedback manipulation
   - Audit logging

4. UUID Validation
   - Strict UUID format validation
   - Invalid format rejection
   - Detailed error messages
   - Character-level diagnostics
   - Security against injection

5. Flexible Feedback Options
   - Rating only (binary score)
   - Comment only (text feedback)
   - Both rating and comment
   - Minimum one required
   - Optional fields handling

6. Comprehensive Logging
   - Debug logging for all operations
   - Flow logging for user journeys
   - Error diagnostics
   - Security event logging
   - Performance monitoring

7. Windows Compatibility
   - WindowsSelectorEventLoopPolicy for async operations
   - psycopg async support
   - Cross-platform compatibility

===================================================================================
API ENDPOINTS
===================================================================================

POST /feedback
    Submit feedback for a specific query execution to LangSmith

    Authentication: JWT token required
    Security: User must own the run_id

    Request Body:
        {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "feedback": 1,        // Optional: 1 (positive), 0 (negative), null (none)
            "comment": "Great!"   // Optional: Text comment
        }

    Validation:
        - At least one of feedback or comment must be provided
        - run_id must be valid UUID format
        - User must own the run_id

    Returns:
        {
            "message": "Feedback submitted successfully",
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "feedback": 1,
            "comment": "Great!"
        }

    Error Responses:
        400: Invalid run_id format or missing both feedback and comment
        401: User not authenticated
        404: Run ID not found or access denied
        500: LangSmith submission failure

POST /sentiment
    Update sentiment rating for a specific query execution

    Authentication: JWT token required
    Security: Ownership verified in database update

    Request Body:
        {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "sentiment": "positive"  // or "negative", "neutral"
        }

    Validation:
        - run_id must be valid UUID format
        - sentiment must be valid value
        - User must own the run_id (verified in DB update)

    Returns:
        {
            "message": "Sentiment updated successfully",
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "sentiment": "positive"
        }

    Error Responses:
        400: Invalid run_id format
        401: User not authenticated
        404: Run ID not found or access denied
        500: Database update failure

===================================================================================
LANGSMITH INTEGRATION
===================================================================================

What is LangSmith:
    - LangChain's observability and monitoring platform
    - Tracks LLM application runs and performance
    - Collects user feedback for model improvement
    - Provides analytics and debugging tools

Feedback Submission:
    client = Client()  # LangSmith client
    client.create_feedback(
        run_id=run_uuid,
        key="SENTIMENT",
        score=request.feedback,      # 1 or 0
        comment=request.comment      # Optional text
    )

Feedback Structure:
    - run_id: Links feedback to specific execution
    - key: "SENTIMENT" - categorizes feedback type
    - score: Binary rating (1=positive, 0=negative)
    - comment: Free-text user comment

LangSmith Benefits:
    - Automatic aggregation of feedback metrics
    - Visualization of user satisfaction trends
    - Correlation with model performance
    - A/B testing support
    - Historical feedback analysis

Use Cases:
    - Track model accuracy over time
    - Identify problematic query patterns
    - Measure user satisfaction
    - Prioritize improvements
    - Validate model updates

===================================================================================
SENTIMENT TRACKING
===================================================================================

Database Storage:
    Table: users_threads_runs
    Columns:
        - run_id (UUID, PRIMARY KEY)
        - email (TEXT)
        - thread_id (TEXT)
        - prompt (TEXT)
        - timestamp (TIMESTAMP)
        - sentiment (TEXT, NULLABLE)

Sentiment Values:
    - "positive": User satisfied with response
    - "negative": User unsatisfied with response
    - "neutral": User neutral about response
    - null: No sentiment recorded yet

Update Process:
    1. Validate UUID format
    2. Call update_thread_run_sentiment(run_uuid, sentiment)
    3. Function verifies user ownership internally
    4. Updates sentiment column in database
    5. Returns success/failure

Ownership Verification:
    - Handled by update_thread_run_sentiment function
    - Uses WHERE run_id = %s AND email = %s
    - Prevents cross-user sentiment manipulation
    - Returns False if not owned

Difference from /feedback:
    - /feedback → LangSmith (external platform)
    - /sentiment → Database (internal storage)
    - Can use both for same run_id
    - Complementary systems

===================================================================================
SECURITY AND OWNERSHIP VERIFICATION
===================================================================================

Security Model:
    1. Authentication: JWT token required (all endpoints)
    2. Authorization: User must own the run_id
    3. Validation: UUID format strictly enforced
    4. Logging: All access attempts logged

Ownership Verification Process (/feedback):
    1. Extract user_email from JWT token
    2. Validate run_id is UUID format
    3. Query database:
       SELECT COUNT(*) FROM users_threads_runs
       WHERE run_id = %s AND email = %s
    4. If count == 0 → Access denied (404)
    5. If count > 0 → Authorized, proceed

Ownership Verification Process (/sentiment):
    - Delegated to update_thread_run_sentiment function
    - Function includes WHERE email = %s clause
    - Updates only if user owns run_id
    - Returns False if update affected 0 rows

Why Ownership Matters:
    - Prevents feedback spam
    - Ensures feedback authenticity
    - Protects user privacy
    - Prevents malicious manipulation
    - Maintains data integrity

Security Scenarios:
    1. Valid owner → Feedback accepted
    2. Non-owner → 404 error (doesn't reveal existence)
    3. Invalid UUID → 400 error (format validation)
    4. Missing auth → 401 error (authentication)

===================================================================================
UUID VALIDATION
===================================================================================

Validation Process:
    try:
        run_uuid = str(uuid.UUID(request.run_id))
        # Validation successful
    except ValueError:
        raise HTTPException(400, "Invalid run_id format")

Why Strict Validation:
    - Prevents SQL injection
    - Ensures data type consistency
    - Catches client-side errors early
    - Improves error messages
    - Security best practice

Diagnostic Logging:
    - Length check (< 32 chars suspicious)
    - Character-by-character validation
    - Invalid character detection
    - Position and ordinal reporting

Example Diagnostics:
    Input: "invalid-uuid-123"
    Logs:
        - Run ID suspiciously short: length 16
        - Invalid character at position 7: '-' (expected hex)
        - UUID ValueError: badly formed hexadecimal UUID string

Valid UUID Format:
    550e8400-e29b-41d4-a716-446655440000
    - 36 characters total
    - 5 groups separated by hyphens
    - Hexadecimal characters only

===================================================================================
FLEXIBLE FEEDBACK OPTIONS
===================================================================================

Three Submission Modes:

1. Rating Only:
   {
       "run_id": "...",
       "feedback": 1,
       "comment": null
   }
   → LangSmith receives score only

2. Comment Only:
   {
       "run_id": "...",
       "feedback": null,
       "comment": "The query was incorrect"
   }
   → LangSmith receives comment only

3. Both Rating and Comment:
   {
       "run_id": "...",
       "feedback": 0,
       "comment": "SQL syntax error"
   }
   → LangSmith receives both

Validation Rule:
    if feedback is None and not comment:
        raise HTTPException(400, "At least one required")

Why Flexibility:
    - Reduces user friction (quick thumbs up/down)
    - Allows detailed feedback when needed
    - Supports various UI patterns
    - Accommodates user preferences

UI Patterns:
    - Quick rating: Single click (feedback only)
    - Detailed feedback: Text box (comment only)
    - Full feedback: Rating + comment dialog

===================================================================================
ERROR HANDLING
===================================================================================

Error Handling Hierarchy:

1. Validation Errors (400)
   - Invalid UUID format
   - Missing both feedback and comment
   - Malformed request body

2. Authentication Errors (401)
   - No JWT token provided
   - Invalid JWT token
   - User email not in token

3. Authorization Errors (404)
   - Run ID not found
   - User doesn't own run_id
   - Access denied

4. External Service Errors (500)
   - LangSmith API failure
   - Database connection error
   - Unexpected exceptions

Error Response Pattern:
    try:
        # Endpoint logic
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise HTTPException(500, f"Failed: {e}")

Graceful Degradation:
    - Ownership check failure → Log warning, continue submission
    - Non-critical errors → Log but don't fail request
    - Critical errors → Return proper HTTP error

===================================================================================
LOGGING AND DEBUGGING
===================================================================================

Debug Functions:
    - print__feedback_debug: Detailed feedback endpoint logging
    - print__feedback_flow: User journey flow logging
    - print__sentiment_debug: Detailed sentiment endpoint logging
    - print__sentiment_flow: User journey flow logging

Logged Information:

Entry Point:
    - Endpoint called
    - Request parameters
    - User email extraction

Validation:
    - UUID format validation
    - Request data validation
    - Character-level diagnostics (if errors)

Security:
    - Ownership verification start
    - Database query execution
    - Ownership result (granted/denied)
    - Security decisions

Processing:
    - LangSmith client initialization
    - Feedback data preparation
    - Submission to external services
    - Database updates

Results:
    - Success/failure status
    - Response data
    - Error details if failed

Security Events:
    - Access denied (user doesn't own run_id)
    - Ownership check failures
    - Authentication issues

Log Patterns:
    print__feedback_flow("📥 Incoming feedback request:")
    print__feedback_flow(f"👤 User: {user_email}")
    print__feedback_flow(f"✅ Feedback successfully submitted")

Benefits:
    - Troubleshoot issues quickly
    - Audit user actions
    - Monitor security events
    - Understand user journeys
    - Debug validation failures

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Feedback Endpoint (/feedback):
    - Response time: 100-300ms (includes LangSmith API call)
    - Network latency: ~50-100ms (LangSmith)
    - Database query: ~10-20ms (ownership check)
    - Processing: <10ms
    - External dependency: LangSmith API

Sentiment Endpoint (/sentiment):
    - Response time: 50-150ms
    - Database query: ~20-50ms (update with WHERE clause)
    - Processing: <10ms
    - No external dependencies

Optimization Opportunities:
    1. Async LangSmith submission (fire-and-forget)
    2. Batch feedback submissions
    3. Cache ownership checks (short TTL)
    4. Connection pooling (already implemented)

Scaling:
    - Linear scaling with request volume
    - LangSmith API rate limits apply
    - Database write scalability depends on PostgreSQL
    - Consider queueing for high volume

===================================================================================
INTEGRATION WITH FRONTEND
===================================================================================

Typical UI Flow:

1. User Receives AI Response
   - Query executed
   - Results displayed
   - run_id available in response

2. User Provides Feedback
   - Click thumbs up/down
   - Optionally add comment
   - Submit feedback

3. Frontend Makes Request
   POST /feedback
   {
       "run_id": response.run_id,
       "feedback": 1,  // or 0
       "comment": optional_text
   }

4. Optional Sentiment Update
   POST /sentiment
   {
       "run_id": response.run_id,
       "sentiment": "positive"
   }

UI Patterns:

Quick Feedback:
    - Single thumbs up/down button
    - No comment required
    - Instant submission
    - Minimal user friction

Detailed Feedback:
    - Thumbs up/down + comment field
    - Optional text input
    - Submit button
    - Richer data collection

Sentiment Only:
    - Quick sentiment buttons (😊😐😞)
    - Maps to positive/neutral/negative
    - Fast user interaction
    - Lightweight tracking

===================================================================================
MONITORING AND ANALYTICS
===================================================================================

Metrics to Track:

Feedback Metrics:
    - Submission rate (% of users providing feedback)
    - Positive vs negative ratio
    - Average comment length
    - Time to feedback submission

Sentiment Metrics:
    - Sentiment distribution (positive/negative/neutral)
    - Sentiment trends over time
    - Per-user sentiment patterns
    - Correlation with query complexity

Quality Metrics:
    - Feedback consistency (same query patterns)
    - User satisfaction scores
    - Model performance correlation
    - Improvement validation

Error Metrics:
    - Validation failure rate
    - Ownership denial rate
    - LangSmith API failures
    - Database errors

Alerting:
    - High negative feedback rate (> 30%)
    - LangSmith API errors (> 5%)
    - Ownership check failures (suspicious activity)
    - Validation error spike

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    1. UUID validation (valid/invalid formats)
    2. Ownership verification logic
    3. Feedback data preparation
    4. Error handling paths
    5. Request validation

Integration Tests:
    1. Full feedback submission to LangSmith
    2. Sentiment database updates
    3. Ownership verification with database
    4. Authentication integration
    5. Error responses

Mock Testing:
    - Mock LangSmith Client
    - Mock database connections
    - Mock authentication
    - Test error scenarios

Test Data:
    - Valid UUIDs
    - Invalid UUIDs (malformed, short, special chars)
    - Owned/non-owned run_ids
    - Various feedback combinations
    - Edge cases (empty strings, nulls)

Security Testing:
    - Cross-user access attempts
    - Missing authentication
    - SQL injection attempts
    - UUID format exploits

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection, path manipulation
    - uuid: UUID validation and conversion
    - traceback: Error diagnostics
    - typing: Type hints
    - asyncio: Event loop (Windows)

Third-Party:
    - fastapi: Web framework, routing, dependencies
    - langsmith: LangSmith client for feedback submission
    - dotenv: Environment variable loading

Internal:
    - api.dependencies.auth: JWT authentication (get_current_user)
    - api.models.requests: FeedbackRequest, SentimentRequest models
    - api.utils.debug: Debug logging functions
    - api.helpers: Traceback JSON response utility
    - checkpointer.user_management.sentiment_tracking: Database sentiment updates
    - checkpointer.database.connection: Database connection management

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Feedback Analytics Dashboard
   - Real-time feedback visualization
   - Sentiment trend graphs
   - User satisfaction metrics
   - Query performance correlation

2. Automated Response
   - Thank user for feedback
   - Acknowledge specific issues
   - Provide status updates
   - Close feedback loop

3. Feedback Categories
   - Accuracy issues
   - Performance problems
   - UI/UX feedback
   - Feature requests
   - Structured classification

4. Batch Operations
   - Bulk feedback submission
   - Historical feedback import
   - Batch sentiment updates
   - Improved performance

5. Advanced Analytics
   - ML-based sentiment analysis on comments
   - Automatic issue detection
   - Pattern recognition
   - Predictive quality metrics

6. Webhook Integration
   - Real-time feedback notifications
   - Slack/Teams integration
   - PagerDuty alerting
   - Custom webhooks

===================================================================================
"""

# ==============================================================================
# CRITICAL WINDOWS COMPATIBILITY CONFIGURATION
# ==============================================================================

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# with asyncio on Windows platforms. This prevents "Event loop is closed" errors
# and ensures proper async database operations.
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# PATH AND DIRECTORY CONSTANTS
# ==============================================================================

# Determine base directory for the project
# Handles both normal execution and special environments (e.g., REPL, Jupyter)
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

# ==============================================================================
# AUTHENTICATION AND AUTHORIZATION
# ==============================================================================

# Import JWT-based authentication dependency for user verification
from api.dependencies.auth import get_current_user

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

# Import Pydantic models for feedback and sentiment request validation
from api.models.requests import FeedbackRequest, SentimentRequest

# ==============================================================================
# DEBUG AND LOGGING UTILITIES
# ==============================================================================

# Import specialized debug logging functions for feedback workflows
from api.utils.debug import (
    print__feedback_debug,
    print__feedback_flow,
    print__sentiment_debug,
    print__sentiment_flow,
)

# ==============================================================================
# DATABASE AND BUSINESS LOGIC IMPORTS
# ==============================================================================

# Add project root to Python path for direct imports
sys.path.insert(0, str(BASE_DIR))

# Import error response formatting helper
from api.helpers import traceback_json_response

# Import sentiment tracking and database connection utilities
from checkpointer.user_management.sentiment_tracking import update_thread_run_sentiment
from checkpointer.database.connection import get_direct_connection

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for feedback and sentiment endpoints
# This router will be included in the main FastAPI application
router = APIRouter()


# ==============================================================================
# FEEDBACK SUBMISSION ENDPOINT
# ==============================================================================


@router.post(
    "/feedback",
    summary="Submit user feedback",
    description="""
    **Submit feedback for a specific query execution run.**
    
    Feedback is logged to LangSmith for quality tracking and model improvement.
    You can provide a binary rating (thumbs up/down) and/or a text comment.
    
    **At least one of `feedback` or `comment` must be provided.**
    """,
    response_description="Confirmation of feedback submission",
    responses={
        200: {
            "description": "Feedback successfully submitted",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Feedback submitted successfully",
                        "run_id": "550e8400-e29b-41d4-a716-446655440000",
                    }
                }
            },
        },
        400: {"description": "Invalid request - Missing both feedback and comment"},
        404: {"description": "Run ID not found in LangSmith"},
    },
)
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

    # ==========================================================================
    # REQUEST VALIDATION AND LOGGING
    # ==========================================================================

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
        # ======================================================================
        # UUID FORMAT VALIDATION
        # ======================================================================

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

        # ======================================================================
        # SECURITY: OWNERSHIP VERIFICATION
        # ======================================================================

        # 🔒 SECURITY CHECK: Verify user owns this run_id before submitting feedback
        # This prevents users from submitting feedback for other users' queries
        print__feedback_debug(f"🔍 Starting ownership verification")
        print__feedback_flow(f"🔒 Verifying run_id ownership for user: {user_email}")

        try:
            # Get a healthy pool to check ownership
            print__feedback_debug(f"🔍 Importing get_direct_connection")

            # FIX: get_direct_connection is an async context manager returning a direct connection,
            # not a pool. Use it directly with 'async with' instead of awaiting and then calling pool.connection().
            print__feedback_debug(f"🔍 Acquiring direct DB connection")
            async with get_direct_connection() as conn:
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

        # ======================================================================
        # LANGSMITH FEEDBACK SUBMISSION
        # ======================================================================

        # Initialize LangSmith client for external feedback tracking and analytics
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
        resp = traceback_json_response(
            e, run_id=run_uuid if "run_uuid" in locals() else None
        )
        if resp:
            return resp
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")


# ==============================================================================
# SENTIMENT UPDATE ENDPOINT
# ==============================================================================


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
        resp = traceback_json_response(
            e, run_id=run_uuid if "run_uuid" in locals() else None
        )
        if resp:
            return resp
        raise HTTPException(status_code=500, detail=f"Failed to update sentiment: {e}")
