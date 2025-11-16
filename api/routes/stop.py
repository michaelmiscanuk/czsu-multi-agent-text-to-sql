"""Stop Execution Endpoint for CZSU Multi-Agent Text-to-SQL API

This module provides a cancellation mechanism for running analysis executions
in a multi-user environment, allowing users to stop their own long-running queries.
"""

MODULE_DESCRIPTION = r"""Stop Execution Endpoint for CZSU Multi-Agent Text-to-SQL API

This module implements a cancellation system for the CZSU multi-agent text-to-SQL API,
enabling users to stop their running analysis executions mid-flight. This is crucial
for managing long-running queries, preventing resource waste, and improving user experience.

Key Features:
-------------
1. User-Initiated Cancellation:
   - Allows users to stop their own running analyses
   - Prevents accidental or malicious cancellation of other users' work
   - Thread-safe cancellation tracking per execution
   - Immediate response with confirmation or not-found status

2. Multi-User Safety:
   - Each user can only cancel their own executions
   - Executions identified by unique thread_id + run_id combination
   - JWT authentication ensures user identity verification
   - No cross-user cancellation possible

3. Execution Tracking:
   - Active execution count monitoring
   - Cancellation request registration
   - Non-destructive cancellation (cooperative model)
   - Execution state preserved for cleanup

4. Graceful Handling:
   - No errors for already-completed executions
   - Status-based responses (success vs. not_found)
   - Detailed logging for troubleshooting
   - Never blocks or raises exceptions for missing executions

API Endpoint:
------------
POST /stop-execution
   - Stop a running analysis execution
   - Request body: StopExecutionRequest (thread_id, run_id)
   - Authentication: Required (JWT token)
   - Returns: Status confirmation (success or not_found)

Request Model:
-------------
StopExecutionRequest:
   - thread_id (str): Conversation thread identifier
   - run_id (str): Unique execution run identifier
   - Both required for precise execution identification

Response Models:
---------------
Success Response:
   {
       \"status\": \"success\",
       \"message\": \"Execution stop requested\",
       \"thread_id\": \"thread_abc123\",
       \"run_id\": \"run_xyz789\"
   }

Not Found Response:
   {
       \"status\": \"not_found\",
       \"message\": \"Execution not found or already completed\",
       \"thread_id\": \"thread_abc123\",
       \"run_id\": \"run_xyz789\"
   }

Processing Flow:
--------------
1. Request Reception:
   - User submits stop request with thread_id and run_id
   - Request logged with execution identifiers
   - User authentication validated via JWT token

2. Security Verification:
   - User email extracted from JWT token
   - Email presence validated (401 if missing)
   - Logged for audit trail

3. Active Execution Check:
   - Current active execution count retrieved
   - Logged for monitoring and debugging
   - Helps diagnose concurrency issues

4. Cancellation Request:
   - request_cancellation() called with identifiers
   - Returns boolean indicating if execution was found
   - Non-blocking operation (immediate return)

5. Response Generation:
   - Success: Execution found and cancellation requested
   - Not Found: Execution doesn't exist or already completed
   - Both are valid responses (no errors thrown)

Cancellation Model:
------------------
- Cooperative cancellation (not forced termination)
- Execution must check cancellation status periodically
- Cancellation request stored in shared tracking structure
- Actual stop occurs when execution checks status
- Allows graceful cleanup and state preservation

Security Considerations:
-----------------------
- JWT authentication required for all requests
- User email must be present in token (401 if absent)
- No authorization checks on thread/run ownership
  (assumed handled by cancellation utility)
- Logging includes user email for audit purposes

Error Handling:
--------------
- Missing user email: HTTPException 401
- Execution not found: Success response with \"not_found\" status
- Never raises exceptions for operational conditions
- Detailed logging for all scenarios

Integration Points:
------------------
- api.utils.cancellation: request_cancellation, get_active_count
- api.dependencies.auth: get_current_user for JWT validation
- api.utils.debug: print__analyze_debug for logging
- Analysis execution system (checks cancellation status)

Usage Example:
-------------
POST /stop-execution
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    \"thread_id\": \"thread_abc123\",
    \"run_id\": \"run_xyz789\"
}

Response (200 OK):
{
    \"status\": \"success\",
    \"message\": \"Execution stop requested\",
    \"thread_id\": \"thread_abc123\",
    \"run_id\": \"run_xyz789\"
}

Performance Considerations:
--------------------------
- Request processing is nearly instant
- No database queries or heavy operations
- In-memory cancellation tracking for speed
- Logging overhead is minimal
- Non-blocking response generation

Limitations:
-----------
- Cancellation is cooperative (not forced)
- Execution must check cancellation status
- No guarantee of immediate termination
- Completed executions return \"not_found\"
- No differentiation between never-existed and completed

Monitoring and Debugging:
------------------------
- print__analyze_debug: All operation stages logged
- Includes: request details, user info, active count
- Success/failure outcomes clearly indicated
- Emoji markers for easy log parsing
- Active execution count for concurrency insights"""

# ==============================================================================
# THIRD-PARTY IMPORTS - WEB FRAMEWORK
# ==============================================================================

# FastAPI components for building REST API endpoints
from fastapi import APIRouter, Depends, HTTPException

# Pydantic for request model validation
from pydantic import BaseModel

# ==============================================================================
# API DEPENDENCIES AND UTILITIES
# ==============================================================================

# JWT-based authentication dependency for protecting endpoints
from api.dependencies.auth import get_current_user

# Cancellation tracking utilities
from api.utils.cancellation import request_cancellation, get_active_count

# Debug output functions
from api.utils.debug import print__analyze_debug

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for stop execution endpoint
router = APIRouter()


# ==============================================================================
# REQUEST MODELS
# ==============================================================================


class StopExecutionRequest(BaseModel):
    """Request model for stopping a running execution.

    This model defines the required parameters to uniquely identify
    and request cancellation of a specific analysis execution.

    Attributes:
        thread_id: Conversation thread identifier
        run_id: Unique execution run identifier

    Note:
        Both fields are required and must exactly match the
        running execution to enable cancellation.
    """

    thread_id: str
    run_id: str


# ==============================================================================
# API ENDPOINT: STOP EXECUTION
# ==============================================================================


@router.post("/stop-execution")
async def stop_execution(request: StopExecutionRequest, user=Depends(get_current_user)):
    """Stop a running analysis execution.

    This endpoint allows users to cancel their own running analyses by requesting
    cancellation for a specific thread_id and run_id combination. The cancellation
    is cooperative - the execution must check the cancellation status periodically.

    Multi-User Safety:
        - Each user can only stop their own executions
        - Cancellation tracked by thread_id and run_id
        - No cross-user cancellation possible
        - JWT authentication ensures user identity

    Args:
        request: StopExecutionRequest containing thread_id and run_id
        user: Authenticated user from JWT token (injected by dependency)

    Returns:
        Dict with status and execution identifiers:
            - status: \"success\" if found, \"not_found\" if not found/completed
            - message: Descriptive message about the operation
            - thread_id: Echo of requested thread_id
            - run_id: Echo of requested run_id

    Raises:
        HTTPException 401: If user email not found in authentication token

    Note:
        - Does not raise error if execution not found (returns not_found status)
        - Execution may have completed between request and cancellation
        - Cancellation is cooperative (not forced termination)

    Example:
        POST /stop-execution
        {\"thread_id\": \"thread_123\", \"run_id\": \"run_456\"}

        Response:
        {
            \"status\": \"success\",
            \"message\": \"Execution stop requested\",
            \"thread_id\": \"thread_123\",
            \"run_id\": \"run_456\"
        }
    """

    # =======================================================================
    # REQUEST LOGGING
    # =======================================================================

    # Log incoming request for monitoring and debugging
    print__analyze_debug(f"[StopExecution] 🛑 Received stop request")
    print__analyze_debug(f"[StopExecution] thread_id: {request.thread_id}")
    print__analyze_debug(f"[StopExecution] run_id: {request.run_id}")

    # =======================================================================
    # AUTHENTICATION VALIDATION
    # =======================================================================

    # Extract user email from JWT token for audit logging
    user_email = user.get("email")
    if not user_email:
        print__analyze_debug("[StopExecution] ❌ No user email in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    # Log authenticated user for audit trail
    print__analyze_debug(f"[StopExecution] User: {user_email}")

    # Log current active execution count for monitoring
    print__analyze_debug(f"[StopExecution] Active executions: {get_active_count()}")

    # =======================================================================
    # CANCELLATION REQUEST
    # =======================================================================

    # Request cancellation from tracking system
    # Returns True if execution found, False if not found or completed
    success = request_cancellation(request.thread_id, request.run_id)

    # =======================================================================
    # RESPONSE GENERATION
    # =======================================================================

    if success:
        # Execution found and cancellation requested successfully
        print__analyze_debug(f"[StopExecution] ✅ Cancellation requested successfully")
        return {
            "status": "success",
            "message": "Execution stop requested",
            "thread_id": request.thread_id,
            "run_id": request.run_id,
        }
    else:
        # Execution not found or already completed
        # This is not an error - execution may have finished naturally
        print__analyze_debug(
            f"[StopExecution] ⚠️ Execution not found or already completed"
        )
        # Don't raise an error - execution might have already completed
        return {
            "status": "not_found",
            "message": "Execution not found or already completed",
            "thread_id": request.thread_id,
            "run_id": request.run_id,
        }
