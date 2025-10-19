"""Stop execution endpoint for cancelling running analyses.

This module provides an endpoint to stop/cancel running analysis executions
in a multi-user environment. Each user can only stop their own executions.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies.auth import get_current_user
from api.utils.cancellation import request_cancellation, get_active_count
from api.utils.debug import print__analyze_debug

router = APIRouter()


class StopExecutionRequest(BaseModel):
    """Request model for stopping an execution."""

    thread_id: str
    run_id: str


@router.post("/stop-execution")
async def stop_execution(request: StopExecutionRequest, user=Depends(get_current_user)):
    """Stop a running analysis execution.

    This endpoint allows users to cancel their own running analyses.
    The cancellation is tracked by thread_id and run_id to ensure
    multi-user safety - each user can only stop their own executions.

    Args:
        request: Contains thread_id and run_id to identify the execution
        user: Authenticated user from JWT token

    Returns:
        Success message if cancellation was requested

    Raises:
        HTTPException: If user is not authenticated or execution not found
    """
    print__analyze_debug(f"[StopExecution] 🛑 Received stop request")
    print__analyze_debug(f"[StopExecution] thread_id: {request.thread_id}")
    print__analyze_debug(f"[StopExecution] run_id: {request.run_id}")

    user_email = user.get("email")
    if not user_email:
        print__analyze_debug("[StopExecution] ❌ No user email in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__analyze_debug(f"[StopExecution] User: {user_email}")
    print__analyze_debug(f"[StopExecution] Active executions: {get_active_count()}")

    # Request cancellation
    success = request_cancellation(request.thread_id, request.run_id)

    if success:
        print__analyze_debug(f"[StopExecution] ✅ Cancellation requested successfully")
        return {
            "status": "success",
            "message": "Execution stop requested",
            "thread_id": request.thread_id,
            "run_id": request.run_id,
        }
    else:
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
