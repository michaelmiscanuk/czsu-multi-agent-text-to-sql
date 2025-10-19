"""Cancellation token management for multi-user execution control.

This module provides a centralized system for managing cancellation requests
in a multi-user environment. Each execution is tracked by thread_id + run_id,
allowing users to cancel only their own running analyses.
"""

import asyncio
from typing import Dict, Tuple
from datetime import datetime, timedelta

# Store cancellation flags per (thread_id, run_id)
# Key: (thread_id, run_id), Value: {"cancelled": bool, "timestamp": datetime}
_cancellation_registry: Dict[Tuple[str, str], Dict] = {}

# Cleanup old entries after 30 minutes
CLEANUP_THRESHOLD = timedelta(minutes=30)


def register_execution(thread_id: str, run_id: str) -> None:
    """Register a new execution that can be cancelled.

    Args:
        thread_id: The thread identifier
        run_id: The run identifier
    """
    key = (thread_id, run_id)
    _cancellation_registry[key] = {"cancelled": False, "timestamp": datetime.now()}
    print(f"[Cancellation] Registered execution: thread={thread_id}, run={run_id}")


def request_cancellation(thread_id: str, run_id: str) -> bool:
    """Request cancellation for a specific execution.

    Args:
        thread_id: The thread identifier
        run_id: The run identifier

    Returns:
        True if the execution was found and marked for cancellation, False otherwise
    """
    key = (thread_id, run_id)
    if key in _cancellation_registry:
        _cancellation_registry[key]["cancelled"] = True
        print(
            f"[Cancellation] Requested cancellation: thread={thread_id}, run={run_id}"
        )
        return True
    else:
        print(f"[Cancellation] Execution not found: thread={thread_id}, run={run_id}")
        return False


def is_cancelled(thread_id: str, run_id: str) -> bool:
    """Check if an execution has been cancelled.

    Args:
        thread_id: The thread identifier
        run_id: The run identifier

    Returns:
        True if the execution was cancelled, False otherwise
    """
    key = (thread_id, run_id)
    if key in _cancellation_registry:
        return _cancellation_registry[key]["cancelled"]
    return False


def unregister_execution(thread_id: str, run_id: str) -> None:
    """Unregister an execution after it completes.

    Args:
        thread_id: The thread identifier
        run_id: The run identifier
    """
    key = (thread_id, run_id)
    if key in _cancellation_registry:
        del _cancellation_registry[key]
        print(
            f"[Cancellation] Unregistered execution: thread={thread_id}, run={run_id}"
        )


def cleanup_old_entries() -> None:
    """Remove old entries from the registry to prevent memory leaks."""
    now = datetime.now()
    keys_to_remove = []

    for key, value in _cancellation_registry.items():
        if now - value["timestamp"] > CLEANUP_THRESHOLD:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del _cancellation_registry[key]

    if keys_to_remove:
        print(f"[Cancellation] Cleaned up {len(keys_to_remove)} old entries")


def get_active_count() -> int:
    """Get the number of active tracked executions."""
    return len(_cancellation_registry)
