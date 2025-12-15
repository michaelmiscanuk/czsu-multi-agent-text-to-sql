"""Lightweight execution lock management for per-user/IP concurrency control.

This module enforces a single in-flight analysis per client identity (currently
mapped to client IP + user email) to ensure users cannot begin a second prompt
while a previous one is still running. The locks are intentionally simple and
process-local; they prevent duplicate work within a single FastAPI instance and
fall back gracefully if a process crashes thanks to time-based expiration.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Deque, Dict

from fastapi import Request

# Maximum time (in seconds) a lock can stay active before being considered stale.
# Align with backend analysis timeout (4 minutes) with buffer to cover retries.
DEFAULT_LOCK_TTL_SECONDS = 600  # 10 minutes
LOCK_TTL_SECONDS = int(
    os.getenv("USER_EXECUTION_LOCK_TTL_SECONDS", DEFAULT_LOCK_TTL_SECONDS)
)

# Allow limited parallel executions per identity (IP+user) to avoid false conflicts
DEFAULT_MAX_SLOTS_PER_IDENTITY = 2
MAX_SLOTS_PER_IDENTITY = max(
    1,
    int(
        os.getenv(
            "USER_EXECUTION_LOCK_MAX_SLOTS",
            str(DEFAULT_MAX_SLOTS_PER_IDENTITY),
        )
    ),
)

# Internal registries guarded by an asyncio lock for thread-safety inside the event loop.
_registry_lock = asyncio.Lock()
_active_slots: Dict[str, Deque[float]] = {}


def _purge_expired(now: float) -> None:
    """Remove lock entries that exceeded the configured TTL."""
    expired_keys = []
    for key, timestamps in _active_slots.items():
        while timestamps and now - timestamps[0] > LOCK_TTL_SECONDS:
            timestamps.popleft()
        if not timestamps:
            expired_keys.append(key)

    for key in expired_keys:
        _active_slots.pop(key, None)


def build_execution_identity(client_ip: str | None, user_email: str | None) -> str:
    """Create a stable identity string for lock tracking."""
    ip_segment = (client_ip or "unknown-ip").strip().lower()
    user_segment = (user_email or "anonymous").strip().lower()
    # Prioritize the IP requirement while keeping the user in the identifier for logging.
    return f"{ip_segment}::{user_segment}"


def get_client_ip(request: Request) -> str:
    """Extract the best-guess client IP address from the incoming request."""
    # Respect reverse-proxy headers first.
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Use the first IP in the chain (original client).
        return forwarded_for.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown-ip"


async def acquire_execution_slot(identity: str) -> bool:
    """Attempt to acquire the execution slot for the provided identity.

    Returns True if the slot was acquired, False if another execution is in progress.
    """
    if not identity:
        return True  # Nothing to guard.

    now = time.time()
    async with _registry_lock:
        _purge_expired(now)
        timestamps = _active_slots.setdefault(identity, deque())
        if len(timestamps) >= MAX_SLOTS_PER_IDENTITY:
            return False

        timestamps.append(now)
        return True


def release_execution_slot(identity: str) -> None:
    """Release an execution slot when analysis completes or fails."""
    if not identity:
        return
    timestamps = _active_slots.get(identity)
    if not timestamps:
        return

    if timestamps:
        timestamps.popleft()

    if not timestamps:
        _active_slots.pop(identity, None)


def has_active_slot(identity: str) -> bool:
    """Utility helper (mainly for debugging/tests) to check if a slot is active."""
    if not identity:
        return False
    return identity in _active_slots
