"""Health checks and monitoring for checkpointer system.

This module handles connection health checks and monitoring
for the PostgreSQL checkpointer system.
"""

from __future__ import annotations


# Re-export the health check function from factory to maintain compatibility
def check_pool_health_and_recreate():
    """Re-export check_pool_health_and_recreate from factory module."""
    from checkpointer.checkpointer.factory import (
        check_pool_health_and_recreate as _check_pool_health_and_recreate,
    )

    return _check_pool_health_and_recreate()
