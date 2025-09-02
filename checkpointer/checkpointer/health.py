"""Health checks and monitoring for checkpointer system.

This module handles connection health checks and monitoring
for the PostgreSQL checkpointer system.
"""
from __future__ import annotations

from checkpointer.checkpointer.factory import get_global_checkpointer
from checkpointer.database.pool_manager import force_close_modern_pools


# This file will contain:
# - check_pool_health_and_recreate() function
async def check_pool_health_and_recreate():
    """Check the health of the global connection pool and recreate if unhealthy."""
    global _GLOBAL_CHECKPOINTER
    try:
        pool = getattr(_GLOBAL_CHECKPOINTER, "pool", None)
        if pool is not None:
            # Try to acquire a connection and run a simple query
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;")
                    result = await cur.fetchone()
                    if result is None or result[0] != 1:
                        raise Exception("Pool health check failed: bad result")
            # If no exception, pool is healthy
            return True
        else:
            return False
    except Exception as e:
        # Pool is unhealthy, recreate it
        from api.utils.debug import print__checkpointers_debug

        print__checkpointers_debug(f"POOL HEALTH CHECK FAILED: {e}, recreating pool...")
    await force_close_modern_pools()
    await get_global_checkpointer()
    print__checkpointers_debug("POOL RECREATED after health check failure.")
    return False
