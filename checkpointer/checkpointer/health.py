"""Connection Health Monitoring and Pool Health Check Re-export Module

This module provides health check functionality for the PostgreSQL checkpointer system
by re-exporting the check_pool_health_and_recreate function from the factory module.
It serves as a compatibility layer and logical grouping for health-related operations.
"""

from __future__ import annotations


# ==============================================================================
# MODULE OVERVIEW
# ==============================================================================
# This module acts as a thin re-export layer for connection pool health monitoring
# functionality. Its primary purpose is to provide a logical import path for
# health-related operations while maintaining backward compatibility.
#
# The actual health check implementation resides in factory.py, and this module
# simply re-exports that functionality for organizational purposes.
# ==============================================================================

MODULE_DESCRIPTION = r"""Connection Health Monitoring and Pool Health Check Re-export Module

This module acts as a thin compatibility and organizational layer for health monitoring
functionality in the checkpointer system. It re-exports the check_pool_health_and_recreate
function from the factory module, providing a logical import path for health-related
operations.

Key Features:
-------------
1. Function Re-export:
   - Re-exports check_pool_health_and_recreate from factory module
   - Provides backward compatibility for existing imports
   - Maintains clean separation of concerns
   - Allows health-specific import paths

2. Compatibility Layer:
   - Preserves existing import patterns
   - Allows migration without breaking changes
   - Provides logical grouping of health functionality
   - Enables future health-specific extensions

Architecture:
-----------
The module uses a simple re-export pattern:

1. Import Pattern:
   - Imports check_pool_health_and_recreate from factory module
   - Re-exports as module-level function
   - Maintains same function signature and behavior

2. Purpose:
   - Organizational: Groups health-related functionality
   - Compatibility: Preserves existing import paths
   - Extensibility: Allows future health-specific additions

Re-exported Functions:
-------------------
1. check_pool_health_and_recreate():
   - Validates connection pool health
   - Executes test query ("SELECT 1")
   - Detects SSL, connection, and timeout errors
   - Automatically recreates pool on failures
   - Returns True if healthy, False if recreated
   
For detailed documentation of this function, see:
    checkpointer.checkpointer.factory.check_pool_health_and_recreate

Usage Example:
-------------
# Import from health module (re-export):
from checkpointer.checkpointer.health import check_pool_health_and_recreate

# Or import directly from factory (original):
from checkpointer.checkpointer.factory import check_pool_health_and_recreate

# Both imports provide the same function
await check_pool_health_and_recreate()

Benefits of Re-export Pattern:
----------------------------
1. Logical Organization:
   - Health-related code grouped together
   - Clear module purpose and responsibility
   - Easier to find health-specific functionality

2. Backward Compatibility:
   - Existing imports continue to work
   - No breaking changes during refactoring
   - Gradual migration possible

3. Future Extensibility:
   - Can add more health-specific functions here
   - Centralized location for health monitoring
   - Potential for health metrics and reporting

Future Enhancements:
------------------
- Add connection pool statistics export
- Implement health check scheduling/automation
- Add health metrics collection and reporting
- Implement health check result caching
- Add alerting for repeated health check failures
- Implement circuit breaker pattern for health checks

Notes:
-----
- This is a thin wrapper/re-export module
- Actual implementation is in factory.py
- Function behavior is identical to factory version
- Provides cleaner import semantics for health operations
- Maintains single source of truth in factory module

Dependencies:
-----------
- checkpointer.checkpointer.factory: Source of re-exported function
"""


# ==============================================================================
# HEALTH CHECK FUNCTION RE-EXPORT
# ==============================================================================
# This function is re-exported from the factory module to provide a logical
# import path for health-related operations. The actual implementation is in
# checkpointer.checkpointer.factory.check_pool_health_and_recreate.


def check_pool_health_and_recreate():
    """Re-export check_pool_health_and_recreate from factory module.

    This function re-exports the check_pool_health_and_recreate function from
    the factory module, providing a compatibility layer and logical import path
    for health monitoring operations.

    The function performs connection pool health checks by:
    1. Acquiring a connection from the pool (10s timeout)
    2. Executing a simple test query ("SELECT 1")
    3. Validating the query result
    4. On failure: detecting error type (SSL, connection, timeout)
    5. On failure: forcing pool closure and recreation
    6. Returning health status (True = healthy, False = recreated)

    Returns:
        Coroutine: Returns the coroutine from the factory module's function.
                  When awaited, returns bool:
                  - True if pool is healthy
                  - False if pool was recreated

    For complete documentation, see:
        checkpointer.checkpointer.factory.check_pool_health_and_recreate

    Usage:
        from checkpointer.checkpointer.health import check_pool_health_and_recreate

        is_healthy = await check_pool_health_and_recreate()
        if is_healthy:
            print("Pool is healthy")
        else:
            print("Pool was recreated")

    Note:
        - This is a re-export for organizational purposes
        - Actual implementation is in factory.py
        - Function behavior is identical to factory version
        - Provides cleaner import semantics for health operations
    """
    # ==============================================================================
    # IMPORT AND DELEGATE
    # ==============================================================================
    # Import the actual implementation from factory module and return it directly.
    # This import is done inside the function to avoid circular import issues that
    # could arise from factory.py potentially importing from this module.
    # The function simply delegates to the factory implementation without any
    # additional processing or wrapping.
    # ==============================================================================
    from checkpointer.checkpointer.factory import (
        check_pool_health_and_recreate as _check_pool_health_and_recreate,
    )

    # Return the coroutine directly - caller will await it
    return _check_pool_health_and_recreate()
