"""PostgreSQL Checkpointer for LangGraph Multi-Agent Text-to-SQL System.

Main entry point with public API for the refactored PostgreSQL checkpointer system.
This module provides comprehensive PostgreSQL-based checkpointing functionality
for the CZSU Multi-Agent Text-to-SQL system using LangGraph's AsyncPostgresSaver.

This package is organized into the following modules:
- config: Configuration constants and environment handling
- globals: Global state management and type definitions
- database: Connection management, pool handling, table setup
- error_handling: Prepared statement cleanup and retry logic
- checkpointer: Checkpointer lifecycle and health monitoring
- user_management: Thread operations and sentiment tracking
"""

# This file will contain the main public API imports and
# Windows event loop policy configuration
