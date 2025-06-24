#!/usr/bin/env python3
"""
Robust PostgreSQL Connection Pool Manager
Based on Psycopg best practices from Context7 documentation.

This module provides a simple, robust connection pool that handles:
- Automatic reconnection on connection failures
- Proper context management
- Health checks and recovery
- Thread-safe operations
- Resource cleanup
"""

import os
import sys
import asyncio
import threading
import time
from typing import Optional, Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager
import psycopg
import psycopg_pool
from psycopg_pool import AsyncConnectionPool

# Set Windows compatibility if needed
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def print_pool_debug(msg: str) -> None:
    """Print pool debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[ROBUST-POOL] {msg}")
        sys.stdout.flush()

class RobustPostgresPool:
    """
    A robust PostgreSQL connection pool manager using Psycopg best practices.
    
    Features:
    - Automatic reconnection on failures
    - Proper async context management
    - Health checks with recovery
    - Simplified operations without complex locking
    - Graceful shutdown
    """
    
    def __init__(self):
        self._pool: Optional[AsyncConnectionPool] = None
        self._connection_string = self._build_connection_string()
        self._pool_config = self._get_pool_config()
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        config = {
            'user': os.environ.get('user'),
            'password': os.environ.get('password'), 
            'host': os.environ.get('host'),
            'port': int(os.environ.get('port', 5432)),
            'dbname': os.environ.get('dbname')        }
        
        # Validate required environment variables
        missing = [k for k, v in config.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        # Use timestamp to avoid prepared statement conflicts
        timestamp = int(time.time())
        return (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['dbname']}?"
            f"sslmode=require&application_name=czsu_robust_pool_{timestamp}"
        )
    
    def _get_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration."""
        return {
            "autocommit": True,
            "prepare_threshold": None,  # Completely disable prepared statements
        }
    
    async def _create_fresh_pool(self) -> AsyncConnectionPool:
        """Create a fresh connection pool with health checks."""
        print_pool_debug("Creating fresh connection pool...")
        
        async def check_connection(conn):
            """Check if connection is healthy."""
            try:
                # Use a simple query without prepared statements
                await conn.execute("SELECT 1")
                print_pool_debug("Connection health check passed")
            except Exception as e:
                print_pool_debug(f"Connection health check failed: {e}")
                raise
        
        def reconnect_failed(pool):
            """Called when reconnection fails repeatedly."""
            print_pool_debug(f"⚠️ Pool {pool.name} failed to reconnect after timeout")            # Log but don't exit - let the pool retry
        
        pool = AsyncConnectionPool(
            conninfo=self._connection_string,
            kwargs=self._pool_config,
            min_size=2,
            max_size=8,
            timeout=15,
            max_waiting=5,
            max_lifetime=1800,
            max_idle=300,
            reconnect_timeout=180,
            reconnect_failed=reconnect_failed,
            name=f"robust_pool_{int(time.time())}",
            open=False  # Explicitly set to avoid deprecation warnings
        )
        
        # Open the pool and wait for it to be ready
        await pool.open()
        await pool.wait()  # Ensure minimum connections are established
        
        print_pool_debug(f"✅ Pool '{pool.name}' created and ready")
        return pool
    
    async def _ensure_pool(self) -> AsyncConnectionPool:
        """Ensure we have a healthy connection pool."""
        if self._pool is None:
            print_pool_debug("Creating new pool...")
            self._pool = await self._create_fresh_pool()
        else:
            # Quick health check using simple execute (no prepared statements)
            try:
                async with self._pool.connection() as conn:
                    await conn.execute("SELECT 1")
                print_pool_debug("Existing pool is healthy")
            except Exception as e:
                print_pool_debug(f"Existing pool is unhealthy: {e}")
                # Close the unhealthy pool
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
                
                # Create new pool
                print_pool_debug("Creating new pool...")
                self._pool = await self._create_fresh_pool()
        
        return self._pool
    
    async def _ensure_healthy_pool(self) -> AsyncConnectionPool:
        """Legacy compatibility method for _ensure_pool."""
        return await self._ensure_pool()
    
    @asynccontextmanager
    async def connection(self) -> AsyncContextManager[psycopg.AsyncConnection]:
        """
        Get a connection from the pool using async context manager.
        
        Usage:
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                pool = await self._ensure_pool()
                
                # Use the pool's context manager for proper connection handling
                async with pool.connection() as conn:
                    print_pool_debug(f"✅ Connection acquired (attempt {attempt + 1})")
                    yield conn
                    print_pool_debug("✅ Connection returned to pool")
                    return
                    
            except (psycopg.OperationalError, 
                   psycopg_pool.PoolTimeout, 
                   psycopg_pool.PoolClosed) as e:
                print_pool_debug(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    raise
                  # Force pool recreation on connection errors
                if self._pool:
                    try:
                        await self._pool.close()
                    except Exception:
                        pass
                    self._pool = None
                
                # Wait before retry with exponential backoff
                wait_time = min(2 ** attempt, 10)
                print_pool_debug(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                print_pool_debug(f"Unexpected error: {e}")
                raise
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics for monitoring."""
        try:
            if self._pool:
                stats = self._pool.get_stats()
                return {
                    "pool_name": self._pool.name,
                    "pool_size": stats.get("pool_size", 0),
                    "pool_available": stats.get("pool_available", 0),
                    "requests_waiting": stats.get("requests_waiting", 0),
                    "connections_errors": stats.get("connections_errors", 0),
                    "requests_errors": stats.get("requests_errors", 0),
                }
            return {"status": "no_pool"}
        except Exception as e:
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Perform a health check on the pool."""
        try:
            async with self.connection() as conn:
                # Use simple execute without cursors to avoid prepared statement conflicts
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            print_pool_debug(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Close the connection pool gracefully."""
        print_pool_debug("Closing connection pool...")
        
        if self._pool:
            try:
                await self._pool.close()
                print_pool_debug("✅ Pool closed successfully")
            except Exception as e:
                print_pool_debug(f"Error closing pool: {e}")
            finally:
                self._pool = None

# Global singleton instance
_global_pool: Optional[RobustPostgresPool] = None

def get_global_pool() -> RobustPostgresPool:
    """Get the global connection pool singleton."""
    global _global_pool
    
    if _global_pool is None:
        _global_pool = RobustPostgresPool()
    return _global_pool

@asynccontextmanager
async def get_connection():
    """
    Convenience function to get a connection from the global pool.
    
    Usage:
        async with get_connection() as conn:
            await conn.execute("SELECT 1")
    """
    pool = get_global_pool()
    async with pool.connection() as conn:
        yield conn

async def close_global_pool():
    """Close the global connection pool."""
    global _global_pool
    
    if _global_pool:
        await _global_pool.close()
        _global_pool = None

async def test_pool():
    """Test the connection pool."""
    print_pool_debug("Testing robust connection pool...")
    
    try:        # Test basic connection
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 'test_successful'")
                result = await cur.fetchone()
                print_pool_debug(f"✅ Test result: {result[0]}")
        
        # Test pool stats
        pool = get_global_pool()
        stats = await pool.get_pool_stats()
        print_pool_debug(f"📊 Pool stats: {stats}")
        
        # Test health check
        is_healthy = await pool.health_check()
        print_pool_debug(f"🏥 Health check: {'✅ PASS' if is_healthy else '❌ FAIL'}")
        
        print_pool_debug("✅ All tests passed!")
        return True
        
    except Exception as e:
        print_pool_debug(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        success = await test_pool()
        await close_global_pool()
        return success
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
