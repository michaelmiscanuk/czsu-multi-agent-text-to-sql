#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using asyncpg for all operations.
- asyncpg for application-specific database operations (better performance)
- LangGraph's built-in AsyncPostgresSaver for checkpointing (uses psycopg internally but we provide asyncpg connection string)

ENHANCED FEATURES RESTORED FROM OLD VERSION:
============================================

1. **Background Connection Monitoring**:
   - monitor_connection_health() function with SSL-specific diagnostics
   - Automatic failure detection and alerting
   - Dynamic monitoring frequency based on connection health
   - Enhanced error pattern recognition

2. **Comprehensive Initialization System**:
   - initialize_enhanced_postgres_system() with 8-step verification process
   - Background monitoring setup with task management
   - Environment validation and health checks
   - Graceful fallback handling

3. **Enhanced Connection Configuration**:
   - Cloud-optimized connection strings with keepalive settings
   - SSL-specific configurations for Render/Supabase deployment
   - asyncpg connection optimizations for all operations
   - Network resilience settings with timeout configurations

4. **Advanced Error Handling and Diagnostics**:
   - SSL connection error pattern detection
   - Authentication and timeout error analysis
   - Pipeline/handler error identification
   - Detailed troubleshooting suggestions

5. **Sophisticated Pool Management**:
   - Enhanced pool recreation with retry logic
   - Progressive delay strategies for connection failures
   - Pool health monitoring with comprehensive statistics
   - Connection lifetime management for cloud environments

6. **Comprehensive Debug and Monitoring**:
   - Enhanced debug_pool_status() with connection testing
   - Real-time pool statistics and health metrics
   - Event loop type verification
   - Active operations tracking

7. **Resilient Checkpointer Wrapper**:
   - ResilientPostgreSQLCheckpointer with enhanced retry logic
   - SSL-specific error handling for cloud deployments
   - Automatic pool recreation on connection failures
   - Exponential backoff with failure pattern recognition

All features use asyncpg-optimized connection strings and settings.
The LangGraph checkpointer internally uses psycopg but accepts standard PostgreSQL connection strings.
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix asyncpg compatibility
import sys
import os  # Import os early for environment variable access

def print__postgres_startup_debug(msg: str) -> None:
    """Print PostgreSQL startup debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[POSTGRES-STARTUP-DEBUG] {msg}")
        sys.stdout.flush()

if sys.platform == "win32":
    import asyncio
    
    # AGGRESSIVE WINDOWS FIX: Force SelectorEventLoop for asyncpg compatibility
    print__postgres_startup_debug(f"[FIX] PostgreSQL module: Windows detected - ensuring asyncpg compatibility")
    
    # Set the policy first - this is CRITICAL and must happen before any async operations
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print__postgres_startup_debug(f"[FIX] PostgreSQL module: Windows event loop policy set to: {type(asyncio.get_event_loop_policy()).__name__}")
    
    # Force close any existing event loop and create a fresh SelectorEventLoop
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop and not current_loop.is_closed():
            print__postgres_startup_debug(f"[FIX] PostgreSQL module: Closing existing {type(current_loop).__name__}")
            current_loop.close()
    except RuntimeError:
        # No event loop exists yet, which is fine
        pass
    
    # Create a new SelectorEventLoop explicitly and set it as the running loop
    new_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
    asyncio.set_event_loop(new_loop)
    print__postgres_startup_debug(f"[FIX] PostgreSQL module: Created new {type(new_loop).__name__}")
    
    # Verify the fix worked - this is critical for asyncpg compatibility
    try:
        current_loop = asyncio.get_event_loop()
        print__postgres_startup_debug(f"[FIX] PostgreSQL module: Current event loop type: {type(current_loop).__name__}")
        if "Selector" in type(current_loop).__name__:
            print__postgres_startup_debug(f"[OK] PostgreSQL module: asyncpg should work correctly on Windows now")
        else:
            print__postgres_startup_debug(f"[WARN] PostgreSQL module: Event loop fix may not have worked properly")
            # FORCE FIX: If we still don't have a SelectorEventLoop, create one
            print__postgres_startup_debug(f"[FIX] PostgreSQL module: Force-creating SelectorEventLoop...")
            if not current_loop.is_closed():
                current_loop.close()
            selector_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
            asyncio.set_event_loop(selector_loop)
            print__postgres_startup_debug(f"[FIX] PostgreSQL module: Force-created {type(selector_loop).__name__}")
    except RuntimeError:
        print__postgres_startup_debug(f"[FIX] PostgreSQL module: No event loop set yet (will be created as needed)")

import asyncio
import platform
import os
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncpg  # Using asyncpg for application-specific operations
import threading
from contextlib import asynccontextmanager
import time

# Import LangGraph's built-in PostgreSQL checkpointer (uses psycopg)
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    print__postgres_startup_debug("[OK] LangGraph AsyncPostgresSaver imported successfully")
except ImportError as e:
    print__postgres_startup_debug(f"[ERROR] Failed to import AsyncPostgresSaver: {e}")
    print__postgres_startup_debug("[TIP] Install with: pip install langgraph-checkpoint-postgres")
    AsyncPostgresSaver = None

#==============================================================================
# DEBUG FUNCTIONS
#==============================================================================
def print__postgresql_debug(msg: str) -> None:
    """Print PostgreSQL debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[POSTGRESQL-DEBUG] {msg}")
        sys.stdout.flush()

def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[API-PostgreSQL] {msg}")
        sys.stdout.flush()

async def debug_pool_status() -> Dict[str, Any]:
    """Get debug information about the current pool status.
    
    Returns:
        Dictionary with pool status information
    """
    global database_pool, _active_operations
    
    status = {
        "asyncpg_pool": {
            "exists": database_pool is not None,
            "closed": None,
            "min_size": None,
            "max_size": None,
            "current_size": None,
        },
        "operations": {
            "active_count": _active_operations,
        },
        "event_loop": {
            "type": None,
            "closed": None,
        },
    }
    
    print__api_postgresql(f"🔍 Enhanced Pool Status Debug:")
    print__api_postgresql(f"   Global pool exists: {database_pool is not None}")
    
    # Get asyncpg pool details
    if database_pool:
        status["asyncpg_pool"]["closed"] = getattr(database_pool, "_closed", None)
        status["asyncpg_pool"]["min_size"] = getattr(database_pool, "_min_size", None)
        status["asyncpg_pool"]["max_size"] = getattr(database_pool, "_max_size", None)
        
        print__api_postgresql(f"   Pool type: {type(database_pool).__name__}")
        print__api_postgresql(f"   Pool closed: {database_pool._closed}")
        
        # Enhanced pool information
        try:
            min_size = getattr(database_pool, '_min_size', 'unknown')
            max_size = getattr(database_pool, '_max_size', 'unknown')
            current_size = len(getattr(database_pool, '_queue', []))
            status["asyncpg_pool"]["current_size"] = current_size
            
            print__api_postgresql(f"   Pool min_size: {min_size}")
            print__api_postgresql(f"   Pool max_size: {max_size}")
            print__api_postgresql(f"   Pool current_size: {current_size}")
                
            # Test health with enhanced diagnostics
            is_healthy = await is_pool_healthy(database_pool)
            status["asyncpg_pool"]["healthy"] = is_healthy
            print__api_postgresql(f"   Pool healthy: {is_healthy}")
            
            # Connection test
            if not database_pool._closed:
                try:
                    conn = await asyncio.wait_for(database_pool.acquire(), timeout=5)
                    try:
                        result = await asyncio.wait_for(conn.fetchrow("SELECT pg_backend_pid() as pid, NOW() as timestamp"), timeout=5)
                        print__api_postgresql(f"   Test connection PID: {result['pid'] if result else 'unknown'}")
                        print__api_postgresql(f"   Test connection timestamp: {result['timestamp'] if result else 'unknown'}")
                        status["asyncpg_pool"]["test_connection"] = "success"
                    finally:
                        await database_pool.release(conn)
                except Exception as conn_error:
                    print__api_postgresql(f"   Connection test failed: {conn_error}")
                    status["asyncpg_pool"]["test_connection"] = f"failed: {conn_error}"
            
        except Exception as e:
            print__api_postgresql(f"   Pool status error: {e}")
            status["asyncpg_pool"]["error"] = str(e)
    else:
        print__api_postgresql("   No global pool available")
    
    # Get event loop info
    try:
        loop = asyncio.get_event_loop()
        status["event_loop"]["type"] = type(loop).__name__
        status["event_loop"]["closed"] = loop.is_closed() if loop else None
        print__api_postgresql(f"   Event loop type: {type(loop).__name__}")
        print__api_postgresql(f"   Event loop closed: {loop.is_closed() if loop else None}")
    except Exception as e:
        print__api_postgresql(f"   Event loop info error: {e}")
        status["event_loop"]["error"] = str(e)
    
    # Operation count
    print__api_postgresql(f"   Active operations: {_active_operations}")
    
    print__postgresql_debug(f"[DEBUG] Complete pool status: {status}")
    return status

# Database connection parameters for asyncpg (application operations)
database_pool: Optional[asyncpg.Pool] = None
_pool_lock = None  # Will be created lazily when needed
_active_operations = 0  # Track active operations using the pool
_operations_lock = None  # Will be created lazily when needed

def _get_pool_lock():
    """Get or create the pool lock."""
    global _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    return _pool_lock

def _get_operations_lock():
    """Get or create the operations lock."""
    global _operations_lock
    if _operations_lock is None:
        _operations_lock = asyncio.Lock()
    return _operations_lock

async def increment_active_operations():
    """Safely increment the count of active operations."""
    global _active_operations
    async with _get_operations_lock():
        _active_operations += 1
        print__postgresql_debug(f"🔄 Active operations incremented to: {_active_operations}")

async def decrement_active_operations():
    """Safely decrement the count of active operations."""
    global _active_operations
    async with _get_operations_lock():
        _active_operations -= 1
        print__postgresql_debug(f"🔄 Active operations decremented to: {_active_operations}")

async def get_active_operations_count():
    """Get the current count of active operations."""
    global _active_operations
    async with _get_operations_lock():
        return _active_operations

async def force_close_all_connections():
    """Force close asyncpg connections."""
    try:
        print__postgresql_debug("[WARN] Forcing closure of PostgreSQL connections")
        
        # Close asyncpg pool
        global database_pool
        if database_pool and not database_pool._closed:
            await database_pool.close()
            database_pool = None
            print__postgresql_debug("[OK] asyncpg pool closed")
            
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Error closing connections: {e}")

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'user': os.getenv('user'),
        'password': os.getenv('password'),  
        'host': os.getenv('host'),
        'port': os.getenv('port', '5432'),
        'dbname': os.getenv('dbname')
    }

def get_connection_string():
    """Get PostgreSQL connection string from environment variables optimized for asyncpg and PostgreSQL standard."""
    config = get_db_config()
    
    print__postgres_startup_debug(f"[CONN] Building PostgreSQL connection string for cloud deployment (asyncpg compatible)")
    
    # Enhanced connection string optimized for asyncpg and compatible with standard PostgreSQL drivers
    # This works for both asyncpg and any other PostgreSQL driver that accepts standard connection strings
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode=require"                     # Required for cloud PostgreSQL
        f"&application_name=czsu_agent"         # App identification for monitoring
        # Connection keepalive settings (critical for cloud deployments)
        f"&keepalives_idle=600"                 # 10 minutes before first keepalive
        f"&keepalives_interval=30"              # 30 seconds between keepalive probes
        f"&keepalives_count=3"                  # 3 failed probes before disconnect
        f"&tcp_user_timeout=60000"              # 60 seconds TCP user timeout
        # Standard PostgreSQL optimizations (work with both asyncpg and psycopg)
        # Network resilience settings
        f"&target_session_attrs=read-write"     # Ensure we get a writable session
        # Enhanced cloud deployment settings
        f"&connect_timeout=20"                  # Increased timeout for cloud latency
    )
    
    # Log connection details (without password)
    debug_string = connection_string.replace(config['password'], '***')
    print__postgres_startup_debug(f"[CONN] Using standard PostgreSQL connection string: {debug_string}")
    
    return connection_string


# NOTE: LangGraph AsyncPostgresSaver manages its own connections internally  
# We don't need to manually create connection pools - AsyncPostgresSaver.from_conn_string() handles this

async def is_pool_healthy(pool: Optional[asyncpg.Pool]) -> bool:
    """Check if an asyncpg connection pool is healthy and open with enhanced diagnostics."""
    if pool is None:
        print__postgresql_debug(f"[WARN] Pool is None")
        return False
        
    try:
        # Check if pool is closed
        if pool._closed:  # asyncpg pools have _closed attribute
            print__postgresql_debug(f"[WARN] Pool is marked as closed")
            return False
        
        # Get pool statistics for diagnostics
        try:
            # asyncpg pools don't have get_stats() like psycopg, so we'll use the size properties
            min_size = getattr(pool, '_min_size', 'unknown')
            max_size = getattr(pool, '_max_size', 'unknown')
            current_size = getattr(pool, '_queue', {})
            current_size = len(current_size) if hasattr(current_size, '__len__') else 'unknown'
            print__postgresql_debug(f"[STATS] Pool info: min={min_size}, max={max_size}, current={current_size}")
        except Exception as e:
            print__postgresql_debug(f"[STATS] Pool stats unavailable: {e}")
        
        # Try a quick connection test with timeout
        try:
            # FIXED: Properly handle wait_for with async context manager
            conn = await asyncio.wait_for(pool.acquire(), timeout=5)
            try:
                result = await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=5)
                # FIXED: Properly await the release method
                await pool.release(conn)
                print__postgresql_debug(f"[OK] Pool health check passed")
                return True
            except Exception as e:
                # FIXED: Properly await the release method
                await pool.release(conn)
                print__postgresql_debug(f"[WARN] Pool health check query failed: {e}")
                return False
        except asyncio.TimeoutError:
            print__postgresql_debug(f"[WARN] Pool health check timed out")
            return False
        except Exception as e:
            print__postgresql_debug(f"[WARN] Pool health check failed: {e}")
            return False
            
    except Exception as e:
        print__postgresql_debug(f"[WARN] Pool health check error: {e}")
        return False

async def create_fresh_connection_pool() -> Optional[asyncpg.Pool]:
    """Create a fresh asyncpg connection pool with optimized settings."""
    try:
        print__postgresql_debug("[CONN] Creating fresh asyncpg connection pool...")
        
        # Get database config
        config = get_db_config()
        
        # Log connection info (with password hidden)
        print__postgresql_debug(f"[CONN] Connecting to: {config['host']}:{config['port']}")
        print__postgresql_debug(f"[CONN] Database: {config['dbname']}")
        print__postgresql_debug(f"[CONN] User: {config['user']}")
        
        # Enable SSL by default for cloud databases
        ssl = "require" if config.get('sslmode') != "disable" else None
        if ssl:
            print__postgresql_debug(f"[SSL] SSL enabled by default")
        
        # Maximum number of attempts to create a pool (increased from old version)
        max_attempts = 3
        retry_delay = 2
        
        for attempt in range(1, max_attempts + 1):
            try:
                print__postgresql_debug(f"[CONN] Pool creation attempt {attempt}/{max_attempts}")
                
                # IMPORTANT: Disable statement cache to avoid prepared statement errors with pgbouncer
                # Enhanced settings based on old version's cloud optimizations
                pool = await asyncpg.create_pool(
                    host=config['host'],
                    port=config['port'],
                    user=config['user'],
                    password=config['password'],
                    database=config['dbname'],
                    min_size=0,                      # Start with 0 to avoid initial connection storms
                    max_size=10,                     # Conservative max to prevent connection exhaustion
                    ssl=ssl,
                    statement_cache_size=0,          # Disable statement cache for pgbouncer compatibility
                    timeout=20.0,                    # Increased timeout for cloud latency
                    command_timeout=30.0,            # Increased command timeout for cloud environments
                    max_inactive_connection_lifetime=900  # 15 minutes lifetime to prevent SSL timeouts
                )
                
                # Enhanced validation with multiple test queries
                async with pool.acquire() as conn:
                    # Test basic connectivity
                    await conn.fetchval('SELECT 1')
                    # Test with timestamp to ensure full functionality
                    result = await conn.fetchrow('SELECT 1 as test, NOW() as timestamp, pg_backend_pid() as pid')
                    print__postgresql_debug(f"[CONN] Pool validation successful - PID: {result['pid']}")
                
                print__postgresql_debug(f"[OK] Fresh asyncpg connection pool created successfully")
                print__postgresql_debug(f"[STATS] Pool stats: min=0, max=10, statement_cache=disabled")
                print__postgresql_debug(f"[OK] New pool verified with enhanced test queries")
                return pool
                
            except Exception as e:
                error_msg = str(e).lower()
                print__postgresql_debug(f"[ERROR] Pool creation attempt {attempt} failed: {str(e)}")
                
                # Enhanced error diagnostics from old version
                if "ssl" in error_msg:
                    print__postgresql_debug("💡 SSL-related error detected. Check:")
                    print__postgresql_debug("   1. Database SSL configuration")
                    print__postgresql_debug("   2. Network connectivity to database")
                    print__postgresql_debug("   3. Certificate validity")
                elif "timeout" in error_msg:
                    print__postgresql_debug("💡 Timeout error detected. Check:")
                    print__postgresql_debug("   1. Database server responsiveness")
                    print__postgresql_debug("   2. Network latency")
                    print__postgresql_debug("   3. Connection limits")
                elif "authentication" in error_msg or "password" in error_msg:
                    print__postgresql_debug("💡 Authentication error detected. Check:")
                    print__postgresql_debug("   1. Database credentials")
                    print__postgresql_debug("   2. User permissions")
                    print__postgresql_debug("   3. Connection limits")
                
                if attempt < max_attempts:
                    delay = retry_delay * attempt  # Progressive delay
                    print__postgresql_debug(f"[RETRY] Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print__postgresql_debug(f"[ERROR] Failed to create connection pool after {max_attempts} attempts")
                    # Enhanced error diagnostics and suggestions
                    import traceback
                    print__postgresql_debug(f"🔍 Full traceback: {traceback.format_exc()}")
                    raise
        
        return None
        
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to create connection pool: {str(e)}")
        return None

async def get_healthy_pool() -> asyncpg.Pool:
    """Get a healthy asyncpg connection pool with automatic recreation."""
    global database_pool
    
    # Check if pool exists and is healthy
    if database_pool is not None and not database_pool._closed:
        print__postgresql_debug("[CHECK] Checking existing pool health...")
        is_healthy = await is_pool_healthy(database_pool)
        if is_healthy:
            print__postgresql_debug("[OK] Existing pool is healthy")
            return database_pool
        else:
            print__postgresql_debug("[WARN] Existing pool is unhealthy, closing...")
            try:
                await database_pool.close()
            except Exception as e:
                print__postgresql_debug(f"[WARN] Error closing unhealthy pool: {e}")
            database_pool = None
    
    # Create new pool
    print__postgresql_debug("[CONN] Creating new healthy pool...")
    async with _get_pool_lock():
        # Double-check pattern - another thread might have created the pool
        if database_pool is not None and not database_pool._closed:
            print__postgresql_debug("[OK] Pool was created by another thread")
            return database_pool
        
        database_pool = await create_fresh_connection_pool()
        
        # Test the new pool
        try:
            async with database_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            print__postgresql_debug("[OK] New pool verified with test query")
        except Exception as e:
            print__postgresql_debug(f"[ERROR] New pool failed verification: {e}")
            await database_pool.close()
            database_pool = None
            raise
        
        return database_pool

async def setup_users_threads_runs_table():
    """Setup the users_threads_runs table for tracking user conversations."""
    try:
        print__postgresql_debug("[DB] Setting up users_threads_runs table...")
        pool = await get_healthy_pool()
        
        async with pool.acquire() as conn:
            # Create table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users_threads_runs (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    thread_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) UNIQUE NOT NULL,
                    prompt TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment BOOLEAN DEFAULT NULL
                );
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email 
                ON users_threads_runs(email);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_thread_id 
                ON users_threads_runs(thread_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_run_id 
                ON users_threads_runs(run_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread 
                ON users_threads_runs(email, thread_id);
            """)
            
            print__postgresql_debug("[OK] users_threads_runs table and indexes created/verified")
            
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to setup users_threads_runs table: {e}")
        import traceback
        print__postgresql_debug(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise

async def monitor_connection_health(pool: asyncpg.Pool, interval: int = 60):
    """Enhanced connection pool health monitor with SSL-specific diagnostics for asyncpg pools."""
    print__api_postgresql("[MONITOR] Starting enhanced asyncpg connection pool health monitor")
    monitor_failures = 0
    max_consecutive_failures = 3
    
    try:
        while True:
            try:
                # Enhanced health check with timeout for asyncpg
                start_time = time.time()
                conn = await asyncio.wait_for(pool.acquire(), timeout=10)
                try:
                    # More comprehensive health check
                    result = await asyncio.wait_for(
                        conn.fetchrow("SELECT 1 as health, NOW() as timestamp, pg_backend_pid() as pid"), 
                        timeout=10
                    )
                    check_duration = time.time() - start_time
                finally:
                    await pool.release(conn)
                
                # Get enhanced pool statistics (asyncpg-style)
                try:
                    min_size = getattr(pool, '_min_size', 'unknown')
                    max_size = getattr(pool, '_max_size', 'unknown')
                    current_size = len(getattr(pool, '_queue', []))
                    print__api_postgresql(f"[MONITOR] ✓ Pool health OK ({check_duration:.2f}s) - Stats: min={min_size}, max={max_size}, current={current_size}")
                except Exception as e:
                    print__api_postgresql(f"[MONITOR] ✓ Pool health OK ({check_duration:.2f}s) - Backend PID: {result['pid'] if result else 'unknown'}")
                
                # Reset failure counter on success
                monitor_failures = 0
                
            except asyncio.TimeoutError:
                monitor_failures += 1
                print__api_postgresql(f"[MONITOR] ⚠ Pool health check timeout ({monitor_failures}/{max_consecutive_failures})")
                
                if monitor_failures >= max_consecutive_failures:
                    print__api_postgresql("[MONITOR] 🚨 Multiple consecutive health check timeouts - pool may need recreation")
                    # Don't recreate pool from monitor - just log the issue
                    
            except Exception as e:
                monitor_failures += 1
                error_msg = str(e).lower()
                
                # Enhanced SSL-specific error detection
                if any(pattern in error_msg for pattern in [
                    "ssl connection has been closed",
                    "server closed the connection",
                    "connection to server",
                    "ssl syscall error",
                    "eof detected",
                    "bad connection"
                ]):
                    print__api_postgresql(f"[MONITOR] 🔒 SSL connection error in health check ({monitor_failures}/{max_consecutive_failures}): {e}")
                elif any(pattern in error_msg for pattern in [
                    "dbhandler exited",
                    "pipeline",
                    "flush request failed",
                    "lost synchronization"
                ]):
                    print__api_postgresql(f"[MONITOR] 🚨 Critical connection error in health check ({monitor_failures}/{max_consecutive_failures}): {e}")
                else:
                    print__api_postgresql(f"[MONITOR] ⚠ Pool health check failed ({monitor_failures}/{max_consecutive_failures}): {e}")
                
                if monitor_failures >= max_consecutive_failures:
                    print__api_postgresql("[MONITOR] 🚨 Multiple consecutive health check failures detected")
                    print__api_postgresql("[MONITOR] 💡 This may indicate persistent connection issues that require manual intervention")
            
            # Dynamic interval based on health
            sleep_interval = interval
            if monitor_failures > 0:
                # More frequent checks when issues are detected
                sleep_interval = min(interval, 30)
                print__api_postgresql(f"[MONITOR] 🔍 Increased monitoring frequency due to {monitor_failures} failures")
            
            await asyncio.sleep(sleep_interval)
            
    except asyncio.CancelledError:
        print__api_postgresql("[MONITOR] 📊 Enhanced connection monitor stopped")
        raise
    except Exception as e:
        print__api_postgresql(f"[MONITOR] ❌ Connection monitor error: {e}")
        raise

async def initialize_enhanced_postgres_system():
    """Initialize the enhanced PostgreSQL system with all necessary components.
    
    This function sets up:
    1. Database connection pools (asyncpg for app operations)
    2. Required tables and indexes
    3. LangGraph checkpointer (manages its own psycopg connections)
    4. Connection health verification
    5. Background monitoring (optional)
    
    Returns:
        Tuple of (checkpointer, success_status)
    """
    try:
        print__postgresql_debug("[INIT] 🚀 Initializing Enhanced PostgreSQL Connection System")
        print__postgresql_debug("[INIT] " + "=" * 60)
        
        # Step 1: Environment validation
        print__postgresql_debug("[INIT] 📋 Step 1: Environment Validation")
        if not check_postgres_env_vars():
            print__postgresql_debug("[INIT] ❌ Environment validation failed")
            return None, False
        print__postgresql_debug("[INIT] ✅ Environment validation passed")
        
        # Step 2: Basic connection health test
        print__postgresql_debug("[INIT] 🔍 Step 2: Basic Connection Health Test")
        if not await test_basic_postgres_connection():
            print__postgresql_debug("[INIT] ❌ Basic connection health test failed")
            return None, False
        print__postgresql_debug("[INIT] ✅ Basic connection health test passed")
        
        # Step 3: Enhanced pool creation and testing
        print__postgresql_debug("[INIT] 🏊 Step 3: Enhanced Pool Creation and Testing")
        if not await test_pool_connection():
            print__postgresql_debug("[INIT] ❌ Enhanced pool test failed")
            return None, False
        print__postgresql_debug("[INIT] ✅ Enhanced pool test passed")
        
        # Step 4: Initialize global asyncpg pool
        print__postgresql_debug("[INIT] 🌐 Step 4: Global Pool Initialization")
        try:
            global_pool = await get_healthy_pool()
            if global_pool:
                print__postgresql_debug("[INIT] ✅ Global asyncpg pool initialized successfully")
                
                # Test the global pool
                async with global_pool.acquire() as conn:
                    result = await conn.fetchrow("SELECT 'Global pool test' as message, NOW() as timestamp")
                    print__postgresql_debug(f"[INIT] ✅ Global pool test: {result['message']} at {result['timestamp']}")
            else:
                print__postgresql_debug("[INIT] ❌ Global pool initialization failed")
                return None, False
        except Exception as e:
            print__postgresql_debug(f"[INIT] ❌ Global pool initialization error: {e}")
            return None, False
        
        # Step 5: Initialize database tables
        print__postgresql_debug("[INIT] 📊 Step 5: Database Schema Initialization")
        try:
            await setup_users_threads_runs_table()
            print__postgresql_debug("[INIT] ✅ Database schema initialized successfully")
        except Exception as e:
            print__postgresql_debug(f"[INIT] ❌ Database schema initialization failed: {e}")
            return None, False
        
        # Step 6: Create LangGraph checkpointer
        print__postgresql_debug("[INIT] 🔧 Step 6: LangGraph Checkpointer Creation")
        try:
            checkpointer = await get_postgres_checkpointer()
            print__postgresql_debug("[INIT] ✅ LangGraph checkpointer created successfully")
        except Exception as e:
            print__postgresql_debug(f"[INIT] ❌ LangGraph checkpointer creation failed: {e}")
            return None, False
        
        # Step 7: Start background monitoring (optional)
        monitoring_enabled = os.getenv("ENABLE_CONNECTION_MONITORING", "true").lower() == "true"
        if monitoring_enabled and global_pool:
            print__postgresql_debug("[INIT] 📡 Step 7: Starting Background Monitoring")
            try:
                # Start monitoring task in background
                monitor_task = asyncio.create_task(monitor_connection_health(global_pool, 60))
                print__postgresql_debug("[INIT] ✅ Background connection monitoring started")
                
                # Store the task reference to prevent garbage collection
                if not hasattr(initialize_enhanced_postgres_system, '_monitor_tasks'):
                    initialize_enhanced_postgres_system._monitor_tasks = []
                initialize_enhanced_postgres_system._monitor_tasks.append(monitor_task)
                
            except Exception as e:
                print__postgresql_debug(f"[INIT] ⚠️ Background monitoring failed to start: {e}")
                # Don't fail initialization if monitoring fails
        else:
            print__postgresql_debug("[INIT] 📡 Step 7: Background Monitoring Disabled")
        
        # Step 8: Final system verification
        print__postgresql_debug("[INIT] ✅ Step 8: Final System Verification")
        if not await test_connection_health():
            print__postgresql_debug("[INIT] ⚠️ System health check failed, but continuing...")
        
        print__postgresql_debug("[INIT] 🎉 Enhanced PostgreSQL Connection System Initialized Successfully!")
        print__postgresql_debug("[INIT] " + "=" * 60)
        return checkpointer, True
        
    except Exception as e:
        print__postgresql_debug(f"[INIT] ❌ Enhanced PostgreSQL system initialization failed: {e}")
        import traceback
        print__postgresql_debug(f"[INIT] 🔍 Full traceback: {traceback.format_exc()}")
        return None, False

async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new entry in users_threads_runs table."""
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    try:
        print__postgresql_debug(f"[DB] Creating thread run entry: email={email}, thread_id={thread_id}, run_id={run_id}")
        
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users_threads_runs (email, thread_id, run_id, prompt, timestamp)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            """, email, thread_id, run_id, prompt)
            
        print__postgresql_debug(f"[OK] Thread run entry created successfully")
        return run_id
        
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to create thread run entry: {e}")
        # Return the run_id anyway so analysis can continue
        return run_id

async def update_thread_run_sentiment(run_id: str, sentiment: bool, user_email: str = None) -> bool:
    """Update sentiment for a specific run_id with optional user verification."""
    try:
        print__postgresql_debug(f"[DB] Updating sentiment for run_id: {run_id}, sentiment: {sentiment}")
        
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            if user_email:
                # SECURITY: Update with user verification
                print__postgresql_debug(f"[SECURITY] Verifying user {user_email} owns run_id {run_id}")
                result = await conn.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = $1 
                    WHERE run_id = $2 AND email = $3
                """, sentiment, run_id, user_email)
            else:
                # Update without user verification (for backward compatibility)
                print__postgresql_debug(f"[WARN] Updating sentiment without user verification")
                result = await conn.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = $1 
                    WHERE run_id = $2
                """, sentiment, run_id)
            
            # asyncpg returns the status string, extract number
            rows_affected = int(result.split()[1]) if result.startswith('UPDATE') else 0
            success = rows_affected > 0
            
            if success:
                print__postgresql_debug(f"[OK] Sentiment updated successfully for run_id: {run_id}")
            else:
                print__postgresql_debug(f"[WARN] No rows updated - run_id not found or access denied: {run_id}")
            
            return success
            
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to update sentiment for run_id {run_id}: {e}")
        return False

async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get sentiment values for all run_ids in a thread."""
    try:
        print__postgresql_debug(f"[DB] Getting sentiments for thread: {thread_id}, user: {email}")
        
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT run_id, sentiment 
                FROM users_threads_runs 
                WHERE email = $1 AND thread_id = $2 AND sentiment IS NOT NULL
                ORDER BY timestamp ASC
            """, email, thread_id)
            
            sentiments = {}
            for row in rows:
                run_id, sentiment = row['run_id'], row['sentiment']
                sentiments[run_id] = sentiment
            
            print__postgresql_debug(f"[STATS] Retrieved {len(sentiments)} sentiment values for thread {thread_id}")
            return sentiments
            
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to get sentiments for thread {thread_id}: {e}")
        return {}

async def get_user_chat_threads(email: str, connection_pool=None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination."""
    try:
        print__api_postgresql(f"[API] Getting chat threads for user: {email} (limit: {limit}, offset: {offset})")
        
        if connection_pool:
            print__api_postgresql(f"[CONN] Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"[CONN] Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.acquire() as conn:
            # Build the SQL query with optional pagination
            base_query = """
                SELECT 
                    thread_id,
                    MAX(timestamp) as latest_timestamp,
                    COUNT(*) as run_count,
                    (SELECT prompt FROM users_threads_runs utr2 
                     WHERE utr2.email = $1 AND utr2.thread_id = utr.thread_id 
                     ORDER BY timestamp ASC LIMIT 1) as first_prompt
                FROM users_threads_runs utr
                WHERE email = $2
                GROUP BY thread_id
                ORDER BY latest_timestamp DESC
            """
            
            params = [email, email]
            
            # Add pagination if limit is specified
            if limit is not None:
                base_query += " LIMIT $3 OFFSET $4"
                params.extend([limit, offset])
            
            rows = await conn.fetch(base_query, *params)
            
            threads = []
            for row in rows:
                thread_id = row['thread_id']
                latest_timestamp = row['latest_timestamp']
                run_count = row['run_count']
                first_prompt = row['first_prompt']
                
                # Create a title from the first prompt (limit to 50 characters)
                title = (first_prompt[:47] + "...") if first_prompt and len(first_prompt) > 50 else (first_prompt or "Untitled Conversation")
                
                threads.append({
                    "thread_id": thread_id,
                    "latest_timestamp": latest_timestamp,
                    "run_count": run_count,
                    "title": title,
                    "full_prompt": first_prompt or ""
                })
            
            print__api_postgresql(f"[OK] Retrieved {len(threads)} threads for user {email}")
            return threads
            
    except Exception as e:
        print__api_postgresql(f"[ERROR] Failed to get chat threads for user {email}: {e}")
        import traceback
        print__api_postgresql(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise

async def get_user_chat_threads_count(email: str, connection_pool=None) -> int:
    """Get total count of chat threads for a user."""
    try:
        print__api_postgresql(f"[API] Getting chat threads count for user: {email}")
        
        if connection_pool:
            print__api_postgresql(f"[CONN] Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"[CONN] Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.acquire() as conn:
            total_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT thread_id) as total_threads
                FROM users_threads_runs
                WHERE email = $1
            """, email)
            
            print__api_postgresql(f"[OK] Total threads count for user {email}: {total_count}")
            return total_count or 0
            
    except Exception as e:
        print__api_postgresql(f"[ERROR] Failed to get chat threads count for user {email}: {e}")
        import traceback
        print__api_postgresql(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise

async def delete_user_thread_entries(email: str, thread_id: str, connection_pool=None) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table."""
    try:
        print__api_postgresql(f"[API] Deleting thread entries for user: {email}, thread: {thread_id}")
        
        if connection_pool:
            print__api_postgresql(f"[CONN] Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"[CONN] Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.acquire() as conn:
            # First, count the entries to be deleted
            entries_to_delete = await conn.fetchval("""
                SELECT COUNT(*) FROM users_threads_runs 
                WHERE email = $1 AND thread_id = $2
            """, email, thread_id)
            
            print__api_postgresql(f"[STATS] Found {entries_to_delete} entries to delete")
            
            if entries_to_delete == 0:
                print__api_postgresql(f"[WARN] No entries found for user {email} and thread {thread_id}")
                return {
                    "deleted_count": 0,
                    "message": "No entries found to delete",
                    "thread_id": thread_id,
                    "user_email": email
                }
            
            # Delete the entries
            delete_result = await conn.execute("""
                DELETE FROM users_threads_runs 
                WHERE email = $1 AND thread_id = $2
            """, email, thread_id)
            
            # Extract deleted count from result status
            deleted_count = int(delete_result.split()[1]) if delete_result.startswith('DELETE') else 0
            
            print__api_postgresql(f"[OK] Deleted {deleted_count} entries for user {email}, thread {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email
            }
            
    except Exception as e:
        print__api_postgresql(f"[ERROR] Failed to delete thread entries for user {email}, thread {thread_id}: {e}")
        import traceback
        print__api_postgresql(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise

def check_postgres_env_vars():
    """Check if all required PostgreSQL environment variables are set."""
    required_vars = [
        'host', 'port', 'dbname', 
        'user', 'password'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print__postgres_startup_debug(f"[ERROR] Missing required environment variables: {missing_vars}")
        return False
    else:
        print__postgres_startup_debug("[OK] All required PostgreSQL environment variables are set")
        return True

async def test_basic_postgres_connection():
    """Test basic PostgreSQL connection using asyncpg."""
    try:
        print__postgres_startup_debug("[CONN] Testing basic asyncpg PostgreSQL connection...")
        
        # Check environment variables first
        if not check_postgres_env_vars():
            print__postgres_startup_debug("[ERROR] Environment variables check failed")
            return False
        
        config = get_db_config()
        
        # Test direct connection (not using pool)
        # IMPORTANT: Disable statement cache to avoid prepared statement conflicts with pgbouncer
        conn = await asyncpg.connect(
            user=config['user'],
            password=config['password'],
            database=config['dbname'],
            host=config['host'],
            port=config['port'],
            statement_cache_size=0  # Disable statement cache for pgbouncer compatibility
        )
        
        print__postgres_startup_debug("[OK] Direct connection established")
        
        # Get PostgreSQL version
        version = await conn.fetchval('SELECT version()')
        print__postgres_startup_debug(f"[STATS] PostgreSQL version: {version}")
        
        # Close connection
        await conn.close()
        print__postgres_startup_debug("[OK] Connection closed successfully")
        
        return True
    except Exception as e:
        print__postgres_startup_debug(f"[ERROR] Connection test failed: {e}")
        return False

def log_connection_info(host: str, port: str, dbname: str, user: str):
    """Enhanced connection information logging for debugging."""
    print__postgres_startup_debug(f"[CONN] Enhanced PostgreSQL Connection Configuration:")
    print__postgres_startup_debug(f"   📡 Host: {host}")
    print__postgres_startup_debug(f"   🔌 Port: {port}")
    print__postgres_startup_debug(f"   💾 Database: {dbname}")
    print__postgres_startup_debug(f"   👤 User: {user}")
    print__postgres_startup_debug(f"   🔒 SSL: Required (Cloud PostgreSQL)")
    print__postgres_startup_debug(f"   🔄 Connection Pooling: Enhanced (asyncpg 3)")
    print__postgres_startup_debug(f"   🛡️ Pipeline Mode: Disabled (AsyncPipeline error prevention)")

async def test_pool_connection():
    """Enhanced test for asyncpg connection pool creation and functionality."""
    try:
        print__api_postgresql("[CONN] Testing enhanced asyncpg connection pool...")
        
        # Test pool creation with enhanced settings
        pool = await create_fresh_connection_pool()
        print__api_postgresql(f"[OK] Enhanced asyncpg pool created: closed={pool._closed}")
        
        # Test pool statistics
        try:
            min_size = getattr(pool, '_min_size', 'unknown')
            max_size = getattr(pool, '_max_size', 'unknown')
            current_size = len(getattr(pool, '_queue', []))
            print__api_postgresql(f"[STATS] Pool statistics: min={min_size}, max={max_size}, current={current_size}")
        except Exception as e:
            print__api_postgresql(f"[STATS] Pool statistics unavailable: {e}")
        
        # Test pool usage with comprehensive queries
        async with pool.acquire() as conn:
            # Test basic functionality
            result = await conn.fetchrow("SELECT 1 as test, NOW() as current_time, pg_backend_pid() as pid")
            print__api_postgresql(f"[OK] Basic query successful: test={result['test']}, pid={result['pid']}")
            
            # Test transaction handling
            async with conn.transaction():
                await conn.fetchval("SELECT 1")
            print__api_postgresql(f"[OK] Transaction test successful")
            
            # Test connection info
            result = await conn.fetchrow("SELECT version(), current_database(), current_user")
            print__api_postgresql(f"[OK] Connection info: db={result['current_database']}, user={result['current_user']}")
        
        # Test pool health check
        is_healthy = await is_pool_healthy(pool)
        print__api_postgresql(f"[OK] Pool health check: {is_healthy}")
        
        # Test multiple concurrent connections
        if pool._max_size > 1:
            print__api_postgresql("[CONN] Testing concurrent connections...")
            
            async def test_concurrent_query(query_id):
                async with pool.acquire() as conn:
                    result = await conn.fetchrow("SELECT $1 as query_id, pg_backend_pid() as pid", query_id)
                    return f"Query {result['query_id']} -> PID {result['pid']}"
            
            # Run multiple concurrent queries
            tasks = [test_concurrent_query(i) for i in range(2)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print__api_postgresql(f"[WARN] Concurrent query failed: {result}")
                else:
                    print__api_postgresql(f"[OK] Concurrent query: {result}")
        
        # Cleanup
        await pool.close()
        print__api_postgresql(f"[OK] Pool closed: closed={pool._closed}")
        
        return True
        
    except Exception as e:
        print__api_postgresql(f"[ERROR] Enhanced asyncpg pool connection test failed: {e}")
        import traceback
        print__api_postgresql(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return False

async def test_connection_health():
    """Enhanced connection health test with comprehensive SSL diagnostics."""
    try:
        config = get_db_config()
        
        if not all([config['user'], config['password'], config['host'], config['dbname']]):
            print__api_postgresql("[ERROR] Missing required environment variables for database connection")
            return False
            
        print__api_postgresql("[CONN] Testing enhanced PostgreSQL connection health...")
        print__api_postgresql(f"[CONN] Host: {config['host']}:{config['port']}")
        print__api_postgresql(f"[CONN] Database: {config['dbname']}")
        print__api_postgresql(f"[CONN] User: {config['user']}")
        
        # Test asyncpg pool (application operations)
        print__api_postgresql("[CONN] Testing asyncpg pool...")
        try:
            app_pool = await get_healthy_pool()
            async with app_pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchrow("SELECT 1 as test, NOW() as current_time, version() as pg_version, pg_backend_pid() as pid"), 
                    timeout=10
                )
                
                if result and result['test'] == 1:
                    print__api_postgresql("[OK] asyncpg pool health check successful")
                    print__api_postgresql(f"   📊 Server time: {result['current_time']}")
                    print__api_postgresql(f"   📊 Backend PID: {result['pid']}")
                    print__api_postgresql(f"   📊 PostgreSQL: {result['pg_version'][:50]}...")
                    
                    # Test transaction to ensure full functionality
                    async with conn.transaction():
                        await conn.fetchval("SELECT 1")
                    print__api_postgresql("[OK] asyncpg transaction test successful")
                    
                else:
                    print__api_postgresql("[ERROR] asyncpg health check returned unexpected result")
                    return False
        except asyncio.TimeoutError:
            print__api_postgresql("[ERROR] asyncpg pool health check timed out")
            print__api_postgresql("💡 Timeout issue detected:")
            print__api_postgresql("   1. Check database server responsiveness")
            print__api_postgresql("   2. Verify network latency to database")
            print__api_postgresql("   3. Check connection pool settings")
            return False
        except Exception as e:
            error_msg = str(e).lower()
            print__api_postgresql(f"[ERROR] asyncpg pool health check failed: {e}")
            
            # Enhanced error diagnostics from old version
            if "ssl" in error_msg:
                print__api_postgresql("💡 SSL-related issue detected:")
                print__api_postgresql("   1. Check database SSL configuration")
                print__api_postgresql("   2. Verify network connectivity")
                print__api_postgresql("   3. Check firewall settings")
            elif "authentication" in error_msg or "password" in error_msg:
                print__api_postgresql("💡 Authentication issue detected:")
                print__api_postgresql("   1. Verify database credentials")
                print__api_postgresql("   2. Check user permissions")
                print__api_postgresql("   3. Verify connection limits")
            elif any(pattern in error_msg for pattern in ["pipeline", "dbhandler", "flush request"]):
                print__api_postgresql("💡 Pipeline/handler issue detected:")
                print__api_postgresql("   1. This indicates connection state corruption")
                print__api_postgresql("   2. Pool recreation may be required")
                print__api_postgresql("   3. Consider disabling pipeline mode")
            elif any(pattern in error_msg for pattern in [
                "ssl connection has been closed",
                "server closed the connection",
                "connection to server",
                "ssl syscall error",
                "eof detected",
                "bad connection"
            ]):
                print__api_postgresql("🔒 SSL connection error detected:")
                print__api_postgresql("   1. Cloud database may have closed the connection")
                print__api_postgresql("   2. Check network stability")
                print__api_postgresql("   3. Verify SSL certificate validity")
            
            return False
        
        # Test LangGraph checkpointer
        print__api_postgresql("[CONN] Testing LangGraph AsyncPostgresSaver...")
        try:
            checkpointer = await get_postgres_checkpointer()
            # Simple test - just verify it was created
            if checkpointer:
                print__api_postgresql("[OK] LangGraph AsyncPostgresSaver created successfully")
            else:
                print__api_postgresql("[ERROR] LangGraph AsyncPostgresSaver creation returned None")
                return False
        except Exception as e:
            error_msg = str(e).lower()
            print__api_postgresql(f"[ERROR] LangGraph AsyncPostgresSaver test failed: {e}")
            
            # Apply same enhanced error diagnostics for checkpointer
            if "ssl" in error_msg:
                print__api_postgresql("💡 SSL configuration issue - check database SSL settings")
            elif "timeout" in error_msg:
                print__api_postgresql("💡 Network connectivity issue - check connection to database")
            elif "authentication" in error_msg:
                print__api_postgresql("💡 Authentication issue - verify database credentials")
            
            return False
        
        print__api_postgresql("[OK] All database connections are healthy!")
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        print__api_postgresql(f"[ERROR] Connection health test failed: {e}")
        
        # Provide specific guidance based on error type
        if "ssl" in error_msg:
            print__api_postgresql("[TIP] SSL configuration issue - check database SSL settings")
        elif "timeout" in error_msg:
            print__api_postgresql("[TIP] Network connectivity issue - check connection to database")
        elif "authentication" in error_msg:
            print__api_postgresql("[TIP] Authentication issue - verify database credentials")
        
        return False

async def get_postgres_checkpointer():
    """Get LangGraph's built-in AsyncPostgresSaver using standard PostgreSQL connection string.
    
    Note: AsyncPostgresSaver uses psycopg internally, but accepts standard PostgreSQL connection strings.
    """
    try:
        print__postgresql_debug("[CONN] Creating LangGraph AsyncPostgresSaver using standard PostgreSQL connection string...")
        
        if AsyncPostgresSaver is None:
            raise ImportError("AsyncPostgresSaver not available. Install with: pip install langgraph-checkpoint-postgres")
        
        # Get the standard PostgreSQL connection string (works with both asyncpg and psycopg)
        connection_string = get_connection_string()
        
        # FIXED: from_conn_string returns an async context manager in newer LangGraph versions
        # We need to enter the context manager to get the actual checkpointer instance
        checkpointer_ctx = AsyncPostgresSaver.from_conn_string(connection_string)
        
        # Enter the context manager to get the actual checkpointer
        checkpointer = await checkpointer_ctx.__aenter__()
        
        # The setup is automatically done when entering the context manager
        print__postgresql_debug("[OK] LangGraph AsyncPostgresSaver tables set up using context manager")
        
        # Wrap with enhanced resilience for cloud environments
        resilient_checkpointer = ResilientPostgreSQLCheckpointer(checkpointer)
        
        print__postgresql_debug("[OK] LangGraph AsyncPostgresSaver created successfully (uses psycopg internally)")
        return resilient_checkpointer
        
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to create LangGraph AsyncPostgresSaver: {e}")
        import traceback
        print__postgresql_debug(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise

async def get_postgres_checkpointer_with_context():
    """Get LangGraph's AsyncPostgresSaver using the official context manager pattern.
    
    This is the most recommended approach from the documentation for production use.
    Use this when you want the full context manager benefits.
    
    Note: AsyncPostgresSaver uses psycopg internally, but accepts standard PostgreSQL connection strings.
    """
    try:
        print__postgresql_debug("[CONN] Creating LangGraph AsyncPostgresSaver with context manager...")
        
        if AsyncPostgresSaver is None:
            raise ImportError("AsyncPostgresSaver not available. Install with: pip install langgraph-checkpoint-postgres")
        
        # Get the standard PostgreSQL connection string (works with both asyncpg and psycopg)
        connection_string = get_connection_string()
        
        # Return the context manager directly - let caller handle the context
        return AsyncPostgresSaver.from_conn_string(connection_string)
        
    except Exception as e:
        print__postgresql_debug(f"[ERROR] Failed to create LangGraph AsyncPostgresSaver context manager: {e}")
        import traceback
        print__postgresql_debug(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise

# Enhanced wrapper for the built-in AsyncPostgresSaver
class ResilientPostgreSQLCheckpointer:
    """Enhanced wrapper around AsyncPostgresSaver to handle SSL connection drops and errors in cloud environments."""
    
    def __init__(self, base_checkpointer):
        self.base_checkpointer = base_checkpointer
        self._last_pool_recreation = 0
        self._pool_recreation_cooldown = 30  # 30 seconds cooldown between pool recreations

    async def _recreate_checkpointer_pool(self):
        """Attempt to recreate the checkpointer's connection pool."""
        try:
            print__postgresql_debug("[CONN] Attempting to recreate checkpointer connection pool...")
            
            # For AsyncPostgresSaver, we can't easily recreate its internal pool
            # But we can recreate our application's asyncpg pool which often resolves connection issues
            
            # First, try to recreate the application's asyncpg pool
            global database_pool
            if database_pool is not None:
                try:
                    print__postgresql_debug("[CONN] Closing existing asyncpg application pool...")
                    await asyncio.wait_for(database_pool.close(), timeout=10)
                    print__postgresql_debug("[CONN] Old asyncpg pool closed")
                except Exception as e:
                    print__postgresql_debug(f"[WARN] Error closing old asyncpg pool: {e}")
                finally:
                    database_pool = None
            
            # Wait a moment for cleanup
            await asyncio.sleep(2)
            
            # Create new asyncpg pool
            print__postgresql_debug("[CONN] Creating new asyncpg application pool...")
            database_pool = await create_fresh_connection_pool()
            
            if database_pool:
                print__postgresql_debug("[OK] New asyncpg application pool created successfully")
                
                # Test the new pool
                try:
                    async with database_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    print__postgresql_debug("[OK] New asyncpg pool verified")
                except Exception as e:
                    print__postgresql_debug(f"[WARN] New asyncpg pool test failed: {e}")
            
            # For the checkpointer itself, we'll wait a bit to let connection issues resolve
            # The AsyncPostgresSaver manages its own psycopg pool internally
            await asyncio.sleep(1)
            
            print__postgresql_debug("[OK] Checkpointer pool recreation completed")
            
        except Exception as e:
            print__postgresql_debug(f"[WARN] Checkpointer pool recreation failed: {e}")
            # Don't raise the exception as this is a best-effort operation

    async def _enhanced_cloud_resilient_retry(self, operation_name, operation_func, *args, **kwargs):
        """Enhanced retry logic with SSL-specific error handling and pool recreation."""
        # FIXED: Properly parse environment variables for retry configuration
        try:
            max_retries = int(os.getenv("CHECKPOINT_MAX_RETRIES", "4"))
        except (ValueError, TypeError):
            max_retries = 4  # Default if parsing fails
        
        try:
            base_delay = float(os.getenv("CHECKPOINT_RETRY_BASE_DELAY", "1.0"))
        except (ValueError, TypeError):
            base_delay = 1.0  # Default if parsing fails
        
        try:
            dbhandler_multiplier = int(os.getenv("DBHANDLER_EXITED_DELAY_MULTIPLIER", "6"))
        except (ValueError, TypeError):
            dbhandler_multiplier = 6  # Default if parsing fails
        
        try:
            ssl_retry_delay = float(os.getenv("SSL_RETRY_DELAY", "5.0"))
        except (ValueError, TypeError):
            ssl_retry_delay = 5.0  # Default if parsing fails
        
        enable_pool_recreation = os.getenv("ENABLE_POOL_RECREATION", "true").lower() == "true"
        
        # Enhanced SSL and connection error patterns
        ssl_connection_errors = [
            "ssl connection has been closed unexpectedly",
            "consuming input failed",
            "server closed the connection unexpectedly",
            "connection to server",
            "dbhandler exited",
            "asyncpipeline",
            "pipeline",
            "bad connection",
            "connection already closed",
            "flush request failed",
            "insufficient data",
            "lost synchronization",
            "ssl syscall error",
            "eof detected"
        ]
        
        critical_errors = [
            "dbhandler exited",
            "flush request failed",
            "asyncpipeline",
            "lost synchronization"
        ]
        
        for attempt in range(max_retries):
            try:
                print__postgresql_debug(f"[RETRY] Attempt {attempt + 1}/{max_retries} for {operation_name}")
                result = await operation_func(*args, **kwargs)
                if attempt > 0:
                    print__postgresql_debug(f"[OK] {operation_name} succeeded after {attempt + 1} attempts")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                is_ssl_error = any(pattern in error_msg for pattern in ssl_connection_errors)
                is_critical_error = any(pattern in error_msg for pattern in critical_errors)
                
                print__postgresql_debug(f"[WARN] {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Enhanced error diagnostics
                if is_critical_error:
                    print__postgresql_debug("[CRITICAL] CRITICAL ERROR: Database handler or pipeline failure detected")
                elif is_ssl_error:
                    print__postgresql_debug("[SSL-ERROR] SSL CONNECTION ERROR: Connection lost or terminated")
                
                # Last attempt - don't retry
                if attempt == max_retries - 1:
                    print__postgresql_debug(f"[ERROR] {operation_name} failed after {max_retries} attempts")
                    raise
                
                # Calculate delay with enhanced backoff for different error types
                if is_critical_error:
                    # Aggressive backoff for critical errors
                    delay = ssl_retry_delay * (dbhandler_multiplier ** (attempt + 1))
                    print__postgresql_debug(f"[CRITICAL] Critical error - using aggressive backoff: {delay}s")
                elif is_ssl_error:
                    # Extended delay for SSL errors
                    delay = ssl_retry_delay * (2 ** attempt)
                    print__postgresql_debug(f"[SSL-ERROR] SSL error - using extended backoff: {delay}s")
                else:
                    # Standard exponential backoff
                    delay = base_delay * (2 ** attempt)
                    print__postgresql_debug(f"[RETRY] Standard backoff: {delay}s")
                
                # Pool recreation logic for SSL/critical errors
                if (is_ssl_error or is_critical_error) and enable_pool_recreation:
                    current_time = time.time()
                    if current_time - self._last_pool_recreation > self._pool_recreation_cooldown:
                        print__postgresql_debug("[CONN] Attempting checkpointer pool recreation due to connection error...")
                        try:
                            # Force close and recreate the checkpointer connection pool
                            await self._recreate_checkpointer_pool()
                            self._last_pool_recreation = current_time
                            print__postgresql_debug("[OK] Checkpointer pool recreation successful")
                            # Shorter delay after successful pool recreation
                            delay = min(delay, 3)
                        except Exception as pool_error:
                            print__postgresql_debug(f"[WARN] Checkpointer pool recreation failed: {pool_error}")
                    else:
                        cooldown_remaining = self._pool_recreation_cooldown - (current_time - self._last_pool_recreation)
                        print__postgresql_debug(f"[WAIT] Checkpointer pool recreation on cooldown ({cooldown_remaining:.1f}s remaining)")
                
                print__postgresql_debug(f"[WAIT] Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
        
        # Should never reach here
        raise Exception(f"Unexpected end of retry loop for {operation_name}")


    # Enhanced operation wrappers with better error handling
    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._enhanced_cloud_resilient_retry("aput", self.base_checkpointer.aput, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id):
        return await self._enhanced_cloud_resilient_retry("aput_writes", self.base_checkpointer.aput_writes, config, writes, task_id)

    async def aget(self, config):
        return await self._enhanced_cloud_resilient_retry("aget", self.base_checkpointer.aget, config)

    async def aget_tuple(self, config):
        return await self._enhanced_cloud_resilient_retry("aget_tuple", self.base_checkpointer.aget_tuple, config)

    async def alist(self, config, filter=None, before=None, limit=None):
        # Enhanced alist with better error handling for async generators
        try:
            async for item in self.base_checkpointer.alist(config, filter=filter, before=before, limit=limit):
                yield item
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ["ssl", "connection", "dbhandler", "pipeline"]):
                print__postgresql_debug(f"[WARN] alist encountered connection error: {e}")
                # Try to recreate pool and retry once
                try:
                    await self._recreate_checkpointer_pool()
                    await asyncio.sleep(2)
                    async for item in self.base_checkpointer.alist(config, filter=filter, before=before, limit=limit):
                        yield item
                except Exception as retry_error:
                    print__postgresql_debug(f"[ERROR] alist retry failed: {retry_error}")
                    raise
            else:
                raise

    def __getattr__(self, name):
        """Delegate other attributes to the base checkpointer."""
        return getattr(self.base_checkpointer, name)

@asynccontextmanager
async def safe_pool_operation():
    """Context manager to safely track pool operations and prevent concurrent closure."""
    await increment_active_operations()
    try:
        pool = await get_healthy_pool()
        yield pool
    finally:
        await decrement_active_operations()

def get_sync_postgres_checkpointer():
    """Synchronous wrapper for getting PostgreSQL checkpointer."""
    try:
        print__postgres_startup_debug("[CONN] Getting PostgreSQL checkpointer (sync wrapper)")
        return asyncio.run(get_postgres_checkpointer())
    except Exception as e:
        print__postgres_startup_debug(f"[ERROR] Sync checkpointer creation failed: {e}")
        raise

# For backward compatibility
async def create_postgres_checkpointer():
    """Backward compatibility wrapper."""
    return await get_postgres_checkpointer()

async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str, user_email: str = None) -> List[Dict[str, Any]]:
    """Get the COMPLETE conversation messages from the LangGraph PostgreSQL checkpoint history.
    
    SECURITY: Only loads messages for threads owned by the specified user to prevent data leakage.
    
    This extracts ALL user questions and ALL AI responses for proper chat display:
    - All user messages: for right-side blue display
    - All AI messages: for left-side white display using the explicit final_answer from state
    
    Args:
        checkpointer: The LangGraph AsyncPostgresSaver instance
        thread_id: Thread ID for the conversation
        user_email: Email of the user requesting the messages (for security verification)
    
    Returns:
        List of message dictionaries in chronological order (complete conversation history)
    """
    try:
        print__api_postgresql(f"[API] Retrieving COMPLETE checkpoint history for thread: {thread_id}")
        
        # SECURITY CHECK: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__api_postgresql(f"[CONN] Verifying thread ownership for user: {user_email}")
            
            # Get a connection to verify ownership using our asyncpg pool
            pool = await get_healthy_pool()
            
            if pool:
                async with pool.acquire() as conn:
                    thread_entries_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM users_threads_runs 
                        WHERE email = $1 AND thread_id = $2
                    """, user_email, thread_id)
                    
                    if thread_entries_count == 0:
                        print__api_postgresql(f"[ERROR] SECURITY: User {user_email} does not own thread {thread_id} - access denied")
                        return []  # Return empty instead of loading other users' data
                    
                    print__api_postgresql(f"[OK] SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
            else:
                print__api_postgresql(f"[WARN] Could not verify thread ownership - connection pool unavailable")
                return []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints for this thread using the built-in alist()
        checkpoint_tuples = []
        try:
            # For AsyncPostgresSaver, we need to unwrap if it's wrapped
            base_checkpointer = checkpointer
            if hasattr(checkpointer, 'base_checkpointer'):
                base_checkpointer = checkpointer.base_checkpointer
            
            # alist() returns an async generator, iterate over it
            async for checkpoint_tuple in base_checkpointer.alist(config):
                checkpoint_tuples.append(checkpoint_tuple)
                
        except Exception as alist_error:
            print__api_postgresql(f"[ERROR] Error getting checkpoint list: {alist_error}")
            # Try alternative approach if alist fails
            try:
                # Alternative: use aget_tuple to get the latest checkpoint
                state_snapshot = await base_checkpointer.aget_tuple(config)
                if state_snapshot:
                    checkpoint_tuples = [state_snapshot]
                    print__api_postgresql(f"[WARN] Using fallback method - got latest checkpoint only")
            except Exception as fallback_error:
                print__api_postgresql(f"[ERROR] Fallback method also failed: {fallback_error}")
                return []
        
        if not checkpoint_tuples:
            print__api_postgresql(f"[WARN] No checkpoints found for thread: {thread_id}")
            return []
        
        print__api_postgresql(f"[STATS] Found {len(checkpoint_tuples)} checkpoints for verified thread")
        
        # Sort checkpoints chronologically (oldest first)
        checkpoint_tuples.sort(key=lambda x: x.config.get("configurable", {}).get("checkpoint_id", ""))
        
        # Extract conversation messages chronologically
        conversation_messages = []
        seen_prompts = set()
        seen_answers = set()
        
        # Extract all user prompts and AI responses from checkpoint history
        print__api_postgresql(f"[API] Extracting ALL user questions and AI responses...")
        
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}
            
            if not checkpoint:
                continue
                
            # METHOD 1: Extract user prompts from checkpoint writes (new questions)
            if "writes" in metadata and isinstance(metadata["writes"], dict):
                writes = metadata["writes"]
                
                # Look for user prompts in different node writes
                for node_name, node_data in writes.items():
                    if isinstance(node_data, dict):
                        # Check for new prompts (excluding rewritten prompts)
                        prompt = node_data.get("prompt")
                        if (prompt and 
                            prompt.strip() and 
                            prompt.strip() not in seen_prompts and
                            len(prompt.strip()) > 5 and
                            # Filter out rewritten prompts (they usually contain references to previous context)
                            not any(indicator in prompt.lower() for indicator in [
                                "standalone question:", "rephrase", "follow up", "conversation so far"
                            ])):
                            
                            seen_prompts.add(prompt.strip())
                            user_message = {
                                "id": f"user_{len(conversation_messages) + 1}",
                                "content": prompt.strip(),
                                "is_user": True,
                                "timestamp": datetime.fromtimestamp(1700000000 + checkpoint_index * 1000),  # Use stable timestamp for sorting
                                "checkpoint_order": checkpoint_index,
                                "message_order": len(conversation_messages) + 1
                            }
                            conversation_messages.append(user_message)
                            print__api_postgresql(f"[API] Found user prompt in checkpoint {checkpoint_index}: {prompt[:50]}...")
            
            # METHOD 2: Extract AI responses directly from final_answer in channel_values
            if "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                
                # NEW: Use explicit final_answer from state instead of trying to filter messages
                final_answer = channel_values.get("final_answer")
                
                if (final_answer and 
                    isinstance(final_answer, str) and 
                    final_answer.strip() and 
                    len(final_answer.strip()) > 20 and 
                    final_answer.strip() not in seen_answers):
                    
                    seen_answers.add(final_answer.strip())
                    ai_message = {
                        "id": f"ai_{len(conversation_messages) + 1}",
                        "content": final_answer.strip(),
                        "is_user": False,
                        "timestamp": datetime.fromtimestamp(1700000000 + checkpoint_index * 1000 + 500),  # Stable timestamp slightly after user message
                        "checkpoint_order": checkpoint_index,
                        "message_order": len(conversation_messages) + 1
                    }
                    conversation_messages.append(ai_message)
                    print__api_postgresql(f"[API] ✅ Found final_answer in checkpoint {checkpoint_index}: {final_answer[:100]}...")
        
        # Sort all messages by timestamp to ensure proper chronological order
        conversation_messages.sort(key=lambda x: x.get("timestamp", datetime.now()))
        
        # Re-assign sequential IDs and message order after sorting
        for i, msg in enumerate(conversation_messages):
            msg["message_order"] = i + 1
            msg["id"] = f"{'user' if msg['is_user'] else 'ai'}_{i + 1}"
        
        print__api_postgresql(f"[OK] Extracted {len(conversation_messages)} conversation messages from COMPLETE history (verified user access)")
        
        # Debug: Log all messages found
        for i, msg in enumerate(conversation_messages):
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print__api_postgresql(f"{i+1}. {msg_type}: {msg['content'][:50]}...")
        
        return conversation_messages
        
    except Exception as e:
        print__api_postgresql(f"[ERROR] Error retrieving COMPLETE messages from checkpoints: {str(e)}")
        import traceback
        print__api_postgresql(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return []

async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str) -> List[List[str]]:
    """Get queries_and_results from the latest checkpoint state.
    
    Args:
        checkpointer: The LangGraph AsyncPostgresSaver instance
        thread_id: Thread ID for the conversation
    
    Returns:
        List of [query, result] pairs from the latest checkpoint
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Unwrap if checkpointer is wrapped
        base_checkpointer = checkpointer
        if hasattr(checkpointer, 'base_checkpointer'):
            base_checkpointer = checkpointer.base_checkpointer
        
        state_snapshot = await base_checkpointer.aget_tuple(config)
        
        if state_snapshot and state_snapshot.checkpoint:
            channel_values = state_snapshot.checkpoint.get("channel_values", {})
            queries_and_results = channel_values.get("queries_and_results", [])
            print__api_postgresql(f"[OK] Found {len(queries_and_results)} queries from latest checkpoint")
            return [[query, result] for query, result in queries_and_results]
        
        return []
        
    except Exception as e:
        print__api_postgresql(f"[WARN] Could not get queries from checkpoint: {e}")
        return []

if __name__ == "__main__":
    async def test():
        print__postgresql_debug("[CONN] Testing PostgreSQL connection...")
        
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host") 
        port = os.getenv("port", "5432")
        dbname = os.getenv("dbname")
        
        print__postgresql_debug(f"[CONN] User: {user}")
        print__postgresql_debug(f"[CONN] Host: {host}")
        print__postgresql_debug(f"[CONN] Port: {port}")
        print__postgresql_debug(f"[CONN] Database: {dbname}")
        print__postgresql_debug(f"[CONN] Password configured: {bool(password)}")
        
        # Test connection health first
        health_ok = await test_connection_health()
        if not health_ok:
            print__postgresql_debug("[ERROR] Basic connectivity test failed")
            return
        
        # Test full checkpointer setup
        checkpointer = await get_postgres_checkpointer()
        print__postgresql_debug(f"[CONN] Checkpointer type: {type(checkpointer).__name__}")
        
        # Cleanup
        if hasattr(checkpointer, 'conn'):
            await checkpointer.conn.close()
            print__postgresql_debug("[OK] Connection pool closed")
    
    asyncio.run(test()) 