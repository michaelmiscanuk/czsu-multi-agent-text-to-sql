#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using AsyncPostgresSaver from LangGraph.
Follows official documentation patterns exactly - no custom wrappers needed.
"""

from __future__ import annotations
import asyncio
import sys
import os
import functools
from typing import Optional, List, Dict, Any, Callable, TypeVar, Awaitable
import time
from datetime import datetime
from contextlib import asynccontextmanager
import traceback
import uuid
import threading
import gc
import psycopg

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# CRITICAL: Windows event loop fix MUST be first for PostgreSQL compatibility
if sys.platform == "win32":
    print("[POSTGRES-STARTUP] Windows detected - setting SelectorEventLoop for PostgreSQL compatibility...")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[POSTGRES-STARTUP] Event loop policy set successfully")

# Type variable for the retry decorator
T = TypeVar('T')

#==============================================================================
# DEBUG FUNCTIONS
#==============================================================================
def print__analysis_tracing_debug(msg: str) -> None:
    """Print analysis tracing debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    analysis_tracing_debug_mode = os.environ.get('print__analysis_tracing_debug', '0')
    if analysis_tracing_debug_mode == '1':
        print(f"[print__analysis_tracing_debug] 🔍 {msg}")
        sys.stdout.flush()

#==============================================================================
# PREPARED STATEMENT ERROR HANDLING
#==============================================================================
def is_prepared_statement_error(error: Exception) -> bool:
    """Check if an error is related to prepared statements."""
    print__analysis_tracing_debug("200 - PREPARED STATEMENT CHECK: Checking if error is prepared statement related")
    error_str = str(error).lower()
    result = any(indicator in error_str for indicator in [
        'prepared statement',
        'does not exist',
        '_pg3_',
        '_pg_',
        'invalidsqlstatementname'
    ])
    print__analysis_tracing_debug(f"201 - PREPARED STATEMENT RESULT: Error is prepared statement related: {result}")
    return result

def retry_on_prepared_statement_error(max_retries: int = 3):
    """Decorator to retry operations that fail due to prepared statement errors."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            print__analysis_tracing_debug(f"202 - RETRY WRAPPER START: Starting {func.__name__} with max_retries={max_retries}")
            last_error = None
            
            for attempt in range(max_retries + 1):
                print__analysis_tracing_debug(f"203 - RETRY ATTEMPT: Attempt {attempt + 1}/{max_retries + 1} for {func.__name__}")
                try:
                    result = await func(*args, **kwargs)
                    print__analysis_tracing_debug(f"204 - RETRY SUCCESS: {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_error = e
                    print__analysis_tracing_debug(f"205 - RETRY ERROR: {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                    
                    # CRITICAL: Add full traceback to see exactly where the f-string error is occurring
                    
                    full_traceback = traceback.format_exc()
                    print__analysis_tracing_debug(f"205.1 - RETRY TRACEBACK: {full_traceback}")
                    
                    if is_prepared_statement_error(e):
                        print__analysis_tracing_debug(f"206 - PREPARED STATEMENT ERROR: Detected prepared statement error in {func.__name__}")
                        
                        if attempt < max_retries:
                            print__analysis_tracing_debug(f"207 - RETRY CLEANUP: Clearing prepared statements before retry {attempt + 2}")
                            try:
                                await clear_prepared_statements()
                                # Also try to recreate the checkpointer if it's a global operation
                                global _global_checkpointer_context, _global_checkpointer
                                if _global_checkpointer_context or _global_checkpointer:
                                    print__analysis_tracing_debug("208 - CHECKPOINTER RECREATION: Recreating checkpointer due to prepared statement error")
                                    await close_async_postgres_saver()
                                    await create_async_postgres_saver()
                            except Exception as cleanup_error:
                                print__analysis_tracing_debug(f"209 - CLEANUP ERROR: Error during cleanup: {cleanup_error}")
                            continue
                    
                    # If it's not a prepared statement error, or we've exhausted retries, re-raise
                    print__analysis_tracing_debug(f"210 - RETRY EXHAUSTED: No more retries for {func.__name__}, re-raising error")
                    raise
            
            # This should never be reached, but just in case
            print__analysis_tracing_debug(f"211 - RETRY FALLBACK: Fallback error re-raise for {func.__name__}")
            raise last_error
        
        return wrapper
    return decorator

#==============================================================================
# SIMPLIFIED GLOBALS - FOLLOWING OFFICIAL PATTERNS
#==============================================================================
_global_checkpointer_context = None  # Store the async context manager
_global_checkpointer: Optional[AsyncPostgresSaver] = None
_connection_string_cache = None  # Cache the connection string to avoid timestamp conflicts

#==============================================================================
# DEBUG FUNCTIONS
#==============================================================================
def print__postgresql_debug(msg: str) -> None:
    """Print PostgreSQL debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[POSTGRESQL-DEBUG] {msg}")
        sys.stdout.flush()

def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled."""
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[API-POSTGRESQL] {msg}")
        sys.stdout.flush()

def get_db_config():
    """Get database configuration from environment variables."""
    print__analysis_tracing_debug("212 - DB CONFIG START: Getting database configuration from environment variables")
    config = {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }
    print__analysis_tracing_debug(f"213 - DB CONFIG RESULT: Configuration retrieved - host: {config['host']}, port: {config['port']}, dbname: {config['dbname']}, user: {config['user']}")
    return config

def get_connection_string():
    """Get PostgreSQL connection string for LangGraph checkpointer.
    
    ENHANCED: Added cloud-optimized connection parameters to prevent connection issues.
    """
    print__analysis_tracing_debug("214 - CONNECTION STRING START: Generating PostgreSQL connection string")
    global _connection_string_cache
    
    if _connection_string_cache is not None:
        print__analysis_tracing_debug("215 - CONNECTION STRING CACHED: Using cached connection string")
        return _connection_string_cache
    
    config = get_db_config()
    
    # Use process ID + startup time + random for truly unique application name
    process_id = os.getpid()
    thread_id = threading.get_ident()
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    
    app_name = f"czsu_langgraph_{process_id}_{thread_id}_{startup_time}_{random_id}"
    print__analysis_tracing_debug(f"216 - CONNECTION STRING APP NAME: Generated unique application name: {app_name}")
    
    # ENHANCED: Cloud-optimized connection string with better timeout and keepalive settings
    _connection_string_cache = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"sslmode=require"
        f"&application_name={app_name}"
        f"&connect_timeout=20"              # Cloud-friendly timeout
        f"&keepalives_idle=600"             # 10 minutes before first keepalive
        f"&keepalives_interval=30"          # 30 seconds between keepalives
        f"&keepalives_count=3"              # 3 failed keepalives before disconnect
        f"&tcp_user_timeout=30000"          # 30 seconds TCP timeout
    )
    
    print__analysis_tracing_debug("217 - CONNECTION STRING COMPLETE: Cloud-optimized PostgreSQL connection string generated")
    
    return _connection_string_cache

def get_connection_kwargs():
    """Get connection kwargs for cloud-optimized connection handling.
    
    FIXED: Based on GitHub issue #2967 and LangGraph discussions, 
    autocommit=False works better with cloud databases like Supabase
    under concurrent load.
    
    Returns connection parameters that should be passed to psycopg connection methods.
    """
    return {
        "autocommit": False,  # CRITICAL FIX: False works better with cloud databases under load
        "prepare_threshold": None,  # Disable prepared statements completely
    }

def check_postgres_env_vars():
    """Check if all required PostgreSQL environment variables are set."""
    print__analysis_tracing_debug("218 - ENV VARS CHECK START: Checking PostgreSQL environment variables")
    required_vars = ['host', 'port', 'dbname', 'user', 'password']
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print__analysis_tracing_debug(f"219 - ENV VARS MISSING: Missing required environment variables: {missing_vars}")
        return False
    else:
        print__analysis_tracing_debug("220 - ENV VARS COMPLETE: All required PostgreSQL environment variables are set")
        return True

async def clear_prepared_statements():
    """Clear any existing prepared statements to avoid conflicts.
    
    SIMPLIFIED: This function is now optional and only used during error recovery.
    Most prepared statement issues are now prevented by connection kwargs.
    """
    print__analysis_tracing_debug("221 - CLEAR PREPARED START: Starting prepared statements cleanup")
    try:
        config = get_db_config()
        # Use a different application name for the cleanup connection
        
        cleanup_app_name = f"czsu_cleanup_{uuid.uuid4().hex[:8]}"
        print__analysis_tracing_debug(f"222 - CLEANUP CONNECTION: Creating cleanup connection with app name: {cleanup_app_name}")
        
        # Create connection string without prepared statement parameters
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"
        
        # Get connection kwargs for disabling prepared statements
        connection_kwargs = get_connection_kwargs()
        
        print__analysis_tracing_debug("223 - PSYCOPG CONNECTION: Establishing psycopg connection for cleanup")
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            print__analysis_tracing_debug("224 - CONNECTION ESTABLISHED: Cleanup connection established successfully")
            async with conn.cursor() as cur:
                print__analysis_tracing_debug("225 - CURSOR CREATED: Database cursor created for prepared statement query")
                # Get all prepared statements for our application
                await cur.execute("""
                    SELECT name FROM pg_prepared_statements 
                    WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                """)
                prepared_statements = await cur.fetchall()
                
                if prepared_statements:
                    print__analysis_tracing_debug(f"226 - PREPARED STATEMENTS FOUND: Found {len(prepared_statements)} prepared statements to clear")
                    
                    # Drop each prepared statement
                    for i, stmt in enumerate(prepared_statements, 1):
                        stmt_name = stmt[0]
                        # OPTIMIZATION: Only log first few statements
                        if i <= 3:
                            print__analysis_tracing_debug(f"227 - CLEARING STATEMENT {i}: Clearing prepared statement: {stmt_name}")
                        try:
                            await cur.execute(f"DEALLOCATE {stmt_name};")
                            if i <= 3:
                                print__analysis_tracing_debug(f"228 - STATEMENT CLEARED {i}: Successfully cleared prepared statement: {stmt_name}")
                        except Exception as e:
                            if i <= 3:
                                print__analysis_tracing_debug(f"229 - STATEMENT ERROR {i}: Could not clear prepared statement {stmt_name}: {e}")
                    
                    if len(prepared_statements) > 3:
                        print__analysis_tracing_debug(f"230 - CLEANUP SUMMARY: Cleared {len(prepared_statements)} prepared statements (showing first 3)")
                    else:
                        print__analysis_tracing_debug(f"230 - CLEANUP COMPLETE: Cleared {len(prepared_statements)} prepared statements")
                else:
                    print__analysis_tracing_debug("231 - NO STATEMENTS: No prepared statements to clear")
                
    except Exception as e:
        print__analysis_tracing_debug(f"232 - CLEANUP ERROR: Error clearing prepared statements (non-fatal): {e}")
        # Don't raise - this is a cleanup operation and shouldn't block checkpointer creation

async def cleanup_all_pools():
    """
    Enhanced cleanup function that properly handles modern connection pools.
    This function ensures all resources are cleaned up properly.
    """
    print__analysis_tracing_debug("CLEANUP ALL POOLS START: Starting comprehensive pool cleanup")
    
    global _global_checkpointer_context, _global_checkpointer
    
    # Clean up the global checkpointer context if it exists
    if _global_checkpointer_context:
        try:
            print__analysis_tracing_debug("CLEANUP: Cleaning up global checkpointer context using modern approach")
            await _global_checkpointer_context.__aexit__(None, None, None)
            print__analysis_tracing_debug("CLEANUP: Global checkpointer context cleaned up successfully")
        except Exception as e:
            print__analysis_tracing_debug(f"CLEANUP ERROR: Error during global checkpointer cleanup: {e}")
        finally:
            _global_checkpointer_context = None
            _global_checkpointer = None
    
    # Force garbage collection to ensure resources are freed
    
    gc.collect()
    print__analysis_tracing_debug("CLEANUP ALL POOLS COMPLETE: All pools and resources cleaned up")

async def force_close_modern_pools():
    """
    Force close any remaining modern connection pools.
    This is a more aggressive cleanup function for troubleshooting.
    """
    print__analysis_tracing_debug("FORCE CLOSE START: Force closing all modern connection pools")
    
    try:
        # Clean up the global state
        await cleanup_all_pools()
        
        # Additional cleanup for any lingering connections
        print__analysis_tracing_debug("FORCE CLOSE: Forcing cleanup of any remaining resources")
        
        # Clear any cached connection strings to force recreation
        global _connection_string_cache
        _connection_string_cache = None
        
        print__analysis_tracing_debug("FORCE CLOSE COMPLETE: Modern pool force close completed")
        
    except Exception as e:
        print__analysis_tracing_debug(f"FORCE CLOSE ERROR: Error during force close: {e}")
        # Don't re-raise - this is a cleanup function

# ENHANCED OFFICIAL ASYNCPOSTGRESSAVER IMPLEMENTATION WITH CONNECTION POOL
@retry_on_prepared_statement_error(max_retries=3)
async def create_async_postgres_saver():
    """
    Create AsyncPostgresSaver using CLOUD-OPTIMIZED connection pool approach.
    
    FIXED: Based on GitHub issue #2967, LangGraph discussions, and cloud best practices:
    - Use autocommit=False for better cloud database compatibility
    - Implement proper connection pool sizing for concurrent scenarios
    - Add connection lifecycle management for high-load scenarios
    - FIXED: Use separate autocommit=True connection for setup to avoid CONCURRENT INDEX errors
    """
    print__analysis_tracing_debug("233 - CREATE SAVER START: Starting AsyncPostgresSaver creation with cloud-optimized connection pool")
    global _global_checkpointer_context, _global_checkpointer
    
    # CRITICAL: Clear any existing state first to avoid conflicts
    if _global_checkpointer_context or _global_checkpointer:
        print__analysis_tracing_debug("234 - EXISTING STATE CLEANUP: Clearing existing checkpointer state to avoid conflicts")
        try:
            if _global_checkpointer_context:
                await _global_checkpointer_context.__aexit__(None, None, None)
        except Exception as e:
            print__analysis_tracing_debug(f"236 - CLEANUP ERROR: Error during state cleanup: {e}")
        finally:
            _global_checkpointer_context = None
            _global_checkpointer = None
            print__analysis_tracing_debug("237 - STATE CLEARED: Global checkpointer state cleared")
    
    if not AsyncPostgresSaver:
        print__analysis_tracing_debug("239 - SAVER UNAVAILABLE: AsyncPostgresSaver not available")
        raise Exception("AsyncPostgresSaver not available")
    
    if not check_postgres_env_vars():
        print__analysis_tracing_debug("240 - ENV VARS MISSING: Missing required PostgreSQL environment variables")
        raise Exception("Missing required PostgreSQL environment variables")
    
    print__analysis_tracing_debug("241 - CLOUD-OPTIMIZED CREATION: Creating AsyncPostgresSaver using cloud-optimized approach")
    
    try:
        # Try modern connection pool approach first (for high concurrency)
        try:
            
            
            # Get connection details
            connection_string = get_connection_string()
            connection_kwargs = get_connection_kwargs()
            
            print__analysis_tracing_debug("242 - MODERN CONNECTION POOL: Setting up AsyncConnectionPool with cloud-optimized settings")
            
            # CLOUD-OPTIMIZED: Smaller pool size, better timeouts for Supabase/cloud databases
            pool = AsyncConnectionPool(
                conninfo=connection_string,
                min_size=1,       # Start small for cloud efficiency
                max_size=5,       # Reduced from 20 to avoid connection limits on cloud services  
                timeout=30,       # Longer timeout for cloud latency
                max_idle=600,     # 10 minutes idle timeout
                max_lifetime=3600, # 1 hour max connection lifetime for cloud stability
                kwargs={
                    **connection_kwargs,
                    "connect_timeout": 20,  # Cloud-friendly connection timeout
                },
                open=False  # CRITICAL: Set to False to avoid deprecation warnings
            )
            
            # MODERN APPROACH: Open the pool explicitly using await (recommended by psycopg)
            print__analysis_tracing_debug("243 - MODERN POOL OPENING: Opening connection pool using modern await approach")
            await pool.open()
            print__analysis_tracing_debug("244 - MODERN POOL OPENED: Connection pool opened successfully using modern approach")
            
            # MODERN APPROACH: Create AsyncPostgresSaver with the pool
            print__analysis_tracing_debug("245 - MODERN SAVER CREATION: Creating AsyncPostgresSaver with modern connection pool")
            
            # Try passing pool directly (newer LangGraph versions support this)
            try:
                checkpointer = AsyncPostgresSaver(
                    pool=pool,
                    serde=None       # Use default serialization
                )
                print__analysis_tracing_debug("246 - MODERN POOL CONSTRUCTOR: Using modern pool constructor")
            except TypeError:
                # Fallback: Create with connection from pool (older versions)
                print__analysis_tracing_debug("247 - MODERN POOL FALLBACK: Using connection from pool approach")
                async with pool.connection() as conn:
                    checkpointer = AsyncPostgresSaver(
                        conn,
                        serde=None
                    )
            
            # Create a modern context manager wrapper for proper cleanup
            class ModernAsyncPostgresSaverContext:
                def __init__(self, checkpointer, pool):
                    self.checkpointer = checkpointer
                    self.pool = pool
                    
                async def __aenter__(self):
                    return self.checkpointer
                    
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    # Close the pool when context exits
                    try:
                        print__analysis_tracing_debug("Pool cleanup: Closing connection pool using modern approach")
                        await self.pool.close()
                        print__analysis_tracing_debug("Pool cleanup: Connection pool closed successfully")
                    except Exception as e:
                        print__analysis_tracing_debug(f"Pool cleanup error: {e}")
            
            _global_checkpointer_context = ModernAsyncPostgresSaverContext(checkpointer, pool)
            
            # Enter the context manager
            print__analysis_tracing_debug("248 - MODERN CONTEXT ENTER: Entering modern async context manager")
            _global_checkpointer = await _global_checkpointer_context.__aenter__()
            
            print__analysis_tracing_debug("249 - MODERN SAVER CREATED: AsyncPostgresSaver created using modern connection pool approach")
            
        except ImportError:
            print__analysis_tracing_debug("250 - MODERN IMPORT ERROR: psycopg_pool not available, using fallback")
            raise ImportError("psycopg_pool required")
            
    except Exception as pool_error:
        print__analysis_tracing_debug(f"251 - MODERN POOL ERROR: Pool approach failed: {pool_error}")
        # Fallback to connection string approach with cloud-optimized settings
        print__analysis_tracing_debug("252 - CLOUD-OPTIMIZED FALLBACK: Using cloud-optimized connection string approach")
        
        connection_string = get_connection_string()
        _global_checkpointer_context = AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            serde=None
        )
        
        _global_checkpointer = await _global_checkpointer_context.__aenter__()
        print__analysis_tracing_debug("253 - CLOUD-OPTIMIZED FALLBACK SUCCESS: Using cloud-optimized fallback approach")
    
    # FIXED: Setup the checkpointer with autocommit=True connection to avoid CONCURRENT INDEX errors
    print__analysis_tracing_debug("254 - SETUP START: Running checkpointer setup with autocommit=True connection")
    await setup_checkpointer_with_autocommit(_global_checkpointer)
    print__analysis_tracing_debug("255 - SETUP COMPLETE: AsyncPostgresSaver setup complete")
    
    # Test the checkpointer to ensure it's working
    print__analysis_tracing_debug("256 - TESTING START: Testing checkpointer")
    test_config = {"configurable": {"thread_id": "setup_test"}}
    test_result = await _global_checkpointer.aget(test_config)
    print__analysis_tracing_debug(f"257 - TESTING COMPLETE: Checkpointer test successful: {test_result is None}")
    
    # Setup custom tables using the same connection approach
    print__analysis_tracing_debug("258 - CUSTOM TABLES: Setting up custom users_threads_runs table")
    await setup_users_threads_runs_table()
    
    print__analysis_tracing_debug("259 - CREATE SAVER SUCCESS: Cloud-optimized AsyncPostgresSaver creation completed successfully")
    return _global_checkpointer

async def setup_checkpointer_with_autocommit(checkpointer):
    """
    Setup the checkpointer using a separate connection with autocommit=True.
    This avoids the "CREATE INDEX CONCURRENTLY cannot run inside a transaction block" error.
    """
    print__analysis_tracing_debug("SETUP AUTOCOMMIT START: Setting up checkpointer with autocommit=True connection")
    
    try:
        # Get connection kwargs with autocommit=True specifically for setup
        setup_connection_kwargs = get_connection_kwargs().copy()
        setup_connection_kwargs["autocommit"] = True  # Override to True for setup only
        
        connection_string = get_connection_string()
        
        print__analysis_tracing_debug("SETUP AUTOCOMMIT: Creating temporary connection with autocommit=True for setup")
        
        # Create a temporary AsyncPostgresSaver with autocommit=True for setup only
        setup_checkpointer_context = AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            serde=None
        )
        
        async with setup_checkpointer_context as setup_checkpointer:
            print__analysis_tracing_debug("SETUP AUTOCOMMIT: Running setup with autocommit=True connection")
            await setup_checkpointer.setup()
            print__analysis_tracing_debug("SETUP AUTOCOMMIT SUCCESS: Setup completed successfully with autocommit=True")
            
    except Exception as e:
        print__analysis_tracing_debug(f"SETUP AUTOCOMMIT ERROR: Error during autocommit setup: {e}")
        # Fallback: try the manual setup approach
        print__analysis_tracing_debug("SETUP AUTOCOMMIT FALLBACK: Trying manual setup with direct connection")
        await manual_checkpointer_setup()

async def manual_checkpointer_setup():
    """
    Manual setup of checkpointer tables using direct connection with autocommit=True.
    This is a fallback if the AsyncPostgresSaver.setup() continues to fail.
    """
    print__analysis_tracing_debug("MANUAL SETUP START: Setting up checkpointer tables manually")
    
    try:        
        # Create connection string with autocommit=True for setup
        connection_string = get_connection_string()
        setup_kwargs = get_connection_kwargs().copy()
        setup_kwargs["autocommit"] = True
        
        print__analysis_tracing_debug("MANUAL SETUP: Creating direct connection with autocommit=True")
        async with await psycopg.AsyncConnection.connect(connection_string, **setup_kwargs) as conn:
            async with conn.cursor() as cur:
                print__analysis_tracing_debug("MANUAL SETUP: Creating checkpoints table")
                
                # Create the basic checkpoints table (simplified version)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint JSONB NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    );
                """)
                
                print__analysis_tracing_debug("MANUAL SETUP: Creating checkpoint_blobs table")
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        channel TEXT NOT NULL,
                        version TEXT NOT NULL,
                        type TEXT NOT NULL,
                        blob BYTEA,
                        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                    );
                """)
                
                print__analysis_tracing_debug("MANUAL SETUP: Creating indexes (non-concurrent)")
                # Create indexes without CONCURRENTLY to avoid transaction issues
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx 
                    ON checkpoints(thread_id);
                """)
                
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx 
                    ON checkpoint_blobs(thread_id);
                """)
                
                print__analysis_tracing_debug("MANUAL SETUP SUCCESS: Checkpointer tables created manually")
                
    except Exception as e:
        print__analysis_tracing_debug(f"MANUAL SETUP ERROR: Failed to setup tables manually: {e}")
        # Don't raise - let the system continue, as the checkpointer might still work

async def close_async_postgres_saver():
    """Close the AsyncPostgresSaver properly using the context manager."""
    print__analysis_tracing_debug("258 - CLOSE SAVER START: Closing AsyncPostgresSaver using official context manager")
    global _global_checkpointer_context, _global_checkpointer
    
    if _global_checkpointer_context:
        try:
            print__analysis_tracing_debug("259 - CLOSE CONTEXT: Exiting async context manager")
            await _global_checkpointer_context.__aexit__(None, None, None)
            print__analysis_tracing_debug("260 - CLOSE SUCCESS: AsyncPostgresSaver closed properly")
        except Exception as e:
            print__analysis_tracing_debug(f"261 - CLOSE ERROR: Error during AsyncPostgresSaver cleanup: {e}")
        finally:
            _global_checkpointer_context = None
            _global_checkpointer = None
            print__analysis_tracing_debug("262 - CLOSE CLEANUP: Global state cleared after close")
    else:
        print__analysis_tracing_debug("263 - CLOSE SKIP: No checkpointer context to close")

@retry_on_prepared_statement_error(max_retries=2)
async def get_global_checkpointer():
    """Get the global checkpointer instance (for API compatibility).
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    print__analysis_tracing_debug("264 - GET GLOBAL START: Getting global checkpointer instance")
    global _global_checkpointer
    
    if _global_checkpointer is None:
        print__analysis_tracing_debug("265 - CREATE NEW: No existing checkpointer, creating new one")
        _global_checkpointer = await create_async_postgres_saver()
        print__analysis_tracing_debug("266 - CREATE SUCCESS: New checkpointer created successfully")
    else:
        print__analysis_tracing_debug("267 - EXISTING FOUND: Using existing global checkpointer")
    
    return _global_checkpointer

# SIMPLIFIED USERS_THREADS_RUNS TABLE MANAGEMENT
# We'll use a simple connection approach since AsyncPostgresSaver manages its own connections
async def setup_users_threads_runs_table():
    """Create the users_threads_runs table using direct connection."""
    print__analysis_tracing_debug("268 - CUSTOM TABLE START: Setting up users_threads_runs table using direct connection")
    try:
        # Use direct connection for table setup (simpler than pool management)
        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        print__analysis_tracing_debug("269 - CUSTOM TABLE CONNECTION: Establishing connection for table setup")
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            print__analysis_tracing_debug("270 - CUSTOM TABLE CONNECTED: Connection established for table creation")
            # Create table with correct schema
            print__analysis_tracing_debug("271 - CREATE TABLE: Creating users_threads_runs table")
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
            print__analysis_tracing_debug("272 - CREATE INDEXES: Creating indexes for better performance")
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email 
                ON users_threads_runs(email);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_thread_id 
                ON users_threads_runs(thread_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread 
                ON users_threads_runs(email, thread_id);
            """)

            print__analysis_tracing_debug("273 - CUSTOM TABLE SUCCESS: users_threads_runs table and indexes created successfully")

    except Exception as e:
        print__analysis_tracing_debug(f"274 - CUSTOM TABLE ERROR: Failed to setup users_threads_runs table: {e}")
        raise

@asynccontextmanager
async def get_direct_connection():
    """Get a direct database connection for users_threads_runs operations."""
    
    connection_string = get_connection_string()
    connection_kwargs = get_connection_kwargs()
    async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
        yield conn

# MISSING FUNCTIONS NEEDED BY API SERVER
async def get_healthy_pool():
    """Get a healthy PostgreSQL connection pool for direct operations."""
    try:
        # For the simplified approach, we use direct connections instead of pooling
        # Return a connection factory that mimics pool behavior
        class DirectConnectionFactory:
            def __init__(self, connection_string, connection_kwargs):
                self.connection_string = connection_string
                self.connection_kwargs = connection_kwargs
            
            @asynccontextmanager
            async def connection(self):
                """Provide a connection that mimics pool.connection() interface."""
                async with await psycopg.AsyncConnection.connect(self.connection_string, **self.connection_kwargs) as conn:
                    yield conn
        
        return DirectConnectionFactory(get_connection_string(), get_connection_kwargs())
        
    except Exception as e:
        print__postgresql_debug(f"Failed to create connection factory: {e}")
        raise

@retry_on_prepared_statement_error(max_retries=2)
async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str, user_email: str = None) -> List[Dict[str, Any]]:
    """
    Get conversation messages from checkpoints - USING OFFICIAL ASYNCPOSTGRESSAVER METHODS.
    
    This function properly extracts messages from LangGraph checkpoints using the official
    AsyncPostgresSaver methods as documented.
    
    OPTIMIZED: Reduce unnecessary database operations and limit checkpoint processing.
    """
    print__analysis_tracing_debug(f"292 - GET CONVERSATION START: Retrieving conversation messages for thread: {thread_id}")
    try:
        # 🔒 SECURITY CHECK: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__analysis_tracing_debug(f"293 - SECURITY CHECK: Verifying thread ownership for user: {user_email}")
            
            try:
                async with get_direct_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("""
                            SELECT COUNT(*) FROM users_threads_runs 
                            WHERE email = %s AND thread_id = %s
                        """, (user_email, thread_id))
                        result = await cur.fetchone()
                        thread_entries_count = result[0] if result else 0
                    
                    if thread_entries_count == 0:
                        print__analysis_tracing_debug(f"294 - SECURITY DENIED: User {user_email} does not own thread {thread_id} - access denied")
                        return []
                    
                    print__analysis_tracing_debug(f"295 - SECURITY GRANTED: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
            except Exception as e:
                print__analysis_tracing_debug(f"296 - SECURITY ERROR: Could not verify thread ownership: {e}")
                return []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # OPTIMIZED: Use alist() method with limit to avoid processing too many checkpoints
        checkpoint_tuples = []
        try:
            print__analysis_tracing_debug(f"297 - ALIST METHOD: Using official AsyncPostgresSaver.alist() method")
            
            # OPTIMIZATION: Limit checkpoints to recent ones only (last 10)
            async for checkpoint_tuple in checkpointer.alist(config, limit=10):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__analysis_tracing_debug(f"298 - ALIST ERROR: Error using alist(): {alist_error}")
            
            # Fallback: use aget_tuple() to get the latest checkpoint only
            if not checkpoint_tuples:
                print__analysis_tracing_debug(f"299 - FALLBACK METHOD: Trying fallback method using aget_tuple()")
                try:
                    state_snapshot = await checkpointer.aget_tuple(config)
                    if state_snapshot:
                        checkpoint_tuples = [state_snapshot]
                        print__analysis_tracing_debug(f"300 - FALLBACK SUCCESS: Using fallback method - got latest checkpoint only")
                except Exception as fallback_error:
                    print__analysis_tracing_debug(f"301 - FALLBACK ERROR: Fallback method also failed: {fallback_error}")
                    return []
        
        if not checkpoint_tuples:
            print__analysis_tracing_debug(f"302 - NO CHECKPOINTS: No checkpoints found for thread: {thread_id}")
            return []
        
        print__analysis_tracing_debug(f"303 - CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for verified thread")
        
        # Sort checkpoints chronologically (oldest first) based on timestamp
        checkpoint_tuples.sort(key=lambda x: x.checkpoint.get("ts", "") if x.checkpoint else "")
        
        # OPTIMIZED: Extract conversation messages more efficiently
        conversation_messages = []
        seen_prompts = set()
        seen_answers = set()
        
        print__analysis_tracing_debug(f"304 - MESSAGE EXTRACTION: Extracting messages from {len(checkpoint_tuples)} checkpoints")
        
        # FIXED: Two-pass approach to ensure proper prompt->answer ordering
        # Pass 1: Extract all user prompts first, in checkpoint order
        user_prompts = []
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}
            
            if not checkpoint:
                continue
                
            # OPTIMIZATION: Only log for significant checkpoints
            if checkpoint_index < 5 or checkpoint_index % 5 == 0:  # Log first 5, then every 5th
                print__analysis_tracing_debug(f"305 - PROCESSING CHECKPOINT (Pass 1): Processing checkpoint {checkpoint_index + 1}/{len(checkpoint_tuples)} for user prompts")
            
            # EXTRACT USER PROMPTS from checkpoint metadata writes
            if "writes" in metadata and isinstance(metadata["writes"], dict):
                writes = metadata["writes"]
                
                for node_name, node_data in writes.items():
                    if isinstance(node_data, dict):
                        prompt = node_data.get("prompt")
                        if (prompt and 
                            prompt.strip() and 
                            prompt.strip() not in seen_prompts and
                            len(prompt.strip()) > 5):
                            
                            # Filter out rewritten prompts
                            if not any(indicator in prompt.lower() for indicator in [
                                "standalone question:", "rephrase", "follow up", "conversation so far",
                                "given the conversation", "rewrite", "context:"
                            ]):
                                seen_prompts.add(prompt.strip())
                                user_prompts.append({
                                    "content": prompt.strip(),
                                    "checkpoint_index": checkpoint_index
                                })
                                print__analysis_tracing_debug(f"306 - USER MESSAGE FOUND: Found user prompt: {prompt[:50]}...")
        
        # Pass 2: Extract AI responses and pair them with user prompts
        ai_responses = []
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            
            if not checkpoint:
                continue
                
            # EXTRACT AI RESPONSES from channel_values - OPTIMIZED extraction
            if "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                
                # Method 1: Look for final_answer (the main AI response)
                final_answer = channel_values.get("final_answer")
                if (final_answer and 
                    isinstance(final_answer, str) and 
                    final_answer.strip() and 
                    len(final_answer.strip()) > 20 and 
                    final_answer.strip() not in seen_answers):
                    
                    seen_answers.add(final_answer.strip())
                    ai_responses.append({
                        "content": final_answer.strip(),
                        "checkpoint_index": checkpoint_index
                    })
                    print__analysis_tracing_debug(f"307 - AI MESSAGE FOUND: Found final_answer: {final_answer[:100]}...")
        
        # Pass 3: Create properly ordered conversation with guaranteed prompt->answer sequence
        print__analysis_tracing_debug(f"308 - MESSAGE PAIRING: Creating conversation with {len(user_prompts)} prompts and {len(ai_responses)} responses")
        
        # Create conversation messages with proper ordering
        conversation_messages = []
        message_counter = 0
        
        # Handle multiple prompts and responses by pairing them chronologically
        for i in range(max(len(user_prompts), len(ai_responses))):
            # Add user prompt if available
            if i < len(user_prompts):
                prompt = user_prompts[i]
                message_counter += 1
                user_message = {
                    "id": f"user_{message_counter}",
                    "content": prompt["content"],
                    "is_user": True,
                    "timestamp": datetime.fromtimestamp(1700000000 + message_counter * 1000),
                    "checkpoint_order": prompt["checkpoint_index"],
                    "message_order": message_counter
                }
                conversation_messages.append(user_message)
                print__analysis_tracing_debug(f"309 - ADDED USER MESSAGE: Message {message_counter}: {prompt['content'][:50]}...")
            
            # Add AI response if available (always after the user prompt)
            if i < len(ai_responses):
                response = ai_responses[i]
                message_counter += 1
                ai_message = {
                    "id": f"ai_{message_counter}",
                    "content": response["content"],
                    "is_user": False,
                    "timestamp": datetime.fromtimestamp(1700000000 + message_counter * 1000),
                    "checkpoint_order": response["checkpoint_index"],
                    "message_order": message_counter
                }
                conversation_messages.append(ai_message)
                print__analysis_tracing_debug(f"310 - ADDED AI MESSAGE: Message {message_counter}: {response['content'][:100]}...")
        
        # Messages are already in correct order, no need to sort by timestamp
        print__analysis_tracing_debug(f"311 - CONVERSATION SUCCESS: Created {len(conversation_messages)} conversation messages in proper order")
        
        # OPTIMIZATION: Only log first few messages in detail
        for i, msg in enumerate(conversation_messages[:6]):  # Show first 6 messages (3 pairs)
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print__analysis_tracing_debug(f"312 - MESSAGE {i+1}: {msg_type}: {msg['content'][:50]}...")
        
        if len(conversation_messages) > 6:
            print__analysis_tracing_debug(f"312 - MESSAGE SUMMARY: ...and {len(conversation_messages) - 6} more messages")
        
        return conversation_messages
        
    except Exception as e:
        print__analysis_tracing_debug(f"311 - CONVERSATION ERROR: Error retrieving messages from checkpoints: {str(e)}")
        print__analysis_tracing_debug(f"312 - CONVERSATION TRACEBACK: Full traceback: {traceback.format_exc()}")
        return []

# HELPER FUNCTIONS FOR COMPATIBILITY - USING DIRECT CONNECTIONS
@retry_on_prepared_statement_error(max_retries=2)
async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new thread run entry in the database.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    print__analysis_tracing_debug(f"286 - CREATE THREAD ENTRY START: Creating thread run entry for user={email}, thread={thread_id}")
    try:
        if not run_id:
            run_id = str(uuid.uuid4())
            print__analysis_tracing_debug(f"287 - GENERATE RUN ID: Generated new run_id: {run_id}")
        
        print__analysis_tracing_debug(f"288 - DATABASE INSERT: Inserting thread run entry with run_id={run_id}")
        
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO users_threads_runs (email, thread_id, run_id, prompt)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        thread_id = EXCLUDED.thread_id,
                        prompt = EXCLUDED.prompt,
                        timestamp = CURRENT_TIMESTAMP
                """, (email, thread_id, run_id, prompt))
        
        print__analysis_tracing_debug(f"289 - CREATE THREAD ENTRY SUCCESS: Thread run entry created successfully: {run_id}")
        return run_id
    except Exception as e:
        print__analysis_tracing_debug(f"290 - CREATE THREAD ENTRY ERROR: Failed to create thread run entry: {e}")
        # Return the run_id even if database storage fails
        if not run_id:
            run_id = str(uuid.uuid4())
        print__analysis_tracing_debug(f"291 - CREATE THREAD ENTRY FALLBACK: Returning run_id despite database error: {run_id}")
        return run_id

@retry_on_prepared_statement_error(max_retries=2)
async def get_user_chat_threads(email: str, connection_pool=None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        print__api_postgresql(f"Getting chat threads for user: {email} (limit: {limit}, offset: {offset})")
        
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                base_query = """
                    SELECT 
                        thread_id,
                        MAX(timestamp) as latest_timestamp,
                        COUNT(*) as run_count,
                        (SELECT prompt FROM users_threads_runs utr2 
                         WHERE utr2.email = %s AND utr2.thread_id = utr.thread_id 
                         ORDER BY timestamp ASC LIMIT 1) as first_prompt
                    FROM users_threads_runs utr
                    WHERE email = %s
                    GROUP BY thread_id
                    ORDER BY latest_timestamp DESC
                """
                
                params = [email, email]
                
                if limit is not None:
                    base_query += " LIMIT %s OFFSET %s"
                    params.extend([limit, offset])
                
                await cur.execute(base_query, params)
                rows = await cur.fetchall()
                
                threads = []
                for row in rows:
                    thread_id = row[0]
                    latest_timestamp = row[1]
                    run_count = row[2]
                    first_prompt = row[3]
                    
                    title = (first_prompt[:47] + "...") if first_prompt and len(first_prompt) > 50 else (first_prompt or "Untitled Conversation")
                    
                    threads.append({
                        "thread_id": thread_id,
                        "latest_timestamp": latest_timestamp,
                        "run_count": run_count,
                        "title": title,
                        "full_prompt": first_prompt or ""
                    })
                
                print__api_postgresql(f"Retrieved {len(threads)} threads for user {email}")
                return threads
            
    except Exception as e:
        print__api_postgresql(f"Failed to get chat threads for user {email}: {e}")
        # Return empty list instead of raising exception to prevent API crashes
        print__api_postgresql(f"Returning empty threads list due to error")
        return []

@retry_on_prepared_statement_error(max_retries=2)
async def get_user_chat_threads_count(email: str, connection_pool=None) -> int:
    """Get total count of chat threads for a user.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(DISTINCT thread_id) as total_threads
                    FROM users_threads_runs
                    WHERE email = %s
                """, (email,))
                
                result = await cur.fetchone()
                total_count = result[0] if result else 0
            
            return total_count or 0
            
    except Exception as e:
        print__api_postgresql(f"Failed to get chat threads count for user {email}: {e}")
        # Return 0 instead of raising exception to prevent API crashes
        print__api_postgresql(f"Returning 0 thread count due to error")
        return 0

@retry_on_prepared_statement_error(max_retries=2)
async def update_thread_run_sentiment(run_id: str, sentiment: bool, user_email: str = None) -> bool:
    """Update sentiment for a thread run.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        print__api_postgresql(f"Updating sentiment for run {run_id}: {sentiment}")
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s
                """, (sentiment, run_id))
                updated = cur.rowcount
        print__api_postgresql(f"Updated sentiment for {updated} entries")
        return int(updated) > 0
    except Exception as e:
        print__api_postgresql(f"Failed to update sentiment: {e}")
        return False

@retry_on_prepared_statement_error(max_retries=2)
async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get all sentiments for a thread.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        print__api_postgresql(f"Getting sentiments for thread {thread_id}")
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT run_id, sentiment 
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s AND sentiment IS NOT NULL
                """, (email, thread_id))
                rows = await cur.fetchall()
        sentiments = {row[0]: row[1] for row in rows}
        print__api_postgresql(f"Retrieved {len(sentiments)} sentiments")
        return sentiments
    except Exception as e:
        print__api_postgresql(f"Failed to get sentiments: {e}")
        return {}

@retry_on_prepared_statement_error(max_retries=2)
async def delete_user_thread_entries(email: str, thread_id: str, connection_pool=None) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        print__api_postgresql(f"Deleting thread entries for user: {email}, thread: {thread_id}")
        
        async with get_direct_connection() as conn:
            # First, count the entries to be deleted
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (email, thread_id))
                result = await cur.fetchone()
                entries_to_delete = result[0] if result else 0
            
            print__api_postgresql(f"Found {entries_to_delete} entries to delete")
            
            if entries_to_delete == 0:
                return {
                    "deleted_count": 0,
                    "message": "No entries found to delete",
                    "thread_id": thread_id,
                    "user_email": email
                }
            
            # Delete the entries
            async with conn.cursor() as cur:
                await cur.execute("""
                    DELETE FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (email, thread_id))
                deleted_count = cur.rowcount
            
            print__api_postgresql(f"Deleted {deleted_count} entries for user {email}, thread {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email
            }
            
    except Exception as e:
        print__api_postgresql(f"Failed to delete thread entries for user {email}, thread {thread_id}: {e}")
        print__api_postgresql(f"Full traceback: {traceback.format_exc()}")
        raise

# UNIFIED CHECKPOINTER MANAGEMENT - SHARED BY MAIN.PY AND API_SERVER.PY
#==============================================================================
# Global checkpointer variable for unified management
_GLOBAL_CHECKPOINTER = None

async def initialize_checkpointer():
    """Initialize the global checkpointer with proper async context management."""
    global _GLOBAL_CHECKPOINTER
    if _GLOBAL_CHECKPOINTER is None:
        try:
            print__analysis_tracing_debug("🚀 UNIFIED: Initializing PostgreSQL Connection System...")
            print__analysis_tracing_debug(f"🔍 UNIFIED: Current global checkpointer state: {_GLOBAL_CHECKPOINTER}")
            
            # Create and initialize the checkpointer using the OFFICIAL AsyncPostgresSaver method
            print__analysis_tracing_debug("🔧 UNIFIED: Creating PostgreSQL checkpointer using official factory method...")
            
            checkpointer = await create_async_postgres_saver()
            
            print__analysis_tracing_debug(f"✅ UNIFIED: Created checkpointer type: {type(checkpointer).__name__}")
            
            # Set the global checkpointer directly (no wrapper needed)
            _GLOBAL_CHECKPOINTER = checkpointer
            
            print__analysis_tracing_debug("✅ UNIFIED: PostgreSQL checkpointer initialized successfully using official AsyncPostgresSaver")
            
        except Exception as e:
            print__analysis_tracing_debug(f"❌ UNIFIED: PostgreSQL checkpointer initialization failed: {e}")
            print__analysis_tracing_debug("🔄 UNIFIED: Falling back to InMemorySaver...")
            _GLOBAL_CHECKPOINTER = MemorySaver()

async def cleanup_checkpointer():
    """Clean up the global checkpointer on shutdown."""
    global _GLOBAL_CHECKPOINTER
    
    print__analysis_tracing_debug("🧹 UNIFIED: Starting checkpointer cleanup...")
    
    if _GLOBAL_CHECKPOINTER:
        try:
            # Check if it's an AsyncPostgresSaver that needs proper cleanup
            if hasattr(_GLOBAL_CHECKPOINTER, '__class__') and 'AsyncPostgresSaver' in str(type(_GLOBAL_CHECKPOINTER)):
                print__analysis_tracing_debug("🔄 UNIFIED: Cleaning up AsyncPostgresSaver...")
                # Use the proper cleanup function
                await close_async_postgres_saver()
            else:
                print__analysis_tracing_debug(f"🔄 UNIFIED: Cleaning up {type(_GLOBAL_CHECKPOINTER).__name__}...")
                # For other types (like MemorySaver), no special cleanup needed
                
        except Exception as e:
            print__analysis_tracing_debug(f"⚠️ UNIFIED: Error during checkpointer cleanup: {e}")
        finally:
            _GLOBAL_CHECKPOINTER = None
            print__analysis_tracing_debug("✅ UNIFIED: Checkpointer cleanup completed")
    else:
        print__analysis_tracing_debug("ℹ️ UNIFIED: No checkpointer to clean up")

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance, initializing if needed."""
    global _GLOBAL_CHECKPOINTER
    
    # With the unified approach, we don't need complex health checking
    if _GLOBAL_CHECKPOINTER is None:
        await initialize_checkpointer()
    
    return _GLOBAL_CHECKPOINTER

# Add the missing function back after setup_users_threads_runs_table 
@retry_on_prepared_statement_error(max_retries=2)
async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str):
    """Get queries and results from the latest checkpoint for a thread.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    print__analysis_tracing_debug(f"279 - GET CHECKPOINT START: Getting queries and results from latest checkpoint for thread: {thread_id}")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get the latest checkpoint
        print__analysis_tracing_debug("280 - GET CHECKPOINT STATE: Getting latest checkpoint state")
        state_snapshot = await checkpointer.aget_tuple(config)
        
        if not state_snapshot or not state_snapshot.checkpoint:
            print__analysis_tracing_debug(f"281 - NO CHECKPOINT: No checkpoint found for thread: {thread_id}")
            return []
        
        print__analysis_tracing_debug("282 - EXTRACT CHECKPOINT: Extracting queries and results from checkpoint")
        # Extract queries and results from checkpoint
        checkpoint = state_snapshot.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        
        # Look for queries_and_results in various places
        queries_and_results = channel_values.get("queries_and_results", [])
        
        if not queries_and_results:
            print__analysis_tracing_debug("283 - SEARCH ITERATIONS: Searching iteration_results for queries")
            # Try to extract from iteration_results
            iteration_results = channel_values.get("iteration_results", {})
            for iteration_key, iteration_data in iteration_results.items():
                if isinstance(iteration_data, dict):
                    iter_queries = iteration_data.get("queries_and_results", [])
                    if iter_queries:
                        queries_and_results.extend(iter_queries)
        
        print__analysis_tracing_debug(f"284 - GET CHECKPOINT SUCCESS: Found {len(queries_and_results)} queries and results for thread: {thread_id}")
        return queries_and_results
        
    except Exception as e:
        print__analysis_tracing_debug(f"285 - GET CHECKPOINT ERROR: Error getting queries and results from checkpoint: {e}")
        return []

#==============================================================================
# MODERN PSYCOPG CONNECTION POOL CONTEXT MANAGER
#==============================================================================
@asynccontextmanager
async def modern_psycopg_pool():
    """
    Modern async context manager for psycopg connection pools.
    Uses the recommended approach from psycopg documentation to avoid deprecation warnings.
    
    Usage:
        async with modern_psycopg_pool() as pool:
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
    """
    print__analysis_tracing_debug("MODERN POOL CONTEXT START: Creating modern psycopg connection pool context")
    
    try:        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        print__analysis_tracing_debug("MODERN POOL CONTEXT: Setting up AsyncConnectionPool with modern context management")
        
        # Use the modern async context manager approach recommended by psycopg
        async with AsyncConnectionPool(
            conninfo=connection_string,
            min_size=1,
            max_size=3,
            timeout=20,
            max_idle=300,
            max_lifetime=1800,
            kwargs={
                **connection_kwargs,
                "connect_timeout": 15,
            },
            open=False  # Explicitly set to avoid deprecation warnings
        ) as pool:
            print__analysis_tracing_debug("MODERN POOL CONTEXT: Pool created and opened using modern context manager")
            yield pool
            print__analysis_tracing_debug("MODERN POOL CONTEXT: Pool will be automatically closed by context manager")
    
    except ImportError as e:
        print__analysis_tracing_debug(f"MODERN POOL CONTEXT ERROR: psycopg_pool not available: {e}")
        raise Exception("psycopg_pool is required for modern connection pool approach")
    except Exception as e:
        print__analysis_tracing_debug(f"MODERN POOL CONTEXT ERROR: Failed to create modern pool: {e}")
        raise

if __name__ == "__main__":    
    async def test():
        print__postgresql_debug("Testing PostgreSQL checkpointer with official AsyncPostgresSaver...")
        
        if not check_postgres_env_vars():
            print__postgresql_debug("Environment variables not set properly")
            return
        
        checkpointer = await create_async_postgres_saver()
        print__postgresql_debug(f"Checkpointer type: {type(checkpointer).__name__}")
        
        # Test a simple operation
        config = {"configurable": {"thread_id": "test_thread"}}
        try:
            # This should work with the official AsyncPostgresSaver
            async for checkpoint in checkpointer.alist(config, limit=1):
                print__postgresql_debug("alist() method works correctly")
                break
            else:
                print__postgresql_debug("No checkpoints found (expected for fresh DB)")
        except Exception as e:
            print__postgresql_debug(f"Error testing alist(): {e}")
        
        await close_async_postgres_saver()
        print__postgresql_debug("Official AsyncPostgresSaver test completed!")
    
    asyncio.run(test())