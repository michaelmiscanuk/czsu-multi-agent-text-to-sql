#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using asyncpg for all operations.
Simplified version based on working commit 108 to fix infinite loops and conversation loading issues.
"""

from __future__ import annotations

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
import sys
import os

def print__postgres_startup_debug(msg: str) -> None:
    """Print PostgreSQL startup debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[POSTGRES-STARTUP-DEBUG] {msg}")
        sys.stdout.flush()

if sys.platform == "win32":
    import asyncio
    print__postgres_startup_debug("Windows detected - ensuring asyncpg compatibility")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio
import platform
import os
import uuid
import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta
import asyncpg
import psycopg
import psycopg_pool
import threading
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Import LangGraph's built-in PostgreSQL checkpointer
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    print__postgres_startup_debug("LangGraph AsyncPostgresSaver imported successfully")
except ImportError as e:
    print__postgres_startup_debug(f"Failed to import AsyncPostgresSaver: {e}")
    AsyncPostgresSaver = None

#==============================================================================
# GLOBALS AND LOCKS - SIMPLIFIED VERSION
#==============================================================================
database_pool: Optional[asyncpg.Pool] = None
psycopg_pool_instance: Optional[psycopg_pool.AsyncConnectionPool] = None
_pool_lock = threading.Lock()
_psycopg_pool_lock = threading.Lock()
_checkpointer_setup_done = False
_checkpointer_setup_lock = asyncio.Lock()
_operations_lock = threading.Lock()
_active_operations = 0

#==============================================================================
# DEBUG FUNCTIONS
#==============================================================================
def print__postgresql_debug(msg: str) -> None:
    """Print PostgreSQL debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[POSTGRESQL-DEBUG] {msg}")
        sys.stdout.flush()

def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled."""
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[API-PostgreSQL] {msg}")
        sys.stdout.flush()

async def debug_pool_status() -> Dict[str, Any]:
    """Get debug information about the current pool status."""
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
    }
    
    if database_pool:
        status["asyncpg_pool"]["closed"] = getattr(database_pool, "_closed", None)
        status["asyncpg_pool"]["min_size"] = getattr(database_pool, "_min_size", None)
        status["asyncpg_pool"]["max_size"] = getattr(database_pool, "_max_size", None)
        
        try:
            current_size = len(getattr(database_pool, '_queue', []))
            status["asyncpg_pool"]["current_size"] = current_size
        except Exception:
            pass
    
    return status

def _get_pool_lock():
    """Get the pool lock for thread-safe operations."""
    return _pool_lock

def _get_operations_lock():
    """Get the operations lock for thread-safe operations."""
    return _operations_lock

async def increment_active_operations():
    """Increment active operations counter."""
    global _active_operations
    with _get_operations_lock():
        _active_operations += 1

async def decrement_active_operations():
    """Decrement active operations counter."""
    global _active_operations
    with _get_operations_lock():
        _active_operations = max(0, _active_operations - 1)

async def get_active_operations_count():
    """Get current active operations count."""
    global _active_operations
    with _get_operations_lock():
        return _active_operations

async def force_close_psycopg_pool():
    """Force close the psycopg connection pool."""
    global psycopg_pool_instance
    print__postgresql_debug("Force closing psycopg connection pool...")
    
    if psycopg_pool_instance:
        try:
            await psycopg_pool_instance.close()
            print__postgresql_debug("Psycopg pool closed successfully")
        except Exception as e:
            print__postgresql_debug(f"Error closing psycopg pool: {e}")
        finally:
            psycopg_pool_instance = None

async def force_close_all_connections():
    """Force close all database connections."""
    global database_pool
    print__postgresql_debug("Force closing all database connections...")
    
    # Close both asyncpg and psycopg pools
    await force_close_psycopg_pool()
    
    if database_pool:
        try:
            await database_pool.close()
            print__postgresql_debug("Database pool closed successfully")
        except Exception as e:
            print__postgresql_debug(f"Error closing database pool: {e}")
        finally:
            database_pool = None

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }

def get_connection_string():
    """Get PostgreSQL connection string for LangGraph checkpointer with proper configuration."""
    config = get_db_config()
    
    # Use only VALID PostgreSQL connection parameters from the official documentation
    # Based on https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNECT-PARAMKEYWORDS
    # Add timestamp to application name to avoid prepared statement conflicts
    timestamp = int(time.time())
    connection_params = [
        "sslmode=require",  # Valid SSL mode parameter
        f"application_name=czsu_langgraph_{timestamp}"  # Unique application name to avoid conflicts
    ]
    
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"{'&'.join(connection_params)}"
    )
    
    return connection_string

async def is_pool_healthy(pool: Optional[asyncpg.Pool]) -> bool:
    """Simple health check for the pool."""
    if not pool or pool._closed:
        return False
    
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        print__postgresql_debug(f"Pool health check failed: {e}")
        return False

async def create_fresh_connection_pool() -> Optional[asyncpg.Pool]:
    """Create a fresh asyncpg connection pool."""
    try:
        config = get_db_config()
        print__postgresql_debug(f"Creating asyncpg pool: host={config['host']}, db={config['dbname']}")
        
        pool = await asyncpg.create_pool(
            user=config['user'],
            password=config['password'],
            database=config['dbname'],
            host=config['host'],
            port=config['port'],
            min_size=2,
            max_size=10,
            statement_cache_size=0  # Disable for pgbouncer compatibility
        )
        
        print__postgresql_debug("Fresh asyncpg pool created successfully")
        return pool
        
    except Exception as e:
        print__postgresql_debug(f"Failed to create fresh pool: {e}")
        return None

async def get_healthy_pool() -> asyncpg.Pool:
    """Get a healthy asyncpg connection pool - SIMPLIFIED VERSION."""
    global database_pool
    
    # Check if pool exists and is healthy
    if database_pool is not None and not database_pool._closed:
        is_healthy = await is_pool_healthy(database_pool)
        if is_healthy:
            return database_pool
        else:
            print__postgresql_debug("Existing pool is unhealthy, closing...")
            try:
                await database_pool.close()
            except Exception:
                pass
            database_pool = None
    
    # Create new pool
    print__postgresql_debug("Creating new healthy pool...")
    with _get_pool_lock():
        # Double-check pattern
        if database_pool is not None and not database_pool._closed:
            return database_pool
        
        database_pool = await create_fresh_connection_pool()
        if not database_pool:
            raise Exception("Failed to create database pool")
        
        return database_pool

async def setup_users_threads_runs_table():
    """Setup the users_threads_runs table for tracking user conversations."""
    try:
        print__postgresql_debug("Setting up users_threads_runs table...")
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
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread 
                ON users_threads_runs(email, thread_id);
            """)
            
            print__postgresql_debug("users_threads_runs table and indexes created/verified")
            
    except Exception as e:
        print__postgresql_debug(f"Failed to setup users_threads_runs table: {e}")
        raise

async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new thread run entry in the database."""
    try:
        if not run_id:
            run_id = str(uuid.uuid4())
        
        print__api_postgresql(f"Creating thread run entry: user={email}, thread={thread_id}, run={run_id}")
        
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users_threads_runs (email, thread_id, run_id, prompt)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (run_id) DO UPDATE SET
                    email = EXCLUDED.email,
                    thread_id = EXCLUDED.thread_id,
                    prompt = EXCLUDED.prompt,
                    timestamp = CURRENT_TIMESTAMP
            """, email, thread_id, run_id, prompt)
        
        print__api_postgresql(f"Thread run entry created successfully: {run_id}")
        return run_id
        
    except Exception as e:
        print__api_postgresql(f"Failed to create thread run entry: {e}")
        raise

async def update_thread_run_sentiment(run_id: str, sentiment: bool, user_email: str = None) -> bool:
    """Update sentiment for a thread run."""
    try:
        print__api_postgresql(f"Updating sentiment for run {run_id}: {sentiment}")
        
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE users_threads_runs 
                SET sentiment = $1 
                WHERE run_id = $2
            """, sentiment, run_id)
        
        updated = result.split()[1] if result.startswith('UPDATE') else 0
        print__api_postgresql(f"Updated sentiment for {updated} entries")
        return int(updated) > 0
        
    except Exception as e:
        print__api_postgresql(f"Failed to update sentiment: {e}")
        return False

async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get all sentiments for a thread."""
    try:
        print__api_postgresql(f"Getting sentiments for thread {thread_id}")
        
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT run_id, sentiment 
                FROM users_threads_runs 
                WHERE email = $1 AND thread_id = $2 AND sentiment IS NOT NULL
            """, email, thread_id)
        
        sentiments = {row['run_id']: row['sentiment'] for row in rows}
        print__api_postgresql(f"Retrieved {len(sentiments)} sentiments")
        return sentiments
        
    except Exception as e:
        print__api_postgresql(f"Failed to get sentiments: {e}")
        return {}

async def get_user_chat_threads(email: str, connection_pool=None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination - SIMPLIFIED VERSION."""
    try:
        print__api_postgresql(f"Getting chat threads for user: {email} (limit: {limit}, offset: {offset})")
        
        if connection_pool:
            pool = connection_pool
        else:
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
            
            print__api_postgresql(f"Retrieved {len(threads)} threads for user {email}")
            return threads
            
    except Exception as e:
        print__api_postgresql(f"Failed to get chat threads for user {email}: {e}")
        import traceback
        print__api_postgresql(f"Full traceback: {traceback.format_exc()}")
        raise

async def get_user_chat_threads_count(email: str, connection_pool=None) -> int:
    """Get total count of chat threads for a user."""
    try:
        print__api_postgresql(f"Getting chat threads count for user: {email}")
        
        if connection_pool:
            pool = connection_pool
        else:
            pool = await get_healthy_pool()
        
        async with pool.acquire() as conn:
            total_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT thread_id) as total_threads
                FROM users_threads_runs
                WHERE email = $1
            """, email)
            
            print__api_postgresql(f"Total threads count for user {email}: {total_count}")
            return total_count or 0
            
    except Exception as e:
        print__api_postgresql(f"Failed to get chat threads count for user {email}: {e}")
        import traceback
        print__api_postgresql(f"Full traceback: {traceback.format_exc()}")
        raise

async def get_or_create_psycopg_pool() -> psycopg_pool.AsyncConnectionPool:
    """Get or create the psycopg connection pool, ensuring it's initialized only once."""
    global psycopg_pool_instance
    if psycopg_pool_instance:
        return psycopg_pool_instance

    # Use a thread-safe lock for pool creation
    with _psycopg_pool_lock:
        # Double-check inside the lock
        if psycopg_pool_instance:
            return psycopg_pool_instance

        print__postgresql_debug("Creating new psycopg connection pool...")
        connection_string = get_connection_string()
        
        # Connection arguments for psycopg, including disabling auto-prepared statements
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        
        # Create pool with open=False to avoid deprecation warning, then open it explicitly
        psycopg_pool_instance = psycopg_pool.AsyncConnectionPool(
            conninfo=connection_string,
            kwargs=connection_kwargs,
            min_size=2,
            max_size=10,
            open=False  # Explicitly set to False to avoid deprecation warning
        )
        
        # Open the pool explicitly using the modern approach
        await psycopg_pool_instance.open()
        print__postgresql_debug("Psycopg connection pool opened successfully.")
        return psycopg_pool_instance

# MODERN CONTEXT MANAGER APPROACH (recommended for new code)
@asynccontextmanager
async def modern_psycopg_pool():
    """
    Modern approach using async context manager for psycopg pool.
    This is the recommended approach that avoids deprecation warnings.
    
    Usage:
        async with modern_psycopg_pool() as pool:
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
    """
    connection_string = get_connection_string()
    
    # Connection arguments for psycopg, including disabling auto-prepared statements
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    
    print__postgresql_debug("Creating modern psycopg pool using context manager...")
    
    # Use the modern context manager approach recommended by psycopg
    async with psycopg_pool.AsyncConnectionPool(
        conninfo=connection_string,
        kwargs=connection_kwargs,
        min_size=2,
        max_size=10,
        open=False  # Explicitly set to avoid warning
    ) as pool:
        print__postgresql_debug("Modern psycopg pool opened successfully")
        yield pool
        print__postgresql_debug("Modern psycopg pool will be closed automatically")

async def cleanup_all_pools():
    """Clean up all connection pools - use this for graceful shutdown."""
    print__postgresql_debug("Starting cleanup of all connection pools...")
    await force_close_all_connections()
    print__postgresql_debug("All connection pools cleaned up successfully")

async def delete_user_thread_entries(email: str, thread_id: str, connection_pool=None) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table."""
    try:
        print__api_postgresql(f"Deleting thread entries for user: {email}, thread: {thread_id}")
        
        if connection_pool:
            pool = connection_pool
        else:
            pool = await get_healthy_pool()
        
        async with pool.acquire() as conn:
            # First, count the entries to be deleted
            entries_to_delete = await conn.fetchval("""
                SELECT COUNT(*) FROM users_threads_runs 
                WHERE email = $1 AND thread_id = $2
            """, email, thread_id)
            
            print__api_postgresql(f"Found {entries_to_delete} entries to delete")
            
            if entries_to_delete == 0:
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
            
            print__api_postgresql(f"Deleted {deleted_count} entries for user {email}, thread {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email
            }
            
    except Exception as e:
        print__api_postgresql(f"Failed to delete thread entries for user {email}, thread {thread_id}: {e}")
        import traceback
        print__api_postgresql(f"Full traceback: {traceback.format_exc()}")
        raise

def check_postgres_env_vars():
    """Check if all required PostgreSQL environment variables are set."""
    required_vars = ['host', 'port', 'dbname', 'user', 'password']
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print__postgres_startup_debug(f"Missing required environment variables: {missing_vars}")
        return False
    else:
        print__postgres_startup_debug("All required PostgreSQL environment variables are set")
        return True

async def test_basic_postgres_connection():
    """Test basic PostgreSQL connection using asyncpg."""
    try:
        print__postgres_startup_debug("Testing basic asyncpg PostgreSQL connection...")
        
        if not check_postgres_env_vars():
            return False
        
        config = get_db_config()
        
        # Test direct connection
        conn = await asyncpg.connect(
            user=config['user'],
            password=config['password'],
            database=config['dbname'],
            host=config['host'],
            port=config['port'],
            statement_cache_size=0
        )
        
        print__postgres_startup_debug("Direct connection established")
        
        # Get PostgreSQL version
        version = await conn.fetchval('SELECT version()')
        print__postgres_startup_debug(f"PostgreSQL version: {version}")
        
        await conn.close()
        print__postgres_startup_debug("Connection closed successfully")
        
        return True
    except Exception as e:
        print__postgres_startup_debug(f"Basic connection test failed: {e}")
        return False

async def test_connection_health():
    """Test the health of database connections."""
    try:
        print__postgresql_debug("Testing connection health...")
        
        # Test basic connection first
        if not await test_basic_postgres_connection():
            print__postgresql_debug("Basic connection test failed")
            return False
        
        # Test pool connection
        pool = await get_healthy_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 'health_check'")
            print__postgresql_debug(f"Pool connection test result: {result}")
        
        print__postgresql_debug("Connection health test passed")
        return True
        
    except Exception as e:
        print__postgresql_debug(f"Connection health test failed: {e}")
        return False

class PostgresCheckpointerManager:
    """
    Manages the LangGraph AsyncPostgresSaver lifecycle by using a connection
    pool to prevent errors related to prepared statements in a multi-worker environment.
    """

    def __init__(self):
        self.pool: Optional[psycopg_pool.AsyncConnectionPool] = None
        self.conn: Optional[psycopg.AsyncConnection] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    async def __aenter__(self) -> AsyncPostgresSaver:
        """Enter the async context, get a connection from the pool, and set up the checkpointer."""
        global _checkpointer_setup_done
        print__postgresql_debug("Starting PostgresCheckpointerManager context (pool-based)...")
        try:
            # Get a connection from the pool
            self.pool = await get_or_create_psycopg_pool()
            await self.pool.wait() # Ensure pool is ready before getting a connection
            self.conn = await self.pool.getconn()
            print__postgresql_debug("Got connection from psycopg pool.")

            # Instantiate the checkpointer with the connection
            self.checkpointer = AsyncPostgresSaver(conn=self.conn)

            # Set up the checkpointer schema once, using a lock to prevent race conditions
            async with _checkpointer_setup_lock:
                if not _checkpointer_setup_done:
                    try:
                        print__postgresql_debug("Running checkpointer setup for the first time...")
                        await self.checkpointer.setup()
                        print__postgresql_debug("AsyncPostgresSaver setup complete.")
                        _checkpointer_setup_done = True
                    except (psycopg.errors.DuplicateTable, psycopg.errors.DuplicateObject):
                        print__postgresql_debug("Checkpointer tables already exist. Skipping setup.")
                        _checkpointer_setup_done = True
                    except psycopg.errors.DuplicatePreparedStatement:
                        print__postgresql_debug("Ignoring duplicate prepared statement error during setup, assuming another worker succeeded.")
                        _checkpointer_setup_done = True
                    except Exception as e:
                        print__postgresql_debug(f"Error during initial checkpointer setup: {e}")
                        # We don't re-raise here to allow other workers to proceed,
                        # but we don't mark setup as done.
                        
            return self.checkpointer
        except Exception as e:
            print__postgresql_debug(f"Failed to setup PostgresCheckpointerManager: {e}")
            # Ensure connection is released on failure
            if self.conn and self.pool:
                await self.pool.putconn(self.conn)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager and release the connection to the pool."""
        print__postgresql_debug("Exiting PostgresCheckpointerManager context...")
        if self.conn and self.pool:
            await self.pool.putconn(self.conn)
            print__postgresql_debug("Psycopg connection returned to pool.")

async def get_postgres_checkpointer():
    """Get a PostgreSQL checkpointer manager - POOL-BASED VERSION."""
    try:
        print__postgresql_debug("Creating PostgreSQL checkpointer manager...")
        
        if not AsyncPostgresSaver:
            raise Exception("AsyncPostgresSaver not available")
        
        # Setup the users_threads_runs table first using our asyncpg pool
        await setup_users_threads_runs_table()
        
        # Create the checkpointer manager which now uses a connection pool
        checkpointer_manager = PostgresCheckpointerManager()
        print__postgresql_debug("PostgresCheckpointerManager created successfully")
        
        return checkpointer_manager
        
    except Exception as e:
        print__postgresql_debug(f"Failed to create PostgreSQL checkpointer manager: {e}")
        raise

async def get_postgres_checkpointer_with_context():
    """Get PostgreSQL checkpointer with proper initialization."""
    return await get_postgres_checkpointer()

# SIMPLIFIED ResilientPostgreSQLCheckpointer WITHOUT INFINITE RETRY LOOPS
class ResilientPostgreSQLCheckpointer:
    """Simplified resilient PostgreSQL checkpointer without aggressive retry logic."""
    
    def __init__(self, base_checkpointer):
        self.base_checkpointer = base_checkpointer
    
    async def _simple_retry(self, operation_name, operation_func, *args, **kwargs):
        """Simple retry logic with maximum 2 attempts."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                print__postgresql_debug(f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)  # Simple 1 second delay
    
    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._simple_retry("aput", self.base_checkpointer.aput, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id):
        return await self._simple_retry("aput_writes", self.base_checkpointer.aput_writes, config, writes, task_id)

    async def aget(self, config):
        return await self._simple_retry("aget", self.base_checkpointer.aget, config)

    async def aget_tuple(self, config):
        return await self._simple_retry("aget_tuple", self.base_checkpointer.aget_tuple, config)

    async def alist(self, config, filter=None, before=None, limit=None):
        """Simple alist without complex retry logic."""
        async for item in self.base_checkpointer.alist(config, filter=filter, before=before, limit=limit):
            yield item

    def __getattr__(self, name):
        """Delegate other attributes to the base checkpointer."""
        return getattr(self.base_checkpointer, name)

@asynccontextmanager
async def safe_pool_operation():
    """Context manager to safely track pool operations."""
    await increment_active_operations()
    try:
        pool = await get_healthy_pool()
        yield pool
    finally:
        await decrement_active_operations()

def get_sync_postgres_checkpointer():
    """Synchronous wrapper for getting PostgreSQL checkpointer."""
    return asyncio.run(get_postgres_checkpointer())

async def create_postgres_checkpointer():
    """Backward compatibility wrapper."""
    return await get_postgres_checkpointer()

async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str, user_email: str = None) -> List[Dict[str, Any]]:
    """Get conversation messages from checkpoints - FIXED VERSION BASED ON WORKING COMMIT 108."""
    try:
        print__api_postgresql(f"🔍 Retrieving COMPLETE checkpoint history for thread: {thread_id}")
        
        # 🔒 SECURITY CHECK: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__api_postgresql(f"🔒 Verifying thread ownership for user: {user_email}")
            
            try:
                # FIXED: Use our asyncpg pool instead of checkpointer.conn
                pool = await get_healthy_pool()
                async with pool.acquire() as conn:
                    thread_entries_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM users_threads_runs 
                        WHERE email = $1 AND thread_id = $2
                    """, user_email, thread_id)
                    
                    if thread_entries_count == 0:
                        print__api_postgresql(f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - access denied")
                        return []  # Return empty instead of loading other users' data
                    
                    print__api_postgresql(f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
            except Exception as e:
                print__api_postgresql(f"⚠ Could not verify thread ownership: {e}")
                return []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints for this thread using alist()
        checkpoint_tuples = []
        try:
            # Unwrap if needed
            base_checkpointer = checkpointer
            if hasattr(checkpointer, 'checkpointer'):
                base_checkpointer = checkpointer.checkpointer
            elif hasattr(checkpointer, 'base_checkpointer'):
                base_checkpointer = checkpointer.base_checkpointer
            
            print__api_postgresql(f"🔍 Using checkpointer type: {type(base_checkpointer).__name__}")
            
            # Get checkpoints using the base checkpointer - handle async iterator properly
            checkpoint_iterator = base_checkpointer.alist(config)
            
            # It's an async iterator - use it properly
            async for checkpoint_tuple in checkpoint_iterator:
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            error_msg = str(alist_error).lower()
            print__api_postgresql(f"❌ Error getting checkpoint list: {alist_error}")
            
            # The retry logic for prepared statements is removed because `prepare_threshold=0`
            # should prevent these errors from happening in the first place.
            
            # If we still don't have checkpoints, try fallback method
            if not checkpoint_tuples:
                print__api_postgresql(f"🔄 Trying fallback method to get latest checkpoint...")
                try:
                    # Alternative: use aget_tuple to get the latest checkpoint
                    state_snapshot = await base_checkpointer.aget_tuple(config)
                    if state_snapshot:
                        checkpoint_tuples = [state_snapshot]
                        print__api_postgresql(f"⚠️ Using fallback method - got latest checkpoint only")
                except Exception as fallback_error:
                    print__api_postgresql(f"❌ Fallback method also failed: {fallback_error}")
                    return []
        
        if not checkpoint_tuples:
            print__api_postgresql(f"⚠ No checkpoints found for thread: {thread_id}")
            return []
        
        print__api_postgresql(f"📄 Found {len(checkpoint_tuples)} checkpoints for verified thread")
        
        # Sort checkpoints chronologically (oldest first)
        checkpoint_tuples.sort(key=lambda x: x.config.get("configurable", {}).get("checkpoint_id", ""))
        
        # Extract conversation messages chronologically
        conversation_messages = []
        seen_prompts = set()
        seen_answers = set()
        
        # Extract all user prompts and AI responses from checkpoint history
        print__api_postgresql(f"🔍 Extracting ALL user questions and AI responses...")
        
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
                            print__api_postgresql(f"🔍 Found user prompt in checkpoint {checkpoint_index}: {prompt[:50]}...")
            
            # METHOD 2: Extract AI responses directly from final_answer in channel_values
            if "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                
                # Use explicit final_answer from state instead of trying to filter messages
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
                    print__api_postgresql(f"🤖 ✅ Found final_answer in checkpoint {checkpoint_index}: {final_answer[:100]}...")
        
        # Sort all messages by timestamp to ensure proper chronological order
        conversation_messages.sort(key=lambda x: x.get("timestamp", datetime.now()))
        
        # Re-assign sequential IDs and message order after sorting
        for i, msg in enumerate(conversation_messages):
            msg["message_order"] = i + 1
            msg["id"] = f"{'user' if msg['is_user'] else 'ai'}_{i + 1}"
        
        print__api_postgresql(f"✅ Extracted {len(conversation_messages)} conversation messages from COMPLETE history (verified user access)")
        
        # Debug: Log all messages found
        for i, msg in enumerate(conversation_messages):
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print__api_postgresql(f"{i+1}. {msg_type}: {msg['content'][:50]}...")
        
        return conversation_messages
        
    except Exception as e:
        print__api_postgresql(f"❌ Error retrieving COMPLETE messages from checkpoints: {str(e)}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        return []

async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str) -> List[List[str]]:
    """Get queries_and_results from the latest checkpoint state."""
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
            print__postgresql_debug(f"✅ Found {len(queries_and_results)} queries from latest checkpoint")
            return [[query, result] for query, result in queries_and_results]
        
        return []
        
    except Exception as e:
        print__postgresql_debug(f"⚠ Could not get queries from checkpoint: {e}")
        return []

if __name__ == "__main__":
    async def test():
        print__postgresql_debug("Testing PostgreSQL connection...")
        
        # Test connection health first
        health_ok = await test_connection_health()
        if not health_ok:
            print__postgresql_debug("Basic connectivity test failed")
            return
        
        # Test checkpointer setup
        checkpointer = await get_postgres_checkpointer()
        print__postgresql_debug(f"Checkpointer type: {type(checkpointer).__name__}")
        
        print__postgresql_debug("All tests passed!")
    
    asyncio.run(test())