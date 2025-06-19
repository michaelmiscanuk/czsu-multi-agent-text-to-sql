#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using the official langgraph checkpoint postgres functionality.
This uses the correct table schemas and implementation from the langgraph library.
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
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
    
    # AGGRESSIVE WINDOWS FIX: Force SelectorEventLoop before any other async operations
    print__postgres_startup_debug(f"🔧 PostgreSQL module: Windows detected - forcing SelectorEventLoop for PostgreSQL compatibility")
    
    # Set the policy first - this is CRITICAL and must happen before any async operations
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print__postgres_startup_debug(f"🔧 PostgreSQL module: Windows event loop policy set to: {type(asyncio.get_event_loop_policy()).__name__}")
    
    # Force close any existing event loop and create a fresh SelectorEventLoop
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop and not current_loop.is_closed():
            print__postgres_startup_debug(f"🔧 PostgreSQL module: Closing existing {type(current_loop).__name__}")
            current_loop.close()
    except RuntimeError:
        # No event loop exists yet, which is fine
        pass
    
    # Create a new SelectorEventLoop explicitly and set it as the running loop
    new_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
    asyncio.set_event_loop(new_loop)
    print__postgres_startup_debug(f"🔧 PostgreSQL module: Created new {type(new_loop).__name__}")
    
    # Verify the fix worked - this is critical for PostgreSQL compatibility
    try:
        current_loop = asyncio.get_event_loop()
        print__postgres_startup_debug(f"🔧 PostgreSQL module: Current event loop type: {type(current_loop).__name__}")
        if "Selector" in type(current_loop).__name__:
            print__postgres_startup_debug(f"✅ PostgreSQL module: PostgreSQL should work correctly on Windows now")
        else:
            print__postgres_startup_debug(f"⚠️ PostgreSQL module: Event loop fix may not have worked properly")
            # FORCE FIX: If we still don't have a SelectorEventLoop, create one
            print__postgres_startup_debug(f"🔧 PostgreSQL module: Force-creating SelectorEventLoop...")
            if not current_loop.is_closed():
                current_loop.close()
            selector_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
            asyncio.set_event_loop(selector_loop)
            print__postgres_startup_debug(f"🔧 PostgreSQL module: Force-created {type(selector_loop).__name__}")
    except RuntimeError:
        print__postgres_startup_debug(f"🔧 PostgreSQL module: No event loop set yet (will be created as needed)")

import asyncio
import platform
import os
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # Correct async import
from langgraph.checkpoint.postgres import PostgresSaver  # Correct sync import
from psycopg_pool import AsyncConnectionPool, ConnectionPool
import threading
from contextlib import asynccontextmanager

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
        print(f"[PostgreSQL-Debug] {msg}")
        import sys
        sys.stdout.flush()

def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[API-PostgreSQL] {msg}")
        import sys
        sys.stdout.flush()

# Database connection parameters
database_pool: Optional[AsyncConnectionPool] = None
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
    """Force close all connections - useful when hitting connection limits."""
    global database_pool
    
    if database_pool is not None:
        try:
            print__postgresql_debug("🧹 Force closing all database connections...")
            await database_pool.close()
            print__postgresql_debug("✓ All connections force closed")
        except Exception as e:
            print__postgresql_debug(f"⚠ Error force closing connections: {e}")
        finally:
            database_pool = None

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
    """Get PostgreSQL connection string from environment variables with basic Supabase configuration."""
    config = get_db_config()
    
    print__postgres_startup_debug(f"🔗 Building connection string for Supabase")
    
    # FIXED: Basic Supabase connection string (prepare_threshold must be set in connection kwargs, not URL)
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode=require"                     # Required for Supabase
        f"&connect_timeout=20"                  # Basic connection timeout
        f"&application_name=czsu_agent"         # App identification
    )
    
    # Log connection details (without password)
    debug_string = connection_string.replace(config['password'], '***')
    print__postgres_startup_debug(f"🔗 Using Supabase connection string (prepared statements disabled via pool kwargs): {debug_string}")
    
    return connection_string

async def is_pool_healthy(pool: Optional[AsyncConnectionPool]) -> bool:
    """Check if a connection pool is healthy and open."""
    if pool is None:
        return False
    try:
        # Check if pool is closed
        if pool.closed:
            print__postgresql_debug(f"⚠ Pool is marked as closed")
            return False
        
        # Try a quick connection test
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        print__postgresql_debug(f"⚠ Pool health check failed: {e}")
        return False

async def create_fresh_connection_pool() -> AsyncConnectionPool:
    """Create a fresh connection pool with simple, reliable configuration."""
    print__postgresql_debug(f"🔄 Creating fresh PostgreSQL connection pool...")
    
    connection_string = get_connection_string()
    
    try:
        # FIXED: Enhanced connection pool with prepared statement protection for Supabase/pgBouncer
        pool = AsyncConnectionPool(
            connection_string, 
            min_size=1,          # Start with 1 connection
            max_size=5,          # Max 5 connections to avoid Supabase limits
            open=False,          # Don't open in constructor to avoid deprecation warning
            kwargs={
                "prepare_threshold": None,      # CRITICAL: Disable prepared statements at connection level
                "autocommit": True,             # Use autocommit for better compatibility
            }
        )
        
        # Open the pool properly
        await pool.open()
        
        # Simple connection test
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        
        print__postgresql_debug(f"✅ Fresh connection pool created successfully with prepared statement protection")
        return pool
        
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to create connection pool: {e}")
        raise

async def get_healthy_pool() -> AsyncConnectionPool:
    """Get a healthy connection pool with proper concurrent access protection."""
    global database_pool
    
    async with _get_pool_lock():  # Ensure only one thread can modify the pool at a time
        print__postgresql_debug(f"🔒 Acquired pool lock for health check")
        
        # Check current active operations before making changes
        active_ops = await get_active_operations_count()
        print__postgresql_debug(f"📊 Current active operations: {active_ops}")
        
        # If we have an existing pool, check if it's healthy
        if database_pool is not None:
            try:
                # Don't close pool if operations are active
                if active_ops > 0:
                    print__postgresql_debug(f"⚠️ {active_ops} operations active - skipping pool health check to prevent closure")
                    return database_pool
                
                is_healthy = await is_pool_healthy(database_pool)
                if is_healthy:
                    print__postgresql_debug(f"✅ Existing pool is healthy")
                    return database_pool
                else:
                    print__postgresql_debug(f"⚠️ Existing pool is unhealthy, will recreate")
                    # Wait for active operations to complete before closing
                    max_wait = 30  # Maximum 30 seconds to wait
                    wait_time = 0
                    while active_ops > 0 and wait_time < max_wait:
                        print__postgresql_debug(f"⏳ Waiting for {active_ops} active operations to complete...")
                        await asyncio.sleep(1)
                        wait_time += 1
                        active_ops = await get_active_operations_count()
                    
                    if active_ops > 0:
                        print__postgresql_debug(f"⚠️ Timeout waiting for operations to complete - will not close pool")
                        return database_pool  # Return existing pool rather than risk breaking active operations
                    
                    # Safe to close now
                    try:
                        if not database_pool.closed:
                            await database_pool.close()
                            print__postgresql_debug(f"🔒 Closed unhealthy pool safely")
                    except Exception as e:
                        print__postgresql_debug(f"⚠️ Error closing unhealthy pool: {e}")
                    database_pool = None
                    
            except Exception as e:
                print__postgresql_debug(f"❌ Error checking pool health: {e}")
                # Don't close pool on health check errors if operations are active
                if active_ops > 0:
                    print__postgresql_debug(f"⚠️ Health check failed but {active_ops} operations active - keeping pool")
                    return database_pool
                # Otherwise, mark for recreation
                database_pool = None
        
        # Create new pool if needed
        if database_pool is None:
            print__postgresql_debug(f"🔄 Creating new connection pool...")
            try:
                database_pool = await create_fresh_connection_pool()
                print__postgresql_debug(f"✅ New connection pool created successfully")
            except Exception as e:
                print__postgresql_debug(f"❌ Failed to create new pool: {e}")
                raise
        
        return database_pool

async def setup_users_threads_runs_table():
    """Setup the users_threads_runs table for chat management."""
    
    # Get a healthy pool
    pool = await get_healthy_pool()
    
    try:
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Create users_threads_runs table with all 5 columns including prompt (50 char limit)
            # Use IF NOT EXISTS to preserve existing data on server restarts
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users_threads_runs (
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    email VARCHAR(255) NOT NULL,
                    thread_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) PRIMARY KEY,
                    prompt VARCHAR(50),
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
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_timestamp 
                ON users_threads_runs(email, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread_timestamp 
                ON users_threads_runs(email, thread_id, timestamp);
            """)
            
            # Index on run_id for feedback/sentiment functionality (explicit, though PK already provides this)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_run_id 
                ON users_threads_runs(run_id);
            """)
            
            # Enable RLS if this is Supabase
            try:
                await conn.execute("ALTER TABLE users_threads_runs ENABLE ROW LEVEL SECURITY")
                print__postgresql_debug("✓ RLS enabled on users_threads_runs")
            except Exception as e:
                if "already enabled" in str(e).lower():
                    print__postgresql_debug("⚠ RLS already enabled on users_threads_runs")
                else:
                    print__postgresql_debug(f"⚠ Warning: Could not enable RLS on users_threads_runs: {e}")
            
            # Create RLS policy
            try:
                await conn.execute('DROP POLICY IF EXISTS "Allow service role full access" ON users_threads_runs')
                await conn.execute("""
                    CREATE POLICY "Allow service role full access" ON users_threads_runs
                    FOR ALL USING (true) WITH CHECK (true)
                """)
                print__postgresql_debug("✓ RLS policy created for users_threads_runs")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print__postgresql_debug("⚠ RLS policy already exists for users_threads_runs")
                else:
                    print__postgresql_debug(f"⚠ Could not create RLS policy for users_threads_runs: {e}")
            
            print__postgresql_debug("✅ users_threads_runs table verified/created (6 columns: timestamp, email, thread_id, run_id, prompt, sentiment)")
            
    except Exception as e:
        print__postgresql_debug(f"❌ Error setting up users_threads_runs table: {str(e)}")
        raise

async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new entry in users_threads_runs table.
    
    Args:
        email: User's email address
        thread_id: Thread ID for the conversation
        prompt: The user's prompt/question for this run (will be truncated to 50 chars)
        run_id: Optional run ID, will generate UUID if not provided
    
    Returns:
        The run_id that was created or provided
    """
    
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    # Truncate prompt to 50 characters to fit database constraint
    truncated_prompt = None
    was_truncated = False
    if prompt:
        if len(prompt) > 50:
            truncated_prompt = prompt[:50]
            was_truncated = True
        else:
            truncated_prompt = prompt
    
    try:
        # Get a healthy pool
        pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Insert new entry with truncated prompt - run_id is primary key so must be unique
            await conn.execute("""
                INSERT INTO users_threads_runs (timestamp, email, thread_id, run_id, prompt)
                VALUES (NOW(), %s, %s, %s, %s)
            """, (email, thread_id, run_id, truncated_prompt))
            
            original_length = len(prompt) if prompt else 0
            truncated_length = len(truncated_prompt) if truncated_prompt else 0
            print__postgresql_debug(f"✓ Created thread run entry: email={email}, thread_id={thread_id}, run_id={run_id}")
            print__postgresql_debug(f"  prompt: '{truncated_prompt}' (original: {original_length} chars, stored: {truncated_length} chars, truncated: {was_truncated})")
            return run_id
            
    except Exception as e:
        print__postgresql_debug(f"❌ Error creating thread run entry: {str(e)}")
        raise

async def update_thread_run_sentiment(run_id: str, sentiment: bool, user_email: str = None) -> bool:
    """Update sentiment for a specific run_id.
    
    Args:
        run_id: The run ID to update
        sentiment: True for thumbs up, False for thumbs down, None to clear
        user_email: User's email address for ownership verification (recommended for security)
    
    Returns:
        True if update was successful, False otherwise
    """
    
    try:
        # Get a healthy pool
        pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # 🔒 SECURITY: If user_email is provided, verify ownership before updating
            if user_email:
                # Check if this user owns the run_id
                ownership_result = await conn.execute("""
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE run_id = %s AND email = %s
                """, (run_id, user_email))
                
                ownership_row = await ownership_result.fetchone()
                ownership_count = ownership_row[0] if ownership_row else 0
                
                if ownership_count == 0:
                    print__postgresql_debug(f"🚫 SECURITY: User {user_email} does not own run_id {run_id} - sentiment update denied")
                    return False
                
                print__postgresql_debug(f"✅ SECURITY: User {user_email} owns run_id {run_id} - sentiment update authorized")
                
                # Update sentiment with user verification
                result = await conn.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s AND email = %s
                """, (sentiment, run_id, user_email))
            else:
                # Legacy mode: Update without user verification (less secure)
                print__postgresql_debug(f"⚠ WARNING: Updating sentiment without user verification for run_id {run_id}")
                result = await conn.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s
                """, (sentiment, run_id))
            
            updated_count = result.rowcount if hasattr(result, 'rowcount') else 0
            
            if updated_count > 0:
                print__postgresql_debug(f"✓ Updated sentiment for run_id {run_id}: {sentiment}")
                return True
            else:
                print__postgresql_debug(f"⚠ No rows updated for run_id {run_id} - run_id may not exist or access denied")
                return False
                
    except Exception as e:
        print__postgresql_debug(f"❌ Error updating sentiment for run_id {run_id}: {str(e)}")
        return False

async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get all sentiment values for a user's thread.
    
    Args:
        email: User's email address
        thread_id: Thread ID to get sentiments for
    
    Returns:
        Dictionary mapping run_id to sentiment value (True/False/None)
    """
    
    try:
        # Get a healthy pool
        pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            # Get all run_ids and their sentiments for this thread
            result = await conn.execute("""
                SELECT run_id, sentiment 
                FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
                ORDER BY timestamp ASC
            """, (email, thread_id))
            
            sentiments = {}
            async for row in result:
                run_id = row[0]
                sentiment = row[1]  # This will be True, False, or None
                sentiments[run_id] = sentiment
            
            print__postgresql_debug(f"✓ Retrieved {len(sentiments)} sentiment values for thread {thread_id}")
            return sentiments
            
    except Exception as e:
        print__postgresql_debug(f"❌ Error retrieving sentiments for thread {thread_id}: {str(e)}")
        return {}

async def get_user_chat_threads(email: str, connection_pool=None) -> List[Dict[str, Any]]:
    """Get all chat threads for a user with first prompt as title, sorted by latest timestamp.
    
    Args:
        email: User's email address
        connection_pool: Optional connection pool to use (defaults to healthy pool)
    
    Returns:
        List of dictionaries with thread information including first prompt as title:
        [{"thread_id": str, "latest_timestamp": datetime, "run_count": int, "title": str, "full_prompt": str}, ...]
    """
    
    # Use provided pool or get a healthy pool
    if connection_pool:
        pool_to_use = connection_pool
    else:
        pool_to_use = await get_healthy_pool()
    
    try:
        async with pool_to_use.connection() as conn:
            # First, let's check if we have any data for this user
            count_result = await conn.execute("""
                SELECT COUNT(*) FROM users_threads_runs WHERE email = %s
            """, (email,))
            
            count_row = await count_result.fetchone()
            total_records = count_row[0] if count_row else 0
            print__postgresql_debug(f"🔍 Total records for user {email}: {total_records}")
            
            if total_records == 0:
                print__postgresql_debug(f"⚠ No records found for user {email}")
                return []
            
            # Get unique threads with their latest timestamp, run count, and first prompt as title
            # We need to get the ORIGINAL prompt from the first run to show proper tooltip
            result = await conn.execute("""
                WITH thread_stats AS (
                    SELECT 
                        thread_id,
                        MAX(timestamp) as latest_timestamp,
                        COUNT(*) as run_count
                    FROM users_threads_runs 
                    WHERE email = %s 
                    GROUP BY thread_id
                ),
                first_prompts AS (
                    SELECT DISTINCT ON (thread_id)
                        thread_id,
                        prompt as first_prompt
                    FROM users_threads_runs 
                    WHERE email = %s AND prompt IS NOT NULL AND prompt != ''
                    ORDER BY thread_id, timestamp ASC
                )
                SELECT 
                    ts.thread_id,
                    ts.latest_timestamp,
                    ts.run_count,
                    COALESCE(fp.first_prompt, 'New Chat') as full_prompt
                FROM thread_stats ts
                LEFT JOIN first_prompts fp ON ts.thread_id = fp.thread_id
                ORDER BY ts.latest_timestamp DESC
            """, (email, email))
            
            threads = []
            async for row in result:
                print__postgresql_debug(f"🔍 Raw row: {row}")
                
                # Get the full prompt from database (already truncated to 50 chars)
                full_prompt = row[3] if row[3] else 'New Chat'
                
                # Create display title (truncate to 47 chars + "..." for UI layout)
                display_title = full_prompt
                if len(full_prompt) > 47:
                    display_title = full_prompt[:47] + "..."
                
                thread_info = {
                    "thread_id": row[0],
                    "latest_timestamp": row[1],
                    "run_count": row[2],
                    "title": display_title,
                    "full_prompt": full_prompt  # For tooltip
                }
                
                print__postgresql_debug(f"🔍 Thread info: title='{display_title}', full_prompt='{full_prompt}'")
                threads.append(thread_info)
            
            print__postgresql_debug(f"✓ Retrieved {len(threads)} chat threads for user: {email}")
            return threads
            
    except Exception as e:
        print__postgresql_debug(f"❌ Error retrieving user chat threads: {str(e)}")
        import traceback
        print__postgresql_debug(f"🔍 Full traceback: {traceback.format_exc()}")
        return []

async def delete_user_thread_entries(email: str, thread_id: str, connection_pool=None) -> Dict[str, Any]:
    """Delete all entries for a specific user's thread.
    
    Args:
        email: User's email address
        thread_id: Thread ID to delete
        connection_pool: Optional connection pool to use (defaults to healthy pool)
    
    Returns:
        Dictionary with deletion results
    """
    
    # Use provided pool or get a healthy pool
    if connection_pool:
        pool_to_use = connection_pool
    else:
        try:
            pool_to_use = await get_healthy_pool()
        except Exception as e:
            print__postgresql_debug(f"⚠ Could not get healthy pool for deletion: {e}")
            return {
                "deleted_count": 0,
                "email": email,
                "thread_id": thread_id,
                "error": f"No connection pool available: {e}"
            }
    
    try:
        async with pool_to_use.connection() as conn:
            await conn.set_autocommit(True)
            
            # Delete entries for this user's thread
            result = await conn.execute("""
                DELETE FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (email, thread_id))
            
            deleted_count = result.rowcount if hasattr(result, 'rowcount') else 0
            
            print__postgresql_debug(f"✓ Deleted {deleted_count} thread entries from users_threads_runs for user: {email}, thread_id: {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "email": email,
                "thread_id": thread_id
            }
            
    except Exception as e:
        print__postgresql_debug(f"❌ Error deleting user thread entries from users_threads_runs: {str(e)}")
        return {
            "deleted_count": 0,
            "email": email,
            "thread_id": thread_id,
            "error": str(e)
        }

def check_postgres_env_vars():
    """Check if all required PostgreSQL environment variables are present."""
    required_vars = ['user', 'password', 'host', 'dbname']
    missing_vars = []
    
    config = get_db_config()
    
    for var in required_vars:
        if not config.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print__postgresql_debug(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    print__postgresql_debug(f"✅ All required PostgreSQL environment variables present")
    return True

async def test_basic_postgres_connection():
    """Test basic PostgreSQL connectivity without pools or langgraph."""
    try:
        import psycopg
        
        config = get_db_config()
        connection_string = get_connection_string()
        
        print__postgresql_debug(f"🔍 Testing basic Supabase connection...")
        print__postgresql_debug(f"🔍 Host: {config['host']}")
        print__postgresql_debug(f"🔍 Port: {config['port']}")
        print__postgresql_debug(f"🔍 Database: {config['dbname']}")
        print__postgresql_debug(f"🔍 User: {config['user']}")
        print__postgresql_debug(f"🔍 SSL Mode: REQUIRED (Supabase)")
        
        # WINDOWS FIX: Ensure we're using SelectorEventLoop for PostgreSQL compatibility
        if sys.platform == "win32":
            print__postgresql_debug(f"🔧 Windows detected - ensuring SelectorEventLoop for Supabase connection")
            
            # Check current event loop type
            try:
                current_loop = asyncio.get_event_loop()
                current_loop_type = type(current_loop).__name__
                print__postgresql_debug(f"🔧 Current event loop type: {current_loop_type}")
                
                # If we're not using a SelectorEventLoop, we need to switch permanently
                if "Selector" not in current_loop_type:
                    print__postgresql_debug(f"🔧 ProactorEventLoop detected - switching to SelectorEventLoop for PostgreSQL compatibility")
                    
                    # Close the ProactorEventLoop
                    if not current_loop.is_closed():
                        current_loop.close()
                    
                    # Create and set a new SelectorEventLoop
                    selector_policy = asyncio.WindowsSelectorEventLoopPolicy()
                    selector_loop = selector_policy.new_event_loop()
                    asyncio.set_event_loop(selector_loop)
                    
                    print__postgresql_debug(f"🔧 Switched to {type(selector_loop).__name__} permanently for PostgreSQL compatibility")
                else:
                    print__postgresql_debug(f"✅ Already using SelectorEventLoop - PostgreSQL should work correctly")
                    
            except RuntimeError:
                # No event loop exists, create a SelectorEventLoop
                print__postgresql_debug(f"🔧 No event loop exists - creating SelectorEventLoop")
                selector_policy = asyncio.WindowsSelectorEventLoopPolicy()
                selector_loop = selector_policy.new_event_loop()
                asyncio.set_event_loop(selector_loop)
                print__postgresql_debug(f"🔧 Created {type(selector_loop).__name__}")
        
        # Test connection with the current event loop (should be SelectorEventLoop on Windows)
        async with await psycopg.AsyncConnection.connect(
            connection_string,
            autocommit=True,
            connect_timeout=15  # Match pool settings
        ) as conn:
            # Simple query test
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 as test, NOW() as current_time, version() as pg_version")
                result = await cur.fetchone()
                print__postgresql_debug(f"✅ Basic Supabase connection successful!")
                print__postgresql_debug(f"   Test result: {result[0]}")
                print__postgresql_debug(f"   Server time: {result[1]}")
                print__postgresql_debug(f"   PostgreSQL version: {result[2][:50]}...")
                return True
                
    except Exception as e:
        error_msg = str(e).lower()
        print__postgresql_debug(f"❌ Basic Supabase connection failed: {e}")
        print__postgresql_debug(f"🔍 Error type: {type(e).__name__}")
        
        # Enhanced error handling for event loop issues
        if "proactoreventloop" in error_msg or "cannot use the 'proactoreventloop'" in error_msg:
            print__postgresql_debug("💡 Event Loop Issue - PostgreSQL requires SelectorEventLoop on Windows:")
            print__postgresql_debug("   1. The application will attempt to switch event loops")
            print__postgresql_debug("   2. If this persists, restart the application")
            print__postgresql_debug("   3. Ensure no other code is forcing ProactorEventLoop")
        elif "ssl" in error_msg:
            print__postgresql_debug("💡 SSL Connection Issue - Supabase requires SSL:")
            print__postgresql_debug("   1. Verify your connection string uses sslmode=require")
            print__postgresql_debug("   2. Check if your IP is whitelisted in Supabase dashboard")
            print__postgresql_debug("   3. Verify your database credentials are correct")
        elif "authentication" in error_msg or "password" in error_msg:
            print__postgresql_debug("💡 Authentication Issue:")
            print__postgresql_debug("   1. Verify your database password is correct")
            print__postgresql_debug("   2. Check your database user has proper permissions")
        elif "timeout" in error_msg or "connection" in error_msg:
            print__postgresql_debug("💡 Connection Timeout Issue:")
            print__postgresql_debug("   1. Check your network connectivity")
            print__postgresql_debug("   2. Verify Supabase service is running")
            print__postgresql_debug("   3. Check if your IP is allowed in Supabase firewall")
        
        return False

async def get_postgres_checkpointer():
    """
    Get a PostgreSQL checkpointer with simple, reliable configuration.
    """
    
    try:
        # Step 1: Check environment variables
        print__postgresql_debug("🔍 Checking environment variables...")
        if not check_postgres_env_vars():
            raise Exception("Missing required PostgreSQL environment variables")
        
        # Step 2: Test basic connectivity  
        print__postgresql_debug("🔍 Testing basic Supabase connectivity...")
        basic_connection_ok = await test_basic_postgres_connection()
        
        if not basic_connection_ok:
            print__postgresql_debug("❌ Basic Supabase connectivity failed")
            raise Exception("Supabase server is not reachable or credentials are invalid")
        
        print__postgresql_debug("✅ Basic Supabase connectivity confirmed")
        
        # Step 3: Create simple connection pool
        print__postgresql_debug("🔍 Creating Supabase connection pool...")
        pool = await get_healthy_pool()
        
        # Step 4: Create checkpointer
        print__postgresql_debug("🔍 Creating PostgreSQL checkpointer...")
        checkpointer = AsyncPostgresSaver(pool)
        
        # Step 5: Setup tables
        print__postgresql_debug("🔍 Setting up database tables...")
        await checkpointer.setup()
        await setup_users_threads_runs_table()
        
        print__postgresql_debug("✅ PostgreSQL checkpointer initialized successfully")
        
        # Step 6: Wrap with resilient checkpointer for basic error handling
        resilient_checkpointer = ResilientPostgreSQLCheckpointer(checkpointer)
        print__postgresql_debug("✅ Wrapped with resilient checkpointer")
        
        return resilient_checkpointer
        
    except Exception as e:
        error_msg = str(e).lower()
        print__postgresql_debug(f"❌ Error creating PostgreSQL checkpointer: {e}")
        
        # Simple error categorization
        if any(keyword in error_msg for keyword in [
            "timeout", "connection", "network", "ssl", "authentication", "password"
        ]):
            print__postgresql_debug("💡 Check Supabase connection and credentials")
        
        raise Exception("Supabase server is not reachable or credentials are invalid")

def get_sync_postgres_checkpointer():
    """
    Get a synchronous PostgreSQL checkpointer using the official library.
    """
    try:
        connection_string = get_connection_string()
        
        # Create sync connection pool with simplified settings for compatibility
        pool = ConnectionPool(
            conninfo=connection_string,
            max_size=3,  # Match async pool settings
            min_size=1,  # Match async pool settings
            timeout=60,  # Timeout for acquiring a connection from pool
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,  # Disable prepared statements
                "connect_timeout": 30,  # Connection establishment timeout
            },
            open=False
        )
        
        # Create checkpointer with the connection pool
        checkpointer = PostgresSaver(pool)
        
        # Setup tables (this creates all required tables with correct schemas)
        checkpointer.setup()
        
        print__postgresql_debug("✅ Sync PostgreSQL checkpointer initialized successfully (max_size=3) with enhanced stability")
        return checkpointer
        
    except Exception as e:
        print__postgresql_debug(f"❌ Error creating sync PostgreSQL checkpointer: {str(e)}")
        raise

# For backward compatibility
async def create_postgres_checkpointer():
    """Backward compatibility wrapper."""
    return await get_postgres_checkpointer()

async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str, user_email: str = None) -> List[Dict[str, Any]]:
    """Get the COMPLETE conversation messages from the LangChain PostgreSQL checkpoint history.
    
    SECURITY: Only loads messages for threads owned by the specified user to prevent data leakage.
    
    This extracts ALL user questions and ALL AI responses for proper chat display:
    - All user messages: for right-side blue display
    - All AI messages: for left-side white display using the explicit final_answer from state
    
    Args:
        checkpointer: The PostgreSQL checkpointer instance
        thread_id: Thread ID for the conversation
        user_email: Email of the user requesting the messages (for security verification)
    
    Returns:
        List of message dictionaries in chronological order (complete conversation history)
    """
    try:
        print__api_postgresql(f"🔍 Retrieving COMPLETE checkpoint history for thread: {thread_id}")
        
        # 🔒 SECURITY CHECK: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__api_postgresql(f"🔒 Verifying thread ownership for user: {user_email}")
            
            # Get a connection to verify ownership
            pool = checkpointer.conn if hasattr(checkpointer, 'conn') else await get_healthy_pool()
            
            if pool:
                async with pool.connection() as conn:
                    ownership_result = await conn.execute("""
                        SELECT COUNT(*) FROM users_threads_runs 
                        WHERE email = %s AND thread_id = %s
                    """, (user_email, thread_id))
                    
                    ownership_row = await ownership_result.fetchone()
                    thread_entries_count = ownership_row[0] if ownership_row else 0
                    
                    if thread_entries_count == 0:
                        print__api_postgresql(f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - access denied")
                        return []  # Return empty instead of loading other users' data
                    
                    print__api_postgresql(f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
            else:
                print__api_postgresql(f"⚠ Could not verify thread ownership - connection pool unavailable")
                # In case of connection issues, don't load data to be safe
                return []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints for this thread using alist()
        checkpoint_tuples = []
        try:
            # Fix: alist() returns an async generator, don't await it
            checkpoint_iterator = checkpointer.alist(config)
            
            # Now we can iterate over the async iterator
            async for checkpoint_tuple in checkpoint_iterator:
                checkpoint_tuples.append(checkpoint_tuple)
                
        except Exception as alist_error:
            print__api_postgresql(f"❌ Error getting checkpoint list: {alist_error}")
            # Try alternative approach if alist fails
            try:
                # Alternative: use aget_tuple to get the latest checkpoint
                state_snapshot = await checkpointer.aget_tuple(config)
                if state_snapshot:
                    checkpoint_tuples = [state_snapshot]
                    print__api_postgresql(f"⚠ Using fallback method - got latest checkpoint only")
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
    """Get queries_and_results from the latest checkpoint state.
    
    Args:
        checkpointer: The PostgreSQL checkpointer instance
        thread_id: Thread ID for the conversation
    
    Returns:
        List of [query, result] pairs from the latest checkpoint
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await checkpointer.aget_tuple(config)
        
        if state_snapshot and state_snapshot.checkpoint:
            channel_values = state_snapshot.checkpoint.get("channel_values", {})
            queries_and_results = channel_values.get("queries_and_results", [])
            print__api_postgresql(f"✅ Found {len(queries_and_results)} queries from latest checkpoint")
            return [[query, result] for query, result in queries_and_results]
        
        return []
        
    except Exception as e:
        print__api_postgresql(f"⚠ Could not get queries from checkpoint: {e}")
        return []

async def setup_rls_policies(pool: AsyncConnectionPool):
    """Setup Row Level Security policies for checkpointer tables in Supabase."""
    try:
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Enable RLS on all checkpointer tables
            tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"]
            
            for table in tables:
                try:
                    await conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                    print__api_postgresql(f"✓ RLS enabled on {table}")
                except Exception as e:
                    if "already enabled" in str(e).lower() or "does not exist" in str(e).lower():
                        print__api_postgresql(f"⚠ RLS already enabled on {table} or table doesn't exist")
                    else:
                        print__api_postgresql(f"⚠ Warning: Could not enable RLS on {table}: {e}")
            
            # Drop existing policies if they exist
            for table in tables:
                try:
                    await conn.execute(f'DROP POLICY IF EXISTS "Allow service role full access" ON {table}')
                except Exception as e:
                    print__api_postgresql(f"⚠ Could not drop existing policy on {table}: {e}")
            
            # Create permissive policies for authenticated users
            for table in tables:
                try:
                    await conn.execute(f"""
                        CREATE POLICY "Allow service role full access" ON {table}
                        FOR ALL USING (true) WITH CHECK (true)
                    """)
                    print__api_postgresql(f"✓ RLS policy created for {table}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print__api_postgresql(f"⚠ RLS policy already exists for {table}")
                    else:
                        print__api_postgresql(f"⚠ Could not create RLS policy for {table}: {e}")
            
        print__api_postgresql("✓ Row Level Security setup completed (with any warnings noted above)")
    except Exception as e:
        print__api_postgresql(f"⚠ Warning: Could not setup RLS policies: {e}")
        # Don't fail the entire setup if RLS setup fails - this is not critical for basic functionality

async def monitor_connection_health(pool: AsyncConnectionPool, interval: int = 60):
    """Monitor connection pool health in the background."""
    try:
        while True:
            try:
                # Quick health check
                async with pool.connection() as conn:
                    await conn.execute("SELECT 1")
                
                # Get pool statistics if available
                try:
                    stats = pool.get_stats()
                    print__api_postgresql(f"✓ Connection pool health OK - Stats: {stats}")
                except AttributeError:
                    print__api_postgresql("✓ Connection pool health check passed")
            except Exception as e:
                print__api_postgresql(f"⚠ Connection pool health check failed: {e}")
            
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        print__api_postgresql("📊 Connection monitor stopped")

def log_connection_info(host: str, port: str, dbname: str, user: str):
    """Log connection information for debugging."""
    print__api_postgresql(f"🔗 PostgreSQL Connection Info:")
    print__api_postgresql(f"   Host: {host}")
    print__api_postgresql(f"   Port: {port}")
    print__api_postgresql(f"   Database: {dbname}")
    print__api_postgresql(f"   User: {user}")
    print__api_postgresql(f"   SSL: Required")

# Test and health check functions
async def test_pool_connection():
    """Test creating and using a connection pool."""
    try:
        print__api_postgresql("🔍 Testing pool connection...")
        
        # Test pool creation
        pool = await create_fresh_connection_pool()
        print__api_postgresql(f"✅ Pool created successfully: closed={pool.closed}")
        
        # Test pool usage
        async with pool.connection() as conn:
            result = await conn.execute("SELECT 1 as test, NOW() as current_time")
            row = await result.fetchone()
            print__api_postgresql(f"✅ Pool query successful: {row}")
        
        # Test pool health check
        is_healthy = await is_pool_healthy(pool)
        print__api_postgresql(f"✅ Pool health check: {is_healthy}")
        
        # Cleanup
        await pool.close()
        print__api_postgresql(f"✅ Pool closed: closed={pool.closed}")
        
        return True
        
    except Exception as e:
        print__api_postgresql(f"❌ Pool connection test failed: {e}")
        return False

async def debug_pool_status():
    """Debug function to show current pool status."""
    global database_pool
    
    print__api_postgresql(f"🔍 Pool Status Debug:")
    print__api_postgresql(f"   Global pool exists: {database_pool is not None}")
    
    if database_pool:
        print__api_postgresql(f"   Pool closed: {database_pool.closed}")
        try:
            # Try to get pool stats if available
            if hasattr(database_pool, 'get_stats'):
                stats = database_pool.get_stats()
                print__api_postgresql(f"   Pool stats: {stats}")
            else:
                print__api_postgresql(f"   Pool stats: Not available")
                
            # Test health
            is_healthy = await is_pool_healthy(database_pool)
            print__api_postgresql(f"   Pool healthy: {is_healthy}")
            
        except Exception as e:
            print__api_postgresql(f"   Pool status error: {e}")
    
    return database_pool

async def test_connection_health():
    """Test the health of the PostgreSQL connection using proper Supabase settings."""
    try:
        config = get_db_config()
        
        if not all([config['user'], config['password'], config['host'], config['dbname']]):
            print__api_postgresql("❌ Missing required environment variables for database connection")
            return False
            
        # Use the same connection string as the main application for consistency
        connection_string = get_connection_string()
        
        print__api_postgresql("🔍 Testing Supabase connection health...")
        
        # Test basic connection using same settings as main pool
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=1,
            min_size=1,
            timeout=10,  # Short timeout for health check
            open=False,  # Don't open in constructor to avoid deprecation warning
            kwargs={
                "prepare_threshold": None,      # CRITICAL: Disable prepared statements for Supabase
                "autocommit": True,             # Use autocommit for better compatibility
            }
        )
        
        await pool.open()  # Open the pool properly
        
        async with pool.connection() as conn:
            result = await conn.execute("SELECT 1 as test, NOW() as current_time")
            row = await result.fetchone()
            if row and row[0] == 1:
                print__api_postgresql("✓ Supabase connection health check successful")
                await pool.close()
                return True
        
        await pool.close()
        return False
        
    except Exception as e:
        error_msg = str(e).lower()
        print__api_postgresql(f"❌ Supabase connection health check failed: {e}")
        
        # Provide specific guidance
        if "ssl" in error_msg:
            print__api_postgresql("💡 SSL Issue: Verify Supabase credentials and IP whitelist")
        elif "timeout" in error_msg:
            print__api_postgresql("💡 Timeout Issue: Check network connectivity to Supabase")
        elif "authentication" in error_msg:
            print__api_postgresql("💡 Auth Issue: Verify database credentials")
            
        return False

class ResilientPostgreSQLCheckpointer:
    """
    A wrapper around PostgreSQLCheckpointer that handles connection failures gracefully
    with simple retry logic.
    """
    
    def __init__(self, base_checkpointer):
        self.base_checkpointer = base_checkpointer
        self.max_retries = 2  # Simple retry count
        
    async def _simple_retry(self, operation_name, operation_func, *args, **kwargs):
        """Simple retry logic for basic connection failures."""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await operation_func(*args, **kwargs)
                if attempt > 0:
                    print__postgresql_debug(f"✅ [{operation_name}] Succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Only retry connection-related errors
                is_retryable = any(keyword in error_msg for keyword in [
                    "connection", "timeout", "network", "ssl", "closed"
                ])
                
                if attempt < self.max_retries and is_retryable:
                    delay = 1 + attempt  # Simple delay: 1s, 2s
                    print__postgresql_debug(f"⚠ [{operation_name}] Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    print__postgresql_debug(f"❌ [{operation_name}] Failed after {attempt + 1} attempts: {e}")
                    raise
    
    # Delegate all operations to the base checkpointer with simple retry
    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._simple_retry("aput", self.base_checkpointer.aput, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id):
        return await self._simple_retry("aput_writes", self.base_checkpointer.aput_writes, config, writes, task_id)

    async def aget(self, config):
        return await self._simple_retry("aget", self.base_checkpointer.aget, config)

    async def aget_tuple(self, config):
        return await self._simple_retry("aget_tuple", self.base_checkpointer.aget_tuple, config)

    async def alist(self, config, filter=None, before=None, limit=None):
        # alist returns an async generator, so we need to yield from it
        # We can't apply retry logic to async generators, so we delegate directly
        async for item in self.base_checkpointer.alist(config):
            yield item

    def __getattr__(self, name):
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

if __name__ == "__main__":
    async def test():
        print__postgresql_debug("Testing PostgreSQL connection...")
        
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host") 
        port = os.getenv("port", "5432")
        dbname = os.getenv("dbname")
        
        print__postgresql_debug(f"User: {user}")
        print__postgresql_debug(f"Host: {host}")
        print__postgresql_debug(f"Port: {port}")
        print__postgresql_debug(f"Database: {dbname}")
        print__postgresql_debug(f"Password configured: {bool(password)}")
        
        # Test connection health first
        health_ok = await test_connection_health()
        if not health_ok:
            print__postgresql_debug("❌ Basic connectivity test failed")
            return
        
        # Test full checkpointer setup
        checkpointer = await get_postgres_checkpointer()
        print__postgresql_debug(f"Checkpointer type: {type(checkpointer).__name__}")
        
        # Cleanup
        if hasattr(checkpointer, 'pool') and checkpointer.pool:
            await checkpointer.pool.close()
            print__postgresql_debug("✓ Connection pool closed")
    
    asyncio.run(test()) 