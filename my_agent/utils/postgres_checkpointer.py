#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using the official langgraph checkpoint postgres functionality.
This uses the correct table schemas and implementation from the langgraph library.
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
if sys.platform == "win32":
    import asyncio
    
    # AGGRESSIVE WINDOWS FIX: Force SelectorEventLoop before any other async operations
    print(f"🔧 PostgreSQL module: Windows detected - forcing SelectorEventLoop for PostgreSQL compatibility")
    
    # Set the policy first - this is CRITICAL and must happen before any async operations
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"🔧 PostgreSQL module: Windows event loop policy set to: {type(asyncio.get_event_loop_policy()).__name__}")
    
    # Force close any existing event loop and create a fresh SelectorEventLoop
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop and not current_loop.is_closed():
            print(f"🔧 PostgreSQL module: Closing existing {type(current_loop).__name__}")
            current_loop.close()
    except RuntimeError:
        # No event loop exists yet, which is fine
        pass
    
    # Create a new SelectorEventLoop explicitly and set it as the running loop
    new_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
    asyncio.set_event_loop(new_loop)
    print(f"🔧 PostgreSQL module: Created new {type(new_loop).__name__}")
    
    # Verify the fix worked - this is critical for PostgreSQL compatibility
    try:
        current_loop = asyncio.get_event_loop()
        print(f"🔧 PostgreSQL module: Current event loop type: {type(current_loop).__name__}")
        if "Selector" in type(current_loop).__name__:
            print(f"✅ PostgreSQL module: PostgreSQL should work correctly on Windows now")
        else:
            print(f"⚠️ PostgreSQL module: Event loop fix may not have worked properly")
            # FORCE FIX: If we still don't have a SelectorEventLoop, create one
            print(f"🔧 PostgreSQL module: Force-creating SelectorEventLoop...")
            if not current_loop.is_closed():
                current_loop.close()
            selector_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
            asyncio.set_event_loop(selector_loop)
            print(f"🔧 PostgreSQL module: Force-created {type(selector_loop).__name__}")
    except RuntimeError:
        print(f"🔧 PostgreSQL module: No event loop set yet (will be created as needed)")

import asyncio
import platform
import os
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # Correct async import
from langgraph.checkpoint.postgres import PostgresSaver  # Correct sync import
from psycopg_pool import AsyncConnectionPool, ConnectionPool

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
_pool_lock = asyncio.Lock()  # Add global lock to prevent multiple pool creation

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
    """Get PostgreSQL connection string from environment variables with proper Supabase SSL configuration."""
    config = get_db_config()
    
    # SUPABASE FIX: Use proper SSL settings for Supabase
    # Supabase REQUIRES SSL connections - never use sslmode=disable
    
    # For Supabase, we need these SSL settings:
    # - sslmode=require: Force SSL connection (required by Supabase)
    # - connect_timeout: Prevent hanging connections
    # - application_name: Help identify connections in Supabase dashboard
    # NOTE: command_timeout is NOT a valid connection parameter - removed
    
    # Check if this is transaction mode (port 6543) for logging purposes
    is_transaction_mode = config['port'] == '6543'
    
    # Enhanced connection string for Supabase stability
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode=require"                     # REQUIRED for Supabase
        f"&connect_timeout=20"                  # Timeout for initial connection 
        f"&application_name=czsu_agent"         # Application identification
        f"&keepalives_idle=600"                 # Keep connection alive (10 minutes)
        f"&keepalives_interval=30"              # Send keepalive every 30 seconds
        f"&keepalives_count=3"                  # 3 failed keepalives before disconnect
        f"&tcp_user_timeout=30000"              # TCP timeout (30 seconds)
    )
    
    # Log transaction mode detection (removed invalid pgbouncer parameter)
    if is_transaction_mode:
        print__postgresql_debug(f"🔧 Detected Supabase transaction mode (port 6543) - using optimized settings")
    
    # Debug: Show what connection string we're actually using (without password)
    debug_string = connection_string.replace(config['password'], '***')
    print__postgresql_debug(f"🔍 Using Supabase-optimized connection string: {debug_string}")
    
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
    """Create a new connection pool with Supabase-optimized settings from environment variables."""
    connection_string = get_connection_string()
    
    # SUPABASE CRITICAL FIX: Check if we're using transaction mode (port 6543)
    config = get_db_config()
    is_transaction_mode = config['port'] == '6543'
    
    # Get pool settings from environment variables with Supabase-friendly defaults
    if is_transaction_mode:
        # TRANSACTION MODE (port 6543): Much more conservative settings
        max_size = int(os.getenv('POSTGRES_POOL_MAX', '1'))    # CRITICAL: Only 1 connection for transaction mode
        min_size = int(os.getenv('POSTGRES_POOL_MIN', '0'))    # Start with 0, create as needed
        timeout = int(os.getenv('POSTGRES_POOL_TIMEOUT', '10')) # Faster timeout for transaction mode
        print__postgresql_debug(f"🔧 TRANSACTION MODE detected (port 6543) - using conservative pool settings")
    else:
        # SESSION MODE (port 5432): Normal settings
        max_size = int(os.getenv('POSTGRES_POOL_MAX', '2'))    # Conservative for Supabase free tier
        min_size = int(os.getenv('POSTGRES_POOL_MIN', '0'))    # Start with 0, create as needed
        timeout = int(os.getenv('POSTGRES_POOL_TIMEOUT', '20')) # Normal timeout for session mode
        print__postgresql_debug(f"🔧 SESSION MODE detected (port 5432) - using normal pool settings")
    
    print__postgresql_debug(f"🔧 Creating Supabase-optimized connection pool with settings: max_size={max_size}, min_size={min_size}, timeout={timeout}")
    
    # Note: Removed invalid pgbouncer parameter that was causing connection failures
    # PostgreSQL drivers don't recognize 'pgbouncer=true' as a valid connection parameter
    
    # Supabase-optimized configuration for SSL connection stability
    pool_kwargs = {
        "autocommit": True,
        "prepare_threshold": None,  # CRITICAL: Disable prepared statements for Supabase transaction mode
        "connect_timeout": 10 if is_transaction_mode else 15  # Shorter timeout for transaction mode
    }
    
    # On Windows, add additional configuration for psycopg compatibility
    if sys.platform == "win32":
        print__postgresql_debug(f"🔧 Windows detected - configuring pool for Supabase + SelectorEventLoop compatibility")
        # Windows-specific psycopg configuration for SSL stability
        # Note: Removed server_settings as it's not a valid connection parameter
    
    # Create pool with Supabase-optimized settings
    pool = AsyncConnectionPool(
        conninfo=connection_string,
        max_size=max_size,
        min_size=min_size,
        timeout=timeout,
        kwargs=pool_kwargs,
        open=False
    )
    
    # Explicitly open the pool with enhanced error handling for Supabase
    try:
        # CRITICAL FIX: Reduce timeout for transaction mode
        pool_open_timeout = 15 if is_transaction_mode else 30
        
        # WINDOWS + SUPABASE FIX: Handle both Windows event loop issues and Supabase SSL
        if sys.platform == "win32":
            print__postgresql_debug(f"🔧 Opening pool with Windows + Supabase compatibility (timeout={pool_open_timeout}s)")
            
            # Try opening the pool in the current context first
            try:
                await asyncio.wait_for(pool.open(), timeout=pool_open_timeout)
                print__postgresql_debug(f"🔗 Pool opened successfully in current context")
            except Exception as e:
                error_msg = str(e).lower()
                
                if "proactoreventloop" in error_msg:
                    print__postgresql_debug(f"🔧 ProactorEventLoop issue detected, switching to SelectorEventLoop...")
                    
                    # Create a new SelectorEventLoop context
                    selector_policy = asyncio.WindowsSelectorEventLoopPolicy()
                    selector_loop = selector_policy.new_event_loop()
                    
                    try:
                        old_loop = asyncio.get_event_loop()
                        asyncio.set_event_loop(selector_loop)
                        
                        # Recreate the pool in the new context
                        pool = AsyncConnectionPool(
                            conninfo=connection_string,
                            max_size=max_size,
                            min_size=min_size,
                            timeout=timeout,
                            kwargs=pool_kwargs,
                            open=False
                        )
                        
                        await asyncio.wait_for(pool.open(), timeout=pool_open_timeout)
                        print__postgresql_debug(f"🔗 Pool opened successfully with SelectorEventLoop")
                        
                        # Restore original loop
                        asyncio.set_event_loop(old_loop)
                        
                    finally:
                        if not selector_loop.is_closed():
                            selector_loop.close()
                elif "ssl" in error_msg or "connection" in error_msg:
                    print__postgresql_debug(f"🔧 SSL/Connection issue with Supabase detected: {error_msg}")
                    raise Exception(f"Supabase SSL connection failed: {error_msg}")
                else:
                    raise
        else:
            # Non-Windows: Normal opening with Supabase timeout
            print__postgresql_debug(f"🔧 Opening pool with timeout={pool_open_timeout}s")
            await asyncio.wait_for(pool.open(), timeout=pool_open_timeout)
            
        mode_description = "TRANSACTION MODE (port 6543)" if is_transaction_mode else "SESSION MODE (port 5432)"
        print__postgresql_debug(f"🔗 Created fresh Supabase connection pool for {mode_description} (max_size={max_size}, min_size={min_size}, timeout={timeout}) with SSL stability")
        return pool
        
    except asyncio.TimeoutError:
        mode_description = "transaction mode (port 6543)" if is_transaction_mode else "session mode (port 5432)"
        print__postgresql_debug(f"❌ Timeout opening Supabase connection pool in {mode_description} - check network connectivity and pool size limits")
        if is_transaction_mode:
            print__postgresql_debug("💡 Transaction mode troubleshooting:")
            print__postgresql_debug("   1. Verify connection to port 6543 (transaction mode)")
            print__postgresql_debug("   2. Use max_size=1 for transaction mode")
            print__postgresql_debug("   3. Check Supabase pool size settings in dashboard")
            print__postgresql_debug("   4. Ensure SSL settings are correct for Supabase")
        raise Exception(f"Supabase {mode_description} connection timeout - check your network and database status")
    except Exception as e:
        error_msg = str(e).lower()
        if "ssl" in error_msg:
            print__postgresql_debug(f"❌ SSL error opening Supabase connection pool: {e}")
            print__postgresql_debug("💡 SSL Troubleshooting tips:")
            print__postgresql_debug("   1. Verify your Supabase credentials are correct")
            print__postgresql_debug("   2. Check if your IP is allowed in Supabase dashboard")
            print__postgresql_debug("   3. Ensure you're using the correct Supabase host URL")
            raise Exception(f"Supabase SSL connection failed: {e}")
        else:
            print__postgresql_debug(f"❌ Error opening Supabase connection pool: {e}")
            raise

async def get_healthy_pool() -> AsyncConnectionPool:
    """Get a healthy connection pool, creating a new one if needed."""
    global database_pool
    
    # Use global lock to prevent multiple pool creation
    async with _pool_lock:
        # Check if current pool is healthy
        if await is_pool_healthy(database_pool):
            return database_pool
        
        # Pool is unhealthy or doesn't exist, close old one if needed
        if database_pool is not None:
            try:
                print__postgresql_debug(f"🔄 Closing unhealthy pool...")
                await database_pool.close()
            except Exception as e:
                print__postgresql_debug(f"⚠ Error closing old pool: {e}")
            finally:
                database_pool = None
        
        # Create new pool
        print__postgresql_debug(f"🆕 Creating new connection pool...")
        database_pool = await create_fresh_connection_pool()
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
    Get a PostgreSQL checkpointer using the official langgraph PostgreSQL implementation.
    This ensures we use the correct table schemas and implementation with enhanced Supabase error handling.
    """
    
    max_attempts = 3
    base_delay = 3  # Start with 3 seconds
    
    for attempt in range(max_attempts):
        try:
            # First check environment variables
            print__postgresql_debug(f"🔍 Step 0 (attempt {attempt + 1}): Checking environment variables...")
            if not check_postgres_env_vars():
                raise Exception("Missing required PostgreSQL environment variables")
            
            # First test basic PostgreSQL connectivity
            print__postgresql_debug(f"🔍 Step 1 (attempt {attempt + 1}): Testing basic Supabase connectivity...")
            basic_connection_ok = await test_basic_postgres_connection()
            
            if not basic_connection_ok:
                if attempt < max_attempts - 1:
                    delay = base_delay * (attempt + 1)  # Progressive delay: 3s, 6s, 9s
                    print__postgresql_debug(f"❌ Basic Supabase connectivity failed - retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print__postgresql_debug("❌ Basic Supabase connectivity failed - cannot proceed")
                    raise Exception("Supabase server is not reachable or credentials are invalid")
            
            print__postgresql_debug(f"✅ Basic Supabase connectivity confirmed (attempt {attempt + 1})")
            
            # Get a healthy connection pool
            print__postgresql_debug(f"🔍 Step 2 (attempt {attempt + 1}): Creating Supabase connection pool...")
            pool = await get_healthy_pool()
            
            print__postgresql_debug(f"🔍 Step 3 (attempt {attempt + 1}): Creating PostgreSQL checkpointer with official library...")
            
            # Create checkpointer with the connection pool
            checkpointer = AsyncPostgresSaver(pool)
            
            print__postgresql_debug(f"🔍 Step 4 (attempt {attempt + 1}): Setting up langgraph tables...")
            # Setup tables (this creates all required tables with correct schemas)
            await checkpointer.setup()
            
            print__postgresql_debug(f"🔍 Step 5 (attempt {attempt + 1}): Setting up custom users_threads_runs table...")
            # Setup our custom users_threads_runs table
            await setup_users_threads_runs_table()
            
            print__postgresql_debug(f"✅ Official PostgreSQL checkpointer initialized successfully (attempt {attempt + 1})")
            
            # Wrap with resilient checkpointer to handle connection failures gracefully
            resilient_checkpointer = ResilientPostgreSQLCheckpointer(checkpointer)
            print__postgresql_debug(f"✅ Wrapped with resilient checkpointer for Supabase connection stability")
            
            return resilient_checkpointer
            
        except Exception as e:
            error_msg = str(e).lower()
            print__postgresql_debug(f"❌ Error creating PostgreSQL checkpointer (attempt {attempt + 1}): {e}")
            
            # Enhanced error categorization for Supabase
            is_retryable = False
            
            if any(keyword in error_msg for keyword in [
                "timeout", "connection", "network", "ssl", "connection reset", 
                "connection closed", "server closed", "temporarily unavailable"
            ]):
                is_retryable = True
                print__postgresql_debug("🔄 Connection/network error detected - this is retryable")
            elif any(keyword in error_msg for keyword in [
                "authentication", "password", "permission denied", "access denied"
            ]):
                print__postgresql_debug("🚫 Authentication error detected - not retryable")
                print__postgresql_debug("💡 Check your Supabase credentials:")
                print__postgresql_debug("   1. Verify database password")
                print__postgresql_debug("   2. Check user permissions")
                print__postgresql_debug("   3. Verify service role key")
                break  # Don't retry auth errors
            elif "missing required" in error_msg:
                print__postgresql_debug("🚫 Configuration error detected - not retryable")
                print__postgresql_debug("💡 Check your environment variables")
                break  # Don't retry config errors
            else:
                is_retryable = True  # Default to retryable for unknown errors
                print__postgresql_debug("❓ Unknown error - attempting retry")
            
            if attempt < max_attempts - 1 and is_retryable:
                # Progressive delay with jitter for Supabase rate limits
                delay = base_delay * (2 ** attempt) + (attempt * 2)  # 3s, 8s, 18s
                print__postgresql_debug(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                if not is_retryable:
                    print__postgresql_debug(f"❌ Non-retryable error encountered")
                else:
                    print__postgresql_debug(f"❌ All {max_attempts} attempts failed")
                    
                # Provide specific guidance based on error type
                if "ssl" in error_msg or "connection" in error_msg:
                    print__postgresql_debug("💡 Supabase Connection Troubleshooting:")
                    print__postgresql_debug("   1. Check Supabase service status")
                    print__postgresql_debug("   2. Verify your IP is whitelisted")
                    print__postgresql_debug("   3. Check network connectivity")
                    print__postgresql_debug("   4. Verify connection string format")
                
                print__postgresql_debug("📋 Environment Check:")
                config = get_db_config()
                print__postgresql_debug(f"   Host: {config.get('host', 'MISSING')}")
                print__postgresql_debug(f"   Port: {config.get('port', 'MISSING')}")
                print__postgresql_debug(f"   Database: {config.get('dbname', 'MISSING')}")
                print__postgresql_debug(f"   User: {config.get('user', 'MISSING')}")
                print__postgresql_debug(f"   Password: {'SET' if config.get('password') else 'MISSING'}")
                
                raise

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

async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str) -> List[Dict[str, Any]]:
    """Get the COMPLETE conversation messages from the LangChain PostgreSQL checkpoint history.
    
    This extracts ALL user questions and ALL AI responses for proper chat display:
    - All user messages: for right-side blue display
    - All AI messages: for left-side white display using the explicit final_answer from state
    
    Args:
        checkpointer: The PostgreSQL checkpointer instance
        thread_id: Thread ID for the conversation
    
    Returns:
        List of message dictionaries in chronological order (complete conversation history)
    """
    try:
        print__api_postgresql(f"🔍 Retrieving COMPLETE checkpoint history for thread: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints for this thread using alist()
        checkpoint_tuples = []
        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoint_tuples.append(checkpoint_tuple)
        
        if not checkpoint_tuples:
            print__api_postgresql(f"⚠ No checkpoints found for thread: {thread_id}")
            return []
        
        print__api_postgresql(f"📄 Found {len(checkpoint_tuples)} checkpoints")
        
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
        
        print__api_postgresql(f"✅ Extracted {len(conversation_messages)} conversation messages from COMPLETE history")
        
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
            kwargs={
                "connect_timeout": 10,
                "autocommit": True,
                "prepare_threshold": None,
                "server_settings": {
                    "application_name": "czsu_agent_health_check",
                    "statement_timeout": "10000",  # 10 second timeout
                }
            },
            open=False
        )
        
        await pool.open()
        
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
    by retrying only checkpoint operations, not the entire LangGraph execution.
    """
    
    def __init__(self, base_checkpointer):
        self.base_checkpointer = base_checkpointer
        self.max_checkpoint_retries = 3
        
    async def _retry_checkpoint_operation(self, operation_name, operation_func, *args, **kwargs):
        """Retry checkpoint operations with exponential backoff for connection issues."""
        
        for attempt in range(self.max_checkpoint_retries):
            try:
                return await operation_func(*args, **kwargs)
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Enhanced error detection for SSL and connection issues
                is_recoverable = any(keyword in error_msg for keyword in [
                    "dbhandler exited",
                    "connection is lost",
                    "ssl connection has been closed",
                    "connection closed",
                    "flush request failed",
                    "pipeline mode",
                    "connection not available",
                    "bad connection",
                    "ssl error",                    # Added SSL error detection
                    "ssl syscall error",           # Added SSL syscall error detection
                    "bad length",                  # Added bad length error (from your specific error)
                    "eof detected",                # Added EOF detected error
                    "connection reset",            # Added connection reset
                    "broken pipe",                 # Added broken pipe
                    "network unreachable"          # Added network issues
                ])
                
                if attempt < self.max_checkpoint_retries - 1 and is_recoverable:
                    delay = 2.0 ** (attempt + 1)  # Exponential backoff: 2s, 4s, 8s
                    print__api_postgresql(f"🔄 Checkpoint operation '{operation_name}' failed (attempt {attempt + 1}): {str(e)}")
                    print__api_postgresql(f"🔄 Retrying checkpoint operation in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    
                    # Enhanced connection recovery for SSL/connection errors
                    if any(keyword in error_msg for keyword in ["ssl", "connection", "eof", "bad length"]):
                        try:
                            print__api_postgresql(f"🔄 SSL/Connection error detected - attempting pool recovery...")
                            
                            # Try multiple recovery strategies
                            if hasattr(self.base_checkpointer, 'conn'):
                                # Strategy 1: Reset connection pool
                                if hasattr(self.base_checkpointer.conn, 'reset'):
                                    print__api_postgresql(f"🔄 Resetting connection pool...")
                                    await self.base_checkpointer.conn.reset()
                                
                                # Strategy 2: Check pool health and recreate if needed
                                if hasattr(self.base_checkpointer.conn, 'closed') and self.base_checkpointer.conn.closed:
                                    print__api_postgresql(f"🔄 Pool is closed, attempting to reopen...")
                                    await self.base_checkpointer.conn.open()
                                
                                # Strategy 3: Force a health check
                                try:
                                    async with self.base_checkpointer.conn.connection() as test_conn:
                                        await test_conn.execute("SELECT 1")
                                    print__api_postgresql(f"✓ Connection pool health check passed")
                                except Exception as health_error:
                                    print__api_postgresql(f"⚠ Connection health check failed: {health_error}")
                                    
                        except Exception as reset_error:
                            print__api_postgresql(f"⚠ Connection recovery failed: {reset_error}")
                    
                    continue
                else:
                    print__api_postgresql(f"❌ Checkpoint operation '{operation_name}' failed after {attempt + 1} attempts: {str(e)}")
                    # For SSL errors, provide specific guidance
                    if any(keyword in error_msg for keyword in ["ssl", "eof detected", "bad length"]):
                        print__api_postgresql(f"💡 SSL Connection Issue: This may be caused by network timeouts, connection pool exhaustion, or database restarts")
                        print__api_postgresql(f"💡 Consider: 1) Check database connectivity, 2) Restart application, 3) Check SSL certificates")
                    raise
        
    async def aput(self, config, checkpoint, metadata, new_versions):
        """Put checkpoint with retry logic."""
        return await self._retry_checkpoint_operation(
            "aput", 
            self.base_checkpointer.aput,
            config, checkpoint, metadata, new_versions
        )
        
    async def aput_writes(self, config, writes, task_id):
        """Put writes with retry logic."""
        return await self._retry_checkpoint_operation(
            "aput_writes",
            self.base_checkpointer.aput_writes,
            config, writes, task_id
        )
    
    async def aget(self, config):
        """Get checkpoint with retry logic."""
        return await self._retry_checkpoint_operation(
            "aget",
            self.base_checkpointer.aget,
            config
        )
        
    async def aget_tuple(self, config):
        """Get tuple with retry logic."""
        return await self._retry_checkpoint_operation(
            "aget_tuple", 
            self.base_checkpointer.aget_tuple,
            config
        )
        
    async def alist(self, config, filter=None, before=None, limit=None):
        """List checkpoints with retry logic - returns async generator."""
        # For alist, we need to handle it differently since it returns an async generator
        # We can't use the retry wrapper directly, so we implement retry logic here
        
        for attempt in range(self.max_checkpoint_retries):
            try:
                # Get the async generator from the base checkpointer
                # Note: AsyncPostgresSaver.alist() only takes config parameter
                async_gen = self.base_checkpointer.alist(config)
                
                # Yield items from the generator with error handling
                async for item in async_gen:
                    yield item
                return  # Successfully completed
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a recoverable database connection error
                is_recoverable = any(keyword in error_msg for keyword in [
                    "dbhandler exited",
                    "connection is lost",
                    "ssl connection has been closed",
                    "connection closed",
                    "flush request failed",
                    "pipeline mode",
                    "connection not available",
                    "bad connection"
                ])
                
                if attempt < self.max_checkpoint_retries - 1 and is_recoverable:
                    delay = 1.5 ** (attempt + 1)  # Progressive delay: 1.5s, 2.25s, 3.38s
                    print__api_postgresql(f"🔄 Checkpoint alist operation failed (attempt {attempt + 1}): {str(e)}")
                    print__api_postgresql(f"🔄 Retrying alist operation in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    
                    # Try to refresh connection pool on SSL/connection errors
                    if any(keyword in error_msg for keyword in ["ssl", "connection"]):
                        try:
                            if hasattr(self.base_checkpointer, 'conn') and hasattr(self.base_checkpointer.conn, 'reset'):
                                print__api_postgresql(f"🔄 Attempting to reset connection pool...")
                                await self.base_checkpointer.conn.reset()
                        except Exception as reset_error:
                            print__api_postgresql(f"⚠ Connection reset failed: {reset_error}")
                    
                    continue
                else:
                    print__api_postgresql(f"❌ Checkpoint alist operation failed after {attempt + 1} attempts: {str(e)}")
                    raise
    
    def __getattr__(self, name):
        """Delegate other operations to base checkpointer."""
        return getattr(self.base_checkpointer, name)

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