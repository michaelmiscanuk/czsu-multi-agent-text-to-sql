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
from datetime import datetime, timedelta
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # Correct async import
from langgraph.checkpoint.postgres import PostgresSaver  # Correct sync import
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from psycopg import AsyncConnection  # Add missing import for direct connection testing
import threading
from contextlib import asynccontextmanager
import time

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
    """Force close all connections in the pool."""
    try:
        print__postgresql_debug("🚨 Forcing closure of all PostgreSQL connections")
        pool = await get_healthy_pool()
        if pool:
            await pool.close()
            print__postgresql_debug("✅ All PostgreSQL connections closed")
    except Exception as e:
        print__postgresql_debug(f"❌ Error closing connections: {e}")

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
    """Get PostgreSQL connection string from environment variables with enhanced cloud deployment configuration."""
    config = get_db_config()
    
    print__postgres_startup_debug(f"🔗 Building connection string for cloud PostgreSQL (Render/Supabase)")
    
    # ENHANCED: Robust cloud-optimized connection string for Render/Supabase deployment
    # Based on psycopg best practices and cloud environment requirements
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode=require"                     # Required for cloud PostgreSQL
        f"&connect_timeout=20"                  # Increased timeout for cloud latency
        f"&application_name=czsu_agent"         # App identification for monitoring
        # Connection keepalive settings (critical for cloud deployments)
        f"&keepalives_idle=600"                 # 10 minutes before first keepalive
        f"&keepalives_interval=30"              # 30 seconds between keepalive probes
        f"&keepalives_count=3"                  # 3 failed probes before disconnect
        f"&tcp_user_timeout=60000"              # 60 seconds TCP user timeout
        # FIXED: Put statement timeouts back in connection string where they work correctly
        f"&statement_timeout=300000"            # 5 minutes statement timeout
        f"&idle_in_transaction_session_timeout=600000"  # 10 minutes idle timeout
        # Network resilience settings
        f"&target_session_attrs=read-write"     # Ensure we get a writable session
    )
    
    # Log connection details (without password)
    debug_string = connection_string.replace(config['password'], '***')
    print__postgres_startup_debug(f"🔗 Using enhanced cloud-optimized connection string: {debug_string}")
    
    return connection_string

async def is_pool_healthy(pool: Optional[AsyncConnectionPool]) -> bool:
    """Check if a connection pool is healthy and open with enhanced diagnostics."""
    if pool is None:
        print__postgresql_debug(f"⚠ Pool is None")
        return False
        
    try:
        # Check if pool is closed
        if pool.closed:
            print__postgresql_debug(f"⚠ Pool is marked as closed")
            return False
        
        # Get pool statistics if available for diagnostics
        try:
            stats = pool.get_stats()
            print__postgresql_debug(f"📊 Pool stats: {stats}")
        except (AttributeError, Exception) as e:
            print__postgresql_debug(f"📊 Pool stats unavailable: {e}")
        
        # Try a quick connection test with timeout
        try:
            async with asyncio.wait_for(pool.connection(), timeout=5) as conn:
                await asyncio.wait_for(conn.execute("SELECT 1"), timeout=5)
                print__postgresql_debug(f"✅ Pool health check passed")
                return True
        except asyncio.TimeoutError:
            print__postgresql_debug(f"⚠ Pool health check timed out")
            return False
        except Exception as e:
            print__postgresql_debug(f"⚠ Pool health check failed: {e}")
            return False
            
    except Exception as e:
        print__postgresql_debug(f"⚠ Pool health check error: {e}")
        return False

async def create_fresh_connection_pool() -> AsyncConnectionPool:
    """Create a new PostgreSQL connection pool with cloud-optimized settings and enhanced error handling."""
    try:
        print__postgresql_debug("🔄 Creating fresh PostgreSQL connection pool...")
        
        # Log connection configuration
        host = get_db_config()["host"]
        port = get_db_config()["port"]
        dbname = get_db_config()["dbname"]
        user = get_db_config()["user"]
        log_connection_info(host, port, dbname, user)
        
        # Construct connection string with SSL configuration
        conninfo = get_connection_string()
        print__postgresql_debug(f"🔗 Connection string configured with enhanced cloud settings")
        
        # CRITICAL: Enhanced configuration for cloud deployment resilience
        print__postgresql_debug("🔧 Configuring for cloud deployment with maximum resilience")
        
        # ENHANCED: Ultra-robust pool settings based on psycopg best practices
        pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=0,  # CHANGED: Start with 0 to avoid initial connection storms
            max_size=2,  # REDUCED: Conservative max to prevent connection exhaustion
            timeout=20,  # INCREASED: Longer timeout for acquiring connections in cloud
            max_idle=120,  # REDUCED: 2 minutes idle time to prevent stale connections
            max_lifetime=900,  # REDUCED: 15 minutes lifetime to prevent SSL timeouts
            reconnect_timeout=60,  # INCREASED: Longer reconnection attempts
            open=False,  # IMPORTANT: Don't open in constructor
            # Enhanced connection configuration function
            configure=None,
            # Connection health check function
            check=AsyncConnectionPool.check_connection,  # ADDED: Built-in health checking
            kwargs={
                "prepare_threshold": None,      # CRITICAL: Disable prepared statements for cloud
                "autocommit": True,             # Use autocommit for better compatibility
                "connect_timeout": 15,          # INCREASED: Connection-level timeout for cloud
                # Pipeline mode disabled to prevent AsyncPipeline errors
                "pipeline": False,              # CRITICAL: Disable pipeline mode
                # FIXED: Removed incorrect options parameter - timeouts are now in connection string
            }
        )
        
        # Explicitly open the pool with retries
        print__postgresql_debug("🔧 Opening connection pool with retry logic...")
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                await asyncio.wait_for(pool.open(), timeout=30)
                print__postgresql_debug("✅ Connection pool opened successfully")
                break
            except asyncio.TimeoutError:
                print__postgresql_debug(f"⚠️ Pool open timeout on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))
            except Exception as e:
                print__postgresql_debug(f"⚠️ Pool open failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))
        
        # Validate the pool with a test query
        try:
            async with asyncio.wait_for(pool.connection(), timeout=10) as conn:
                await asyncio.wait_for(conn.execute("SELECT 1, NOW(), version()"), timeout=10)
            print__postgresql_debug("✅ Pool validation successful")
        except Exception as e:
            print__postgresql_debug(f"❌ Pool validation failed: {e}")
            await pool.close()
            raise
        
        print__postgresql_debug("✅ Fresh connection pool created with enhanced cloud resilience")
        return pool
        
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to create connection pool: {e}")
        # Enhanced error diagnostics
        error_msg = str(e).lower()
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
        
        import traceback
        print__postgresql_debug(f"🔍 Full traceback: {traceback.format_exc()}")
        raise

async def get_healthy_pool() -> AsyncConnectionPool:
    """Get a healthy PostgreSQL connection pool with automatic recreation."""
    global database_pool
    
    # Check if pool exists and is healthy
    if database_pool is not None and not database_pool.closed:
        print__postgresql_debug("🔍 Checking existing pool health...")
        is_healthy = await is_pool_healthy(database_pool)
        if is_healthy:
            print__postgresql_debug("✅ Existing pool is healthy")
            return database_pool
        else:
            print__postgresql_debug("⚠️ Existing pool is unhealthy, closing...")
            try:
                await database_pool.close()
            except Exception as e:
                print__postgresql_debug(f"⚠️ Error closing unhealthy pool: {e}")
            database_pool = None
    
    # Create new pool
    print__postgresql_debug("🔄 Creating new healthy pool...")
    async with _get_pool_lock():
        # Double-check pattern - another thread might have created the pool
        if database_pool is not None and not database_pool.closed:
            print__postgresql_debug("✅ Pool was created by another thread")
            return database_pool
        
        database_pool = await create_fresh_connection_pool()
        
        # Test the new pool
        try:
            async with database_pool.connection() as conn:
                await conn.execute("SELECT 1")
            print__postgresql_debug("✅ New pool verified with test query")
        except Exception as e:
            print__postgresql_debug(f"❌ New pool failed verification: {e}")
            await database_pool.close()
            database_pool = None
            raise
        
        return database_pool

async def setup_users_threads_runs_table():
    """Setup the users_threads_runs table for tracking user conversations."""
    try:
        print__postgresql_debug("🔧 Setting up users_threads_runs table...")
        pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
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
            
            print__postgresql_debug("✅ users_threads_runs table and indexes created/verified")
            
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to setup users_threads_runs table: {e}")
        import traceback
        print__postgresql_debug(f"🔍 Full traceback: {traceback.format_exc()}")
        raise

async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new entry in users_threads_runs table."""
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    try:
        print__postgresql_debug(f"📝 Creating thread run entry: email={email}, thread_id={thread_id}, run_id={run_id}")
        
        pool = await get_healthy_pool()
        async with pool.connection() as conn:
            await conn.execute("""
                INSERT INTO users_threads_runs (email, thread_id, run_id, prompt, timestamp)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (email, thread_id, run_id, prompt))
            
        print__postgresql_debug(f"✅ Thread run entry created successfully")
        return run_id
        
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to create thread run entry: {e}")
        # Return the run_id anyway so analysis can continue
        return run_id

async def update_thread_run_sentiment(run_id: str, sentiment: bool, user_email: str = None) -> bool:
    """Update sentiment for a specific run_id with optional user verification."""
    try:
        print__postgresql_debug(f"💭 Updating sentiment for run_id: {run_id}, sentiment: {sentiment}")
        
        pool = await get_healthy_pool()
        async with pool.connection() as conn:
            if user_email:
                # 🔒 SECURITY: Update with user verification
                print__postgresql_debug(f"🔒 Verifying user {user_email} owns run_id {run_id}")
                result = await conn.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s AND email = %s
                """, (sentiment, run_id, user_email))
            else:
                # Update without user verification (for backward compatibility)
                print__postgresql_debug(f"⚠️ Updating sentiment without user verification")
                result = await conn.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s
                """, (sentiment, run_id))
            
            rows_affected = result.rowcount if hasattr(result, 'rowcount') else 0
            success = rows_affected > 0
            
            if success:
                print__postgresql_debug(f"✅ Sentiment updated successfully for run_id: {run_id}")
            else:
                print__postgresql_debug(f"⚠️ No rows updated - run_id not found or access denied: {run_id}")
            
            return success
            
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to update sentiment for run_id {run_id}: {e}")
        return False

async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get sentiment values for all run_ids in a thread."""
    try:
        print__postgresql_debug(f"💭 Getting sentiments for thread: {thread_id}, user: {email}")
        
        pool = await get_healthy_pool()
        async with pool.connection() as conn:
            result = await conn.execute("""
                SELECT run_id, sentiment 
                FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s AND sentiment IS NOT NULL
                ORDER BY timestamp ASC
            """, (email, thread_id))
            
            sentiments = {}
            async for row in result:
                run_id, sentiment = row
                sentiments[run_id] = sentiment
            
            print__postgresql_debug(f"📊 Retrieved {len(sentiments)} sentiment values for thread {thread_id}")
            return sentiments
            
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to get sentiments for thread {thread_id}: {e}")
        return {}

async def get_user_chat_threads(email: str, connection_pool=None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination."""
    try:
        print__api_postgresql(f"📋 Getting chat threads for user: {email} (limit: {limit}, offset: {offset})")
        
        if connection_pool:
            print__api_postgresql(f"🔗 Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"🔄 Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            # Build the SQL query with optional pagination
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
            
            # Add pagination if limit is specified
            if limit is not None:
                base_query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            
            result = await conn.execute(base_query, params)
            
            threads = []
            async for row in result:
                thread_id, latest_timestamp, run_count, first_prompt = row
                
                # Create a title from the first prompt (limit to 50 characters)
                title = (first_prompt[:47] + "...") if first_prompt and len(first_prompt) > 50 else (first_prompt or "Untitled Conversation")
                
                threads.append({
                    "thread_id": thread_id,
                    "latest_timestamp": latest_timestamp,
                    "run_count": run_count,
                    "title": title,
                    "full_prompt": first_prompt or ""
                })
            
            print__api_postgresql(f"✅ Retrieved {len(threads)} threads for user {email}")
            return threads
            
    except Exception as e:
        print__api_postgresql(f"❌ Failed to get chat threads for user {email}: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        raise

async def get_user_chat_threads_count(email: str, connection_pool=None) -> int:
    """Get total count of chat threads for a user."""
    try:
        print__api_postgresql(f"📊 Getting chat threads count for user: {email}")
        
        if connection_pool:
            print__api_postgresql(f"🔗 Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"🔄 Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            result = await conn.execute("""
                SELECT COUNT(DISTINCT thread_id) as total_threads
                FROM users_threads_runs
                WHERE email = %s
            """, (email,))
            
            row = await result.fetchone()
            total_count = row[0] if row else 0
            
            print__api_postgresql(f"✅ Total threads count for user {email}: {total_count}")
            return total_count
            
    except Exception as e:
        print__api_postgresql(f"❌ Failed to get chat threads count for user {email}: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        raise

async def delete_user_thread_entries(email: str, thread_id: str, connection_pool=None) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table."""
    try:
        print__api_postgresql(f"🗑️ Deleting thread entries for user: {email}, thread: {thread_id}")
        
        if connection_pool:
            print__api_postgresql(f"🔗 Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"🔄 Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            # First, count the entries to be deleted
            count_result = await conn.execute("""
                SELECT COUNT(*) FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (email, thread_id))
            
            count_row = await count_result.fetchone()
            entries_to_delete = count_row[0] if count_row else 0
            
            print__api_postgresql(f"📊 Found {entries_to_delete} entries to delete")
            
            if entries_to_delete == 0:
                print__api_postgresql(f"⚠️ No entries found for user {email} and thread {thread_id}")
                return {
                    "deleted_count": 0,
                    "message": "No entries found to delete",
                    "thread_id": thread_id,
                    "user_email": email
                }
            
            # Delete the entries
            delete_result = await conn.execute("""
                DELETE FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (email, thread_id))
            
            deleted_count = delete_result.rowcount if hasattr(delete_result, 'rowcount') else 0
            
            print__api_postgresql(f"✅ Deleted {deleted_count} entries for user {email}, thread {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email
            }
            
    except Exception as e:
        print__api_postgresql(f"❌ Failed to delete thread entries for user {email}, thread {thread_id}: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
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
        print__postgres_startup_debug(f"❌ Missing required environment variables: {missing_vars}")
        return False
    else:
        print__postgres_startup_debug("✅ All required PostgreSQL environment variables are set")
        return True

async def test_basic_postgres_connection():
    """Test basic PostgreSQL connection without creating a pool."""
    try:
        print__postgres_startup_debug("🔍 Testing basic PostgreSQL connection...")
        
        # Check environment variables first
        if not check_postgres_env_vars():
            print__postgres_startup_debug("❌ Environment variables check failed")
            return False
        
        connection_string = get_connection_string()
        
        # Test direct connection (not using pool)
        conn = await AsyncConnection.connect(connection_string)
        print__postgres_startup_debug("✅ Direct connection established")
        
        # Test a simple query
        cursor = await conn.execute("SELECT version()")
        result = await cursor.fetchone()
        version = result[0] if result else "Unknown"
        print__postgres_startup_debug(f"📊 PostgreSQL version: {version}")
        
        await conn.close()
        print__postgres_startup_debug("✅ Connection closed successfully")
        
        return True
        
    except Exception as e:
        print__postgres_startup_debug(f"❌ Basic connection test failed: {e}")
        return False

def log_connection_info(host: str, port: str, dbname: str, user: str):
    """Enhanced connection information logging for debugging."""
    print__postgres_startup_debug(f"🔗 Enhanced PostgreSQL Connection Configuration:")
    print__postgres_startup_debug(f"   📡 Host: {host}")
    print__postgres_startup_debug(f"   🔌 Port: {port}")
    print__postgres_startup_debug(f"   💾 Database: {dbname}")
    print__postgres_startup_debug(f"   👤 User: {user}")
    print__postgres_startup_debug(f"   🔒 SSL: Required (Cloud PostgreSQL)")
    print__postgres_startup_debug(f"   🔄 Connection Pooling: Enhanced (psycopg 3)")
    print__postgres_startup_debug(f"   🛡️ Pipeline Mode: Disabled (AsyncPipeline error prevention)")

async def test_pool_connection():
    """Enhanced test for connection pool creation and functionality."""
    try:
        print__api_postgresql("🔍 Testing enhanced connection pool...")
        
        # Test pool creation with enhanced settings
        pool = await create_fresh_connection_pool()
        print__api_postgresql(f"✅ Enhanced pool created: closed={pool.closed}")
        
        # Test pool statistics
        try:
            stats = pool.get_stats()
            print__api_postgresql(f"📊 Pool statistics: {stats}")
        except (AttributeError, Exception) as e:
            print__api_postgresql(f"📊 Pool statistics unavailable: {e}")
        
        # Test pool usage with comprehensive queries
        async with pool.connection() as conn:
            # Test basic functionality
            result = await conn.execute("SELECT 1 as test, NOW() as current_time, pg_backend_pid() as pid")
            row = await result.fetchone()
            print__api_postgresql(f"✅ Basic query successful: test={row[0]}, pid={row[2]}")
            
            # Test transaction handling
            await conn.execute("BEGIN")
            await conn.execute("SELECT 1")
            await conn.execute("COMMIT")
            print__api_postgresql(f"✅ Transaction test successful")
            
            # Test connection info
            result = await conn.execute("SELECT version(), current_database(), current_user")
            row = await result.fetchone()
            print__api_postgresql(f"✅ Connection info: db={row[1]}, user={row[2]}")
        
        # Test pool health check
        is_healthy = await is_pool_healthy(pool)
        print__api_postgresql(f"✅ Pool health check: {is_healthy}")
        
        # Test multiple concurrent connections (if max_size > 1)
        if hasattr(pool, 'max_size') and pool.max_size > 1:
            print__api_postgresql("🔍 Testing concurrent connections...")
            
            async def test_concurrent_query(query_id):
                async with pool.connection() as conn:
                    result = await conn.execute("SELECT %s as query_id, pg_backend_pid() as pid", [query_id])
                    row = await result.fetchone()
                    return f"Query {row[0]} -> PID {row[1]}"
            
            # Run multiple concurrent queries
            tasks = [test_concurrent_query(i) for i in range(2)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print__api_postgresql(f"⚠️ Concurrent query failed: {result}")
                else:
                    print__api_postgresql(f"✅ Concurrent query: {result}")
        
        # Cleanup
        await pool.close()
        print__api_postgresql(f"✅ Pool closed: closed={pool.closed}")
        
        return True
        
    except Exception as e:
        print__api_postgresql(f"❌ Enhanced pool connection test failed: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

async def debug_pool_status():
    """Enhanced debug function to show comprehensive pool status."""
    global database_pool
    
    print__api_postgresql(f"🔍 Enhanced Pool Status Debug:")
    print__api_postgresql(f"   Global pool exists: {database_pool is not None}")
    
    if database_pool:
        print__api_postgresql(f"   Pool type: {type(database_pool).__name__}")
        print__api_postgresql(f"   Pool closed: {database_pool.closed}")
        
        # Enhanced pool information
        try:
            if hasattr(database_pool, 'min_size'):
                print__api_postgresql(f"   Pool min_size: {database_pool.min_size}")
            if hasattr(database_pool, 'max_size'):
                print__api_postgresql(f"   Pool max_size: {database_pool.max_size}")
            if hasattr(database_pool, 'timeout'):
                print__api_postgresql(f"   Pool timeout: {database_pool.timeout}")
                
            # Try to get enhanced pool stats
            if hasattr(database_pool, 'get_stats'):
                stats = database_pool.get_stats()
                print__api_postgresql(f"   Pool stats: {stats}")
            else:
                print__api_postgresql(f"   Pool stats: Not available")
                
            # Test health with enhanced diagnostics
            is_healthy = await is_pool_healthy(database_pool)
            print__api_postgresql(f"   Pool healthy: {is_healthy}")
            
            # Connection test
            if not database_pool.closed:
                try:
                    async with asyncio.wait_for(database_pool.connection(), timeout=5) as conn:
                        result = await asyncio.wait_for(conn.execute("SELECT pg_backend_pid()"), timeout=5)
                        row = await result.fetchone()
                        print__api_postgresql(f"   Test connection PID: {row[0] if row else 'unknown'}")
                except Exception as conn_error:
                    print__api_postgresql(f"   Connection test failed: {conn_error}")
            
        except Exception as e:
            print__api_postgresql(f"   Pool status error: {e}")
    else:
        print__api_postgresql("   No global pool available")
    
    return database_pool

# Enhanced startup and initialization functions
async def initialize_enhanced_postgres_system():
    """Initialize the enhanced PostgreSQL connection system with comprehensive testing."""
    try:
        print__api_postgresql("🚀 Initializing Enhanced PostgreSQL Connection System")
        print__api_postgresql("=" * 60)
        
        # Step 1: Environment validation
        print__api_postgresql("📋 Step 1: Environment Validation")
        env_ok = check_postgres_env_vars()
        if not env_ok:
            print__api_postgresql("❌ Environment validation failed")
            return False
        print__api_postgresql("✅ Environment validation passed")
        
        # Step 2: Basic connection health test
        print__api_postgresql("\n🔍 Step 2: Basic Connection Health Test")
        health_ok = await test_connection_health()
        if not health_ok:
            print__api_postgresql("❌ Basic connection health test failed")
            return False
        print__api_postgresql("✅ Basic connection health test passed")
        
        # Step 3: Enhanced pool creation and testing
        print__api_postgresql("\n🏊 Step 3: Enhanced Pool Creation and Testing")
        pool_ok = await test_pool_connection()
        if not pool_ok:
            print__api_postgresql("❌ Enhanced pool test failed")
            return False
        print__api_postgresql("✅ Enhanced pool test passed")
        
        # Step 4: Initialize global pool
        print__api_postgresql("\n🌐 Step 4: Global Pool Initialization")
        try:
            global_pool = await get_healthy_pool()
            if global_pool:
                print__api_postgresql("✅ Global pool initialized successfully")
                
                # Test the global pool
                async with global_pool.connection() as conn:
                    result = await conn.execute("SELECT 'Global pool test' as message, NOW() as timestamp")
                    row = await result.fetchone()
                    print__api_postgresql(f"✅ Global pool test: {row[0]} at {row[1]}")
            else:
                print__api_postgresql("❌ Global pool initialization failed")
                return False
        except Exception as e:
            print__api_postgresql(f"❌ Global pool initialization error: {e}")
            return False
        
        # Step 5: Initialize database tables
        print__api_postgresql("\n📊 Step 5: Database Schema Initialization")
        try:
            await setup_users_threads_runs_table()
            print__api_postgresql("✅ Database schema initialized successfully")
        except Exception as e:
            print__api_postgresql(f"❌ Database schema initialization failed: {e}")
            return False
        
        # Step 6: Start background monitoring (optional)
        monitoring_enabled = os.getenv("ENABLE_CONNECTION_MONITORING", "true").lower() == "true"
        if monitoring_enabled and global_pool:
            print__api_postgresql("\n📡 Step 6: Starting Background Monitoring")
            try:
                # Start monitoring task in background
                monitor_task = asyncio.create_task(monitor_connection_health(global_pool, 60))
                print__api_postgresql("✅ Background connection monitoring started")
                
                # Store the task reference to prevent garbage collection
                if not hasattr(initialize_enhanced_postgres_system, '_monitor_tasks'):
                    initialize_enhanced_postgres_system._monitor_tasks = []
                initialize_enhanced_postgres_system._monitor_tasks.append(monitor_task)
                
            except Exception as e:
                print__api_postgresql(f"⚠️ Background monitoring failed to start: {e}")
                # Don't fail initialization if monitoring fails
        else:
            print__api_postgresql("\n📡 Step 6: Background Monitoring Disabled")
        
        print__api_postgresql("\n🎉 Enhanced PostgreSQL Connection System Initialized Successfully!")
        print__api_postgresql("=" * 60)
        return True
        
    except Exception as e:
        print__api_postgresql(f"❌ Enhanced PostgreSQL system initialization failed: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

# Enhanced wrapper for backward compatibility
async def get_postgres_checkpointer():
    """Get enhanced PostgreSQL checkpointer with improved error handling."""
    try:
        print__postgresql_debug("🔄 Creating enhanced PostgreSQL checkpointer...")
        
        # Ensure the enhanced system is initialized
        pool = await get_healthy_pool()
        if not pool:
            raise Exception("Failed to get healthy connection pool")
        
        # Create the base checkpointer
        base_checkpointer = AsyncPostgresSaver(
            pool,
            migrations_table="langgraph_migrations",
            checkpoints_table="langgraph_checkpoints",
            writes_table="langgraph_writes"
        )
        
        # Set up the tables
        await base_checkpointer.setup()
        print__postgresql_debug("✅ Enhanced checkpointer tables set up")
        
        # Wrap with enhanced resilience
        resilient_checkpointer = ResilientPostgreSQLCheckpointer(base_checkpointer)
        
        print__postgresql_debug("✅ Enhanced PostgreSQL checkpointer created successfully")
        return resilient_checkpointer
        
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to create enhanced PostgreSQL checkpointer: {e}")
        import traceback
        print__postgresql_debug(f"🔍 Full traceback: {traceback.format_exc()}")
        raise

def get_sync_postgres_checkpointer():
    """Synchronous wrapper for getting PostgreSQL checkpointer."""
    try:
        print__postgres_startup_debug("🔄 Getting PostgreSQL checkpointer (sync wrapper)")
        return asyncio.run(get_postgres_checkpointer())
    except Exception as e:
        print__postgres_startup_debug(f"❌ Sync checkpointer creation failed: {e}")
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


async def monitor_connection_health(pool: AsyncConnectionPool, interval: int = 60):
    """Enhanced connection pool health monitor with SSL-specific diagnostics."""
    print__api_postgresql("🔍 Starting enhanced connection pool health monitor")
    monitor_failures = 0
    max_consecutive_failures = 3
    
    try:
        while True:
            try:
                # Enhanced health check with timeout
                start_time = time.time()
                async with asyncio.wait_for(pool.connection(), timeout=10) as conn:
                    # More comprehensive health check
                    result = await asyncio.wait_for(
                        conn.execute("SELECT 1 as health, NOW() as timestamp, pg_backend_pid() as pid"), 
                        timeout=10
                    )
                    row = await result.fetchone()
                    check_duration = time.time() - start_time
                
                # Get enhanced pool statistics
                try:
                    stats = pool.get_stats()
                    print__api_postgresql(f"✓ Pool health OK ({check_duration:.2f}s) - Stats: {stats}")
                except (AttributeError, Exception):
                    print__api_postgresql(f"✓ Pool health OK ({check_duration:.2f}s) - Backend PID: {row[2] if row else 'unknown'}")
                
                # Reset failure counter on success
                monitor_failures = 0
                
            except asyncio.TimeoutError:
                monitor_failures += 1
                print__api_postgresql(f"⚠ Pool health check timeout ({monitor_failures}/{max_consecutive_failures})")
                
                if monitor_failures >= max_consecutive_failures:
                    print__api_postgresql("🚨 Multiple consecutive health check timeouts - pool may need recreation")
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
                    print__api_postgresql(f"🔒 SSL connection error in health check ({monitor_failures}/{max_consecutive_failures}): {e}")
                elif any(pattern in error_msg for pattern in [
                    "dbhandler exited",
                    "pipeline",
                    "flush request failed",
                    "lost synchronization"
                ]):
                    print__api_postgresql(f"🚨 Critical connection error in health check ({monitor_failures}/{max_consecutive_failures}): {e}")
                else:
                    print__api_postgresql(f"⚠ Pool health check failed ({monitor_failures}/{max_consecutive_failures}): {e}")
                
                if monitor_failures >= max_consecutive_failures:
                    print__api_postgresql("🚨 Multiple consecutive health check failures detected")
                    print__api_postgresql("💡 This may indicate persistent connection issues that require manual intervention")
            
            # Dynamic interval based on health
            sleep_interval = interval
            if monitor_failures > 0:
                # More frequent checks when issues are detected
                sleep_interval = min(interval, 30)
                print__api_postgresql(f"🔍 Increased monitoring frequency due to {monitor_failures} failures")
            
            await asyncio.sleep(sleep_interval)
            
    except asyncio.CancelledError:
        print__api_postgresql("📊 Enhanced connection monitor stopped")
        raise
    except Exception as e:
        print__api_postgresql(f"❌ Connection monitor error: {e}")
        raise

async def test_connection_health():
    """Enhanced connection health test with comprehensive SSL diagnostics."""
    try:
        config = get_db_config()
        
        if not all([config['user'], config['password'], config['host'], config['dbname']]):
            print__api_postgresql("❌ Missing required environment variables for database connection")
            return False
            
        # Use the same connection string as the main application for consistency
        connection_string = get_connection_string()
        
        print__api_postgresql("🔍 Testing enhanced cloud PostgreSQL connection health...")
        print__api_postgresql(f"🔗 Host: {config['host']}:{config['port']}")
        print__api_postgresql(f"🔗 Database: {config['dbname']}")
        print__api_postgresql(f"🔗 User: {config['user']}")
        
        # ENHANCED: Use the same robust settings as main pool for consistency
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=1,
            min_size=0,  # Start with 0 for health check
            timeout=10,  # Reasonable timeout for health check
            max_idle=60,  # Short idle time for health check
            max_lifetime=300,  # Short lifetime for health check
            reconnect_timeout=30,  # Shorter reconnect timeout for health check
            open=False,  # Don't open in constructor
            check=AsyncConnectionPool.check_connection,  # Use built-in health checking
            kwargs={
                "prepare_threshold": None,      # CRITICAL: Disable prepared statements
                "autocommit": True,             # Use autocommit for compatibility
                "connect_timeout": 10,          # Connection timeout for health check
                "pipeline": False,              # CRITICAL: Disable pipeline mode
                # FIXED: Removed incorrect options parameter - timeouts are now in connection string
            }
        )
        
        # Explicitly open the pool with timeout
        try:
            await asyncio.wait_for(pool.open(), timeout=20)
            print__api_postgresql("✅ Health check pool opened successfully")
        except asyncio.TimeoutError:
            print__api_postgresql("❌ Health check pool open timeout")
            return False
        except Exception as e:
            print__api_postgresql(f"❌ Health check pool open failed: {e}")
            return False
        
        # Comprehensive connection test
        try:
            async with pool.connection() as conn:
                # Test multiple operations to ensure full functionality
                result = await asyncio.wait_for(
                    conn.execute("SELECT 1 as test, NOW() as current_time, version() as pg_version, pg_backend_pid() as pid"), 
                    timeout=10
                )
                row = await result.fetchone()
                
                if row and row[0] == 1:
                    print__api_postgresql("✅ Enhanced connection health check successful")
                    print__api_postgresql(f"   📊 Server time: {row[1]}")
                    print__api_postgresql(f"   📊 Backend PID: {row[3]}")
                    print__api_postgresql(f"   📊 PostgreSQL: {row[2][:50]}...")
                    
                    # Test a simple transaction to ensure full functionality
                    await conn.execute("BEGIN")
                    await conn.execute("SELECT 1")
                    await conn.execute("COMMIT")
                    print__api_postgresql("✅ Transaction test successful")
                    
                    await pool.close()
                    return True
                else:
                    print__api_postgresql("❌ Health check query returned unexpected result")
                    await pool.close()
                    return False
                    
        except asyncio.TimeoutError:
            print__api_postgresql("❌ Health check query timeout")
        except Exception as e:
            error_msg = str(e).lower()
            print__api_postgresql(f"❌ Health check query failed: {e}")
            
            # Enhanced error diagnostics
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
            elif "timeout" in error_msg:
                print__api_postgresql("💡 Timeout issue detected:")
                print__api_postgresql("   1. Check database server responsiveness")
                print__api_postgresql("   2. Verify network latency")
                print__api_postgresql("   3. Check connection pool settings")
            elif any(pattern in error_msg for pattern in ["pipeline", "dbhandler", "flush request"]):
                print__api_postgresql("💡 Pipeline/handler issue detected:")
                print__api_postgresql("   1. This indicates connection state corruption")
                print__api_postgresql("   2. Pool recreation may be required")
                print__api_postgresql("   3. Consider disabling pipeline mode")
        
        await pool.close()
        return False
        
    except Exception as e:
        error_msg = str(e).lower()
        print__api_postgresql(f"❌ Connection health test failed: {e}")
        
        # Provide specific guidance based on error type
        if "ssl" in error_msg:
            print__api_postgresql("💡 SSL configuration issue - check database SSL settings")
        elif "timeout" in error_msg:
            print__api_postgresql("💡 Network connectivity issue - check connection to database")
        elif "authentication" in error_msg:
            print__api_postgresql("💡 Authentication issue - verify database credentials")
        
        return False

class ResilientPostgreSQLCheckpointer:
    """Enhanced wrapper around PostgresSaver to handle SSL connection drops and AsyncPipeline errors in cloud environments."""
    
    def __init__(self, base_checkpointer):
        self.base_checkpointer = base_checkpointer
        self._last_pool_recreation = 0
        self._pool_recreation_cooldown = 30  # 30 seconds cooldown between pool recreations

    async def _enhanced_cloud_resilient_retry(self, operation_name, operation_func, *args, **kwargs):
        """Enhanced retry logic with SSL-specific error handling and pool recreation."""
        max_retries = int(os.getenv("CHECKPOINT_MAX_RETRIES", "4"))  # Increased default
        base_delay = float(os.getenv("CHECKPOINT_RETRY_BASE_DELAY", "1.0"))
        dbhandler_multiplier = int(os.getenv("DBHANDLER_EXITED_DELAY_MULTIPLIER", "6"))  # Increased
        ssl_retry_delay = float(os.getenv("SSL_RETRY_DELAY", "5.0"))  # Increased
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
                print__postgresql_debug(f"🔄 Attempt {attempt + 1}/{max_retries} for {operation_name}")
                result = await operation_func(*args, **kwargs)
                if attempt > 0:
                    print__postgresql_debug(f"✅ {operation_name} succeeded after {attempt + 1} attempts")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                is_ssl_error = any(pattern in error_msg for pattern in ssl_connection_errors)
                is_critical_error = any(pattern in error_msg for pattern in critical_errors)
                
                print__postgresql_debug(f"⚠️ {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Enhanced error diagnostics
                if is_critical_error:
                    print__postgresql_debug("🚨 CRITICAL ERROR: Database handler or pipeline failure detected")
                elif is_ssl_error:
                    print__postgresql_debug("🔒 SSL CONNECTION ERROR: Connection lost or terminated")
                
                # Last attempt - don't retry
                if attempt == max_retries - 1:
                    print__postgresql_debug(f"❌ {operation_name} failed after {max_retries} attempts")
                    raise
                
                # Calculate delay with enhanced backoff for different error types
                if is_critical_error:
                    # Aggressive backoff for critical errors
                    delay = ssl_retry_delay * (dbhandler_multiplier ** (attempt + 1))
                    print__postgresql_debug(f"🚨 Critical error - using aggressive backoff: {delay}s")
                elif is_ssl_error:
                    # Extended delay for SSL errors
                    delay = ssl_retry_delay * (2 ** attempt)
                    print__postgresql_debug(f"🔒 SSL error - using extended backoff: {delay}s")
                else:
                    # Standard exponential backoff
                    delay = base_delay * (2 ** attempt)
                    print__postgresql_debug(f"🔄 Standard backoff: {delay}s")
                
                # Pool recreation logic for SSL/critical errors
                if (is_ssl_error or is_critical_error) and enable_pool_recreation:
                    current_time = time.time()
                    if current_time - self._last_pool_recreation > self._pool_recreation_cooldown:
                        print__postgresql_debug("🔄 Attempting pool recreation due to connection error...")
                        try:
                            # Force close and recreate the connection pool
                            await self._recreate_connection_pool()
                            self._last_pool_recreation = current_time
                            print__postgresql_debug("✅ Pool recreation successful")
                            # Shorter delay after successful pool recreation
                            delay = min(delay, 3)
                        except Exception as pool_error:
                            print__postgresql_debug(f"⚠️ Pool recreation failed: {pool_error}")
                    else:
                        cooldown_remaining = self._pool_recreation_cooldown - (current_time - self._last_pool_recreation)
                        print__postgresql_debug(f"⏳ Pool recreation on cooldown ({cooldown_remaining:.1f}s remaining)")
                
                print__postgresql_debug(f"⏳ Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
        
        # Should never reach here
        raise Exception(f"Unexpected end of retry loop for {operation_name}")

    async def _recreate_connection_pool(self):
        """Recreate the connection pool to handle persistent connection issues."""
        global database_pool
        
        try:
            print__postgresql_debug("🔄 Recreating connection pool...")
            
            # Close existing pool if it exists
            if database_pool is not None:
                try:
                    await asyncio.wait_for(database_pool.close(), timeout=10)
                    print__postgresql_debug("✅ Old pool closed")
                except Exception as e:
                    print__postgresql_debug(f"⚠️ Error closing old pool: {e}")
                finally:
                    database_pool = None
            
            # Wait a moment for cleanup
            await asyncio.sleep(1)
            
            # Create new pool
            database_pool = await create_fresh_connection_pool()
            
            # Update base checkpointer if it has a pool attribute
            if hasattr(self.base_checkpointer, 'pool'):
                old_pool = self.base_checkpointer.pool
                self.base_checkpointer.pool = database_pool
                # Close old pool if different
                if old_pool and old_pool != database_pool:
                    try:
                        await old_pool.close()
                    except:
                        pass
            
            print__postgresql_debug("✅ Connection pool recreation completed")
            
        except Exception as e:
            print__postgresql_debug(f"❌ Failed to recreate connection pool: {e}")
            raise

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
                print__postgresql_debug(f"⚠️ alist encountered connection error: {e}")
                # Try to recreate pool and retry once
                try:
                    await self._recreate_connection_pool()
                    await asyncio.sleep(2)
                    async for item in self.base_checkpointer.alist(config, filter=filter, before=before, limit=limit):
                        yield item
                except Exception as retry_error:
                    print__postgresql_debug(f"❌ alist retry failed: {retry_error}")
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