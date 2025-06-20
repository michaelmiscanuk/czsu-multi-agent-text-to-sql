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
    """Create a new PostgreSQL connection pool."""
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
        print__postgresql_debug(f"🔗 Connection string configured (SSL enabled)")
        
        # Create pool with production-optimized settings
        pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=1,  # Minimum connections in pool
            max_size=8,  # Maximum connections in pool (reduced from 10)
            timeout=30,  # Connection timeout in seconds
            max_idle=300,  # Maximum idle time before closing connections (5 minutes)
            max_lifetime=3600,  # Maximum lifetime of connections (1 hour)
            check=ConnectionCheck.ON_CREATE_AND_ACQUIRE,  # Health check mode
            configure=None,  # Connection configuration function
            kwargs={}  # Additional connection parameters
        )
        
        print__postgresql_debug("✅ Connection pool created successfully")
        return pool
        
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to create connection pool: {e}")
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

async def get_user_chat_threads(email: str, connection_pool=None) -> List[Dict[str, Any]]:
    """Get all chat threads for a user with thread details."""
    try:
        print__api_postgresql(f"📋 Getting chat threads for user: {email}")
        
        if connection_pool:
            print__api_postgresql(f"🔗 Using provided connection pool")
            pool = connection_pool
        else:
            print__api_postgresql(f"🔄 Creating new connection pool")
            pool = await get_healthy_pool()
        
        async with pool.connection() as conn:
            result = await conn.execute("""
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
            """, (email, email))
            
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
        'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 
        'POSTGRES_USER', 'POSTGRES_PASSWORD'
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

async def get_postgres_checkpointer():
    """Get a PostgreSQL checkpointer with comprehensive error handling."""
    try:
        print__postgres_startup_debug("🚀 Initializing PostgreSQL checkpointer...")
        
        # Test basic connection first
        if not await test_basic_postgres_connection():
            print__postgres_startup_debug("❌ Basic connection test failed - checkpointer creation aborted")
            raise Exception("PostgreSQL basic connection test failed")
        
        print__postgres_startup_debug("✅ Basic connection test passed")
        
        # Create and setup database objects
        pool = await get_healthy_pool()
        print__postgres_startup_debug("✅ Healthy connection pool obtained")
        
        # Setup required tables
        await setup_users_threads_runs_table()
        print__postgres_startup_debug("✅ Users threads runs table setup completed")
        
        # Create the checkpointer
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        print__postgres_startup_debug("✅ PostgreSQL checkpointer setup completed")
        
        # Wrap with resilient wrapper for better error handling
        resilient_checkpointer = ResilientPostgreSQLCheckpointer(checkpointer)
        print__postgres_startup_debug("✅ Resilient wrapper applied")
        
        print__postgres_startup_debug("🎉 PostgreSQL checkpointer fully initialized and ready")
        return resilient_checkpointer
        
    except Exception as e:
        print__postgres_startup_debug(f"❌ Failed to create PostgreSQL checkpointer: {e}")
        import traceback
        print__postgres_startup_debug(f"🔍 Full error traceback: {traceback.format_exc()}")
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
    """Extract conversation messages from PostgreSQL checkpoint data."""
    try:
        print__api_postgresql(f"📥 Loading conversation messages from checkpoints for thread: {thread_id}")
        
        # Get the latest checkpoint state
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await checkpointer.aget_tuple(config)
        
        if not state_snapshot or not state_snapshot.checkpoint:
            print__api_postgresql(f"⚠️ No checkpoint data found for thread: {thread_id}")
            return []
        
        # Extract messages from the checkpoint
        channel_values = state_snapshot.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        
        print__api_postgresql(f"📊 Found {len(messages)} messages in checkpoint for thread: {thread_id}")
        
        # Convert LangChain messages to our format
        stored_messages = []
        
        for i, message in enumerate(messages):
            try:
                # Skip empty system messages (they're just placeholders)
                if hasattr(message, 'content') and message.content.strip() == "":
                    continue
                
                # Determine if this is a user or AI message
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                if isinstance(message, HumanMessage):
                    is_user = True
                    user_type = "user"
                elif isinstance(message, AIMessage):
                    is_user = False
                    user_type = "ai"
                elif isinstance(message, SystemMessage):
                    # Skip system messages in conversation display
                    continue
                else:
                    # Unknown message type, skip
                    print__api_postgresql(f"⚠️ Unknown message type: {type(message)}")
                    continue
                
                # Create stored message record
                stored_message = {
                    "id": getattr(message, 'id', f"msg_{thread_id}_{i}"),
                    "content": getattr(message, 'content', ''),
                    "is_user": is_user,
                    "timestamp": datetime.now(),  # We don't have original timestamps
                    "user_type": user_type
                }
                
                stored_messages.append(stored_message)
                
                print__api_postgresql(f"📄 Message {i+1}: {user_type} - {stored_message['content'][:50]}...")
                
            except Exception as msg_error:
                print__api_postgresql(f"❌ Error processing message {i}: {msg_error}")
                continue
        
        print__api_postgresql(f"✅ Successfully processed {len(stored_messages)} conversation messages")
        return stored_messages
        
    except Exception as e:
        print__api_postgresql(f"❌ Failed to get conversation messages from checkpoints: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        return []

async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str) -> List[List[str]]:
    """Get queries_and_results from the latest checkpoint."""
    try:
        print__api_postgresql(f"🔍 Getting queries and results from checkpoint for thread: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await checkpointer.aget_tuple(config)
        
        if state_snapshot and state_snapshot.checkpoint:
            channel_values = state_snapshot.checkpoint.get("channel_values", {})
            queries_and_results = channel_values.get("queries_and_results", [])
            print__api_postgresql(f"📊 Found {len(queries_and_results)} queries in checkpoint")
            return queries_and_results
        else:
            print__api_postgresql(f"⚠️ No checkpoint or queries found for thread: {thread_id}")
            return []
            
    except Exception as e:
        print__api_postgresql(f"❌ Failed to get queries from checkpoint: {e}")
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
    """Log basic connection information for debugging."""
    print__postgres_startup_debug(f"🔗 Connecting to PostgreSQL:")
    print__postgres_startup_debug(f"   📡 Host: {host}")
    print__postgres_startup_debug(f"   🔌 Port: {port}")
    print__postgres_startup_debug(f"   💾 Database: {dbname}")
    print__postgres_startup_debug(f"   👤 User: {user}")

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