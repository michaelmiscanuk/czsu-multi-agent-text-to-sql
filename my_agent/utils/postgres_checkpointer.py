#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using AsyncPostgresSaver from LangGraph.
Follows official documentation patterns exactly - no custom wrappers needed.
"""

from __future__ import annotations

import sys
import os
import functools
from typing import Optional, List, Dict, Any, Callable, TypeVar, Awaitable

# CRITICAL: Windows event loop fix MUST be first for PostgreSQL compatibility
if sys.platform == "win32":
    import asyncio
    print(f"[POSTGRES-STARTUP] Windows detected - setting SelectorEventLoop for PostgreSQL compatibility...")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"[POSTGRES-STARTUP] Event loop policy set successfully")

import time
from datetime import datetime
from contextlib import asynccontextmanager

# Import LangGraph's built-in PostgreSQL checkpointer
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    print(f"[POSTGRES-STARTUP] LangGraph AsyncPostgresSaver imported successfully")
except ImportError as e:
    print(f"[POSTGRES-STARTUP] Failed to import AsyncPostgresSaver: {e}")
    AsyncPostgresSaver = None

# Type variable for the retry decorator
T = TypeVar('T')

#==============================================================================
# PREPARED STATEMENT ERROR HANDLING
#==============================================================================
def is_prepared_statement_error(error: Exception) -> bool:
    """Check if an error is related to prepared statements."""
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in [
        'prepared statement',
        'does not exist',
        '_pg3_',
        '_pg_',
        'invalidsqlstatementname'
    ])

def retry_on_prepared_statement_error(max_retries: int = 3):
    """Decorator to retry operations that fail due to prepared statement errors."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if is_prepared_statement_error(e):
                        print__postgresql_debug(f"🔄 Attempt {attempt + 1}/{max_retries + 1} - Prepared statement error: {e}")
                        
                        if attempt < max_retries:
                            print__postgresql_debug("🧹 Clearing prepared statements and retrying...")
                            try:
                                await clear_prepared_statements()
                                # Also try to recreate the checkpointer if it's a global operation
                                global _global_checkpointer_context, _global_checkpointer
                                if _global_checkpointer_context or _global_checkpointer:
                                    print__postgresql_debug("🔄 Recreating checkpointer due to prepared statement error...")
                                    await close_async_postgres_saver()
                                    await create_async_postgres_saver()
                            except Exception as cleanup_error:
                                print__postgresql_debug(f"⚠️ Error during cleanup: {cleanup_error}")
                            continue
                    
                    # If it's not a prepared statement error, or we've exhausted retries, re-raise
                    raise
            
            # This should never be reached, but just in case
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
    return {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }

def get_connection_string():
    """Get PostgreSQL connection string for LangGraph checkpointer.
    
    CRITICAL FIX: Use truly unique application names to avoid prepared statement conflicts.
    ENHANCED FIX: Add connection parameters to reduce prepared statement issues.
    """
    global _connection_string_cache
    
    if _connection_string_cache is not None:
        return _connection_string_cache
    
    config = get_db_config()
    
    # Use process ID + startup time + random for truly unique application name
    import os
    import uuid
    process_id = os.getpid()
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    
    # ENHANCED: Add connection parameters to reduce prepared statement issues
    _connection_string_cache = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"sslmode=require"
        f"&application_name=czsu_langgraph_{process_id}_{startup_time}_{random_id}"
        f"&connect_timeout=30"
    )
    
    print__postgresql_debug(f"🔗 Generated enhanced connection string with app name: czsu_langgraph_{process_id}_{startup_time}_{random_id}")
    
    return _connection_string_cache

def check_postgres_env_vars():
    """Check if all required PostgreSQL environment variables are set."""
    required_vars = ['host', 'port', 'dbname', 'user', 'password']
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print__postgresql_debug(f"Missing required environment variables: {missing_vars}")
        return False
    else:
        print__postgresql_debug("All required PostgreSQL environment variables are set")
        return True

async def clear_prepared_statements():
    """Clear any existing prepared statements to avoid conflicts.
    
    This is necessary because psycopg caches prepared statements globally
    and multiple checkpointer instances can conflict.
    
    Uses a completely separate connection to avoid interfering with checkpointer.
    """
    try:
        config = get_db_config()
        # Use a different application name for the cleanup connection
        import uuid
        cleanup_app_name = f"czsu_cleanup_{uuid.uuid4().hex[:8]}"
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"
        
        import psycopg
        async with await psycopg.AsyncConnection.connect(connection_string) as conn:
            async with conn.cursor() as cur:
                # Get all prepared statements for our application
                await cur.execute("""
                    SELECT name FROM pg_prepared_statements 
                    WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                """)
                prepared_statements = await cur.fetchall()
                
                if prepared_statements:
                    print__postgresql_debug(f"🧹 Found {len(prepared_statements)} prepared statements to clear")
                    
                    # Drop each prepared statement
                    for stmt in prepared_statements:
                        stmt_name = stmt[0]
                        try:
                            await cur.execute(f"DEALLOCATE {stmt_name};")
                            print__postgresql_debug(f"🧹 Cleared prepared statement: {stmt_name}")
                        except Exception as e:
                            print__postgresql_debug(f"⚠️ Could not clear prepared statement {stmt_name}: {e}")
                    
                    print__postgresql_debug(f"✅ Cleared {len(prepared_statements)} prepared statements")
                else:
                    print__postgresql_debug("✅ No prepared statements to clear")
                
    except Exception as e:
        print__postgresql_debug(f"⚠️ Error clearing prepared statements (non-fatal): {e}")
        # Don't raise - this is a cleanup operation and shouldn't block checkpointer creation

# ENHANCED OFFICIAL ASYNCPOSTGRESSAVER IMPLEMENTATION
@retry_on_prepared_statement_error(max_retries=3)
async def create_async_postgres_saver():
    """
    Create AsyncPostgresSaver using the OFFICIAL pattern from documentation.
    This follows the exact pattern shown in the AsyncPostgresSaver docs.
    
    CRITICAL FIX: Clear prepared statements first to avoid conflicts.
    ENHANCED FIX: Add retry logic for prepared statement errors.
    """
    global _global_checkpointer_context, _global_checkpointer
    
    # CRITICAL: Clear any existing state first to avoid conflicts
    if _global_checkpointer_context or _global_checkpointer:
        print__postgresql_debug("🧹 Clearing existing checkpointer state to avoid conflicts...")
        try:
            if _global_checkpointer_context:
                await _global_checkpointer_context.__aexit__(None, None, None)
        except Exception as e:
            print__postgresql_debug(f"⚠️ Error during state cleanup: {e}")
        finally:
            _global_checkpointer_context = None
            _global_checkpointer = None
    
    # CRITICAL: Clear prepared statements to avoid conflicts
    print__postgresql_debug("🧹 Clearing prepared statements to avoid conflicts...")
    await clear_prepared_statements()
    
    if not AsyncPostgresSaver:
        raise Exception("AsyncPostgresSaver not available")
    
    if not check_postgres_env_vars():
        raise Exception("Missing required PostgreSQL environment variables")
    
    print__postgresql_debug("🚀 Creating AsyncPostgresSaver using official from_conn_string...")
    
    try:
        # ENHANCED: Use connection string with better timeout settings
        connection_string = get_connection_string()
        
        # CORRECT USAGE: from_conn_string returns AsyncIterator[AsyncPostgresSaver]
        # According to docs, this should be used as an async context manager
        # We need to store the async context manager for later cleanup
        _global_checkpointer_context = AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # Disable pipeline mode for stability
            serde=None  # Use default serialization
        )
        
        # CORRECT: Use async context manager protocol properly
        # The __aenter__ method returns the actual AsyncPostgresSaver instance
        _global_checkpointer = await _global_checkpointer_context.__aenter__()
        
        print__postgresql_debug("✅ AsyncPostgresSaver created using official factory method")
        print__postgresql_debug(f"✅ Checkpointer type: {type(_global_checkpointer).__name__}")
        
        # Setup the checkpointer (creates tables) - REQUIRED by docs
        print__postgresql_debug("🔧 Running checkpointer setup (required by docs)...")
        await _global_checkpointer.setup()
        print__postgresql_debug("✅ AsyncPostgresSaver setup complete - LangGraph tables created")
        
        # Verify the checkpointer is working by testing a simple operation
        print__postgresql_debug("🧪 Testing checkpointer with a simple operation...")
        test_config = {"configurable": {"thread_id": "setup_test"}}
        test_result = await _global_checkpointer.aget(test_config)
        print__postgresql_debug(f"✅ Checkpointer test successful: {test_result is None} (expected None for new thread)")
        
        # Now setup our custom users_threads_runs table using the same connection approach
        await setup_users_threads_runs_table()
        
        return _global_checkpointer
        
    except Exception as e:
        print__postgresql_debug(f"❌ Failed to create AsyncPostgresSaver: {e}")
        import traceback
        print__postgresql_debug(f"🔍 Full traceback: {traceback.format_exc()}")
        
        # Clean up on failure
        if _global_checkpointer_context:
            try:
                await _global_checkpointer_context.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print__postgresql_debug(f"⚠️ Error during cleanup: {cleanup_error}")
            _global_checkpointer_context = None
        _global_checkpointer = None
        raise

async def close_async_postgres_saver():
    """Close the AsyncPostgresSaver properly using the context manager."""
    global _global_checkpointer_context, _global_checkpointer
    
    print__postgresql_debug("🔄 Closing AsyncPostgresSaver using official context manager...")
    
    if _global_checkpointer_context:
        try:
            await _global_checkpointer_context.__aexit__(None, None, None)
            print__postgresql_debug("✅ AsyncPostgresSaver closed properly")
        except Exception as e:
            print__postgresql_debug(f"⚠️ Error during AsyncPostgresSaver cleanup: {e}")
        finally:
            _global_checkpointer_context = None
            _global_checkpointer = None

@retry_on_prepared_statement_error(max_retries=2)
async def get_global_checkpointer():
    """Get the global checkpointer instance (for API compatibility).
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    global _global_checkpointer
    
    if _global_checkpointer is None:
        _global_checkpointer = await create_async_postgres_saver()
    
    return _global_checkpointer

# SIMPLIFIED USERS_THREADS_RUNS TABLE MANAGEMENT
# We'll use a simple connection approach since AsyncPostgresSaver manages its own connections
async def setup_users_threads_runs_table():
    """Create the users_threads_runs table using direct connection."""
    try:
        print__postgresql_debug("Setting up users_threads_runs table using direct connection...")
        
        # Use direct connection for table setup (simpler than pool management)
        import psycopg
        
        async with await psycopg.AsyncConnection.connect(get_connection_string()) as conn:
            # Create table with correct schema
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

            print__postgresql_debug("users_threads_runs table and indexes created successfully")

    except Exception as e:
        print__postgresql_debug(f"Failed to setup users_threads_runs table: {e}")
        raise

@asynccontextmanager
async def get_direct_connection():
    """Get a direct database connection for users_threads_runs operations."""
    import psycopg
    
    async with await psycopg.AsyncConnection.connect(get_connection_string()) as conn:
        yield conn

# MISSING FUNCTIONS NEEDED BY API SERVER
async def get_healthy_pool():
    """Get a healthy PostgreSQL connection pool for direct operations."""
    try:
        # For the simplified approach, we use direct connections instead of pooling
        # Return a connection factory that mimics pool behavior
        class DirectConnectionFactory:
            def __init__(self, connection_string):
                self.connection_string = connection_string
            
            @asynccontextmanager
            async def connection(self):
                """Provide a connection that mimics pool.connection() interface."""
                import psycopg
                async with await psycopg.AsyncConnection.connect(self.connection_string) as conn:
                    yield conn
        
        return DirectConnectionFactory(get_connection_string())
        
    except Exception as e:
        print__postgresql_debug(f"Failed to create connection factory: {e}")
        raise

@retry_on_prepared_statement_error(max_retries=2)
async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str, user_email: str = None) -> List[Dict[str, Any]]:
    """
    Get conversation messages from checkpoints - USING OFFICIAL ASYNCPOSTGRESSAVER METHODS.
    
    This function properly extracts messages from LangGraph checkpoints using the official
    AsyncPostgresSaver methods as documented.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        print__api_postgresql(f"🔍 Retrieving conversation messages for thread: {thread_id}")
        
        # 🔒 SECURITY CHECK: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__api_postgresql(f"🔒 Verifying thread ownership for user: {user_email}")
            
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
                        print__api_postgresql(f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - access denied")
                        return []
                    
                    print__api_postgresql(f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
            except Exception as e:
                print__api_postgresql(f"⚠ Could not verify thread ownership: {e}")
                return []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use the OFFICIAL AsyncPostgresSaver alist() method as documented
        checkpoint_tuples = []
        try:
            print__api_postgresql(f"🔍 Using official AsyncPostgresSaver.alist() method")
            
            # Use alist() method exactly as shown in the documentation
            async for checkpoint_tuple in checkpointer.alist(config, limit=50):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__api_postgresql(f"❌ Error using alist(): {alist_error}")
            
            # Fallback: use aget_tuple() to get the latest checkpoint
            if not checkpoint_tuples:
                print__api_postgresql(f"🔄 Trying fallback method using aget_tuple()...")
                try:
                    state_snapshot = await checkpointer.aget_tuple(config)
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
        
        # Sort checkpoints chronologically (oldest first) based on timestamp
        checkpoint_tuples.sort(key=lambda x: x.checkpoint.get("ts", "") if x.checkpoint else "")
        
        # Extract conversation messages chronologically
        conversation_messages = []
        seen_prompts = set()
        seen_answers = set()
        
        print__api_postgresql(f"🔍 Extracting messages from {len(checkpoint_tuples)} checkpoints...")
        
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}
            
            if not checkpoint:
                continue
                
            print__api_postgresql(f"🔍 Processing checkpoint {checkpoint_index + 1}/{len(checkpoint_tuples)}")
            
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
                                user_message = {
                                    "id": f"user_{len(conversation_messages) + 1}",
                                    "content": prompt.strip(),
                                    "is_user": True,
                                    "timestamp": datetime.fromtimestamp(1700000000 + checkpoint_index * 1000),
                                    "checkpoint_order": checkpoint_index,
                                    "message_order": len(conversation_messages) + 1
                                }
                                conversation_messages.append(user_message)
                                print__api_postgresql(f"👤 Found user prompt: {prompt[:50]}...")
            
            # EXTRACT AI RESPONSES from channel_values
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
                    ai_message = {
                        "id": f"ai_{len(conversation_messages) + 1}",
                        "content": final_answer.strip(),
                        "is_user": False,
                        "timestamp": datetime.fromtimestamp(1700000000 + checkpoint_index * 1000 + 500),
                        "checkpoint_order": checkpoint_index,
                        "message_order": len(conversation_messages) + 1
                    }
                    conversation_messages.append(ai_message)
                    print__api_postgresql(f"🤖 Found final_answer: {final_answer[:100]}...")
                
                # Method 2: Look for messages with AI content (fallback)
                elif "messages" in channel_values:
                    messages = channel_values["messages"]
                    if isinstance(messages, list) and messages:
                        for msg in reversed(messages):
                            if (hasattr(msg, 'content') and 
                                msg.content and 
                                getattr(msg, 'type', None) == 'ai' and
                                len(msg.content.strip()) > 20 and
                                msg.content.strip() not in seen_answers):
                                
                                seen_answers.add(msg.content.strip())
                                ai_message = {
                                    "id": f"ai_{len(conversation_messages) + 1}",
                                    "content": msg.content.strip(),
                                    "is_user": False,
                                    "timestamp": datetime.fromtimestamp(1700000000 + checkpoint_index * 1000 + 500),
                                    "checkpoint_order": checkpoint_index,
                                    "message_order": len(conversation_messages) + 1
                                }
                                conversation_messages.append(ai_message)
                                print__api_postgresql(f"🤖 Found AI message: {msg.content[:100]}...")
                                break
        
        # Sort all messages by timestamp to ensure proper chronological order
        conversation_messages.sort(key=lambda x: x.get("timestamp", datetime.now()))
        
        # Re-assign sequential IDs and message order after sorting
        for i, msg in enumerate(conversation_messages):
            msg["message_order"] = i + 1
            msg["id"] = f"{'user' if msg['is_user'] else 'ai'}_{i + 1}"
        
        print__api_postgresql(f"✅ Extracted {len(conversation_messages)} conversation messages")
        
        # Debug: Log all messages found
        for i, msg in enumerate(conversation_messages):
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print__api_postgresql(f"{i+1}. {msg_type}: {msg['content'][:50]}...")
        
        return conversation_messages
        
    except Exception as e:
        print__api_postgresql(f"❌ Error retrieving messages from checkpoints: {str(e)}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        return []

# HELPER FUNCTIONS FOR COMPATIBILITY - USING DIRECT CONNECTIONS
@retry_on_prepared_statement_error(max_retries=2)
async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new thread run entry in the database.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        import uuid
        if not run_id:
            run_id = str(uuid.uuid4())
        
        print__api_postgresql(f"Creating thread run entry: user={email}, thread={thread_id}, run={run_id}")
        
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
        
        print__api_postgresql(f"Thread run entry created successfully: {run_id}")
        return run_id
    except Exception as e:
        print__api_postgresql(f"Failed to create thread run entry: {e}")
        # Return the run_id even if database storage fails
        import uuid
        if not run_id:
            run_id = str(uuid.uuid4())
        print__api_postgresql(f"Returning run_id despite database error: {run_id}")
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
        import traceback
        print__api_postgresql(f"Full traceback: {traceback.format_exc()}")
        raise

# BACKWARD COMPATIBILITY FUNCTIONS
async def get_postgres_checkpointer():
    """Backward compatibility wrapper."""
    return await get_global_checkpointer()

# Add the missing function back after setup_users_threads_runs_table
@retry_on_prepared_statement_error(max_retries=2)
async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str):
    """Get queries and results from the latest checkpoint for a thread.
    
    ENHANCED: Add retry logic for prepared statement errors.
    """
    try:
        print__postgresql_debug(f"Getting queries and results from latest checkpoint for thread: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get the latest checkpoint
        state_snapshot = await checkpointer.aget_tuple(config)
        
        if not state_snapshot or not state_snapshot.checkpoint:
            print__postgresql_debug(f"No checkpoint found for thread: {thread_id}")
            return []
        
        # Extract queries and results from checkpoint
        checkpoint = state_snapshot.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        
        # Look for queries_and_results in various places
        queries_and_results = channel_values.get("queries_and_results", [])
        
        if not queries_and_results:
            # Try to extract from iteration_results
            iteration_results = channel_values.get("iteration_results", {})
            for iteration_key, iteration_data in iteration_results.items():
                if isinstance(iteration_data, dict):
                    iter_queries = iteration_data.get("queries_and_results", [])
                    if iter_queries:
                        queries_and_results.extend(iter_queries)
        
        print__postgresql_debug(f"Found {len(queries_and_results)} queries and results for thread: {thread_id}")
        return queries_and_results
        
    except Exception as e:
        print__postgresql_debug(f"Error getting queries and results from checkpoint: {e}")
        return []

if __name__ == "__main__":
    import asyncio
    
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