#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using the official langgraph checkpoint postgres functionality.
This uses the correct table schemas and implementation from the langgraph library.
"""

import asyncio
import platform
import os
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # Correct async import
from langgraph.checkpoint.postgres import PostgresSaver  # Correct sync import
from psycopg_pool import AsyncConnectionPool, ConnectionPool

# Fix for Windows ProactorEventLoop issue with psycopg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Database connection parameters
database_pool: Optional[AsyncConnectionPool] = None

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
    """Get PostgreSQL connection string from environment variables."""
    config = get_db_config()
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require"

async def setup_users_threads_runs_table():
    """Setup the users_threads_runs table for chat management."""
    global database_pool
    
    if database_pool is None:
        connection_string = get_connection_string()
        database_pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=3,
            min_size=1,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            }
        )
    
    try:
        async with database_pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Create users_threads_runs table with only 4 columns as requested
            # Use IF NOT EXISTS to preserve existing data on server restarts
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users_threads_runs (
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    email VARCHAR(255) NOT NULL,
                    thread_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) PRIMARY KEY
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
            
            # Enable RLS if this is Supabase
            try:
                await conn.execute("ALTER TABLE users_threads_runs ENABLE ROW LEVEL SECURITY")
                print("✓ RLS enabled on users_threads_runs")
            except Exception as e:
                if "already enabled" in str(e).lower():
                    print("⚠ RLS already enabled on users_threads_runs")
                else:
                    print(f"⚠ Warning: Could not enable RLS on users_threads_runs: {e}")
            
            # Create RLS policy
            try:
                await conn.execute('DROP POLICY IF EXISTS "Allow service role full access" ON users_threads_runs')
                await conn.execute("""
                    CREATE POLICY "Allow service role full access" ON users_threads_runs
                    FOR ALL USING (true) WITH CHECK (true)
                """)
                print("✓ RLS policy created for users_threads_runs")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("⚠ RLS policy already exists for users_threads_runs")
                else:
                    print(f"⚠ Could not create RLS policy for users_threads_runs: {e}")
            
            print("✅ users_threads_runs table verified/created (4 columns: timestamp, email, thread_id, run_id)")
            
    except Exception as e:
        print(f"❌ Error setting up users_threads_runs table: {str(e)}")
        raise

async def create_thread_run_entry(email: str, thread_id: str, run_id: str = None) -> str:
    """Create a new entry in users_threads_runs table.
    
    Args:
        email: User's email address
        thread_id: Thread ID for the conversation
        run_id: Optional run ID, will generate UUID if not provided
    
    Returns:
        The run_id that was created or provided
    """
    global database_pool
    
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    try:
        async with database_pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Insert new entry - run_id is primary key so must be unique
            await conn.execute("""
                INSERT INTO users_threads_runs (timestamp, email, thread_id, run_id)
                VALUES (NOW(), %s, %s, %s)
            """, (email, thread_id, run_id))
            
            print(f"✓ Created thread run entry: email={email}, thread_id={thread_id}, run_id={run_id}")
            return run_id
            
    except Exception as e:
        print(f"❌ Error creating thread run entry: {str(e)}")
        raise

async def get_user_chat_threads(email: str) -> List[Dict[str, Any]]:
    """Get all chat threads for a user, sorted by latest timestamp.
    
    Args:
        email: User's email address
    
    Returns:
        List of dictionaries with thread information:
        [{"thread_id": str, "latest_timestamp": datetime, "run_count": int}, ...]
    """
    global database_pool
    
    try:
        async with database_pool.connection() as conn:
            # Get unique threads with their latest timestamp and run count
            result = await conn.execute("""
                SELECT 
                    thread_id,
                    MAX(timestamp) as latest_timestamp,
                    COUNT(*) as run_count
                FROM users_threads_runs 
                WHERE email = %s 
                GROUP BY thread_id
                ORDER BY MAX(timestamp) DESC
            """, (email,))
            
            threads = []
            async for row in result:
                threads.append({
                    "thread_id": row[0],
                    "latest_timestamp": row[1],
                    "run_count": row[2]
                })
            
            print(f"✓ Retrieved {len(threads)} chat threads for user: {email}")
            return threads
            
    except Exception as e:
        print(f"❌ Error retrieving user chat threads: {str(e)}")
        return []

async def delete_user_thread_entries(email: str, thread_id: str) -> Dict[str, Any]:
    """Delete all entries for a specific user's thread.
    
    Args:
        email: User's email address
        thread_id: Thread ID to delete
    
    Returns:
        Dictionary with deletion results
    """
    global database_pool
    
    try:
        async with database_pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Delete entries for this user's thread
            result = await conn.execute("""
                DELETE FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (email, thread_id))
            
            deleted_count = result.rowcount if hasattr(result, 'rowcount') else 0
            
            print(f"✓ Deleted {deleted_count} thread entries for user: {email}, thread_id: {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "email": email,
                "thread_id": thread_id
            }
            
    except Exception as e:
        print(f"❌ Error deleting user thread entries: {str(e)}")
        return {
            "deleted_count": 0,
            "email": email,
            "thread_id": thread_id,
            "error": str(e)
        }

async def get_postgres_checkpointer():
    """
    Get a PostgreSQL checkpointer using the official langgraph PostgreSQL implementation.
    This ensures we use the correct table schemas and implementation.
    """
    global database_pool
    
    try:
        if database_pool is None:
            connection_string = get_connection_string()
            
            # Create connection pool
            database_pool = AsyncConnectionPool(
                conninfo=connection_string,
                max_size=3,
                min_size=1,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                }
            )
            
            print("🔗 Creating PostgreSQL checkpointer with official library...")
            
        # Create checkpointer with the connection pool
        checkpointer = AsyncPostgresSaver(database_pool)
        
        # Setup tables (this creates all required tables with correct schemas)
        await checkpointer.setup()
        
        # Setup our custom users_threads_runs table
        await setup_users_threads_runs_table()
        
        print("✅ Official PostgreSQL checkpointer initialized successfully")
        return checkpointer
        
    except Exception as e:
        print(f"❌ Error creating PostgreSQL checkpointer: {str(e)}")
        raise

def get_sync_postgres_checkpointer():
    """
    Get a synchronous PostgreSQL checkpointer using the official library.
    """
    try:
        connection_string = get_connection_string()
        
        # Create sync connection pool
        pool = ConnectionPool(
            conninfo=connection_string,
            max_size=3,
            min_size=1,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            }
        )
        
        # Create checkpointer with the connection pool
        checkpointer = PostgresSaver(pool)
        
        # Setup tables (this creates all required tables with correct schemas)
        checkpointer.setup()
        
        print("✅ Sync PostgreSQL checkpointer initialized successfully")
        return checkpointer
        
    except Exception as e:
        print(f"❌ Error creating sync PostgreSQL checkpointer: {str(e)}")
        raise

# For backward compatibility
async def create_postgres_checkpointer():
    """Backward compatibility wrapper."""
    return await get_postgres_checkpointer()

async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str) -> List[Dict[str, Any]]:
    """Get conversation messages from the LangChain PostgreSQL checkpoint history.
    
    This extracts both user questions and final AI responses for proper chat display:
    - User messages: for right-side blue display
    - AI messages: for left-side white display
    
    Args:
        checkpointer: The PostgreSQL checkpointer instance
        thread_id: Thread ID for the conversation
    
    Returns:
        List of message dictionaries in chronological order (user questions + AI answers)
    """
    try:
        print(f"[API-PostgreSQL] 🔍 Retrieving checkpoint history for thread: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # First, get the latest state to extract the original prompt
        latest_state = None
        try:
            state_snapshot = await checkpointer.aget_tuple(config)
            if state_snapshot and state_snapshot.checkpoint:
                latest_state = state_snapshot.checkpoint.get("channel_values", {})
                print(f"[API-PostgreSQL] 📊 Retrieved latest state with keys: {list(latest_state.keys())}")
        except Exception as e:
            print(f"[API-PostgreSQL] ⚠ Could not get latest state: {e}")
        
        # Get all checkpoints for this thread using alist()
        checkpoint_tuples = []
        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoint_tuples.append(checkpoint_tuple)
        
        if not checkpoint_tuples:
            print(f"[API-PostgreSQL] ⚠ No checkpoints found for thread: {thread_id}")
            return []
        
        print(f"[API-PostgreSQL] 📄 Found {len(checkpoint_tuples)} checkpoints")
        
        # Extract conversation messages chronologically
        conversation_messages = []
        
        # Sort checkpoints chronologically (oldest first) using checkpoint order
        checkpoint_tuples.sort(key=lambda x: x.config.get("configurable", {}).get("checkpoint_id", ""))
        
        # Track seen content to avoid duplicates
        seen_user_prompts = set()
        seen_final_answers = set()
        
        # STEP 1: Extract the original user prompt from the latest state or first checkpoint
        original_prompt = None
        
        # Try to get prompt from latest state first
        if latest_state and "prompt" in latest_state:
            original_prompt = latest_state["prompt"]
            print(f"[API-PostgreSQL] 📝 Found original prompt from latest state: '{original_prompt}'")
        
        # If not found in latest state, try first checkpoint metadata
        if not original_prompt and checkpoint_tuples:
            first_checkpoint = checkpoint_tuples[0]
            metadata = first_checkpoint.metadata or {}
            
            if "writes" in metadata and isinstance(metadata["writes"], dict):
                writes = metadata["writes"]
                if "__start__" in writes and isinstance(writes["__start__"], dict):
                    start_data = writes["__start__"]
                    if "prompt" in start_data:
                        original_prompt = start_data["prompt"]
                        print(f"[API-PostgreSQL] 📝 Found original prompt from first checkpoint: '{original_prompt}'")
        
        # Add the original user prompt as the first message
        if original_prompt and original_prompt.strip():
            user_message = {
                "id": "user_1",
                "content": original_prompt.strip(),
                "is_user": True,
                "timestamp": datetime.now(),
                "checkpoint_order": 0,
                "message_order": 1
            }
            conversation_messages.append(user_message)
            seen_user_prompts.add(original_prompt.strip())
            print(f"[API-PostgreSQL] 👤 Added original user prompt: {original_prompt}")
        else:
            print(f"[API-PostgreSQL] ⚠ Could not find original user prompt in state or checkpoints")
        
        # STEP 2: Extract ONLY final formatted answers (with id starting with "run--")
        print(f"[API-PostgreSQL] 🔍 Extracting AI responses from checkpoints...")
        
        # Only look at the LAST few checkpoints to get the final answer
        # The final answer should be in one of the last checkpoints after format_answer_node
        last_checkpoints = checkpoint_tuples[-5:] if len(checkpoint_tuples) > 5 else checkpoint_tuples
        
        for i, checkpoint_tuple in enumerate(last_checkpoints):
            checkpoint = checkpoint_tuple.checkpoint
            if checkpoint and "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                messages = channel_values.get("messages", [])
                
                if messages and len(messages) > 1:
                    # LangGraph state has [summary (SystemMessage), last_message]
                    last_message = messages[-1] if messages else None
                    
                    if last_message:
                        msg_type = type(last_message).__name__
                        msg_content = getattr(last_message, 'content', str(last_message))
                        msg_id = getattr(last_message, 'id', None)
                        
                        # More strict filtering for ONLY the final answer from format_answer_node
                        is_final_answer = (
                            msg_type == "AIMessage" and
                            msg_id and msg_id.startswith("run--") and
                            len(msg_content.strip()) > 10 and
                            msg_content.strip() not in seen_final_answers and
                            # Must NOT contain intermediate patterns
                            not any(keyword in msg_content.lower() for keyword in [
                                "query:", "result:", "select ", "from ", "where ", 
                                "executed a query", "decision:", "schema details", "kolik lidí žije",
                                "how many people live", "analysis:", "breakdown:", "summary:"
                            ]) and
                            # Must be a direct answer (not a question or instruction)
                            not msg_content.strip().endswith('?') and
                            # Should contain actual data/numbers (typical of final answers)
                            any(char.isdigit() for char in msg_content)
                        )
                        
                        if is_final_answer:
                            seen_final_answers.add(msg_content.strip())
                            ai_message = {
                                "id": f"ai_{len(conversation_messages) + 1}",
                                "content": msg_content.strip(),
                                "is_user": False,
                                "timestamp": datetime.now(),
                                "checkpoint_order": len(checkpoint_tuples) + i,  # Ensure it comes after user message
                                "message_order": len(conversation_messages) + 1
                            }
                            conversation_messages.append(ai_message)
                            print(f"[API-PostgreSQL] 🤖 Final answer extracted (id={msg_id}): {msg_content[:50]}...")
                            
                            # Only take the FIRST (most recent) final answer we find
                            break
        
        # Sort all messages by checkpoint order to ensure proper chronological order
        conversation_messages.sort(key=lambda x: (x.get("checkpoint_order", 0), x.get("message_order", 0)))
        
        # Re-assign message order after sorting
        for i, msg in enumerate(conversation_messages):
            msg["message_order"] = i + 1
            msg["id"] = f"{'user' if msg['is_user'] else 'ai'}_{i + 1}"
        
        print(f"[API-PostgreSQL] ✅ Extracted {len(conversation_messages)} conversation messages")
        
        # Debug: Log the actual messages found
        for i, msg in enumerate(conversation_messages):
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print(f"[API-PostgreSQL] {i+1}. {msg_type}: {msg['content'][:50]}...")
        
        return conversation_messages
        
    except Exception as e:
        print(f"[API-PostgreSQL] ❌ Error retrieving messages from checkpoints: {str(e)}")
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
            print(f"[API-PostgreSQL] �� Found {len(queries_and_results)} queries from latest checkpoint")
            return [[query, result] for query, result in queries_and_results]
        
        return []
        
    except Exception as e:
        print(f"[API-PostgreSQL] ⚠ Could not get queries from checkpoint: {e}")
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
                    print(f"✓ RLS enabled on {table}")
                except Exception as e:
                    if "already enabled" in str(e).lower() or "does not exist" in str(e).lower():
                        print(f"⚠ RLS already enabled on {table} or table doesn't exist")
                    else:
                        print(f"⚠ Warning: Could not enable RLS on {table}: {e}")
            
            # Drop existing policies if they exist
            for table in tables:
                try:
                    await conn.execute(f'DROP POLICY IF EXISTS "Allow service role full access" ON {table}')
                except Exception as e:
                    print(f"⚠ Could not drop existing policy on {table}: {e}")
            
            # Create permissive policies for authenticated users
            for table in tables:
                try:
                    await conn.execute(f"""
                        CREATE POLICY "Allow service role full access" ON {table}
                        FOR ALL USING (true) WITH CHECK (true)
                    """)
                    print(f"✓ RLS policy created for {table}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"⚠ RLS policy already exists for {table}")
                    else:
                        print(f"⚠ Could not create RLS policy for {table}: {e}")
            
        print("✓ Row Level Security setup completed (with any warnings noted above)")
    except Exception as e:
        print(f"⚠ Warning: Could not setup RLS policies: {e}")
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
                    print(f"✓ Connection pool health OK - Stats: {stats}")
                except AttributeError:
                    print("✓ Connection pool health check passed")
            except Exception as e:
                print(f"⚠ Connection pool health check failed: {e}")
            
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        print("📊 Connection monitor stopped")

def log_connection_info(host: str, port: str, dbname: str, user: str):
    """Log connection information for debugging."""
    print(f"🔗 PostgreSQL Connection Info:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Database: {dbname}")
    print(f"   User: {user}")
    print(f"   SSL: Required")

# Test and health check functions
async def test_connection_health():
    """Test the health of the PostgreSQL connection."""
    try:
        user = os.getenv("user")
        password = os.getenv("password") 
        host = os.getenv("host")
        port = os.getenv("port", "5432")
        dbname = os.getenv("dbname")
        
        if not all([user, password, host, dbname]):
            print("❌ Missing required environment variables for database connection")
            return False
            
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
        # Test basic connection
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=1,
            min_size=1,
            timeout=5,
            kwargs={"sslmode": "require", "connect_timeout": 5},
            open=False
        )
        
        await pool.open()
        
        async with pool.connection() as conn:
            result = await conn.execute("SELECT 1 as test")
            row = await result.fetchone()
            if row and row[0] == 1:
                print("✓ Database connection test successful")
                return True
        
        await pool.close()
        return False
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    async def test():
        print("Testing PostgreSQL connection...")
        
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host") 
        port = os.getenv("port", "5432")
        dbname = os.getenv("dbname")
        
        print(f"User: {user}")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Database: {dbname}")
        print(f"Password configured: {bool(password)}")
        
        # Test connection health first
        health_ok = await test_connection_health()
        if not health_ok:
            print("❌ Basic connectivity test failed")
            return
        
        # Test full checkpointer setup
        checkpointer = await get_postgres_checkpointer()
        print(f"Checkpointer type: {type(checkpointer).__name__}")
        
        # Cleanup
        if hasattr(checkpointer, 'pool') and checkpointer.pool:
            await checkpointer.pool.close()
            print("✓ Connection pool closed")
    
    asyncio.run(test()) 