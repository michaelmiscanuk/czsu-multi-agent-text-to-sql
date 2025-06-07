"""Simple PostgreSQL checkpointer for AsyncPostgresSaver."""

import os
import sys
import asyncio
from typing import Optional
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def create_postgres_checkpointer() -> AsyncPostgresSaver:
    """Create AsyncPostgresSaver with robust connection settings for Supabase."""
    
    # Get connection parameters
    user = os.getenv("user")
    password = os.getenv("password") 
    host = os.getenv("host")
    port = os.getenv("port", "5432")
    dbname = os.getenv("dbname")
    
    # Check required parameters
    if not all([user, password, host, dbname]):
        missing = [k for k, v in {"user": user, "password": password, "host": host, "dbname": dbname}.items() if not v]
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Create connection string with optimized parameters for Supabase
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    # Optimized connection kwargs for Supabase stability
    connection_kwargs = {
        "sslmode": "require",
        "connect_timeout": 30,
        "application_name": "czsu-langgraph-checkpointer",
        # Disable autocommit for better transaction control
        "autocommit": False,
        # Connection pool specific optimizations
        "prepare_threshold": 0,  # Disable prepared statements for better compatibility
    }
    
    # Create connection pool with conservative settings for stability
    pool = AsyncConnectionPool(
        conninfo=connection_string,
        max_size=5,  # Reduced from 20 to avoid overwhelming Supabase
        min_size=1,
        max_idle=1800,  # 30 minutes
        max_lifetime=3600,  # 1 hour - close connections after this time
        timeout=30,  # Pool timeout
        kwargs=connection_kwargs,
        open=False
    )
    
    # Open pool manually
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    
    # Setup tables with proper error handling for concurrent access
    try:
        await checkpointer.setup()
    except Exception as e:
        error_str = str(e)
        if "CREATE INDEX CONCURRENTLY cannot run inside a transaction block" in error_str:
            # Handle the specific case where concurrent index creation fails
            print("⚠ Handling concurrent index creation issue...")
            # Let's try a manual setup with non-concurrent indexes
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                
                # Create tables without concurrent indexes
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint JSONB NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoint_writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value JSONB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                """)
                
                # Create regular (non-concurrent) indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS ix_checkpoints_thread_id 
                    ON checkpoints (thread_id, checkpoint_ns)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS ix_checkpoint_writes_thread_id 
                    ON checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id)
                """)
                print("✓ Database tables created successfully")
        else:
            # Re-raise other setup errors
            raise
    
    return checkpointer

async def get_postgres_checkpointer() -> AsyncPostgresSaver:
    """Get PostgreSQL checkpointer with fallback to InMemorySaver."""
    try:
        checkpointer = await create_postgres_checkpointer()
        print("✓ Connected to PostgreSQL for persistent checkpointing")
        return checkpointer
    except Exception as e:
        print(f"⚠ PostgreSQL failed: {e}")
        print("⚠ Using InMemorySaver (non-persistent)")
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()

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
        
        checkpointer = await get_postgres_checkpointer()
        print(f"Checkpointer type: {type(checkpointer).__name__}")
        
        # Cleanup
        if hasattr(checkpointer, 'pool') and checkpointer.pool:
            await checkpointer.pool.close()
    
    asyncio.run(test()) 