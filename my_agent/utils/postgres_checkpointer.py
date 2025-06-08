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

async def setup_rls_policies(pool: AsyncConnectionPool):
    """Setup Row Level Security policies for checkpointer tables in Supabase."""
    try:
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Enable RLS on checkpoints table
            await conn.execute("ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY")
            
            # Enable RLS on checkpoint_writes table  
            await conn.execute("ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY")
            
            # Drop existing policies if they exist (separate statements)
            await conn.execute('DROP POLICY IF EXISTS "Allow service role full access" ON checkpoints')
            await conn.execute('DROP POLICY IF EXISTS "Allow service role full access" ON checkpoint_writes')
            
            # Create permissive policies for authenticated users
            await conn.execute("""
                CREATE POLICY "Allow service role full access" ON checkpoints
                FOR ALL USING (true) WITH CHECK (true)
            """)
            
            await conn.execute("""
                CREATE POLICY "Allow service role full access" ON checkpoint_writes
                FOR ALL USING (true) WITH CHECK (true)
            """)
            
        print("✓ Row Level Security policies configured successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not setup RLS policies: {e}")
        # Don't fail the entire setup if RLS setup fails

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
    
    # Log connection info for debugging
    log_connection_info(host, port, dbname, user)
    
    # Create connection string with optimized parameters for Supabase
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    # Optimized connection kwargs for Supabase stability
    connection_kwargs = {
        "sslmode": "require",
        "connect_timeout": 10,  # Connection timeout
        "application_name": "czsu-langgraph-checkpointer",
        # Disable autocommit for better transaction control
        "autocommit": False,
        # Connection pool specific optimizations
        "prepare_threshold": 0,  # Disable prepared statements for better compatibility
    }
    
    print(f"⚙️  Pool Configuration:")
    print(f"   Max Size: 3")
    print(f"   Min Size: 1") 
    print(f"   Max Idle: 300s")
    print(f"   Max Lifetime: 1800s")
    print(f"   Timeout: 10s")
    
    # Create connection pool with conservative settings for stability
    pool = AsyncConnectionPool(
        conninfo=connection_string,
        max_size=3,  # Further reduced to avoid overwhelming Supabase
        min_size=1,
        max_idle=300,  # 5 minutes - shorter idle time
        max_lifetime=1800,  # 30 minutes - shorter lifetime
        timeout=10,  # Shorter pool timeout
        kwargs=connection_kwargs,
        open=False
    )
    
    # Open pool manually with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Opening connection pool (attempt {attempt + 1}/{max_retries})...")
            await pool.open()
            print(f"✅ Connection pool opened successfully")
            break
        except Exception as e:
            print(f"⚠ Pool open attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    checkpointer = AsyncPostgresSaver(pool)
    
    # Setup tables with proper error handling for concurrent access
    try:
        print("🔨 Setting up checkpointer tables...")
        await checkpointer.setup()
        print("✓ Checkpointer tables setup completed")
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
                    );
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
                    );
                """)
                
                # Create regular (non-concurrent) indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS ix_checkpoints_thread_id 
                    ON checkpoints (thread_id, checkpoint_ns);
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS ix_checkpoint_writes_thread_id 
                    ON checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id);
                """)
                print("✓ Database tables created successfully")
        elif "already exists" in error_str.lower():
            print("✓ Tables already exist, continuing...")
        else:
            # Re-raise other setup errors
            raise
    
    # Setup Row Level Security policies
    await setup_rls_policies(pool)
    
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