#!/usr/bin/env python3
"""
PostgreSQL checkpointer module using the official langgraph checkpoint postgres functionality.
This uses the correct table schemas and implementation from the langgraph library.
"""

import asyncio
import platform
import os
from typing import Optional
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