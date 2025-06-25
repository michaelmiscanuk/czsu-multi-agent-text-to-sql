#!/usr/bin/env python3
"""
Test script for PostgreSQL connection to Supabase
"""

import sys
import os

# CRITICAL: Windows event loop fix MUST be first for PostgreSQL compatibility
if sys.platform == "win32":
    import asyncio
    print("🔧 Windows detected - setting SelectorEventLoop for PostgreSQL compatibility...")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✅ Event loop policy set successfully")

import asyncio
import psycopg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_connection():
    """Test PostgreSQL connection with Supabase credentials from .env file."""
    
    # Get connection details from environment variables
    config = {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }
    
    # Check if all required variables are set
    missing_vars = []
    for key, value in config.items():
        if not value:
            missing_vars.append(key)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Make sure your .env file contains: user, password, host, port, dbname")
        return
    
    # Build connection string
    conn_string = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"sslmode=require&application_name=czsu_test_connection"
    )
    
    print("🔍 Testing PostgreSQL connection...")
    print(f"📍 Host: {config['host']}:{config['port']}")
    print(f"🗄️ Database: {config['dbname']}")
    print(f"👤 User: {config['user']}")
    print()
    
    try:
        async with await psycopg.AsyncConnection.connect(conn_string) as conn:
            print("✅ PostgreSQL connection successful!")
            
            # Test basic query
            async with conn.cursor() as cur:
                await cur.execute('SELECT version();')
                result = await cur.fetchone()
                print(f"🐘 PostgreSQL version: {result[0][:80]}...")
                
                # Test table creation (for checkpointer compatibility)
                await cur.execute("SELECT 1;")
                test_result = await cur.fetchone()
                if test_result[0] == 1:
                    print("✅ Basic query execution works")
                
                # Check if checkpointer tables exist
                await cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name LIKE '%checkpoint%';
                """)
                checkpoint_tables = await cur.fetchall()
                
                if checkpoint_tables:
                    print(f"📋 Found {len(checkpoint_tables)} checkpoint-related tables:")
                    for table in checkpoint_tables:
                        print(f"   - {table[0]}")
                else:
                    print("📋 No checkpoint tables found (expected for fresh setup)")
                
                # Check if users_threads_runs table exists
                await cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'users_threads_runs'
                    );
                """)
                table_exists = await cur.fetchone()
                
                if table_exists[0]:
                    await cur.execute("SELECT COUNT(*) FROM users_threads_runs;")
                    count = await cur.fetchone()
                    print(f"👥 users_threads_runs table exists with {count[0]} entries")
                else:
                    print("👥 users_threads_runs table not found (will be created on first use)")
            
            print()
            print("🎉 All PostgreSQL connection tests passed!")
            print("💡 Your Supabase connection is ready for the application")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("1. Check your .env file contains correct Supabase credentials")
        print("2. Verify your Supabase project is active")
        print("3. Check if your IP is allowed in Supabase settings")
        print("4. Ensure the password doesn't contain special characters that need escaping")

if __name__ == "__main__":
    asyncio.run(test_connection()) 