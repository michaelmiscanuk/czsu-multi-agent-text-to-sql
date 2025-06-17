#!/usr/bin/env python3
"""
Test script to verify Supabase PostgreSQL connection with new SSL settings.
Run this to test if the connection issues are resolved.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Windows compatibility fix - must be set BEFORE any async operations
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("🔧 Windows: Set SelectorEventLoop policy for PostgreSQL compatibility")

async def test_supabase_connection():
    """Test the new Supabase connection settings."""
    
    print("🚀 Testing Supabase PostgreSQL Connection")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ['user', 'password', 'host', 'dbname']
    config = {}
    missing_vars = []
    
    for var in required_vars:
        config[var] = os.getenv(var)
        if not config[var]:
            missing_vars.append(var)
    
    config['port'] = os.getenv('port', '5432')
    
    print("📋 Environment Variables:")
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Database: {config['dbname']}")
    print(f"   User: {config['user']}")
    print(f"   Password: {'✅ SET' if config['password'] else '❌ MISSING'}")
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {missing_vars}")
        print("💡 Make sure your .env file contains:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
        return False
    
    # Test with new connection string format
    print(f"\n🔍 Testing connection with new Supabase-optimized settings...")
    
    try:
        import psycopg
        from psycopg_pool import AsyncConnectionPool
        
        # Build the new connection string (same as in the fixed code)
        connection_string = (
            f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
            f"?sslmode=require"                     # REQUIRED for Supabase
            f"&connect_timeout=20"                  # Timeout for initial connection 
            f"&application_name=czsu_agent_test"    # Application identification
            f"&keepalives_idle=600"                 # Keep connection alive (10 minutes)
            f"&keepalives_interval=30"              # Send keepalive every 30 seconds
            f"&keepalives_count=3"                  # 3 failed keepalives before disconnect
            f"&tcp_user_timeout=30000"              # TCP timeout (30 seconds)
        )
        
        # Show connection string (without password)
        debug_string = connection_string.replace(config['password'], '***')
        print(f"🔗 Connection string: {debug_string}")
        
        print(f"\n🧪 Test 1: Basic connection test...")
        
        # Test basic connection
        async with await psycopg.AsyncConnection.connect(
            connection_string,
            autocommit=True,
            connect_timeout=15
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 as test, NOW() as current_time, version() as pg_version")
                result = await cur.fetchone()
                
                print(f"✅ Basic connection successful!")
                print(f"   Test result: {result[0]}")
                print(f"   Server time: {result[1]}")
                print(f"   PostgreSQL version: {result[2][:60]}...")
        
        print(f"\n🧪 Test 2: Connection pool test...")
        
        # Test connection pool
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=2,
            min_size=0,
            timeout=20,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,
                "connect_timeout": 15
            },
            open=False
        )
        
        await pool.open()
        print(f"✅ Connection pool opened successfully!")
        
        # Test pool connection
        async with pool.connection() as conn:
            result = await conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            row = await result.fetchone()
            table_count = row[0] if row else 0
            print(f"✅ Pool connection test successful! Found {table_count} public tables")
        
        await pool.close()
        print(f"✅ Connection pool closed successfully!")
        
        print(f"\n🎉 All tests passed! Your Supabase connection is working correctly.")
        print(f"\n💡 Next steps:")
        print(f"   1. Restart your application")
        print(f"   2. The connection errors should be resolved")
        print(f"   3. Monitor logs for 'Basic Supabase connection successful'")
        
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"\n❌ Connection test failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Provide specific troubleshooting guidance
        print(f"\n💡 Troubleshooting:")
        
        if "ssl" in error_msg:
            print("🔐 SSL Connection Issue:")
            print("   1. Verify your Supabase project is active")
            print("   2. Check if your IP is whitelisted in Supabase dashboard")
            print("   3. Verify your database credentials are correct")
            print("   4. Try connecting from Supabase dashboard first")
            
        elif "authentication" in error_msg or "password" in error_msg:
            print("🔑 Authentication Issue:")
            print("   1. Double-check your database password")
            print("   2. Verify you're using the correct user (usually 'postgres')")
            print("   3. Make sure you're using the service role key if required")
            
        elif "timeout" in error_msg or "connection" in error_msg:
            print("⏰ Connection/Timeout Issue:")
            print("   1. Check your internet connection")
            print("   2. Verify Supabase service is running (check status page)")
            print("   3. Try reducing connection timeout settings")
            print("   4. Check for firewall blocking the connection")
            
        elif "host" in error_msg or "name resolution" in error_msg:
            print("🌐 DNS/Host Issue:")
            print("   1. Verify your Supabase host URL is correct")
            print("   2. Check if you can ping the host")
            print("   3. Try using IP address instead of hostname")
            
        else:
            print("❓ Unknown Issue:")
            print("   1. Check Supabase dashboard for any alerts")
            print("   2. Try connecting with a different client (e.g., pgAdmin)")
            print("   3. Contact Supabase support if issue persists")
        
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_supabase_connection())
        
        if result:
            print(f"\n✅ SUCCESS: Connection test completed successfully")
            sys.exit(0)
        else:
            print(f"\n❌ FAILED: Connection test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during test: {e}")
        sys.exit(1) 