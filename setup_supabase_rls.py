#!/usr/bin/env python3
"""
Script to setup Row Level Security (RLS) for LangGraph checkpointer tables in Supabase.

This script should be run once to configure the security policies for your 
checkpointer tables in Supabase. It enables RLS and creates appropriate policies.

Usage:
    python setup_supabase_rls.py
"""

import os
import sys
import asyncio
import psycopg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def setup_rls_for_supabase():
    """Setup Row Level Security policies for checkpointer tables."""
    
    # Get connection parameters
    user = os.getenv("user")
    password = os.getenv("password") 
    host = os.getenv("host")
    port = os.getenv("port", "5432")
    dbname = os.getenv("dbname")
    
    # Check required parameters
    if not all([user, password, host, dbname]):
        missing = [k for k, v in {"user": user, "password": password, "host": host, "dbname": dbname}.items() if not v]
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        return False
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    
    try:
        print(f"🔗 Connecting to database at {host}...")
        
        async with await psycopg.AsyncConnection.connect(connection_string) as conn:
            print("✅ Connected successfully!")
            
            # Enable autocommit for DDL operations
            await conn.set_autocommit(True)
            
            print("\n📋 Setting up Row Level Security...")
            
            # Check if tables exist first
            tables_exist = await conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name IN ('checkpoints', 'checkpoint_writes') 
                AND table_schema = 'public'
            """)
            
            existing_tables = [row[0] async for row in tables_exist]
            print(f"📊 Found existing tables: {existing_tables}")
            
            if not existing_tables:
                print("⚠️  No checkpointer tables found. Please run your application first to create them.")
                return False
            
            # Enable RLS on both tables
            if 'checkpoints' in existing_tables:
                print("🔒 Enabling RLS on checkpoints table...")
                await conn.execute("ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;")
                
                # Drop existing policies if they exist
                await conn.execute('DROP POLICY IF EXISTS "Allow service role full access" ON checkpoints;')
                
                # Create permissive policy for service role
                await conn.execute("""
                    CREATE POLICY "Allow service role full access" ON checkpoints
                    FOR ALL 
                    USING (true) 
                    WITH CHECK (true);
                """)
                print("✅ RLS configured for checkpoints table")
            
            if 'checkpoint_writes' in existing_tables:
                print("🔒 Enabling RLS on checkpoint_writes table...")
                await conn.execute("ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY;")
                
                # Drop existing policies if they exist
                await conn.execute('DROP POLICY IF EXISTS "Allow service role full access" ON checkpoint_writes;')
                
                # Create permissive policy for service role
                await conn.execute("""
                    CREATE POLICY "Allow service role full access" ON checkpoint_writes
                    FOR ALL 
                    USING (true) 
                    WITH CHECK (true);
                """)
                print("✅ RLS configured for checkpoint_writes table")
            
            # Verify RLS status
            print("\n🔍 Verifying RLS status...")
            rls_status = await conn.execute("""
                SELECT schemaname, tablename, rowsecurity 
                FROM pg_tables 
                WHERE tablename IN ('checkpoints', 'checkpoint_writes')
                AND schemaname = 'public'
            """)
            
            async for row in rls_status:
                schema, table, rls_enabled = row
                status = "✅ ENABLED" if rls_enabled else "❌ DISABLED"
                print(f"  {table}: RLS {status}")
            
            # Show policies
            print("\n📜 Current RLS policies:")
            policies = await conn.execute("""
                SELECT tablename, policyname, cmd, qual, with_check
                FROM pg_policies 
                WHERE tablename IN ('checkpoints', 'checkpoint_writes')
            """)
            
            policy_count = 0
            async for row in policies:
                table, policy, cmd, qual, with_check = row
                print(f"  {table}: {policy} ({cmd})")
                policy_count += 1
            
            if policy_count == 0:
                print("  No policies found")
            
            print(f"\n🎉 RLS setup completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error setting up RLS: {e}")
        return False

async def test_connection():
    """Test basic database connectivity."""
    user = os.getenv("user")
    password = os.getenv("password") 
    host = os.getenv("host")
    port = os.getenv("port", "5432")
    dbname = os.getenv("dbname")
    
    print("🧪 Testing database connection...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Database: {dbname}")
    print(f"   User: {user}")
    print(f"   Password: {'✅ Set' if password else '❌ Missing'}")
    
    if not all([user, password, host, dbname]):
        print("❌ Missing required connection parameters")
        return False
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    
    try:
        async with await psycopg.AsyncConnection.connect(connection_string) as conn:
            result = await conn.execute("SELECT version()")
            version = await result.fetchone()
            print(f"✅ Connection successful!")
            print(f"   PostgreSQL version: {version[0]}")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Supabase RLS Setup Script")
    print("=" * 50)
    
    async def main():
        # Test connection first
        if not await test_connection():
            print("\n❌ Cannot proceed with RLS setup - connection failed")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        
        # Setup RLS
        success = await setup_rls_for_supabase()
        
        if success:
            print("\n🎉 All done! Your Supabase database is now properly secured.")
            print("\nNext steps:")
            print("1. Restart your application")
            print("2. The 'Row Level Security is disabled' warnings should disappear")
            print("3. Your checkpointer tables are now properly secured")
        else:
            print("\n❌ Setup failed. Please check the errors above and try again.")
            sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 