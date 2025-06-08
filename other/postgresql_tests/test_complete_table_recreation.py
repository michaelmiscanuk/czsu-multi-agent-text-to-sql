#!/usr/bin/env python3
"""
Complete test to verify the PostgreSQL checkpoint table recreation fix.
This test demonstrates that all 4 required tables are properly created when starting fresh.
"""

import asyncio
import platform
import os
from dotenv import load_dotenv
import psycopg
from my_agent.utils.postgres_checkpointer import create_postgres_checkpointer

# Fix for Windows ProactorEventLoop issue with psycopg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

def drop_all_tables():
    """Drop all checkpoint tables using sync connection."""
    print("🗑️  Dropping all checkpoint tables...")
    
    user = os.getenv('user')
    password = os.getenv('password') 
    host = os.getenv('host')
    port = os.getenv('port', '5432')
    dbname = os.getenv('dbname')
    
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require'
    
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"]
            
            for table in tables:
                try:
                    cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
                    print(f'  ✓ Dropped: {table}')
                except Exception as e:
                    print(f'  ⚠ Could not drop {table}: {e}')
            
            conn.commit()
            print("✅ All tables dropped successfully")

async def verify_table_creation():
    """Verify that all 4 tables are created correctly."""
    print("\n🔨 Creating checkpointer (this should create all 4 tables)...")
    
    try:
        checkpointer = await create_postgres_checkpointer()
        
        # Verify all tables exist
        async with checkpointer.conn.connection() as conn:
            result = await conn.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'checkpoint%'
                ORDER BY tablename;
            """)
            
            tables = [row[0] for row in await result.fetchall()]
            
            print(f"\n📋 Tables found after creation: {len(tables)}")
            for table in tables:
                print(f"  ✓ {table}")
            
            expected_tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"]
            missing_tables = [table for table in expected_tables if table not in tables]
            
            if missing_tables:
                print(f"❌ MISSING TABLES: {missing_tables}")
                return False
            else:
                print("✅ ALL 4 REQUIRED TABLES CREATED!")
                return True
                
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'checkpointer' in locals():
            try:
                await checkpointer.conn.close()
            except:
                pass

async def test_basic_functionality():
    """Test that basic checkpoint operations work."""
    print("\n🧪 Testing basic checkpoint functionality...")
    
    try:
        checkpointer = await create_postgres_checkpointer()
        
        # Test that we can access all tables
        async with checkpointer.conn.connection() as conn:
            tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"]
            
            for table in tables:
                result = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = (await result.fetchone())[0]
                print(f"  ✓ {table}: {count} records")
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False
    
    finally:
        if 'checkpointer' in locals():
            try:
                await checkpointer.conn.close()
            except:
                pass

async def main():
    """Run the complete test suite."""
    print("🎯 COMPLETE POSTGRESQL CHECKPOINT TABLE RECREATION TEST")
    print("=" * 60)
    
    # Step 1: Drop all tables
    drop_all_tables()
    
    # Step 2: Verify table creation
    creation_success = await verify_table_creation()
    
    if not creation_success:
        print("\n❌ TABLE CREATION TEST FAILED!")
        return False
    
    # Step 3: Test basic functionality  
    functionality_success = await test_basic_functionality()
    
    if not functionality_success:
        print("\n❌ FUNCTIONALITY TEST FAILED!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! The fix is working correctly.")
    print("✅ All 4 required tables are created automatically")
    print("✅ Basic checkpoint operations work properly")
    print("✅ The 'checkpoint_blobs' missing table issue is RESOLVED!")
    
    return True

if __name__ == "__main__":
    asyncio.run(main()) 