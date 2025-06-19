#!/usr/bin/env python3
import asyncio
import os
import sys

# Set event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def check_db():
    from psycopg_pool import AsyncConnectionPool
    
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host") 
    port = os.getenv("port", "5432")
    dbname = os.getenv("dbname")
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    
    print(f"Checking database: {host}:{port}/{dbname}")
    
    pool = AsyncConnectionPool(
        conninfo=connection_string,
        max_size=1,
        min_size=1,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    )
    
    async with pool.connection() as conn:
        # Check if users_threads_runs table exists
        print("\n📋 Checking users_threads_runs table...")
        result = await conn.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'users_threads_runs'
            )
        """)
        
        table_exists = await result.fetchone()
        print(f"Table exists: {table_exists[0] if table_exists else False}")
        
        if table_exists and table_exists[0]:
            # Get table structure
            print("\n📊 Table structure:")
            result = await conn.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'users_threads_runs'
                ORDER BY ordinal_position
            """)
            
            async for row in result:
                print(f"  - {row[0]}: {row[1]}")
            
            # Count all records
            result = await conn.execute("SELECT COUNT(*) FROM users_threads_runs")
            count = await result.fetchone()
            print(f"\n📈 Total records: {count[0] if count else 0}")
            
            # Get sample data
            if count and count[0] > 0:
                print("\n📄 Sample records:")
                result = await conn.execute("""
                    SELECT timestamp, email, thread_id, run_id 
                    FROM users_threads_runs 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """)
                
                async for row in result:
                    print(f"  {row[0]} | {row[1]} | {row[2][:12]}... | {row[3][:12]}...")
                
                # Check unique emails
                result = await conn.execute("SELECT DISTINCT email FROM users_threads_runs")
                emails = []
                async for row in result:
                    emails.append(row[0])
                print(f"\n👥 Unique users: {emails}")
                
                # Check unique threads per user
                for email in emails[:3]:  # Show first 3 users
                    result = await conn.execute("""
                        SELECT COUNT(DISTINCT thread_id) 
                        FROM users_threads_runs 
                        WHERE email = %s
                    """, (email,))
                    thread_count = await result.fetchone()
                    print(f"  {email}: {thread_count[0] if thread_count else 0} threads")
        
        # Also check if checkpoint tables exist
        print("\n📋 Checking LangGraph checkpoint tables...")
        checkpoint_tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]
        
        for table in checkpoint_tables:
            result = await conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table,))
            
            exists = await result.fetchone()
            if exists and exists[0]:
                result = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = await result.fetchone()
                print(f"  ✓ {table}: {count[0] if count else 0} records")
            else:
                print(f"  ✗ {table}: not found")
    
    await pool.close()

if __name__ == "__main__":
    asyncio.run(check_db()) 