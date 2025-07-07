#!/usr/bin/env python3
"""
Debug script to directly query PostgreSQL checkpoints database.
"""

import asyncio
import os
import json
import sys
from datetime import datetime
from psycopg_pool import AsyncConnectionPool

async def query_checkpoints_database():
    """Directly query the PostgreSQL checkpoints database to see what's stored."""
    
    # Get connection details from environment (same as postgres_checkpointer.py)
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host") 
    port = os.getenv("port", "5432")
    dbname = os.getenv("dbname")
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    
    # Thread ID from the debug logs
    thread_id = "645ef712-42df-40fd-a27e-b4be7cf0e500"
    
    print(f"🔍 Querying checkpoints for thread: {thread_id}")
    print(f"📡 Connecting to: {host}:{port}/{dbname}")
    
    try:
        # Create connection pool (same as postgres_checkpointer.py)
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=1,
            min_size=1,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            }
        )
        
        async with pool.connection() as conn:
            print("✅ Connected to PostgreSQL")
            
            # 1. Check what tables exist
            print("\n📋 Available checkpoint tables:")
            result = await conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name LIKE 'checkpoint%' OR table_name = 'users_threads_runs'
                ORDER BY table_name
            """)
            async for row in result:
                print(f"  - {row[0]}")
            
            # 2. Count checkpoints for our thread
            print(f"\n📊 Checkpoint counts for thread {thread_id}:")
            
            result = await conn.execute("""
                SELECT COUNT(*) as count
                FROM checkpoints 
                WHERE thread_id = %s
            """, (thread_id,))
            count = await result.fetchone()
            print(f"  - checkpoints table: {count[0] if count else 0} records")
            
            result = await conn.execute("""
                SELECT COUNT(*) as count
                FROM checkpoint_writes 
                WHERE thread_id = %s
            """, (thread_id,))
            count = await result.fetchone()
            print(f"  - checkpoint_writes table: {count[0] if count else 0} records")
            
            # 3. Get the actual checkpoint data with messages
            print(f"\n💾 Checkpoint data for thread {thread_id}:")
            result = await conn.execute("""
                SELECT 
                    checkpoint_id,
                    thread_ts,
                    checkpoint::text as checkpoint_data
                FROM checkpoints 
                WHERE thread_id = %s 
                ORDER BY thread_ts
                LIMIT 20
            """, (thread_id,))
            
            checkpoints = []
            async for row in result:
                checkpoints.append(row)
            
            print(f"  Found {len(checkpoints)} checkpoints")
            
            for i, (checkpoint_id, thread_ts, checkpoint_data) in enumerate(checkpoints):
                print(f"\n  🔹 Checkpoint {i+1}: {checkpoint_id}")
                print(f"    Timestamp: {thread_ts}")
                
                try:
                    # Parse the checkpoint JSON
                    checkpoint_json = json.loads(checkpoint_data)
                    
                    # Look for messages in channel_values
                    if 'channel_values' in checkpoint_json:
                        channel_values = checkpoint_json['channel_values']
                        
                        if 'messages' in channel_values:
                            messages = channel_values['messages']
                            print(f"    Messages count: {len(messages)}")
                            
                            for j, msg in enumerate(messages):
                                msg_type = msg.get('type', 'unknown')
                                msg_id = msg.get('id', 'no-id')
                                content = msg.get('content', '')
                                content_preview = content[:100] + "..." if len(content) > 100 else content
                                
                                print(f"      Message {j+1}: {msg_type} (id={msg_id})")
                                print(f"        Content: {content_preview}")
                        
                        # Check for queries_and_results
                        if 'queries_and_results' in channel_values:
                            queries = channel_values['queries_and_results']
                            print(f"    Queries count: {len(queries)}")
                            
                        # Check for top_selection_codes
                        if 'top_selection_codes' in channel_values:
                            codes = channel_values['top_selection_codes']
                            print(f"    Selection codes: {codes}")
                    
                except json.JSONDecodeError as e:
                    print(f"    ❌ Failed to parse checkpoint JSON: {e}")
                except Exception as e:
                    print(f"    ⚠ Error processing checkpoint: {e}")
            
            # 4. Check our custom table
            print(f"\n👥 User threads runs for thread {thread_id}:")
            result = await conn.execute("""
                SELECT timestamp, email, run_id
                FROM users_threads_runs 
                WHERE thread_id = %s
                ORDER BY timestamp DESC
                LIMIT 10
            """, (thread_id,))
            
            runs = []
            async for row in result:
                runs.append(row)
            
            print(f"  Found {len(runs)} run entries")
            for timestamp, email, run_id in runs:
                print(f"    {timestamp}: {email} -> {run_id}")
        
        await pool.close()
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(query_checkpoints_database()) 