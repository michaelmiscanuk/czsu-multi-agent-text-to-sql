#!/usr/bin/env python3
"""
Test script to verify chat functionality works with the fixed checkpoint tables.
"""

import asyncio
import platform
import os
from dotenv import load_dotenv
from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer

# Fix for Windows ProactorEventLoop issue with psycopg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

async def test_chat_functionality():
    """Test that chat functionality works with the checkpoint tables."""
    print("🧪 Testing chat functionality with PostgreSQL checkpointer...")
    
    try:
        # Get the checkpointer (this should work now with all tables created)
        checkpointer = await get_postgres_checkpointer()
        print(f"✓ Checkpointer type: {type(checkpointer).__name__}")
        
        # Verify it's using PostgreSQL (not InMemorySaver fallback)
        if hasattr(checkpointer, 'conn'):
            print("✅ Using PostgreSQL checkpointer (persistent storage)")
            
            # Quick verification that we can access the database
            async with checkpointer.conn.connection() as conn:
                result = await conn.execute("SELECT COUNT(*) as count FROM checkpoints")
                row = await result.fetchone()
                print(f"✓ Current checkpoints count: {row[0] if row else 0}")
                
                # Check that checkpoint_blobs table exists and is accessible
                result = await conn.execute("SELECT COUNT(*) as count FROM checkpoint_blobs")
                row = await result.fetchone()
                print(f"✓ Current checkpoint_blobs count: {row[0] if row else 0}")
                
        else:
            print("⚠ Using InMemorySaver fallback (non-persistent)")
            
        print("✅ Chat functionality test passed! Ready for use.")
        return True
        
    except Exception as e:
        print(f"❌ Chat functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Close the checkpointer connection pool
        if 'checkpointer' in locals() and hasattr(checkpointer, 'conn'):
            try:
                await checkpointer.conn.close()
                print("🔒 Connection pool closed")
            except Exception as e:
                print(f"⚠ Warning: Could not close connection pool: {e}")

if __name__ == "__main__":
    asyncio.run(test_chat_functionality()) 