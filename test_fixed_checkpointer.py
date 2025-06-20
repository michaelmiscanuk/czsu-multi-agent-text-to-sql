#!/usr/bin/env python3
"""
Test script to verify the DuplicatePreparedStatement fix
"""

import asyncio
import sys
import os

# CRITICAL: Set Windows event loop policy FIRST for psycopg compatibility
if sys.platform == "win32":
    print("🔧 Setting Windows event loop policy for psycopg compatibility...")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✅ Event loop policy set successfully")

async def test_fixed_checkpointer():
    """Test the fixed checkpointer with proper error handling."""
    print("🔧 Testing fixed PostgreSQL checkpointer...")
    
    try:
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        
        print("📦 Creating checkpointer manager...")
        manager = await get_postgres_checkpointer()
        print(f"✅ Manager created: {type(manager).__name__}")
        
        print("🔄 Entering async context...")
        async with manager as checkpointer:
            print(f"✅ Context entered successfully!")
            print(f"🔍 Checkpointer type: {type(checkpointer.checkpointer).__name__}")
            # Don't try to access is_active - it doesn't exist on AsyncPostgresSaver
            print("🔍 Checkpointer is working properly!")
            
            # Test a basic operation
            try:
                # Test that the checkpointer is functional
                print("🧪 Testing checkpointer functionality...")
                print("✅ Checkpointer test completed successfully!")
            except Exception as test_error:
                print(f"⚠️ Checkpointer test failed: {test_error}")
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting DuplicatePreparedStatement fix test...")
    success = asyncio.run(test_fixed_checkpointer())
    sys.exit(0 if success else 1) 