#!/usr/bin/env python3
"""
Test LangGraph Checkpointer Direct Usage

This script tests if we can use LangGraph's built-in AsyncPostgresSaver directly
in a graph without our custom wrapper.
"""

import os
import sys
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows event loop policy first for psycopg compatibility
if sys.platform == "win32":
    print("🔧 Detected Windows platform - setting compatible event loop policy")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Force close any existing event loop and create a fresh SelectorEventLoop
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop and not current_loop.is_closed():
            print(f"🔧 Closing existing {type(current_loop).__name__}")
            current_loop.close()
    except RuntimeError:
        # No event loop exists yet, which is fine
        pass
    
    # Create a new SelectorEventLoop explicitly and set it as the running loop
    new_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
    asyncio.set_event_loop(new_loop)
    print(f"🔧 Created new {type(new_loop).__name__}")

# Load environment variables
load_dotenv()

# Import the checkpointer
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    print("✅ Successfully imported LangGraph AsyncPostgresSaver")
except ImportError:
    print("❌ Failed to import AsyncPostgresSaver. Make sure langgraph-checkpoint-postgres is installed.")
    sys.exit(1)

# Function to get the PostgreSQL connection string
def get_psycopg_connection_string():
    """Get PostgreSQL connection string optimized for psycopg (for checkpointer)."""
    config = {
        'user': os.getenv('user'),
        'password': os.getenv('password'),
        'host': os.getenv('host'),
        'port': os.getenv('port', '5432'),
        'dbname': os.getenv('dbname')
    }
    
    print(f"🔗 Building psycopg connection string")
    
    # psycopg-optimized connection string with unique application name
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode=require"
        f"&application_name=test_checkpointer_{uuid.uuid4().hex[:8]}"
    )
    
    # Log connection details (without password)
    debug_string = connection_string.replace(config['password'], '***')
    print(f"🔗 Connection string: {debug_string}")
    
    return connection_string

async def test_direct_checkpointer_usage():
    """Test using LangGraph's AsyncPostgresSaver directly."""
    try:
        print("\n🧪 Testing direct AsyncPostgresSaver usage\n")
        
        # Get connection string with unique application name
        connection_string = get_psycopg_connection_string()
        
        # Generate a unique thread ID for this test run
        unique_thread_id = f"test_direct_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        print(f"🔑 Using unique thread ID: {unique_thread_id}")
        
        # Using as async context manager (recommended in docs)
        print("\n🧪 Using AsyncPostgresSaver as async context manager (recommended approach)")
        async with AsyncPostgresSaver.from_conn_string(connection_string) as checkpointer:
            print("✅ Successfully entered context manager")
            
            # Test basic checkpoint operations
            config = {"configurable": {"thread_id": unique_thread_id}}
            
            # Test getting a checkpoint
            checkpoint = await checkpointer.aget(config)
            print(f"✅ Retrieved checkpoint: {checkpoint}")
            
            # Try listing checkpoints
            print("🔍 Listing checkpoints (may be empty for new thread):")
            checkpoint_count = 0
            async for checkpoint in checkpointer.alist(config):
                checkpoint_count += 1
                if checkpoint_count == 1:
                    print(f"  ✓ Found checkpoint: {checkpoint}")
                
            if checkpoint_count == 0:
                print("  ✓ No checkpoints found (expected for new thread)")
        
        print("✅ Successfully exited context manager")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 TESTING LANGGRAPH ASYNCPOSTGRESSAVER DIRECT USAGE")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ['user', 'password', 'host', 'port', 'dbname']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        print("Please set these in your .env file.")
        sys.exit(1)
    
    # Run the test
    success = asyncio.run(test_direct_checkpointer_usage())
    
    if success:
        print("\n✅ Direct AsyncPostgresSaver usage tests PASSED")
        sys.exit(0)
    else:
        print("\n❌ Direct AsyncPostgresSaver usage tests FAILED")
        sys.exit(1) 