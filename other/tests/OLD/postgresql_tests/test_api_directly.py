#!/usr/bin/env python3
"""
Test script to directly test the message loading API functionality.
"""

import asyncio
import sys
# Updated imports to use new modular structure
from api.routes.messages import get_chat_messages
from my_agent.utils.postgres_checkpointer import get_healthy_checkpointer
from unittest.mock import MagicMock

class MockUser:
    """Mock user for testing."""
    def get(self, key):
        if key == "email":
            return "test@example.com"
        return None

async def test_direct_api():
    """Test the API function directly."""
    
    print("🧪 Testing message loading API directly...")
    
    try:
        # Test with a sample thread_id
        test_thread_id = "test-thread-123"
        mock_user = MockUser()
        
        print(f"🔍 Testing API with thread: {test_thread_id}")
        
        # Call the API function directly
        messages = await get_chat_messages(test_thread_id, mock_user)
        
        print(f"✅ API call successful - returned {len(messages)} messages")
        
        for msg in messages:
            print(f"  - {msg.user}: {msg.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct_api())
    sys.exit(0 if success else 1) 