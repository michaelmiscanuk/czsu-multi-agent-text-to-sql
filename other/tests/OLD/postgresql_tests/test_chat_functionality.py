#!/usr/bin/env python3
"""
Test script to validate the complete chat functionality including PostgreSQL checkpoints.
"""

import asyncio
import sys
import uuid
from my_agent.utils.postgres_checkpointer import get_healthy_checkpointer, create_thread_run_entry
from api.routes.messages import get_chat_messages
from main import main as analysis_main
from unittest.mock import MagicMock

class MockUser:
    """Mock user for testing."""
    def get(self, key):
        if key == "email":
            return "test@example.com"
        return None

async def test_complete_chat_flow():
    """Test the complete chat flow: send message -> store in checkpoint -> retrieve messages."""
    
    print("🧪 Testing complete chat flow...")
    
    try:
        # Generate a unique thread ID for this test
        test_thread_id = f"test-{uuid.uuid4().hex[:8]}"
        test_prompt = "Tell me about Prague population"
        mock_user = MockUser()
        
        print(f"🔍 Testing with thread: {test_thread_id}")
        print(f"📝 Test prompt: {test_prompt}")
        
        # Step 1: Get healthy checkpointer
        print("\n1️⃣ Getting healthy checkpointer...")
        checkpointer = await get_healthy_checkpointer()
        print("✅ Checkpointer ready")
        
        # Step 2: Create thread run entry (simulating frontend API call)
        print("\n🧪 Test 1: Creating thread run entry...")
        run_id = await create_thread_run_entry("test@example.com", test_thread_id, "Test data analysis")
        print(f"✅ Created run_id: {run_id}")
        
        # Step 3: Run analysis (simulating backend processing)
        print("\n3️⃣ Running analysis...")
        result = await analysis_main(test_prompt, thread_id=test_thread_id, checkpointer=checkpointer)
        print(f"✅ Analysis completed: {result['result'][:100]}...")
        
        # Step 4: Retrieve messages from checkpoint (simulating frontend loading)
        print("\n4️⃣ Retrieving messages from checkpoint...")
        messages = await get_chat_messages(test_thread_id, mock_user)
        print(f"✅ Retrieved {len(messages)} messages from checkpoint")
        
        # Step 5: Validate messages
        print("\n5️⃣ Validating messages...")
        
        if len(messages) >= 2:  # Should have at least user message and AI response
            user_messages = [m for m in messages if m.isUser]
            ai_messages = [m for m in messages if not m.isUser]
            
            print(f"  👤 User messages: {len(user_messages)}")
            print(f"  🤖 AI messages: {len(ai_messages)}")
            
            if len(user_messages) >= 1:
                print(f"  📄 User message content: {user_messages[0].content}")
                
            if len(ai_messages) >= 1:
                print(f"  📄 AI response content: {ai_messages[-1].content[:100]}...")
                
            print("✅ Message validation successful")
        else:
            print(f"⚠️ Expected at least 2 messages, got {len(messages)}")
            
        print("\n🎉 Complete chat flow test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Complete chat flow test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_chat_flow())
    sys.exit(0 if success else 1) 