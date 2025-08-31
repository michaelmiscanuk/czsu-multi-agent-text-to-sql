#!/usr/bin/env python3
"""
Test script to validate the new message loading functionality from LangGraph checkpoints.
"""

import asyncio
import sys
import uuid
import json
from checkpointer.postgres_checkpointer import get_postgres_checkpointer
from main import main as analysis_main


async def test_message_loading():
    """Test loading messages from a checkpoint."""

    print("🧪 Testing message loading from LangGraph checkpoints...")

    try:
        # Step 1: Create a test conversation
        test_thread_id = f"test-conversation-{uuid.uuid4().hex[:8]}"
        test_prompt = "What is the population of Prague?"

        print(f"🔍 Creating test conversation with thread: {test_thread_id}")
        print(f"📝 Test prompt: {test_prompt}")

        # Get checkpointer
        checkpointer = await get_postgres_checkpointer()
        print("✅ PostgreSQL checkpointer initialized")

        # Step 2: Run a complete analysis to create conversation data
        print("\n📊 Running analysis to create conversation...")
        result = await analysis_main(
            test_prompt, thread_id=test_thread_id, checkpointer=checkpointer
        )
        print(f"✅ Analysis completed successfully")
        print(f"📄 Result preview: {result['result'][:100]}...")

        # Step 3: Test retrieving the state using the same method as the API
        print(f"\n🔍 Testing state retrieval for thread: {test_thread_id}")
        config = {"configurable": {"thread_id": test_thread_id}}

        # Use the async method like the API does
        state_snapshot = await checkpointer.aget_tuple(config)

        if not state_snapshot or not state_snapshot.checkpoint:
            print(f"❌ No state found for thread {test_thread_id}")
            return False

        # Extract messages from the checkpoint's channel_values
        channel_values = state_snapshot.checkpoint.get("channel_values", {})
        messages_data = channel_values.get("messages", [])

        if not messages_data:
            print(f"⚠️ No messages found in state for thread {test_thread_id}")
            return False

        print(f"✅ Found {len(messages_data)} messages in checkpoint")

        # Step 4: Simulate the frontend message conversion
        print("\n🔄 Converting messages to frontend format...")
        chat_messages = []
        for i, msg in enumerate(messages_data):
            if hasattr(msg, "content") and hasattr(msg, "type"):
                is_user = msg.type == "human"

                chat_message = {
                    "id": f"{msg.type}-{i}-{test_thread_id}",
                    "threadId": test_thread_id,
                    "user": "test@example.com" if is_user else "AI",
                    "content": str(msg.content),
                    "isUser": is_user,
                    "createdAt": 1000 * (i + 1),
                    "error": None,
                    "meta": (
                        getattr(msg, "additional_kwargs", {})
                        if hasattr(msg, "additional_kwargs")
                        else None
                    ),
                    "queriesAndResults": None,
                    "isLoading": False,
                    "startedAt": None,
                    "isError": False,
                }
                chat_messages.append(chat_message)

        print(f"✅ Converted {len(chat_messages)} messages to frontend format")

        # Step 5: Display the messages
        print("\n📋 Message Summary:")
        for i, msg in enumerate(chat_messages):
            user_type = "👤 User" if msg["isUser"] else "🤖 AI"
            content_preview = (
                msg["content"][:80] + "..."
                if len(msg["content"]) > 80
                else msg["content"]
            )
            print(f"  {i+1}. {user_type}: {content_preview}")

        # Step 6: Verify we have both user and AI messages
        user_messages = [m for m in chat_messages if m["isUser"]]
        ai_messages = [m for m in chat_messages if not m["isUser"]]

        print(f"\n📊 Message Statistics:")
        print(f"  👤 User messages: {len(user_messages)}")
        print(f"  🤖 AI messages: {len(ai_messages)}")

        if len(user_messages) >= 1 and len(ai_messages) >= 1:
            print("✅ Message loading test PASSED - Found both user and AI messages")
            return True
        else:
            print("❌ Message loading test FAILED - Missing expected message types")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_message_loading())
    sys.exit(0 if success else 1)
