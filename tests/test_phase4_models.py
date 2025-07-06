#!/usr/bin/env python3
"""
Test for Phase 4: Extract Models (Request and Response Models)
Based on test_concurrency.py pattern - imports functionality from main scripts
"""

# CRITICAL: Set Windows event loop policy FIRST, before other imports
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Constants
try:
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import asyncio
import time
import httpx
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from pydantic import ValidationError

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import functionality from the new modular structure (not reimplementing!)
from api.models.requests import AnalyzeRequest, FeedbackRequest, SentimentRequest
from api.models.responses import ChatThreadResponse, PaginatedChatThreadsResponse, ChatMessage

def test_phase4_models():
    """Test Phase 4 models functionality by importing and validating all models."""
    
    print("🔍 Testing Phase 4: Models extraction...")
    print("=" * 60)
    
    # Test imports
    print("📦 Testing model imports...")
    
    # Verify request models are imported correctly
    assert AnalyzeRequest is not None, "AnalyzeRequest should be importable"
    assert FeedbackRequest is not None, "FeedbackRequest should be importable"
    assert SentimentRequest is not None, "SentimentRequest should be importable"
    print("✅ Request models imported successfully")
    
    # Verify response models are imported correctly
    assert ChatThreadResponse is not None, "ChatThreadResponse should be importable"
    assert PaginatedChatThreadsResponse is not None, "PaginatedChatThreadsResponse should be importable"
    assert ChatMessage is not None, "ChatMessage should be importable"
    print("✅ Response models imported successfully")
    
    # Test AnalyzeRequest validation
    print("\n🧪 Testing AnalyzeRequest validation...")
    
    # Valid request
    valid_analyze = AnalyzeRequest(
        prompt="Test prompt for analysis",
        thread_id="test-thread-123"
    )
    assert valid_analyze.prompt == "Test prompt for analysis"
    assert valid_analyze.thread_id == "test-thread-123"
    print("✅ Valid AnalyzeRequest created successfully")
    
    # Test prompt validation - expect ValidationError for empty prompt
    try:
        AnalyzeRequest(prompt="", thread_id="test-thread")
        assert False, "Empty prompt should raise ValidationError"
    except ValidationError as e:
        assert "string_too_short" in str(e) or "min_length" in str(e) or "empty" in str(e).lower()
        print("✅ Empty prompt validation works")
    except ValueError as e:
        assert "empty" in str(e).lower()
        print("✅ Empty prompt validation works (ValueError)")
    
    # Test whitespace-only prompt - should be trimmed and then fail
    try:
        AnalyzeRequest(prompt="   ", thread_id="test-thread")
        assert False, "Whitespace-only prompt should raise ValidationError"
    except (ValidationError, ValueError) as e:
        print("✅ Whitespace-only prompt validation works")
    
    # Test thread_id validation
    try:
        AnalyzeRequest(prompt="Valid prompt", thread_id="")
        assert False, "Empty thread_id should raise ValidationError"
    except (ValidationError, ValueError) as e:
        print("✅ Empty thread_id validation works")
    
    # Test FeedbackRequest validation
    print("\n🧪 Testing FeedbackRequest validation...")
    
    # Valid request with feedback
    test_uuid = str(uuid.uuid4())
    valid_feedback = FeedbackRequest(
        run_id=test_uuid,
        feedback=1,
        comment="Great response!"
    )
    assert valid_feedback.run_id == test_uuid
    assert valid_feedback.feedback == 1
    assert valid_feedback.comment == "Great response!"
    print("✅ Valid FeedbackRequest with feedback created successfully")
    
    # Valid request with comment only
    valid_feedback_comment_only = FeedbackRequest(
        run_id=test_uuid,
        comment="Just a comment"
    )
    assert valid_feedback_comment_only.feedback is None
    assert valid_feedback_comment_only.comment == "Just a comment"
    print("✅ Valid FeedbackRequest with comment only created successfully")
    
    # Test run_id UUID validation
    try:
        FeedbackRequest(run_id="invalid-uuid", feedback=1)
        assert False, "Invalid UUID should raise ValueError"
    except (ValidationError, ValueError) as e:
        assert "uuid" in str(e).lower()
        print("✅ Invalid UUID validation works")
    
    # Test feedback range validation (ge=0, le=1)
    try:
        FeedbackRequest(run_id=test_uuid, feedback=2)
        assert False, "Feedback > 1 should raise validation error"
    except (ValidationError, ValueError):
        print("✅ Feedback range validation works")
    
    try:
        FeedbackRequest(run_id=test_uuid, feedback=-1)
        assert False, "Feedback < 0 should raise validation error"
    except (ValidationError, ValueError):
        print("✅ Feedback negative validation works")
    
    # Test empty comment handling
    empty_comment_feedback = FeedbackRequest(
        run_id=test_uuid,
        feedback=1,
        comment=""
    )
    assert empty_comment_feedback.comment is None
    print("✅ Empty comment conversion to None works")
    
    # Test SentimentRequest validation
    print("\n🧪 Testing SentimentRequest validation...")
    
    # Valid request with positive sentiment
    valid_sentiment = SentimentRequest(
        run_id=test_uuid,
        sentiment=True
    )
    assert valid_sentiment.sentiment is True
    print("✅ Valid SentimentRequest with positive sentiment created successfully")
    
    # Valid request with negative sentiment
    valid_sentiment_neg = SentimentRequest(
        run_id=test_uuid,
        sentiment=False
    )
    assert valid_sentiment_neg.sentiment is False
    print("✅ Valid SentimentRequest with negative sentiment created successfully")
    
    # Valid request with null sentiment (to clear)
    valid_sentiment_null = SentimentRequest(
        run_id=test_uuid,
        sentiment=None
    )
    assert valid_sentiment_null.sentiment is None
    print("✅ Valid SentimentRequest with null sentiment created successfully")
    
    # Test run_id UUID validation for sentiment
    try:
        SentimentRequest(run_id="not-a-uuid", sentiment=True)
        assert False, "Invalid UUID should raise ValueError"
    except (ValidationError, ValueError) as e:
        assert "uuid" in str(e).lower()
        print("✅ SentimentRequest UUID validation works")
    
    # Test ChatThreadResponse
    print("\n🧪 Testing ChatThreadResponse...")
    
    thread_response = ChatThreadResponse(
        thread_id="test-thread-123",
        latest_timestamp=datetime.now(),
        run_count=5,
        title="Test Thread Title",
        full_prompt="This is the full prompt for testing"
    )
    assert thread_response.thread_id == "test-thread-123"
    assert isinstance(thread_response.latest_timestamp, datetime)
    assert thread_response.run_count == 5
    print("✅ ChatThreadResponse created successfully")
    
    # Test PaginatedChatThreadsResponse
    print("\n🧪 Testing PaginatedChatThreadsResponse...")
    
    paginated_response = PaginatedChatThreadsResponse(
        threads=[thread_response],
        total_count=10,
        page=1,
        limit=5,
        has_more=True
    )
    assert len(paginated_response.threads) == 1
    assert paginated_response.total_count == 10
    assert paginated_response.has_more is True
    print("✅ PaginatedChatThreadsResponse created successfully")
    
    # Test ChatMessage
    print("\n🧪 Testing ChatMessage...")
    
    chat_message = ChatMessage(
        id="msg-123",
        threadId="thread-456",
        user="test@example.com",
        content="Hello, this is a test message",
        isUser=True,
        createdAt=int(time.time() * 1000),
        error=None,
        meta={"source": "test"},
        queriesAndResults=[["SELECT * FROM test", "Result data"]],
        isLoading=False,
        startedAt=None,
        isError=False
    )
    assert chat_message.id == "msg-123"
    assert chat_message.isUser is True
    assert chat_message.meta["source"] == "test"
    assert len(chat_message.queriesAndResults) == 1
    print("✅ ChatMessage created successfully")
    
    # Test ChatMessage with minimal fields
    minimal_message = ChatMessage(
        id="msg-minimal",
        threadId="thread-minimal",
        user="user@test.com",
        content="Minimal message",
        isUser=False,
        createdAt=int(time.time() * 1000)
    )
    assert minimal_message.error is None
    assert minimal_message.meta is None
    assert minimal_message.isLoading is None
    print("✅ Minimal ChatMessage created successfully")
    
    print("\n" + "=" * 60)
    print("✅ All Phase 4 model tests passed successfully!")
    print("✅ Request models: AnalyzeRequest, FeedbackRequest, SentimentRequest")
    print("✅ Response models: ChatThreadResponse, PaginatedChatThreadsResponse, ChatMessage")
    print("✅ All validation rules working correctly")
    
    return True

async def main():
    """Main test runner."""
    print(f"Testing Phase 4 model extraction...")
    
    # Run model tests
    test_success = test_phase4_models()
    
    if test_success:
        print("✅ Phase 4 tests completed successfully")
        return True
    else:
        print("❌ Phase 4 tests failed")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 