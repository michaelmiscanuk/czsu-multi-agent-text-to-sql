"""
Test for Phase 4: Extract Models (Request and Response Models)
Tests ONLY the Pydantic models in api/models/requests.py and api/models/responses.py
No API endpoint testing - just pure model validation.
Following test patterns from test_phase8_catalog.py and using helpers.py functions.
"""

from typing import Dict, Any
from datetime import datetime
import time
import sys
import uuid
import os

from pathlib import Path

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

from tests.helpers import (
    BaseTestResults,
    save_traceback_report,
)

# Import Pydantic models for validation testing
from pydantic import ValidationError
from api.models.requests import AnalyzeRequest, FeedbackRequest, SentimentRequest
from api.models.responses import (
    ChatMessage,
    ChatThreadResponse,
    PaginatedChatThreadsResponse,
)

# Test configuration - following Phase 8 pattern
REQUIRED_MODELS = {
    "AnalyzeRequest",
    "FeedbackRequest",
    "SentimentRequest",
    "ChatMessage",
    "ChatThreadResponse",
    "PaginatedChatThreadsResponse",
}

# Model validation test cases - following Phase 8 TEST_QUERIES pattern
MODEL_VALIDATION_TESTS = [
    # AnalyzeRequest tests
    {
        "model": "AnalyzeRequest",
        "data": {"prompt": "What are the top sales?", "thread_id": "test-123"},
        "description": "Valid AnalyzeRequest",
        "should_succeed": True,
    },
    {
        "model": "AnalyzeRequest",
        "data": {"prompt": "", "thread_id": "test"},
        "description": "Empty prompt should fail",
        "should_succeed": False,
    },
    {
        "model": "AnalyzeRequest",
        "data": {"prompt": "   ", "thread_id": "test"},
        "description": "Whitespace-only prompt should fail",
        "should_succeed": False,
    },
    {
        "model": "AnalyzeRequest",
        "data": {"prompt": "Valid", "thread_id": ""},
        "description": "Empty thread_id should fail",
        "should_succeed": False,
    },
    {
        "model": "AnalyzeRequest",
        "data": {"prompt": "x" * 10001, "thread_id": "test"},
        "description": "Prompt too long (>10000 chars) should fail",
        "should_succeed": False,
    },
    {
        "model": "AnalyzeRequest",
        "data": {"prompt": "x" * 10000, "thread_id": "test"},
        "description": "Prompt exactly at 10000 char limit should succeed",
        "should_succeed": True,
    },
    # FeedbackRequest tests
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()), "feedback": 1, "comment": "Great!"},
        "description": "Complete feedback",
        "should_succeed": True,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()), "feedback": 0},
        "description": "Feedback only",
        "should_succeed": True,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()), "comment": "Just comment"},
        "description": "Comment only",
        "should_succeed": True,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4())},
        "description": "Run ID only (model level validation)",
        "should_succeed": True,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": "invalid-uuid", "feedback": 1},
        "description": "Invalid UUID format should fail",
        "should_succeed": False,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()), "feedback": 2},
        "description": "Feedback > 1 should fail",
        "should_succeed": False,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()), "feedback": -1},
        "description": "Feedback < 0 should fail",
        "should_succeed": False,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()).upper(), "feedback": 1},
        "description": "Uppercase UUID should work",
        "should_succeed": True,
    },
    {
        "model": "FeedbackRequest",
        "data": {"run_id": str(uuid.uuid4()), "comment": "", "feedback": 1},
        "description": "Empty comment conversion to None",
        "should_succeed": True,
    },
    # SentimentRequest tests
    {
        "model": "SentimentRequest",
        "data": {"run_id": str(uuid.uuid4()), "sentiment": True},
        "description": "Positive sentiment",
        "should_succeed": True,
    },
    {
        "model": "SentimentRequest",
        "data": {"run_id": str(uuid.uuid4()), "sentiment": False},
        "description": "Negative sentiment",
        "should_succeed": True,
    },
    {
        "model": "SentimentRequest",
        "data": {"run_id": str(uuid.uuid4()), "sentiment": None},
        "description": "Null sentiment",
        "should_succeed": True,
    },
    {
        "model": "SentimentRequest",
        "data": {"run_id": "not-a-uuid", "sentiment": True},
        "description": "Invalid UUID format should fail",
        "should_succeed": False,
    },
    # Response model tests
    {
        "model": "ChatMessage",
        "data": {
            "id": "msg-123",
            "threadId": "thread-456",
            "user": "test@example.com",
            "createdAt": int(time.time() * 1000),
        },
        "description": "Valid ChatMessage",
        "should_succeed": True,
    },
    {
        "model": "ChatThreadResponse",
        "data": {
            "thread_id": "test-thread-123",
            "latest_timestamp": datetime.now(),
            "run_count": 5,
            "title": "Test Thread",
            "full_prompt": "Test prompt",
        },
        "description": "Valid ChatThreadResponse",
        "should_succeed": True,
    },
    {
        "model": "PaginatedChatThreadsResponse",
        "data": {
            "threads": [],
            "total_count": 0,
            "page": 1,
            "limit": 10,
            "has_more": False,
        },
        "description": "Valid PaginatedChatThreadsResponse",
        "should_succeed": True,
    },
]


def _get_model_class(model_name: str):
    """Get the model class by name - following Phase 8 validation pattern."""
    model_classes = {
        "AnalyzeRequest": AnalyzeRequest,
        "FeedbackRequest": FeedbackRequest,
        "SentimentRequest": SentimentRequest,
        "ChatMessage": ChatMessage,
        "ChatThreadResponse": ChatThreadResponse,
        "PaginatedChatThreadsResponse": PaginatedChatThreadsResponse,
    }
    return model_classes.get(model_name)


def _validate_model_instance(model_name: str, instance: Any, test_data: Dict):
    """Validate model instance properties - following Phase 8 validation pattern."""
    if model_name == "AnalyzeRequest":
        assert (
            len(instance.prompt.strip()) > 0
        ), "Prompt should not be empty after validation"
        assert (
            len(instance.thread_id.strip()) > 0
        ), "Thread ID should not be empty after validation"

    elif model_name == "FeedbackRequest":
        # Validate UUID format
        uuid.UUID(instance.run_id)  # Will raise ValueError if invalid

        # Check empty comment conversion to None
        if "comment" in test_data and test_data["comment"] == "":
            assert instance.comment is None, "Empty comment should be converted to None"

        # Check feedback range
        if hasattr(instance, "feedback") and instance.feedback is not None:
            assert 0 <= instance.feedback <= 1, "Feedback must be between 0 and 1"

    elif model_name == "SentimentRequest":
        # Validate UUID format
        uuid.UUID(instance.run_id)  # Will raise ValueError if invalid

        # Sentiment can be True, False, or None
        assert instance.sentiment in [
            True,
            False,
            None,
        ], "Sentiment must be boolean or None"

    elif model_name == "ChatMessage":
        assert instance.id, "ChatMessage must have an id"
        assert instance.threadId, "ChatMessage must have a threadId"
        assert instance.user, "ChatMessage must have a user"
        assert isinstance(instance.createdAt, int), "createdAt must be an integer"

    elif model_name == "ChatThreadResponse":
        assert instance.thread_id, "ChatThreadResponse must have a thread_id"
        assert isinstance(
            instance.latest_timestamp, datetime
        ), "latest_timestamp must be datetime"
        assert isinstance(instance.run_count, int), "run_count must be an integer"
        assert instance.title, "ChatThreadResponse must have a title"
        assert instance.full_prompt, "ChatThreadResponse must have a full_prompt"

    elif model_name == "PaginatedChatThreadsResponse":
        assert isinstance(instance.threads, list), "threads must be a list"
        assert isinstance(instance.total_count, int), "total_count must be an integer"
        assert isinstance(instance.page, int), "page must be an integer"
        assert isinstance(instance.limit, int), "limit must be an integer"
        assert isinstance(instance.has_more, bool), "has_more must be a boolean"


def validate_model(
    test_id: str,
    model_name: str,
    test_data: Dict,
    description: str,
    should_succeed: bool,
    results: BaseTestResults,
):
    """Validate a single model - following Phase 8 make_catalog_request pattern."""
    start_time = time.time()

    try:
        model_class = _get_model_class(model_name)
        if not model_class:
            error_obj = Exception(f"Unknown model: {model_name}")
            results.add_error(test_id, model_name, description, error_obj, 0)
            return

        # Try to create the model instance
        instance = model_class(**test_data)

        validation_time = time.time() - start_time

        if should_succeed:
            # Validate the instance properties
            _validate_model_instance(model_name, instance, test_data)

            print(f"✅ Test {test_id}: {description}")
            results.add_result(
                test_id,
                model_name,
                description,
                {"model_valid": True, "instance_created": True},
                validation_time,
                200,  # Success status
            )
        else:
            print(f"❌ Test {test_id}: Expected failure but succeeded - {description}")
            error_obj = Exception(
                "Expected validation error but model creation succeeded"
            )
            results.add_error(
                test_id, model_name, description, error_obj, validation_time
            )

    except (ValidationError, ValueError, AssertionError) as e:
        validation_time = time.time() - start_time

        if should_succeed:
            print(
                f"❌ Test {test_id}: Expected success but failed - {description}: {str(e)}"
            )
            results.add_error(test_id, model_name, description, e, validation_time)
        else:
            print(f"✅ Test {test_id}: Expected failure - {description}: {str(e)}")
            results.add_result(
                test_id,
                model_name,
                description,
                {"model_valid": False, "validation_error": str(e)},
                validation_time,
                422,  # Validation error status
            )

    except Exception as e:
        validation_time = time.time() - start_time
        print(f"❌ Test {test_id}: Unexpected error - {description}: {str(e)}")
        results.add_error(test_id, model_name, description, e, validation_time)


def run_model_validation_tests() -> BaseTestResults:
    """Run all model validation tests - following Phase 8 run_catalog_tests pattern."""
    print("🧪 Testing Pydantic Model Validation...")
    print("=" * 60)

    results = BaseTestResults(required_endpoints=REQUIRED_MODELS)
    results.start_time = datetime.now()

    for i, test_case in enumerate(MODEL_VALIDATION_TESTS, 1):
        test_id = f"model_test_{i}"
        validate_model(
            test_id,
            test_case["model"],
            test_case["data"],
            test_case["description"],
            test_case["should_succeed"],
            results,
        )

    results.end_time = datetime.now()
    return results


def analyze_model_test_results(results: BaseTestResults):
    """Analyze and print model test results - following Phase 8 pattern."""
    print("\n📊 Model Validation Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Validation Time: {summary['average_response_time']:.3f}s")

    tested_models = set(r["endpoint"] for r in results.results)
    tested_models.update(e["endpoint"] for e in results.errors)

    if not tested_models.issuperset(REQUIRED_MODELS):
        missing_models = REQUIRED_MODELS - tested_models
        print(f"❌ Missing models: {', '.join(missing_models)}")

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Errors:")
        for error in results.errors:
            print(f"  {error['test_id']}: {error['error']}")

    # Save traceback information (following Phase 8 pattern)
    save_traceback_report(report_type="model_validation", test_results=results)

    return summary


def main():
    """Main test execution function - following Phase 8 main() pattern."""
    print("🚀 Phase 4 Models Test - Pure Model Validation")
    print("=" * 60)
    print(
        "Testing Pydantic models in api/models/requests.py and api/models/responses.py"
    )
    print("Following test patterns from test_phase8_catalog.py")
    print("=" * 60)

    try:
        # Run model validation tests
        results = run_model_validation_tests()

        # Analyze results
        summary = analyze_model_test_results(results)

        # Determine overall test success - following Phase 8 success criteria
        test_passed = (
            summary["total_requests"] > 0
            and summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
            and len(results.errors) == 0
        )

        # Print detailed results - following Phase 8 format
        print("\n" + "=" * 60)
        print("📋 PHASE 4 MODEL VALIDATION RESULTS")
        print("=" * 60)

        print(f"📊 Total Tests: {summary['total_requests']}")
        print(f"✅ Successful: {summary['successful_requests']}")
        print(f"❌ Failed: {summary['failed_requests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

        if summary["successful_requests"] > 0:
            print(f"⏱️ Average Validation Time: {summary['average_response_time']:.3f}s")

        tested_models = set(r["endpoint"] for r in results.results)
        tested_models.update(e["endpoint"] for e in results.errors)

        if tested_models.issuperset(REQUIRED_MODELS):
            print("✅ All required models tested")
        else:
            missing_models = REQUIRED_MODELS - tested_models
            print(f"❌ Missing models: {', '.join(missing_models)}")

        # Error analysis
        if len(results.errors) > 0:
            print(f"❌ Test failed: {len(results.errors)} validation errors")
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No model validations succeeded")
        elif summary["failed_requests"] > 0:
            print(f"❌ Test failed: {summary['failed_requests']} tests failed")

        print("\n" + "=" * 60)
        print(f"🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        print("=" * 60)

        # Summary of what was tested - following Phase 8 format
        print("\n📝 TESTED MODELS:")
        print("📦 Request Models:")
        print(
            "   • AnalyzeRequest - prompt validation, length limits, thread_id validation"
        )
        print(
            "   • FeedbackRequest - UUID validation, feedback range (0-1), optional fields"
        )
        print(
            "   • SentimentRequest - UUID validation, boolean/null sentiment handling"
        )
        print("📦 Response Models:")
        print("   • ChatMessage - field validation, optional fields")
        print("   • ChatThreadResponse - datetime handling, required fields")
        print("   • PaginatedChatThreadsResponse - list validation, pagination fields")
        print("🔧 Validation Features:")
        print("   • Field constraint testing (string lengths, numeric ranges)")
        print("   • UUID format validation (case sensitivity)")
        print("   • Empty field handling and conversion (empty string to None)")
        print("   • Boundary testing (exact limits)")
        print("   • Custom validator testing (whitespace validation)")

        print(f"\n📊 TOTAL VALIDATIONS: {len(MODEL_VALIDATION_TESTS)}")
        print("📊 Using helpers.py functions for result tracking and error reporting")

        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")

        # Save error report using helpers.py - following Phase 8 pattern
        test_context = {
            "Test Type": "Model Validation",
            "Total Model Tests": len(MODEL_VALIDATION_TESTS),
            "Required Models": list(REQUIRED_MODELS),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")

        # Final error handling using helpers.py
        test_context = {
            "Test Type": "Model Validation",
            "Total Model Tests": len(MODEL_VALIDATION_TESTS),
            "Required Models": list(REQUIRED_MODELS),
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)
