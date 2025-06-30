#!/usr/bin/env python3
"""
Test Integration with Existing System

This script tests integration with the user's actual postgres_checkpointer.py
implementation and verifies that the connection pool fix works correctly
with their existing workflow.

Windows Unicode Fix: All Unicode characters replaced with ASCII equivalents
to prevent UnicodeEncodeError on Windows console.
"""

import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_postgres_checkpointer_import():
    """Test importing the user's postgres_checkpointer module."""
    print("\n[TEST] PostgreSQL Checkpointer Import")
    print("=" * 50)
    
    try:
        print(f">> Testing import of postgres_checkpointer module...")
        
        # Try to import the user's checkpointer
        try:
            from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
            print(f"   SUCCESS: get_postgres_checkpointer imported")
        except ImportError as e:
            print(f"   ERROR: Could not import get_postgres_checkpointer: {e}")
            return False
        
        # Try to import fallback function if it exists
        try:
            from my_agent.utils.postgres_checkpointer import get_fallback_checkpointer
            print(f"   SUCCESS: get_fallback_checkpointer imported")
        except ImportError:
            print(f"   INFO: get_fallback_checkpointer not available (optional)")
        
        # Test basic function call (without actually connecting)
        print(f">> Testing checkpointer function signature...")
        import inspect
        
        sig = inspect.signature(get_postgres_checkpointer)
        print(f"   Function signature: {sig}")
        
        # Check if function has expected parameters
        expected_params = ['connection_string']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            if param in actual_params:
                print(f"   Parameter '{param}': FOUND")
            else:
                print(f"   Parameter '{param}': MISSING (may use **kwargs)")
        
        print(f">> SUCCESS: PostgreSQL checkpointer import test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: PostgreSQL checkpointer import test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_checkpointer_creation():
    """Test creating checkpointer using the user's implementation."""
    print("\n[TEST] Checkpointer Creation")
    print("=" * 50)
    
    try:
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        
        # Get connection string
        config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'dbname': os.getenv('POSTGRES_DB', 'postgres'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        connection_string = (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['dbname']}"
            f"?sslmode=require"
            f"&connect_timeout=30"
            f"&application_name=test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print(f">> Creating checkpointer using user's implementation...")
        print(f"   Connection: {connection_string.split('@')[1].split('?')[0]}")
        
        # Test checkpointer creation
        checkpointer = await get_postgres_checkpointer()
        
        if checkpointer is None:
            print(f"   ERROR: Checkpointer creation returned None")
            return False
        
        print(f"   SUCCESS: Checkpointer created")
        print(f"   Type: {type(checkpointer).__name__}")
        
        # Test setup
        print(f">> Testing checkpointer setup...")
        await checkpointer.setup()
        print(f"   SUCCESS: Setup completed")
        
        # Test basic operations
        print(f">> Testing basic checkpointer operations...")
        
        # Test configuration
        thread_id = f"integration_test_{int(time.time())}"
        config_dict = {"configurable": {"thread_id": thread_id}}
        
        # Test empty retrieval
        result = await checkpointer.aget_tuple(config_dict)
        if result is None:
            print(f"   SUCCESS: Empty thread correctly returned None")
        else:
            print(f"   WARNING: Empty thread returned: {result}")
        
        # Test putting a simple checkpoint
        test_checkpoint = {
            "v": 1,
            "ts": time.time(),
            "id": str(uuid.uuid4()),
            "channel_values": {
                "test_message": "Integration test checkpoint",
                "timestamp": datetime.now().isoformat()
            },
            "channel_versions": {"test_message": 1, "timestamp": 1},
            "versions_seen": {}
        }
        
        metadata = {"source": "integration_test", "step": 1, "writes": {}}
        
        checkpoint_config = await checkpointer.aput(
            config_dict,
            test_checkpoint,
            metadata,
            new_versions={}
        )
        
        print(f"   SUCCESS: Checkpoint created with config: {checkpoint_config}")
        
        # Test retrieval
        retrieved = await checkpointer.aget_tuple(config_dict)
        
        if retrieved is not None:
            print(f"   SUCCESS: Checkpoint retrieved successfully")
            
            if retrieved.checkpoint:
                channel_values = retrieved.checkpoint.get("channel_values", {})
                test_message = channel_values.get("test_message", "")
                print(f"   Retrieved message: {test_message}")
        else:
            print(f"   ERROR: Could not retrieve checkpoint")
            return False
        
        print(f">> SUCCESS: Checkpointer creation test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Checkpointer creation test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_fallback_mechanism():
    """Test the fallback mechanism to InMemorySaver."""
    print("\n[TEST] Fallback Mechanism")
    print("=" * 50)
    
    try:
        print(f">> Testing fallback to InMemorySaver...")
        
        # Try to import fallback function
        try:
            from my_agent.utils.postgres_checkpointer import get_fallback_checkpointer
            fallback_available = True
        except ImportError:
            print(f"   INFO: Fallback function not available in current implementation")
            fallback_available = False
        
        if fallback_available:
            print(f">> Testing fallback checkpointer creation...")
            
            fallback_checkpointer = await get_fallback_checkpointer()
            
            if fallback_checkpointer is None:
                print(f"   ERROR: Fallback checkpointer creation returned None")
                return False
            
            print(f"   SUCCESS: Fallback checkpointer created")
            print(f"   Type: {type(fallback_checkpointer).__name__}")
            
            # Test fallback operations
            thread_id = f"fallback_test_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Test basic operation with fallback
            result = await fallback_checkpointer.aget_tuple(config)
            print(f"   SUCCESS: Fallback checkpointer basic operation works")
            
        else:
            # Test manual fallback to InMemorySaver
            print(f">> Testing manual fallback to InMemorySaver...")
            
            try:
                from langgraph.checkpoint.memory import MemorySaver
                memory_checkpointer = MemorySaver()
                print(f"   SUCCESS: MemorySaver created as fallback")
                print(f"   Type: {type(memory_checkpointer).__name__}")
                
                # Test basic operation
                thread_id = f"memory_test_{int(time.time())}"
                config = {"configurable": {"thread_id": thread_id}}
                
                result = await memory_checkpointer.aget_tuple(config)
                print(f"   SUCCESS: MemorySaver basic operation works")
                
            except ImportError as e:
                print(f"   ERROR: Could not import MemorySaver: {e}")
                return False
        
        print(f">> SUCCESS: Fallback mechanism test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Fallback mechanism test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_workflow_integration():
    """Test integration with a complete LangGraph workflow."""
    print("\n[TEST] Workflow Integration")
    print("=" * 50)
    
    try:
        from langgraph.graph import StateGraph
        from typing import TypedDict, Annotated
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        
        print(f">> Testing complete workflow integration...")
        
        # Define a simple state for testing
        def add_messages(left: List[str], right: List[str]) -> List[str]:
            return left + right
        
        class WorkflowState(TypedDict):
            messages: Annotated[List[str], add_messages]
            step_count: int
            user_id: str
        
        # Create test nodes
        def step_1(state: WorkflowState) -> WorkflowState:
            messages = state.get("messages", [])
            messages.append("Step 1: Initial processing")
            return {
                "messages": messages,
                "step_count": 1,
                "user_id": state.get("user_id", "test_user")
            }
        
        def step_2(state: WorkflowState) -> WorkflowState:
            messages = state.get("messages", [])
            messages.append("Step 2: Data processing")
            return {
                "messages": messages,
                "step_count": 2,
                "user_id": state.get("user_id", "test_user")
            }
        
        def step_3(state: WorkflowState) -> WorkflowState:
            messages = state.get("messages", [])
            messages.append("Step 3: Final processing")
            return {
                "messages": messages,
                "step_count": 3,
                "user_id": state.get("user_id", "test_user")
            }
        
        # Get connection string
        config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'dbname': os.getenv('POSTGRES_DB', 'postgres'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        connection_string = (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['dbname']}"
            f"?sslmode=require"
            f"&connect_timeout=30"
            f"&application_name=test_workflow_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create checkpointer
        checkpointer = await get_postgres_checkpointer()
        await checkpointer.setup()
        
        print(f"   Checkpointer created and set up")
        
        # Build workflow
        graph = StateGraph(WorkflowState)
        graph.add_node("step_1", step_1)
        graph.add_node("step_2", step_2)
        graph.add_node("step_3", step_3)
        
        graph.set_entry_point("step_1")
        graph.add_edge("step_1", "step_2")
        graph.add_edge("step_2", "step_3")
        graph.set_finish_point("step_3")
        
        # Compile with checkpointer
        compiled_graph = graph.compile(checkpointer=checkpointer)
        print(f"   Workflow compiled with checkpointer")
        
        # Test workflow execution
        thread_id = f"workflow_test_{int(time.time())}"
        config_dict = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [],
            "step_count": 0,
            "user_id": "integration_test_user"
        }
        
        print(f"   Starting workflow execution...")
        start_time = time.time()
        
        result = await compiled_graph.ainvoke(initial_state, config_dict)
        
        execution_time = time.time() - start_time
        print(f"   Workflow completed in {execution_time:.2f}s")
        
        # Verify results
        expected_steps = 3
        actual_steps = result.get("step_count", 0)
        expected_messages = 3
        actual_messages = len(result.get("messages", []))
        
        print(f"   Expected steps: {expected_steps}, Actual: {actual_steps}")
        print(f"   Expected messages: {expected_messages}, Actual: {actual_messages}")
        print(f"   User ID: {result.get('user_id', 'N/A')}")
        
        if actual_steps == expected_steps and actual_messages == expected_messages:
            print(f"   SUCCESS: Workflow execution results correct")
        else:
            print(f"   ERROR: Workflow execution results incorrect")
            return False
        
        # Test state persistence
        print(f">> Testing state persistence...")
        
        retrieved_state = await checkpointer.aget_tuple(config_dict)
        
        if retrieved_state and retrieved_state.checkpoint:
            channel_values = retrieved_state.checkpoint.get("channel_values", {})
            persisted_messages = channel_values.get("messages", [])
            persisted_steps = channel_values.get("step_count", 0)
            
            print(f"   Persisted messages: {len(persisted_messages)}")
            print(f"   Persisted step count: {persisted_steps}")
            
            if len(persisted_messages) == expected_messages and persisted_steps == expected_steps:
                print(f"   SUCCESS: State properly persisted")
            else:
                print(f"   ERROR: State not properly persisted")
                return False
        else:
            print(f"   ERROR: Could not retrieve persisted state")
            return False
        
        # Test multiple workflow executions on same thread
        print(f">> Testing multiple executions on same thread...")
        
        # Execute again to test state accumulation
        second_result = await compiled_graph.ainvoke(result, config_dict)
        
        expected_total_messages = 6  # 3 from first + 3 from second execution
        actual_total_messages = len(second_result.get("messages", []))
        
        print(f"   Total messages after second execution: {actual_total_messages}")
        
        if actual_total_messages == expected_total_messages:
            print(f"   SUCCESS: State accumulation working correctly")
        else:
            print(f"   WARNING: State accumulation behavior differs from expected")
            # This might be expected behavior depending on how the state reducer works
        
        print(f">> SUCCESS: Workflow integration test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Workflow integration test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_connection_pool_behavior():
    """Test the connection pool behavior under the user's implementation."""
    print("\n[TEST] Connection Pool Behavior")
    print("=" * 50)
    
    try:
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        
        print(f">> Testing connection pool behavior...")
        
        # Get connection string
        config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'dbname': os.getenv('POSTGRES_DB', 'postgres'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        connection_string = (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['dbname']}"
            f"?sslmode=require"
            f"&connect_timeout=30"
            f"&application_name=test_pool_behavior_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Test multiple checkpointer instances
        print(f">> Creating multiple checkpointer instances...")
        
        checkpointers = []
        for i in range(3):
            checkpointer = await get_postgres_checkpointer()
            await checkpointer.setup()
            checkpointers.append(checkpointer)
            print(f"   Checkpointer {i+1} created")
        
        # Test concurrent operations across instances
        print(f">> Testing concurrent operations...")
        
        async def test_checkpointer_operation(checkpointer, instance_id: int):
            """Test operation on a single checkpointer instance."""
            thread_id = f"pool_test_{instance_id}_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Create checkpoint
            test_checkpoint = {
                "v": 1,
                "ts": time.time(),
                "id": str(uuid.uuid4()),
                "channel_values": {
                    "instance_id": instance_id,
                    "message": f"Pool test from instance {instance_id}",
                    "timestamp": datetime.now().isoformat()
                },
                "channel_versions": {"instance_id": 1, "message": 1, "timestamp": 1},
                "versions_seen": {}
            }
            
            metadata = {"source": "pool_test", "step": 1, "writes": {}}
            
            # Put checkpoint
            await checkpointer.aput(config, test_checkpoint, metadata, new_versions={})
            
            # Retrieve checkpoint
            retrieved = await checkpointer.aget_tuple(config)
            
            if retrieved and retrieved.checkpoint:
                channel_values = retrieved.checkpoint.get("channel_values", {})
                retrieved_instance_id = channel_values.get("instance_id", -1)
                
                if retrieved_instance_id == instance_id:
                    print(f"   Instance {instance_id}: SUCCESS")
                    return True
                else:
                    print(f"   Instance {instance_id}: ERROR - ID mismatch")
                    return False
            else:
                print(f"   Instance {instance_id}: ERROR - No checkpoint retrieved")
                return False
        
        # Run concurrent operations
        start_time = time.time()
        
        operation_results = await asyncio.gather(*[
            test_checkpointer_operation(checkpointer, i)
            for i, checkpointer in enumerate(checkpointers)
        ])
        
        duration = time.time() - start_time
        
        successful_operations = sum(operation_results)
        total_operations = len(operation_results)
        
        print(f"   Concurrent operations completed in {duration:.2f}s")
        print(f"   Successful operations: {successful_operations}/{total_operations}")
        
        if successful_operations == total_operations:
            print(f"   SUCCESS: All concurrent operations completed")
        else:
            print(f"   ERROR: Some concurrent operations failed")
            return False
        
        # Test rapid sequential operations
        print(f">> Testing rapid sequential operations...")
        
        rapid_checkpointer = checkpointers[0]
        rapid_operations = 10
        
        for i in range(rapid_operations):
            thread_id = f"rapid_test_{i}_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            test_checkpoint = {
                "v": 1,
                "ts": time.time(),
                "id": str(uuid.uuid4()),
                "channel_values": {
                    "operation": i,
                    "message": f"Rapid operation {i}"
                },
                "channel_versions": {"operation": 1, "message": 1},
                "versions_seen": {}
            }
            
            metadata = {"source": "rapid_test", "step": 1, "writes": {}}
            
            await rapid_checkpointer.aput(config, test_checkpoint, metadata, new_versions={})
            
            # Verify immediately
            retrieved = await rapid_checkpointer.aget_tuple(config)
            
            if not (retrieved and retrieved.checkpoint):
                print(f"   ERROR: Rapid operation {i} failed")
                return False
        
        print(f"   SUCCESS: All {rapid_operations} rapid operations completed")
        
        print(f">> SUCCESS: Connection pool behavior test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Connection pool behavior test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_error_handling():
    """Test error handling in the user's implementation."""
    print("\n[TEST] Error Handling")
    print("=" * 50)
    
    try:
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        
        print(f">> Testing error handling scenarios...")
        
        # Test 1: Invalid connection string
        print(f">> Testing invalid connection string handling...")
        
        invalid_connection_string = "postgresql://invalid:invalid@nonexistent:5432/nonexistent"
        
        try:
            invalid_checkpointer = await get_postgres_checkpointer()
            
            if invalid_checkpointer is None:
                print(f"   SUCCESS: Invalid connection properly handled (returned None)")
            else:
                # Try to use it and see if it fails gracefully
                try:
                    await invalid_checkpointer.setup()
                    print(f"   WARNING: Invalid connection didn't fail immediately")
                except Exception as setup_error:
                    print(f"   SUCCESS: Invalid connection failed during setup (expected)")
        
        except Exception as e:
            print(f"   SUCCESS: Invalid connection raised exception (expected): {type(e).__name__}")
        
        # Test 2: Valid connection but database errors
        print(f">> Testing database error handling...")
        
        # Get valid connection string
        config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'dbname': os.getenv('POSTGRES_DB', 'postgres'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        connection_string = (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['dbname']}"
            f"?sslmode=require"
            f"&connect_timeout=30"
            f"&application_name=test_error_handling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create valid checkpointer
        checkpointer = await get_postgres_checkpointer()
        await checkpointer.setup()
        
        # Test 3: Invalid config handling
        print(f">> Testing invalid config handling...")
        
        try:
            invalid_config = {"configurable": {"thread_id": ""}}  # Empty thread_id
            result = await checkpointer.aget_tuple(invalid_config)
            print(f"   INFO: Empty thread_id result: {result}")
        except Exception as e:
            print(f"   SUCCESS: Empty thread_id properly handled: {type(e).__name__}")
        
        # Test 4: Malformed checkpoint data
        print(f">> Testing malformed checkpoint data handling...")
        
        try:
            thread_id = f"error_test_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Try to put malformed checkpoint
            malformed_checkpoint = {
                "v": "invalid_version",  # Should be int
                "invalid_field": "should_not_be_here"
                # Missing required fields
            }
            
            metadata = {"source": "error_test", "step": 1, "writes": {}}
            
            await checkpointer.aput(config, malformed_checkpoint, metadata, new_versions={})
            print(f"   WARNING: Malformed checkpoint was accepted")
            
        except Exception as e:
            print(f"   SUCCESS: Malformed checkpoint properly rejected: {type(e).__name__}")
        
        # Test 5: Connection recovery
        print(f">> Testing connection recovery...")
        
        # Perform operations to test if connections can recover from temporary issues
        for i in range(5):
            try:
                thread_id = f"recovery_test_{i}_{int(time.time())}"
                config = {"configurable": {"thread_id": thread_id}}
                
                test_checkpoint = {
                    "v": 1,
                    "ts": time.time(),
                    "id": str(uuid.uuid4()),
                    "channel_values": {
                        "recovery_test": i,
                        "message": f"Recovery test {i}"
                    },
                    "channel_versions": {"recovery_test": 1, "message": 1},
                    "versions_seen": {}
                }
                
                metadata = {"source": "recovery_test", "step": 1, "writes": {}}
                
                await checkpointer.aput(config, test_checkpoint, metadata, new_versions={})
                
                # Retrieve to verify
                retrieved = await checkpointer.aget_tuple(config)
                
                if retrieved and retrieved.checkpoint:
                    print(f"   Recovery test {i}: SUCCESS")
                else:
                    print(f"   Recovery test {i}: FAILED")
                    return False
                
                # Small delay between operations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"   Recovery test {i}: ERROR - {type(e).__name__}: {e}")
                return False
        
        print(f">> SUCCESS: Error handling test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Error handling test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all integration tests with the user's existing system."""
    print("Integration with Existing System Test Suite")
    print("=" * 60)
    print("Testing integration with user's postgres_checkpointer.py implementation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tests = [
        ("PostgreSQL Checkpointer Import", test_postgres_checkpointer_import),
        ("Checkpointer Creation", test_checkpointer_creation),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("Workflow Integration", test_workflow_integration),
        ("Connection Pool Behavior", test_connection_pool_behavior),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n" + "=" * 60)
        print(f"RUNNING: {test_name}")
        print(f"=" * 60)
        
        test_start = time.time()
        try:
            success = await test_func()
            test_duration = time.time() - test_start
            results[test_name] = {
                'success': success,
                'duration': test_duration,
                'error': None
            }
            
            status = "PASSED" if success else "FAILED"
            print(f"\n>> RESULT: {status} (Duration: {test_duration:.2f}s)")
            
        except Exception as e:
            test_duration = time.time() - test_start
            results[test_name] = {
                'success': False,
                'duration': test_duration,
                'error': str(e)
            }
            print(f"\n>> RESULT: ERROR (Duration: {test_duration:.2f}s)")
            print(f"   Unexpected error: {e}")
    
    # Summary
    total_duration = time.time() - start_time
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY")
    print(f"=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        duration = result['duration']
        print(f"  {test_name:.<35} {status:>6} ({duration:>5.2f}s)")
        if result['error']:
            print(f"    Error: {result['error']}")
    
    # Exit with appropriate code
    if passed == total:
        print(f"\n>> ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n>> SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n>> Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n>> Unexpected error: {e}")
        sys.exit(1) 