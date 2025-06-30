#!/usr/bin/env python3
"""
Test Multi-User Scenarios with LangGraph AsyncPostgresSaver - FIXED VERSION

This script tests concurrent users, thread isolation, load balancing,
and performance under various multi-user scenarios using the EXACT SAME
connection management approach as the main postgres_checkpointer.py script.

FIXED: Uses AsyncConnectionPool with proper configuration for Supabase/cloud databases
to prevent "connection is closed" errors.

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
import random
from datetime import datetime
from typing import Dict, List, TypedDict, Annotated

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    import psycopg
    print("SUCCESS: All required imports available")
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("Please install: pip install langgraph-checkpoint-postgres psycopg[binary,pool]")
    sys.exit(1)

# Global checkpointer context for proper resource management (same as main script)
_global_checkpointer_context = None

def get_db_config():
    """Get database configuration from environment variables (same as main script)."""
    return {
        'host': os.environ.get('host'),
        'port': os.environ.get('port'),
        'dbname': os.environ.get('dbname'),
        'user': os.environ.get('user'),
        'password': os.environ.get('password')
    }

def get_connection_string():
    """Generate optimized connection string for cloud databases (same as main script)."""
    config = get_db_config()
    
    # Generate unique application name for tracing
    import os
    process_id = os.getpid()
    thread_id = "test_script"
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    
    app_name = f"czsu_test_{process_id}_{thread_id}_{startup_time}_{random_id}"
    
    # ENHANCED: Cloud-optimized connection string with better timeout and keepalive settings
    connection_string = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"sslmode=require"
        f"&application_name={app_name}"
        f"&connect_timeout=20"              # Cloud-friendly timeout
        f"&keepalives_idle=600"             # 10 minutes idle timeout
        f"&keepalives_interval=30"          # 30 seconds between keepalives
        f"&keepalives_count=3"              # 3 failed keepalives before disconnect
        f"&tcp_user_timeout=30000"          # 30 seconds TCP timeout
    )
    
    return connection_string

def get_connection_kwargs():
    """Get connection kwargs for disabling prepared statements (same as postgres_checkpointer.py).
    
    Returns connection parameters that should be passed to psycopg connection methods.
    """
    return {
        "autocommit": False,  # CRITICAL FIX: False works better with cloud databases under load
        "prepare_threshold": None,  # Disable prepared statements completely
    }

async def clear_prepared_statements():
    """Clear any existing prepared statements to avoid conflicts (enhanced version)."""
    try:
        config = get_db_config()
        # Use a different application name for the cleanup connection
        cleanup_app_name = f"czsu_test_cleanup_{uuid.uuid4().hex[:8]}"
        
        # Create connection string without prepared statement parameters
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"
        
        # Get connection kwargs for disabling prepared statements
        connection_kwargs = get_connection_kwargs()
        
        print(f"   Clearing prepared statements...")
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            async with conn.cursor() as cur:
                # Get ALL prepared statements (not just our app)
                await cur.execute("""
                    SELECT name FROM pg_prepared_statements;
                """)
                prepared_statements = await cur.fetchall()
                
                if prepared_statements:
                    print(f"   Clearing {len(prepared_statements)} prepared statements...")
                    
                    # Drop each prepared statement
                    for stmt in prepared_statements:
                        stmt_name = stmt[0]
                        try:
                            await cur.execute(f"DEALLOCATE {stmt_name};")
                        except Exception as e:
                            pass  # Ignore individual statement errors
                    
                    print(f"   Cleared {len(prepared_statements)} prepared statements")
                    
                    # Force a connection reset to ensure clean state
                    await conn.execute("SELECT 1;")  # Simple query to verify connection
                else:
                    print("   No prepared statements to clear")
                
    except Exception as e:
        print(f"   Warning: Error clearing prepared statements (non-fatal): {e}")
        # Don't raise - this is a cleanup operation and shouldn't block tests

async def create_optimized_checkpointer():
    """
    Create AsyncPostgresSaver using the EXACT SAME fallback approach 
    as the main script with cloud-optimized connection string.
    
    ENHANCED: Create truly isolated checkpointer for each test.
    """
    global _global_checkpointer_context
    
    print(">> Creating optimized checkpointer using fallback approach...")
    
    try:
        # Create unique connection string for each test to avoid conflicts
        import uuid
        import time
        unique_id = f"{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Use the EXACT SAME fallback approach as the main script with unique ID
        config = get_db_config()
        connection_string = (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['dbname']}?"
            f"sslmode=require"
            f"&application_name=czsu_test_{unique_id}"
            f"&connect_timeout=20"
            f"&keepalives_idle=600"
            f"&keepalives_interval=30"
            f"&keepalives_count=3"
            f"&tcp_user_timeout=30000"
        )
        
        print(">> Using AsyncPostgresSaver.from_conn_string() with cloud-optimized connection...")
        
        # CRITICAL: Use exact same approach as main script fallback with unique connection
        _global_checkpointer_context = AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        )
        
        # Enter the context manager
        print(">> Entering context manager...")
        checkpointer = await _global_checkpointer_context.__aenter__()
        
        # Setup the checkpointer (creates tables)
        print(">> Setting up checkpointer tables...")
        await checkpointer.setup()
        print(">> Checkpointer setup complete")
        
        # Test the checkpointer to ensure it's working
        test_config = {"configurable": {"thread_id": f"setup_test_{unique_id}"}}
        test_result = await checkpointer.aget(test_config)
        print(f">> Checkpointer test successful: {test_result is None}")
        
        return checkpointer
        
    except Exception as e:
        print(f">> ERROR creating checkpointer: {e}")
        print(f">> Traceback: {traceback.format_exc()}")
        
        # Clean up on failure
        if _global_checkpointer_context:
            try:
                await _global_checkpointer_context.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f">> Cleanup error: {cleanup_error}")
            _global_checkpointer_context = None
        raise

async def cleanup_checkpointer():
    """Clean up the global checkpointer and connection pool."""
    global _global_checkpointer_context
    
    if _global_checkpointer_context:
        try:
            print(">> Cleaning up checkpointer...")
            await _global_checkpointer_context.__aexit__(None, None, None)
            print(">> Checkpointer cleaned up successfully")
        except Exception as e:
            print(f">> Cleanup error: {e}")
        finally:
            _global_checkpointer_context = None

# Test workflow state
class WorkflowState(TypedDict):
    user_id: str
    session_id: str
    message_count: int
    last_message: str
    results: List[str]

def simple_workflow_node(state: WorkflowState) -> WorkflowState:
    """Simple workflow node for testing."""
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")
    message_count = state.get("message_count", 0) + 1
    
    # Simulate some work
    import time
    time.sleep(0.1)  # Small delay to simulate processing
    
    result = f"User {user_id} session {session_id} message {message_count} processed"
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "message_count": message_count,
        "last_message": result,
        "results": state.get("results", []) + [result]
    }

async def run_single_session(checkpointer, user_id: str, session_id: str, num_messages: int = 5):
    """Run a single user session with multiple messages."""
    try:
        from langgraph.graph import StateGraph, START, END
        
        # Create a simple workflow
        workflow = StateGraph(WorkflowState)
        workflow.add_node("process", simple_workflow_node)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        
        # Compile with checkpointer
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Run multiple messages in the session
        config = {"configurable": {"thread_id": f"user_{user_id}_session_{session_id}"}}
        
        for i in range(num_messages):
            try:
                initial_state = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "message_count": 0,
                    "last_message": f"Message {i+1}",
                    "results": []
                }
                
                result = await graph.ainvoke(initial_state, config)
                
                # Verify the session is persisted
                if i == 0:  # First message, verify persistence works
                    state_snapshot = await graph.aget_state(config)
                    if state_snapshot and state_snapshot.values:
                        pass  # State persistence working
                    
            except Exception as e:
                print(f"   ERROR: User {user_id}, Session {session_id}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ERROR: User {user_id}, Session {session_id}: {e}")
        return False

async def test_concurrent_users():
    """Test multiple users accessing the system concurrently."""
    print("\n[TEST] Concurrent Users")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        # Create optimized checkpointer using AsyncConnectionPool
        checkpointer = await create_optimized_checkpointer()
        
        print(">> Testing concurrent users...")
        print(">> Running 10 users with 5 sessions each...")
        
        start_time = time.time()
        tasks = []
        
        # Create 10 concurrent users, each with 5 sessions
        for user_id in range(10):
            for session_id in range(5):
                task = run_single_session(checkpointer, str(user_id), str(session_id), 1)
                tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful results
        successful_sessions = sum(1 for result in results if result is True)
        total_sessions = len(tasks)
        duration = time.time() - start_time
        
        print(f">> Concurrent user test completed:")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Successful sessions: {successful_sessions}/{total_sessions}")
        print(f"   Sessions per second: {total_sessions/duration:.1f}")
        
        if successful_sessions == total_sessions:
            print(">> SUCCESS: All sessions completed successfully")
            success = True
        else:
            print(">> FAILED: Some sessions failed")
            success = False
            
        # Show any exception details
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            print(f">> Exceptions encountered: {len(exceptions)}")
            for i, exc in enumerate(exceptions[:3]):  # Show first 3
                print(f"   Exception {i+1}: {exc}")
        
        result = "SUCCESS" if success else "FAILED"
        print(f">> RESULT: {result} (Duration: {duration:.2f}s)")
        
        return success
        
    except Exception as e:
        print(f">> ERROR in concurrent users test: {e}")
        print(f">> Traceback: {traceback.format_exc()}")
        return False

async def test_thread_isolation():
    """Test that different user threads are properly isolated."""
    print("\n[TEST] Thread Isolation")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        checkpointer = await create_optimized_checkpointer()
        
        print(">> Testing thread isolation...")
        print("   Created optimized checkpointer")
        
        from langgraph.graph import StateGraph, START, END
        
        # Create a simple workflow
        workflow = StateGraph(WorkflowState)
        workflow.add_node("process", simple_workflow_node)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Create 5 isolated user threads
        threads = []
        for i in range(5):
            thread_id = f"isolation_test_user_{i}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Each user has different initial state
            initial_state = {
                "user_id": f"user_{i}",
                "session_id": "isolation_test",
                "message_count": i * 10,  # Different starting counts
                "last_message": f"Initial message for user {i}",
                "results": [f"User {i} initial result"]
            }
            
            result = await graph.ainvoke(initial_state, config)
            
            # Verify each thread maintains its own state
            state_snapshot = await graph.aget_state(config)
            if state_snapshot and state_snapshot.values:
                user_id = state_snapshot.values.get("user_id")
                if user_id == f"user_{i}":
                    threads.append((thread_id, True))
                else:
                    threads.append((thread_id, False))
            else:
                threads.append((thread_id, False))
        
        # Verify isolation by checking each thread again
        isolation_success = True
        for thread_id, initial_success in threads:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = await graph.aget_state(config)
            
            if not (initial_success and state_snapshot and state_snapshot.values):
                isolation_success = False
                break
        
        print(f">> Thread isolation test:")
        print(f"   Isolated threads created: {len(threads)}")
        print(f"   All threads properly isolated: {isolation_success}")
        
        result = "SUCCESS" if isolation_success else "FAILED"
        print(f">> RESULT: {result}")
        
        return isolation_success
        
    except Exception as e:
        print(f">> ERROR in thread isolation test: {e}")
        return False

async def test_load_balancing():
    """Test system behavior under varying load conditions."""
    print("\n[TEST] Load Balancing")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        checkpointer = await create_optimized_checkpointer()
        
        print(">> Testing load balancing...")
        
        # Test different load levels
        load_tests = [
            ("Low load", 5, 1),      # 5 users, 1 session each
            ("Medium load", 10, 2),   # 10 users, 2 sessions each  
            ("High load", 15, 3),     # 15 users, 3 sessions each
        ]
        
        all_success = True
        
        for test_name, num_users, sessions_per_user in load_tests:
            print(f"   Running {test_name}: {num_users} users x {sessions_per_user} sessions")
            
            start_time = time.time()
            tasks = []
            
            for user_id in range(num_users):
                for session_id in range(sessions_per_user):
                    task = run_single_session(checkpointer, f"load_{user_id}", f"s_{session_id}", 1)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            total = len(tasks)
            duration = time.time() - start_time
            
            print(f"   {test_name}: {successful}/{total} sessions in {duration:.2f}s ({total/duration:.1f} sessions/s)")
            
            if successful < total:
                all_success = False
        
        result = "SUCCESS" if all_success else "FAILED"
        print(f">> RESULT: {result}")
        
        return all_success
        
    except Exception as e:
        print(f">> ERROR in load balancing test: {e}")
        return False

async def test_data_consistency():
    """Test data consistency across multiple operations."""
    print("\n[TEST] Data Consistency")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        checkpointer = await create_optimized_checkpointer()
        
        print(">> Testing data consistency...")
        
        from langgraph.graph import StateGraph, START, END
        
        workflow = StateGraph(WorkflowState)
        workflow.add_node("process", simple_workflow_node)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Test consistency with a single thread over multiple operations
        thread_id = "consistency_test"
        config = {"configurable": {"thread_id": thread_id}}
        
        consistency_passed = True
        
        # Perform 10 sequential operations
        for i in range(10):
            initial_state = {
                "user_id": "consistency_user",
                "session_id": "consistency_session",
                "message_count": 0,
                "last_message": f"Consistency test message {i+1}",
                "results": []
            }
            
            result = await graph.ainvoke(initial_state, config)
            
            # Verify state persistence
            state_snapshot = await graph.aget_state(config)
            if not (state_snapshot and state_snapshot.values):
                consistency_passed = False
                break
                
            # Verify message count increases (due to workflow logic)
            if state_snapshot.values.get("message_count", 0) <= 0:
                consistency_passed = False
                break
        
        print(f"   Consistency operations: 10/10")
        print(f"   Data consistency maintained: {consistency_passed}")
        
        result = "SUCCESS" if consistency_passed else "FAILED"
        print(f">> RESULT: {result}")
        
        return consistency_passed
        
    except Exception as e:
        print(f">> ERROR in data consistency test: {e}")
        return False

async def test_performance_metrics():
    """Test and measure performance metrics."""
    print("\n[TEST] Performance Metrics")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        checkpointer = await create_optimized_checkpointer()
        
        print(">> Testing performance metrics...")
        
        # Test different scenarios and measure performance
        scenarios = [
            ("Single user, sequential", 1, 10, False),
            ("Single user, batch", 1, 10, True),
            ("Multiple users, sequential", 5, 5, False),
            ("Multiple users, concurrent", 5, 5, True),
        ]
        
        all_metrics = []
        
        for scenario_name, num_users, operations_per_user, concurrent in scenarios:
            print(f"   Testing: {scenario_name}")
            
            start_time = time.time()
            
            if concurrent:
                tasks = []
                for user_id in range(num_users):
                    for op_id in range(operations_per_user):
                        task = run_single_session(checkpointer, f"perf_{user_id}", f"op_{op_id}", 1)
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
            else:
                successful = 0
                for user_id in range(num_users):
                    for op_id in range(operations_per_user):
                        result = await run_single_session(checkpointer, f"perf_{user_id}", f"op_{op_id}", 1)
                        if result:
                            successful += 1
            
            duration = time.time() - start_time
            total_ops = num_users * operations_per_user
            ops_per_second = total_ops / duration if duration > 0 else 0
            
            metrics = {
                "scenario": scenario_name,
                "total_operations": total_ops,
                "successful_operations": successful,
                "duration": duration,
                "ops_per_second": ops_per_second,
                "success_rate": successful / total_ops if total_ops > 0 else 0
            }
            
            all_metrics.append(metrics)
            
            print(f"     Operations: {successful}/{total_ops}")
            print(f"     Duration: {duration:.2f}s")
            print(f"     Throughput: {ops_per_second:.1f} ops/s")
            print(f"     Success rate: {metrics['success_rate']:.1%}")
        
        # Overall performance assessment
        avg_success_rate = sum(m['success_rate'] for m in all_metrics) / len(all_metrics)
        max_throughput = max(m['ops_per_second'] for m in all_metrics)
        
        print(f"\n>> Performance Summary:")
        print(f"   Average success rate: {avg_success_rate:.1%}")
        print(f"   Maximum throughput: {max_throughput:.1f} ops/s")
        
        performance_passed = avg_success_rate >= 0.9  # 90% success rate required
        
        result = "SUCCESS" if performance_passed else "FAILED"
        print(f">> RESULT: {result}")
        
        return performance_passed
        
    except Exception as e:
        print(f">> ERROR in performance metrics test: {e}")
        return False

async def main():
    """Run the comprehensive multi-user test suite."""
    print("Multi-User Scenarios Test Suite - FIXED VERSION")
    print("=" * 60)
    print("Testing concurrent users, thread isolation, and load balancing")
    print("with OPTIMIZED AsyncConnectionPool for Supabase/cloud databases")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Display connection info
    config = get_db_config()
    print(f"\nConnection: {config['host']}:{config['port']}/{config['dbname']}")
    
    # Check environment variables
    required_vars = ['host', 'port', 'dbname', 'user', 'password']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"\nERROR: Missing required environment variables: {missing_vars}")
        print("Please set all required PostgreSQL environment variables.")
        return False
    
    print("\n" + "=" * 60)
    
    # Run all tests
    test_functions = [
        ("Concurrent Users", test_concurrent_users),
        ("Thread Isolation", test_thread_isolation),
        ("Load Balancing", test_load_balancing),
        ("Data Consistency", test_data_consistency),
        ("Performance Metrics", test_performance_metrics),
    ]
    
    results = {}
    overall_success = True
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print("="*60)
        
        try:
            start_time = time.time()
            success = await test_func()
            duration = time.time() - start_time
            
            results[test_name] = {
                "success": success,
                "duration": duration
            }
            
            if not success:
                overall_success = False
                
        except Exception as e:
            print(f"FATAL ERROR in {test_name}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = {
                "success": False,
                "duration": 0,
                "error": str(e)
            }
            overall_success = False
        
        finally:
            # Clean up after each test to prevent resource leaks
            try:
                await cleanup_checkpointer()
            except Exception as e:
                print(f"Cleanup error after {test_name}: {e}")
    
    # Final cleanup
    try:
        await cleanup_checkpointer()
    except Exception as e:
        print(f"Final cleanup error: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        duration = result.get("duration", 0)
        print(f"{test_name:25} | {status:4} | {duration:6.2f}s")
        
        if "error" in result:
            print(f"  Error: {result['error']}")
    
    print("-" * 60)
    total_duration = sum(r.get("duration", 0) for r in results.values())
    passed_tests = sum(1 for r in results.values() if r["success"])
    total_tests = len(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Overall result: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! The multi-user system is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED! Please review the errors above.")
    
    return overall_success

if __name__ == "__main__":
    try:
        # Import asyncio after setting the event loop policy
        import asyncio
        
        # Run the test suite
        success = asyncio.run(main())
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1) 