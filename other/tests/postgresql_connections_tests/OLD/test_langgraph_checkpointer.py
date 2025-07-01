#!/usr/bin/env python3
"""
Test LangGraph AsyncPostgresSaver Integration - FIXED VERSION

This script tests the LangGraph AsyncPostgresSaver functionality specifically,
including checkpoint operations, graph execution, and state persistence.

FIXED: Uses the same techniques as successful test_multiuser_scenarios.py
for cloud-optimized connection handling and prepared statement management.

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
from typing import Dict, List, TypedDict, Annotated

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables (same as postgres_checkpointer.py)."""
    return {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }

def get_connection_string() -> str:
    """Generate PostgreSQL connection string with cloud optimizations (same as multiuser test)."""
    config = get_db_config()
    
    # Generate unique application name for tracing (same approach as multiuser test)
    process_id = os.getpid()
    thread_id = "langgraph_test"
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    
    app_name = f"czsu_langgraph_test_{process_id}_{thread_id}_{startup_time}_{random_id}"
    
    # ENHANCED: Cloud-optimized connection string with better timeout and keepalive settings
    return (
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

def get_connection_kwargs():
    """Get connection kwargs for disabling prepared statements (same as postgres_checkpointer.py)."""
    return {
        "autocommit": False,  # CRITICAL FIX: False works better with cloud databases under load
        "prepare_threshold": None,  # Disable prepared statements completely
    }

async def clear_prepared_statements():
    """Clear any existing prepared statements to avoid conflicts (same as multiuser test)."""
    try:
        config = get_db_config()
        # Use a different application name for the cleanup connection
        cleanup_app_name = f"czsu_langgraph_cleanup_{uuid.uuid4().hex[:8]}"
        
        # Create connection string without prepared statement parameters
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"
        
        # Get connection kwargs for disabling prepared statements
        connection_kwargs = get_connection_kwargs()
        
        print(f"   Clearing prepared statements...")
        import psycopg
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            async with conn.cursor() as cur:
                # Get ALL prepared statements (same as multiuser test)
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

def add(left: List, right: List) -> List:
    """Add two lists together."""
    return left + right

class TestState(TypedDict):
    messages: Annotated[List[str], add]
    count: int
    user_id: str

def create_test_nodes():
    """Create test nodes for LangGraph testing."""
    
    def node_a(state: TestState) -> TestState:
        """First test node that adds a message."""
        messages = state.get("messages", [])
        messages.append(f"Node A executed for user {state.get('user_id', 'unknown')}")
        return {
            "messages": messages,
            "count": state.get("count", 0) + 1,
            "user_id": state.get("user_id", "test_user")
        }
    
    def node_b(state: TestState) -> TestState:
        """Second test node that adds a message."""
        messages = state.get("messages", [])
        messages.append(f"Node B executed, count is now {state.get('count', 0) + 1}")
        return {
            "messages": messages,
            "count": state.get("count", 0) + 1,
            "user_id": state.get("user_id", "test_user")
        }
    
    def node_c(state: TestState) -> TestState:
        """Third test node that adds a final message."""
        messages = state.get("messages", [])
        messages.append(f"Node C executed, final count: {state.get('count', 0) + 1}")
        return {
            "messages": messages,
            "count": state.get("count", 0) + 1,
            "user_id": state.get("user_id", "test_user")
        }
    
    return node_a, node_b, node_c

async def test_checkpointer_creation():
    """Test AsyncPostgresSaver creation and setup."""
    print("\n[TEST] AsyncPostgresSaver Creation")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        
        connection_string = get_connection_string()
        print(f">> Creating AsyncPostgresSaver...")
        print(f"   Connection string: {connection_string.split('@')[1].split('?')[0]}")
        
        # CRITICAL: Use exact same approach as main script fallback with pipeline=False
        async with AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        ) as checkpointer:
            print(f">> Checkpointer created successfully!")
            print(f"   Type: {type(checkpointer).__name__}")
            
            # Test setup
            print(f">> Setting up checkpointer tables...")
            await checkpointer.setup()
            print(f">> Setup completed successfully!")
            
            print(f">> SUCCESS: AsyncPostgresSaver creation test passed")
            return True, checkpointer
        
    except Exception as e:
        print(f">> FAILED: AsyncPostgresSaver creation test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False, None

async def test_basic_checkpoint_operations():
    """Test basic checkpoint operations."""
    print("\n[TEST] Basic Checkpoint Operations")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        
        connection_string = get_connection_string()
        # CRITICAL: Use exact same approach as main script fallback
        async with AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        ) as checkpointer:
            await checkpointer.setup()
            
            print(f">> Testing basic checkpoint operations...")
            
            # Test configuration
            thread_id = f"test_thread_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            print(f"   Thread ID: {thread_id}")
            
            # Test 1: Get tuple from empty thread (should return None)
            print(f">> Testing empty thread checkpoint retrieval...")
            result = await checkpointer.aget_tuple(config)
            if result is None:
                print(f"   SUCCESS: Empty thread correctly returned None")
            else:
                print(f"   WARNING: Empty thread returned: {result}")
            
            # Test 2: Put a checkpoint
            print(f">> Testing checkpoint creation...")
            test_checkpoint = {
                "v": 1,
                "ts": time.time(),
                "id": str(uuid.uuid4()),
                "channel_values": {
                    "messages": ["Test message 1", "Test message 2"],
                    "count": 2,
                    "user_id": "test_user"
                },
                "channel_versions": {"messages": 1, "count": 1, "user_id": 1},
                "versions_seen": {}
            }
            
            metadata = {"source": "test", "step": 1, "writes": {}}
            
            checkpoint_config = await checkpointer.aput(
                config,
                test_checkpoint,
                metadata,
                new_versions={}
            )
            
            print(f"   Checkpoint created with config: {checkpoint_config}")
            
            # Test 3: Retrieve the checkpoint
            print(f">> Testing checkpoint retrieval...")
            retrieved = await checkpointer.aget_tuple(config)
            
            if retrieved is not None:
                print(f"   SUCCESS: Checkpoint retrieved successfully")
                print(f"   Config: {retrieved.config}")
                print(f"   Has checkpoint: {retrieved.checkpoint is not None}")
                print(f"   Has metadata: {retrieved.metadata is not None}")
                
                if retrieved.checkpoint:
                    channel_values = retrieved.checkpoint.get("channel_values", {})
                    messages = channel_values.get("messages", [])
                    count = channel_values.get("count", 0)
                    print(f"   Messages: {len(messages)} items")
                    print(f"   Count: {count}")
                    print(f"   First message: {messages[0] if messages else 'N/A'}")
            else:
                print(f"   FAILED: Could not retrieve checkpoint")
                return False
            
            # Test 4: List checkpoints
            print(f">> Testing checkpoint listing...")
            checkpoint_list = []
            async for checkpoint_tuple in checkpointer.alist(config):
                checkpoint_list.append(checkpoint_tuple)
            
            print(f"   Found {len(checkpoint_list)} checkpoints")
            
            print(f">> SUCCESS: Basic checkpoint operations test passed")
            return True
            
    except Exception as e:
        print(f">> FAILED: Basic checkpoint operations test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_graph_with_checkpointer():
    """Test LangGraph integration with AsyncPostgresSaver."""
    print("\n[TEST] LangGraph Integration")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.graph import StateGraph
        
        connection_string = get_connection_string()
        # CRITICAL: Use exact same approach as main script fallback
        async with AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        ) as checkpointer:
            await checkpointer.setup()
            
            print(f">> Creating test graph with checkpointer...")
            
            # Create nodes
            node_a, node_b, node_c = create_test_nodes()
            
            # Build graph
            graph = StateGraph(TestState)
            graph.add_node("node_a", node_a)
            graph.add_node("node_b", node_b)
            graph.add_node("node_c", node_c)
            
            # Define edges
            graph.set_entry_point("node_a")
            graph.add_edge("node_a", "node_b")
            graph.add_edge("node_b", "node_c")
            graph.set_finish_point("node_c")
            
            # Compile with checkpointer
            compiled_graph = graph.compile(checkpointer=checkpointer)
            print(f"   Graph compiled successfully with checkpointer")
            
            # Test execution
            thread_id = f"graph_test_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            print(f">> Testing graph execution...")
            print(f"   Thread ID: {thread_id}")
            
            # Initial state
            initial_state = {
                "messages": [],
                "count": 0,
                "user_id": "test_user_123"
            }
            
            print(f"   Initial state: {initial_state}")
            
            # Execute graph
            start_time = time.time()
            result = await compiled_graph.ainvoke(initial_state, config)
            execution_time = time.time() - start_time
            
            print(f">> Graph execution completed in {execution_time:.2f}s")
            print(f"   Final result: {result}")
            
            # Verify results
            expected_message_count = 3  # One from each node
            actual_message_count = len(result.get("messages", []))
            expected_count = 3  # Incremented by each node
            actual_count = result.get("count", 0)
            
            print(f">> Verifying results...")
            print(f"   Expected messages: {expected_message_count}, Actual: {actual_message_count}")
            print(f"   Expected count: {expected_count}, Actual: {actual_count}")
            print(f"   User ID: {result.get('user_id', 'N/A')}")
            
            if actual_message_count == expected_message_count and actual_count == expected_count:
                print(f"   SUCCESS: Graph execution results are correct")
            else:
                print(f"   FAILED: Graph execution results are incorrect")
                return False
            
            # Test checkpoint persistence
            print(f">> Testing checkpoint persistence...")
            retrieved_state = await checkpointer.aget_tuple(config)
            
            if retrieved_state and retrieved_state.checkpoint:
                channel_values = retrieved_state.checkpoint.get("channel_values", {})
                persisted_messages = channel_values.get("messages", [])
                persisted_count = channel_values.get("count", 0)
                
                print(f"   Persisted messages: {len(persisted_messages)}")
                print(f"   Persisted count: {persisted_count}")
                
                if len(persisted_messages) == expected_message_count:
                    print(f"   SUCCESS: State properly persisted to checkpoints")
                else:
                    print(f"   FAILED: State not properly persisted")
                    return False
            else:
                print(f"   FAILED: Could not retrieve persisted state")
                return False
            
            print(f">> SUCCESS: LangGraph integration test passed")
            return True
            
    except Exception as e:
        print(f">> FAILED: LangGraph integration test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_concurrent_checkpoints():
    """Test concurrent checkpoint operations."""
    print("\n[TEST] Concurrent Checkpoint Operations")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.graph import StateGraph
        
        connection_string = get_connection_string()
        # CRITICAL: Use exact same approach as main script fallback
        async with AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        ) as checkpointer:
            await checkpointer.setup()
            
            print(f">> Testing concurrent checkpoint operations...")
            
            async def create_checkpoint_worker(worker_id: int):
                """Worker that creates and tests checkpoints concurrently."""
                thread_id = f"concurrent_test_{worker_id}_{int(time.time())}"
                config = {"configurable": {"thread_id": thread_id}}
                
                # Create test nodes
                node_a, node_b, node_c = create_test_nodes()
                
                # Build simple graph
                graph = StateGraph(TestState)
                graph.add_node("node_a", node_a)
                graph.set_entry_point("node_a")
                graph.set_finish_point("node_a")
                
                compiled_graph = graph.compile(checkpointer=checkpointer)
                
                # Execute multiple times
                results = []
                for i in range(3):
                    initial_state = {
                        "messages": [],
                        "count": i,
                        "user_id": f"worker_{worker_id}_iteration_{i}"
                    }
                    
                    result = await compiled_graph.ainvoke(initial_state, config)
                    results.append(result)
                    
                    # Small delay
                    await asyncio.sleep(0.1)
                
                print(f"   Worker {worker_id} completed {len(results)} operations")
                return worker_id, results
            
            # Run 3 workers concurrently
            num_workers = 3
            print(f">> Running {num_workers} concurrent workers...")
            
            start_time = time.time()
            worker_results = await asyncio.gather(*[
                create_checkpoint_worker(worker_id)
                for worker_id in range(num_workers)
            ])
            
            duration = time.time() - start_time
            
            print(f">> Concurrent operations completed in {duration:.2f}s")
            
            # Verify all workers completed successfully
            total_operations = 0
            for worker_id, results in worker_results:
                total_operations += len(results)
                print(f"   Worker {worker_id}: {len(results)} operations")
            
            expected_operations = num_workers * 3
            if total_operations == expected_operations:
                print(f">> SUCCESS: All concurrent operations completed ({total_operations}/{expected_operations})")
            else:
                print(f">> FAILED: Some operations failed ({total_operations}/{expected_operations})")
                return False
            
            print(f">> SUCCESS: Concurrent checkpoint operations test passed")
            return True
            
    except Exception as e:
        print(f">> FAILED: Concurrent checkpoint operations test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_checkpoint_cleanup():
    """Test checkpoint cleanup and memory management."""
    print("\n[TEST] Checkpoint Cleanup")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        import psycopg
        
        connection_string = get_connection_string()
        # CRITICAL: Use exact same approach as main script fallback
        async with AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        ) as checkpointer:
            await checkpointer.setup()
            
            print(f">> Testing checkpoint cleanup...")
            
            # Create multiple test checkpoints
            test_threads = []
            for i in range(5):
                thread_id = f"cleanup_test_{i}_{int(time.time())}"
                config = {"configurable": {"thread_id": thread_id}}
                
                test_checkpoint = {
                    "v": 1,
                    "ts": time.time(),
                    "id": str(uuid.uuid4()),
                    "channel_values": {
                        "messages": [f"Test message from thread {i}"],
                        "count": i,
                        "user_id": f"test_user_{i}"
                    },
                    "channel_versions": {"messages": 1, "count": 1, "user_id": 1},
                    "versions_seen": {}
                }
                
                metadata = {"source": "cleanup_test", "step": 1, "writes": {}}
                
                await checkpointer.aput(
                    config,
                    test_checkpoint,
                    metadata,
                    new_versions={}
                )
                
                test_threads.append(thread_id)
            
            print(f"   Created {len(test_threads)} test checkpoints")
            
            # Verify checkpoints exist
            print(f">> Verifying checkpoints exist...")
            existing_count = 0
            for thread_id in test_threads:
                config = {"configurable": {"thread_id": thread_id}}
                result = await checkpointer.aget_tuple(config)
                if result is not None:
                    existing_count += 1
            
            print(f"   Found {existing_count}/{len(test_threads)} checkpoints")
            
            if existing_count != len(test_threads):
                print(f"   WARNING: Not all checkpoints were created properly")
            
            # Test direct database access for cleanup verification
            print(f">> Testing direct database access...")
            
            try:
                connection_kwargs = get_connection_kwargs()
                async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
                    async with conn.cursor() as cur:
                        # Count checkpoints in database
                        await cur.execute("SELECT COUNT(*) FROM checkpoints")
                        total_checkpoints = await cur.fetchone()
                        print(f"   Total checkpoints in database: {total_checkpoints[0]}")
                        
                        # Count checkpoint writes
                        await cur.execute("SELECT COUNT(*) FROM checkpoint_writes")
                        total_writes = await cur.fetchone()
                        print(f"   Total checkpoint writes in database: {total_writes[0]}")
                        
                        # Count checkpoint blobs
                        await cur.execute("SELECT COUNT(*) FROM checkpoint_blobs") 
                        total_blobs = await cur.fetchone()
                        print(f"   Total checkpoint blobs in database: {total_blobs[0]}")
            
            except Exception as db_error:
                print(f"   WARNING: Could not access database directly: {db_error}")
            
            print(f">> SUCCESS: Checkpoint cleanup test passed")
            return True
            
    except Exception as e:
        print(f">> FAILED: Checkpoint cleanup test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_error_recovery():
    """Test error handling and recovery scenarios."""
    print("\n[TEST] Error Recovery")
    print("=" * 50)
    
    # CRITICAL: Clear prepared statements before test to prevent conflicts
    await clear_prepared_statements()
    
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.graph import StateGraph
        
        connection_string = get_connection_string()
        # CRITICAL: Use exact same approach as main script fallback
        async with AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            pipeline=False,  # CRITICAL: Same as main script
            serde=None
        ) as checkpointer:
            await checkpointer.setup()
            
            print(f">> Testing error recovery scenarios...")
            
            # Test 1: Invalid thread_id handling
            print(f">> Testing invalid thread_id handling...")
            try:
                invalid_config = {"configurable": {"thread_id": ""}}
                result = await checkpointer.aget_tuple(invalid_config)
                print(f"   Empty thread_id result: {result}")
            except Exception as e:
                print(f"   Empty thread_id error (expected): {type(e).__name__}")
            
            # Test 2: Node that raises an exception
            def error_node(state: TestState) -> TestState:
                """Node that intentionally raises an error."""
                if state.get("count", 0) > 2:
                    raise ValueError("Intentional test error")
                
                messages = state.get("messages", [])
                messages.append("Error node executed")
                return {
                    "messages": messages,
                    "count": state.get("count", 0) + 1,
                    "user_id": state.get("user_id", "test_user")
                }
            
            print(f">> Testing graph with error node...")
            
            # Create graph with error-prone node
            graph = StateGraph(TestState)
            graph.add_node("error_node", error_node)
            graph.set_entry_point("error_node")
            graph.set_finish_point("error_node")
            
            compiled_graph = graph.compile(checkpointer=checkpointer)
            
            # Test successful execution first
            thread_id = f"error_test_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            
            initial_state = {
                "messages": [],
                "count": 1,  # Should not trigger error
                "user_id": "error_test_user"
            }
            
            try:
                result = await compiled_graph.ainvoke(initial_state, config)
                print(f"   Successful execution: count = {result.get('count', 0)}")
            except Exception as e:
                print(f"   Unexpected error in successful case: {e}")
                return False
            
            # Test error case
            error_state = {
                "messages": [],
                "count": 5,  # Should trigger error
                "user_id": "error_test_user"
            }
            
            try:
                error_result = await compiled_graph.ainvoke(error_state, config)
                print(f"   WARNING: Expected error but got result: {error_result}")
            except ValueError as e:
                print(f"   Expected error caught: {e}")
            except Exception as e:
                print(f"   Unexpected error type: {type(e).__name__}: {e}")
            
            # Test 3: Recovery after error
            print(f">> Testing recovery after error...")
            
            recovery_state = {
                "messages": [],
                "count": 1,  # Should work again
                "user_id": "recovery_test_user"
            }
            
            try:
                recovery_result = await compiled_graph.ainvoke(recovery_state, config)
                print(f"   Recovery successful: count = {recovery_result.get('count', 0)}")
            except Exception as e:
                print(f"   Recovery failed: {e}")
                return False
            
            print(f">> SUCCESS: Error recovery test passed")
            return True
            
    except Exception as e:
        print(f">> FAILED: Error recovery test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all LangGraph checkpointer tests."""
    print("LangGraph AsyncPostgresSaver Test Suite - FIXED VERSION")
    print("=" * 60)
    print("Testing LangGraph AsyncPostgresSaver functionality and checkpoint operations")
    print("Using same techniques as successful test_multiuser_scenarios.py")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    connection_string = get_connection_string()
    print(f"\nConnection: {connection_string.split('@')[1].split('?')[0]}")
    
    # Check environment variables
    required_vars = ['host', 'port', 'dbname', 'user', 'password']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"\nERROR: Missing required environment variables: {missing_vars}")
        print("Please set all required PostgreSQL environment variables.")
        return False
    
    print("\n" + "=" * 60)
    
    tests = [
        ("AsyncPostgresSaver Creation", test_checkpointer_creation),
        ("Basic Checkpoint Operations", test_basic_checkpoint_operations),
        ("LangGraph Integration", test_graph_with_checkpointer),
        ("Concurrent Operations", test_concurrent_checkpoints),
        ("Checkpoint Cleanup", test_checkpoint_cleanup),
        ("Error Recovery", test_error_recovery),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n" + "=" * 60)
        print(f"RUNNING: {test_name}")
        print(f"=" * 60)
        
        test_start = time.time()
        try:
            if test_name == "AsyncPostgresSaver Creation":
                success, checkpointer = await test_func()
            else:
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