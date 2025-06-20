#!/usr/bin/env python3
"""
Comprehensive PostgreSQL Checkpointer Test Suite
Tests both asyncpg (application operations) and LangGraph AsyncPostgresSaver (checkpointing)
"""

import asyncio
import os
import sys
import uuid
import time
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import our PostgreSQL module
from my_agent.utils.postgres_checkpointer import (
    get_healthy_pool,
    setup_users_threads_runs_table,
    create_thread_run_entry,
    update_thread_run_sentiment,
    get_thread_run_sentiments,
    get_user_chat_threads,
    get_user_chat_threads_count,
    delete_user_thread_entries,
    get_postgres_checkpointer,
    get_postgres_checkpointer_with_context,
    get_conversation_messages_from_checkpoints,
    get_queries_and_results_from_latest_checkpoint,
    test_connection_health,
    test_basic_postgres_connection,
    test_pool_connection,
    check_postgres_env_vars,
    force_close_all_connections,
    get_connection_string
)

class TestColors:
    """ANSI color codes for test output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class PostgreSQLTestSuite:
    """Comprehensive test suite for PostgreSQL operations"""
    
    def __init__(self):
        self.test_results = []
        self.test_email = f"test_user_{uuid.uuid4().hex[:8]}@example.com"
        self.test_thread_id = f"test_thread_{uuid.uuid4().hex[:8]}"
        self.cleanup_items = []
        
    def log_test(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Log test result with colored output"""
        status = f"{TestColors.OKGREEN}✅ PASS{TestColors.ENDC}" if success else f"{TestColors.FAIL}❌ FAIL{TestColors.ENDC}"
        duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
        print(f"{status} {test_name}{duration_str}")
        if message:
            print(f"    {message}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })
    
    async def test_environment_setup(self):
        """Test 1: Environment Setup"""
        print(f"\n{TestColors.HEADER}[ENV] Test 1: Environment Setup{TestColors.ENDC}")
        
        try:
            # Check environment variables
            env_ok = check_postgres_env_vars()
            self.log_test("Environment Variables", env_ok, "Missing required environment variables")
            
            if not env_ok:
                print(f"{TestColors.FAIL}[ERROR] Cannot continue without proper environment setup{TestColors.ENDC}")
                return False
            
            # Test connection string generation
            conn_string = get_connection_string()
            self.log_test("Connection String", bool(conn_string), "Failed to generate connection string")
            
            # Test basic connection
            basic_conn = await test_basic_postgres_connection()
            self.log_test("Basic Connection", basic_conn, "Direct asyncpg connection failed")
            
            # Test connection health
            health = await test_connection_health()
            self.log_test("Connection Health", health, "Connection health check failed")
            
            return True
            
        except Exception as e:
            self.log_test("Environment Setup", False, f"Unexpected exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def test_asyncpg_pool_operations(self):
        """Test 2: AsyncPG Pool Operations"""
        print(f"\n{TestColors.HEADER}[POOL] Test 2: AsyncPG Pool Operations{TestColors.ENDC}")
        
        try:
            # Test pool creation
            pool = await get_healthy_pool()
            self.log_test("Pool Creation", pool is not None, "Failed to create connection pool")
            
            if not pool:
                return False
            
            # Test basic pool query
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                self.log_test("Basic Pool Query", result == 1, f"Expected 1, got {result}")
            
            # Test transaction
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("SELECT 1")
                self.log_test("Transaction", True, "Transaction completed successfully")
            
            # Test multiple queries
            async with pool.acquire() as conn:
                await conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, value TEXT)")
                await conn.execute("INSERT INTO test_table (value) VALUES ($1), ($2)", "test1", "test2")
                rows = await conn.fetch("SELECT * FROM test_table")
                self.log_test("Multiple Queries", len(rows) >= 2, f"Expected at least 2 rows, got {len(rows)}")
            
            # Test pool health check
            is_healthy = await is_pool_healthy(pool)
            self.log_test("Pool Health Check", is_healthy, "Pool health check failed")
            
            return True
            
        except Exception as e:
            self.log_test("AsyncPG Pool Operations", False, f"Unexpected exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def test_users_threads_runs_table(self):
        """Test 3: Users Threads Runs Table"""
        print(f"\n{TestColors.HEADER}[TABLE] Test 3: Users Threads Runs Table{TestColors.ENDC}")
        
        try:
            # Setup table
            await setup_users_threads_runs_table()
            self.log_test("Table Setup", True, "Table setup completed")
            
            # Test data insertion
            test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
            test_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
            test_prompt = "This is a test prompt for comprehensive testing"
            
            run_id = await create_thread_run_entry(test_email, test_thread_id, test_prompt)
            self.log_test("Data Insertion", run_id is not None, "Failed to create thread run entry")
            
            # Add to cleanup list
            self.cleanup_items.append(('thread_entry', test_email, test_thread_id))
            
            # Test data retrieval - single thread
            threads = await get_user_chat_threads(test_email, limit=5)
            found_thread = False
            for thread in threads:
                if thread["thread_id"] == test_thread_id:
                    found_thread = True
                    break
            
            self.log_test("Data Retrieval (Single Thread)", found_thread, "Thread not found in results")
            
            # Test thread count
            count = await get_user_chat_threads_count(test_email)
            self.log_test("Thread Count", count >= 1, f"Expected at least 1, got {count}")
            
            # Test sentiment update
            sentiment_result = await update_thread_run_sentiment(run_id, True, test_email)
            self.log_test("Sentiment Update", sentiment_result, "Failed to update sentiment")
            
            # Test sentiment retrieval
            sentiments = await get_thread_run_sentiments(test_email, test_thread_id)
            self.log_test("Sentiment Retrieval", 
                         run_id in sentiments and sentiments[run_id] is True,
                         "Sentiment not found or incorrect")
            
            return True
            
        except Exception as e:
            self.log_test("Users Threads Runs Table", False, f"Unexpected exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def test_langgraph_checkpointer(self):
        """Test 4: LangGraph Checkpointer"""
        print(f"\n{TestColors.HEADER}[CHECKPOINT] Test 4: LangGraph Checkpointer{TestColors.ENDC}")
        
        try:
            # Create checkpointer
            checkpointer = await get_postgres_checkpointer()
            self.log_test("Checkpointer Creation", checkpointer is not None, "Failed to create checkpointer")
            
            if not checkpointer:
                return False
            
            # Test checkpointer get (empty)
            test_thread_id = f"test_checkpointer_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": test_thread_id}}
            
            result = await checkpointer.aget(config)
            self.log_test("Checkpointer Get (Empty)", result is None, f"Expected None for new thread, got {result}")
            
            # Test checkpointer put
            test_checkpoint = {
                "channel_values": {
                    "messages": [{"content": "Test message", "role": "user"}],
                    "final_answer": "This is a test answer"
                }
            }
            
            test_metadata = {"test_metadata": True}
            
            await checkpointer.aput(config, test_checkpoint, test_metadata, {})
            self.log_test("Checkpointer Put", True, "Put operation completed")
            
            # Test checkpointer get (with data)
            result = await checkpointer.aget(config)
            self.log_test("Checkpointer Get (With Data)", 
                         result is not None and "channel_values" in result,
                         "Failed to retrieve checkpoint data")
            
            # Test checkpointer get_tuple
            result_tuple = await checkpointer.aget_tuple(config)
            self.log_test("Checkpointer Get Tuple", 
                         result_tuple is not None and hasattr(result_tuple, 'checkpoint') and hasattr(result_tuple, 'metadata'),
                         "Failed to retrieve checkpoint tuple")
            
            if result_tuple:
                metadata_ok = result_tuple.metadata is not None and "test_metadata" in result_tuple.metadata
                self.log_test("Checkpointer Metadata", metadata_ok, "Metadata not preserved correctly")
            
            # Test checkpointer list
            checkpoints = []
            async for checkpoint in checkpointer.alist(config):
                checkpoints.append(checkpoint)
            
            self.log_test("Checkpointer List", len(checkpoints) > 0, f"Expected at least 1 checkpoint, got {len(checkpoints)}")
            
            return True
            
        except Exception as e:
            self.log_test("LangGraph Checkpointer", False, f"Unexpected exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def test_conversation_message_extraction(self):
        """Test 5: Conversation Message Extraction"""
        print(f"\n{TestColors.HEADER}[EXTRACT] Test 5: Conversation Message Extraction{TestColors.ENDC}")
        
        try:
            # Create checkpointer
            checkpointer = await get_postgres_checkpointer()
            
            # Create a test thread with multiple checkpoints
            test_thread_id = f"test_conversation_{uuid.uuid4().hex[:8]}"
            test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
            config = {"configurable": {"thread_id": test_thread_id}}
            
            # Create a database entry for this thread (for security check)
            await create_thread_run_entry(test_email, test_thread_id, "Test conversation extraction")
            self.cleanup_items.append(('thread_entry', test_email, test_thread_id))
            
            # Create multiple checkpoints with user questions and AI answers
            for i in range(3):
                checkpoint = {
                    "channel_values": {
                        "messages": [{"content": f"Test question {i}", "role": "user"}],
                        "final_answer": f"Test answer {i}"
                    }
                }
                metadata = {"writes": {"user_input": {"prompt": f"Test question {i}"}}}
                await checkpointer.aput(config, checkpoint, metadata, {})
            
            # Test message extraction
            messages = await get_conversation_messages_from_checkpoints(checkpointer, test_thread_id, test_email)
            
            self.log_test("Message Extraction", len(messages) > 0, f"Expected messages, got {len(messages)}")
            
            # Test queries extraction
            checkpoint = {
                "channel_values": {
                    "queries_and_results": [
                        ["SELECT * FROM test", "Test result"]
                    ]
                }
            }
            await checkpointer.aput(config, checkpoint, {}, {})
            
            queries = await get_queries_and_results_from_latest_checkpoint(checkpointer, test_thread_id)
            self.log_test("Queries Extraction", len(queries) > 0, f"Expected queries, got {len(queries)}")
            
            return True
            
        except Exception as e:
            self.log_test("Conversation Message Extraction", False, f"Unexpected exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def test_concurrent_operations(self):
        """Test 6: Concurrent Operations"""
        print(f"\n{TestColors.HEADER}[CONCUR] Test 6: Concurrent Operations{TestColors.ENDC}")
        
        try:
            async def concurrent_asyncpg_task(task_id: int):
                """Run concurrent asyncpg operations"""
                pool = await get_healthy_pool()
                async with pool.acquire() as conn:
                    # Create a temporary table specific to this task
                    table_name = f"concurrent_test_{task_id}"
                    await conn.execute(f"CREATE TEMPORARY TABLE {table_name} (id SERIAL PRIMARY KEY, value TEXT)")
                    
                    # Insert some data
                    for i in range(5):
                        await conn.execute(f"INSERT INTO {table_name} (value) VALUES ($1)", f"value_{task_id}_{i}")
                    
                    # Read the data back
                    rows = await conn.fetch(f"SELECT * FROM {table_name}")
                    return len(rows)
            
            async def concurrent_checkpointer_task(task_id: int):
                """Run concurrent checkpointer operations"""
                checkpointer = await get_postgres_checkpointer()
                thread_id = f"concurrent_test_{task_id}_{uuid.uuid4().hex[:8]}"
                config = {"configurable": {"thread_id": thread_id}}
                
                # Write a checkpoint
                checkpoint = {
                    "channel_values": {
                        "messages": [{"content": f"Concurrent test {task_id}", "role": "user"}],
                        "final_answer": f"Concurrent answer {task_id}"
                    }
                }
                await checkpointer.aput(config, checkpoint, {}, {})
                
                # Read it back
                result = await checkpointer.aget(config)
                return result is not None
            
            # Run concurrent asyncpg tasks
            asyncpg_tasks = [concurrent_asyncpg_task(i) for i in range(5)]
            asyncpg_results = await asyncio.gather(*asyncpg_tasks, return_exceptions=True)
            
            asyncpg_success = all(isinstance(r, int) and r == 5 for r in asyncpg_results)
            self.log_test("Concurrent AsyncPG", asyncpg_success, 
                         f"Some tasks failed: {[r for r in asyncpg_results if not (isinstance(r, int) and r == 5)]}")
            
            # Run concurrent checkpointer tasks
            checkpointer_tasks = [concurrent_checkpointer_task(i) for i in range(5)]
            checkpointer_results = await asyncio.gather(*checkpointer_tasks, return_exceptions=True)
            
            checkpointer_success = all(isinstance(r, bool) and r for r in checkpointer_results)
            self.log_test("Concurrent Checkpointer", checkpointer_success,
                         f"Some tasks failed: {[r for r in checkpointer_results if not (isinstance(r, bool) and r)]}")
            
            return asyncpg_success and checkpointer_success
            
        except Exception as e:
            self.log_test("Concurrent Operations", False, f"Unexpected exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def test_error_handling_and_resilience(self):
        """Test 7: Error Handling and Resilience"""
        print(f"\n{TestColors.HEADER}[ERROR] Test 7: Error Handling and Resilience{TestColors.ENDC}")
        
        start_time = time.time()
        try:
            # Test invalid query handling
            try:
                pool = await get_healthy_pool()
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT invalid_column FROM non_existent_table")
                self.log_test("Invalid Query Handling", False, "Should have raised an exception")
            except Exception:
                self.log_test("Invalid Query Handling", True, "Properly handled invalid query")
            
            # Test checkpointer error handling
            try:
                checkpointer = await get_postgres_checkpointer()
                invalid_config = {"configurable": {"thread_id": None}}  # Invalid config
                await checkpointer.aget(invalid_config)
                self.log_test("Checkpointer Error Handling", False, "Should have raised an exception")
            except Exception:
                self.log_test("Checkpointer Error Handling", True, "Properly handled invalid checkpointer config")
            
            # Test connection recovery
            pool = await get_healthy_pool()
            if pool and not pool._closed:
                # Test that pool can handle connection after error
                async with pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    self.log_test("Connection Recovery", result == 1, "Connection recovered after error")
            
            duration = time.time() - start_time
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Error Handling and Resilience", False, f"Unexpected exception: {str(e)}", duration)
            traceback.print_exc()
            return False
    
    async def test_performance_benchmarks(self):
        """Test 8: Performance Benchmarks"""
        print(f"\n{TestColors.HEADER}[PERF] Test 8: Performance Benchmarks{TestColors.ENDC}")
        
        try:
            # Benchmark asyncpg operations
            start_time = time.time()
            pool = await get_healthy_pool()
            
            # Single connection performance
            async with pool.acquire() as conn:
                for i in range(100):
                    await conn.fetchval("SELECT $1", i)
            
            single_conn_duration = time.time() - start_time
            self.log_test("AsyncPG Single Connection (100 queries)", True,
                         f"Completed in {single_conn_duration:.3f}s ({100/single_conn_duration:.1f} qps)")
            
            # Concurrent connection performance
            async def benchmark_task():
                async with pool.acquire() as conn:
                    for i in range(20):
                        await conn.fetchval("SELECT $1", i)
            
            start_time = time.time()
            await asyncio.gather(*[benchmark_task() for _ in range(5)])
            concurrent_duration = time.time() - start_time
            
            self.log_test("AsyncPG Concurrent (5x20 queries)", True,
                         f"Completed in {concurrent_duration:.3f}s ({100/concurrent_duration:.1f} qps)")
            
            # Checkpointer performance
            start_time = time.time()
            checkpointer = await get_postgres_checkpointer()
            
            for i in range(10):
                config = {"configurable": {"thread_id": f"perf_test_{i}"}}
                await checkpointer.aget(config)
            
            checkpointer_duration = time.time() - start_time
            self.log_test("Checkpointer Performance (10 gets)", True,
                         f"Completed in {checkpointer_duration:.3f}s ({10/checkpointer_duration:.1f} ops/s)")
            
            return True
            
        except Exception as e:
            self.log_test("Performance Benchmarks", False, f"Exception: {str(e)}")
            traceback.print_exc()
            return False
    
    async def cleanup_test_data(self):
        """Cleanup test data"""
        print(f"\n{TestColors.HEADER}[CLEANUP] Cleanup Test Data{TestColors.ENDC}")
        
        try:
            for item_type, *args in self.cleanup_items:
                if item_type == 'thread_entry':
                    email, thread_id = args
                    result = await delete_user_thread_entries(email, thread_id)
                    self.log_test(f"Cleanup Thread {thread_id}", result['deleted_count'] > 0,
                                 f"Deleted {result['deleted_count']} entries")
            
            # Force close all connections
            await force_close_all_connections()
            self.log_test("Connection Cleanup", True, "All connections closed")
            
            return True
            
        except Exception as e:
            self.log_test("Cleanup", False, f"Exception: {str(e)}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{TestColors.HEADER}[SUMMARY] TEST SUMMARY{TestColors.ENDC}")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"{TestColors.OKGREEN}Passed: {passed_tests}{TestColors.ENDC}")
        print(f"{TestColors.FAIL}Failed: {failed_tests}{TestColors.ENDC}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n{TestColors.FAIL}Failed Tests:{TestColors.ENDC}")
            for result in self.test_results:
                if not result['success']:
                    print(f"  [FAIL] {result['test']}: {result['message']}")
        
        # Save detailed results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'results': self.test_results
            }, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return failed_tests == 0

async def main():
    """Run the comprehensive test suite"""
    print(f"{TestColors.BOLD}{TestColors.HEADER}")
    print("[TEST] COMPREHENSIVE POSTGRESQL CHECKPOINTER TEST SUITE")
    print("Testing both AsyncPG (application) and LangGraph AsyncPostgresSaver (checkpointing)")
    print("=" * 80)
    print(f"{TestColors.ENDC}")
    
    test_suite = PostgreSQLTestSuite()
    
    try:
        # Run all tests
        tests = [
            test_suite.test_environment_setup(),
            test_suite.test_asyncpg_pool_operations(),
            test_suite.test_users_threads_runs_table(),
            test_suite.test_langgraph_checkpointer(),
            test_suite.test_conversation_message_extraction(),
            test_suite.test_concurrent_operations(),
            test_suite.test_error_handling_and_resilience(),
            test_suite.test_performance_benchmarks(),
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"{TestColors.FAIL}Test {i+1} crashed: {result}{TestColors.ENDC}")
                traceback.print_exc()
        
        # Cleanup
        await test_suite.cleanup_test_data()
        
        # Print summary
        success = test_suite.print_summary()
        
        if success:
            print(f"\n{TestColors.OKGREEN}{TestColors.BOLD}[SUCCESS] ALL TESTS PASSED!{TestColors.ENDC}")
            return 0
        else:
            print(f"\n{TestColors.FAIL}{TestColors.BOLD}[ERROR] SOME TESTS FAILED{TestColors.ENDC}")
            return 1
            
    except Exception as e:
        print(f"{TestColors.FAIL}Test suite crashed: {e}{TestColors.ENDC}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 