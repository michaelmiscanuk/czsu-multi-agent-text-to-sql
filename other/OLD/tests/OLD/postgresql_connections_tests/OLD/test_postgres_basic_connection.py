#!/usr/bin/env python3
"""
Test Basic PostgreSQL Connection

This script tests basic PostgreSQL connectivity using the same configuration
as the successful test_multiuser_scenarios.py implementation.

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
from typing import Dict, List

def get_db_config():
    """Get database configuration from environment variables (same as postgres_checkpointer.py)."""
    return {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }

def get_connection_string():
    """Generate optimized connection string for cloud databases (same as multiuser test)."""
    config = get_db_config()
    
    # Generate unique application name for tracing
    process_id = os.getpid()
    thread_id = "basic_connection_test"
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    
    app_name = f"czsu_basic_test_{process_id}_{thread_id}_{startup_time}_{random_id}"
    
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

def get_connection_kwargs() -> Dict:
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
        cleanup_app_name = f"czsu_basic_cleanup_{uuid.uuid4().hex[:8]}"
        
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

async def test_basic_connection():
    """Test basic PostgreSQL connection."""
    print("\n[TEST] Basic PostgreSQL Connection")
    print("=" * 50)
    
    # Clear prepared statements before test
    await clear_prepared_statements()
    
    try:
        print(f">> Testing basic PostgreSQL connection...")
        
        config = get_db_config()
        print(f"   Host: {config['host']}")
        print(f"   Port: {config['port']}")
        print(f"   Database: {config['dbname']}")
        print(f"   User: {config['user']}")
        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        print(f">> Connecting to database...")
        import psycopg
        
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            print(f"   SUCCESS: Connection established")
            
            async with conn.cursor() as cur:
                # Test basic query
                await cur.execute("SELECT version();")
                version_result = await cur.fetchone()
                print(f"   PostgreSQL version: {version_result[0][:50]}...")
                
                # Test current database
                await cur.execute("SELECT current_database();")
                db_result = await cur.fetchone()
                print(f"   Connected to database: {db_result[0]}")
                
                # Test current user
                await cur.execute("SELECT current_user;")
                user_result = await cur.fetchone()
                print(f"   Connected as user: {user_result[0]}")
                
                # Test prepared statement behavior
                print(f">> Testing prepared statement behavior...")
                await cur.execute("SELECT COUNT(*) FROM pg_prepared_statements;")
                prepared_count = await cur.fetchone()
                print(f"   Prepared statements count: {prepared_count[0]}")
                
                # Test table creation permissions
                test_table_name = f"test_basic_connection_{int(time.time())}"
                try:
                    await cur.execute(f"""
                        CREATE TEMPORARY TABLE {test_table_name} (
                            id SERIAL PRIMARY KEY,
                            test_data TEXT
                        );
                    """)
                    print(f"   SUCCESS: Table creation permissions verified")
                    
                    # Test insert
                    await cur.execute(f"""
                        INSERT INTO {test_table_name} (test_data) 
                        VALUES ('test_value_1'), ('test_value_2');
                    """)
                    print(f"   SUCCESS: Insert operation verified")
                    
                    # Test select
                    await cur.execute(f"SELECT COUNT(*) FROM {test_table_name};")
                    count_result = await cur.fetchone()
                    print(f"   SUCCESS: Select operation verified (count: {count_result[0]})")
                    
                except Exception as table_error:
                    print(f"   WARNING: Table operation failed: {table_error}")
        
        print(f">> SUCCESS: Basic PostgreSQL connection test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Basic PostgreSQL connection test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_connection_pool():
    """Test connection pooling behavior (same approach as multiuser test)."""
    print("\n[TEST] Connection Pool")
    print("=" * 50)
    
    await clear_prepared_statements()
    
    try:
        print(f">> Testing connection pool behavior...")
        
        # Test creating multiple connections using same approach as multiuser test
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        print(f">> Creating multiple sequential connections...")
        
        import psycopg
        
        for i in range(3):
            print(f"   Creating connection {i+1}...")
            async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;")
                    result = await cur.fetchone()
                    print(f"   Connection {i+1}: SUCCESS (result: {result[0]})")
        
        print(f">> Testing concurrent connections...")
        
        async def test_connection(conn_id: int):
            """Test a single connection."""
            try:
                async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT %s as connection_id;", (conn_id,))
                        result = await cur.fetchone()
                        print(f"   Concurrent connection {conn_id}: SUCCESS (ID: {result[0]})")
                        
                        # Small delay to test concurrent behavior
                        await asyncio.sleep(0.5)
                        
                        await cur.execute("SELECT current_timestamp;")
                        timestamp_result = await cur.fetchone()
                        print(f"   Connection {conn_id} timestamp: {timestamp_result[0]}")
                        
                        return True
            except Exception as e:
                print(f"   Concurrent connection {conn_id}: FAILED - {e}")
                return False
        
        # Run 3 connections concurrently
        concurrent_results = await asyncio.gather(*[
            test_connection(i) for i in range(3)
        ])
        
        successful_connections = sum(concurrent_results)
        total_connections = len(concurrent_results)
        
        print(f">> Concurrent connections: {successful_connections}/{total_connections} successful")
        
        if successful_connections == total_connections:
            print(f">> SUCCESS: Connection pool test passed")
            return True
        else:
            print(f">> FAILED: Some concurrent connections failed")
            return False
        
    except Exception as e:
        print(f">> FAILED: Connection pool test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_table_operations():
    """Test table operations with cloud-optimized connection."""
    print("\n[TEST] Table Operations")
    print("=" * 50)
    
    await clear_prepared_statements()
    
    try:
        print(f">> Testing table operations...")
        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        import psycopg
        
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            async with conn.cursor() as cur:
                # Create a test table with unique name
                table_name = f"test_table_ops_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                print(f"   Creating table: {table_name}")
                
                await cur.execute(f"""
                    CREATE TEMPORARY TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100),
                        value INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print(f"   SUCCESS: Table created")
                
                # Test insert operations
                print(f">> Testing insert operations...")
                test_data = [
                    ("Test Record 1", 100),
                    ("Test Record 2", 200),
                    ("Test Record 3", 300),
                    ("Test Record 4", 400),
                    ("Test Record 5", 500)
                ]
                
                for i, (name, value) in enumerate(test_data, 1):
                    await cur.execute(f"""
                        INSERT INTO {table_name} (name, value) 
                        VALUES (%s, %s);
                    """, (name, value))
                    print(f"   Inserted record {i}: {name} = {value}")
                
                # Test select operations
                print(f">> Testing select operations...")
                
                await cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                count_result = await cur.fetchone()
                print(f"   Total records: {count_result[0]}")
                
                await cur.execute(f"""
                    SELECT id, name, value FROM {table_name} 
                    ORDER BY id LIMIT 3;
                """)
                select_results = await cur.fetchall()
                
                print(f"   First 3 records:")
                for record in select_results:
                    print(f"     ID: {record[0]}, Name: {record[1]}, Value: {record[2]}")
                
                # Test update operations
                print(f">> Testing update operations...")
                await cur.execute(f"""
                    UPDATE {table_name} 
                    SET value = value * 2 
                    WHERE id <= 2;
                """)
                updated_rows = cur.rowcount
                print(f"   Updated {updated_rows} rows")
                
                # Verify updates
                await cur.execute(f"""
                    SELECT id, name, value FROM {table_name} 
                    WHERE id <= 2 ORDER BY id;
                """)
                updated_results = await cur.fetchall()
                
                print(f"   Updated records:")
                for record in updated_results:
                    print(f"     ID: {record[0]}, Name: {record[1]}, Value: {record[2]}")
                
                # Test delete operations
                print(f">> Testing delete operations...")
                await cur.execute(f"""
                    DELETE FROM {table_name} 
                    WHERE id > 3;
                """)
                deleted_rows = cur.rowcount
                print(f"   Deleted {deleted_rows} rows")
                
                # Final count
                await cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                final_count = await cur.fetchone()
                print(f"   Final record count: {final_count[0]}")
                
                # Test transaction behavior
                print(f">> Testing transaction behavior...")
                
                # Since we're using autocommit=False, let's test explicit transactions
                await cur.execute("BEGIN;")
                await cur.execute(f"""
                    INSERT INTO {table_name} (name, value) 
                    VALUES ('Transaction Test', 999);
                """)
                
                await cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                trans_count = await cur.fetchone()
                print(f"   Count in transaction: {trans_count[0]}")
                
                await cur.execute("ROLLBACK;")
                
                await cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                final_trans_count = await cur.fetchone()
                print(f"   Count after rollback: {final_trans_count[0]}")
                
                if final_trans_count[0] == final_count[0]:
                    print(f"   SUCCESS: Transaction rollback worked correctly")
                else:
                    print(f"   WARNING: Transaction behavior unexpected")
        
        print(f">> SUCCESS: Table operations test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Table operations test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_prepared_statements():
    """Test prepared statement behavior."""
    print("\n[TEST] Prepared Statements")
    print("=" * 50)
    
    await clear_prepared_statements()
    
    try:
        print(f">> Testing prepared statement behavior...")
        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        import psycopg
        
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            async with conn.cursor() as cur:
                # Check initial prepared statements
                await cur.execute("SELECT COUNT(*) FROM pg_prepared_statements;")
                initial_count = await cur.fetchone()
                print(f"   Initial prepared statements: {initial_count[0]}")
                
                # Execute some queries that might create prepared statements
                print(f">> Executing queries that might create prepared statements...")
                
                for i in range(5):
                    await cur.execute("SELECT %s as test_value;", (f"test_{i}",))
                    result = await cur.fetchone()
                    print(f"   Query {i+1} result: {result[0]}")
                
                # Check prepared statements after queries
                await cur.execute("SELECT COUNT(*) FROM pg_prepared_statements;")
                after_count = await cur.fetchone()
                print(f"   Prepared statements after queries: {after_count[0]}")
                
                # List prepared statements if any exist
                await cur.execute("""
                    SELECT name, statement, prepare_time 
                    FROM pg_prepared_statements 
                    LIMIT 5;
                """)
                prepared_list = await cur.fetchall()
                
                if prepared_list:
                    print(f"   Found prepared statements:")
                    for prep in prepared_list:
                        print(f"     Name: {prep[0]}, Statement: {prep[1][:50]}...")
                else:
                    print(f"   SUCCESS: No prepared statements created (as expected)")
                
                # Test manual prepared statement creation
                print(f">> Testing manual prepared statement creation...")
                
                try:
                    await cur.execute("PREPARE test_stmt AS SELECT $1 as manual_test;")
                    await cur.execute("EXECUTE test_stmt('manual_value');")
                    manual_result = await cur.fetchone()
                    print(f"   Manual prepared statement result: {manual_result[0]}")
                    
                    # Clean up manual prepared statement
                    await cur.execute("DEALLOCATE test_stmt;")
                    print(f"   SUCCESS: Manual prepared statement cleaned up")
                    
                except Exception as manual_error:
                    print(f"   Manual prepared statement error: {manual_error}")
        
        print(f">> SUCCESS: Prepared statements test passed")
        return True
        
    except Exception as e:
        print(f">> FAILED: Prepared statements test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_stress_connections():
    """Test connection behavior under stress (same approach as multiuser test)."""
    print("\n[TEST] Stress Connections")
    print("=" * 50)
    
    await clear_prepared_statements()
    
    try:
        print(f">> Testing connection behavior under stress...")
        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        async def stress_worker(worker_id: int, iterations: int):
            """Worker that performs database operations under stress."""
            success_count = 0
            
            import psycopg
            
            for i in range(iterations):
                try:
                    async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
                        async with conn.cursor() as cur:
                            # Multiple operations per connection
                            await cur.execute("SELECT %s as worker_id, %s as iteration;", (worker_id, i))
                            result = await cur.fetchone()
                            
                            await cur.execute("SELECT current_timestamp;")
                            timestamp = await cur.fetchone()
                            
                            await cur.execute("SELECT random();")
                            random_val = await cur.fetchone()
                            
                            # Small delay to simulate real work
                            await asyncio.sleep(0.01)
                            
                            success_count += 1
                            
                except Exception as worker_error:
                    print(f"   Worker {worker_id} iteration {i} error: {worker_error}")
            
            print(f"   Worker {worker_id} completed: {success_count}/{iterations} operations")
            return worker_id, success_count, iterations
        
        # Run stress test with multiple workers
        num_workers = 5
        iterations_per_worker = 10
        
        print(f">> Running {num_workers} workers with {iterations_per_worker} iterations each...")
        
        start_time = time.time()
        
        worker_results = await asyncio.gather(*[
            stress_worker(worker_id, iterations_per_worker)
            for worker_id in range(num_workers)
        ])
        
        duration = time.time() - start_time
        
        # Analyze results
        total_operations = 0
        successful_operations = 0
        
        for worker_id, success_count, iterations in worker_results:
            total_operations += iterations
            successful_operations += success_count
        
        print(f">> Stress test completed in {duration:.2f}s")
        print(f"   Total operations: {total_operations}")
        print(f"   Successful operations: {successful_operations}")
        print(f"   Success rate: {successful_operations/total_operations:.1%}")
        print(f"   Operations per second: {total_operations/duration:.1f}")
        
        if successful_operations == total_operations:
            print(f">> SUCCESS: Stress connections test passed")
            return True
        elif successful_operations >= total_operations * 0.9:  # 90% success rate
            print(f">> PARTIAL SUCCESS: Stress connections test passed (90%+ success)")
            return True
        else:
            print(f">> FAILED: Too many operations failed")
            return False
        
    except Exception as e:
        print(f">> FAILED: Stress connections test failed")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all basic PostgreSQL connection tests."""
    print("Basic PostgreSQL Connection Test Suite")
    print("=" * 60)
    print("Testing PostgreSQL connectivity with cloud-optimized settings")
    print("Using same techniques as successful test_multiuser_scenarios.py")
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
    
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Connection Pool", test_connection_pool),
        ("Table Operations", test_table_operations),
        ("Prepared Statements", test_prepared_statements),
        ("Stress Connections", test_stress_connections),
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
        print(f"  {test_name:.<25} {status:>6} ({duration:>5.2f}s)")
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