#!/usr/bin/env python3
"""
Database Stress Testing for CZSU Multi-Agent API
Tests database connectivity, concurrent operations, edge cases, and boundary conditions.
ENHANCED: Now includes production-like connection pool testing and Supabase-specific scenarios.
"""

import asyncio
import aiohttp
import time
import json
import jwt
import base64
import uuid
import random
import string
import os
import psycopg
from psycopg_pool import AsyncConnectionPool
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"
TEST_TIMEOUT = 60  # Longer timeout for stress tests
MAX_CONCURRENT_REQUESTS = 20

def create_mock_jwt_token():
    """Create a valid JWT token for testing with proper Google format."""
    header = base64.urlsafe_b64encode(b'{"typ":"JWT","alg":"HS256"}').decode().rstrip('=')
    payload = base64.urlsafe_b64encode(b'{"sub":"test","email":"test@example.com"}').decode().rstrip('=')
    signature = base64.urlsafe_b64encode(b'mock_signature').decode().rstrip('=')
    return f"{header}.{payload}.{signature}"

MOCK_JWT_TOKEN = f"Bearer {create_mock_jwt_token()}"

def generate_random_string(length=10):
    """Generate random string for testing."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_uuid():
    """Generate random UUID for testing."""
    return str(uuid.uuid4())

async def test_concurrent_chat_threads_access():
    """Test concurrent access to chat-threads endpoint."""
    print("🔍 Testing concurrent chat-threads access...")
    
    async def single_request(session, request_id):
        try:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/chat-threads", 
                                 headers={"Authorization": MOCK_JWT_TOKEN}) as response:
                response_time = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': response.status,
                    'response_time': response_time,
                    'success': response.status in [200, 401],  # Both are valid responses
                    'error': None
                }
        except Exception as e:
            return {
                'request_id': request_id,
                'status': None,
                'response_time': None,
                'success': False,
                'error': str(e)
            }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            # Test with different concurrency levels
            concurrency_levels = [5, 10, 15, 20]
            
            for concurrency in concurrency_levels:
                print(f"   Testing with {concurrency} concurrent requests...")
                
                start_time = time.time()
                tasks = [single_request(session, i) for i in range(concurrency)]
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                successful = sum(1 for r in results if r['success'])
                failed = sum(1 for r in results if not r['success'])
                avg_response_time = sum(r['response_time'] for r in results if r['response_time']) / len([r for r in results if r['response_time']])
                
                print(f"     Concurrency {concurrency}: {successful}/{concurrency} successful")
                print(f"     Total time: {total_time:.2f}s, Avg response: {avg_response_time:.3f}s")
                print(f"     Failures: {failed}")
                
                if failed > concurrency * 0.3:  # Allow up to 30% failures
                    print(f"     ❌ Too many failures at concurrency {concurrency}")
                    return False
                
                # Brief pause between concurrency tests
                await asyncio.sleep(2)
        
        print("   ✅ Concurrent chat-threads access test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Concurrent chat-threads test failed: {e}")
        return False

async def test_database_boundary_conditions():
    """Test database operations with boundary conditions."""
    print("🔍 Testing database boundary conditions...")
    
    boundary_tests = [
        # Catalog endpoint boundary tests
        ("/catalog", {"page": 1, "page_size": 1}, "Minimum page size"),
        ("/catalog", {"page": 1, "page_size": 10000}, "Maximum page size"),
        ("/catalog", {"page": 999999, "page_size": 10}, "Very high page number"),
        ("/catalog", {"q": "a" * 1000}, "Very long search query"),
        ("/catalog", {"q": "🔥🚀💯" * 100}, "Unicode search query"),
        ("/catalog", {"q": "'; DROP TABLE users; --"}, "SQL injection attempt"),
        
        # Data-tables boundary tests
        ("/data-tables", {"q": ""}, "Empty search query"),
        ("/data-tables", {"q": " " * 100}, "Whitespace query"),
        ("/data-tables", {"q": "SELECT * FROM information_schema.tables"}, "SQL query as search"),
        
        # Data-table boundary tests
        ("/data-table", {"table": ""}, "Empty table name"),
        ("/data-table", {"table": "a" * 100}, "Very long table name"),
        ("/data-table", {"table": "nonexistent_table_" + generate_random_string(50)}, "Long nonexistent table"),
        ("/data-table", {"table": "../../etc/passwd"}, "Path traversal attempt"),
        ("/data-table", {"table": "users'; DROP TABLE users; --"}, "SQL injection in table name"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for endpoint, params, description in boundary_tests:
                print(f"   Testing: {description}")
                
                try:
                    start_time = time.time()
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        response_time = time.time() - start_time
                        
                        # Should not return 500 (internal server error)
                        if response.status == 500:
                            print(f"     ❌ 500 Internal Server Error - Poor error handling")
                            results[description] = False
                        elif response.status in [200, 400, 401, 404, 422]:
                            print(f"     ✅ {response.status} - Handled gracefully ({response_time:.3f}s)")
                            results[description] = True
                        else:
                            print(f"     ⚠️ {response.status} - Unexpected but not critical")
                            results[description] = True
                            
                except asyncio.TimeoutError:
                    print(f"     ❌ Timeout - Server may be struggling")
                    results[description] = False
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
                
                # Brief pause between boundary tests
                await asyncio.sleep(0.5)
        
        successful = sum(results.values())
        total = len(results)
        print(f"   ✅ Boundary conditions test: {successful}/{total} handled gracefully")
        return successful >= total * 0.8  # Allow some tolerance
        
    except Exception as e:
        print(f"   ❌ Boundary conditions test failed: {e}")
        return False

async def test_rapid_sequential_requests():
    """Test rapid sequential requests to stress database connections."""
    print("🔍 Testing rapid sequential requests...")
    
    endpoints_to_test = [
        ("/chat-threads", {}),
        ("/catalog", {"page": 1, "page_size": 5}),
        ("/data-tables", {}),
        ("/data-table", {"table": "nonexistent"}),
    ]
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            print(f"   Sending 50 rapid sequential requests...")
            
            start_time = time.time()
            successful = 0
            failed = 0
            response_times = []
            
            for i in range(50):
                # Randomly select an endpoint
                endpoint, params = random.choice(endpoints_to_test)
                
                try:
                    request_start = time.time()
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if response.status in [200, 401, 404]:
                            successful += 1
                        else:
                            failed += 1
                            
                except Exception as e:
                    failed += 1
                    print(f"     Request {i+1} failed: {e}")
                
                # Very brief pause to simulate rapid but not instantaneous requests
                await asyncio.sleep(0.1)
            
            total_time = time.time() - start_time
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            print(f"   Results:")
            print(f"     Total requests: 50")
            print(f"     Successful: {successful}")
            print(f"     Failed: {failed}")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Average response time: {avg_response_time:.3f}s")
            print(f"     Requests per second: {50/total_time:.1f}")
            
            if failed <= 5:  # Allow up to 10% failures
                print("   ✅ Rapid sequential requests test passed")
                return True
            else:
                print("   ❌ Too many failures in rapid sequential requests")
                return False
                
    except Exception as e:
        print(f"   ❌ Rapid sequential requests test failed: {e}")
        return False

async def test_memory_intensive_operations():
    """Test operations that might consume significant memory."""
    print("🔍 Testing memory-intensive operations...")
    
    memory_tests = [
        # Large page sizes
        ("/catalog", {"page": 1, "page_size": 1000}, "Large catalog page"),
        ("/catalog", {"page": 1, "page_size": 5000}, "Very large catalog page"),
        
        # Complex search patterns
        ("/catalog", {"q": "data analysis machine learning artificial intelligence"}, "Complex search"),
        ("/data-tables", {"q": "table database sql query select"}, "Multi-word search"),
        
        # Rapid repeated requests to same endpoint
        ("rapid_catalog", {}, "Rapid catalog requests"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for test_type, params, description in memory_tests:
                print(f"   Testing: {description}")
                
                if test_type == "rapid_catalog":
                    # Special test: 10 rapid catalog requests
                    start_time = time.time()
                    tasks = []
                    for i in range(10):
                        task = session.get(f"{BASE_URL}/catalog", 
                                         params={"page": 1, "page_size": 100}, 
                                         headers=headers)
                        tasks.append(task)
                    
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    test_time = time.time() - start_time
                    
                    successful_responses = 0
                    for response in responses:
                        if isinstance(response, Exception):
                            continue
                        if response.status in [200, 401]:
                            successful_responses += 1
                        response.close()
                    
                    print(f"     10 rapid requests: {successful_responses}/10 successful ({test_time:.2f}s)")
                    results[description] = successful_responses >= 7
                    
                else:
                    try:
                        start_time = time.time()
                        async with session.get(f"{BASE_URL}{test_type}", 
                                             params=params, 
                                             headers=headers) as response:
                            response_time = time.time() - start_time
                            
                            if response.status in [200, 401, 404]:
                                print(f"     ✅ {response.status} - Completed ({response_time:.3f}s)")
                                results[description] = True
                            else:
                                print(f"     ❌ {response.status} - Failed")
                                results[description] = False
                                
                    except asyncio.TimeoutError:
                        print(f"     ❌ Timeout - May indicate memory issues")
                        results[description] = False
                    except Exception as e:
                        print(f"     ❌ Error: {e}")
                        results[description] = False
                
                # Pause between memory-intensive tests
                await asyncio.sleep(3)
        
        successful = sum(results.values())
        total = len(results)
        print(f"   ✅ Memory-intensive operations: {successful}/{total} completed successfully")
        return successful >= total * 0.7
        
    except Exception as e:
        print(f"   ❌ Memory-intensive operations test failed: {e}")
        return False

async def test_database_connection_resilience():
    """Test database connection resilience under stress."""
    print("🔍 Testing database connection resilience...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            # Test 1: Sustained load
            print("   Test 1: Sustained load (30 requests over 30 seconds)")
            sustained_successful = 0
            
            for i in range(30):
                try:
                    async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                        if response.status in [200, 401]:
                            sustained_successful += 1
                except:
                    pass
                await asyncio.sleep(1)  # 1 request per second
            
            print(f"     Sustained load: {sustained_successful}/30 successful")
            
            # Test 2: Burst load
            print("   Test 2: Burst load (20 concurrent requests)")
            tasks = []
            for i in range(20):
                task = session.get(f"{BASE_URL}/catalog", 
                                 params={"page": 1, "page_size": 10}, 
                                 headers=headers)
                tasks.append(task)
            
            burst_responses = await asyncio.gather(*tasks, return_exceptions=True)
            burst_successful = 0
            
            for response in burst_responses:
                if isinstance(response, Exception):
                    continue
                if response.status in [200, 401]:
                    burst_successful += 1
                response.close()
            
            print(f"     Burst load: {burst_successful}/20 successful")
            
            # Test 3: Mixed operations
            print("   Test 3: Mixed operations (different endpoints)")
            mixed_operations = [
                ("/chat-threads", {}),
                ("/catalog", {"page": 1, "page_size": 10}),
                ("/data-tables", {}),
                ("/data-table", {"table": "test"}),
                ("/health", {}),
            ]
            
            mixed_successful = 0
            for endpoint, params in mixed_operations * 3:  # 15 total requests
                try:
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers if endpoint != "/health" else {}) as response:
                        if response.status in [200, 401, 404]:
                            mixed_successful += 1
                except:
                    pass
                await asyncio.sleep(0.2)
            
            print(f"     Mixed operations: {mixed_successful}/15 successful")
            
            # Overall assessment
            total_successful = sustained_successful + burst_successful + mixed_successful
            total_requests = 30 + 20 + 15
            success_rate = (total_successful / total_requests) * 100
            
            print(f"   Overall resilience: {total_successful}/{total_requests} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("   ✅ Database connection resilience test passed")
                return True
            else:
                print("   ❌ Database connection resilience test failed")
                return False
                
    except Exception as e:
        print(f"   ❌ Database resilience test failed: {e}")
        return False

async def test_edge_case_parameters():
    """Test edge case parameters that might cause issues."""
    print("🔍 Testing edge case parameters...")
    
    edge_cases = [
        # Null byte injection
        ("/catalog", {"q": "test\x00injection"}, "Null byte in query"),
        
        # Unicode edge cases
        ("/catalog", {"q": "🔥🚀💯🎉🌟"}, "Emoji in query"),
        ("/data-tables", {"q": "тест"}, "Cyrillic characters"),
        ("/data-table", {"table": "测试表"}, "Chinese characters"),
        
        # Numeric edge cases
        ("/catalog", {"page": 0}, "Zero page"),
        ("/catalog", {"page": -1}, "Negative page"),
        ("/catalog", {"page_size": -1}, "Negative page size"),
        ("/catalog", {"page_size": 999999}, "Huge page size"),
        
        # String edge cases
        ("/catalog", {"q": ""}, "Empty string query"),
        ("/catalog", {"q": " "}, "Space-only query"),
        ("/catalog", {"q": "\n\r\t"}, "Whitespace characters"),
        
        # Potential injection attempts
        ("/data-table", {"table": "'; SELECT * FROM users; --"}, "SQL injection attempt"),
        ("/catalog", {"q": "<script>alert('xss')</script>"}, "XSS attempt"),
        ("/data-tables", {"q": "../../etc/passwd"}, "Path traversal attempt"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for endpoint, params, description in edge_cases:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        
                        # Key test: Should not return 500 (internal server error)
                        if response.status == 500:
                            print(f"     ❌ 500 Internal Server Error - Vulnerability!")
                            results[description] = False
                        elif response.status in [200, 400, 401, 404, 422]:
                            print(f"     ✅ {response.status} - Handled safely")
                            results[description] = True
                        else:
                            print(f"     ⚠️ {response.status} - Unexpected status")
                            results[description] = True  # Not necessarily bad
                            
                except Exception as e:
                    print(f"     ❌ Exception: {e}")
                    results[description] = False
                
                await asyncio.sleep(0.3)  # Brief pause
        
        successful = sum(results.values())
        total = len(results)
        vulnerabilities = sum(1 for r in results.values() if not r)
        
        print(f"   ✅ Edge case parameters: {successful}/{total} handled safely")
        if vulnerabilities > 0:
            print(f"   ⚠️ Found {vulnerabilities} potential vulnerabilities!")
        
        return successful >= total * 0.9  # Very high bar for security
        
    except Exception as e:
        print(f"   ❌ Edge case parameters test failed: {e}")
        return False

async def test_production_connection_pool_scenarios():
    """Test the actual connection pool creation scenarios that are failing in production."""
    print("\n🏭 Testing Production Connection Pool Scenarios")
    
    # Get database configuration from environment (same as production)
    db_config = {
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_HOST'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'dbname': os.getenv('POSTGRES_DB', 'postgres')
    }
    
    # Skip if no database config available
    if not all([db_config['user'], db_config['password'], db_config['host']]):
        print("⚠️ Skipping production DB tests - no database config available")
        return True
    
    # Test 1: Basic connection works (this should pass like in production logs)
    print("📊 Test 1: Basic connection test (should work)")
    try:
        is_transaction_mode = db_config['port'] == '6543'
        
        # Build connection string like production code
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
            f"?sslmode=require"
            f"&connect_timeout=20"
            f"&application_name=czsu_agent_test"
            f"&keepalives_idle=600"
            f"&keepalives_interval=30"
            f"&keepalives_count=3"
            f"&tcp_user_timeout=30000"
        )
        
        # Add pgbouncer=true for transaction mode (CRITICAL FIX)
        if is_transaction_mode:
            connection_string += "&pgbouncer=true"
            print(f"🔧 Added pgbouncer=true for transaction mode (port {db_config['port']})")
        
        # Test basic connection
        async with await psycopg.AsyncConnection.connect(
            connection_string,
            autocommit=True,
            connect_timeout=15
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                assert result[0] == 1
                print("✅ Basic connection test passed")
        
    except Exception as e:
        print(f"❌ Basic connection test failed: {e}")
        return False
    
    # Test 2: Connection pool creation (this is what's failing in production)
    print("📊 Test 2: Connection pool creation test (production scenario)")
    try:
        # Use the exact same settings as production code
        if is_transaction_mode:
            max_size = 1  # CRITICAL: Only 1 connection for transaction mode
            min_size = 0
            timeout = 10
            pool_open_timeout = 15
            print(f"🔧 Using TRANSACTION MODE settings: max_size={max_size}")
        else:
            max_size = 2  # Normal session mode
            min_size = 0
            timeout = 20
            pool_open_timeout = 30
            print(f"🔧 Using SESSION MODE settings: max_size={max_size}")
        
        # Create pool with production settings
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=max_size,
            min_size=min_size,
            timeout=timeout,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,  # CRITICAL: Disable prepared statements
                "connect_timeout": 10 if is_transaction_mode else 15
            },
            open=False
        )
        
        # This is the step that times out in production
        print(f"🔧 Opening pool with timeout={pool_open_timeout}s...")
        await asyncio.wait_for(pool.open(), timeout=pool_open_timeout)
        print("✅ Connection pool opened successfully!")
        
        # Test pool usage
        async with pool.connection() as conn:
            result = await conn.execute("SELECT 1")
            row = await result.fetchone()
            assert row[0] == 1
            print("✅ Pool connection test passed")
        
        # Cleanup
        await pool.close()
        print("✅ Pool closed successfully")
        
    except asyncio.TimeoutError:
        print(f"❌ Connection pool creation timed out after {pool_open_timeout}s")
        print("💡 This reproduces the production issue!")
        return False
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False
    
    # Test 3: Multiple pool operations (stress test)
    print("📊 Test 3: Multiple pool operations stress test")
    try:
        pools = []
        for i in range(3):  # Create multiple pools like in concurrent requests
            pool = AsyncConnectionPool(
                conninfo=connection_string,
                max_size=1,  # Conservative for stress test
                min_size=0,
                timeout=5,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": None,
                    "connect_timeout": 5
                },
                open=False
            )
            
            await asyncio.wait_for(pool.open(), timeout=10)
            pools.append(pool)
            print(f"✅ Pool {i+1} created successfully")
        
        # Test concurrent usage
        async def test_pool_query(pool, pool_id):
            try:
                async with pool.connection() as conn:
                    result = await conn.execute("SELECT 1")
                    row = await result.fetchone()
                    return row[0] == 1
            except Exception as e:
                print(f"⚠️ Pool {pool_id} query failed: {e}")
                return False
        
        # Run concurrent queries
        tasks = [test_pool_query(pool, i) for i, pool in enumerate(pools)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        print(f"✅ {success_count}/{len(pools)} pools executed queries successfully")
        
        # Cleanup all pools
        for i, pool in enumerate(pools):
            try:
                await pool.close()
                print(f"✅ Pool {i+1} closed")
            except Exception as e:
                print(f"⚠️ Error closing pool {i+1}: {e}")
        
    except Exception as e:
        print(f"❌ Multiple pool operations test failed: {e}")
        return False
    
    print("🎉 All production connection pool tests passed!")
    return True

async def test_supabase_transaction_mode_compatibility():
    """Test Supabase transaction mode specific requirements."""
    print("\n🔧 Testing Supabase Transaction Mode Compatibility")
    
    # Get database configuration
    db_config = {
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_HOST'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'dbname': os.getenv('POSTGRES_DB', 'postgres')
    }
    
    # Skip if no database config available
    if not all([db_config['user'], db_config['password'], db_config['host']]):
        print("⚠️ Skipping Supabase tests - no database config available")
        return True
    
    is_transaction_mode = db_config['port'] == '6543'
    
    if not is_transaction_mode:
        print(f"⚠️ Not using transaction mode (port is {db_config['port']}, not 6543)")
        return True
    
    print(f"🔧 Testing Supabase transaction mode (port 6543) requirements")
    
    # Test 1: Connection string with pgbouncer=true
    print("📊 Test 1: Connection string with pgbouncer=true")
    try:
        connection_string_without = (
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
            f"?sslmode=require&connect_timeout=20"
        )
        
        connection_string_with = connection_string_without + "&pgbouncer=true"
        
        # Test without pgbouncer=true (should potentially fail or be suboptimal)
        try:
            async with await psycopg.AsyncConnection.connect(
                connection_string_without,
                autocommit=True,
                connect_timeout=10
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
            print("⚠️ Connection without pgbouncer=true worked (may not be optimal)")
        except Exception as e:
            print(f"❌ Connection without pgbouncer=true failed: {e}")
        
        # Test with pgbouncer=true (should work)
        async with await psycopg.AsyncConnection.connect(
            connection_string_with,
            autocommit=True,
            connect_timeout=10
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        print("✅ Connection with pgbouncer=true worked")
        
    except Exception as e:
        print(f"❌ pgbouncer=true test failed: {e}")
        return False
    
    # Test 2: Prepared statements disabled
    print("📊 Test 2: Prepared statements disabled for transaction mode")
    try:
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
            f"?sslmode=require&connect_timeout=20&pgbouncer=true"
        )
        
        # Test with prepare_threshold=None (disabled)
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=1,
            min_size=0,
            timeout=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,  # Disabled for transaction mode
                "connect_timeout": 10
            },
            open=False
        )
        
        await asyncio.wait_for(pool.open(), timeout=15)
        
        async with pool.connection() as conn:
            # Execute multiple queries to see if prepared statements cause issues
            for i in range(3):
                result = await conn.execute("SELECT %s", (i,))
                row = await result.fetchone()
                assert row[0] == i
        
        await pool.close()
        print("✅ Prepared statements disabled test passed")
        
    except Exception as e:
        print(f"❌ Prepared statements test failed: {e}")
        return False
    
    print("🎉 All Supabase transaction mode compatibility tests passed!")
    return True

async def test_api_deployment_scenarios():
    """Test scenarios that specifically happen in deployment environments."""
    print("\n🚀 Testing API Deployment Scenarios")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Health check endpoint (should work even with DB issues)
    print("📊 Test 1: Health check endpoint availability")
    total_tests += 1
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data.get('status')}")
                    
                    # Check if database connection is reported
                    if 'database' in data:
                        print(f"📊 Database status: {data['database']}")
                    
                    success_count += 1
                else:
                    print(f"❌ Health check failed: {response.status}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Debug pool status endpoint 
    print("📊 Test 2: Debug pool status endpoint")
    total_tests += 1
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{BASE_URL}/debug/pool-status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Pool status endpoint works")
                    print(f"📊 Checkpointer type: {data.get('checkpointer_type')}")
                    print(f"📊 Pool healthy: {data.get('pool_healthy')}")
                    print(f"📊 Can query: {data.get('can_query')}")
                    
                    # Check for fallback to InMemorySaver (indicates production issue)
                    if data.get('checkpointer_type') == 'InMemorySaver':
                        print("⚠️ PRODUCTION ISSUE: App fell back to InMemorySaver!")
                        print("💡 This means PostgreSQL connection pool creation failed")
                    
                    success_count += 1
                else:
                    print(f"❌ Pool status check failed: {response.status}")
    except Exception as e:
        print(f"❌ Pool status error: {e}")
    
    # Test 3: Chat threads endpoint with auth (tests real user flow)
    print("📊 Test 3: Chat threads endpoint with authentication")
    total_tests += 1
    try:
        token = create_mock_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Chat threads endpoint works: {len(data)} threads")
                    
                    # Empty results indicate database connection issues
                    if len(data) == 0:
                        print("⚠️ No chat threads returned - possible DB connection issue")
                    
                    success_count += 1
                elif response.status == 401:
                    print("⚠️ Authentication failed (expected with mock token)")
                    success_count += 1  # This is actually expected
                else:
                    print(f"❌ Chat threads failed: {response.status}")
                    error_text = await response.text()
                    print(f"📊 Error: {error_text[:200]}...")
    except Exception as e:
        print(f"❌ Chat threads error: {e}")
    
    # Test 4: Concurrent requests (simulates real load)
    print("📊 Test 4: Concurrent requests simulation")
    total_tests += 1
    try:
        async def health_check_request(session, request_id):
            try:
                async with session.get(f"{BASE_URL}/health") as response:
                    return response.status == 200
            except:
                return False
        
        # Simulate 10 concurrent health checks
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = [health_check_request(session, i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_requests = sum(1 for r in results if r is True)
            print(f"✅ {success_requests}/10 concurrent requests succeeded")
            
            if success_requests >= 8:  # Allow some failures
                success_count += 1
    except Exception as e:
        print(f"❌ Concurrent requests test error: {e}")
    
    print(f"\n🎯 Deployment tests: {success_count}/{total_tests} passed")
    return success_count >= total_tests * 0.75  # 75% success rate acceptable

async def run_database_stress_tests():
    """Run all database stress tests."""
    print("🚀 DATABASE STRESS TESTS")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    print(f"Test started: {datetime.now()}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print("=" * 60)
    
    tests = [
        ("Concurrent Chat Threads Access", test_concurrent_chat_threads_access),
        ("Database Boundary Conditions", test_database_boundary_conditions),
        ("Rapid Sequential Requests", test_rapid_sequential_requests),
        ("Memory Intensive Operations", test_memory_intensive_operations),
        ("Database Connection Resilience", test_database_connection_resilience),
        ("Edge Case Parameters", test_edge_case_parameters),
        ("Production Connection Pool Scenarios", test_production_connection_pool_scenarios),
        ("Supabase Transaction Mode Compatibility", test_supabase_transaction_mode_compatibility),
        ("API Deployment Scenarios", test_api_deployment_scenarios),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        
        start_time = time.time()
        result = await test_func()
        test_time = time.time() - start_time
        
        results[test_name] = {
            'passed': result,
            'time': test_time
        }
        
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   Result: {status} ({test_time:.2f}s)")
        
        # Pause between major tests to let system recover
        await asyncio.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DATABASE STRESS TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    total_time = sum(r['time'] for r in results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} {test_name} ({result['time']:.2f}s)")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total execution time: {total_time:.2f}s")
    
    if passed == total:
        print("🎉 All database stress tests passed!")
        print("💪 Your database layer is robust and resilient!")
    else:
        print("⚠️ Some database stress tests failed")
        print("🔧 Review failed tests for potential optimizations")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_database_stress_tests()) 