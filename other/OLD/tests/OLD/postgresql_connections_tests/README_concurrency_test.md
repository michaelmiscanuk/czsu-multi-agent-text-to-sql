# PostgreSQL Concurrency Test

This test file (`test_concurrency.py`) is designed to test the concurrency behavior of the `/analyze` endpoint with PostgreSQL database connections.

## What it tests

- **Concurrent Request Handling**: Makes 2 simultaneous requests to the `/analyze` endpoint
- **Database Connection Stability**: Verifies that PostgreSQL connections work properly under concurrent load
- **Performance Analysis**: Measures response times and identifies potential bottlenecks
- **Error Detection**: Catches and reports connection errors, prepared statement issues, etc.

## Prerequisites

1. **Environment Variables**: Ensure your PostgreSQL environment variables are set:
   - `host`: PostgreSQL server host
   - `port`: PostgreSQL server port
   - `dbname`: Database name
   - `user`: Database username
   - `password`: Database password

2. **Dependencies**: Install required packages:
   ```bash
   pip install pytest httpx asyncio
   ```

## How to run

### Method 1: Direct execution
```bash
cd other/tests/postgresql_connections_tests
python test_concurrency.py
```

### Method 2: Using pytest
```bash
cd other/tests/postgresql_connections_tests
pytest test_concurrency.py -v
```

### Method 3: From project root
```bash
python -m pytest other/tests/postgresql_connections_tests/test_concurrency.py -v
```

## What the test does

1. **Environment Setup**: 
   - Checks PostgreSQL environment variables
   - Overrides JWT authentication for testing

2. **Database Connectivity Test**:
   - Creates a database checkpointer
   - Tests basic database operations
   - Closes connections properly

3. **Concurrency Test**:
   - Generates 2 unique thread IDs
   - Makes 2 simultaneous requests with different prompts:
     - "What is the total revenue for this year?"
     - "Show me the customer demographics breakdown"
   - Measures response times and success rates

4. **Analysis**:
   - Reports success/failure rates
   - Analyzes response time consistency
   - Identifies potential connection issues

## Sample Output

```
🚀 PostgreSQL Concurrency Test Starting...
============================================================
🔧 Setting up test environment...
✅ PostgreSQL environment variables are configured
✅ Authentication dependency overridden for testing
🔍 Testing database connectivity...
🔧 Creating database checkpointer...
✅ Database checkpointer created successfully
✅ Database connectivity test passed
✅ Database connection closed properly
✅ Database connectivity confirmed - proceeding with concurrency test
🎯 Starting concurrency test with 2 simultaneous requests...
📋 Test threads: test_thread_a1b2c3d4, test_thread_e5f6g7h8
📋 Test prompts: ['What is the total revenue for this year?', 'Show me the customer demographics breakdown']
⚡ Executing concurrent requests...
🚀 Starting request for thread test_thread_a1b2c3d4
🚀 Starting request for thread test_thread_e5f6g7h8
📝 Thread test_thread_a1b2c3d4 - Status: 200, Time: 2.45s
✅ Result added: Thread test_thread_a1b2c3d4, Status 200, Time 2.45s
📝 Thread test_thread_e5f6g7h8 - Status: 200, Time: 2.67s
✅ Result added: Thread test_thread_e5f6g7h8, Status 200, Time 2.67s

============================================================
📊 CONCURRENCY TEST RESULTS ANALYSIS
============================================================
🔢 Total Requests: 2
✅ Successful: 2
❌ Failed: 0
📈 Success Rate: 100.0%
⏱️  Total Test Time: 2.78s
⚡ Avg Response Time: 2.56s
🏆 Best Response Time: 2.45s
🐌 Worst Response Time: 2.67s
🎯 Concurrent Requests Completed: ✅ YES

📋 Individual Request Results:
  1. ✅ Thread: test_thread_a1... | Status: 200 | Time: 2.45s
     Run ID: 12345678-1234-1234-1234-123456789abc
  2. ✅ Thread: test_thread_e5... | Status: 200 | Time: 2.67s
     Run ID: 87654321-4321-4321-4321-cba987654321

🔍 CONCURRENCY ANALYSIS:
✅ Both requests completed - database connection handling appears stable
✅ Response times are consistent - good concurrent performance

🏁 OVERALL TEST RESULT: ✅ PASSED
```

## Interpreting Results

- **✅ PASSED**: Both concurrent requests completed successfully
- **❌ FAILED**: Issues with concurrent request handling detected
- **Response Time Analysis**: 
  - Consistent times = good connection pooling
  - Highly variable times = potential connection contention
- **Error Reports**: Shows specific database or connection errors if any occur

## Troubleshooting

If the test fails:

1. **Check Environment Variables**: Ensure all PostgreSQL variables are set correctly
2. **Database Connectivity**: Verify you can connect to the database manually
3. **Dependencies**: Make sure all required packages are installed
4. **Logs**: Look at the detailed error output for specific issues

## Customization

You can modify the test by:

- Changing `TEST_PROMPTS` to use different prompts
- Adjusting the number of concurrent requests in `run_concurrency_test()`
- Modifying success criteria in the `main()` function
- Adding more sophisticated analysis in `analyze_concurrency_results()` 