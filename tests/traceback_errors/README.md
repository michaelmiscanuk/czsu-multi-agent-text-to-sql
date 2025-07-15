# Server Traceback Capture System

This directory contains comprehensive traceback reports from test executions, including detailed server-side error information that is normally not visible to HTTP clients.

## Overview

When testing API endpoints, you typically only see generic error messages like "Internal Server Error" from the client side. However, the server logs contain detailed tracebacks with the actual error information. This system captures those server-side tracebacks and saves them to files for debugging.

## Features

### 1. Server Log Capture
- Captures server-side logs during test execution
- Intercepts logging from FastAPI, Uvicorn, and Starlette
- Preserves full exception tracebacks with stack traces

### 2. Detailed Error Analysis
- Extracts exception types, messages, and timestamps
- Associates server errors with specific test cases
- Provides context about when and where errors occurred

### 3. Comprehensive Reporting
- Saves multiple types of reports:
  - `*_traceback_failures.txt` - General test failure reports
  - `*_server_tracebacks.txt` - Detailed server-side traceback reports
  - `*_exception_traceback.txt` - Client-side exception reports

## File Types

### Test Failure Reports (`*_traceback_failures.txt`)
Contains:
- Test execution summary
- Individual test failures with error messages
- Server tracebacks associated with each failure
- Performance metrics and timing information

### Server Traceback Reports (`*_server_tracebacks.txt`)
Contains:
- Pure server-side error information
- Full exception tracebacks from the server
- Exception types and messages
- Timestamps and logging levels

### Exception Reports (`*_exception_traceback.txt`)
Contains:
- Client-side exceptions that occurred during testing
- Full Python tracebacks from the test execution
- Context information about when the exception occurred

## Usage

### In Test Files

```python
from tests.helpers import (
    make_request_with_traceback_capture,
    extract_detailed_error_info,
    save_server_traceback_report
)

# Make a request with server traceback capture
result = await make_request_with_traceback_capture(
    client, "GET", "/api/endpoint", headers=headers
)

# Extract detailed error information
error_info = extract_detailed_error_info(result)

# Check for server-side errors
if error_info['server_tracebacks']:
    print(f"Server errors captured: {len(error_info['server_tracebacks'])}")
    for tb in error_info['server_tracebacks']:
        print(f"  {tb['exception_type']}: {tb['exception_message']}")
```

### Automatic Integration

The system is automatically integrated into test files like `test_phase8_catalog.py`. When tests run:

1. Server logs are captured during each HTTP request
2. Any server-side exceptions are extracted and stored
3. Reports are automatically generated when tests complete
4. Files are saved to `tests/traceback_errors/` with descriptive names

## Benefits

### For Debugging
- **Full Context**: See exactly what went wrong on the server side
- **Stack Traces**: Complete call stacks showing where errors occurred
- **Timing**: Know when errors happened during test execution

### For Development
- **Bug Detection**: Catch server-side bugs that don't show up in HTTP responses
- **Performance**: Identify slow operations or resource issues
- **Validation**: Ensure error handling works correctly

### For Testing
- **Comprehensive**: Capture both client and server-side error information
- **Automated**: No manual intervention needed to capture tracebacks
- **Persistent**: All error information is saved to files for later analysis

## Example Output

When a server error occurs, you'll see detailed information like:

```
SERVER TRACEBACK #1:
Timestamp: 2025-01-11T21:21:31.226173
Level: ERROR
Exception Type: NameError
Exception Message: name 'db_pFDGDFGFDGath' is not defined

FULL SERVER TRACEBACK:
Traceback (most recent call last):
  File "/.../fastapi/routing.py", line 214, in run_endpoint_function
    return await run_in_threadpool(dependant.call, **values)
  File "/.../starlette/concurrency.py", line 37, in run_in_threadpool
    return await anyio.to_thread.run_sync(func)
  File "/.../api/routes/catalog.py", line 105, in get_data_table
    with sqlite3.connect(db_pFDGDFGFDGath) as conn:
NameError: name 'db_pFDGDFGFDGath' is not defined
```

## Demo

Run the demo script to see the functionality in action:

```bash
python test_traceback_demo.py
```

This will:
1. Make a request that triggers a server error
2. Capture the server-side traceback
3. Display the captured information
4. Save a detailed report to a file

## Configuration

The system automatically captures logs from these sources:
- `uvicorn` - Web server logs
- `uvicorn.error` - Server error logs
- `uvicorn.access` - Access logs
- `fastapi` - FastAPI framework logs
- `starlette` - Starlette framework logs
- `app` - Application-specific logs

Log levels captured: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Files Generated

Each test run may generate multiple files:
- `test_phase8_catalog_traceback_failures.txt` - Main test failure report
- `test_phase8_catalog_server_tracebacks.txt` - Server traceback details
- `test_phase8_catalog_exception_traceback.txt` - Client exception details
- `demo_traceback_test_server_tracebacks.txt` - Demo script output

## Maintenance

Files in this directory are automatically overwritten on each test run to ensure fresh data. If you need to preserve specific reports, copy them to another location before running tests again. 