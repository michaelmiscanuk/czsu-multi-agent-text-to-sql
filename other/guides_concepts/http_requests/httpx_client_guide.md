# HTTPX and AsyncClient: Technical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [HTTPX vs Requests](#httpx-vs-requests)
3. [AsyncClient Deep Dive](#asyncclient-deep-dive)
4. [HTTP Methods](#http-methods)
5. [Configuration Options](#configuration-options)
6. [Context Managers](#context-managers)
7. [Authentication](#authentication)
8. [Error Handling](#error-handling)
9. [Advanced Features](#advanced-features)
10. [Performance Optimization](#performance-optimization)
11. [Best Practices](#best-practices)

## Introduction

HTTPX is a fully featured HTTP client library for Python that supports both synchronous and asynchronous requests. It's designed as a drop-in replacement for the popular `requests` library, with additional support for async/await syntax.

### Key Features
- **Async/await support** - Works with Python's asyncio
- **HTTP/2 support** - Better performance for modern web services
- **Type hints** - Full type annotation support
- **Timeouts** - Comprehensive timeout configuration
- **Connection pooling** - Efficient resource management
- **Streaming** - Handle large responses efficiently

## HTTPX vs Requests

### Synchronous Comparison
```python
# Requests (synchronous only)
import requests

response = requests.get("https://api.example.com/data")
data = response.json()

# HTTPX (synchronous)
import httpx

response = httpx.get("https://api.example.com/data")
data = response.json()
```

### Asynchronous Support
```python
# HTTPX (asynchronous)
import httpx
import asyncio

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

data = asyncio.run(fetch_data())
```

### Performance Comparison
```python
import time
import asyncio
import httpx
import requests

# Synchronous (requests)
def fetch_sync():
    urls = ["https://api.example.com/data"] * 10
    responses = []
    for url in urls:
        response = requests.get(url)
        responses.append(response)
    return responses

# Asynchronous (httpx)
async def fetch_async():
    urls = ["https://api.example.com/data"] * 10
    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(
            *[client.get(url) for url in urls]
        )
    return responses

# Timing comparison
start = time.time()
sync_responses = fetch_sync()
sync_time = time.time() - start

start = time.time()
async_responses = asyncio.run(fetch_async())
async_time = time.time() - start

print(f"Synchronous: {sync_time:.2f}s")
print(f"Asynchronous: {async_time:.2f}s")
print(f"Speedup: {sync_time/async_time:.2f}x")
```

## AsyncClient Deep Dive

### Basic Usage
```python
import httpx
import asyncio

async def basic_usage():
    # Create and use client
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/users")
        return response.json()

# Run the async function
result = asyncio.run(basic_usage())
```

### Client Configuration
```python
import httpx
from httpx import Timeout

async def configured_client():
    # Comprehensive client configuration
    client = httpx.AsyncClient(
        base_url="https://api.example.com",
        timeout=Timeout(10.0, connect=5.0, read=10.0, write=5.0, pool=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
        headers={
            "User-Agent": "MyApp/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        cookies={"session": "abc123"},
        follow_redirects=True,
        verify=True,  # SSL verification
        cert=None,    # Client certificate
        trust_env=True  # Use environment proxy settings
    )
    
    async with client:
        response = await client.get("/users")
        return response.json()

result = asyncio.run(configured_client())
```

### Client Lifecycle Management
```python
import httpx
import asyncio

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(base_url=self.base_url)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
    
    async def get_users(self):
        response = await self.client.get("/users")
        return response.json()
    
    async def get_user(self, user_id: int):
        response = await self.client.get(f"/users/{user_id}")
        return response.json()

# Usage
async def main():
    async with APIClient("https://api.example.com") as client:
        users = await client.get_users()
        user = await client.get_user(1)
        return users, user

asyncio.run(main())
```

## HTTP Methods

### GET Requests
```python
import httpx

async def get_examples():
    async with httpx.AsyncClient() as client:
        # Simple GET
        response = await client.get("https://api.example.com/users")
        
        # GET with parameters
        response = await client.get(
            "https://api.example.com/users",
            params={"page": 1, "limit": 10}
        )
        
        # GET with headers
        response = await client.get(
            "https://api.example.com/users",
            headers={"Authorization": "Bearer token123"}
        )
        
        return response.json()
```

### POST Requests
```python
import httpx

async def post_examples():
    async with httpx.AsyncClient() as client:
        # POST with JSON data
        user_data = {"name": "John", "email": "john@example.com"}
        response = await client.post(
            "https://api.example.com/users",
            json=user_data
        )
        
        # POST with form data
        form_data = {"username": "john", "password": "secret"}
        response = await client.post(
            "https://api.example.com/login",
            data=form_data
        )
        
        # POST with files
        files = {"file": ("test.txt", open("test.txt", "rb"), "text/plain")}
        response = await client.post(
            "https://api.example.com/upload",
            files=files
        )
        
        return response.json()
```

### PUT, PATCH, DELETE Requests
```python
import httpx

async def other_methods():
    async with httpx.AsyncClient() as client:
        # PUT request (full update)
        user_data = {"name": "John Updated", "email": "john@example.com"}
        response = await client.put(
            "https://api.example.com/users/1",
            json=user_data
        )
        
        # PATCH request (partial update)
        patch_data = {"name": "John Patched"}
        response = await client.patch(
            "https://api.example.com/users/1",
            json=patch_data
        )
        
        # DELETE request
        response = await client.delete("https://api.example.com/users/1")
        
        # HEAD request (metadata only)
        response = await client.head("https://api.example.com/users/1")
        
        return response.status_code
```

## Configuration Options

### Timeout Configuration
```python
import httpx
from httpx import Timeout

async def timeout_examples():
    # Simple timeout (applies to all operations)
    client = httpx.AsyncClient(timeout=10.0)
    
    # Granular timeout control
    timeout = Timeout(
        connect=5.0,    # Time to establish connection
        read=10.0,      # Time to read response
        write=5.0,      # Time to write request
        pool=10.0       # Time to get connection from pool
    )
    client = httpx.AsyncClient(timeout=timeout)
    
    # Per-request timeout
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.example.com/slow-endpoint",
            timeout=30.0  # Override client timeout
        )
    
    # No timeout
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get("https://api.example.com/data")
```

### Connection Limits
```python
import httpx

async def connection_limits():
    limits = httpx.Limits(
        max_keepalive_connections=10,  # Keep-alive connections
        max_connections=100,           # Total connections
        keepalive_expiry=30.0         # Keep-alive timeout
    )
    
    async with httpx.AsyncClient(limits=limits) as client:
        # Client will manage connections efficiently
        response = await client.get("https://api.example.com/data")
```

### SSL and Certificate Configuration
```python
import httpx
import ssl

async def ssl_configuration():
    # Disable SSL verification (not recommended for production)
    client = httpx.AsyncClient(verify=False)
    
    # Custom SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    client = httpx.AsyncClient(verify=ssl_context)
    
    # Client certificate
    client = httpx.AsyncClient(cert=("client.pem", "client.key"))
    
    # CA bundle
    client = httpx.AsyncClient(verify="/path/to/ca-bundle.pem")
```

## Context Managers

### Async Context Manager Pattern
```python
import httpx
import asyncio

async def context_manager_example():
    # Automatic resource cleanup
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        # Client is automatically closed when exiting context
        return response.json()

# Multiple requests with same client
async def multiple_requests():
    async with httpx.AsyncClient(base_url="https://api.example.com") as client:
        # All requests use the same connection pool
        users = await client.get("/users")
        posts = await client.get("/posts")
        comments = await client.get("/comments")
        
        return {
            "users": users.json(),
            "posts": posts.json(),
            "comments": comments.json()
        }
```

### Manual Resource Management
```python
import httpx
import asyncio

async def manual_management():
    # Manual client management (not recommended)
    client = httpx.AsyncClient()
    try:
        response = await client.get("https://api.example.com/data")
        return response.json()
    finally:
        # Always close the client
        await client.aclose()

# Better approach with context manager
async def better_approach():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

## Authentication

### Bearer Token Authentication
```python
import httpx

async def bearer_token_auth():
    token = "your-jwt-token-here"
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient(headers=headers) as client:
        response = await client.get("https://api.example.com/protected")
        return response.json()
```

### Basic Authentication
```python
import httpx

async def basic_auth():
    auth = httpx.BasicAuth("username", "password")
    
    async with httpx.AsyncClient(auth=auth) as client:
        response = await client.get("https://api.example.com/protected")
        return response.json()
```

### Custom Authentication
```python
import httpx

class CustomAuth(httpx.Auth):
    def __init__(self, token):
        self.token = token
    
    def auth_flow(self, request):
        # Add custom authentication logic
        request.headers["X-Custom-Auth"] = self.token
        yield request

async def custom_auth():
    auth = CustomAuth("my-custom-token")
    
    async with httpx.AsyncClient(auth=auth) as client:
        response = await client.get("https://api.example.com/protected")
        return response.json()
```

## Error Handling

### HTTP Status Errors
```python
import httpx

async def handle_http_errors():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("https://api.example.com/data")
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            return None
```

### Network Errors
```python
import httpx

async def handle_network_errors():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("https://api.example.com/data")
            return response.json()
        except httpx.RequestError as e:
            print(f"Network error: {e}")
            return None
        except httpx.TimeoutException as e:
            print(f"Request timed out: {e}")
            return None
```

### Comprehensive Error Handling
```python
import httpx
import asyncio

async def robust_request(url: str, max_retries: int = 3):
    """Make a robust HTTP request with retries and error handling"""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                # Client error (4xx) - don't retry
                print(f"Client error: {e.response.status_code}")
                return None
            else:
                # Server error (5xx) - retry
                print(f"Server error: {e.response.status_code}, retrying...")
                
        except httpx.TimeoutException:
            print(f"Request timed out, attempt {attempt + 1}/{max_retries}")
            
        except httpx.RequestError as e:
            print(f"Network error: {e}, attempt {attempt + 1}/{max_retries}")
        
        # Wait before retry
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    print("All retry attempts failed")
    return None
```

## Advanced Features

### Streaming Responses
```python
import httpx
import asyncio

async def stream_response():
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", "https://api.example.com/large-data") as response:
            async for chunk in response.aiter_bytes():
                # Process chunk by chunk
                print(f"Received chunk: {len(chunk)} bytes")
                
        # Or stream lines
        async with client.stream("GET", "https://api.example.com/logs") as response:
            async for line in response.aiter_lines():
                print(f"Log line: {line}")
```

### Request and Response Hooks
```python
import httpx
import time

async def request_hook(request):
    print(f"Sending request: {request.method} {request.url}")
    request.headers["X-Request-ID"] = str(time.time())

async def response_hook(response):
    print(f"Received response: {response.status_code}")
    response.read()  # Ensure response is read

async def with_hooks():
    async with httpx.AsyncClient(
        event_hooks={
            "request": [request_hook],
            "response": [response_hook]
        }
    ) as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

### Connection Pooling
```python
import httpx
import asyncio

async def connection_pooling_example():
    # Shared connection pool across requests
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=10)
    ) as client:
        
        # Multiple requests will reuse connections
        tasks = [
            client.get("https://api.example.com/users"),
            client.get("https://api.example.com/posts"),
            client.get("https://api.example.com/comments")
        ]
        
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

## Performance Optimization

### Connection Reuse
```python
import httpx
import asyncio

async def optimized_requests():
    # Single client for multiple requests
    async with httpx.AsyncClient(
        base_url="https://api.example.com",
        limits=httpx.Limits(max_keepalive_connections=20)
    ) as client:
        
        # All requests reuse the same connection pool
        results = await asyncio.gather(
            client.get("/endpoint1"),
            client.get("/endpoint2"),
            client.get("/endpoint3"),
            client.get("/endpoint4"),
            client.get("/endpoint5")
        )
        
        return [r.json() for r in results]
```

### Batch Processing
```python
import httpx
import asyncio

async def batch_requests(urls, batch_size=10):
    """Process URLs in batches to avoid overwhelming the server"""
    results = []
    
    async with httpx.AsyncClient() as client:
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[client.get(url) for url in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
    
    return results
```

### Memory-Efficient Streaming
```python
import httpx
import asyncio

async def memory_efficient_download(url, filename):
    """Download large files without loading into memory"""
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            with open(filename, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
    
    print(f"Downloaded {filename}")
```

## Best Practices

### 1. Always Use Context Managers
```python
# Good
async def good_example():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# Bad - resource leak
async def bad_example():
    client = httpx.AsyncClient()
    response = await client.get("https://api.example.com/data")
    # Client is never closed!
    return response.json()
```

### 2. Configure Timeouts
```python
# Good - explicit timeout
async with httpx.AsyncClient(timeout=10.0) as client:
    response = await client.get("https://api.example.com/data")

# Bad - no timeout (can hang forever)
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com/data")
```

### 3. Handle Errors Properly
```python
async def robust_request():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        print(f"Network error: {e}")
        return None
```

### 4. Reuse Clients for Multiple Requests
```python
# Good - reuse client
async def multiple_requests():
    async with httpx.AsyncClient() as client:
        response1 = await client.get("https://api.example.com/users")
        response2 = await client.get("https://api.example.com/posts")
        return response1.json(), response2.json()

# Bad - new client for each request
async def wasteful_requests():
    async with httpx.AsyncClient() as client1:
        response1 = await client1.get("https://api.example.com/users")
    
    async with httpx.AsyncClient() as client2:
        response2 = await client2.get("https://api.example.com/posts")
    
    return response1.json(), response2.json()
```

### 5. Use Type Hints
```python
from typing import Dict, List, Optional
import httpx

async def typed_request(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[Dict[str, str]] = None
) -> Optional[Dict]:
    """Make a typed HTTP request"""
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None
```

### 6. Use Base URLs
```python
# Good - use base URL
async with httpx.AsyncClient(base_url="https://api.example.com") as client:
    users = await client.get("/users")
    posts = await client.get("/posts")

# Less efficient - repeat base URL
async with httpx.AsyncClient() as client:
    users = await client.get("https://api.example.com/users")
    posts = await client.get("https://api.example.com/posts")
```

## Real-World Example: Your Test Code

Here's how the concepts apply to your specific test code:

```python
import httpx
import asyncio

async def test_chat_endpoints():
    # Create client with configuration
    async with httpx.AsyncClient(
        base_url="http://localhost:8000",  # Base URL for all requests
        timeout=httpx.Timeout(30.0)       # 30-second timeout
    ) as client:
        
        # Authentication header
        token = "your-jwt-token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test different endpoints
        results = []
        
        # GET request with path parameter
        response = await client.get(
            f"/chat/{thread_id}/sentiments",
            headers=headers
        )
        results.append(response.json())
        
        # GET request with query parameters
        response = await client.get(
            "/chat-threads",
            headers=headers,
            params={"page": 1, "limit": 10}
        )
        results.append(response.json())
        
        # DELETE request
        response = await client.delete(
            f"/chat/{thread_id}",
            headers=headers
        )
        results.append(response.json())
        
        # POST request with JSON body
        response = await client.post(
            "/debug/set-env",
            headers=headers,
            json={"debug": "1"}
        )
        results.append(response.json())
        
        return results

# Run the test
asyncio.run(test_chat_endpoints())
```

## Summary

HTTPX with AsyncClient provides a powerful, flexible way to make HTTP requests in async Python code. Key takeaways:

- **Use context managers** for automatic resource cleanup
- **Configure timeouts** to prevent hanging requests
- **Reuse clients** for multiple requests to the same API
- **Handle errors** appropriately for robust applications
- **Use async/await** for concurrent request processing
- **Leverage connection pooling** for better performance

The combination of HTTPX's AsyncClient with Python's async/await syntax makes it ideal for building high-performance applications that need to make multiple HTTP requests efficiently. 