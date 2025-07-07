# Async/Await in Python: Technical Guide

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem with Synchronous Code](#the-problem-with-synchronous-code)
3. [How Async/Await Works](#how-asyncawait-works)
4. [Event Loop Deep Dive](#event-loop-deep-dive)
5. [Coroutines vs Functions](#coroutines-vs-functions)
6. [Practical Examples](#practical-examples)
7. [Common Patterns](#common-patterns)
8. [Performance Considerations](#performance-considerations)
9. [Best Practices](#best-practices)

## Introduction

Asynchronous programming in Python allows you to write concurrent code that can handle multiple operations without blocking. The `async`/`await` syntax, introduced in Python 3.5, provides a clean way to write asynchronous code that looks similar to synchronous code.

## The Problem with Synchronous Code

### Traditional Synchronous Execution
```python
import time
import requests

def fetch_data_sync():
    print("Starting request 1...")
    response1 = requests.get("https://api.example.com/data1")  # Blocks for 2 seconds
    print("Request 1 complete")
    
    print("Starting request 2...")
    response2 = requests.get("https://api.example.com/data2")  # Blocks for 2 seconds
    print("Request 2 complete")
    
    return response1, response2

# Total time: ~4 seconds (sequential)
start = time.time()
fetch_data_sync()
print(f"Total time: {time.time() - start:.2f} seconds")
```

### The I/O Bottleneck
- **CPU sits idle** during network requests
- **No parallelism** - one operation at a time
- **Poor resource utilization** - especially bad for I/O-bound operations

## How Async/Await Works

### Basic Async Function
```python
import asyncio
import aiohttp

async def fetch_data_async():
    print("Starting request 1...")
    async with aiohttp.ClientSession() as session:
        response1 = await session.get("https://api.example.com/data1")  # Can pause here
        print("Request 1 complete")
        
        print("Starting request 2...")
        response2 = await session.get("https://api.example.com/data2")  # Can pause here
        print("Request 2 complete")
        
        return response1, response2

# Still sequential, but can be made concurrent
asyncio.run(fetch_data_async())
```

### Concurrent Execution
```python
async def fetch_single_url(session, url, name):
    print(f"Starting {name}...")
    response = await session.get(url)
    print(f"{name} complete")
    return response

async def fetch_data_concurrent():
    async with aiohttp.ClientSession() as session:
        # Both requests run concurrently
        response1, response2 = await asyncio.gather(
            fetch_single_url(session, "https://api.example.com/data1", "Request 1"),
            fetch_single_url(session, "https://api.example.com/data2", "Request 2")
        )
        return response1, response2

# Total time: ~2 seconds (concurrent)
start = time.time()
asyncio.run(fetch_data_concurrent())
print(f"Total time: {time.time() - start:.2f} seconds")
```

## Event Loop Deep Dive

### What is the Event Loop?
The event loop is like a **restaurant manager** that coordinates multiple waiters (coroutines) serving customers. Instead of having one waiter handle one customer from start to finish, the manager switches waiters between customers when they're waiting for something (like the kitchen to prepare food).

**Key Responsibilities:**
- **Manages coroutines** - keeps track of all running async functions
- **Schedules execution** - decides when to run each coroutine
- **Handles I/O** - manages file, network, and database operations
- **Coordinates tasks** - switches between coroutines when they await

### Restaurant Analogy
```python
# Think of this like a restaurant:

# Traditional (synchronous) restaurant:
# 1 waiter serves 1 customer completely, others wait in line
def synchronous_restaurant():
    serve_customer_1()  # Everyone waits while this happens
    serve_customer_2()  # Everyone waits while this happens
    serve_customer_3()  # Everyone waits while this happens

# Async restaurant with event loop manager:
# 1 waiter can serve multiple customers, switching when waiting
async def async_restaurant():
    # Waiter takes order from customer 1
    await take_order_customer_1()    # Waiter goes to kitchen
    # While kitchen cooks, waiter serves customer 2
    await take_order_customer_2()    # Waiter goes to kitchen again
    # While both meals cook, waiter serves customer 3
    await take_order_customer_3()    # Now waiter delivers ready meals
```

### How the Event Loop Works - Step by Step

```python
import asyncio
import time

# Global counter to show execution order
step_counter = 0

def next_step():
    global step_counter
    step_counter += 1
    return step_counter

async def cook_meal(name, cook_time):
    """Simulate cooking a meal"""
    step = next_step()
    print(f"🍳 Step {step}: Started cooking {name} (takes {cook_time}s)")
    await asyncio.sleep(cook_time)  # This is where we yield control
    step = next_step()
    print(f"✅ Step {step}: {name} is ready!")
    return f"{name} meal"

async def serve_customer(customer_name, meal, cook_time):
    """Simulate serving one customer"""
    step = next_step()
    print(f"👋 Step {step}: Greeting {customer_name}")
    
    # Take the order (instant)
    step = next_step()
    print(f"📝 Step {step}: Taking order from {customer_name}: {meal}")
    
    # Send order to kitchen and wait (this is where we "await")
    meal_result = await cook_meal(meal, cook_time)
    
    # Serve the meal (instant)
    step = next_step()
    print(f"🍽️ Step {step}: Serving {meal_result} to {customer_name}")
    return f"{customer_name} served"

async def simple_restaurant_demo():
    """Simple demo showing event loop switching with numbered steps"""
    global step_counter
    step_counter = 0  # Reset counter
    
    print("🏪 Restaurant Demo: Watch the Step Numbers!")
    print("=" * 50)
    
    # Start serving 3 customers concurrently
    # Watch how the step numbers show the switching!
    results = await asyncio.gather(
        serve_customer("Alice", "Pasta", 2),
        serve_customer("Bob", "Burger", 1),
        serve_customer("Carol", "Salad", 0.5)
    )
    
    print("=" * 50)
    print("🎉 All customers served!")
    print(f"📊 Total steps: {step_counter}")
    return results

# Run the demo
asyncio.run(simple_restaurant_demo())
```

**Expected Output (showing the switching):**
```
🏪 Restaurant Demo: Watch the Step Numbers!
==================================================
👋 Step 1: Greeting Alice
📝 Step 2: Taking order from Alice: Pasta
🍳 Step 3: Started cooking Pasta (takes 2s)
👋 Step 4: Greeting Bob
📝 Step 5: Taking order from Bob: Burger
🍳 Step 6: Started cooking Burger (takes 1s)
👋 Step 7: Greeting Carol
📝 Step 8: Taking order from Carol: Salad
🍳 Step 9: Started cooking Salad (takes 0.5s)
✅ Step 10: Salad is ready!
🍽️ Step 11: Serving Salad meal to Carol
✅ Step 12: Burger is ready!
🍽️ Step 13: Serving Burger meal to Bob
✅ Step 14: Pasta is ready!
🍽️ Step 15: Serving Pasta meal to Alice
==================================================
🎉 All customers served!
📊 Total steps: 15
```

**What the numbers show:**
- **Steps 1-3**: Alice starts (greeting → order → cooking starts)
- **Steps 4-6**: Bob starts (greeting → order → cooking starts) 
- **Steps 7-9**: Carol starts (greeting → order → cooking starts)
- **Steps 10-11**: Carol finishes first (salad was fastest)
- **Steps 12-13**: Bob finishes next (burger was medium)
- **Steps 14-15**: Alice finishes last (pasta took longest)

**Key Insight**: See how steps 1-9 happen quickly (no waiting), then steps 10-15 happen as the cooking completes. The event loop switches between customers when they're waiting for cooking!

### Event Loop States - Visual Representation

```python
import asyncio
import time

async def demonstrate_states():
    """Show how coroutines move through different states"""
    
    async def task_with_states(name, delay):
        print(f"🟢 RUNNING: {name} starts")
        
        print(f"🟡 SUSPENDED: {name} going to sleep for {delay}s")
        await asyncio.sleep(delay)  # SUSPENDED state
        
        print(f"🟢 RUNNING: {name} wakes up")
        print(f"🔵 COMPLETED: {name} finished")
        return f"{name} result"
    
    # Create multiple tasks
    tasks = [
        asyncio.create_task(task_with_states("Task1", 1)),
        asyncio.create_task(task_with_states("Task2", 2)),
        asyncio.create_task(task_with_states("Task3", 0.5))
    ]
    
    # Event loop manages all tasks
    results = await asyncio.gather(*tasks)
    return results

asyncio.run(demonstrate_states())
```

### Event Loop Lifecycle - Under the Hood

```python
import asyncio
import time

async def detailed_lifecycle_demo():
    """Detailed demonstration of event loop lifecycle"""
    
    print("📋 Event Loop Lifecycle Demo")
    print("=" * 40)
    
    async def step_by_step_task(name, steps):
        for i, (action, delay) in enumerate(steps, 1):
            print(f"  🔄 {name} - Step {i}: {action}")
            if delay > 0:
                print(f"    ⏳ {name} - Waiting {delay}s...")
                await asyncio.sleep(delay)  # Yield control to event loop
                print(f"    ✅ {name} - Wait complete")
            else:
                print(f"    ⚡ {name} - Instant action")
        
        print(f"  🏁 {name} - All steps complete")
        return f"{name} finished"
    
    # Define different task workflows
    task_a_steps = [
        ("Initialize", 0),
        ("Connect to API", 1),
        ("Process data", 0.5),
        ("Save results", 0.2)
    ]
    
    task_b_steps = [
        ("Load config", 0),
        ("Query database", 1.5),
        ("Generate report", 0.3)
    ]
    
    task_c_steps = [
        ("Validate input", 0),
        ("Call external service", 0.8),
        ("Clean up", 0.1)
    ]
    
    # Run all tasks concurrently
    # Watch how the event loop switches between them
    results = await asyncio.gather(
        step_by_step_task("TaskA", task_a_steps),
        step_by_step_task("TaskB", task_b_steps),
        step_by_step_task("TaskC", task_c_steps)
    )
    
    print("=" * 40)
    print("🎯 All tasks completed!")
    return results

asyncio.run(detailed_lifecycle_demo())
```

### Event Loop Inspection - Advanced Understanding

```python
import asyncio
import time

async def inspect_event_loop():
    """Inspect the event loop while it's running"""
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    
    print(f"🔍 Event Loop Information:")
    print(f"  Type: {type(loop)}")
    print(f"  Running: {loop.is_running()}")
    print(f"  Debug mode: {loop.get_debug()}")
    
    async def monitored_task(name, duration):
        start_time = time.time()
        print(f"📊 {name} - Starting (loop time: {loop.time():.2f})")
        
        # Simulate work with periodic checks
        elapsed = 0
        while elapsed < duration:
            await asyncio.sleep(0.1)  # Small sleep to yield control
            elapsed = time.time() - start_time
            
            # Check how many tasks are pending
            pending_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
            print(f"  📈 {name} - Progress: {elapsed:.1f}s, Pending tasks: {pending_tasks}")
        
        print(f"  ✅ {name} - Completed in {elapsed:.1f}s")
        return f"{name} done"
    
    # Start multiple monitored tasks
    tasks = [
        asyncio.create_task(monitored_task("Monitor-A", 1.0)),
        asyncio.create_task(monitored_task("Monitor-B", 1.5)),
        asyncio.create_task(monitored_task("Monitor-C", 0.8))
    ]
    
    # Wait for all tasks with timeout
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3.0)
        print("🎉 All monitored tasks completed!")
        return results
    except asyncio.TimeoutError:
        print("⏰ Tasks timed out!")
        return None

asyncio.run(inspect_event_loop())
```

### Event Loop Control - Practical Examples

```python
import asyncio
import time

# Example 1: Manual event loop control
def manual_event_loop_control():
    """Demonstrate manual event loop management"""
    
    async def background_task():
        for i in range(5):
            print(f"🔄 Background task iteration {i+1}")
            await asyncio.sleep(0.5)
        return "Background complete"
    
    async def main_task():
        print("🚀 Starting main task")
        
        # Start background task
        bg_task = asyncio.create_task(background_task())
        
        # Do other work
        for i in range(3):
            print(f"⚡ Main task work {i+1}")
            await asyncio.sleep(0.7)
        
        # Wait for background task to complete
        bg_result = await bg_task
        print(f"✅ Main task complete. Background: {bg_result}")
        
        return "Main complete"
    
    # Manual loop management
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        print("🎯 Starting manual event loop")
        result = loop.run_until_complete(main_task())
        print(f"🏆 Final result: {result}")
    finally:
        loop.close()
        print("🔒 Event loop closed")

# Example 2: Event loop with callbacks
async def event_loop_with_callbacks():
    """Show how event loop handles callbacks"""
    
    def callback_function(name, delay):
        print(f"📞 Callback {name} executed after {delay}s")
    
    async def callback_demo():
        loop = asyncio.get_event_loop()
        
        # Schedule callbacks at different times
        loop.call_later(1.0, callback_function, "First", 1.0)
        loop.call_later(0.5, callback_function, "Second", 0.5)
        loop.call_later(1.5, callback_function, "Third", 1.5)
        
        print("⏰ Callbacks scheduled, waiting...")
        await asyncio.sleep(2.0)  # Wait for callbacks to execute
        print("✅ Callback demo complete")
    
    await callback_demo()

# Run the examples
print("=" * 60)
print("🎮 MANUAL EVENT LOOP CONTROL")
print("=" * 60)
manual_event_loop_control()

print("\n" + "=" * 60)
print("📞 EVENT LOOP WITH CALLBACKS")
print("=" * 60)
asyncio.run(event_loop_with_callbacks())
```

### Real-World Event Loop Patterns

```python
import asyncio
import aiohttp
import time

async def real_world_event_loop_patterns():
    """Show real-world patterns of event loop usage"""
    
    # Pattern 1: Producer-Consumer with event loop
    async def producer(queue, name, items):
        """Produce items and put them in queue"""
        for i in range(items):
            item = f"{name}-item-{i+1}"
            await queue.put(item)
            print(f"📦 Producer {name} created: {item}")
            await asyncio.sleep(0.1)  # Simulate work
        
        await queue.put(None)  # Signal completion
        print(f"🏁 Producer {name} finished")
    
    async def consumer(queue, name):
        """Consume items from queue"""
        consumed = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            
            print(f"🔄 Consumer {name} processing: {item}")
            await asyncio.sleep(0.05)  # Simulate processing
            consumed += 1
            queue.task_done()
        
        print(f"✅ Consumer {name} finished, processed {consumed} items")
        return consumed
    
    # Pattern 2: HTTP requests with connection pooling
    async def http_request_pattern():
        """Show event loop managing HTTP connections"""
        urls = [
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/2",
            "https://httpbin.org/delay/1.5"
        ]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, url in enumerate(urls):
                task = asyncio.create_task(fetch_url(session, url, f"Request-{i+1}"))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
    
    async def fetch_url(session, url, name):
        """Fetch URL with timing"""
        start = time.time()
        try:
            async with session.get(url) as response:
                data = await response.json()
                elapsed = time.time() - start
                print(f"🌐 {name} completed in {elapsed:.2f}s")
                return data
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return None
    
    # Run producer-consumer pattern
    print("🏭 Producer-Consumer Pattern")
    print("-" * 40)
    
    queue = asyncio.Queue(maxsize=5)
    
    # Start producer and consumer concurrently
    producer_task = asyncio.create_task(producer(queue, "Factory", 5))
    consumer_task = asyncio.create_task(consumer(queue, "Worker"))
    
    # Wait for both to complete
    await asyncio.gather(producer_task, consumer_task)
    
    print("\n🌐 HTTP Request Pattern")
    print("-" * 40)
    
    # Run HTTP pattern (commented out to avoid actual HTTP calls)
    # http_results = await http_request_pattern()
    print("HTTP pattern demonstrated above (commented out for demo)")

# Run the real-world patterns
asyncio.run(real_world_event_loop_patterns())
```

### Key Takeaways

1. **Event Loop is a Manager**: It coordinates multiple tasks, not executes them all at once
2. **Cooperative Multitasking**: Tasks voluntarily yield control with `await`
3. **Single Thread**: All async code runs in one thread, but doesn't block
4. **I/O Efficiency**: Perfect for network/disk operations, not CPU-intensive work
5. **State Management**: Tasks move between Running → Suspended → Ready → Completed

### Mental Model
```
Think of the event loop like a DJ at a party:
- Multiple people (coroutines) want to request songs (tasks)
- DJ (event loop) takes requests and queues them
- When a song (I/O operation) is playing, DJ can take more requests
- DJ switches between songs smoothly without dead air
- Everyone gets their turn, no one waits unnecessarily
```

## Coroutines vs Functions

### Regular Functions
```python
def regular_function():
    print("This runs immediately")
    return "result"

# Executes immediately when called
result = regular_function()
```

### Coroutines
```python
async def coroutine_function():
    print("This runs when awaited")
    return "result"

# Creates a coroutine object, doesn't execute
coro = coroutine_function()
print(type(coro))  # <class 'coroutine'>

# Must be awaited or run in event loop
result = asyncio.run(coro)
```

### Coroutine Objects
```python
async def example():
    await asyncio.sleep(1)
    return "Hello"

# Create coroutine object
coro = example()

# Different ways to execute:
# 1. Using asyncio.run()
result = asyncio.run(coro)

# 2. Using await (inside another async function)
async def caller():
    result = await example()
    return result

# 3. Using asyncio.create_task()
async def with_task():
    task = asyncio.create_task(example())
    result = await task
    return result
```

## Practical Examples

### HTTP Requests (Your Test Case)
```python
import asyncio
import httpx

async def test_endpoint(client, endpoint):
    print(f"Testing {endpoint}...")
    response = await client.get(endpoint)
    print(f"✅ {endpoint} - Status: {response.status_code}")
    return response

async def run_tests():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Sequential execution
        await test_endpoint(client, "/health")
        await test_endpoint(client, "/users")
        await test_endpoint(client, "/posts")
        
        # Concurrent execution
        results = await asyncio.gather(
            test_endpoint(client, "/endpoint1"),
            test_endpoint(client, "/endpoint2"),
            test_endpoint(client, "/endpoint3")
        )
        return results

asyncio.run(run_tests())
```

### Database Operations
```python
import asyncio
import asyncpg

async def fetch_user(pool, user_id):
    async with pool.acquire() as conn:
        result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        return result

async def main():
    # Create connection pool
    pool = await asyncpg.create_pool(
        host="localhost",
        database="mydb",
        user="user",
        password="password"
    )
    
    # Fetch multiple users concurrently
    users = await asyncio.gather(
        fetch_user(pool, 1),
        fetch_user(pool, 2),
        fetch_user(pool, 3)
    )
    
    await pool.close()
    return users

asyncio.run(main())
```

## Common Patterns

### 1. Gathering Results
```python
async def gather_example():
    # Run multiple coroutines concurrently
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    return results
```

### 2. Creating Tasks
```python
async def task_example():
    # Create tasks for background execution
    task1 = asyncio.create_task(fetch_data("url1"))
    task2 = asyncio.create_task(fetch_data("url2"))
    
    # Do other work...
    await some_other_work()
    
    # Wait for tasks to complete
    result1 = await task1
    result2 = await task2
    
    return result1, result2
```

### 3. Timeout Handling
```python
async def with_timeout():
    try:
        result = await asyncio.wait_for(
            slow_operation(), 
            timeout=5.0
        )
        return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
```

### 4. Error Handling
```python
async def error_handling_example():
    try:
        result = await risky_operation()
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
```

## Performance Considerations

### When to Use Async
✅ **Good for:**
- I/O-bound operations (network, disk, database)
- Multiple concurrent operations
- Web servers and APIs
- Chat applications, real-time features

❌ **Not good for:**
- CPU-bound operations (use multiprocessing instead)
- Simple scripts with minimal I/O
- Single operation with no concurrency

### Performance Metrics
```python
import time
import asyncio

async def measure_performance():
    # Sequential
    start = time.time()
    for i in range(10):
        await asyncio.sleep(0.1)
    sequential_time = time.time() - start
    
    # Concurrent
    start = time.time()
    await asyncio.gather(*[asyncio.sleep(0.1) for _ in range(10)])
    concurrent_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.2f}x")

asyncio.run(measure_performance())
```

## Best Practices

### 1. Always Use Async Context Managers
```python
# Good
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com")

# Bad - doesn't properly clean up
client = httpx.AsyncClient()
response = await client.get("https://api.example.com")
# Connection might not be closed properly
```

### 2. Handle Exceptions Properly
```python
async def robust_request():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com")
            return response.json()
    except httpx.RequestError as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 3. Use Type Hints
```python
from typing import List, Optional
import httpx

async def fetch_users(client: httpx.AsyncClient) -> Optional[List[dict]]:
    try:
        response = await client.get("/users")
        return response.json()
    except Exception:
        return None
```

### 4. Don't Block the Event Loop
```python
import asyncio
import time

# Bad - blocks event loop
async def blocking_operation():
    time.sleep(5)  # Blocks everything!
    return "done"

# Good - yields control
async def non_blocking_operation():
    await asyncio.sleep(5)  # Yields control to event loop
    return "done"
```

### 5. Use asyncio.gather() for Concurrency
```python
async def concurrent_requests():
    async with httpx.AsyncClient() as client:
        # All requests run concurrently
        responses = await asyncio.gather(
            client.get("/endpoint1"),
            client.get("/endpoint2"),
            client.get("/endpoint3"),
            return_exceptions=True  # Don't fail fast
        )
        return responses
```

### 6. Proper Event Loop Management
```python
# Good - for simple scripts
async def main():
    # Your async code here
    pass

if __name__ == "__main__":
    asyncio.run(main())

# Good - for more complex scenarios
async def main():
    # Your async code here
    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
```

## Summary

Async/await in Python provides a powerful way to write concurrent code that can handle multiple I/O operations efficiently. The key concepts are:

- **Coroutines** can be suspended and resumed
- **Event loop** manages execution and scheduling
- **await** points are where control can be yielded
- **Concurrency** comes from running multiple operations together
- **Performance** gains are significant for I/O-bound operations

The pattern you see in your test code is a perfect example of async/await in action, where HTTP requests can be made efficiently without blocking the entire program. 