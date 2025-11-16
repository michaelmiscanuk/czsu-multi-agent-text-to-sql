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
            "https://httpbin.org/delay/5",
            "https://httpbin.org/delay/2",
            "https://httpbin.org/delay/1",
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
