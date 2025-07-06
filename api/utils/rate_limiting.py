# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Constants
try:
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import time
import asyncio
from collections import defaultdict

# Import rate limiting globals from api.config.settings
from api.config.settings import (
    rate_limit_storage,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    RATE_LIMIT_BURST,
    RATE_LIMIT_MAX_WAIT
)

def check_rate_limit_with_throttling(client_ip: str) -> dict:
    """Check rate limits and return throttling information instead of boolean."""
    now = time.time()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check burst limit (last 10 seconds)
    recent_requests = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < 10
    ]
    
    # Check window limit
    window_requests = len(rate_limit_storage[client_ip])
    
    # Calculate suggested wait time based on current load
    suggested_wait = 0
    
    if len(recent_requests) >= RATE_LIMIT_BURST:
        # Burst limit exceeded - calculate wait time until oldest burst request expires
        oldest_burst = min(recent_requests)
        suggested_wait = max(0, 10 - (now - oldest_burst))
    elif window_requests >= RATE_LIMIT_REQUESTS:
        # Window limit exceeded - calculate wait time until oldest request expires
        oldest_window = min(rate_limit_storage[client_ip])
        suggested_wait = max(0, RATE_LIMIT_WINDOW - (now - oldest_window))
    
    return {
        "allowed": len(recent_requests) < RATE_LIMIT_BURST and window_requests < RATE_LIMIT_REQUESTS,
        "suggested_wait": min(suggested_wait, RATE_LIMIT_MAX_WAIT),
        "burst_count": len(recent_requests),
        "window_count": window_requests,
        "burst_limit": RATE_LIMIT_BURST,
        "window_limit": RATE_LIMIT_REQUESTS
    }

async def wait_for_rate_limit(client_ip: str) -> bool:
    """Wait for rate limit to allow request, with maximum wait time."""
    # Import debug function
    from api.utils.debug import print__debug
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        rate_info = check_rate_limit_with_throttling(client_ip)
        
        if rate_info["allowed"]:
            # Add current request to tracking
            rate_limit_storage[client_ip].append(time.time())
            return True
        
        if rate_info["suggested_wait"] <= 0:
            # Should be allowed but isn't - might be a race condition
            await asyncio.sleep(0.1)
            continue
            
        if rate_info["suggested_wait"] > RATE_LIMIT_MAX_WAIT:
            # Wait time too long, give up
            print__debug(f"⚠️ Rate limit wait time ({rate_info['suggested_wait']:.1f}s) exceeds maximum ({RATE_LIMIT_MAX_WAIT}s) for {client_ip}")
            return False
            
        # Wait for the suggested time
        print__debug(f"⏳ Throttling request from {client_ip}: waiting {rate_info['suggested_wait']:.1f}s (burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, window: {rate_info['window_count']}/{rate_info['window_limit']}, attempt {attempt + 1})")
        await asyncio.sleep(rate_info["suggested_wait"])
    
    print__debug(f"❌ Rate limit exceeded after {max_attempts} attempts for {client_ip}")
    return False

def check_rate_limit(client_ip: str) -> bool:
    """Check if client IP is within rate limits."""
    now = time.time()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check burst limit (last 10 seconds)
    recent_requests = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < 10
    ]
    
    if len(recent_requests) >= RATE_LIMIT_BURST:
        return False
    
    # Check window limit
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True 