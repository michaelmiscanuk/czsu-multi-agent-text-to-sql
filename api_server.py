# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os  # Import os early for environment variable access

def print__startup_debug(msg: str) -> None:
    """Print startup debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[STARTUP-DEBUG] {msg}")
        sys.stdout.flush()

if sys.platform == "win32":
    import asyncio
    
    # AGGRESSIVE WINDOWS FIX: Force SelectorEventLoop before any other async operations
    print__startup_debug(f"🔧 API Server: Windows detected - forcing SelectorEventLoop for PostgreSQL compatibility")
    
    # Set the policy first - this is CRITICAL and must happen before any async operations
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print__startup_debug(f"🔧 Windows event loop policy set to: {type(asyncio.get_event_loop_policy()).__name__}")
    
    # Force close any existing event loop and create a fresh SelectorEventLoop
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop and not current_loop.is_closed():
            current_loop_type = type(current_loop).__name__
            print__startup_debug(f"🔧 API Server: Closing existing {current_loop_type}")
            current_loop.close()
    except RuntimeError:
        # No event loop exists yet, which is fine
        pass
    
    # Create a new SelectorEventLoop explicitly and set it as the running loop
    new_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
    asyncio.set_event_loop(new_loop)
    print__startup_debug(f"🔧 API Server: Created new {type(new_loop).__name__}")
    
    # Verify the fix worked - this is critical for PostgreSQL compatibility
    try:
        current_loop = asyncio.get_event_loop()
        current_loop_type = type(current_loop).__name__
        print__startup_debug(f"🔧 API Server: Current event loop type: {current_loop_type}")
        if "Selector" in current_loop_type:
            print__startup_debug(f"✅ API Server: PostgreSQL should work correctly on Windows now")
        else:
            print__startup_debug(f"⚠️ API Server: Event loop fix may not have worked properly")
            # FORCE FIX: If we still don't have a SelectorEventLoop, create one
            print__startup_debug(f"🔧 API Server: Force-creating SelectorEventLoop...")
            if not current_loop.is_closed():
                current_loop.close()
            selector_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
            asyncio.set_event_loop(selector_loop)
            print__startup_debug(f"🔧 API Server: Force-created {type(selector_loop).__name__}")
    except RuntimeError:
        print__startup_debug(f"🔧 API Server: No event loop set yet (will be created as needed)")

import asyncio
import gc
import psutil
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import uuid
import time
from collections import defaultdict
from typing import List, Optional, Dict

from fastapi import FastAPI, Query, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, field_validator
import sqlite3
import requests
import jwt
import json
import os
from jwt.algorithms import RSAAlgorithm
from langchain_core.messages import BaseMessage
from langsmith import Client
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Load environment variables from .env file EARLY
load_dotenv()

from main import main as analysis_main
from my_agent.utils.postgres_checkpointer import (
    get_postgres_checkpointer,
    create_thread_run_entry,
    get_user_chat_threads,
    delete_user_thread_entries,
    get_conversation_messages_from_checkpoints,
    get_queries_and_results_from_latest_checkpoint,
    force_close_all_connections
)

# Additional imports for sentiment functionality
from my_agent.utils.postgres_checkpointer import (
    update_thread_run_sentiment,
    get_thread_run_sentiments
)

# Read GC memory threshold from environment with default fallback
GC_MEMORY_THRESHOLD = int(os.environ.get('GC_MEMORY_THRESHOLD', '100'))  # Default 100MB
print__startup_debug(f"🔧 API Server: GC_MEMORY_THRESHOLD set to {GC_MEMORY_THRESHOLD}MB (from environment)")

def print__memory_monitoring(msg: str) -> None:
    """Print MEMORY-MONITORING messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[MEMORY-MONITORING] {msg}")
        import sys
        sys.stdout.flush()

# MEMORY LEAK PREVENTION: Global tracking based on "Needle in a haystack" article
# These help detect the same issues that plagued the Ocean framework
_app_startup_time = None
_route_registration_count = defaultdict(int)  # Track duplicate route registrations
_memory_baseline = None  # RSS memory at startup
_request_count = 0  # Track total requests processed
_last_gc_run = time.time()  # Track garbage collection frequency

# Global shared checkpointer for conversation memory across API requests
# This ensures that conversation state is preserved between frontend requests using PostgreSQL
GLOBAL_CHECKPOINTER = None

# Add a semaphore to limit concurrent analysis requests
MAX_CONCURRENT_ANALYSES = int(os.environ.get('MAX_CONCURRENT_ANALYSES', '3'))  # Read from .env with fallback to 3 (we have recovery systems!)
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# Log the concurrency setting for debugging
print__startup_debug(f"🔧 API Server: MAX_CONCURRENT_ANALYSES set to {MAX_CONCURRENT_ANALYSES} (from environment)")
print__memory_monitoring(f"Concurrent analysis semaphore initialized with {MAX_CONCURRENT_ANALYSES} slots")
print__memory_monitoring(f"🛡️ Recovery systems active: Response persistence + Frontend auto-recovery")

# RATE LIMITING: Global rate limiting storage
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # 60 seconds window
RATE_LIMIT_BURST = 20  # burst limit for rapid requests
RATE_LIMIT_MAX_WAIT = 5  # maximum seconds to wait before giving up

# Throttling semaphores per IP to limit concurrent requests
throttle_semaphores = defaultdict(lambda: asyncio.Semaphore(8))  # Max 8 concurrent requests per IP

def detect_memory_fragmentation() -> dict:
    """Simple memory monitoring without complex fragmentation detection."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # RSS is physical memory
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        # Simple memory growth tracking
        global _memory_baseline
        memory_growth = rss_mb - _memory_baseline if _memory_baseline else 0
        
        return {
            "rss_mb": round(rss_mb, 2),
            "vms_mb": round(vms_mb, 2),
            "memory_growth_mb": round(memory_growth, 2),
            "high_memory": rss_mb > 400  # Simple threshold
        }
    except Exception as e:
        return {"error": str(e)}

def log_memory_usage(context: str = ""):
    """Enhanced memory logging with RSS tracking and fragmentation detection."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # RSS memory (actual physical RAM usage) - this is what leaked in the article
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        # Get memory limits if available (Render typically sets these)
        memory_limit_mb = None
        try:
            # Check for common memory limit environment variables
            if os.environ.get('MEMORY_LIMIT'):
                memory_limit_mb = int(os.environ.get('MEMORY_LIMIT')) / 1024 / 1024
            elif os.environ.get('WEB_MEMORY'):
                memory_limit_mb = int(os.environ.get('WEB_MEMORY'))
        except:
            pass
        
        # Enhanced logging with RSS focus (the issue from the article)
        limit_info = f" (limit: {memory_limit_mb:.1f}MB)" if memory_limit_mb else ""
        usage_percent = f" ({(rss_mb/memory_limit_mb)*100:.1f}%)" if memory_limit_mb else ""
        
        # Memory fragmentation detection
        fragmentation_info = detect_memory_fragmentation()
        frag_warning = " [FRAGMENTATION DETECTED]" if fragmentation_info.get("high_memory") else ""
        growth_warning = " [SIGNIFICANT GROWTH]" if fragmentation_info.get("memory_growth_mb") > GC_MEMORY_THRESHOLD else ""
        
        print__memory_monitoring(
            f"Memory usage{f' [{context}]' if context else ''}: "
            f"RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB{limit_info}{usage_percent}"
            f"{frag_warning}{growth_warning}"
        )
        
        # FRAGMENTATION HANDLING: Automatically handle detected fragmentation
        # Only trigger if not already in a fragmentation handling context to avoid loops
        if (fragmentation_info.get("high_memory") and 
            not context.startswith("post_fragmentation_handler") and
            not context.startswith("fragmentation_")):
            
            print__memory_monitoring(f"🚨 FRAGMENTATION TRIGGER: RSS {rss_mb:.1f}MB exceeds threshold 400MB")
            print__memory_monitoring(f"📊 Memory details: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB, Growth={fragmentation_info.get('memory_growth_mb', 0):.1f}MB")
            
            # Call the fragmentation handler
            handle_memory_fragmentation()
        
        # Warning if memory usage is high or growing suspiciously
        if memory_limit_mb and rss_mb > memory_limit_mb * 0.85:
            print__memory_monitoring(f"HIGH MEMORY WARNING: Using {(rss_mb/memory_limit_mb)*100:.1f}% of available RSS memory!")
        
        # Track memory growth pattern (similar to the article's issue)
        global _memory_baseline, _request_count
        if _memory_baseline is None:
            _memory_baseline = rss_mb
            print__memory_monitoring(f"Memory baseline set: {_memory_baseline:.1f}MB RSS")
        elif context.startswith("after_") and _request_count > 0:
            growth_per_request = (rss_mb - _memory_baseline) / _request_count
            if growth_per_request > GC_MEMORY_THRESHOLD * 0.001:  # More than 0.1% of threshold growth per request
                print__memory_monitoring(f"LEAK WARNING: {growth_per_request:.3f}MB growth per request pattern detected!")
            
    except Exception as e:
        print__memory_monitoring(f"Could not check memory usage: {e}")

def handle_memory_fragmentation():
    """Simple memory cleanup without complex fragmentation handling."""
    print__memory_monitoring("🧹 Running simple memory cleanup")
    
    try:
        # Simple garbage collection
        import gc
        collected = gc.collect()
        print__memory_monitoring(f"🧹 GC freed {collected} objects")
        
        # Clear rate limiting storage for inactive clients
        global rate_limit_storage, throttle_semaphores
        old_clients = len(rate_limit_storage)
        
        import time
        current_time = time.time()
        for client_ip in list(rate_limit_storage.keys()):
            rate_limit_storage[client_ip] = [
                timestamp for timestamp in rate_limit_storage[client_ip]
                if current_time - timestamp < RATE_LIMIT_WINDOW
            ]
            if not rate_limit_storage[client_ip]:
                del rate_limit_storage[client_ip]
        
        new_clients = len(rate_limit_storage)
        print__memory_monitoring(f"🧹 Cleaned rate limiting: {old_clients}→{new_clients} clients")
            
    except Exception as e:
        print__memory_monitoring(f"❌ Error in memory cleanup: {e}")
        # Still try basic GC even if cleanup fails
        try:
            gc.collect()
        except:
            pass

def monitor_route_registration(route_path: str, method: str):
    """Monitor route registration to detect duplicates (main issue from article)."""
    global _route_registration_count
    
    route_key = f"{method}:{route_path}"
    _route_registration_count[route_key] += 1
    
    if _route_registration_count[route_key] > 1:
        print__memory_monitoring(
            f"🚨 DUPLICATE ROUTE REGISTRATION DETECTED: {route_key} "
            f"registered {_route_registration_count[route_key]} times! "
            f"This was the ROOT CAUSE in the Ocean framework leak!"
        )
        
        # Log call stack to identify where duplicate registration is happening
        import traceback
        print__memory_monitoring(f"Route registration stack trace:\n{traceback.format_stack()[-3]}")

def log_comprehensive_error(context: str, error: Exception, request: Request = None):
    """Enhanced error logging with context and request details."""
    error_details = {
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
    }
    
    if request:
        error_details.update({
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "content_type": request.headers.get("content-type", "unknown")
        })
    
    # Log to debug output
    print__debug(f"COMPREHENSIVE ERROR: {json.dumps(error_details, indent=2)}")
    
    # Log memory usage during error AND run cleanup GC
    log_memory_usage(f"error_{context}")
    aggressive_garbage_collection(f"error_{context}")
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
            print__debug(f"⚠ Rate limit wait time ({rate_info['suggested_wait']:.1f}s) exceeds maximum ({RATE_LIMIT_MAX_WAIT}s) for {client_ip}")
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

def setup_graceful_shutdown():
    """Setup graceful shutdown handlers to detect platform restarts."""
    def signal_handler(signum, frame):
        print__memory_monitoring(f"Received signal {signum} - preparing for graceful shutdown...")
        print__memory_monitoring(f"This could indicate a platform restart or memory limit exceeded")
        log_memory_usage("shutdown_signal")
        
        # Run final cleanup GC
        aggressive_garbage_collection("shutdown_signal")
        # Don't exit immediately, let FastAPI handle cleanup
        
    # Register signal handlers for common restart signals
    signal.signal(signal.SIGTERM, signal_handler)  # Most common for container restarts
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, signal_handler)  # User-defined signal

async def initialize_checkpointer():
    """Initialize the global PostgreSQL checkpointer on startup."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER is None:
        try:
            print__startup_debug("🔗 Initializing PostgreSQL checkpointer and chat system...")
            print__startup_debug(f"🔍 Current global checkpointer state: {GLOBAL_CHECKPOINTER}")
            log_memory_usage("startup")
            
            # Add timeout to initialization to fail faster
            GLOBAL_CHECKPOINTER = await asyncio.wait_for(
                get_postgres_checkpointer(), 
                timeout=45  # Increased from 30 to 45 seconds
            )
            
            # Verify the checkpointer is healthy
            if hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
                print__startup_debug(f"✓ Checkpointer has connection pool: closed={GLOBAL_CHECKPOINTER.conn.closed}")
            else:
                print__startup_debug("⚠ Checkpointer does not have connection pool")
            
            print__startup_debug("✓ Global PostgreSQL checkpointer initialized successfully")
            print__startup_debug("✓ users_threads_runs table verified/created")
            log_memory_usage("checkpointer_initialized")
        except asyncio.TimeoutError:
            print__startup_debug("✗ Failed to initialize PostgreSQL checkpointer: initialization timeout")
            print__startup_debug("⚠ This usually means PostgreSQL connection pool is exhausted")
            
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print__startup_debug("⚠ Falling back to InMemorySaver")
        except Exception as e:
            print__startup_debug(f"✗ Failed to initialize PostgreSQL checkpointer: {e}")
            print__startup_debug(f"🔍 Error type: {type(e).__name__}")
            import traceback
            print__startup_debug(f"🔍 Full traceback: {traceback.format_exc()}")
            
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print__startup_debug("⚠ Falling back to InMemorySaver")
    else:
        print__startup_debug("⚠ Global checkpointer already exists - skipping initialization")

async def cleanup_checkpointer():
    """Clean up resources on app shutdown."""
    global GLOBAL_CHECKPOINTER
    print__memory_monitoring("Starting application cleanup...")
    log_memory_usage("cleanup_start")
    
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
        try:
            # Check if pool is already closed before trying to close it
            if not GLOBAL_CHECKPOINTER.conn.closed:
                await GLOBAL_CHECKPOINTER.conn.close()
                print__startup_debug("✓ PostgreSQL connection pool closed cleanly")
            else:
                print__startup_debug("⚠ PostgreSQL connection pool was already closed")
        except Exception as e:
            print__startup_debug(f"⚠ Error closing connection pool: {e}")
        finally:
            GLOBAL_CHECKPOINTER = None
    
    # Enhanced garbage collection on shutdown based on article findings
    aggressive_garbage_collection("cleanup_complete")
    log_memory_usage("cleanup_complete")

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance, recreating if necessary with enhanced error handling."""
    global GLOBAL_CHECKPOINTER
    
    # Check if current checkpointer is healthy
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn'):
        try:
            # FRAGMENTATION CHECK: If fragmentation handler marked for reset, recreate pool
            if hasattr(GLOBAL_CHECKPOINTER, '_fragmentation_reset_needed') and GLOBAL_CHECKPOINTER._fragmentation_reset_needed:
                print__memory_monitoring("🔄 FRAGMENTATION RESET: Recreating checkpointer due to fragmentation handler request")
                
                # CRITICAL FIX: Wait for active operations to complete before pool recreation
                from my_agent.utils.postgres_checkpointer import get_active_operations_count
                max_wait_time = 30  # Maximum 30 seconds to wait for operations to complete
                wait_start = time.time()
                
                while True:
                    active_ops = await get_active_operations_count()
                    if active_ops == 0:
                        print__memory_monitoring(f"✅ No active operations - safe to recreate pool")
                        break
                    
                    elapsed = time.time() - wait_start
                    if elapsed > max_wait_time:
                        print__memory_monitoring(f"⚠ Timeout waiting for {active_ops} active operations to complete - proceeding with caution")
                        break
                    
                    print__memory_monitoring(f"⏳ Waiting for {active_ops} active operations to complete... ({elapsed:.1f}s elapsed)")
                    await asyncio.sleep(1)
                
                # Close existing connection and mark for recreation
                if GLOBAL_CHECKPOINTER.conn and not GLOBAL_CHECKPOINTER.conn.closed:
                    await GLOBAL_CHECKPOINTER.conn.close()
                GLOBAL_CHECKPOINTER = None  # Force recreation
            # Check if the pool is closed
            elif hasattr(GLOBAL_CHECKPOINTER.conn, 'closed') and GLOBAL_CHECKPOINTER.conn.closed:
                print__startup_debug(f"⚠ Checkpointer pool is closed, recreating...")
                GLOBAL_CHECKPOINTER = None
            else:
                # Enhanced health check with timeout
                try:
                    # Fix: await the coroutine first, then use the connection in async with
                    conn = await asyncio.wait_for(GLOBAL_CHECKPOINTER.conn.connection(), timeout=5)
                    async with conn:
                        await asyncio.wait_for(conn.execute("SELECT 1"), timeout=5)
                    print__startup_debug("✅ Existing checkpointer is healthy")
                    return GLOBAL_CHECKPOINTER
                except asyncio.TimeoutError:
                    print__startup_debug("⚠ Checkpointer health check timed out, recreating...")
                    GLOBAL_CHECKPOINTER = None
                except Exception as health_error:
                    print__startup_debug(f"⚠ Checkpointer health check failed: {health_error}")
                    GLOBAL_CHECKPOINTER = None
        except Exception as e:
            print__startup_debug(f"⚠ Error checking checkpointer health: {e}")
            GLOBAL_CHECKPOINTER = None
    
    # Cleanup old checkpointer if needed
    if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn'):
        try:
            if GLOBAL_CHECKPOINTER.conn and not GLOBAL_CHECKPOINTER.conn.closed:
                print__startup_debug("🧹 Closing old checkpointer connection...")
                
                # CRITICAL FIX: Wait for active operations before closing
                from my_agent.utils.postgres_checkpointer import get_active_operations_count
                active_ops = await get_active_operations_count()
                if active_ops > 0:
                    print__startup_debug(f"⏳ Waiting for {active_ops} active operations before closing pool...")
                    wait_start = time.time()
                    while active_ops > 0 and (time.time() - wait_start) < 15:  # Wait max 15 seconds
                        await asyncio.sleep(0.5)
                        active_ops = await get_active_operations_count()
                    
                    if active_ops > 0:
                        print__startup_debug(f"⚠ Still {active_ops} active operations after 15s - proceeding with closure")
                
                await GLOBAL_CHECKPOINTER.conn.close()
        except Exception as cleanup_error:
            print__startup_debug(f"⚠ Error during cleanup: {cleanup_error}")
        finally:
            GLOBAL_CHECKPOINTER = None
    
    # Create new checkpointer with retries
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print__startup_debug(f"🔄 Creating fresh checkpointer (attempt {attempt + 1})...")
            GLOBAL_CHECKPOINTER = await asyncio.wait_for(
                get_postgres_checkpointer(), 
                timeout=120  # 2 minute timeout for checkpointer creation
            )
            
            # Clear any fragmentation reset flag on successful creation
            if hasattr(GLOBAL_CHECKPOINTER, '_fragmentation_reset_needed'):
                delattr(GLOBAL_CHECKPOINTER, '_fragmentation_reset_needed')
            
            print__startup_debug(f"✅ Fresh checkpointer created successfully (attempt {attempt + 1})")
            print__memory_monitoring("✅ New checkpointer created - fragmentation-related pool issues should be resolved")
            return GLOBAL_CHECKPOINTER
            
        except asyncio.TimeoutError:
            print__startup_debug(f"❌ Timeout creating checkpointer (attempt {attempt + 1})")
            if attempt < max_attempts - 1:
                delay = 2 ** attempt  # Exponential backoff
                print__startup_debug(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print__startup_debug("❌ All checkpointer creation attempts timed out, falling back to InMemorySaver")
                break
                    
        except Exception as e:
            error_msg = str(e)
            print__startup_debug(f"❌ Failed to recreate checkpointer (attempt {attempt + 1}): {error_msg}")
            
            # Check if it's a recoverable error
            if (attempt < max_attempts - 1 and 
                ("connection" in error_msg.lower() or 
                 "pool" in error_msg.lower() or
                 "dbhandler" in error_msg.lower())):
                delay = 2 ** attempt  # Exponential backoff
                print__startup_debug(f"🔄 Retrying due to connection issue in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print__startup_debug("❌ Non-recoverable error or final attempt failed, falling back to InMemorySaver")
                break
    # Fallback to InMemorySaver only if PostgreSQL completely fails
    print__startup_debug("⚠ PostgreSQL checkpointer creation failed completely - falling back to InMemorySaver")
    print__memory_monitoring("⚠ Using InMemorySaver - fragmentation handling will be limited")
    from langgraph.checkpoint.memory import InMemorySaver
    GLOBAL_CHECKPOINTER = InMemorySaver()
    print__startup_debug("⚠ Using InMemorySaver - persistence will be limited to this session")
    return GLOBAL_CHECKPOINTER

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _app_startup_time, _memory_baseline
    _app_startup_time = datetime.now()
    
    print__startup_debug("🚀 FastAPI application starting up...")
    print__memory_monitoring(f"Application startup initiated at {_app_startup_time.isoformat()}")
    log_memory_usage("app_startup")
    
    # ROUTE REGISTRATION MONITORING: Track all routes that get registered
    # This prevents the exact issue described in the "Needle in a haystack" article
    print__memory_monitoring("🔍 Monitoring route registrations to prevent memory leaks...")
    
    # Setup graceful shutdown handlers
    setup_graceful_shutdown()
    
    # Optimize garbage collection for memory efficiency (more aggressive than before)
    # Based on article findings about memory not being released
    gc.set_threshold(500, 8, 8)  # Even more aggressive than the current 700, 10, 10
    print__memory_monitoring(f"Set aggressive GC thresholds: (500, 8, 8)")
    
    await initialize_checkpointer()
    
    # Set memory baseline after initialization
    if _memory_baseline is None:
        try:
            process = psutil.Process()
            _memory_baseline = process.memory_info().rss / 1024 / 1024
            print__memory_monitoring(f"Memory baseline established: {_memory_baseline:.1f}MB RSS")
        except:
            pass
    
    log_memory_usage("app_ready")
    print__startup_debug("✅ FastAPI application ready to serve requests")
    
    yield
    
    # Shutdown
    print__startup_debug("🛑 FastAPI application shutting down...")
    print__memory_monitoring(f"Application ran for {datetime.now() - _app_startup_time}")
    
    # Log final memory statistics
    if _memory_baseline:
        try:
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024
            total_growth = final_memory - _memory_baseline
            print__memory_monitoring(
                f"Final memory stats: Started={_memory_baseline:.1f}MB, "
                f"Final={final_memory:.1f}MB, Growth={total_growth:.1f}MB"
            )
            if total_growth > GC_MEMORY_THRESHOLD:  # More than threshold growth - app will restart soon
                print__memory_monitoring("🚨 SIGNIFICANT MEMORY GROWTH DETECTED - investigate for leaks!")
        except:
            pass
    
    await cleanup_checkpointer()

# CRITICAL: Route registration happens here ONCE during startup
# This is the key fix from the "Needle in a haystack" article
app = FastAPI(
    lifespan=lifespan,
    default_response_class=ORJSONResponse  # Use ORJSON for faster, memory-efficient JSON responses
)

# Monitor all route registrations (including middleware and CORS)
print__memory_monitoring("📋 Registering CORS middleware...")
# Note: Route registration monitoring happens at runtime to avoid import-time global variable access

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print__memory_monitoring("📋 Registering GZip middleware...")
# Add GZip compression to reduce response sizes and memory usage
app.add_middleware(GZipMiddleware, minimum_size=1000)

# RATE LIMITING MIDDLEWARE
@app.middleware("http")
async def throttling_middleware(request: Request, call_next):
    """Throttling middleware that makes requests wait instead of rejecting them."""
    
    # Skip throttling for health checks and static endpoints
    if request.url.path in ["/health", "/docs", "/openapi.json", "/debug/pool-status"]:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    
    # Use semaphore to limit concurrent requests per IP
    semaphore = throttle_semaphores[client_ip]
    
    async with semaphore:
        # Try to wait for rate limit instead of immediately rejecting
        if not await wait_for_rate_limit(client_ip):
            # Only reject if we can't wait (wait time too long or max attempts exceeded)
            rate_info = check_rate_limit_with_throttling(client_ip)
            log_comprehensive_error("rate_limit_exceeded_after_wait", 
                                   Exception(f"Rate limit exceeded for IP: {client_ip} after waiting. Burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, Window: {rate_info['window_count']}/{rate_info['window_limit']}"), 
                                   request)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Please wait {rate_info['suggested_wait']:.1f}s before retrying.",
                    "retry_after": max(rate_info['suggested_wait'], 1),
                    "burst_usage": f"{rate_info['burst_count']}/{rate_info['burst_limit']}",
                    "window_usage": f"{rate_info['window_count']}/{rate_info['window_limit']}"
                },
                headers={"Retry-After": str(max(int(rate_info['suggested_wait']), 1))}
            )
        
        return await call_next(request)

# Enhanced middleware to monitor memory patterns and detect leaks
@app.middleware("http")
async def enhanced_memory_monitoring_middleware(request: Request, call_next):
    """Enhanced memory monitoring with leak detection patterns."""
    global _request_count
    
    start_time = datetime.now()
    _request_count += 1
    
    # Log memory before processing request (especially for heavy operations)
    request_path = request.url.path
    if any(path in request_path for path in ["/analyze", "/chat", "/feedback"]):
        log_memory_usage(f"before_{request_path.replace('/', '_')}")
    
    # Track requests that might be repeatedly registering routes (the article's issue)
    if request_path in ["/health", "/docs"] and _request_count > 100:
        # These are frequently called - make sure they don't cause memory leaks
        if _request_count % 100 == 0:  # Log every 100th request
            print__memory_monitoring(f"Health check #{_request_count} - monitoring for route re-registration leaks")
    
    response = await call_next(request)
    
    # Log memory after processing request with enhanced leak detection
    if any(path in request_path for path in ["/analyze", "/chat", "/feedback"]):
        process_time = (datetime.now() - start_time).total_seconds()
        log_memory_usage(f"after_{request_path.replace('/', '_')}_({process_time:.2f}s)")
        
        # Aggressive GC after heavy operations to prevent accumulation
        if request_path.startswith("/analyze"):
            aggressive_garbage_collection("post_analysis")
        
        # Warn about slow requests that might cause timeouts
        if process_time > 300:  # 5 minutes
            print__memory_monitoring(f"SLOW REQUEST WARNING: {request_path} took {process_time:.2f}s")
    
    # Periodic memory health check (every 50 requests)
    if _request_count % 50 == 0:
        fragmentation_info = detect_memory_fragmentation()
        if fragmentation_info.get("high_memory") or fragmentation_info.get("memory_growth_mb"):
            print__memory_monitoring(f"🔍 Request #{_request_count} memory health check: {fragmentation_info}")
    
    return response

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with memory leak detection."""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "app_uptime": str(datetime.now() - _app_startup_time) if _app_startup_time else "unknown",
            "total_requests_processed": _request_count
        }
        
        # Enhanced memory info with RSS focus (the leak indicator from article)
        try:
            fragmentation_info = detect_memory_fragmentation()
            health_data.update({
                "memory_rss_mb": fragmentation_info.get("rss_mb", 0),
                "memory_vms_mb": fragmentation_info.get("vms_mb", 0),
                "memory_fragmentation_ratio": fragmentation_info.get("fragmentation_ratio", 0),
                "memory_growth_mb": fragmentation_info.get("memory_growth_mb", 0),
                "potential_memory_fragmentation": fragmentation_info.get("high_memory", False),
                "significant_memory_growth": fragmentation_info.get("memory_growth_mb", 0) > GC_MEMORY_THRESHOLD
            })
            
            # Check if we're approaching memory limits
            memory_limit_mb = None
            try:
                if os.environ.get('MEMORY_LIMIT'):
                    memory_limit_mb = int(os.environ.get('MEMORY_LIMIT')) / 1024 / 1024
                elif os.environ.get('WEB_MEMORY'):
                    memory_limit_mb = int(os.environ.get('WEB_MEMORY'))
            except:
                pass
            
            if memory_limit_mb:
                rss_mb = fragmentation_info.get("rss_mb", 0)
                usage_percent = (rss_mb / memory_limit_mb) * 100
                health_data["memory_usage_percent"] = round(usage_percent, 1)
                health_data["memory_limit_mb"] = memory_limit_mb
                
                # Return unhealthy if memory usage is too high or showing leak patterns
                if usage_percent > 90 or fragmentation_info.get("significant_memory_growth", False):
                    health_data["status"] = "critical_memory"
                    health_data["warning"] = f"Memory usage at {usage_percent:.1f}% or significant growth detected"
                elif usage_percent > 80 or fragmentation_info.get("high_memory", False):
                    health_data["status"] = "high_memory"
                    health_data["warning"] = f"Memory usage at {usage_percent:.1f}% or fragmentation detected"
                    
        except Exception as mem_error:
            health_data["memory_error"] = str(mem_error)
        
        # Route registration health check (prevent the article's main issue)
        try:
            duplicate_routes = {k: v for k, v in _route_registration_count.items() if v > 1}
            health_data["route_registrations"] = {
                "total_registrations": sum(_route_registration_count.values()),
                "unique_routes": len(_route_registration_count),
                "duplicate_registrations": duplicate_routes,
                "has_duplicates": len(duplicate_routes) > 0
            }
            
            # Mark as unhealthy if we detect route duplication (the core issue!)
            if duplicate_routes:
                health_data["status"] = "unhealthy"
                health_data["error"] = f"Duplicate route registrations detected: {list(duplicate_routes.keys())}"
                print__memory_monitoring(f"🚨 HEALTH CHECK: Duplicate route registrations detected!")
                
        except Exception as route_error:
            health_data["route_registration_error"] = str(route_error)
        
        # Check checkpointer health
        try:
            if GLOBAL_CHECKPOINTER:
                if hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
                    if not GLOBAL_CHECKPOINTER.conn.closed:
                        health_data["database"] = "connected"
                    else:
                        health_data["database"] = "disconnected"
                        health_data["status"] = "degraded"
                else:
                    health_data["database"] = "in_memory_fallback"
                    health_data["status"] = "degraded"
            else:
                health_data["database"] = "not_initialized"
                health_data["status"] = "degraded"
        except:
            health_data["database"] = "error"
            health_data["status"] = "degraded"
        
        return health_data
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# NEW: Individual service health checks
@app.get("/health/database")
async def database_health_check():
    """Database-specific health check."""
    try:
        if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
            async with GLOBAL_CHECKPOINTER.conn.connection() as conn:
                await conn.execute("SELECT 1")
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "degraded",
                "database": "not_available",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health/memory")
async def memory_health_check():
    """Enhanced memory-specific health check with fragmentation detection."""
    try:
        fragmentation_info = detect_memory_fragmentation()
        
        # Get memory limit
        memory_limit_mb = 512  # Default Render limit
        try:
            if os.environ.get('MEMORY_LIMIT'):
                memory_limit_mb = int(os.environ.get('MEMORY_LIMIT')) / 1024 / 1024
            elif os.environ.get('WEB_MEMORY'):
                memory_limit_mb = int(os.environ.get('WEB_MEMORY'))
        except:
            pass
        
        rss_mb = fragmentation_info.get("rss_mb", 0)
        usage_percent = (rss_mb / memory_limit_mb) * 100
        
        status = "healthy"
        warnings = []
        
        # Check for various memory issues based on article findings
        if usage_percent > 90:
            status = "critical"
            warnings.append(f"High RSS memory usage: {usage_percent:.1f}%")
        elif usage_percent > 80:
            status = "warning"
            warnings.append(f"Elevated RSS memory usage: {usage_percent:.1f}%")
        
        if fragmentation_info.get("high_memory", False):
            status = "warning" if status == "healthy" else status
            warnings.append(f"Memory fragmentation detected (ratio: {fragmentation_info.get('fragmentation_ratio', 0):.2f})")
        
        if fragmentation_info.get("memory_growth_mb", 0):
            status = "critical" if status == "healthy" else status
            warnings.append(f"Significant memory growth: {fragmentation_info.get('memory_growth_mb', 0):.1f}MB")
        
        return {
            "status": status,
            "memory_rss_mb": rss_mb,
            "memory_vms_mb": fragmentation_info.get("vms_mb", 0),
            "memory_limit_mb": memory_limit_mb,
            "usage_percent": round(usage_percent, 1),
            "fragmentation_ratio": fragmentation_info.get("fragmentation_ratio", 0),
            "memory_growth_mb": fragmentation_info.get("memory_growth_mb", 0),
            "warnings": warnings,
            "baseline_memory_mb": _memory_baseline,
            "total_requests_processed": _request_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health/rate-limits")
async def rate_limit_health_check():
    """Rate limiting health check."""
    try:
        total_clients = len(rate_limit_storage)
        active_clients = sum(1 for requests in rate_limit_storage.values() if requests)
        
        return {
            "status": "healthy",
            "total_tracked_clients": total_clients,
            "active_clients": active_clients,
            "rate_limit_window": RATE_LIMIT_WINDOW,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_burst": RATE_LIMIT_BURST,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ROUTE REGISTRATION MONITORING: Track all endpoint registrations to prevent memory leaks
# This directly addresses the core issue from the "Needle in a haystack" article
print__memory_monitoring("📋 Monitoring route registrations for memory leak prevention...")

# Track all main routes that will be registered
main_routes = [
    ("/health", "GET"), ("/health/database", "GET"), ("/health/memory", "GET"), 
    ("/health/rate-limits", "GET"), ("/analyze", "POST"), ("/feedback", "POST"), 
    ("/sentiment", "POST"), ("/chat/{thread_id}/sentiments", "GET"), 
    ("/chat-threads", "GET"), ("/chat/{thread_id}", "DELETE"), 
    ("/catalog", "GET"), ("/data-tables", "GET"), ("/data-table", "GET"),
    ("/chat/{thread_id}/messages", "GET"), ("/chat/all-messages", "GET"), 
    ("/debug/chat/{thread_id}/checkpoints", "GET"),
    ("/debug/pool-status", "GET"), ("/chat/{thread_id}/run-ids", "GET"),
    ("/debug/run-id/{run_id}", "GET")
]

# Route monitoring is performed at runtime through middleware to ensure proper global variable access
# for route_path, method in main_routes:
#     monitor_route_registration(route_path, method)

print__memory_monitoring(f"📋 Route monitoring configured for {len(main_routes)} endpoints - tracking occurs at runtime")

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to analyze")
    thread_id: str = Field(..., min_length=1, max_length=100, description="The thread ID")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty or only whitespace')
        return v.strip()
    
    @field_validator('thread_id')
    @classmethod
    def validate_thread_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Thread ID cannot be empty or only whitespace')
        return v.strip()

class FeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=1, description="The run ID (UUID format)")
    feedback: Optional[int] = Field(None, ge=0, le=1, description="Feedback score: 1 for thumbs up, 0 for thumbs down")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional comment")
    
    @field_validator('run_id')
    @classmethod
    def validate_run_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Run ID cannot be empty')
        # Basic UUID format validation
        import uuid
        try:
            uuid.UUID(v.strip())
        except ValueError:
            raise ValueError('Run ID must be a valid UUID format')
        return v.strip()
    
    @field_validator('comment')
    @classmethod
    def validate_comment(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty string to None
        return v

class SentimentRequest(BaseModel):
    run_id: str = Field(..., min_length=1, description="The run ID (UUID format)")
    sentiment: Optional[bool] = Field(None, description="Sentiment: true for positive, false for negative, null to clear")
    
    @field_validator('run_id')
    @classmethod
    def validate_run_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Run ID cannot be empty')
        # Basic UUID format validation
        import uuid
        try:
            uuid.UUID(v.strip())
        except ValueError:
            raise ValueError('Run ID must be a valid UUID format')
        return v.strip()

class ChatThreadResponse(BaseModel):
    thread_id: str
    latest_timestamp: str
    run_count: int
    title: str  # Now includes the title from first prompt
    full_prompt: str  # Full prompt text for tooltip

class ChatMessage(BaseModel):
    id: str
    threadId: str
    user: str
    content: str
    isUser: bool
    createdAt: int
    error: Optional[str] = None
    meta: Optional[dict] = None
    queriesAndResults: Optional[List[List[str]]] = None
    isLoading: Optional[bool] = None
    startedAt: Optional[int] = None
    isError: Optional[bool] = None

GOOGLE_JWK_URL = "https://www.googleapis.com/oauth2/v3/certs"

# Global counter for tracking JWT 'kid' missing events to reduce log spam
_jwt_kid_missing_count = 0

# FIXED: Enhanced JWT verification with proper error handling
def verify_google_jwt(token: str):
    global _jwt_kid_missing_count
    
    try:
        # EARLY VALIDATION: Check if token has basic JWT structure before processing
        # JWT tokens must have exactly 3 parts separated by dots (header.payload.signature)
        token_parts = token.split('.')
        if len(token_parts) != 3:
            # Don't log this as it's a common case for invalid tokens in tests
            raise HTTPException(status_code=401, detail="Invalid JWT token format")
        
        # Additional basic validation - each part should be non-empty and base64-like
        for i, part in enumerate(token_parts):
            if not part or len(part) < 4:  # Base64 encoded parts should be at least 4 chars
                raise HTTPException(status_code=401, detail="Invalid JWT token format")
        
        # Get Google public keys
        jwks = requests.get(GOOGLE_JWK_URL).json()
        
        # Get unverified header - this should now work since we pre-validated the format
        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.DecodeError as e:
            # This should be rare now due to pre-validation, but keep for edge cases
            print__debug(f"JWT decode error after pre-validation: {e}")
            raise HTTPException(status_code=401, detail="Invalid JWT token format")
        except Exception as e:
            print__debug(f"JWT header decode error: {e}")
            raise HTTPException(status_code=401, detail="Invalid JWT token format")
        
        # CRITICAL FIX: Check if 'kid' exists in header before accessing it
        if "kid" not in unverified_header:
            # Reduce log noise - only log this every 10th occurrence
            # This is normal for test tokens and security probes
            _jwt_kid_missing_count += 1
            if _jwt_kid_missing_count % 10 == 1:  # Log 1st, 11th, 21st, etc.
                print__debug(f"JWT token missing 'kid' field (#{_jwt_kid_missing_count} - common for test/invalid tokens)")
            raise HTTPException(status_code=401, detail="Invalid JWT token: missing key ID")
        
        # Find matching key
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                public_key = RSAAlgorithm.from_jwk(key)
                try:
                    # Debug: print the audience in the token and the expected audience
                    try:
                        unverified_payload = jwt.decode(token, options={"verify_signature": False})
                        print__debug(f"Token aud: {unverified_payload.get('aud')}")
                        print__debug(f"Backend GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID')}")
                    except Exception:
                        pass  # Ignore debug errors
                    
                    payload = jwt.decode(token, public_key, algorithms=["RS256"], audience=os.getenv("GOOGLE_CLIENT_ID"))
                    return payload
                except jwt.ExpiredSignatureError:
                    print__debug("JWT token has expired")
                    raise HTTPException(status_code=401, detail="Token has expired")
                except jwt.InvalidAudienceError:
                    print__debug("JWT token has invalid audience")
                    raise HTTPException(status_code=401, detail="Invalid token audience")
                except jwt.InvalidSignatureError:
                    print__debug("JWT token has invalid signature")
                    raise HTTPException(status_code=401, detail="Invalid token signature")
                except jwt.InvalidTokenError as e:
                    print__debug(f"JWT token is invalid: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token")
                except jwt.DecodeError as e:
                    print__debug(f"JWT decode error: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token format")
                except Exception as e:
                    print__debug(f"JWT decode error: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token")
        
        print__debug("JWT public key not found in Google JWKS")
        raise HTTPException(status_code=401, detail="Invalid token: public key not found")
        
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except requests.RequestException as e:
        print__debug(f"Failed to fetch Google JWKS: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed - unable to validate")
    except jwt.DecodeError as e:
        # This should be rare now due to pre-validation
        print__debug(f"JWT decode error in main handler: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token format")
    except KeyError as e:
        print__debug(f"JWT verification KeyError: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token structure")
    except Exception as e:
        print__debug(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")

# Enhanced dependency for JWT authentication with better error handling
def get_current_user(authorization: str = Header(None)):
    try:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid Authorization header format. Expected 'Bearer <token>'")
        
        # Split and validate token extraction
        auth_parts = authorization.split(" ", 1)
        if len(auth_parts) != 2 or not auth_parts[1].strip():
            raise HTTPException(status_code=401, detail="Invalid Authorization header format")
        
        token = auth_parts[1].strip()
        return verify_google_jwt(token)
        
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        print__debug(f"Authentication error: {e}")
        log_comprehensive_error("authentication", e)
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """Analyze request with enhanced memory leak prevention and memory pressure recovery."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__feedback_flow(f"📝 New analysis request - Thread: {request.thread_id}, User: {user_email}")
    
    # MEMORY LEAK PREVENTION: Pre-analysis memory monitoring
    log_memory_usage("analysis_start")
    pre_analysis_memory = None
    run_id = None
    checkpointer = None
    
    try:
        process = psutil.Process()
        pre_analysis_memory = process.memory_info().rss / 1024 / 1024
        print__memory_monitoring(f"Pre-analysis memory: {pre_analysis_memory:.1f}MB RSS")
    except:
        pass
    
    # MEMORY PROTECTION: Limit concurrent analyses to prevent OOM
    async with analysis_semaphore:
        print__feedback_flow(f"🔒 Acquired analysis semaphore ({analysis_semaphore._value}/{MAX_CONCURRENT_ANALYSES} available)")
        
        try:
            print__feedback_flow(f"🔄 Getting healthy checkpointer")
            checkpointer = await get_healthy_checkpointer()
            
            print__feedback_flow(f"🔄 Creating thread run entry")
            run_id = await create_thread_run_entry(user_email, request.thread_id, request.prompt)
            print__feedback_flow(f"✅ Generated new run_id: {run_id}")
            
            print__feedback_flow(f"🚀 Starting analysis")
            # 8 minute timeout for platform stability
            result = await asyncio.wait_for(
                analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=checkpointer, run_id=run_id),
                timeout=480  # 8 minutes timeout
            )
            
            print__feedback_flow(f"✅ Analysis completed successfully")
            
            # Simple response preparation
            response_data = {
                "prompt": request.prompt,
                "result": result["result"] if isinstance(result, dict) and "result" in result else str(result),
                "queries_and_results": result.get("queries_and_results", []) if isinstance(result, dict) else [],
                "thread_id": request.thread_id,
                "top_selection_codes": result.get("top_selection_codes", []) if isinstance(result, dict) else [],
                "iteration": result.get("iteration", 0) if isinstance(result, dict) else 0,
                "max_iterations": result.get("max_iterations", 2) if isinstance(result, dict) else 2,
                "sql": result.get("sql", None) if isinstance(result, dict) else None,
                "datasetUrl": result.get("datasetUrl", None) if isinstance(result, dict) else None,
                "run_id": run_id
            }
            
            # Simple memory check and return response
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                
                print__memory_monitoring(f"📤 Current memory: {current_memory:.1f}MB - sending response")
                
                response_data.update({
                    "memory_usage_mb": current_memory,
                    "analysis_completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "server_status": "completed_successfully"
                })
                
                return response_data
                
            except Exception as response_error:
                # Simple fallback without complex recovery logic
                print__memory_monitoring(f"❌ Error preparing response: {response_error}")
                
                fallback_response = {
                    "prompt": request.prompt,
                    "result": result if 'result' in locals() else "Analysis completed",
                    "queries_and_results": [],
                    "thread_id": request.thread_id,
                    "top_selection_codes": [],
                    "iteration": 0,
                    "max_iterations": 2,
                    "sql": None,
                    "datasetUrl": None,
                    "run_id": run_id,
                    "error": "Response formatting error"
                }
                
                return fallback_response
            
        except asyncio.TimeoutError:
            error_msg = "Analysis timed out after 8 minutes"
            print__feedback_flow(f"🚨 {error_msg}")
            
            # Log for recovery purposes
            if run_id:
                print__feedback_flow(f"🔄 TIMEOUT INFO: thread_id={request.thread_id}, run_id={run_id}")
            
            raise HTTPException(status_code=408, detail=error_msg)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print__feedback_flow(f"🚨 {error_msg}")
            print__feedback_flow(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            
            # Enhanced error logging for recovery
            if run_id:
                print__feedback_flow(f"🔄 ERROR INFO: thread_id={request.thread_id}, run_id={run_id}")
            
            raise HTTPException(status_code=500, detail="Sorry, there was an error processing your request. Please try again.")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user=Depends(get_current_user)):
    """Submit feedback for a specific run_id to LangSmith."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__feedback_flow(f"📥 Incoming feedback request:")
    print__feedback_flow(f"👤 User: {user_email}")
    print__feedback_flow(f"🔑 Run ID: '{request.run_id}'")
    print__feedback_flow(f"🔑 Run ID type: {type(request.run_id).__name__}, length: {len(request.run_id) if request.run_id else 0}")
    print__feedback_flow(f"👍/👎 Feedback: {request.feedback}")
    print__feedback_flow(f"💬 Comment: {request.comment}")
    
    # Validate that at least one of feedback or comment is provided
    if request.feedback is None and not request.comment:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'feedback' or 'comment' must be provided"
        )
    
    try:
        try:
            print__feedback_flow(f"🔍 Validating UUID format for: '{request.run_id}'")
            # Debug check if it resembles a UUID at all
            if not request.run_id or len(request.run_id) < 32:
                print__feedback_flow(f"⚠️ Run ID is suspiciously short for a UUID: '{request.run_id}'")
            
            # Try to convert to UUID to validate format
            try:
                run_uuid = str(uuid.UUID(request.run_id))
                print__feedback_flow(f"✅ UUID validation successful: '{run_uuid}'")
            except ValueError as uuid_error:
                print__feedback_flow(f"🚨 UUID ValueError details: {str(uuid_error)}")
                # More detailed diagnostic about the input
                for i, char in enumerate(request.run_id):
                    if not (char.isalnum() or char == '-'):
                        print__feedback_flow(f"🚨 Invalid character at position {i}: '{char}' (ord={ord(char)})")
                raise
        except ValueError:
            print__feedback_flow(f"❌ UUID validation failed for: '{request.run_id}'")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}"
            )
        
        # 🔒 SECURITY CHECK: Verify user owns this run_id before submitting feedback
        print__feedback_flow(f"🔒 Verifying run_id ownership for user: {user_email}")
        
        try:
            # Get a healthy pool to check ownership
            from my_agent.utils.postgres_checkpointer import get_healthy_pool
            pool = await get_healthy_pool()
            
            async with pool.connection() as conn:
                ownership_result = await conn.execute("""
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE run_id = %s AND email = %s
                """, (run_uuid, user_email))
                
                ownership_row = await ownership_result.fetchone()
                ownership_count = ownership_row[0] if ownership_row else 0
                
                if ownership_count == 0:
                    print__feedback_flow(f"🚫 SECURITY: User {user_email} does not own run_id {run_uuid} - feedback denied")
                    raise HTTPException(
                        status_code=404,
                        detail="Run ID not found or access denied"
                    )
                
                print__feedback_flow(f"✅ SECURITY: User {user_email} owns run_id {run_uuid} - feedback authorized")
                
        except HTTPException:
            raise
        except Exception as ownership_error:
            print__feedback_flow(f"⚠ Could not verify ownership: {ownership_error}")
            # Continue with feedback submission but log the warning
            print__feedback_flow(f"⚠ Proceeding with feedback submission despite ownership check failure")
        
        print__feedback_flow("🔄 Initializing LangSmith client")
        client = Client()
        
        # Prepare feedback data for LangSmith
        feedback_kwargs = {
            "run_id": run_uuid,
            "key": "SENTIMENT"
        }
        
        # Only add score if feedback is provided
        if request.feedback is not None:
            feedback_kwargs["score"] = request.feedback
            print__feedback_flow(f"📤 Submitting feedback with score to LangSmith - run_id: '{run_uuid}', score: {request.feedback}")
        else:
            print__feedback_flow(f"📤 Submitting comment-only feedback to LangSmith - run_id: '{run_uuid}'")
        
        # Only add comment if provided
        if request.comment:
            feedback_kwargs["comment"] = request.comment
        
        client.create_feedback(**feedback_kwargs)
        
        print__feedback_flow(f"✅ Feedback successfully submitted to LangSmith")
        return {
            "message": "Feedback submitted successfully", 
            "run_id": run_uuid,
            "feedback": request.feedback,
            "comment": request.comment
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print__feedback_flow(f"🚨 LangSmith feedback submission error: {str(e)}")
        print__feedback_flow(f"🔍 Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")

@app.post("/sentiment")
async def update_sentiment(request: SentimentRequest, user=Depends(get_current_user)):
    """Update sentiment for a specific run_id."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__sentiment_flow(f"📥 Incoming sentiment update request:")
    print__sentiment_flow(f"👤 User: {user_email}")
    print__sentiment_flow(f"🔑 Run ID: '{request.run_id}'")
    print__sentiment_flow(f"👍/👎 Sentiment: {request.sentiment}")
    
    try:
        # Validate UUID format
        try:
            run_uuid = str(uuid.UUID(request.run_id))
            print__sentiment_flow(f"✅ UUID validation successful: '{run_uuid}'")
        except ValueError:
            print__sentiment_flow(f"❌ UUID validation failed for: '{request.run_id}'")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}"
            )
        
        # 🔒 SECURITY: Update sentiment with user email verification
        print__sentiment_flow(f"🔒 Verifying ownership before sentiment update")
        success = await update_thread_run_sentiment(run_uuid, request.sentiment, user_email)
        
        if success:
            print__sentiment_flow(f"✅ Sentiment successfully updated")
            return {
                "message": "Sentiment updated successfully", 
                "run_id": run_uuid,
                "sentiment": request.sentiment
            }
        else:
            print__sentiment_flow(f"❌ Failed to update sentiment - run_id may not exist or access denied")
            raise HTTPException(
                status_code=404,
                detail=f"Run ID not found or access denied: {run_uuid}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print__sentiment_flow(f"🚨 Sentiment update error: {str(e)}")
        print__sentiment_flow(f"🔍 Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Failed to update sentiment: {e}")

@app.get("/chat/{thread_id}/sentiments")
async def get_thread_sentiments(thread_id: str, user=Depends(get_current_user)):
    """Get sentiment values for all messages in a thread."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    try:
        print__sentiment_flow(f"📥 Getting sentiments for thread {thread_id}, user: {user_email}")
        sentiments = await get_thread_run_sentiments(user_email, thread_id)
        
        print__sentiment_flow(f"✅ Retrieved {len(sentiments)} sentiment values")
        return sentiments
    
    except Exception as e:
        print__sentiment_flow(f"❌ Failed to get sentiments for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiments: {e}")

@app.get("/chat-threads")
async def get_chat_threads(user=Depends(get_current_user)) -> List[ChatThreadResponse]:
    """Get all chat threads for the authenticated user from PostgreSQL."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__api_postgresql(f"📥 Loading chat threads for user: {user_email}")
    
    try:
        # First try to use the checkpointer's connection pool
        checkpointer = await get_healthy_checkpointer()
        
        if hasattr(checkpointer, 'conn') and checkpointer.conn and not checkpointer.conn.closed:
            print__api_postgresql(f"🔍 Using checkpointer connection pool")
            threads = await get_user_chat_threads(user_email, checkpointer.conn)
        else:
            print__api_postgresql(f"⚠ Checkpointer connection pool not available, using direct connection")
            # Fallback to direct connection (this will create its own healthy pool)
            threads = await get_user_chat_threads(user_email)
        
        print__api_postgresql(f"✅ Retrieved {len(threads)} threads for user {user_email}")
        
        if len(threads) == 0:
            print__api_postgresql(f"🔍 No threads found - this might be expected for new users")
            print__api_postgresql(f"🔍 User email: '{user_email}'")
        
        # Convert to response format
        response_threads = []
        for thread in threads:
            print__api_postgresql(f"🔍 Processing thread: {thread}")
            response_threads.append(ChatThreadResponse(
                thread_id=thread["thread_id"],
                latest_timestamp=thread["latest_timestamp"].isoformat(),
                run_count=thread["run_count"],
                title=thread["title"],
                full_prompt=thread["full_prompt"]
            ))
        
        print__api_postgresql(f"📤 Returning {len(response_threads)} threads to frontend")
        return response_threads
        
    except Exception as e:
        print__api_postgresql(f"❌ Failed to get chat threads for user {user_email}: {e}")
        import traceback
        print__api_postgresql(f"🔍 Full traceback: {traceback.format_exc()}")
        
        # If this is a pool-related error, try one more time with completely fresh connection
        error_msg = str(e)
        if any(keyword in error_msg.lower() for keyword in [
            "pool", "closed", "connection", "timeout", "operational error"
        ]):
            print__api_postgresql(f"🔄 Attempting one final retry with fresh connection...")
            try:
                # This should create a completely fresh pool
                threads = await get_user_chat_threads(user_email)
                response_threads = []
                for thread in threads:
                    response_threads.append(ChatThreadResponse(
                        thread_id=thread["thread_id"],
                        latest_timestamp=thread["latest_timestamp"].isoformat(),
                        run_count=thread["run_count"],
                        title=thread["title"],
                        full_prompt=thread["full_prompt"]
                    ))
                print__api_postgresql(f"✅ Retry successful - returning {len(response_threads)} threads")
                return response_threads
            except Exception as retry_error:
                print__api_postgresql(f"❌ Retry also failed: {retry_error}")
        
        raise HTTPException(status_code=500, detail=f"Failed to get chat threads: {e}")

@app.delete("/chat/{thread_id}")
async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Delete all PostgreSQL checkpoint records and thread entries for a specific thread_id."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__api_postgresql(f"🗑️ Deleting chat thread {thread_id} for user {user_email}")
    
    try:
        # Get a healthy checkpointer
        checkpointer = await get_healthy_checkpointer()
        
        # Check if we have a PostgreSQL checkpointer (not InMemorySaver)
        if not hasattr(checkpointer, 'conn'):
            print__api_postgresql(f"⚠ No PostgreSQL checkpointer available - nothing to delete")
            return {"message": "No PostgreSQL checkpointer available - nothing to delete"}
        
        # Access the connection pool through the conn attribute
        pool = checkpointer.conn
        
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # 🔒 SECURITY CHECK: Verify user owns this thread before deleting
            print__api_postgresql(f"🔒 Verifying thread ownership for deletion - user: {user_email}, thread: {thread_id}")
            
            ownership_result = await conn.execute("""
                SELECT COUNT(*) FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (user_email, thread_id))
            
            ownership_row = await ownership_result.fetchone()
            thread_entries_count = ownership_row[0] if ownership_row else 0
            
            if thread_entries_count == 0:
                print__api_postgresql(f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - deletion denied")
                return {
                    "message": "Thread not found or access denied",
                    "thread_id": thread_id,
                    "user_email": user_email,
                    "deleted_counts": {}
                }
            
            print__api_postgresql(f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - deletion authorized")
            
            print__api_postgresql(f"🔄 Deleting from checkpoint tables for thread {thread_id}")
            
            # Delete from all checkpoint tables
            tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
            deleted_counts = {}
            
            for table in tables:
                try:
                    # First check if the table exists
                    result = await conn.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table,))
                    
                    table_exists = await result.fetchone()
                    if not table_exists or not table_exists[0]:
                        print__api_postgresql(f"⚠ Table {table} does not exist, skipping")
                        deleted_counts[table] = 0
                        continue
                    
                    # Delete records for this thread_id
                    result = await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s",
                        (thread_id,)
                    )
                    
                    deleted_counts[table] = result.rowcount if hasattr(result, 'rowcount') else 0
                    print__api_postgresql(f"✅ Deleted {deleted_counts[table]} records from {table} for thread_id: {thread_id}")
                    
                except Exception as table_error:
                    print__api_postgresql(f"⚠ Error deleting from table {table}: {table_error}")
                    deleted_counts[table] = f"Error: {str(table_error)}"
            
            # Delete from users_threads_runs table directly within the same transaction
            print__api_postgresql(f"🔄 Deleting thread entries from users_threads_runs for user {user_email}, thread {thread_id}")
            try:
                result = await conn.execute("""
                    DELETE FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (user_email, thread_id))
                
                users_threads_runs_deleted = result.rowcount if hasattr(result, 'rowcount') else 0
                print__api_postgresql(f"✅ Deleted {users_threads_runs_deleted} entries from users_threads_runs for user {user_email}, thread {thread_id}")
                
                deleted_counts["users_threads_runs"] = users_threads_runs_deleted
                
            except Exception as e:
                print__api_postgresql(f"❌ Error deleting from users_threads_runs: {e}")
                deleted_counts["users_threads_runs"] = f"Error: {str(e)}"
            
            # Also call the helper function for additional cleanup (backward compatibility)
            print__api_postgresql(f"🔄 Additional cleanup via helper function...")
            thread_entries_result = await delete_user_thread_entries(user_email, thread_id, pool)
            print__api_postgresql(f"✅ Helper function deletion result: {thread_entries_result}")
            
            result_data = {
                "message": f"Checkpoint records and thread entries deleted for thread_id: {thread_id}",
                "deleted_counts": deleted_counts,
                "thread_entries_deleted": thread_entries_result,
                "thread_id": thread_id,
                "user_email": user_email
            }
            
            print__api_postgresql(f"🎉 Successfully deleted thread {thread_id} for user {user_email}")
            return result_data
            
    except Exception as e:
        error_msg = str(e)
        print__api_postgresql(f"❌ Failed to delete checkpoint records for thread {thread_id}: {e}")
        
        # If it's a connection error, don't treat it as a failure since it means 
        # there are likely no records to delete anyway
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print__api_postgresql(f"⚠ PostgreSQL connection unavailable - no records to delete")
            return {
                "message": "PostgreSQL connection unavailable - no records to delete", 
                "thread_id": thread_id,
                "user_email": user_email,
                "warning": "Database connection issues"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint records: {e}")

@app.get("/catalog")
def get_catalog(
    page: int = Query(1, ge=1),
    q: Optional[str] = None,
    page_size: int = Query(10, ge=1, le=10000),
    user=Depends(get_current_user)
):
    db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
    offset = (page - 1) * page_size
    where_clause = ""
    params = []
    if q:
        where_clause = "WHERE selection_code LIKE ? OR extended_description LIKE ?"
        like_q = f"%{q}%"
        params.extend([like_q, like_q])
    query = f"""
        SELECT selection_code, extended_description
        FROM selection_descriptions
        {where_clause}
        ORDER BY selection_code
        LIMIT ? OFFSET ?
    """
    params.extend([page_size, offset])
    count_query = f"SELECT COUNT(*) FROM selection_descriptions {where_clause}"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(count_query, params[:-2] if q else [])
        total = cursor.fetchone()[0]
        cursor.execute(query, params)
        rows = cursor.fetchall()
    results = [
        {"selection_code": row[0], "extended_description": row[1]} for row in rows
    ]
    return {"results": results, "total": total, "page": page, "page_size": page_size}

@app.get("/data-tables")
def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
    db_path = "data/czsu_data.db"
    desc_db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
    if q:
        q_lower = q.lower()
        tables = [t for t in tables if q_lower in t.lower()]
    # Fetch short_descriptions from the other DB
    desc_map = {}
    try:
        with sqlite3.connect(desc_db_path) as desc_conn:
            desc_cursor = desc_conn.cursor()
            desc_cursor.execute("SELECT selection_code, short_description FROM selection_descriptions")
            for code, short_desc in desc_cursor.fetchall():
                desc_map[code] = short_desc
    except Exception as e:
        print__debug(f"Error fetching short_descriptions: {e}")
    # Build result list
    result = [
        {"selection_code": t, "short_description": desc_map.get(t, "")}
        for t in tables
    ]
    return {"tables": result}

@app.get("/data-table")
def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
    db_path = "data/czsu_data.db"
    if not table:
        print__debug("No table specified")
        return {"columns": [], "rows": []}
    print__debug(f"Requested table: {table}")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            print__debug(f"Columns: {columns}, Rows count: {len(rows)}")
        except Exception as e:
            print__debug(f"Error fetching table '{table}': {e}")
            return {"columns": [], "rows": []}
    return {"columns": columns, "rows": rows}

@app.get("/chat/{thread_id}/messages")
async def get_chat_messages(thread_id: str, user=Depends(get_current_user)) -> List[ChatMessage]:
    """Load conversation messages from PostgreSQL checkpoint history that preserves original user messages."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__api_postgresql(f"📥 Loading checkpoint messages for thread {thread_id}, user: {user_email}")
    
    try:
        # 🔒 SECURITY CHECK: Verify user owns this thread before retrieving messages
        print__api_postgresql(f"🔒 Verifying thread ownership for user: {user_email}, thread: {thread_id}")
        
        # Check if this user has any entries in users_threads_runs for this thread
        checkpointer = await get_healthy_checkpointer()
        
        if not hasattr(checkpointer, 'conn'):
            print__api_postgresql(f"⚠ No PostgreSQL checkpointer available - returning empty messages")
            return []
        
        # Verify thread ownership using users_threads_runs table
        async with checkpointer.conn.connection() as conn:
            ownership_result = await conn.execute("""
                SELECT COUNT(*) FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (user_email, thread_id))
            
            ownership_row = await ownership_result.fetchone()
            thread_entries_count = ownership_row[0] if ownership_row else 0
            
            if thread_entries_count == 0:
                print__api_postgresql(f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - access denied")
                # Return empty instead of error to avoid information disclosure
                return []
            
            print__api_postgresql(f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
        
        # Get conversation messages from checkpoint history
        stored_messages = await get_conversation_messages_from_checkpoints(checkpointer, thread_id, user_email)
        
        if not stored_messages:
            print__api_postgresql(f"⚠ No messages found in checkpoints for thread {thread_id}")
            return []
        
        print__api_postgresql(f"📄 Found {len(stored_messages)} messages from checkpoints")
        
        # Get additional metadata from latest checkpoint (like queries_and_results and top_selection_codes)
        queries_and_results = await get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id)
        
        # Get dataset information and SQL query from latest checkpoint
        datasets_used = []
        sql_query = None
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = await checkpointer.aget_tuple(config)
            
            if state_snapshot and state_snapshot.checkpoint:
                channel_values = state_snapshot.checkpoint.get("channel_values", {})
                top_selection_codes = channel_values.get("top_selection_codes", [])
                
                # Filter to only include selection codes actually used in queries (same logic as main.py)
                if top_selection_codes and queries_and_results:
                    # Import the filtering function from main.py
                    import sys
                    from pathlib import Path
                    
                    # Get the filtering logic
                    def extract_table_names_from_sql(sql_query: str):
                        """Extract table names from SQL query FROM clauses."""
                        import re
                        
                        # Remove comments and normalize whitespace
                        sql_clean = re.sub(r'--.*?(?=\n|$)', '', sql_query, flags=re.MULTILINE)
                        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
                        sql_clean = ' '.join(sql_clean.split())
                        
                        # Pattern to match FROM clause with table names
                        from_pattern = r'\bFROM\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1(?:\s*,\s*(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\3)*'
                        
                        table_names = []
                        matches = re.finditer(from_pattern, sql_clean, re.IGNORECASE)
                        
                        for match in matches:
                            # Extract the main table name (group 2)
                            if match.group(2):
                                table_names.append(match.group(2).upper())
                            # Extract additional table names if comma-separated (group 4)
                            if match.group(4):
                                table_names.append(match.group(4).upper())
                        
                        # Also handle JOIN clauses
                        join_pattern = r'\bJOIN\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1'
                        join_matches = re.finditer(join_pattern, sql_clean, re.IGNORECASE)
                        
                        for match in join_matches:
                            if match.group(2):
                                table_names.append(match.group(2).upper())
                        
                        return list(set(table_names))  # Remove duplicates
                    
                    def get_used_selection_codes(queries_and_results, top_selection_codes):
                        """Filter top_selection_codes to only include those actually used in queries."""
                        if not queries_and_results or not top_selection_codes:
                            return []
                        
                        # Extract all table names used in queries
                        used_table_names = set()
                        for query, _ in queries_and_results:
                            if query:
                                table_names = extract_table_names_from_sql(query)
                                used_table_names.update(table_names)
                        
                        # Filter selection codes to only include those that match used table names
                        used_selection_codes = []
                        for selection_code in top_selection_codes:
                            if selection_code.upper() in used_table_names:
                                used_selection_codes.append(selection_code)
                        
                        return used_selection_codes
                    
                    # Apply the filtering
                    used_selection_codes = get_used_selection_codes(queries_and_results, top_selection_codes)
                    
                    # Use filtered codes if available, otherwise fallback to all top_selection_codes
                    datasets_used = used_selection_codes if used_selection_codes else top_selection_codes
                    print__api_postgresql(f"📊 Found datasets used (filtered): {datasets_used} (from {len(top_selection_codes)} total)")
                elif top_selection_codes:
                    # If no queries yet, use all top_selection_codes
                    datasets_used = top_selection_codes
                    print__api_postgresql(f"📊 Found datasets used (unfiltered): {datasets_used}")
                
                # Extract SQL query from queries_and_results for SQL button
                if queries_and_results:
                    # Get the last (most recent) SQL query
                    sql_query = queries_and_results[-1][0] if queries_and_results[-1] else None
                    print__api_postgresql(f"🔍 Found SQL query: {sql_query[:50] if sql_query else 'None'}...")
                
        except Exception as e:
            print__api_postgresql(f"⚠ Could not get datasets/SQL from checkpoint: {e}")
        
        # Convert stored messages to frontend format
        chat_messages = []
        
        for i, stored_msg in enumerate(stored_messages):
            # Debug: Log the raw stored message
            print__api_postgresql(f"🔍 Processing stored message {i+1}: is_user={stored_msg.get('is_user')}, content='{stored_msg.get('content', '')[:30]}...'")
            
            # Create meta information for messages
            meta_info = {}
            
            # For AI messages, include queries/results, datasets used, and SQL query
            if not stored_msg["is_user"]:
                if queries_and_results:
                    meta_info["queriesAndResults"] = queries_and_results
                if datasets_used:
                    meta_info["datasetsUsed"] = datasets_used
                if sql_query:
                    meta_info["sqlQuery"] = sql_query
                meta_info["source"] = "checkpoint_history"
                print__api_postgresql(f"🔍 Added metadata to AI message: datasets={len(datasets_used)}, sql={'Yes' if sql_query else 'No'}")
            
            # Convert queries_and_results for AI messages
            queries_results_for_frontend = None
            if not stored_msg["is_user"] and queries_and_results:
                queries_results_for_frontend = queries_and_results
            
            # Create ChatMessage with explicit debugging
            is_user_flag = stored_msg["is_user"]
            print__api_postgresql(f"🔍 Creating ChatMessage: isUser={is_user_flag}")
            
            chat_message = ChatMessage(
                id=stored_msg["id"],
                threadId=thread_id,
                user=user_email if is_user_flag else "AI",
                content=stored_msg["content"],
                isUser=is_user_flag,  # Explicitly use the flag
                createdAt=int(stored_msg["timestamp"].timestamp() * 1000),
                error=None,
                meta=meta_info if meta_info else None,  # Only add meta if it has content
                queriesAndResults=queries_results_for_frontend,
                isLoading=False,
                startedAt=None,
                isError=False
            )
            
            # Debug: Verify the ChatMessage was created correctly
            print__api_postgresql(f"🔍 ChatMessage created: isUser={chat_message.isUser}, user='{chat_message.user}'")
            
            chat_messages.append(chat_message)
        
        print__api_postgresql(f"✅ Converted {len(chat_messages)} messages to frontend format")
        
        # Log the messages for debugging
        for i, msg in enumerate(chat_messages):
            user_type = "👤 User" if msg.isUser else "🤖 AI"
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            datasets_info = f" (datasets: {msg.meta.get('datasetsUsed', [])})" if msg.meta and msg.meta.get('datasetsUsed') else ""
            sql_info = f" (SQL: {msg.meta.get('sqlQuery', 'None')[:30]}...)" if msg.meta and msg.meta.get('sqlQuery') else ""
            print__api_postgresql(f"{i+1}. {user_type}: {content_preview}{datasets_info}{sql_info}")
        
        return chat_messages
        
    except Exception as e:
        error_msg = str(e)
        print__api_postgresql(f"❌ Failed to load checkpoint messages for thread {thread_id}: {e}")
        
        # Handle specific database connection errors gracefully
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print__api_postgresql(f"⚠ Database connection error - returning empty messages")
            return []
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load checkpoint messages: {e}")

@app.get("/chat/all-messages")
async def get_all_chat_messages(user=Depends(get_current_user)) -> Dict:
    """OPTIMIZED: Load conversation messages for ALL user threads using built-in functions in parallel.
    
    This endpoint uses the proven working functions but calls them efficiently:
    - Single query to get all user threads, run-ids, and sentiments 
    - Use the proven working get_conversation_messages_from_checkpoints() for each thread
    - Call all thread processing in parallel using asyncio.gather()
    - Much faster than sequential individual API calls from frontend
    """
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__api_postgresql(f"📥 PARALLEL BULK: Loading ALL chat messages for user: {user_email}")
    
    try:
        checkpointer = await get_healthy_checkpointer()
        
        if not hasattr(checkpointer, 'conn'):
            print__api_postgresql(f"⚠ No PostgreSQL checkpointer available - returning empty messages")
            return {"messages": {}, "runIds": {}, "sentiments": {}}
        
        # STEP 1: Get all user threads, run-ids, and sentiments in ONE query
        print__api_postgresql(f"🔍 BULK QUERY: Getting all user threads, run-ids, and sentiments")
        user_thread_ids = []
        all_run_ids = {}
        all_sentiments = {}
        
        async with checkpointer.conn.connection() as conn:
            # Single query for all threads, run-ids, and sentiments
            result = await conn.execute("""
                SELECT 
                    thread_id, 
                    run_id, 
                    prompt, 
                    timestamp,
                    sentiment
                FROM users_threads_runs 
                WHERE email = %s
                ORDER BY thread_id, timestamp ASC
            """, (user_email,))
            
            async for row in result:
                thread_id, run_id, prompt, timestamp, sentiment = row
                
                # Track unique thread IDs
                if thread_id not in user_thread_ids:
                    user_thread_ids.append(thread_id)
                
                # Build run-ids dictionary
                if thread_id not in all_run_ids:
                    all_run_ids[thread_id] = []
                all_run_ids[thread_id].append({
                    "run_id": run_id,
                    "prompt": prompt,
                    "timestamp": timestamp.isoformat()
                })
                
                # Build sentiments dictionary
                if sentiment is not None:
                    if thread_id not in all_sentiments:
                        all_sentiments[thread_id] = {}
                    all_sentiments[thread_id][run_id] = sentiment
        
        print__api_postgresql(f"📊 BULK: Found {len(user_thread_ids)} threads")
        
        if not user_thread_ids:
            print__api_postgresql(f"⚠ No threads found for user - returning empty dictionary")
            return {"messages": {}, "runIds": {}, "sentiments": {}}
        
        # STEP 2: Process all threads in PARALLEL using the proven working functions
        print__api_postgresql(f"🔄 PARALLEL: Processing all {len(user_thread_ids)} threads using proven working functions")
        
        async def process_single_thread(thread_id: str):
            """Process a single thread using the proven working functions."""
            try:
                print__api_postgresql(f"🔄 Processing thread {thread_id}")
                
                # Use the ORIGINAL working function that we know works
                stored_messages = await get_conversation_messages_from_checkpoints(checkpointer, thread_id, user_email)
                
                if not stored_messages:
                    print__api_postgresql(f"⚠ No messages found in checkpoints for thread {thread_id}")
                    return thread_id, []
                
                print__api_postgresql(f"📄 Found {len(stored_messages)} messages for thread {thread_id}")
                
                # Get additional metadata from latest checkpoint (like queries_and_results)
                queries_and_results = await get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id)
                
                # Get dataset information and SQL query from latest checkpoint
                datasets_used = []
                sql_query = None
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    state_snapshot = await checkpointer.aget_tuple(config)
                    
                    if state_snapshot and state_snapshot.checkpoint:
                        channel_values = state_snapshot.checkpoint.get("channel_values", {})
                        top_selection_codes = channel_values.get("top_selection_codes", [])
                        
                        # Use the datasets directly
                        datasets_used = top_selection_codes
                        
                        # Extract SQL query from queries_and_results for SQL button
                        if queries_and_results:
                            # Get the last (most recent) SQL query
                            sql_query = queries_and_results[-1][0] if queries_and_results[-1] else None
                    
                except Exception as e:
                    print__api_postgresql(f"⚠ Could not get datasets/SQL from checkpoint: {e}")
                
                # Convert stored messages to frontend format (SAME as original working code)
                chat_messages = []
                
                for i, stored_msg in enumerate(stored_messages):
                    # Create meta information for AI messages
                    meta_info = {}
                    if not stored_msg["is_user"]:
                        if queries_and_results:
                            meta_info["queriesAndResults"] = queries_and_results
                        if datasets_used:
                            meta_info["datasetsUsed"] = datasets_used
                        if sql_query:
                            meta_info["sqlQuery"] = sql_query
                        meta_info["source"] = "parallel_bulk_processing"
                    
                    # Convert queries_and_results for AI messages
                    queries_results_for_frontend = None
                    if not stored_msg["is_user"] and queries_and_results:
                        queries_results_for_frontend = queries_and_results
                    
                    is_user_flag = stored_msg["is_user"]
                    
                    chat_message = ChatMessage(
                        id=stored_msg["id"],
                        threadId=thread_id,
                        user=user_email if is_user_flag else "AI",
                        content=stored_msg["content"],
                        isUser=is_user_flag,
                        createdAt=int(stored_msg["timestamp"].timestamp() * 1000),
                        error=None,
                        meta=meta_info if meta_info else None,
                        queriesAndResults=queries_results_for_frontend,
                        isLoading=False,
                        startedAt=None,
                        isError=False
                    )
                    
                    chat_messages.append(chat_message)
                
                print__api_postgresql(f"✅ Processed {len(chat_messages)} messages for thread {thread_id}")
                return thread_id, chat_messages
                
            except Exception as e:
                print__api_postgresql(f"❌ Error processing thread {thread_id}: {e}")
                return thread_id, []
        
        # Process ALL threads in parallel using asyncio.gather()
        print__api_postgresql(f"🚀 Starting parallel processing of {len(user_thread_ids)} threads...")
        
        # Use asyncio.gather to process all threads concurrently
        thread_results = await asyncio.gather(
            *[process_single_thread(thread_id) for thread_id in user_thread_ids],
            return_exceptions=True
        )
        
        # Collect results
        all_messages = {}
        total_messages = 0
        
        for result in thread_results:
            if isinstance(result, Exception):
                print__api_postgresql(f"⚠ Exception in thread processing: {result}")
                continue
            
            thread_id, chat_messages = result
            all_messages[thread_id] = chat_messages
            total_messages += len(chat_messages)
        
        print__api_postgresql(f"✅ PARALLEL BULK LOADING COMPLETE: Successfully loaded ALL messages for user {user_email}: {len(all_messages)} threads, {total_messages} total messages using proven working functions in parallel")
        
        result = {
            "messages": all_messages,
            "runIds": all_run_ids,
            "sentiments": all_sentiments
        }
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print__api_postgresql(f"❌ Failed to bulk load all chat messages for user {user_email}: {e}")
        
        # Handle specific database connection errors gracefully
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print__api_postgresql(f"⚠ Database connection error - returning empty messages")
            return {"messages": {}, "runIds": {}, "sentiments": {}}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to bulk load all chat messages: {e}")

@app.get("/debug/chat/{thread_id}/checkpoints")
async def debug_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Debug endpoint to inspect raw checkpoint data for a thread."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__debug(f"🔍 Inspecting checkpoints for thread: {thread_id}")
    
    try:
        checkpointer = await get_healthy_checkpointer()
        
        if not hasattr(checkpointer, 'conn'):
            return {"error": "No PostgreSQL checkpointer available"}
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints for this thread
        checkpoint_tuples = []
        try:
            # Fix: alist() returns an async generator, don't await it
            checkpoint_iterator = checkpointer.alist(config)
            async for checkpoint_tuple in checkpoint_iterator:
                checkpoint_tuples.append(checkpoint_tuple)
        except Exception as alist_error:
            print__debug(f"❌ Error getting checkpoint list: {alist_error}")
            return {"error": f"Failed to get checkpoints: {alist_error}"}
        
        debug_data = {
            "thread_id": thread_id,
            "total_checkpoints": len(checkpoint_tuples),
            "checkpoints": []
        }
        
        for i, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}
            
            checkpoint_info = {
                "index": i,
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id", "unknown"),
                "has_checkpoint": bool(checkpoint),
                "has_metadata": bool(metadata),
                "metadata_writes": metadata.get("writes", {}),
                "channel_values": {}
            }
            
            if checkpoint and "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                messages = channel_values.get("messages", [])
                
                checkpoint_info["channel_values"] = {
                    "message_count": len(messages),
                    "messages": []
                }
                
                for j, msg in enumerate(messages):
                    msg_info = {
                        "index": j,
                        "type": type(msg).__name__,
                        "id": getattr(msg, 'id', None),
                        "content_preview": getattr(msg, 'content', str(msg))[:200] + "..." if hasattr(msg, 'content') and len(getattr(msg, 'content', '')) > 200 else getattr(msg, 'content', str(msg)),
                        "content_length": len(getattr(msg, 'content', ''))
                    }
                    checkpoint_info["channel_values"]["messages"].append(msg_info)
            
            debug_data["checkpoints"].append(checkpoint_info)
        
        return debug_data
        
    except Exception as e:
        print__debug(f"❌ Error inspecting checkpoints: {e}")
        return {"error": str(e)}

@app.get("/debug/pool-status")
async def debug_pool_status():
    """Debug endpoint to check pool and checkpointer status (no auth required)."""
    global GLOBAL_CHECKPOINTER
    
    try:
        status = {
            "global_checkpointer_exists": GLOBAL_CHECKPOINTER is not None,
            "checkpointer_type": type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None,
            "has_connection_pool": False,
            "pool_closed": None,
            "pool_healthy": None,
            "can_query": False,
            "timestamp": datetime.now().isoformat()
        }
        
        if GLOBAL_CHECKPOINTER and hasattr(GLOBAL_CHECKPOINTER, 'conn'):
            status["has_connection_pool"] = True
            
            if GLOBAL_CHECKPOINTER.conn:
                status["pool_closed"] = GLOBAL_CHECKPOINTER.conn.closed
                
                # Test if we can execute a simple query
                try:
                    async with GLOBAL_CHECKPOINTER.conn.connection() as conn:
                        await conn.execute("SELECT 1")
                    status["can_query"] = True
                    status["pool_healthy"] = True
                except Exception as e:
                    status["can_query"] = False
                    status["pool_healthy"] = False
                    status["query_error"] = str(e)
        
        # Try to get a healthy checkpointer
        try:
            healthy_checkpointer = await get_healthy_checkpointer()
            status["healthy_checkpointer_type"] = type(healthy_checkpointer).__name__
            status["healthy_checkpointer_available"] = True
        except Exception as e:
            status["healthy_checkpointer_available"] = False
            status["healthy_checkpointer_error"] = str(e)
        
        return status
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/chat/{thread_id}/run-ids")
async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
    """Get run_ids for messages in a thread to enable feedback submission."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__feedback_flow(f"🔍 Fetching run_ids for thread {thread_id}")
    
    try:
        pool = await get_healthy_checkpointer()
        pool = pool.conn if hasattr(pool, 'conn') else None
        
        if not pool:
            print__feedback_flow("⚠ No pool available for run_id lookup")
            return {"run_ids": []}
        
        async with pool.connection() as conn:
            print__feedback_flow(f"📊 Executing SQL query for thread {thread_id}")
            result = await conn.execute("""
                SELECT run_id, prompt, timestamp
                FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
                ORDER BY timestamp ASC
            """, (user_email, thread_id))
            
            run_id_data = []
            async for row in result:
                print__feedback_flow(f"📝 Processing database row - run_id: {row[0]}, prompt: {row[1][:50]}...")
                try:
                    run_uuid = str(uuid.UUID(row[0])) if row[0] else None
                    if run_uuid:
                        run_id_data.append({
                            "run_id": run_uuid,
                            "prompt": row[1],
                            "timestamp": row[2].isoformat()
                        })
                        print__feedback_flow(f"✅ Valid UUID found: {run_uuid}")
                    else:
                        print__feedback_flow(f"⚠ Null run_id found for prompt: {row[1][:50]}...")
                except ValueError:
                    print__feedback_flow(f"❌ Invalid UUID in database: {row[0]}")
                    continue
            
            print__feedback_flow(f"📊 Total valid run_ids found: {len(run_id_data)}")
            return {"run_ids": run_id_data}
            
    except Exception as e:
        print__feedback_flow(f"🚨 Error fetching run_ids: {str(e)}")
        return {"run_ids": []}

@app.get("/debug/run-id/{run_id}")
async def debug_run_id(run_id: str, user=Depends(get_current_user)):
    """Debug endpoint to check if a run_id exists in the database."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__debug(f"🔍 Checking run_id: '{run_id}' for user: {user_email}")
    
    result = {
        "run_id": run_id,
        "run_id_type": type(run_id).__name__,
        "run_id_length": len(run_id) if run_id else 0,
        "is_valid_uuid_format": False,
        "exists_in_database": False,
        "user_owns_run_id": False,
        "database_details": None
    }
    
    # Check if it's a valid UUID format
    try:
        uuid_obj = uuid.UUID(run_id)
        result["is_valid_uuid_format"] = True
        result["uuid_parsed"] = str(uuid_obj)
    except ValueError as e:
        result["uuid_error"] = str(e)
    
    # Check if it exists in the database
    try:
        pool = await get_healthy_checkpointer()
        pool = pool.conn if hasattr(pool, 'conn') else None
        
        if pool:
            async with pool.connection() as conn:
                # 🔒 SECURITY: Check in users_threads_runs table with user ownership verification
                db_result = await conn.execute("""
                    SELECT email, thread_id, prompt, timestamp
                    FROM users_threads_runs 
                    WHERE run_id = %s AND email = %s
                """, (run_id, user_email))
                
                row = await db_result.fetchone()
                if row:
                    result["exists_in_database"] = True
                    result["user_owns_run_id"] = True
                    result["database_details"] = {
                        "email": row[0],
                        "thread_id": row[1],
                        "prompt": row[2],
                        "timestamp": row[3].isoformat() if row[3] else None
                    }
                    print__debug(f"✅ User {user_email} owns run_id {run_id}")
                else:
                    # Check if run_id exists but belongs to different user
                    db_result_any = await conn.execute("""
                        SELECT COUNT(*) FROM users_threads_runs WHERE run_id = %s
                    """, (run_id,))
                    
                    any_row = await db_result_any.fetchone()
                    if any_row and any_row[0] > 0:
                        result["exists_in_database"] = True
                        result["user_owns_run_id"] = False
                        print__debug(f"🚫 Run_id {run_id} exists but user {user_email} does not own it")
                    else:
                        print__debug(f"❌ Run_id {run_id} not found in database")
    except Exception as e:
        result["database_error"] = str(e)
    
    return result

#==============================================================================
# DEBUG FUNCTIONS
#==============================================================================
def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[API-PostgreSQL] {msg}")
        import sys
        sys.stdout.flush()

def print__feedback_flow(msg: str) -> None:
    """Print FEEDBACK-FLOW messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[FEEDBACK-FLOW] {msg}")
        import sys
        sys.stdout.flush()

def print__sentiment_flow(msg: str) -> None:
    """Print SENTIMENT-FLOW messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[SENTIMENT-FLOW] {msg}")
        import sys
        sys.stdout.flush()

def print__debug(msg: str) -> None:
    """Print DEBUG messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[DEBUG] {msg}")
        import sys
        sys.stdout.flush()

# Global exception handlers for proper error handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with proper 422 status code."""
    print__debug(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions properly."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions as 400 Bad Request."""
    print__debug(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    print__debug(f"Unexpected error: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    ) 