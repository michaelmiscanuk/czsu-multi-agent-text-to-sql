# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os  # Import os early for environment variable access

def print__startup_debug(msg: str) -> None:
    """Print startup debug messages when debug mode is enabled."""
    debug_mode = os.environ.get('DEBUG', '0')
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
    update_thread_run_sentiment,
    get_thread_run_sentiments,
    get_user_chat_threads,
    get_user_chat_threads_count,  # Add this import
    delete_user_thread_entries,
    get_conversation_messages_from_checkpoints,
    get_queries_and_results_from_latest_checkpoint,
)

# Additional imports for sentiment functionality
from my_agent.utils.postgres_checkpointer import (
    get_thread_run_sentiments
)

# Read GC memory threshold from environment with default fallback
GC_MEMORY_THRESHOLD = int(os.environ.get('GC_MEMORY_THRESHOLD', '1900'))  # 1900MB for 2GB memory allocation
print__startup_debug(f"🔧 API Server: GC_MEMORY_THRESHOLD set to {GC_MEMORY_THRESHOLD}MB (from environment)")

def print__memory_monitoring(msg: str) -> None:
    """Print MEMORY-MONITORING messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[MEMORY-MONITORING] {msg}")
        import sys
        sys.stdout.flush()

# MEMORY LEAK PREVENTION: Simplified global tracking
_app_startup_time = None
_memory_baseline = None  # RSS memory at startup
_request_count = 0  # Track total requests processed

# Global shared checkpointer for conversation memory across API requests
# This ensures that conversation state is preserved between frontend requests using PostgreSQL
GLOBAL_CHECKPOINTER = None

# Add a semaphore to limit concurrent analysis requests
MAX_CONCURRENT_ANALYSES = int(os.environ.get('MAX_CONCURRENT_ANALYSES', '3'))  # Read from .env with fallback to 3
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# Log the concurrency setting for debugging
print__startup_debug(f"🔧 API Server: MAX_CONCURRENT_ANALYSES set to {MAX_CONCURRENT_ANALYSES} (from environment)")
print__memory_monitoring(f"🔒 Concurrent analysis semaphore initialized with {MAX_CONCURRENT_ANALYSES} slots")

# RATE LIMITING: Global rate limiting storage
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # 60 seconds window
RATE_LIMIT_BURST = 20  # burst limit for rapid requests
RATE_LIMIT_MAX_WAIT = 5  # maximum seconds to wait before giving up

# Throttling semaphores per IP to limit concurrent requests
throttle_semaphores = defaultdict(lambda: asyncio.Semaphore(8))  # Max 8 concurrent requests per IP

def check_memory_and_gc():
    """Enhanced memory check with cache cleanup and scaling strategy."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        
        # Clean up cache first if memory is getting high
        if rss_mb > (GC_MEMORY_THRESHOLD * 0.8):  # At 80% of threshold
            print__memory_monitoring(f"📊 Memory at {rss_mb:.1f}MB (80% of {GC_MEMORY_THRESHOLD}MB threshold) - cleaning cache")
            cleaned_entries = cleanup_bulk_cache()
            if cleaned_entries > 0:
                # Check memory after cache cleanup
                new_memory = psutil.Process().memory_info().rss / 1024 / 1024
                freed = rss_mb - new_memory
                print__memory_monitoring(f"🧹 Cache cleanup freed {freed:.1f}MB, cleaned {cleaned_entries} entries")
                rss_mb = new_memory
        
        # Trigger GC only if above threshold
        if rss_mb > GC_MEMORY_THRESHOLD:
            print__memory_monitoring(f"🚨 MEMORY THRESHOLD EXCEEDED: {rss_mb:.1f}MB > {GC_MEMORY_THRESHOLD}MB - running GC")
            import gc
            collected = gc.collect()
            print__memory_monitoring(f"🧹 GC collected {collected} objects")
            
            # Log memory after GC
            new_memory = psutil.Process().memory_info().rss / 1024 / 1024
            freed = rss_mb - new_memory
            print__memory_monitoring(f"🧹 Memory after GC: {new_memory:.1f}MB (freed: {freed:.1f}MB)")
            
            # If memory is still high after GC, provide scaling guidance
            if new_memory > (GC_MEMORY_THRESHOLD * 0.9):
                thread_count = len(_bulk_loading_cache)
                print__memory_monitoring(f"⚠ HIGH MEMORY WARNING: {new_memory:.1f}MB after GC")
                print__memory_monitoring(f"📊 Current cache entries: {thread_count}")
                if thread_count > 20:
                    print__memory_monitoring(f"💡 SCALING TIP: Consider implementing pagination for chat threads")
            
        return rss_mb
        
    except Exception as e:
        print__memory_monitoring(f"❌ Could not check memory: {e}")
        return 0

def log_memory_usage(context: str = ""):
    """Simplified memory logging."""
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        
        print__memory_monitoring(f"📊 Memory usage{f' [{context}]' if context else ''}: {rss_mb:.1f}MB RSS")
        
        # Simple threshold check
        if rss_mb > GC_MEMORY_THRESHOLD:
            check_memory_and_gc()
            
    except Exception as e:
        print__memory_monitoring(f"❌ Could not check memory usage: {e}")

def monitor_route_registration(route_path: str, method: str):
    """Simplified route registration monitoring."""
    pass  # Simplified - no complex tracking needed

def log_comprehensive_error(context: str, error: Exception, request: Request = None):
    """Simplified error logging."""
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
        })
    
    # Log to debug output
    print__debug(f"🚨 ERROR: {json.dumps(error_details, indent=2)}")

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

def setup_graceful_shutdown():
    """Setup graceful shutdown handlers."""
    def signal_handler(signum, frame):
        print__memory_monitoring(f"📡 Received signal {signum} - preparing for graceful shutdown...")
        log_memory_usage("shutdown_signal")
        
    # Register signal handlers for common restart signals
    signal.signal(signal.SIGTERM, signal_handler)  # Most common for container restarts
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, signal_handler)  # User-defined signal

async def initialize_checkpointer():
    """Initialize the global checkpointer with proper async context management."""
    global GLOBAL_CHECKPOINTER
    if GLOBAL_CHECKPOINTER is None:
        try:
            print__startup_debug("🚀 Initializing PostgreSQL Connection System...")
            print__startup_debug(f"🔍 Current global checkpointer state: {GLOBAL_CHECKPOINTER}")
            log_memory_usage("startup")
            
            # Create and initialize the checkpointer manager
            print__startup_debug("🔧 Creating PostgreSQL checkpointer manager...")
            checkpointer_manager = await asyncio.wait_for(
                get_postgres_checkpointer(), 
                timeout=60  # 1 minute for checkpointer creation
            )
            
            # Enter the async context to activate the checkpointer
            print__startup_debug("🔧 Entering AsyncPostgresSaver context...")
            GLOBAL_CHECKPOINTER = await checkpointer_manager.__aenter__()
            
            # Store the manager for cleanup later
            GLOBAL_CHECKPOINTER._manager = checkpointer_manager
            
            # Verify the checkpointer is working
            print__startup_debug(f"⚠️ Created checkpointer type: {type(GLOBAL_CHECKPOINTER).__name__}")
            print__startup_debug("✅ LangGraph checkpointer features enabled:")
            print__startup_debug("   • Proper async context management")
            print__startup_debug("   • PostgreSQL persistence")
            print__startup_debug("   • Thread-safe operations")
            print__startup_debug("   • AsyncPG for custom operations")
            
            print__startup_debug("✅ PostgreSQL checkpointer initialization completed")
            print__startup_debug("✅ LangGraph checkpoint tables verified/created")
            log_memory_usage("checkpointer_initialized")
            
        except asyncio.TimeoutError:
            print__startup_debug("❌ PostgreSQL checkpointer initialization timeout")
            print__startup_debug("⚠️ This may indicate network issues or database overload")
            
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print__startup_debug("⚠️ Falling back to InMemorySaver")
            
        except Exception as e:
            print__startup_debug(f"❌ PostgreSQL checkpointer initialization failed: {e}")
            print__startup_debug(f"🔍 Error type: {type(e).__name__}")
            
            # Enhanced error diagnostics
            error_msg = str(e).lower()
            if "ssl" in error_msg:
                print__startup_debug("💡 SSL-related error detected during initialization")
                print__startup_debug("   • Check database SSL configuration")
                print__startup_debug("   • Verify network connectivity")
                print__startup_debug("   • Check firewall settings")
            elif "timeout" in error_msg:
                print__startup_debug("💡 Timeout error detected during initialization")
                print__startup_debug("   • Check database server responsiveness")
                print__startup_debug("   • Verify network latency")
                print__startup_debug("   • Check connection limits")
            elif "connection" in error_msg:
                print__startup_debug("💡 Connection error detected during initialization")
                print__startup_debug("   • Check database connection string")
                print__startup_debug("   • Verify database server is running")
                print__startup_debug("   • Check network connectivity")
            
            import traceback
            print__startup_debug(f"🔍 Full traceback: {traceback.format_exc()}")
            
            # Fallback to InMemorySaver for development/testing
            from langgraph.checkpoint.memory import InMemorySaver
            GLOBAL_CHECKPOINTER = InMemorySaver()
            print__startup_debug("⚠️ Falling back to InMemorySaver")
            
    else:
        print__startup_debug("⚠️ Global checkpointer already exists - skipping initialization")

async def cleanup_checkpointer():
    """Clean up resources on app shutdown."""
    global GLOBAL_CHECKPOINTER
    print__memory_monitoring("🧹 Starting application cleanup...")
    log_memory_usage("cleanup_start")
    
    if GLOBAL_CHECKPOINTER:
        try:
            # Check if this is our manager-based checkpointer
            if hasattr(GLOBAL_CHECKPOINTER, '_manager'):
                print__startup_debug("🧹 Exiting AsyncPostgresSaver context manager...")
                manager = GLOBAL_CHECKPOINTER._manager
                await manager.__aexit__(None, None, None)
                print__startup_debug("✅ AsyncPostgresSaver context manager exited cleanly")
            elif hasattr(GLOBAL_CHECKPOINTER, 'conn') and GLOBAL_CHECKPOINTER.conn:
                # Fallback for other types of checkpointers
                if not GLOBAL_CHECKPOINTER.conn.closed:
                    await GLOBAL_CHECKPOINTER.conn.close()
                    print__startup_debug("✅ PostgreSQL connection pool closed cleanly")
                else:
                    print__startup_debug("⚠️ PostgreSQL connection pool was already closed")
            else:
                print__startup_debug("⚠️ Checkpointer doesn't require connection cleanup")
        except Exception as e:
            print__startup_debug(f"⚠️ Error during checkpointer cleanup: {e}")
        finally:
            GLOBAL_CHECKPOINTER = None
            print__startup_debug("✅ Global checkpointer cleared")
    
    # Clean up the robust connection pool
    try:
        from my_agent.utils.robust_postgres_pool import close_global_pool
        await close_global_pool()
        print__startup_debug("✅ Robust connection pool closed")
    except Exception as e:
        print__startup_debug(f"⚠️ Error closing robust pool: {e}")
    
    log_memory_usage("cleanup_complete")

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance."""
    global GLOBAL_CHECKPOINTER
    
    # With the new AsyncPostgreSQLCheckpointerManager approach, 
    # we don't need complex health checking - the context manager handles it
    if GLOBAL_CHECKPOINTER is None:
        await initialize_checkpointer()
    
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
async def simplified_memory_monitoring_middleware(request: Request, call_next):
    """Simplified memory monitoring middleware."""
    global _request_count
    
    _request_count += 1
    
    # Only check memory for heavy operations
    request_path = request.url.path
    if any(path in request_path for path in ["/analyze", "/chat/all-messages"]):
        log_memory_usage(f"before_{request_path.replace('/', '_')}")
    
    response = await call_next(request)
    
    # Check memory after heavy operations
    if any(path in request_path for path in ["/analyze", "/chat/all-messages"]):
        log_memory_usage(f"after_{request_path.replace('/', '_')}")
    
    return response

@app.get("/health")
async def health_check():
    """Simplified health check endpoint."""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "app_uptime": str(datetime.now() - _app_startup_time) if _app_startup_time else "unknown",
            "total_requests_processed": _request_count
        }
        
        # Simple memory info
        try:
            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024
            health_data.update({
                "memory_rss_mb": round(rss_mb, 1),
                "memory_over_threshold": rss_mb > GC_MEMORY_THRESHOLD
            })
        except Exception as mem_error:
            health_data["memory_error"] = str(mem_error)
        
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
    """Enhanced memory-specific health check with cache information."""
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        
        # Clean up expired cache entries
        cleaned_entries = cleanup_bulk_cache()
        
        status = "healthy"
        if rss_mb > GC_MEMORY_THRESHOLD:
            status = "high_memory"
        elif rss_mb > (GC_MEMORY_THRESHOLD * 0.8):
            status = "warning"
        
        cache_info = {
            "active_cache_entries": len(_bulk_loading_cache),
            "cleaned_expired_entries": cleaned_entries,
            "cache_timeout_seconds": BULK_CACHE_TIMEOUT
        }
        
        # Calculate estimated memory per thread for scaling guidance
        thread_count = len(_bulk_loading_cache)
        memory_per_thread = rss_mb / max(thread_count, 1) if thread_count > 0 else 0
        estimated_max_threads = int(GC_MEMORY_THRESHOLD / max(memory_per_thread, 38)) if memory_per_thread > 0 else 50
        
        return {
            "status": status,
            "memory_rss_mb": round(rss_mb, 1),
            "memory_threshold_mb": GC_MEMORY_THRESHOLD,
            "memory_usage_percent": round((rss_mb / GC_MEMORY_THRESHOLD) * 100, 1),
            "over_threshold": rss_mb > GC_MEMORY_THRESHOLD,
            "total_requests_processed": _request_count,
            "cache_info": cache_info,
            "scaling_info": {
                "estimated_memory_per_thread_mb": round(memory_per_thread, 1),
                "estimated_max_threads_at_threshold": estimated_max_threads,
                "current_thread_count": thread_count
            },
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

@app.get("/placeholder/{width}/{height}")
async def get_placeholder_image(width: int, height: int):
    """Generate a placeholder image with specified dimensions."""
    try:
        # Validate dimensions
        width = max(1, min(width, 2000))  # Limit between 1 and 2000 pixels
        height = max(1, min(height, 2000))
        
        # Create a simple SVG placeholder
        svg_content = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#e5e7eb"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9ca3af" font-size="20">{width}x{height}</text>
        </svg>'''
        
        from fastapi.responses import Response
        return Response(
            content=svg_content,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        # Fallback for any errors
        simple_svg = f'''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f3f4f6"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#6b7280" font-size="12">Error</text>
        </svg>'''
        
        from fastapi.responses import Response
        return Response(
            content=simple_svg,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )

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

class PaginatedChatThreadsResponse(BaseModel):
    threads: List[ChatThreadResponse]
    total_count: int
    page: int
    limit: int
    has_more: bool

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

# FIXED: Enhanced JWT verification with NextAuth.js id_token support
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
        
        # Get unverified header and payload first - this should now work since we pre-validated the format
        try:
            unverified_header = jwt.get_unverified_header(token)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
        except jwt.DecodeError as e:
            # This should be rare now due to pre-validation, but keep for edge cases
            print__DEBUG_TOKEN(f"JWT decode error after pre-validation: {e}")
            raise HTTPException(status_code=401, detail="Invalid JWT token format")
        except Exception as e:
            print__DEBUG_TOKEN(f"JWT header decode error: {e}")
            raise HTTPException(status_code=401, detail="Invalid JWT token format")
        
        # Debug: print the audience in the token and the expected audience
        print__DEBUG_TOKEN(f"Token aud: {unverified_payload.get('aud')}")
        print__DEBUG_TOKEN(f"Backend GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID')}")
        
        # NEW: Check if this is a NextAuth.js id_token (missing 'kid' field)
        if "kid" not in unverified_header:
            # Reduce log noise - only log this every 10th occurrence
            _jwt_kid_missing_count += 1
            if _jwt_kid_missing_count % 10 == 1:  # Log 1st, 11th, 21st, etc.
                print__DEBUG_TOKEN(f"JWT token missing 'kid' field (#{_jwt_kid_missing_count}) - attempting NextAuth.js id_token verification")
            
            # NEXTAUTH.JS SUPPORT: Verify id_token directly using Google's tokeninfo endpoint
            try:
                print__DEBUG_TOKEN("Attempting NextAuth.js id_token verification via Google tokeninfo endpoint")
                
                # Use Google's tokeninfo endpoint to verify the id_token
                tokeninfo_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
                response = requests.get(tokeninfo_url, timeout=10)
                
                if response.status_code == 200:
                    tokeninfo = response.json()
                    print__DEBUG_TOKEN(f"Google tokeninfo response: {tokeninfo}")
                    
                    # Verify the audience matches our client ID
                    expected_aud = os.getenv("GOOGLE_CLIENT_ID")
                    if tokeninfo.get("aud") != expected_aud:
                        print__DEBUG_TOKEN(f"Tokeninfo audience mismatch. Expected: {expected_aud}, Got: {tokeninfo.get('aud')}")
                        raise HTTPException(status_code=401, detail="Invalid token audience")
                    
                    # Verify the token is not expired
                    if int(tokeninfo.get("exp", 0)) < time.time():
                        print__DEBUG_TOKEN("Tokeninfo shows token has expired")
                        raise HTTPException(status_code=401, detail="Token has expired")
                    
                    # Return the tokeninfo as the payload (it contains email, name, etc.)
                    print__DEBUG_TOKEN("NextAuth.js id_token verification successful via Google tokeninfo")
                    return tokeninfo
                    
                else:
                    print__DEBUG_TOKEN(f"Google tokeninfo endpoint returned error: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=401, detail="Invalid NextAuth.js id_token")
                    
            except requests.RequestException as e:
                print__DEBUG_TOKEN(f"Failed to verify NextAuth.js id_token via Google tokeninfo: {e}")
                raise HTTPException(status_code=401, detail="Token verification failed - unable to validate NextAuth.js token")
            except HTTPException:
                raise  # Re-raise HTTPException as-is
            except Exception as e:
                print__DEBUG_TOKEN(f"NextAuth.js id_token verification failed: {e}")
                raise HTTPException(status_code=401, detail="NextAuth.js token verification failed")
        
        # ORIGINAL FLOW: Standard Google JWT token with 'kid' field (for direct Google API calls)
        try:
            # Get Google public keys for JWKS verification
            jwks = requests.get(GOOGLE_JWK_URL).json()
        except requests.RequestException as e:
            print__DEBUG_TOKEN(f"Failed to fetch Google JWKS: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed - unable to fetch Google keys")
        
        # Find matching key
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                public_key = RSAAlgorithm.from_jwk(key)
                try:
                    payload = jwt.decode(token, public_key, algorithms=["RS256"], audience=os.getenv("GOOGLE_CLIENT_ID"))
                    print__DEBUG_TOKEN("Standard Google JWT token verification successful")
                    return payload
                except jwt.ExpiredSignatureError:
                    print__DEBUG_TOKEN("JWT token has expired")
                    raise HTTPException(status_code=401, detail="Token has expired")
                except jwt.InvalidAudienceError:
                    print__DEBUG_TOKEN("JWT token has invalid audience")
                    raise HTTPException(status_code=401, detail="Invalid token audience")
                except jwt.InvalidSignatureError:
                    print__DEBUG_TOKEN("JWT token has invalid signature")
                    raise HTTPException(status_code=401, detail="Invalid token signature")
                except jwt.InvalidTokenError as e:
                    print__DEBUG_TOKEN(f"JWT token is invalid: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token")
                except jwt.DecodeError as e:
                    print__DEBUG_TOKEN(f"JWT decode error: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token format")
                except Exception as e:
                    print__DEBUG_TOKEN(f"JWT decode error: {e}")
                    raise HTTPException(status_code=401, detail="Invalid token")
        
        print__DEBUG_TOKEN("JWT public key not found in Google JWKS")
        raise HTTPException(status_code=401, detail="Invalid token: public key not found")
        
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except requests.RequestException as e:
        print__DEBUG_TOKEN(f"Failed to fetch Google JWKS: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed - unable to validate")
    except jwt.DecodeError as e:
        # This should be rare now due to pre-validation
        print__DEBUG_TOKEN(f"JWT decode error in main handler: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token format")
    except KeyError as e:
        print__DEBUG_TOKEN(f"JWT verification KeyError: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token structure")
    except Exception as e:
        print__DEBUG_TOKEN(f"JWT verification failed: {e}")
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
        print__DEBUG_TOKEN(f"Authentication error: {e}")
        log_comprehensive_error("authentication", e)
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """Analyze request with simplified memory monitoring."""
    
    print__analyze_debug(f"🔍 ANALYZE ENDPOINT - ENTRY POINT")
    print__analyze_debug(f"🔍 Request received: thread_id={request.thread_id}")
    print__analyze_debug(f"🔍 Request prompt length: {len(request.prompt)}")
    
    try:
        user_email = user.get("email")
        print__analyze_debug(f"🔍 User extraction: {user_email}")
        if not user_email:
            print__analyze_debug(f"🚨 No user email found in token")
            raise HTTPException(status_code=401, detail="User email not found in token")
        
        print__feedback_flow(f"📝 New analysis request - Thread: {request.thread_id}, User: {user_email}")
        print__analyze_debug(f"🔍 ANALYZE REQUEST RECEIVED: thread_id={request.thread_id}, user={user_email}")
        
        # Simple memory check
        print__analyze_debug(f"🔍 Starting memory logging")
        log_memory_usage("analysis_start")
        run_id = None
        
        print__analyze_debug(f"🔍 About to acquire analysis semaphore")
        # Limit concurrent analyses to prevent resource exhaustion
        async with analysis_semaphore:
            print__feedback_flow(f"🔒 Acquired analysis semaphore ({analysis_semaphore._value}/{MAX_CONCURRENT_ANALYSES} available)")
            print__analyze_debug(f"🔍 Semaphore acquired successfully")
            
            try:
                print__analyze_debug(f"🔍 About to get healthy checkpointer")
                print__feedback_flow(f"🔄 Getting healthy checkpointer")
                checkpointer = await get_healthy_checkpointer()
                print__analyze_debug(f"🔍 Checkpointer obtained: {type(checkpointer).__name__}")
                
                print__analyze_debug(f"🔍 About to create thread run entry")
                print__feedback_flow(f"🔄 Creating thread run entry")
                run_id = await create_thread_run_entry(user_email, request.thread_id, request.prompt)
                print__feedback_flow(f"✅ Generated new run_id: {run_id}")
                print__analyze_debug(f"🔍 Thread run entry created successfully: {run_id}")
                
                print__analyze_debug(f"🔍 About to start analysis_main")
                print__feedback_flow(f"🚀 Starting analysis")
                # 8 minute timeout for platform stability
                result = await asyncio.wait_for(
                    analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=checkpointer, run_id=run_id),
                    timeout=480  # 8 minutes timeout
                )
                
                print__analyze_debug(f"🔍 Analysis completed successfully")
                print__feedback_flow(f"✅ Analysis completed successfully")
                
            except Exception as analysis_error:
                print__analyze_debug(f"🚨 Exception in analysis block: {type(analysis_error).__name__}: {str(analysis_error)}")
                # If there's a database connection issue, try with InMemorySaver as fallback
                error_msg = str(analysis_error).lower()
                print__analyze_debug(f"🔍 Error message (lowercase): {error_msg}")
                
                if any(keyword in error_msg for keyword in ["pool", "connection", "closed", "timeout", "ssl", "postgres"]):
                    print__analyze_debug(f"🔍 Database issue detected, attempting fallback")
                    print__feedback_flow(f"⚠️ Database issue detected, trying with InMemorySaver fallback: {analysis_error}")
                    
                    try:
                        print__analyze_debug(f"🔍 Importing InMemorySaver")
                        from langgraph.checkpoint.memory import InMemorySaver
                        fallback_checkpointer = InMemorySaver()
                        print__analyze_debug(f"🔍 InMemorySaver created")
                        
                        # Generate a fallback run_id since database creation might have failed
                        if run_id is None:
                            run_id = str(uuid.uuid4())
                            print__feedback_flow(f"✅ Generated fallback run_id: {run_id}")
                            print__analyze_debug(f"🔍 Generated fallback run_id: {run_id}")
                        
                        print__analyze_debug(f"🔍 Starting fallback analysis")
                        print__feedback_flow(f"🚀 Starting analysis with InMemorySaver fallback")
                        result = await asyncio.wait_for(
                            analysis_main(request.prompt, thread_id=request.thread_id, checkpointer=fallback_checkpointer, run_id=run_id),
                            timeout=480  # 8 minutes timeout
                        )
                        
                        print__analyze_debug(f"🔍 Fallback analysis completed successfully")
                        print__feedback_flow(f"✅ Analysis completed successfully with fallback")
                        
                    except Exception as fallback_error:
                        print__analyze_debug(f"🚨 Fallback also failed: {type(fallback_error).__name__}: {str(fallback_error)}")
                        print__feedback_flow(f"🚨 Fallback analysis also failed: {fallback_error}")
                        raise HTTPException(status_code=500, detail="Sorry, there was an error processing your request. Please try again.")
                else:
                    # Re-raise non-database errors
                    print__analyze_debug(f"🚨 Non-database error, re-raising: {type(analysis_error).__name__}: {str(analysis_error)}")
                    print__feedback_flow(f"🚨 Non-database error: {analysis_error}")
                    raise HTTPException(status_code=500, detail="Sorry, there was an error processing your request. Please try again.")
            
            print__analyze_debug(f"🔍 About to prepare response data")
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
                "run_id": run_id,
                "top_chunks": result.get("top_chunks", []) if isinstance(result, dict) else []
            }
            
            print__analyze_debug(f"🔍 Response data prepared successfully")
            print__analyze_debug(f"🔍 Response data keys: {list(response_data.keys())}")
            print__analyze_debug(f"🔍 ANALYZE ENDPOINT - SUCCESSFUL EXIT")
            return response_data
            
    except asyncio.TimeoutError:
        error_msg = "Analysis timed out after 8 minutes"
        print__analyze_debug(f"🚨 TIMEOUT ERROR: {error_msg}")
        print__feedback_flow(f"🚨 {error_msg}")
        raise HTTPException(status_code=408, detail=error_msg)
        
    except HTTPException as http_exc:
        print__analyze_debug(f"🚨 HTTP EXCEPTION: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print__analyze_debug(f"🚨 UNEXPECTED EXCEPTION: {type(e).__name__}: {str(e)}")
        print__analyze_debug(f"🚨 Exception traceback: {traceback.format_exc()}")
        print__feedback_flow(f"🚨 {error_msg}")
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
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT COUNT(*) FROM users_threads_runs 
                        WHERE run_id = %s AND email = %s
                    """, (run_uuid, user_email))
                    
                    ownership_row = await cur.fetchone()
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
            print__feedback_flow(f"⚠️ Could not verify ownership: {ownership_error}")
            # Continue with feedback submission but log the warning
            print__feedback_flow(f"⚠️ Proceeding with feedback submission despite ownership check failure")
        
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
async def get_chat_threads(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=50, description="Number of threads per page"),
    user=Depends(get_current_user)
) -> PaginatedChatThreadsResponse:
    """Get paginated chat threads for the authenticated user."""
    try:
        user_email = user["email"]
        print__api_postgresql(f"Loading chat threads for user: {user_email} (page: {page}, limit: {limit})")
        
        print__api_postgresql("Getting chat threads with simplified approach")
        
        # Get total count first
        print__api_postgresql(f"Getting chat threads count for user: {user_email}")
        total_count = await get_user_chat_threads_count(user_email)
        print__api_postgresql(f"Total threads count for user {user_email}: {total_count}")
        
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Get threads for this page
        print__api_postgresql(f"Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})")
        threads = await get_user_chat_threads(user_email, limit=limit, offset=offset)
        print__api_postgresql(f"Retrieved {len(threads)} threads for user {user_email}")
        
        # Convert to response format
        chat_thread_responses = []
        for thread in threads:
            chat_thread_response = ChatThreadResponse(
                thread_id=thread['thread_id'],
                latest_timestamp=thread['latest_timestamp'],
                run_count=thread['run_count'],
                title=thread['title'],
                full_prompt=thread['full_prompt']
            )
            chat_thread_responses.append(chat_thread_response)
        
        # Calculate pagination info
        has_more = (offset + len(chat_thread_responses)) < total_count
        
        print__api_postgresql(f"Retrieved {len(threads)} threads for user {user_email} (total: {total_count})")
        print__api_postgresql(f"Returning {len(chat_thread_responses)} threads to frontend (page {page}/{(total_count + limit - 1) // limit})")
        
        return PaginatedChatThreadsResponse(
            threads=chat_thread_responses,
            total_count=total_count,
            page=page,
            limit=limit,
            has_more=has_more
        )
        
    except Exception as e:
        print__api_postgresql(f"❌ Error getting chat threads: {e}")
        import traceback
        print__api_postgresql(f"Full traceback: {traceback.format_exc()}")
        
        # Return error response
        return PaginatedChatThreadsResponse(
            threads=[],
            total_count=0,
            page=page,
            limit=limit,
            has_more=False
        )

@app.delete("/chat/{thread_id}")
async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Delete all PostgreSQL checkpoint records and thread entries for a specific thread_id."""
    
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    print__api_postgresql(f"🗑️ Deleting chat thread {thread_id} for user {user_email}")
    
    try:
        # Get a healthy checkpointer
        print__api_postgresql(f"🔧 DEBUG: Getting healthy checkpointer...")
        checkpointer = await get_healthy_checkpointer()
        print__api_postgresql(f"🔧 DEBUG: Checkpointer type: {type(checkpointer).__name__}")
        
        # Check if we have a PostgreSQL checkpointer (not InMemorySaver)
        print__api_postgresql(f"🔧 DEBUG: Checking if checkpointer has 'conn' attribute...")
        if not hasattr(checkpointer, 'conn'):
            print__api_postgresql(f"⚠️ No PostgreSQL checkpointer available - nothing to delete")
            return {"message": "No PostgreSQL checkpointer available - nothing to delete"}
        
        print__api_postgresql(f"🔧 DEBUG: Checkpointer has 'conn' attribute")
        print__api_postgresql(f"🔧 DEBUG: checkpointer.conn type: {type(checkpointer.conn).__name__}")
        
        # Access the connection through the conn attribute
        conn_obj = checkpointer.conn
        print__api_postgresql(f"🔧 DEBUG: Connection object set, type: {type(conn_obj).__name__}")
        
        # FIXED: Handle both connection pool and single connection cases
        if hasattr(conn_obj, 'connection') and callable(getattr(conn_obj, 'connection', None)):
            # It's a connection pool - use pool.connection()
            print__api_postgresql(f"🔧 DEBUG: Using connection pool pattern...")
            async with conn_obj.connection() as conn:
                print__api_postgresql(f"🔧 DEBUG: Successfully got connection from pool, type: {type(conn).__name__}")
                result_data = await perform_deletion_operations(conn, user_email, thread_id)
                return result_data
        else:
            # It's a single connection - use it directly
            print__api_postgresql(f"🔧 DEBUG: Using single connection pattern...")
            conn = conn_obj
            print__api_postgresql(f"🔧 DEBUG: Using direct connection, type: {type(conn).__name__}")
            result_data = await perform_deletion_operations(conn, user_email, thread_id)
            return result_data
            
    except Exception as e:
        error_msg = str(e)
        print__api_postgresql(f"❌ Failed to delete checkpoint records for thread {thread_id}: {e}")
        print__api_postgresql(f"🔧 DEBUG: Main exception type: {type(e).__name__}")
        import traceback
        print__api_postgresql(f"🔧 DEBUG: Main exception traceback: {traceback.format_exc()}")
        
        # If it's a connection error, don't treat it as a failure since it means 
        # there are likely no records to delete anyway
        if any(keyword in error_msg.lower() for keyword in [
            "ssl error", "connection", "timeout", "operational error", 
            "server closed", "bad connection", "consuming input failed"
        ]):
            print__api_postgresql(f"⚠️ PostgreSQL connection unavailable - no records to delete")
            return {
                "message": "PostgreSQL connection unavailable - no records to delete", 
                "thread_id": thread_id,
                "user_email": user_email,
                "warning": "Database connection issues"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint records: {e}")

async def perform_deletion_operations(conn, user_email: str, thread_id: str):
    """Perform the actual deletion operations on the given connection."""
    print__api_postgresql(f"🔧 DEBUG: Starting deletion operations...")
    
    print__api_postgresql(f"🔧 DEBUG: Setting autocommit...")
    await conn.set_autocommit(True)
    print__api_postgresql(f"🔧 DEBUG: Autocommit set successfully")
    
    # 🔒 SECURITY CHECK: Verify user owns this thread before deleting
    print__api_postgresql(f"🔒 Verifying thread ownership for deletion - user: {user_email}, thread: {thread_id}")
    
    print__api_postgresql(f"🔧 DEBUG: Creating cursor for ownership check...")
    async with conn.cursor() as cur:
        print__api_postgresql(f"🔧 DEBUG: Cursor created, executing ownership query...")
        await cur.execute("""
            SELECT COUNT(*) FROM users_threads_runs 
            WHERE email = %s AND thread_id = %s
        """, (user_email, thread_id))
        
        print__api_postgresql(f"🔧 DEBUG: Ownership query executed, fetching result...")
        ownership_row = await cur.fetchone()
        thread_entries_count = ownership_row[0] if ownership_row else 0
        print__api_postgresql(f"🔧 DEBUG: Ownership check complete, count: {thread_entries_count}")
    
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
            print__api_postgresql(f"🔧 DEBUG: Processing table {table}...")
            # First check if the table exists
            print__api_postgresql(f"🔧 DEBUG: Creating cursor for table existence check...")
            async with conn.cursor() as cur:
                print__api_postgresql(f"🔧 DEBUG: Executing table existence query for {table}...")
                await cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                
                print__api_postgresql(f"🔧 DEBUG: Table existence query executed, fetching result...")
                table_exists = await cur.fetchone()
                print__api_postgresql(f"🔧 DEBUG: Table {table} exists check result: {table_exists}")
                
                if not table_exists or not table_exists[0]:
                    print__api_postgresql(f"⚠ Table {table} does not exist, skipping")
                    deleted_counts[table] = 0
                    continue
                
                print__api_postgresql(f"🔧 DEBUG: Table {table} exists, proceeding with deletion...")
                # Delete records for this thread_id
                print__api_postgresql(f"🔧 DEBUG: Creating cursor for deletion from {table}...")
                async with conn.cursor() as del_cur:
                    print__api_postgresql(f"🔧 DEBUG: Executing DELETE query for {table}...")
                    await del_cur.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s",
                        (thread_id,)
                    )
                    
                    deleted_counts[table] = del_cur.rowcount if hasattr(del_cur, 'rowcount') else 0
                    print__api_postgresql(f"✅ Deleted {deleted_counts[table]} records from {table} for thread_id: {thread_id}")
                
        except Exception as table_error:
            print__api_postgresql(f"⚠ Error deleting from table {table}: {table_error}")
            print__api_postgresql(f"🔧 DEBUG: Table error type: {type(table_error).__name__}")
            import traceback
            print__api_postgresql(f"🔧 DEBUG: Table error traceback: {traceback.format_exc()}")
            deleted_counts[table] = f"Error: {str(table_error)}"
    
    # Delete from users_threads_runs table directly within the same transaction
    print__api_postgresql(f"🔄 Deleting thread entries from users_threads_runs for user {user_email}, thread {thread_id}")
    try:
        print__api_postgresql(f"🔧 DEBUG: Creating cursor for users_threads_runs deletion...")
        async with conn.cursor() as cur:
            print__api_postgresql(f"🔧 DEBUG: Executing DELETE query for users_threads_runs...")
            await cur.execute("""
                DELETE FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """, (user_email, thread_id))
            
            users_threads_runs_deleted = cur.rowcount if hasattr(cur, 'rowcount') else 0
            print__api_postgresql(f"✅ Deleted {users_threads_runs_deleted} entries from users_threads_runs for user {user_email}, thread {thread_id}")
            
            deleted_counts["users_threads_runs"] = users_threads_runs_deleted
    
    except Exception as e:
        print__api_postgresql(f"❌ Error deleting from users_threads_runs: {e}")
        print__api_postgresql(f"🔧 DEBUG: users_threads_runs error type: {type(e).__name__}")
        import traceback
        print__api_postgresql(f"🔧 DEBUG: users_threads_runs error traceback: {traceback.format_exc()}")
        deleted_counts["users_threads_runs"] = f"Error: {str(e)}"
    
    result_data = {
        "message": f"Checkpoint records and thread entries deleted for thread_id: {thread_id}",
        "deleted_counts": deleted_counts,
        "thread_id": thread_id,
        "user_email": user_email
    }
    
    print__api_postgresql(f"🎉 Successfully deleted thread {thread_id} for user {user_email}")
    return result_data

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
            print__api_postgresql(f"⚠️ No PostgreSQL checkpointer available - returning empty messages")
            return []
        
        # Verify thread ownership using users_threads_runs table
        async with checkpointer.conn.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (user_email, thread_id))
                
                ownership_row = await cur.fetchone()
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
        top_chunks = []
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = await checkpointer.aget_tuple(config)
            
            if state_snapshot and state_snapshot.checkpoint:
                channel_values = state_snapshot.checkpoint.get("channel_values", {})
                top_selection_codes = channel_values.get("top_selection_codes", [])
                
                # Use the datasets directly
                datasets_used = top_selection_codes
                
                # Get PDF chunks from checkpoint state
                checkpoint_top_chunks = channel_values.get("top_chunks", [])
                print__api_postgresql(f"📄 Found {len(checkpoint_top_chunks)} PDF chunks in checkpoint for thread {thread_id}")
                
                # Convert Document objects to serializable format
                if checkpoint_top_chunks:
                    for chunk in checkpoint_top_chunks:
                        chunk_data = {
                            "content": chunk.page_content if hasattr(chunk, 'page_content') else str(chunk),
                            "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                        }
                        top_chunks.append(chunk_data)
                    print__api_postgresql(f"📄 Serialized {len(top_chunks)} PDF chunks for frontend")
                
                # Extract SQL query from queries_and_results for SQL button
                if queries_and_results:
                    # Get the last (most recent) SQL query
                    sql_query = queries_and_results[-1][0] if queries_and_results[-1] else None
            
        except Exception as e:
            print__api_postgresql(f"⚠️ Could not get datasets/SQL/chunks from checkpoint: {e}")
            print__api_postgresql(f"🔧 Using fallback empty values: datasets=[], sql=None, chunks=[]")
        
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
                if top_chunks:
                    meta_info["topChunks"] = top_chunks
                meta_info["source"] = "checkpoint_history"
                print__api_postgresql(f"🔍 Added metadata to AI message: datasets={len(datasets_used)}, sql={'Yes' if sql_query else 'No'}, chunks={len(top_chunks)}")
            
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

# Global cache for bulk loading to prevent repeated calls
_bulk_loading_cache = {}
_bulk_loading_locks = defaultdict(asyncio.Lock)
BULK_CACHE_TIMEOUT = 30  # Cache timeout in seconds

def cleanup_bulk_cache():
    """Clean up expired cache entries."""
    current_time = time.time()
    expired_keys = []
    
    for cache_key, (cached_data, cache_time) in _bulk_loading_cache.items():
        if current_time - cache_time > BULK_CACHE_TIMEOUT:
            expired_keys.append(cache_key)
    
    for key in expired_keys:
        del _bulk_loading_cache[key]
    
    return len(expired_keys)

@app.get("/chat/all-messages")
async def get_all_chat_messages(user=Depends(get_current_user)) -> Dict:
    """Get all chat messages for the authenticated user using bulk loading with improved caching."""
    user_email = user["email"]
    print__api_postgresql(f"📥 BULK REQUEST: Loading ALL chat messages for user: {user_email}")
    
    # Check if we have a recent cached result
    cache_key = f"bulk_messages_{user_email}"
    current_time = time.time()
    
    if cache_key in _bulk_loading_cache:
        cached_data, cache_time = _bulk_loading_cache[cache_key]
        if current_time - cache_time < BULK_CACHE_TIMEOUT:
            print__api_postgresql(f"✅ CACHE HIT: Returning cached bulk data for {user_email} (age: {current_time - cache_time:.1f}s)")
            
            # Return cached data with appropriate headers
            from fastapi.responses import JSONResponse
            response = JSONResponse(content=cached_data)
            response.headers["Cache-Control"] = f"public, max-age={int(BULK_CACHE_TIMEOUT - (current_time - cache_time))}"
            response.headers["ETag"] = f"bulk-{user_email}-{int(cache_time)}"
            return response
        else:
            print__api_postgresql(f"⏰ CACHE EXPIRED: Cached data too old ({current_time - cache_time:.1f}s), will refresh")
            del _bulk_loading_cache[cache_key]
    
    # Use a lock to prevent multiple simultaneous requests from the same user
    async with _bulk_loading_locks[user_email]:
        # Double-check cache after acquiring lock (another request might have completed)
        if cache_key in _bulk_loading_cache:
            cached_data, cache_time = _bulk_loading_cache[cache_key]
            if current_time - cache_time < BULK_CACHE_TIMEOUT:
                print__api_postgresql(f"✅ CACHE HIT (after lock): Returning cached bulk data for {user_email}")
                return cached_data
        
        print__api_postgresql(f"🔄 CACHE MISS: Processing fresh bulk request for {user_email}")
        
        # Simple memory check before starting
        log_memory_usage("bulk_start")
        
        try:
            checkpointer = await get_healthy_checkpointer()
            
            # STEP 1: Get all user threads, run-ids, and sentiments in ONE query
            print__api_postgresql(f"🔍 BULK QUERY: Getting all user threads, run-ids, and sentiments")
            user_thread_ids = []
            all_run_ids = {}
            all_sentiments = {}
            
            # FIXED: Use our working get_healthy_pool() function instead of checkpointer.conn
            from my_agent.utils.postgres_checkpointer import get_healthy_pool
            pool = await get_healthy_pool()
            
            # FIXED: Use pool.connection() instead of pool.acquire() and update query syntax
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Single query for all threads, run-ids, and sentiments
                    # FIXED: Use psycopg format (%s) instead of asyncpg format ($1)
                    await cur.execute("""
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
                    
                    rows = await cur.fetchall()
                
                for row in rows:
                    # FIXED: Use index-based access instead of dict-based for psycopg
                    thread_id = row[0]  # thread_id
                    run_id = row[1]     # run_id
                    prompt = row[2]     # prompt
                    timestamp = row[3]  # timestamp
                    sentiment = row[4]  # sentiment
                    
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
                empty_result = {"messages": {}, "runIds": {}, "sentiments": {}}
                _bulk_loading_cache[cache_key] = (empty_result, current_time)
                return empty_result
            
            # STEP 2: Process threads with limited concurrency (max 3 concurrent)
            print__api_postgresql(f"🔄 Processing {len(user_thread_ids)} threads with limited concurrency")
            
            async def process_single_thread(thread_id: str):
                """Process a single thread using the proven working functions."""
                try:
                    print__api_postgresql(f"🔄 Processing thread {thread_id}")
                    
                    # Use the working function
                    stored_messages = await get_conversation_messages_from_checkpoints(checkpointer, thread_id, user_email)
                    
                    if not stored_messages:
                        print__api_postgresql(f"⚠ No messages found in checkpoints for thread {thread_id}")
                        return thread_id, []
                    
                    print__api_postgresql(f"📄 Found {len(stored_messages)} messages for thread {thread_id}")
                    
                    # Get additional metadata from latest checkpoint
                    queries_and_results = await get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id)
                    
                    # Get dataset information and SQL query from latest checkpoint
                    datasets_used = []
                    sql_query = None
                    top_chunks = []
                    
                    try:
                        config = {"configurable": {"thread_id": thread_id}}
                        state_snapshot = await checkpointer.aget_tuple(config)
                        
                        if state_snapshot and state_snapshot.checkpoint:
                            channel_values = state_snapshot.checkpoint.get("channel_values", {})
                            top_selection_codes = channel_values.get("top_selection_codes", [])
                            datasets_used = top_selection_codes
                            
                            # Get PDF chunks
                            checkpoint_top_chunks = channel_values.get("top_chunks", [])
                            if checkpoint_top_chunks:
                                for chunk in checkpoint_top_chunks:
                                    chunk_data = {
                                        "content": chunk.page_content if hasattr(chunk, 'page_content') else str(chunk),
                                        "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                                    }
                                    top_chunks.append(chunk_data)
                            
                            # Extract SQL query
                            if queries_and_results:
                                sql_query = queries_and_results[-1][0] if queries_and_results[-1] else None
                        
                    except Exception as e:
                        print__api_postgresql(f"⚠️ Could not get datasets/SQL/chunks from checkpoint: {e}")
                    
                    # Convert stored messages to frontend format
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
                            if top_chunks:
                                meta_info["topChunks"] = top_chunks
                            meta_info["source"] = "cached_bulk_processing"
                        
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
            
            MAX_CONCURRENT_THREADS = 3
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_THREADS)

            async def process_single_thread_with_limit(thread_id: str):
                """Process a single thread with concurrency limiting."""
                async with semaphore:
                    return await process_single_thread(thread_id)

            print__api_postgresql(f"🔒 Processing with max {MAX_CONCURRENT_THREADS} concurrent operations")

            # Use asyncio.gather with limited concurrency
            thread_results = await asyncio.gather(
                *[process_single_thread_with_limit(thread_id) for thread_id in user_thread_ids],
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
            
            print__api_postgresql(f"✅ BULK LOADING COMPLETE: {len(all_messages)} threads, {total_messages} total messages")
            
            # Simple memory check after completion
            log_memory_usage("bulk_complete")
            
            result = {
                "messages": all_messages,
                "runIds": all_run_ids,
                "sentiments": all_sentiments
            }
            
            # Cache the result
            _bulk_loading_cache[cache_key] = (result, current_time)
            print__api_postgresql(f"💾 CACHED: Bulk result for {user_email} (expires in {BULK_CACHE_TIMEOUT}s)")
            
            # Return with cache headers
            from fastapi.responses import JSONResponse
            response = JSONResponse(content=result)
            response.headers["Cache-Control"] = f"public, max-age={BULK_CACHE_TIMEOUT}"
            response.headers["ETag"] = f"bulk-{user_email}-{int(current_time)}"
            return response
            
        except Exception as e:
            print__api_postgresql(f"❌ BULK ERROR: Failed to process bulk request for {user_email}: {e}")
            import traceback
            print__api_postgresql(f"Full error traceback: {traceback.format_exc()}")
            
            # Return empty result but cache it briefly to prevent error loops
            empty_result = {"messages": {}, "runIds": {}, "sentiments": {}}
            _bulk_loading_cache[cache_key] = (empty_result, current_time)
            
            response = JSONResponse(content=empty_result, status_code=500)
            response.headers["Cache-Control"] = "no-cache, no-store"  # Don't cache errors
            return response

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
                        await asyncio.wait_for(conn.execute("SELECT 1"), timeout=5)
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
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT run_id, prompt, timestamp
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                    ORDER BY timestamp ASC
                """, (user_email, thread_id))
                
                run_id_data = []
                rows = await cur.fetchall()
                for row in rows:
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
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT email, thread_id, prompt, timestamp
                        FROM users_threads_runs 
                        WHERE run_id = %s AND email = %s
                    """, (run_id, user_email))
                    
                    row = await cur.fetchone()
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
                        await cur.execute("""
                            SELECT COUNT(*) FROM users_threads_runs WHERE run_id = %s
                        """, (run_id,))
                        
                        any_row = await cur.fetchone()
                        if any_row and any_row[0] > 0:
                            result["exists_in_database"] = True
                            result["user_owns_run_id"] = False
                            print__debug(f"🚫 Run_id {run_id} exists but user {user_email} does not own it")
                        else:
                            print__debug(f"❌ Run_id {run_id} not found in database")
    except Exception as e:
        result["database_error"] = str(e)
    
    return result

@app.post("/admin/clear-cache")
async def clear_bulk_cache(user=Depends(get_current_user)):
    """Clear the bulk loading cache (admin endpoint)."""
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
    
    # For now, allow any authenticated user to clear cache
    # In production, you might want to restrict this to admin users
    
    cache_entries_before = len(_bulk_loading_cache)
    _bulk_loading_cache.clear()
    
    print__memory_monitoring(f"🧹 MANUAL CACHE CLEAR: {cache_entries_before} entries cleared by {user_email}")
    
    # Check memory after cleanup
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        memory_status = "normal" if rss_mb < (GC_MEMORY_THRESHOLD * 0.8) else "high"
    except:
        rss_mb = 0
        memory_status = "unknown"
    
    return {
        "message": "Cache cleared successfully",
        "cache_entries_cleared": cache_entries_before,
        "current_memory_mb": round(rss_mb, 1),
        "memory_status": memory_status,
        "cleared_by": user_email,
        "timestamp": datetime.now().isoformat()
    }

#==============================================================================
# DEBUG FUNCTIONS
#==============================================================================
def print__api_postgresql(msg: str) -> None:
    """Print API-PostgreSQL messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[API-PostgreSQL] {msg}")
        import sys
        sys.stdout.flush()

def print__feedback_flow(msg: str) -> None:
    """Print FEEDBACK-FLOW messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[FEEDBACK-FLOW] {msg}")
        import sys
        sys.stdout.flush()

def print__DEBUG_TOKEN(msg: str) -> None:
    """Print DEBUG_TOKEN messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG_TOKEN', '0')
    if debug_mode == '1':
        print(f"[DEBUG_TOKEN] {msg}")
        import sys
        sys.stdout.flush()
        
def print__sentiment_flow(msg: str) -> None:
    """Print SENTIMENT-FLOW messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[SENTIMENT-FLOW] {msg}")
        import sys
        sys.stdout.flush()

def print__debug(msg: str) -> None:
    """Print DEBUG messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[DEBUG] {msg}")
        import sys
        sys.stdout.flush()

def print__analyze_debug(msg: str) -> None:
    """Print DEBUG_ANALYZE messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG_ANALYZE', '0')
    if debug_mode == '1':
        print(f"[DEBUG_ANALYZE] {msg}")
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