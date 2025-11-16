"""
MODULE_DESCRIPTION: Health Check Endpoints - System Monitoring and Diagnostics

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module implements comprehensive health check endpoints for monitoring the
operational status, performance, and resource utilization of the CZSU Multi-Agent
Text-to-SQL API. It provides five specialized health check endpoints:

1. /health - Overall system health check
2. /health/database - Database connectivity and performance
3. /health/memory - Memory usage and cache monitoring
4. /health/rate-limits - Rate limiting status
5. /health/prepared-statements - PostgreSQL prepared statement tracking

These endpoints enable:
    - Production monitoring and alerting
    - Performance troubleshooting
    - Resource optimization
    - Capacity planning
    - Incident response

Health Check Philosophy:
    - Fast response times (< 100ms)
    - No authentication required (monitoring tools need access)
    - Standard HTTP status codes (200, 503, 500)
    - Detailed diagnostic information
    - Actionable metrics

===================================================================================
KEY FEATURES
===================================================================================

1. Overall System Health (/health)
   - Uptime tracking
   - Memory usage monitoring
   - Database connectivity verification
   - Garbage collection metrics
   - Version information
   - HTTP 503 on degraded state

2. Database Health (/health/database)
   - Connection pool status
   - Query latency measurement
   - AsyncPostgresSaver verification
   - Fallback detection (memory vs PostgreSQL)
   - Read operation testing

3. Memory Health (/health/memory)
   - RSS memory usage tracking
   - Cache entry monitoring
   - Automatic cache cleanup
   - Memory threshold alerting
   - Scaling recommendations
   - Memory per thread estimation

4. Rate Limit Health (/health/rate-limits)
   - Active client tracking
   - Rate limit configuration display
   - Client usage statistics
   - Window and burst settings

5. Prepared Statements Health (/health/prepared-statements)
   - PostgreSQL prepared statement counting
   - Statement name listing
   - Connection configuration verification
   - Prepared statement bloat detection

6. Automatic Maintenance
   - Garbage collection on health check
   - Expired cache cleanup
   - Resource optimization
   - Proactive memory management

7. Production-Ready Design
   - No authentication required
   - Suitable for load balancers
   - Prometheus/Grafana compatible
   - Standard monitoring patterns

===================================================================================
API ENDPOINTS
===================================================================================

GET /health
    Overall system health check with memory and database verification

    Authentication: None (publicly accessible)

    Returns:
        {
            "status": "healthy",  // or "degraded", "error"
            "timestamp": "2024-01-15T10:30:00",
            "uptime_seconds": 3600.5,
            "memory": {
                "rss_mb": 850.5,
                "vms_mb": 1200.0,
                "percent": 15.2
            },
            "database": {
                "healthy": true,
                "checkpointer_type": "AsyncPostgresSaver",
                "error": null
            },
            "garbage_collector": {
                "objects_collected": 150,
                "gc_run": true
            },
            "version": "1.0.0"
        }

    Status Codes:
        200: System healthy
        503: System degraded (database issues)
        500: System error

GET /health/database
    Detailed database connectivity and performance check

    Authentication: None

    Returns:
        {
            "timestamp": "2024-01-15T10:30:00",
            "checkpointer_available": true,
            "checkpointer_type": "AsyncPostgresSaver",
            "database_connection": "healthy",
            "read_latency_ms": 12.5,
            "read_test": "passed"
        }

    Status Codes:
        200: Database healthy
        503: Database degraded/error
        500: Check failed

GET /health/memory
    Memory usage, cache status, and scaling recommendations

    Authentication: None

    Returns:
        {
            "status": "healthy",  // or "warning", "high_memory"
            "memory_rss_mb": 850.5,
            "memory_threshold_mb": 1900,
            "memory_usage_percent": 44.8,
            "over_threshold": false,
            "cache_info": {
                "active_cache_entries": 25,
                "cleaned_expired_entries": 5,
                "cache_timeout_seconds": 60
            },
            "scaling_info": {
                "estimated_memory_per_thread_mb": 34.0,
                "estimated_max_threads_at_threshold": 55,
                "current_thread_count": 25
            },
            "timestamp": "2024-01-15T10:30:00"
        }

    Status Levels:
        healthy: < 80% of threshold
        warning: 80-100% of threshold
        high_memory: > threshold

GET /health/rate-limits
    Rate limiting configuration and usage statistics

    Authentication: None

    Returns:
        {
            "status": "healthy",
            "total_tracked_clients": 150,
            "active_clients": 45,
            "rate_limit_window": 60,
            "rate_limit_requests": 100,
            "rate_limit_burst": 20,
            "timestamp": "2024-01-15T10:30:00"
        }

GET /health/prepared-statements
    PostgreSQL prepared statement monitoring

    Authentication: None

    Returns:
        {
            "status": "healthy",
            "checkpointer_status": "healthy",
            "prepared_statements_count": 15,
            "prepared_statement_names": "_pg3_stmt1, _pg3_stmt2, ...",
            "connection_kwargs": {
                "prepare_threshold": null
            },
            "timestamp": "2024-01-15T10:30:00"
        }

===================================================================================
OVERALL SYSTEM HEALTH CHECK
===================================================================================

Purpose:
    Primary health check endpoint for load balancers, monitoring systems, and
    uptime checks. Provides high-level system status.

Health Checks Performed:
    1. Memory Usage
       - RSS (Resident Set Size) in MB
       - VMS (Virtual Memory Size) in MB
       - Memory percentage of total system

    2. Database Connectivity
       - Verify GLOBAL_CHECKPOINTER exists
       - Test AsyncPostgresSaver with aget_tuple()
       - Measure query latency
       - Detect fallback to memory storage

    3. Garbage Collection
       - Force GC run on every health check
       - Count collected objects
       - Free unused memory

Status Determination:
    - healthy: All systems operational
    - degraded: Database issues detected
    - error: Health check itself failed

HTTP Status Codes:
    200: System healthy
    503: System degraded (database down)
    500: Health check error

Use Cases:
    - Load balancer health checks (remove unhealthy instances)
    - Uptime monitoring (Pingdom, UptimeRobot)
    - Alerting triggers (PagerDuty, Slack)
    - Kubernetes liveness probes
    - API status pages

Response Time:
    Target: < 100ms
    Typical: 20-50ms (fast database check)
    Max acceptable: 1000ms

===================================================================================
DATABASE HEALTH CHECK
===================================================================================

Purpose:
    Detailed database connectivity and performance monitoring for diagnosing
    database-related issues.

Tests Performed:
    1. Checkpointer Availability
       - Verify GLOBAL_CHECKPOINTER is initialized
       - Check checkpointer type (AsyncPostgresSaver vs MemorySaver)

    2. Connection Test
       - Execute aget_tuple() read operation
       - Measure query latency in milliseconds
       - Verify no connection errors

    3. Fallback Detection
       - Detect if using memory fallback
       - Identify PostgreSQL unavailability

Metrics Returned:
    - read_latency_ms: Database query response time
    - checkpointer_type: Expected "AsyncPostgresSaver"
    - database_connection: healthy/error/using_memory_fallback
    - read_test: passed/failed

Alerting Thresholds:
    - Latency > 100ms: Warning
    - Latency > 500ms: Critical
    - read_test failed: Critical
    - Fallback mode: Warning

Use Cases:
    - Database performance monitoring
    - Connection pool exhaustion detection
    - Network latency tracking
    - Database failover verification

===================================================================================
MEMORY HEALTH CHECK
===================================================================================

Purpose:
    Monitor memory usage, cache status, and provide scaling recommendations to
    prevent OOM (Out Of Memory) errors.

Metrics Collected:
    1. Memory Usage
       - memory_rss_mb: Resident Set Size (actual RAM usage)
       - memory_threshold_mb: Configured limit (GC_MEMORY_THRESHOLD)
       - memory_usage_percent: Usage as % of threshold
       - over_threshold: Boolean flag for critical state

    2. Cache Status
       - active_cache_entries: Number of cached bulk loads
       - cleaned_expired_entries: Entries removed this check
       - cache_timeout_seconds: TTL configuration

    3. Scaling Recommendations
       - estimated_memory_per_thread_mb: Average memory per cached thread
       - estimated_max_threads_at_threshold: Capacity before hitting limit
       - current_thread_count: Current cache size

Automatic Cleanup:
    - Calls cleanup_bulk_cache() on every check
    - Removes expired cache entries
    - Frees memory proactively
    - Prevents cache bloat

Status Levels:
    healthy: < 80% of threshold
        - System has headroom
        - No action needed

    warning: 80-100% of threshold
        - System approaching limit
        - Monitor closely
        - Consider scaling

    high_memory: > threshold
        - System over limit
        - GC will trigger aggressively
        - Risk of performance degradation
        - Action required

Scaling Guidance:
    - estimated_memory_per_thread_mb: ~30-40MB typical
    - If approaching threshold:
      * Clear cache (/admin/clear-cache)
      * Reduce MAX_CONCURRENT_BULK_THREADS
      * Increase server memory
      * Scale horizontally

Use Cases:
    - Capacity planning
    - Memory leak detection
    - Cache size optimization
    - Scaling decisions

===================================================================================
RATE LIMIT HEALTH CHECK
===================================================================================

Purpose:
    Monitor rate limiting status and track client usage patterns.

Metrics Provided:
    1. Client Tracking
       - total_tracked_clients: All clients in rate limit storage
       - active_clients: Clients with pending requests

    2. Configuration
       - rate_limit_window: Time window in seconds (default 60)
       - rate_limit_requests: Max requests per window (default 100)
       - rate_limit_burst: Burst allowance (default 20)

How Rate Limiting Works:
    - Each client IP tracked separately
    - Sliding window algorithm
    - Request timestamps stored
    - Automatic cleanup of old requests
    - Burst allowance for spikes

Status Information:
    - Always returns "healthy" unless error
    - High active_clients may indicate:
      * Heavy usage
      * Potential abuse
      * Need for scaling

Use Cases:
    - Detect abuse patterns
    - Monitor API usage
    - Capacity planning
    - Rate limit tuning

Alerting:
    - active_clients > 100: High usage
    - Frequent rate limit hits: Consider increasing limits

===================================================================================
PREPARED STATEMENTS HEALTH CHECK
===================================================================================

Purpose:
    Monitor PostgreSQL prepared statement usage to prevent memory bloat and
    connection pool issues.

What Are Prepared Statements:
    - Pre-compiled SQL queries stored in database
    - Improve performance via query plan caching
    - Consume database memory
    - Can accumulate over time (bloat)

Checks Performed:
    1. Checkpointer Status
       - Verify checkpointer availability
       - Check initialization status

    2. Database Query
       - Count prepared statements matching _pg3_% or _pg_%
       - List statement names
       - Query pg_prepared_statements system table

    3. Connection Configuration
       - Verify prepare_threshold setting
       - Check if prepared statements disabled

Metrics:
    - prepared_statements_count: Number of statements
    - prepared_statement_names: Comma-separated list
    - connection_kwargs: Configuration including prepare_threshold

Healthy Thresholds:
    - 0-50 statements: Normal
    - 50-200 statements: Monitor
    - > 200 statements: Consider cleanup

Prepared Statement Bloat:
    Symptoms:
        - High prepared_statements_count
        - Database memory usage high
        - Connection pool exhaustion

    Solutions:
        - Call /admin/clear-prepared-statements
        - Set prepare_threshold=null (disable caching)
        - Restart database connections

Use Cases:
    - Database memory monitoring
    - Connection pool health
    - Performance optimization
    - Troubleshooting slow queries

===================================================================================
AUTOMATIC MAINTENANCE
===================================================================================

Garbage Collection:
    - Triggered on /health endpoint
    - Runs gc.collect()
    - Frees unreferenced objects
    - Returns object count collected
    - Prevents memory leaks

Cache Cleanup:
    - Triggered on /health/memory endpoint
    - Calls cleanup_bulk_cache()
    - Removes expired entries
    - Based on BULK_CACHE_TIMEOUT
    - Reduces memory footprint

Benefits:
    - Proactive resource management
    - No manual intervention needed
    - Prevents gradual degradation
    - Maintains performance

Frequency:
    - Every health check call
    - Typically every 30-60 seconds (monitoring tools)
    - Low overhead operation

===================================================================================
MONITORING INTEGRATION
===================================================================================

Load Balancer Integration:
    - Configure health check: GET /health
    - Interval: 30 seconds
    - Timeout: 5 seconds
    - Unhealthy threshold: 2 consecutive failures
    - Healthy threshold: 2 consecutive successes

Prometheus Metrics:
    - Scrape /health/memory for memory_rss_mb
    - Scrape /health/database for read_latency_ms
    - Scrape /health/rate-limits for active_clients
    - Convert JSON to Prometheus format

Grafana Dashboards:
    - Memory usage graph (rss_mb over time)
    - Database latency graph
    - Cache size trend
    - Rate limit usage

Alerting Rules:
    - memory_usage_percent > 80 → Warning
    - memory_usage_percent > 100 → Critical
    - database read_test failed → Critical
    - read_latency_ms > 500 → Warning
    - prepared_statements_count > 200 → Warning

Kubernetes Probes:
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 30

    readinessProbe:
      httpGet:
        path: /health/database
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Response Times (Target):
    /health: 20-50ms
    /health/database: 30-100ms (includes DB query)
    /health/memory: 10-30ms
    /health/rate-limits: < 10ms
    /health/prepared-statements: 50-150ms (DB query)

Resource Usage:
    - CPU: Minimal (< 1% per check)
    - Memory: Negligible (few KB)
    - Network: Single DB roundtrip (database checks)
    - Disk: None

Scalability:
    - Can handle 100+ checks/second
    - No database writes
    - Minimal locking
    - Stateless operations

Optimization:
    - Cache health status (5-10 second TTL)
    - Batch multiple checks
    - Async operations
    - Connection pooling

===================================================================================
ERROR HANDLING
===================================================================================

Error Handling Pattern:
    try:
        # Health check logic
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return JSONResponse(500, {...})

Error Response Structure:
    {
        "status": "error",
        "error": "Error message",
        "timestamp": "2024-01-15T10:30:00"
    }

HTTP Status Codes:
    200: Success
    503: Service degraded but operational
    500: Health check failed (serious issue)

Graceful Degradation:
    - Return partial data if possible
    - Include error details in response
    - Don't crash on individual check failure
    - Log errors for investigation

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

Public Accessibility:
    - No authentication required
    - Load balancers need unrestricted access
    - Monitoring tools need access
    - Consider IP whitelist in production

Information Disclosure:
    - No sensitive data exposed
    - Version number is public
    - Memory usage is aggregate only
    - No user data included

Rate Limiting:
    - Health checks exempt from rate limits
    - Prevents monitoring from being blocked
    - Consider separate rate limit for health endpoints

DDoS Protection:
    - Health checks are lightweight
    - No expensive operations
    - Quick response times
    - Minimal attack surface

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    1. Memory calculation accuracy
    2. Status level determination
    3. Scaling recommendation logic
    4. Error handling paths

Integration Tests:
    1. Database connectivity test
    2. Full health check with real DB
    3. Memory monitoring accuracy
    4. Cache cleanup effectiveness

Mock Testing:
    - Mock psutil for memory checks
    - Mock checkpointer for DB checks
    - Mock gc.collect() for GC tests
    - Test error scenarios

Load Testing:
    - Verify health check performance under load
    - Test concurrent health check calls
    - Measure response time degradation
    - Verify no memory leaks

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection
    - gc: Garbage collection
    - time: Uptime and latency measurement
    - datetime: Timestamps
    - asyncio: Event loop (Windows)

Third-Party:
    - psutil: Memory and process monitoring
    - fastapi: Web framework, routing
    - dotenv: Environment loading
    - psycopg: PostgreSQL async driver

Internal:
    - api.config.settings: Global state, configuration
    - api.utils.memory: Cache cleanup utilities
    - api.helpers: Error response formatting
    - checkpointer.checkpointer.factory: Checkpointer access
    - checkpointer.config: Database configuration
    - checkpointer.database.connection: Connection utilities

===================================================================================
ENVIRONMENT CONFIGURATION
===================================================================================

Environment Variables:
    GC_MEMORY_THRESHOLD (int, default=1900)
        - Memory threshold in MB
        - Triggers high_memory status
        - Used for scaling calculations

    BULK_CACHE_TIMEOUT (int, default=60)
        - Cache entry TTL in seconds
        - Affects cleanup frequency

    RATE_LIMIT_WINDOW (int, default=60)
    RATE_LIMIT_REQUESTS (int, default=100)
    RATE_LIMIT_BURST (int, default=20)
        - Rate limiting configuration

===================================================================================
MAINTENANCE AND OPERATIONS
===================================================================================

Regular Monitoring Tasks:
    1. Check memory trends weekly
    2. Review database latency monthly
    3. Analyze cache hit rates
    4. Monitor prepared statement growth
    5. Review scaling recommendations

Incident Response:
    1. Check /health for overall status
    2. Check /health/database for DB issues
    3. Check /health/memory for OOM risk
    4. Check logs for error details
    5. Take corrective action

Capacity Planning:
    - Use scaling_info for thread capacity
    - Monitor memory trends
    - Plan for peak usage
    - Add headroom (30-50%)

Performance Tuning:
    - Adjust GC_MEMORY_THRESHOLD based on instance size
    - Tune BULK_CACHE_TIMEOUT for cache hit rate
    - Optimize MAX_CONCURRENT_BULK_THREADS
    - Monitor and adjust rate limits

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Enhanced Metrics
   - CPU usage tracking
   - Disk I/O monitoring
   - Network bandwidth usage
   - Thread pool status

2. Historical Tracking
   - Store health check history
   - Trend analysis
   - Anomaly detection
   - Predictive alerting

3. Advanced Diagnostics
   - Slow query detection
   - Connection pool details
   - Cache hit/miss rates
   - Request latency percentiles

4. Integration Features
   - Prometheus exporter endpoint
   - StatsD metrics emission
   - CloudWatch integration
   - Datadog integration

5. Self-Healing
   - Automatic cache clear on high memory
   - Connection pool reset on errors
   - Prepared statement cleanup triggers
   - Graceful degradation modes

===================================================================================
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

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
import gc
import time
from datetime import datetime

import psutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse

# Import globals and utilities from config/utils
from api.config.settings import (
    BULK_CACHE_TIMEOUT,
    GLOBAL_CHECKPOINTER,
    RATE_LIMIT_BURST,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    _bulk_loading_cache,
    rate_limit_storage,
    start_time,
)

# Import memory-related variables from memory.py
from api.utils.memory import cleanup_bulk_cache
from api.helpers import traceback_json_response

# Create router for health endpoints
router = APIRouter()


@router.get("/health")
async def health_check():
    """Enhanced health check with memory monitoring and database verification."""
    try:
        # Memory check
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # Database check with proper AsyncPostgresSaver handling
        database_healthy = True
        database_error = None
        checkpointer_type = "Unknown"

        try:
            if GLOBAL_CHECKPOINTER:
                checkpointer_type = type(GLOBAL_CHECKPOINTER).__name__

                if "AsyncPostgresSaver" in checkpointer_type:
                    # Test AsyncPostgresSaver with a simple operation
                    test_config = {"configurable": {"thread_id": "health_check_test"}}

                    # Use aget_tuple() which is a basic read operation
                    result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
                    # If we get here without exception, the database is healthy
                    database_healthy = True
                else:
                    # For other checkpointer types (like MemorySaver)
                    database_healthy = True

        except Exception as e:
            database_healthy = False
            database_error = str(e)

        # Response
        status = "healthy" if database_healthy else "degraded"

        health_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - start_time,
            "memory": {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(memory_percent, 2),
            },
            "database": {
                "healthy": database_healthy,
                "checkpointer_type": checkpointer_type,
                "error": database_error,
            },
            "version": "1.0.0",
        }

        # Run garbage collection
        collected = gc.collect()
        health_data["garbage_collector"] = {
            "objects_collected": collected,
            "gc_run": True,
        }

        if not database_healthy:
            return JSONResponse(status_code=503, content=health_data)

        return health_data

    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.get("/health/database")
async def database_health_check():
    """Detailed database health check."""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "checkpointer_available": GLOBAL_CHECKPOINTER is not None,
            "checkpointer_type": (
                type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None
            ),
        }

        if GLOBAL_CHECKPOINTER and "AsyncPostgresSaver" in str(
            type(GLOBAL_CHECKPOINTER)
        ):
            # Test AsyncPostgresSaver functionality
            try:
                test_config = {"configurable": {"thread_id": "db_health_test"}}

                # Test basic read operation
                start_time_local = time.time()
                _result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
                read_latency = time.time() - start_time_local

                health_status.update(
                    {
                        "database_connection": "healthy",
                        "read_latency_ms": round(read_latency * 1000, 2),
                        "read_test": "passed",
                    }
                )

            except Exception as e:
                health_status.update(
                    {
                        "database_connection": "error",
                        "error": str(e),
                        "read_test": "failed",
                    }
                )
                return JSONResponse(status_code=503, content=health_status)
        else:
            health_status.update(
                {
                    "database_connection": "using_memory_fallback",
                    "note": "PostgreSQL checkpointer not available",
                }
            )

        return health_status

    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return JSONResponse(
            status_code=500,
            content={
                "timestamp": datetime.now().isoformat(),
                "database_connection": "error",
                "error": str(e),
            },
        )


@router.get("/health/memory")
async def memory_health_check():
    """Enhanced memory-specific health check with cache information."""
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024

        # Clean up expired cache entries
        cleaned_entries = cleanup_bulk_cache()

        # Get memory threshold from environment
        gc_memory_threshold = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))

        status = "healthy"
        if rss_mb > gc_memory_threshold:
            status = "high_memory"
        elif rss_mb > (gc_memory_threshold * 0.8):
            status = "warning"

        cache_info = {
            "active_cache_entries": len(_bulk_loading_cache),
            "cleaned_expired_entries": cleaned_entries,
            "cache_timeout_seconds": BULK_CACHE_TIMEOUT,
        }

        # Calculate estimated memory per thread for scaling guidance
        thread_count = len(_bulk_loading_cache)
        memory_per_thread = rss_mb / max(thread_count, 1) if thread_count > 0 else 0
        estimated_max_threads = (
            int(gc_memory_threshold / max(memory_per_thread, 38))
            if memory_per_thread > 0
            else 50
        )

        return {
            "status": status,
            "memory_rss_mb": round(rss_mb, 1),
            "memory_threshold_mb": gc_memory_threshold,
            "memory_usage_percent": round((rss_mb / gc_memory_threshold) * 100, 1),
            "over_threshold": rss_mb > gc_memory_threshold,
            "cache_info": cache_info,
            "scaling_info": {
                "estimated_memory_per_thread_mb": round(memory_per_thread, 1),
                "estimated_max_threads_at_threshold": estimated_max_threads,
                "current_thread_count": thread_count,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/health/rate-limits")
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
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/health/prepared-statements")
async def prepared_statements_health_check():
    """Health check for prepared statements and database connection status."""
    try:
        from checkpointer.checkpointer.factory import get_global_checkpointer

        # Check if we can get a checkpointer
        try:
            checkpointer = await get_global_checkpointer()
            checkpointer_status = "healthy" if checkpointer else "unavailable"
        except Exception as e:
            checkpointer_status = f"error: {str(e)}"

        # Check prepared statements in the database
        try:
            import psycopg

            from checkpointer.config import get_db_config
            from checkpointer.database.connection import get_connection_kwargs

            config = get_db_config()
            # Create connection string without prepared statement parameters
            connection_string = (
                f"postgresql://{config['user']}:{config['password']}@"
                f"{config['host']}:{config['port']}/{config['dbname']}?sslmode=require"
            )

            # Get connection kwargs for disabling prepared statements
            connection_kwargs = get_connection_kwargs()

            async with await psycopg.AsyncConnection.connect(
                connection_string, **connection_kwargs
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT COUNT(*) as count, 
                               STRING_AGG(name, ', ') as statement_names
                        FROM pg_prepared_statements 
                        WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                    """
                    )
                    result = await cur.fetchone()

                    # Fix: Handle psycopg Row object properly - check if it exists and has data
                    prepared_count = result[0] if result else 0
                    statement_names = (
                        result[1]
                        if result and len(result) > 1 and result[1]
                        else "none"
                    )

                    return {
                        "status": "healthy",
                        "checkpointer_status": checkpointer_status,
                        "prepared_statements_count": prepared_count,
                        "prepared_statement_names": statement_names,
                        "connection_kwargs": connection_kwargs,
                        "timestamp": datetime.now().isoformat(),
                    }

        except Exception as db_error:
            return {
                "status": "degraded",
                "checkpointer_status": checkpointer_status,
                "database_error": str(db_error),
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
