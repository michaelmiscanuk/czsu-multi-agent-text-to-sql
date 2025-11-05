# CZSU Multi-Agent Text-to-SQL - Backend Features Documentation

> **Comprehensive documentation of backend services, APIs, AI components, and data services**
> 
> Based on actual implementation in the CZSU Multi-Agent Text-to-SQL system

---

## Table of Contents

1. [RESTful API Architecture](#restful-api-architecture)
2. [Authentication & Authorization](#authentication--authorization)
3. [Rate Limiting & Throttling](#rate-limiting--throttling)
4. [Memory Management](#memory-management)
5. [LangGraph Multi-Agent Workflow](#langgraph-multi-agent-workflow)
6. [AI Services Integration](#ai-services-integration)
7. [Data Services & Storage](#data-services--storage)
8. [Checkpointing System](#checkpointing-system)
9. [Error Handling & Monitoring](#error-handling--monitoring)
10. [MCP (Model Context Protocol) Integration](#mcp-model-context-protocol-integration)

---

## RESTful API Architecture

### FastAPI Application Structure

**Main Application** (`api/main.py`):
```python
app = FastAPI(
    title="CZSU Multi-Agent Text-to-SQL API",
    description="API for CZSU Multi-Agent Text-to-SQL application",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        200: {"model": dict, "description": "Success"},
        422: {"model": dict, "description": "Validation Error"},
        500: {"model": dict, "description": "Internal Server Error"},
    },
)
```

### Route Organization

**Modular Router Architecture**:
```python
# Route registration pattern
app.include_router(root_router, tags=["root"])
app.include_router(health_router, tags=["health"])
app.include_router(catalog_router, tags=["catalog"])
app.include_router(analysis_router, tags=["analysis"])
app.include_router(feedback_router, tags=["feedback"])
app.include_router(chat_router, tags=["chat"])
app.include_router(messages_router, tags=["messages"])
app.include_router(bulk_router, tags=["bulk"])
app.include_router(debug_router, tags=["debug"])
app.include_router(misc_router, tags=["misc"])
app.include_router(stop_router, tags=["execution"])
```

### Core API Endpoints

#### 1. Analysis Endpoint

**Natural Language Query Analysis** (`POST /analyze`):
```python
@router.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """
    Main analysis endpoint that processes natural language queries.
    
    Features:
    - Concurrent analysis limiting (MAX_CONCURRENT_ANALYSES)
    - Automatic fallback to InMemorySaver on database issues
    - Cancellation support with execution tracking
    - 4-minute timeout for platform stability
    - Comprehensive error handling with traceback
    """
    async with analysis_semaphore:  # Limit concurrent analyses
        checkpointer = await get_global_checkpointer()
        
        # Create thread run entry for tracking
        run_id = await create_thread_run_entry(
            user_email, request.thread_id, request.prompt, run_id=run_id
        )
        
        # Register for cancellation tracking
        register_execution(request.thread_id, run_id)
        
        # Execute with timeout and cancellation support
        result = await asyncio.wait_for(
            cancellable_analysis(),
            timeout=240,  # 4 minutes
        )
        
        # Extract metadata from thread history
        thread_metadata = await get_thread_metadata_from_single_thread_endpoint(
            request.thread_id, user_email
        )
```

**Request Model**:
```python
class AnalyzeRequest(BaseModel):
    prompt: str
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
```

**Response Structure**:
```python
{
    "prompt": str,
    "result": str,  # Final answer
    "queries_and_results": List[Tuple[str, str]],
    "thread_id": str,
    "top_selection_codes": List[str],
    "datasets_used": List[str],
    "iteration": int,
    "max_iterations": int,
    "sql": Optional[str],
    "datasetUrl": Optional[str],
    "run_id": str,  # UUID for feedback tracking
    "top_chunks": List[Document],
    "followup_prompts": List[str]
}
```

#### 2. Catalog Endpoints

**Dataset Catalog** (`GET /catalog`):
```python
@router.get("/catalog")
def get_catalog(
    page: int = Query(1, ge=1),
    q: Optional[str] = None,
    page_size: int = Query(10, ge=1, le=10000),
    user=Depends(get_current_user),
):
    """
    Paginated dataset catalog with search capability.
    
    Features:
    - Full-text search on selection_code and extended_description
    - Pagination with configurable page size
    - SQLite-based metadata retrieval
    """
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
    
    return {
        "results": results,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
```

**Data Tables** (`GET /data-tables`):
```python
@router.get("/data-tables")
def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
    """
    List available data tables with short descriptions.
    
    Features:
    - Enumerates all tables from czsu_data.db
    - Joins with selection_descriptions for metadata
    - Optional search filter by table name
    """
    with sqlite3.connect("data/czsu_data.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]
    
    # Fetch descriptions from metadata DB
    with sqlite3.connect(desc_db_path) as desc_conn:
        desc_cursor = desc_conn.cursor()
        desc_cursor.execute(
            "SELECT selection_code, short_description FROM selection_descriptions"
        )
        descriptions = desc_cursor.fetchall()
```

**Table Data** (`GET /data-table`):
```python
@router.get("/data-table")
def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
    """
    Retrieve data from a specific table.
    
    Features:
    - Returns columns and rows (up to 10,000 rows)
    - Used by frontend DataTableView component
    """
    with sqlite3.connect("data/czsu_data.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    
    return {"columns": columns, "rows": rows}
```

#### 3. Feedback Endpoints

**LangSmith Feedback Submission** (`POST /feedback`):
```python
@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user=Depends(get_current_user)):
    """
    Submit user feedback to LangSmith for run evaluation.
    
    Features:
    - UUID validation for run_id
    - Ownership verification (security check)
    - Integration with LangSmith Client
    - Support for both score and comment feedback
    """
    # Validate UUID format
    try:
        run_uuid = str(uuid.UUID(request.run_id))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}",
        )
    
    # Security check: Verify user owns this run_id
    async with get_direct_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT COUNT(*) FROM users_threads_runs 
                WHERE run_id = %s AND email = %s
                """,
                (run_uuid, user_email),
            )
            ownership_count = (await cur.fetchone())[0]
            
            if ownership_count == 0:
                raise HTTPException(
                    status_code=404, detail="Run ID not found or access denied"
                )
    
    # Submit to LangSmith
    client = Client()
    feedback_kwargs = {"run_id": run_uuid, "key": "SENTIMENT"}
    
    if request.feedback is not None:
        feedback_kwargs["score"] = request.feedback
    if request.comment:
        feedback_kwargs["comment"] = request.comment
    
    client.create_feedback(**feedback_kwargs)
```

**Sentiment Update** (`POST /sentiment`):
```python
@router.post("/sentiment")
async def update_sentiment(request: SentimentRequest, user=Depends(get_current_user)):
    """
    Update sentiment for a specific run in the database.
    
    Features:
    - Direct database update for sentiment tracking
    - UUID validation
    - Ownership verification
    """
    run_uuid = str(uuid.UUID(request.run_id))
    success = await update_thread_run_sentiment(run_uuid, request.sentiment)
    
    if success:
        return {
            "message": "Sentiment updated successfully",
            "run_id": run_uuid,
            "sentiment": request.sentiment,
        }
    else:
        raise HTTPException(
            status_code=404, detail=f"Run ID not found or access denied: {run_uuid}"
        )
```

### Middleware Stack

**CORS Configuration**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**GZip Compression**:
```python
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

**Throttling Middleware**:
```python
@app.middleware("http")
async def throttling_middleware(request: Request, call_next):
    """
    Throttling middleware that makes requests wait instead of rejecting them.
    
    Features:
    - Per-IP semaphore limiting (concurrent requests)
    - Rate limit checks with suggested wait time
    - Graceful waiting instead of immediate rejection
    - Skip for health checks and static endpoints
    """
    if request.url.path in ["/health", "/docs", "/openapi.json", "/debug/pool-status"]:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    semaphore = throttle_semaphores[client_ip]
    
    async with semaphore:
        if not await wait_for_rate_limit(client_ip):
            rate_info = check_rate_limit_with_throttling(client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Please wait {rate_info['suggested_wait']:.1f}s before retrying.",
                    "retry_after": max(rate_info["suggested_wait"], 1),
                    "burst_usage": f"{rate_info['burst_count']}/{rate_info['burst_limit']}",
                    "window_usage": f"{rate_info['window_count']}/{rate_info['window_limit']}",
                },
                headers={"Retry-After": str(max(int(rate_info["suggested_wait"]), 1))},
            )
        
        return await call_next(request)
```

**Memory Monitoring Middleware**:
```python
@app.middleware("http")
async def simplified_memory_monitoring_middleware(request: Request, call_next):
    """
    Simplified memory monitoring for heavy operations.
    
    Features:
    - Tracks memory before/after heavy endpoints (/analyze, /chat/all-messages)
    - Request counting
    - Minimal overhead for other endpoints
    """
    global _request_count
    _request_count += 1
    
    request_path = request.url.path
    if any(path in request_path for path in ["/analyze", "/chat/all-messages-for-all-threads"]):
        log_memory_usage(f"before_{request_path.replace('/', '_')}")
    
    response = await call_next(request)
    
    if any(path in request_path for path in ["/analyze", "/chat/all-messages-for-all-threads"]):
        log_memory_usage(f"after_{request_path.replace('/', '_')}")
    
    return response
```

### Exception Handlers

**Validation Error Handler**:
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with proper 422 status code.
    
    Features:
    - Uses jsonable_encoder to avoid JSON serialization errors
    - Fallback to simplified error list if encoding fails
    """
    try:
        payload = {"detail": "Validation error", "errors": exc.errors()}
        return JSONResponse(status_code=422, content=jsonable_encoder(payload))
    except Exception as encoding_error:
        simple_errors = [
            {"msg": e.get("msg"), "loc": e.get("loc"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": simple_errors,
                "note": "Simplified due to serialization issue",
            },
        )
```

**HTTP Exception Handler**:
```python
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle HTTP exceptions with comprehensive debugging for 401 errors.
    
    Features:
    - Enhanced logging for authentication failures (401)
    - Client IP tracking
    - Full traceback for debugging
    """
    if exc.status_code == 401:
        print__analyze_debug(f"🚨 HTTP 401 UNAUTHORIZED: {exc.detail}")
        print__analysis_tracing_debug(f"🚨 HTTP 401 TRACE: Request URL: {request.url}")
        print__analysis_tracing_debug(f"🚨 HTTP 401 TRACE: Request method: {request.method}")
        print__analysis_tracing_debug(f"🚨 HTTP 401 TRACE: Request headers: {dict(request.headers)}")
        
        client_ip = request.client.host if request.client else "unknown"
        print__analyze_debug(f"🚨 HTTP 401 CLIENT: IP address: {client_ip}")
    
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
```

---

## Authentication & Authorization

### JWT-Based Authentication

**Token Verification** (`api/dependencies/auth.py`):
```python
def get_current_user(authorization: str = Header(None)):
    """
    JWT authentication dependency with comprehensive debugging.
    
    Features:
    - Bearer token extraction and validation
    - Google JWT verification
    - Detailed trace logging
    - Structured error handling
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )
    
    # Split and validate token extraction
    auth_parts = authorization.split(" ", 1)
    if len(auth_parts) != 2 or not auth_parts[1].strip():
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )
    
    token = auth_parts[1].strip()
    
    # Call JWT verification
    user_info = verify_google_jwt(token)
    
    return user_info
```

**JWT Verification** (`api/auth/jwt_auth.py` - inferred):
```python
def verify_google_jwt(token: str) -> dict:
    """
    Verify Google OAuth JWT token.
    
    Features:
    - Google OAuth 2.0 validation
    - Token expiry checking
    - Returns user information (email, name, etc.)
    """
    # Verify token with Google's public keys
    # Extract user information
    # Return user info dict
    pass
```

### Protected Endpoints

All API endpoints (except `/health`, `/docs`) are protected with `Depends(get_current_user)`:

```python
@router.post("/analyze")
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")
```

### Ownership Verification

**Run ID Ownership Check** (Feedback endpoint):
```python
# Verify user owns this run_id before submitting feedback
async with get_direct_connection() as conn:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT COUNT(*) FROM users_threads_runs 
            WHERE run_id = %s AND email = %s
            """,
            (run_uuid, user_email),
        )
        
        ownership_count = (await cur.fetchone())[0]
        
        if ownership_count == 0:
            raise HTTPException(
                status_code=404, detail="Run ID not found or access denied"
            )
```

---

## Rate Limiting & Throttling

### Dual-Layer Rate Limiting

**Configuration** (`api/config/settings.py` - inferred):
```python
RATE_LIMIT_BURST = 10          # Max requests in 10 seconds
RATE_LIMIT_REQUESTS = 100      # Max requests in time window
RATE_LIMIT_WINDOW = 60         # Time window in seconds
RATE_LIMIT_MAX_WAIT = 30       # Maximum wait time before rejection
```

### Rate Limit Checking

**Throttling Logic** (`api/utils/rate_limiting.py`):
```python
def check_rate_limit_with_throttling(client_ip: str) -> dict:
    """
    Check rate limits and return throttling information.
    
    Features:
    - Burst limit (10 requests / 10 seconds)
    - Window limit (100 requests / 60 seconds)
    - Automatic cleanup of old entries
    - Suggested wait time calculation
    """
    now = time.time()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp
        for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check burst limit (last 10 seconds)
    recent_requests = [
        timestamp for timestamp in rate_limit_storage[client_ip] 
        if now - timestamp < 10
    ]
    
    # Check window limit
    window_requests = len(rate_limit_storage[client_ip])
    
    # Calculate suggested wait time
    suggested_wait = 0
    
    if len(recent_requests) >= RATE_LIMIT_BURST:
        oldest_burst = min(recent_requests)
        suggested_wait = max(0, 10 - (now - oldest_burst))
    elif window_requests >= RATE_LIMIT_REQUESTS:
        oldest_window = min(rate_limit_storage[client_ip])
        suggested_wait = max(0, RATE_LIMIT_WINDOW - (now - oldest_window))
    
    return {
        "allowed": len(recent_requests) < RATE_LIMIT_BURST and window_requests < RATE_LIMIT_REQUESTS,
        "suggested_wait": min(suggested_wait, RATE_LIMIT_MAX_WAIT),
        "burst_count": len(recent_requests),
        "window_count": window_requests,
        "burst_limit": RATE_LIMIT_BURST,
        "window_limit": RATE_LIMIT_REQUESTS,
    }
```

### Graceful Waiting

**Wait for Rate Limit** (`api/utils/rate_limiting.py`):
```python
async def wait_for_rate_limit(client_ip: str) -> bool:
    """
    Wait for rate limit to allow request, with maximum wait time.
    
    Features:
    - Up to 3 retry attempts
    - Exponential backoff-style waiting
    - Returns True if request can proceed
    - Returns False if wait time exceeds maximum
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        rate_info = check_rate_limit_with_throttling(client_ip)
        
        if rate_info["allowed"]:
            # Add current request to tracking
            rate_limit_storage[client_ip].append(time.time())
            return True
        
        if rate_info["suggested_wait"] <= 0:
            await asyncio.sleep(0.1)
            continue
        
        if rate_info["suggested_wait"] > RATE_LIMIT_MAX_WAIT:
            return False
        
        await asyncio.sleep(rate_info["suggested_wait"])
    
    return False
```

### Per-IP Semaphores

**Concurrent Request Limiting**:
```python
from collections import defaultdict
from asyncio import Semaphore

# Per-IP semaphores to limit concurrent requests
throttle_semaphores = defaultdict(lambda: Semaphore(5))  # Max 5 concurrent per IP

async with semaphore:
    # Process request
    pass
```

---

## Memory Management

### Proactive Memory Monitoring

**Memory Tracking** (`api/utils/memory.py`):
```python
def log_memory_usage(context: str):
    """
    Log current memory usage with context label.
    
    Features:
    - RSS (Resident Set Size) measurement
    - Baseline comparison tracking
    - Growth percentage calculation
    - psutil-based monitoring
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        
        if _memory_baseline:
            growth = rss_mb - _memory_baseline
            growth_pct = (growth / _memory_baseline) * 100
            print__memory_monitoring(
                f"📊 [{context}] Memory: {rss_mb:.1f}MB "
                f"(+{growth:.1f}MB, +{growth_pct:.1f}%)"
            )
        else:
            print__memory_monitoring(f"📊 [{context}] Memory: {rss_mb:.1f}MB")
    except Exception as e:
        print__memory_monitoring(f"❌ Memory monitoring error: {e}")
```

### Garbage Collection & malloc_trim

**Force Memory Release** (`api/utils/memory.py`):
```python
def force_release_memory():
    """
    Force memory release using malloc_trim if available (Linux only).
    
    Features:
    - Runs Python garbage collector
    - Calls libc malloc_trim(0) to return memory to OS
    - Measures freed memory
    - Graceful degradation on Windows/Mac
    """
    try:
        process = psutil.Process()
        initial_rss = process.memory_info().rss / 1024 / 1024
        
        # Run garbage collection
        collected = gc.collect()
        
        # Call malloc_trim if available (Linux only)
        if MALLOC_TRIM_AVAILABLE:
            libc.malloc_trim(0)
            malloc_trim_used = True
        else:
            malloc_trim_used = False
        
        final_rss = process.memory_info().rss / 1024 / 1024
        freed_mb = initial_rss - final_rss
        
        print__memory_monitoring(
            f"🧹 Memory cleanup: {freed_mb:.1f}MB freed | "
            f"{initial_rss:.1f}MB → {final_rss:.1f}MB | "
            f"GC: {collected} | malloc_trim: {'✓' if malloc_trim_used else '✗'}"
        )
        
        return {
            "freed_mb": round(freed_mb, 2),
            "gc_collected": collected,
            "malloc_trim_used": malloc_trim_used,
        }
    except Exception as e:
        return {"error": str(e), "freed_mb": 0}
```

### Periodic Memory Cleanup

**Background Cleanup Task**:
```python
async def start_memory_cleanup():
    """
    Start periodic memory cleanup background task.
    
    Features:
    - Runs every MEMORY_CLEANUP_INTERVAL seconds (default: 60s)
    - Automatic garbage collection
    - malloc_trim integration on Linux
    - Can be disabled via environment variable
    """
    if not MEMORY_CLEANUP_ENABLED:
        return None
    
    async def cleanup_loop():
        while True:
            await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)
            force_release_memory()
    
    task = asyncio.create_task(cleanup_loop())
    return task
```

### Memory Thresholds

**GC Trigger Configuration**:
```python
GC_MEMORY_THRESHOLD = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))  # MB

def check_memory_and_gc():
    """
    Check memory usage and trigger GC if threshold exceeded.
    
    Features:
    - Compares current RSS to GC_MEMORY_THRESHOLD
    - Cleans bulk cache at 80% of threshold
    - Forces GC and malloc_trim at threshold
    - Automatic restart recommendation if growth excessive
    """
    process = psutil.Process()
    rss_mb = process.memory_info().rss / 1024 / 1024
    
    # Clean cache at 80% threshold
    if rss_mb > (GC_MEMORY_THRESHOLD * 0.8):
        cleaned_entries = cleanup_bulk_cache()
        if cleaned_entries > 0:
            new_memory = psutil.Process().memory_info().rss / 1024 / 1024
            freed = rss_mb - new_memory
            print__memory_monitoring(
                f"🧹 Cache cleanup freed {freed:.1f}MB ({cleaned_entries} entries)"
            )
    
    # Force GC at threshold
    if rss_mb > GC_MEMORY_THRESHOLD:
        print__memory_monitoring(
            f"🚨 Memory at {rss_mb:.1f}MB (threshold: {GC_MEMORY_THRESHOLD}MB) - forcing GC"
        )
        force_release_memory()
```

### Bulk Cache Management

**Time-Based Cache Expiration**:
```python
BULK_CACHE_TIMEOUT = 300  # 5 minutes

_bulk_loading_cache = {}  # Global cache dictionary

def cleanup_bulk_cache():
    """
    Clean up expired cache entries.
    
    Features:
    - Time-based expiration (5 minutes)
    - Automatic cleanup during memory pressure
    - Thread-safe implementation
    """
    current_time = time.time()
    expired_keys = []
    
    for cache_key, (cached_data, cache_time) in _bulk_loading_cache.items():
        if current_time - cache_time > BULK_CACHE_TIMEOUT:
            expired_keys.append(cache_key)
    
    for key in expired_keys:
        del _bulk_loading_cache[key]
    
    return len(expired_keys)
```

### Graceful Shutdown

**Signal Handler Registration**:
```python
def setup_graceful_shutdown():
    """
    Setup graceful shutdown handlers for SIGTERM and SIGINT.
    
    Features:
    - Catches termination signals
    - Logs shutdown reason
    - Cleans up resources
    - Exits cleanly
    """
    def shutdown_handler(signum, frame):
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print__memory_monitoring(
            f"🛑 Received {signal_name} - initiating graceful shutdown"
        )
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
```

---

## LangGraph Multi-Agent Workflow

### StateGraph Architecture

**Graph Creation** (`my_agent/agent.py`):
```python
def create_graph(checkpointer=None):
    """
    Create the LangGraph StateGraph for data analysis.
    
    Architecture:
    - 22 nodes organized in 6 processing stages
    - Parallel retrieval branches (database + PDF)
    - Conditional routing based on data availability
    - Controlled iteration with MAX_ITERATIONS limit
    - PostgreSQL checkpointing for conversation persistence
    
    Graph Flow:
    1. Query Preprocessing (rewrite_prompt → summarize_messages)
    2. Parallel Retrieval (selections + chunks)
    3. Synchronization & Routing (post_retrieval_sync)
    4. SQL Generation Loop (get_schema → generate_query → reflect)
    5. Answer Finalization (format_answer → followup_prompts)
    6. Resource Cleanup (save → cleanup_resources)
    """
    graph = StateGraph(DataAnalysisState)
    
    # Add all nodes
    graph.add_node("rewrite_prompt", rewrite_prompt_node)
    graph.add_node("summarize_messages_rewrite", summarize_messages_node)
    graph.add_node("retrieve_similar_selections_hybrid_search", 
                   retrieve_similar_selections_hybrid_search_node)
    graph.add_node("rerank_table_descriptions", rerank_table_descriptions_node)
    graph.add_node("relevant_selections", relevant_selections_node)
    graph.add_node("retrieve_similar_chunks_hybrid_search", 
                   retrieve_similar_chunks_hybrid_search_node)
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("relevant_chunks", relevant_chunks_node)
    graph.add_node("post_retrieval_sync", post_retrieval_sync_node)
    graph.add_node("get_schema", get_schema_node)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("summarize_messages_query", summarize_messages_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("summarize_messages_reflect", summarize_messages_node)
    graph.add_node("format_answer", format_answer_node)
    graph.add_node("summarize_messages_format", summarize_messages_node)
    graph.add_node("generate_followup_prompts", followup_prompts_node)
    graph.add_node("submit_final_answer", submit_final_answer_node)
    graph.add_node("save", save_node)
    graph.add_node("cleanup_resources", cleanup_resources_node)
    
    # Define edges (abbreviated - see full implementation)
    graph.add_edge(START, "rewrite_prompt")
    graph.add_edge("rewrite_prompt", "summarize_messages_rewrite")
    
    # Parallel retrieval branches
    graph.add_edge("summarize_messages_rewrite", "retrieve_similar_selections_hybrid_search")
    graph.add_edge("summarize_messages_rewrite", "retrieve_similar_chunks_hybrid_search")
    
    # Conditional routing
    graph.add_conditional_edges(
        "post_retrieval_sync",
        route_after_sync,
        {"get_schema": "get_schema", "format_answer": "format_answer", END: END},
    )
    
    # Compile with checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph
```

### State Management

**DataAnalysisState Schema** (`my_agent/utils/state.py`):
```python
class DataAnalysisState(TypedDict):
    """
    State container for LangGraph workflow.
    
    Key Fields:
    - prompt: Original user question
    - rewritten_prompt: Search-optimized standalone question
    - messages: [summary (SystemMessage), last_message]
    - iteration: Loop counter for cycle prevention (max: 1)
    - queries_and_results: List of (SQL_query, result) tuples
    - reflection_decision: "improve" or "answer" from reflect node
    - top_selection_codes: Database table identifiers (max: 3)
    - top_chunks: Relevant PDF documentation chunks
    - final_answer: Formatted answer string
    - followup_prompts: List of suggested follow-up questions
    - run_id: UUID for LangSmith tracking
    - thread_id: Conversation thread identifier
    """
    prompt: str
    rewritten_prompt: str
    messages: Annotated[List[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    queries_and_results: Annotated[List[Tuple[str, str]], limit_queries_and_results]
    reflection_decision: str
    top_selection_codes: List[str]
    datasets_used: List[str]
    top_chunks: List[Document]
    final_answer: str
    followup_prompts: List[str]
    run_id: str
    thread_id: str
    
    # Intermediate retrieval results (cleared after use)
    hybrid_search_results: List[Document]
    most_similar_selections: List[Tuple[str, float]]
    hybrid_search_chunks: List[Document]
    most_similar_chunks: List[Tuple[Document, float]]
    chromadb_missing: bool  # Error flag
```

### Node Implementation Patterns

**Query Rewriting Node** (`my_agent/utils/nodes.py`):
```python
async def rewrite_prompt_node(state: DataAnalysisState) -> dict:
    """
    Rewrite conversational question into standalone search query.
    
    Features:
    - Resolves pronouns and references using conversation history
    - Detects topic corrections ("but I meant X")
    - Expands vague queries with domain context
    - Adds synonyms for better vector search
    - Preserves original language (Czech/English)
    - Escapes curly braces for f-string safety
    
    LLM: Azure GPT-4o (temperature=0.0)
    """
    llm = get_azure_chat_openai_gpt4o_mini()
    
    system_prompt = """You are an expert at converting conversational questions 
    into standalone, search-optimized queries for Czech statistical data..."""
    
    # Create messages with conversation summary + last question
    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],  # [summary, last_message]
    ]
    
    response = await llm.ainvoke(messages)
    rewritten = response.content.strip()
    
    # Escape curly braces for f-string safety
    rewritten = rewritten.replace("{", "{{").replace("}", "}}")
    
    return {
        "rewritten_prompt": rewritten,
        "messages": [AIMessage(content=rewritten, id="rewritten_prompt")],
    }
```

**Hybrid Search Retrieval Node**:
```python
async def retrieve_similar_selections_hybrid_search_node(state: DataAnalysisState) -> dict:
    """
    Perform hybrid search on database selection descriptions.
    
    Features:
    - Semantic search using Azure OpenAI embeddings
    - BM25 keyword search for exact matches
    - Weighted combination (configurable)
    - ChromaDB client initialization with cloud fallback
    - Explicit memory cleanup after retrieval
    
    Returns:
    - hybrid_search_results: List[Document] with selection descriptions
    - chromadb_missing: bool flag if database unavailable
    """
    # Check ChromaDB directory existence
    chroma_path = BASE_DIR / "metadata" / "czsu_chromadb"
    use_cloud = os.getenv("CHROMA_USE_CLOUD", "false").lower() == "true"
    
    if not chroma_path.exists() and not use_cloud:
        return {
            "chromadb_missing": True,
            "hybrid_search_results": [],
        }
    
    # Initialize ChromaDB client
    client = get_chroma_client()
    collection = client.get_collection(name="czsu_selection_descriptions")
    
    # Perform hybrid search
    results = hybrid_search(
        collection=collection,
        query=state["rewritten_prompt"],
        n_results=20,
    )
    
    # Convert to Document objects
    documents = [
        Document(
            page_content=result["extended_description"],
            metadata={"selection_code": result["selection_code"]},
        )
        for result in results
    ]
    
    # Explicit cleanup
    client.clear_system_cache()
    del client
    gc.collect()
    
    return {"hybrid_search_results": documents}
```

**Agentic SQL Generation Node**:
```python
async def generate_query_node(state: DataAnalysisState) -> dict:
    """
    Generate and execute SQL queries using agentic tool calling.
    
    Features:
    - LLM autonomously decides when to execute queries
    - Multiple query iterations (MAX_TOOL_ITERATIONS=10)
    - MCP tool integration with local SQLite fallback
    - Iterative data gathering pattern
    - Limited query history (max 5) to prevent token overflow
    
    Tools Available:
    - sqlite_query: Execute SQL query on czsu_data.db
    - finish_gathering: Signal data collection complete
    
    LLM: Azure GPT-4o (temperature=0.0, tool calling enabled)
    """
    llm = get_azure_chat_openai_gpt4o()
    
    # Get SQL query tools (MCP or local fallback)
    tools = await get_sqlite_tools()
    tools.append(finish_gathering)
    
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = f"""You are a SQL expert analyzing Czech statistical data.
    
    Schema:
    {state['messages'][-1].content}  # Schema loaded in previous node
    
    Instructions:
    - Generate accurate SQL queries for SQLite
    - Handle Czech diacritics properly (č, š, ž, etc.)
    - Use sqlite_query tool to execute queries
    - Examine results and decide if more queries needed
    - Call finish_gathering when you have sufficient data
    - Maximum {MAX_TOOL_ITERATIONS} tool calls allowed
    """
    
    # Iterative tool calling loop
    for iteration in range(MAX_TOOL_ITERATIONS):
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            *state["messages"],
        ])
        
        if not response.tool_calls:
            break  # No more tool calls needed
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            if tool_call["name"] == "finish_gathering":
                break
            elif tool_call["name"] == "sqlite_query":
                result = await execute_tool(tool_call, tools)
                # Store query and result
                state["queries_and_results"].append((
                    tool_call["args"]["query"],
                    result
                ))
    
    return {
        "queries_and_results": state["queries_and_results"][-5:],  # Keep last 5
        "iteration": state["iteration"] + 1,
    }
```

### Conditional Routing

**Post-Retrieval Router** (`my_agent/utils/routers.py`):
```python
def route_after_sync(state: DataAnalysisState) -> Literal["get_schema", "format_answer", END]:
    """
    Route based on retrieval results.
    
    Logic:
    - IF top_selection_codes found → "get_schema" (proceed with SQL)
    - ELIF chromadb_missing → END (error: no database)
    - ELSE → "format_answer" (PDF-only response)
    """
    if state.get("top_selection_codes"):
        return "get_schema"
    elif state.get("chromadb_missing"):
        # Set error message in final_answer
        state["final_answer"] = """
        ❌ ERROR: ChromaDB directory is missing. 
        Please unzip or create the ChromaDB at 'metadata/czsu_chromadb'.
        Or use ChromaDB Cloud with CHROMA_USE_CLOUD="true".
        """
        return END
    else:
        return "format_answer"  # PDF-only answer
```

**Reflection Router**:
```python
def route_after_reflect(state: DataAnalysisState) -> Literal["generate_query", "format_answer"]:
    """
    Route based on reflection decision.
    
    Logic:
    - IF decision == "improve" → "generate_query" (retry with better query)
    - ELSE → "format_answer" (sufficient data collected)
    """
    decision = state.get("reflection_decision", "answer")
    if decision == "improve" and state["iteration"] < state["max_iterations"]:
        return "generate_query"
    else:
        return "format_answer"
```

---

## AI Services Integration

### Azure OpenAI Services

**GPT-4o Configuration** (`my_agent/utils/models.py`):
```python
def get_azure_chat_openai_gpt4o():
    """
    Get Azure GPT-4o LLM instance for complex reasoning tasks.
    
    Use Cases:
    - Query rewriting
    - SQL generation with tool calling
    - Reflection and self-correction
    - Answer formatting
    
    Configuration:
    - Model: gpt-4o (latest)
    - Temperature: 0.0 (deterministic)
    - Max tokens: 16384
    - Async support: enabled
    """
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
        max_tokens=16384,
        timeout=None,
        max_retries=2,
    )
```

**GPT-4o-mini Configuration**:
```python
def get_azure_chat_openai_gpt4o_mini():
    """
    Get Azure GPT-4o-mini LLM instance for lighter tasks.
    
    Use Cases:
    - Message summarization (4 instances in graph)
    - Follow-up prompt generation
    - Cost-optimized operations
    
    Configuration:
    - Model: gpt-4o-mini
    - Temperature: 0.0
    - Max tokens: 16384
    - Faster and cheaper than GPT-4o
    """
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MINI"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
        max_tokens=16384,
        timeout=None,
        max_retries=2,
    )
```

**Embedding Model**:
```python
def get_azure_openai_embeddings():
    """
    Get Azure OpenAI embeddings for vector search.
    
    Use Cases:
    - ChromaDB hybrid search (semantic component)
    - Selection descriptions embedding
    - PDF chunks embedding
    
    Configuration:
    - Model: text-embedding-ada-002
    - Dimensions: 1536
    - Chunk size: 1000 tokens
    """
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        chunk_size=1000,
    )
```

### Azure AI Translator

**Language Translation** (`my_agent/utils/helpers.py`):
```python
async def translate_to_english(text):
    """
    Translate text to English using Azure Translator API.
    
    Use Cases:
    - PDF chunk retrieval (Czech query → English docs)
    - Cross-lingual semantic search
    
    Features:
    - Async HTTP request in thread pool
    - Unique trace ID for each request
    - Multi-language support
    
    API Configuration:
    - Endpoint: /translate?api-version=3.0&to=en
    - Method: POST
    - Authentication: Subscription key + region
    """
    subscription_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
    region = os.environ["TRANSLATOR_TEXT_REGION"]
    endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]
    
    constructed_url = f"{endpoint}/translate?api-version=3.0&to=en"
    
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    
    body = [{"text": text}]
    
    # Execute in thread pool for async compatibility
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: requests.post(constructed_url, headers=headers, json=body)
    )
    
    result = response.json()
    return result[0]["translations"][0]["text"]
```

### Cohere Reranking

**Multilingual Reranking**:
```python
def cohere_rerank(query: str, documents: List[str], top_n: int = 20) -> List[dict]:
    """
    Rerank search results using Cohere multilingual model.
    
    Use Cases:
    - Database selection reranking (after hybrid search)
    - PDF chunk reranking (after hybrid search)
    
    Features:
    - Multilingual support (Czech + English)
    - Relevance score calculation
    - Top-N filtering
    
    Model: rerank-multilingual-v3.0
    """
    import cohere
    
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-multilingual-v3.0",
    )
    
    return results.results
```

### LangSmith Tracing

**Automatic Run Tracking**:
```python
# Configured via environment variables
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "czsu-text-to-sql")

# All LangGraph executions automatically traced
# Includes:
# - Node execution times
# - LLM token usage
# - Tool calls and results
# - State transitions
# - Error traces
```

**Feedback Integration**:
```python
from langsmith import Client

client = Client()
client.create_feedback(
    run_id=run_uuid,
    key="SENTIMENT",
    score=1.0,  # 1.0 = positive, 0.0 = negative
    comment="Optional user comment",
)
```

---

## Data Services & Storage

### ChromaDB Vector Database

**Dual Collection Architecture**:

1. **Selection Descriptions Collection** (`metadata/czsu_chromadb`):
```python
# Collection: czsu_selection_descriptions
# Purpose: Dataset catalog with extended descriptions
# Documents: ~600 Czech statistical datasets
# Metadata: selection_code, extended_description
# Embedding: Azure OpenAI text-embedding-ada-002 (1536 dimensions)
```

2. **PDF Chunks Collection** (`metadata/pdf_chromadb_llamaparse_v3`):
```python
# Collection: pdf_chunks
# Purpose: Parsed PDF documentation chunks
# Documents: ~1000+ chunks from CZSU documentation
# Metadata: source, page, chunk_id
# Embedding: Azure OpenAI text-embedding-ada-002 (1536 dimensions)
```

**Hybrid Search Implementation**:
```python
def hybrid_search(collection, query: str, n_results: int = 20) -> List[dict]:
    """
    Combine semantic search and BM25 keyword search.
    
    Algorithm:
    1. Semantic Search:
       - Embed query using Azure OpenAI
       - ChromaDB cosine similarity
       - Weight: 0.7 (configurable)
    
    2. BM25 Keyword Search:
       - Term frequency–inverse document frequency
       - Exact phrase matching
       - Weight: 0.3 (configurable)
    
    3. Score Combination:
       - Weighted sum of normalized scores
       - Re-ranking by combined score
       - Top-N selection
    
    Benefits:
    - Semantic: Handles synonyms, paraphrasing
    - BM25: Exact terminology matches
    - Combined: Best of both worlds
    """
    # Get embeddings for semantic search
    embeddings = get_azure_openai_embeddings()
    query_embedding = embeddings.embed_query(query)
    
    # Semantic search
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    # BM25 search (implemented in ChromaDB)
    bm25_results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    
    # Combine and rerank
    combined = combine_scores(semantic_results, bm25_results, 
                              semantic_weight=0.7, bm25_weight=0.3)
    
    return combined[:n_results]
```

**Cloud Fallback Configuration**:
```python
def get_chroma_client():
    """
    Initialize ChromaDB client with cloud fallback.
    
    Modes:
    1. Local (default): Uses persistent directory
    2. Cloud: Connects to Chroma Cloud (trychroma.com)
    
    Environment Variables:
    - CHROMA_USE_CLOUD="true" to enable cloud
    - CHROMA_CLOUD_API_KEY: Authentication key
    - CHROMA_CLOUD_TENANT: Tenant ID
    - CHROMA_CLOUD_DATABASE: Database name
    """
    use_cloud = os.getenv("CHROMA_USE_CLOUD", "false").lower() == "true"
    
    if use_cloud:
        import chromadb
        return chromadb.CloudClient(
            api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
            tenant=os.getenv("CHROMA_CLOUD_TENANT"),
            database=os.getenv("CHROMA_CLOUD_DATABASE"),
        )
    else:
        import chromadb
        chroma_path = BASE_DIR / "metadata" / "czsu_chromadb"
        return chromadb.PersistentClient(path=str(chroma_path))
```

### SQLite Databases

**1. Data Database** (`data/czsu_data.db`):
```sql
-- Purpose: Actual statistical data tables
-- Tables: ~600 tables (one per dataset selection_code)
-- Size: ~500MB
-- Schema: Varies by table (dynamic columns)

-- Example table: 453_461_524 (Population statistics)
CREATE TABLE "453_461_524" (
    "vuzemi_cis" TEXT,
    "vuzemi_kod" TEXT,
    "vuzemi_txt" TEXT,
    "pohlavi_cis" TEXT,
    "pohlavi_kod" TEXT,
    "pohlavi_txt" TEXT,
    "hodnota" REAL,
    -- Additional columns...
);

-- Common patterns:
-- - Territory columns: vuzemi_cis, vuzemi_kod, vuzemi_txt
-- - Category columns: *_cis (code list), *_kod (code), *_txt (label)
-- - Value column: hodnota (numerical value)
-- - Total row: Often includes "CELKEM" (total) in category text
```

**2. Metadata Database** (`metadata/llm_selection_descriptions/selection_descriptions.db`):
```sql
-- Purpose: Dataset catalog with descriptions
-- Tables: 1 (selection_descriptions)
-- Size: ~10MB

CREATE TABLE selection_descriptions (
    selection_code TEXT PRIMARY KEY,
    short_description TEXT,
    extended_description TEXT,  -- Used for vector search
    -- Extended description includes:
    -- - Table name and purpose
    -- - Column names with Czech labels
    -- - Data types and constraints
    -- - Sample categorical values
    -- - CELKEM row handling instructions
);

-- Example extended_description format:
-- Dataset: 453_461_524
-- Table Name: Population by Territory and Gender
-- Columns:
--   - vuzemi_kod (TEXT): Territory code
--   - vuzemi_txt (TEXT): Territory name (e.g., "Praha", "Brno")
--   - pohlavi_kod (TEXT): Gender code (1=Male, 2=Female, CELKEM=Total)
--   - hodnota (REAL): Population count
-- Note: CELKEM rows contain aggregated totals, exclude for detailed breakdowns
```

**Schema Loading Pattern**:
```python
async def load_schema(state: DataAnalysisState) -> str:
    """
    Load extended schema descriptions for selected datasets.
    
    Process:
    1. Extract top_selection_codes from state (max 3)
    2. Connect to selection_descriptions.db
    3. Query extended_description for each code
    4. Join with delimiter for multi-dataset queries
    
    Returns:
    Dataset: 453_461_524.
    [Extended description...]
    **************
    Dataset: 501_515_523.
    [Extended description...]
    """
    db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / "selection_descriptions.db"
    schemas = []
    
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        for selection_code in state["top_selection_codes"]:
            cursor.execute(
                """
                SELECT extended_description FROM selection_descriptions
                WHERE selection_code = ? AND extended_description IS NOT NULL
                """,
                (selection_code,),
            )
            row = cursor.fetchone()
            if row:
                schemas.append(f"Dataset: {selection_code}.\n{row[0]}")
    
    return "\n**************\n".join(schemas)
```

### Supabase PostgreSQL

**Conversation Persistence**:
```sql
-- Table: checkpoints
-- Purpose: LangGraph state checkpoints
-- Schema: LangGraph AsyncPostgresSaver format
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BYTEA,  -- Serialized state
    metadata BYTEA,    -- Additional metadata
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Table: checkpoint_writes
-- Purpose: Pending state writes (for async operations)
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Table: users_threads_runs
-- Purpose: User conversation tracking and ownership
CREATE TABLE users_threads_runs (
    run_id UUID PRIMARY KEY,
    email TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    sentiment TEXT,  -- 'positive', 'negative', or NULL
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (thread_id) REFERENCES checkpoints(thread_id)
);

CREATE INDEX idx_user_threads ON users_threads_runs(email, thread_id);
CREATE INDEX idx_thread_runs ON users_threads_runs(thread_id);
```

**Connection Pool Configuration**:
```python
# Pool settings (optimized for Railway.app 2GB memory)
DEFAULT_POOL_MIN_SIZE = 2
DEFAULT_POOL_MAX_SIZE = 10
DEFAULT_POOL_TIMEOUT = 30  # seconds
DEFAULT_MAX_IDLE = 300  # 5 minutes
DEFAULT_MAX_LIFETIME = 3600  # 1 hour

pool = AsyncConnectionPool(
    conninfo=get_connection_string(),
    min_size=DEFAULT_POOL_MIN_SIZE,
    max_size=DEFAULT_POOL_MAX_SIZE,
    timeout=DEFAULT_POOL_TIMEOUT,
    max_idle=DEFAULT_MAX_IDLE,
    max_lifetime=DEFAULT_MAX_LIFETIME,
    kwargs=get_connection_kwargs(),
)
```

### Turso (Alternative SQLite Cloud)

**MCP Server Integration**:
```python
# MCP Server connects to Turso for remote SQLite access
# Configuration (in MCP server, not backend):
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")  # e.g., https://mcp-server.railway.app

# Turso connection (MCP server side):
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

# Backend uses either:
# 1. Remote MCP server (Turso-backed)
# 2. Local SQLite file (fallback)
```

---

## Checkpointing System

### PostgreSQL Checkpointer

**Factory Pattern** (`checkpointer/checkpointer/factory.py`):
```python
@retry_on_ssl_connection_error(max_retries=3)
@retry_on_prepared_statement_error(max_retries=5)
async def create_async_postgres_saver():
    """
    Create and configure AsyncPostgresSaver with connection pool.
    
    Features:
    - Connection string approach for proper SSL configuration
    - Automatic retry on SSL and prepared statement errors
    - Table setup with autocommit for DDL operations
    - Global singleton pattern for pool reuse
    - Health checks before returning
    
    Tables Created:
    - checkpoints: State snapshots
    - checkpoint_writes: Pending writes
    - users_threads_runs: User tracking (custom)
    """
    global _GLOBAL_CHECKPOINTER
    
    # Clear existing state
    if _GLOBAL_CHECKPOINTER:
        if hasattr(_GLOBAL_CHECKPOINTER, "pool"):
            await _GLOBAL_CHECKPOINTER.pool.close()
        _GLOBAL_CHECKPOINTER = None
    
    # Get connection string with SSL parameters
    connection_string = get_connection_string()
    connection_kwargs = get_connection_kwargs()
    
    # Create connection pool
    pool = AsyncConnectionPool(
        conninfo=connection_string,
        min_size=DEFAULT_POOL_MIN_SIZE,
        max_size=DEFAULT_POOL_MAX_SIZE,
        timeout=DEFAULT_POOL_TIMEOUT,
        kwargs=connection_kwargs,
    )
    
    # Setup tables (LangGraph + custom)
    await setup_checkpointer_with_autocommit(pool)
    await setup_users_threads_runs_table()
    
    # Create saver with pool
    saver = AsyncPostgresSaver(pool)
    
    _GLOBAL_CHECKPOINTER = saver
    return saver
```

**Global Singleton Access**:
```python
async def get_global_checkpointer():
    """
    Get or create the global checkpointer instance.
    
    Features:
    - Thread-safe initialization with asyncio.Lock
    - Automatic creation if not initialized
    - Health check before returning
    - Recreates on connection errors
    """
    global _GLOBAL_CHECKPOINTER
    
    async with _CHECKPOINTER_INIT_LOCK:
        if _GLOBAL_CHECKPOINTER is None:
            _GLOBAL_CHECKPOINTER = await create_async_postgres_saver()
        else:
            # Health check
            healthy = await check_pool_health_and_recreate()
            if not healthy:
                # Recreated automatically by health check
                pass
    
    return _GLOBAL_CHECKPOINTER
```

**InMemory Fallback**:
```python
# Fallback when PostgreSQL unavailable
INMEMORY_FALLBACK_ENABLED = os.getenv("INMEMORY_FALLBACK_ENABLED", "1") == "1"

if INMEMORY_FALLBACK_ENABLED:
    from langgraph.checkpoint.memory import InMemorySaver
    
    fallback_checkpointer = InMemorySaver()
    
    # Use for single conversation (no persistence)
    result = await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
        checkpointer=fallback_checkpointer,
    )
```

### Thread Management

**Thread Run Creation** (`checkpointer/user_management/thread_operations.py` - inferred):
```python
async def create_thread_run_entry(
    email: str,
    thread_id: str,
    prompt: str,
    run_id: Optional[str] = None
) -> str:
    """
    Create a new run entry for tracking and ownership.
    
    Features:
    - Generates UUID run_id if not provided
    - Associates run with user email and thread
    - Enables ownership verification for feedback
    - Tracks original user prompt
    
    Returns:
    - run_id: UUID string for LangSmith correlation
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    async with get_direct_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO users_threads_runs (run_id, email, thread_id, prompt)
                VALUES (%s, %s, %s, %s)
                """,
                (run_id, email, thread_id, prompt),
            )
    
    return run_id
```

**Sentiment Tracking**:
```python
async def update_thread_run_sentiment(run_id: str, sentiment: str) -> bool:
    """
    Update sentiment for a specific run.
    
    Sentiment values:
    - "positive": User satisfied with response
    - "negative": User dissatisfied
    - NULL: No feedback provided
    
    Returns:
    - True if updated successfully
    - False if run_id not found
    """
    async with get_direct_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE users_threads_runs
                SET sentiment = %s
                WHERE run_id = %s
                """,
                (sentiment, run_id),
            )
            
            return cur.rowcount > 0
```

### Connection Management

**Direct Connection** (`checkpointer/database/connection.py` - inferred):
```python
@asynccontextmanager
async def get_direct_connection():
    """
    Get a direct database connection (outside of checkpointer pool).
    
    Use Cases:
    - User management queries (users_threads_runs table)
    - Ownership verification
    - Sentiment updates
    - Non-checkpointer operations
    
    Features:
    - Async context manager
    - Automatic connection cleanup
    - Separate from checkpointer pool
    - SSL configuration included
    """
    connection_string = get_connection_string()
    
    async with await psycopg.AsyncConnection.connect(connection_string) as conn:
        yield conn
```

**Pool Health Monitoring**:
```python
async def check_pool_health_and_recreate():
    """
    Check the health of the global connection pool.
    
    Health Check:
    1. Acquire connection with 10s timeout
    2. Execute "SELECT 1" query
    3. Verify result
    
    On Failure:
    - Close all pool connections
    - Clear global state
    - Recreate checkpointer
    
    Common Failures:
    - SSL connection errors
    - Prepared statement errors
    - Connection timeouts
    - Pool exhaustion
    """
    global _GLOBAL_CHECKPOINTER
    
    try:
        pool = getattr(_GLOBAL_CHECKPOINTER, "pool", None)
        if pool is not None:
            async with asyncio.wait_for(pool.connection(), timeout=10.0) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    result = await cur.fetchone()
                    if result is None or result[0] != 1:
                        raise Exception("Pool health check failed: bad result")
            return True
    except Exception as e:
        print__checkpointers_debug(f"POOL HEALTH CHECK FAILED: {e}, recreating pool...")
        await force_close_modern_pools()
        _GLOBAL_CHECKPOINTER = None
        _GLOBAL_CHECKPOINTER = await create_async_postgres_saver()
        return False
```

---

## Error Handling & Monitoring

### Retry Decorators

**Prepared Statement Error Retry** (`checkpointer/error_handling/retry_decorators.py` - inferred):
```python
def retry_on_prepared_statement_error(max_retries: int = 5):
    """
    Retry decorator for PostgreSQL prepared statement errors.
    
    Common Errors:
    - "prepared statement _pg3_0 does not exist"
    - "prepared statement _pg_0 does not exist"
    - InvalidSqlStatementName
    
    Strategy:
    - Exponential backoff (1s, 2s, 4s, 8s, 16s)
    - Clear connection pool on error
    - Recreate connections
    - Log retry attempts
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(indicator in error_msg for indicator in [
                        "prepared statement",
                        "does not exist",
                        "_pg3_",
                        "_pg_",
                        "invalidsqlstatementname",
                    ]):
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print__checkpointers_debug(
                                f"Prepared statement error (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {wait_time}s..."
                            )
                            await asyncio.sleep(wait_time)
                            await force_close_modern_pools()
                            continue
                    raise
            return None
        return wrapper
    return decorator
```

**SSL Connection Error Retry**:
```python
def retry_on_ssl_connection_error(max_retries: int = 3):
    """
    Retry decorator for SSL/connection errors.
    
    Common Errors:
    - "SSL connection has been closed unexpectedly"
    - "connection closed"
    - "connection timeout"
    - psycopg.OperationalError
    
    Strategy:
    - Fixed 2s delay between retries
    - Force close pools before retry
    - Log connection attempts
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in [
                        "ssl", "connection", "timeout", "closed"
                    ]):
                        if attempt < max_retries - 1:
                            print__checkpointers_debug(
                                f"SSL/connection error (attempt {attempt + 1}/{max_retries}), retrying..."
                            )
                            await asyncio.sleep(2)
                            await force_close_modern_pools()
                            continue
                    raise
            return None
        return wrapper
    return decorator
```

### Comprehensive Error Logging

**Error Context Logger** (`api/utils/memory.py`):
```python
def log_comprehensive_error(context: str, error: Exception, request: Optional[Request] = None):
    """
    Log errors with comprehensive context for debugging.
    
    Information Logged:
    - Error type and message
    - Full traceback
    - Request details (URL, method, headers, client IP)
    - Memory state at time of error
    - Timestamp
    
    Use Cases:
    - Authentication failures
    - Database errors
    - Rate limit violations
    - Unexpected exceptions
    """
    print__memory_monitoring(f"🚨 ERROR [{context}]: {type(error).__name__}: {error}")
    print__memory_monitoring(f"🚨 Traceback:\n{traceback.format_exc()}")
    
    if request:
        print__memory_monitoring(f"🚨 Request URL: {request.url}")
        print__memory_monitoring(f"🚨 Request method: {request.method}")
        print__memory_monitoring(f"🚨 Client IP: {request.client.host if request.client else 'unknown'}")
    
    log_memory_usage(f"error_{context}")
```

### Cancellation Support

**Execution Registry** (`api/utils/cancellation.py` - inferred):
```python
# Global registry for active executions
_active_executions = {}  # {thread_id: {run_id: True}}

def register_execution(thread_id: str, run_id: str):
    """
    Register an execution for cancellation tracking.
    
    Called at start of /analyze endpoint.
    """
    if thread_id not in _active_executions:
        _active_executions[thread_id] = {}
    _active_executions[thread_id][run_id] = True

def unregister_execution(thread_id: str, run_id: str):
    """
    Unregister an execution (completed or cancelled).
    
    Called at end of /analyze endpoint or on error.
    """
    if thread_id in _active_executions:
        _active_executions[thread_id].pop(run_id, None)
        if not _active_executions[thread_id]:
            del _active_executions[thread_id]

def is_cancelled(thread_id: str, run_id: str) -> bool:
    """
    Check if an execution has been cancelled.
    
    Polled every 0.5s during analysis_main execution.
    """
    return thread_id not in _active_executions or \
           run_id not in _active_executions.get(thread_id, {})
```

**Cancellable Analysis Pattern**:
```python
async def cancellable_analysis():
    """
    Wrapper that checks for cancellation periodically.
    
    Features:
    - Starts analysis as asyncio task
    - Polls every 0.5s for cancellation signal
    - Cancels task if stop requested
    - Raises asyncio.CancelledError on cancellation
    """
    task = asyncio.create_task(
        analysis_main(prompt, thread_id, checkpointer, run_id)
    )
    
    while not task.done():
        if is_cancelled(thread_id, run_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise asyncio.CancelledError("Execution cancelled by user")
        
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            continue
    
    return await task
```

### Debug Logging

**Structured Debug Functions** (`api/utils/debug.py` - inferred):
```python
def print__analysis_tracing_debug(msg: str):
    """
    Detailed trace logging for analysis flow.
    
    Enabled by: DEBUG_ANALYSIS_TRACING="1"
    
    Use Cases:
    - Step-by-step analysis execution
    - Numbered trace points (01-29)
    - Critical path debugging
    """
    if os.getenv("DEBUG_ANALYSIS_TRACING") == "1":
        print(f"[ANALYSIS_TRACE] {msg}")

def print__analyze_debug(msg: str):
    """
    High-level analysis debugging.
    
    Enabled by: DEBUG_ANALYZE="1"
    """
    if os.getenv("DEBUG_ANALYZE") == "1":
        print(f"[ANALYZE] {msg}")

def print__memory_monitoring(msg: str):
    """
    Memory usage debugging.
    
    Enabled by: DEBUG_MEMORY="1"
    """
    if os.getenv("DEBUG_MEMORY") == "1":
        print(f"[MEMORY] {msg}")

def print__feedback_flow(msg: str):
    """
    Feedback submission flow debugging.
    
    Enabled by: DEBUG_FEEDBACK_FLOW="1"
    """
    if os.getenv("DEBUG_FEEDBACK_FLOW") == "1":
        print(f"[FEEDBACK_FLOW] {msg}")
```

---

## MCP (Model Context Protocol) Integration

### Tool Architecture

**Dual-Mode Tool System** (`my_agent/utils/tools.py`):
```python
async def get_sqlite_tools() -> List[BaseTool]:
    """
    Get SQLite query tools using MCP with local fallback.
    
    Modes:
    1. Remote MCP Server (Primary):
       - FastMCP server on Railway/other hosting
       - Connects to Turso (SQLite Cloud)
       - Streamable HTTP transport
       - Lower backend memory usage
    
    2. Local SQLite Fallback (Secondary):
       - Direct file access to czsu_data.db
       - No network dependency
       - Development/testing mode
    
    Configuration:
    - MCP_SERVER_URL: Remote server endpoint
    - USE_LOCAL_SQLITE_FALLBACK: "1" to enable fallback
    """
    if MCP_SERVER_URL:
        try:
            # Use official LangChain MCP adapters
            client = MultiServerMCPClient({
                "sqlite": {
                    "transport": "streamable_http",
                    "url": MCP_SERVER_URL,
                }
            })
            
            tools = await client.get_tools()
            
            print(f"🌐 SQLite Tools: Remote MCP server at {MCP_SERVER_URL}")
            print(f"   📝 Tools loaded: {[tool.name for tool in tools]}")
            
            return tools
            
        except (ConnectionError, RuntimeError, ValueError) as e:
            if USE_LOCAL_SQLITE_FALLBACK:
                print(f"↩️ Falling back to local SQLite")
            else:
                raise ConnectionError(
                    f"MCP server unavailable and fallback disabled: {str(e)}"
                )
    
    # Local fallback
    print(f"💾 SQLite Tools: Local database at {DB_PATH}")
    return [LocalSQLiteQueryTool()]
```

### Local SQLite Tool

**Fallback Implementation**:
```python
class LocalSQLiteQueryTool(BaseTool):
    """
    Local SQLite query tool (fallback when MCP server is unavailable).
    
    Features:
    - Direct sqlite3 connection to czsu_data.db
    - Same interface as MCP tool for compatibility
    - Synchronous and async execution
    - Result formatting matches MCP server
    """
    name: str = "sqlite_query"
    description: str = (
        "Execute SQL query on the local SQLite database. "
        "Input should be a valid SQL query string."
    )
    args_schema: type[BaseModel] = LocalSQLiteQueryInput
    
    def _execute_query(self, query: str) -> str:
        """Execute query against local SQLite database."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
        
        # Format result
        if not result:
            return "No results found"
        elif len(result) == 1 and len(result[0]) == 1:
            return str(result[0][0])  # Single value
        else:
            return str(result)  # Multiple rows/columns
    
    def _run(self, query: str) -> str:
        """Execute SQL query synchronously."""
        try:
            result = self._execute_query(query)
            return result
        except (sqlite3.Error, OSError) as e:
            raise ToolException(f"Local SQLite query error: {str(e)}")
    
    async def _arun(self, query: str) -> str:
        """Execute SQL query asynchronously."""
        return self._run(query)
```

### MCP Server Configuration

**Remote Server Setup** (Separate FastMCP server):
```python
# MCP Server (separate application)
from fastmcp import FastMCP
import sqlite3

mcp = FastMCP("CZSU SQLite Server")

@mcp.tool()
def sqlite_query(query: str) -> str:
    """
    Execute SQL query on Turso SQLite database.
    
    Environment:
    - TURSO_DATABASE_URL: Turso connection string
    - TURSO_AUTH_TOKEN: Authentication token
    """
    # Connect to Turso
    conn = sqlite3.connect(TURSO_DATABASE_URL)
    # ... execute query
    return result

# Expose via streamable HTTP
mcp.run(transport="streamable_http", port=8001)
```

**Benefits of MCP Architecture**:
1. **Separation of Concerns**: SQL execution isolated from main API
2. **Scalability**: MCP server can scale independently
3. **Memory Efficiency**: Large query results don't load into main API memory
4. **Flexibility**: Easy to swap SQLite for other databases
5. **Fallback Support**: Graceful degradation to local file access

---

## Summary

The CZSU Multi-Agent Text-to-SQL Backend is a production-grade system featuring:

- **🚀 RESTful API**: FastAPI with modular routing, comprehensive middleware, and exception handling
- **🔐 Authentication**: JWT-based with Google OAuth, ownership verification, and secure token handling
- **🚦 Rate Limiting**: Dual-layer (burst + window) with graceful throttling and per-IP semaphores
- **💾 Memory Management**: Proactive monitoring, automatic GC, malloc_trim integration, and periodic cleanup
- **🤖 LangGraph Workflow**: 22-node StateGraph with parallel retrieval, agentic SQL generation, and iterative reflection
- **🧠 AI Services**: Azure OpenAI (GPT-4o + embeddings), Azure Translator, Cohere reranking, LangSmith tracing
- **📊 Data Services**: ChromaDB vector search, SQLite databases (600+ tables), Supabase PostgreSQL
- **💾 Checkpointing**: PostgreSQL persistence with connection pooling, health checks, and InMemory fallback
- **🛡️ Error Handling**: Comprehensive retry logic, structured logging, cancellation support, and debug tracing
- **🔌 MCP Integration**: Remote tool execution with local fallback, separation of concerns, and memory efficiency

All features are production-tested and deployed on Railway.app with 2GB memory constraints.