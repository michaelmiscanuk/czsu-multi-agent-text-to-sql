# Rate Limiting & Throttling Middleware

## Summary

The CZSU API implements intelligent rate limiting middleware to protect against API abuse, DDoS attacks, and resource exhaustion while maintaining a positive user experience. Unlike traditional rate limiters that immediately reject requests exceeding limits, our implementation uses a **wait-instead-of-reject** strategy that attempts to delay requests for up to 5 seconds before giving up.

**Why We Use This:**

The multi-agent text-to-SQL system is computationally expensive, with each query potentially triggering multiple LLM API calls, database operations, and vector searches. Without rate limiting, a malicious actor or poorly configured client could easily overwhelm the server, causing memory exhaustion and service degradation for all users. The two-tier protection strategy (burst + window limits) handles both rapid-fire attacks and sustained abuse patterns.

The wait-and-retry approach significantly improves user experience compared to instant rejection. When a legitimate user encounters a rate limit (perhaps from refreshing a page or retrying a failed request), the middleware automatically waits for capacity instead of forcing the client to implement complex retry logic. This is especially valuable for interactive web applications where users expect immediate feedback rather than cryptic "too many requests" errors.

**Key Benefits:**

- **Server Protection:** Prevents resource exhaustion from abusive clients (max 8 concurrent requests per IP, 100 requests per minute)
- **Fair Resource Allocation:** Per-IP limits ensure one user can't monopolize server capacity
- **Better UX:** Automatic wait-and-retry reduces failed requests and eliminates need for client-side retry logic
- **Graceful Degradation:** System remains responsive under load by queuing requests rather than crashing
- **Security:** Mitigates DDoS attacks and credential stuffing attempts through temporal rate controls

## Overview

The CZSU API implements a **wait-instead-of-reject** rate limiting strategy that provides a better user experience by attempting to delay requests instead of immediately rejecting them when limits are exceeded.

## Architecture

### Component Stack

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Application (api/main.py)                      │
│  └─ Middleware Registration                             │
│     └─ setup_throttling_middleware(app)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Throttling Middleware (api/middleware/rate_limiting.py)│
│  └─ throttling_middleware(request, call_next)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Rate Limiting Logic (api/utils/rate_limiting.py)       │
│  ├─ wait_for_rate_limit(client_ip)                      │
│  ├─ check_rate_limit_with_throttling(client_ip)         │
│  └─ check_rate_limit(client_ip)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Configuration & Storage (api/config/settings.py)       │
│  ├─ throttle_semaphores (per-IP concurrency)            │
│  ├─ rate_limit_storage (timestamp tracking)             │
│  └─ Rate limit constants (REQUESTS, WINDOW, BURST)      │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

Defined in `api/config/settings.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_REQUESTS` | 100 | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | 60s | Time window for rate limiting |
| `RATE_LIMIT_BURST` | 20 | Additional burst allowance |
| `RATE_LIMIT_MAX_WAIT` | 5s | Max wait time before rejection |

### Concurrency Control

```python
# api/config/settings.py
throttle_semaphores = defaultdict(lambda: asyncio.Semaphore(8))
```

**Per-IP Limits:**
- Maximum 8 concurrent requests per IP address
- Auto-creates semaphore for new IPs
- Prevents resource monopolization

## Request Flow

### 1. Middleware Entry (`throttling_middleware`)

**Location:** `api/middleware/rate_limiting.py`

```python
async def throttling_middleware(request: Request, call_next):
```

**Steps:**
1. Check if endpoint is exempted
2. Extract client IP address
3. Acquire semaphore for IP (limits concurrency)
4. Call `wait_for_rate_limit(client_ip)`
5. If successful: Process request
6. If failed: Return 429 response

### 2. Wait Logic (`wait_for_rate_limit`)

**Location:** `api/utils/rate_limiting.py`

```python
async def wait_for_rate_limit(client_ip: str) -> bool:
```

**Algorithm:**
```
FOR attempt = 1 TO 3:
    rate_info = check_rate_limit_with_throttling(client_ip)
    
    IF allowed:
        Add timestamp to storage
        RETURN True
    
    IF suggested_wait > RATE_LIMIT_MAX_WAIT:
        RETURN False  # Give up
    
    SLEEP(suggested_wait)
    
RETURN False  # Max attempts exceeded
```

### 3. Rate Check (`check_rate_limit_with_throttling`)

**Location:** `api/utils/rate_limiting.py`

```python
def check_rate_limit_with_throttling(client_ip: str) -> dict:
```

**Returns:**
```python
{
    "allowed": bool,              # Request allowed?
    "suggested_wait": float,      # Seconds to wait
    "burst_count": int,           # Requests in last 10s
    "window_count": int,          # Requests in window
    "burst_limit": int,           # Burst limit (20)
    "window_limit": int           # Window limit (100)
}
```

**Logic:**

1. **Cleanup Old Entries:**
   ```python
   rate_limit_storage[client_ip] = [
       ts for ts in rate_limit_storage[client_ip]
       if now - ts < RATE_LIMIT_WINDOW
   ]
   ```

2. **Check Burst Limit (10s window):**
   ```python
   recent_requests = [ts for ts in storage if now - ts < 10]
   if len(recent_requests) >= RATE_LIMIT_BURST:
       # Calculate wait until oldest burst request expires
   ```

3. **Check Window Limit (60s window):**
   ```python
   if len(storage) >= RATE_LIMIT_REQUESTS:
       # Calculate wait until oldest request expires
   ```

## Exempted Endpoints

**Location:** `api/middleware/rate_limiting.py`

```python
if request.url.path in ["/health", "/docs", "/openapi.json", "/debug/pool-status"]:
    return await call_next(request)
```

**Reason:**
- `/health` - Monitoring must always work
- `/docs`, `/openapi.json` - Documentation access
- `/debug/pool-status` - Troubleshooting

## Response Format

### Success (200-299)
Request processed normally

### Rate Limited (429)

```json
{
    "detail": "Rate limit exceeded. Please wait 3.2s before retrying.",
    "retry_after": 4,
    "burst_usage": "25/20",
    "window_usage": "110/100"
}
```

**Headers:**
```
Retry-After: 4
```

## Data Storage

### Rate Limit Tracking

**Location:** `api/config/settings.py`

```python
rate_limit_storage = defaultdict(list)
```

**Structure:**
```python
{
    "192.168.1.100": [1700000001.5, 1700000002.3, ...],  # Unix timestamps
    "10.0.0.50": [1700000010.1, 1700000011.8, ...],
    ...
}
```

**Cleanup:** Automatically removes timestamps older than `RATE_LIMIT_WINDOW` (60s)

### Semaphore Storage

**Location:** `api/config/settings.py`

```python
throttle_semaphores = defaultdict(lambda: asyncio.Semaphore(8))
```

**Structure:**
```python
{
    "192.168.1.100": Semaphore(8),  # Max 8 concurrent
    "10.0.0.50": Semaphore(8),
    ...
}
```

## Example Scenarios

### Scenario 1: Normal Usage

```
Client makes 50 requests in 60 seconds
├─ burst_count: 5/20 (last 10s)
├─ window_count: 50/100 (last 60s)
└─ Result: ✅ Allowed immediately
```

### Scenario 2: Burst Exceeded, Wait Succeeds

```
Client makes 25 requests in 8 seconds
├─ burst_count: 25/20 (EXCEEDED)
├─ window_count: 25/100 (OK)
├─ suggested_wait: 2.3s (until oldest burst expires)
├─ Action: Sleep 2.3s
└─ Result: ✅ Allowed after wait
```

### Scenario 3: Long Wait Required, Rejected

```
Client makes 120 requests in 10 seconds
├─ burst_count: 120/20 (EXCEEDED)
├─ window_count: 120/100 (EXCEEDED)
├─ suggested_wait: 52.7s (until window clears)
├─ max_wait: 5s (configured)
└─ Result: ❌ Rejected with 429
```

### Scenario 4: Concurrency Limit

```
Client has 8 concurrent requests running
├─ New request arrives
├─ Semaphore full (8/8)
├─ Action: Wait for semaphore
└─ Result: Queued until slot available
```

## Integration Points

### 1. Main Application

**File:** `api/main.py`

```python
from api.middleware.rate_limiting import setup_throttling_middleware

# Register middleware
setup_throttling_middleware(app)
```

### 2. Logging

**File:** `api/middleware/rate_limiting.py`

```python
from api.utils.memory import log_comprehensive_error

log_comprehensive_error(
    "rate_limit_exceeded_after_wait",
    Exception(error_msg),
    request
)
```

### 3. Debug Output

**File:** `api/utils/rate_limiting.py`

```python
from api.utils.debug import print__debug

print__debug(f"⏳ Throttling request from {client_ip}: waiting {wait}s")
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Exempted endpoints | <0.01ms overhead |
| Within limits | ~0.5-1ms (timestamp check) |
| With wait | 100ms - 5000ms (sleep time) |
| Rejected | ~0.5-1ms (check + response) |
| Memory per IP | ~1-5 KB (timestamps + semaphore) |

## Testing

### Unit Test Example

```python
def test_rate_limit_enforcement():
    client_ip = "192.168.1.100"
    
    # Make 120 burst requests
    for i in range(120):
        result = wait_for_rate_limit(client_ip)
    
    # 121st should be rejected
    assert not wait_for_rate_limit(client_ip)
```

### Load Test Considerations

- Simulate burst traffic patterns
- Test concurrent request limits (8 per IP)
- Verify wait times under load
- Check semaphore behavior

## Security Considerations

1. **DDoS Protection:** Per-IP limits prevent flooding
2. **IP Spoofing:** Uses `request.client.host` (trusted by reverse proxy)
3. **Bypass Prevention:** No authentication bypass for rate limits
4. **Information Disclosure:** Generic error messages only

## Monitoring

### Key Metrics

- Rate limit rejections per IP
- Total rejections per time period
- Average wait times
- Semaphore saturation (8/8 slots full)

### Log Analysis

```bash
# Find rate limit events
grep "rate_limit_exceeded_after_wait" logs/

# Identify problematic IPs
grep "Rate limit exceeded" logs/ | cut -d' ' -f5 | sort | uniq -c | sort -rn
```

## Future Enhancements

- [ ] Per-user rate limits (authenticated)
- [ ] Tiered limits (free vs premium)
- [ ] Redis-based distributed storage
- [ ] Adaptive rate limiting based on server load
- [ ] Real-time monitoring dashboard
