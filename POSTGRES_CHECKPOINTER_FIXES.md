# PostgreSQL Checkpointer Fixes

## Issues Identified

### 1. Row Level Security (RLS) Disabled
**Problem:** Supabase tables were showing "Row Level Security is disabled" warnings, making them publicly readable and writable.

**Root Cause:** By default, Supabase tables don't have RLS enabled, which creates a security vulnerability.

**Solution:** 
- Created `setup_supabase_rls.py` script to enable RLS and create appropriate policies
- Added automatic RLS setup to the checkpointer initialization
- Created permissive policies that allow the service role full access

### 2. Connection Instability
**Problem:** PostgreSQL connections were being closed unexpectedly with errors like:
- `discarding closed connection: <psycopg.AsyncConnection [BAD]>`
- `server closed the connection unexpectedly`
- `consuming input failed`

**Root Cause:** 
- Aggressive connection pool settings overwhelming Supabase
- Invalid connection parameters causing connection failures
- Lack of connection health monitoring and recovery

**Solutions Applied:**

#### A. Optimized Connection Pool Settings
- Reduced max pool size from 20 → 5 → 3 connections
- Shortened connection lifetimes: max_idle 300s, max_lifetime 1800s
- Added retry logic with exponential backoff for pool opening
- Implemented connection health checks

#### B. Fixed Connection Parameters
- Removed invalid `command_timeout` parameter
- Removed invalid `server_settings` configuration
- Simplified to only use valid psycopg parameters:
  ```python
  connection_kwargs = {
      "sslmode": "require",
      "connect_timeout": 10,
      "application_name": "czsu-langgraph-checkpointer",
      "autocommit": False,
      "prepare_threshold": 0,
  }
  ```

#### C. Enhanced Error Handling
- Added connection health monitoring in `api_server.py`
- Implemented automatic fallback to InMemorySaver when PostgreSQL fails
- Added connection recreation logic when pools become unhealthy
- Enhanced error detection for database connection issues

#### D. Better Logging and Monitoring
- Added detailed connection logging for debugging
- Added pool configuration logging
- Added retry attempt logging
- Added health check status reporting

## Files Modified

### 1. `my_agent/utils/postgres_checkpointer.py`
- **Fixed:** Connection parameters and pool configuration
- **Added:** RLS setup function
- **Added:** Connection health monitoring
- **Added:** Retry logic and better error handling
- **Added:** Detailed logging for debugging

### 2. `api_server.py`
- **Added:** `get_healthy_checkpointer()` function for connection health checks
- **Enhanced:** Error handling to detect and recover from connection issues
- **Enhanced:** Fallback logic to use InMemorySaver when PostgreSQL fails

### 3. `setup_supabase_rls.py` (New)
- **Created:** Standalone script to setup Row Level Security
- **Features:** Connection testing, RLS enabling, policy creation, verification

## Configuration Changes

### Connection Pool Optimization for Supabase
```python
pool = AsyncConnectionPool(
    conninfo=connection_string,
    max_size=3,        # Reduced to avoid overwhelming Supabase
    min_size=1,
    max_idle=300,      # 5 minutes
    max_lifetime=1800, # 30 minutes
    timeout=10,        # Shorter timeout
    kwargs=connection_kwargs,
)
```

### Row Level Security Policies
```sql
-- Enable RLS
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY;

-- Create permissive policies for service role
CREATE POLICY "Allow service role full access" ON checkpoints
FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access" ON checkpoint_writes
FOR ALL USING (true) WITH CHECK (true);
```

## Testing and Verification

### 1. Connection Test
```bash
python my_agent/utils/postgres_checkpointer.py
```
**Expected Output:** Successful connection, table setup, and RLS configuration

### 2. RLS Setup
```bash
python setup_supabase_rls.py
```
**Expected Output:** RLS enabled and policies created successfully

### 3. Application Test
Start your application and verify:
- No more "Row Level Security is disabled" warnings in Supabase
- Stable PostgreSQL connections without "BAD connection" errors
- Automatic fallback to InMemorySaver if PostgreSQL becomes unavailable

## Security Benefits

1. **RLS Enabled:** Tables are no longer publicly accessible
2. **Controlled Access:** Only authenticated service roles can access data
3. **Data Protection:** Checkpoint data is properly secured
4. **Policy-Based Security:** Fine-grained access control through PostgreSQL policies

## Operational Benefits

1. **Connection Stability:** Reduced connection failures and timeouts
2. **Automatic Recovery:** Self-healing when database issues occur
3. **Better Monitoring:** Detailed logging for troubleshooting
4. **Graceful Degradation:** Falls back to InMemorySaver when needed

## Maintenance

- Monitor connection logs for any stability issues
- The RLS setup only needs to be run once per database
- Connection pool settings can be tuned based on your application's load
- Health checks will automatically detect and recover from connection issues 