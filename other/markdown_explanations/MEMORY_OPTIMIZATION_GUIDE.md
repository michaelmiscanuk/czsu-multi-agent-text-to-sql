# Memory Optimization Guide

## The Problem

Your memory profiling showed two major memory consumers:

```
1    201.18 MB    [heap]     - Python heap memory
2    120.24 MB    [anon]     - Anonymous memory regions
```

**This is NOT a memory leak!** This is expected behavior from glibc's memory allocator (malloc).

### Why Memory Doesn't Get Released

When Python's garbage collector frees objects, the memory is released back to the **malloc pool**, but NOT back to the operating system. The malloc implementation keeps this memory in reserve for future allocations (for performance reasons).

For applications with variable workload (like yours), this means:
- When you process requests → memory grows
- When requests stop → Python GC frees objects
- **BUT** the RSS (Resident Set Size) stays high because malloc doesn't return memory to the OS

In memory-constrained environments (Railway with 500MB limit), this causes your app to stay near the limit even when idle.

## The Solution

We implemented a **three-layer memory optimization strategy**:

### 1. **force_release_memory()** - Active Memory Release
```python
def force_release_memory():
    """Forces memory to be returned to the OS"""
    # Step 1: Run Python garbage collector
    gc.collect()
    
    # Step 2: Call malloc_trim(0) to return memory to OS
    libc.malloc_trim(0)
```

This function combines:
- **Python GC**: Frees Python objects
- **malloc_trim()**: Forces glibc to return free memory to the OS

### 2. **Periodic Background Cleanup** - Automatic Memory Management

A background asyncio task runs every `MEMORY_CLEANUP_INTERVAL` seconds (configurable via environment variable):

```python
async def _memory_cleanup_loop():
    while True:
        await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)
        
        # Check current memory
        if memory > 100MB:
            cleanup_bulk_cache()      # Clear expired cache entries
            force_release_memory()    # Release memory to OS
```

**This task:**
- ✅ Runs in the background automatically
- ✅ Doesn't block your API requests
- ✅ Only cleans when memory is above threshold
- ✅ Logs its activity so you can see it working

### 3. **Enhanced Threshold Checks** - Proactive Management

The existing `check_memory_and_gc()` function now uses `force_release_memory()` when your app exceeds the memory threshold.

## How It Works

### On Startup
```python
# In api/main.py lifespan
start_memory_cleanup()  # Starts background task using MEMORY_CLEANUP_INTERVAL env var
```

This creates an asyncio task that runs **continuously in the background** while your uvicorn app is running.

### During Operation
Every `MEMORY_CLEANUP_INTERVAL` seconds, the cleanup task:
1. Checks current RSS memory
2. If > 100MB: Runs cache cleanup + memory release
3. Logs the results
4. Goes back to sleep for `MEMORY_CLEANUP_INTERVAL` seconds

### On Shutdown
```python
await stop_memory_cleanup()  # Gracefully stops the background task
```

## About libc.so.6

**Q: Do we need to install libc.so.6 in railway.toml?**

**A: NO!** `libc.so.6` is the **GNU C Library (glibc)** - the core C library on Linux.

- ✅ It's part of the base Linux system
- ✅ Python itself depends on it
- ✅ It's ALWAYS present on Linux systems
- ✅ If Python runs, libc is there

**You do NOT need to add it to railway.toml.**

### Fallback Handling

The code already has graceful fallback:

```python
try:
    libc = ctypes.CDLL("libc.so.6")
    MALLOC_TRIM_AVAILABLE = True
except (OSError, AttributeError):
    libc = None
    MALLOC_TRIM_AVAILABLE = False
```

If malloc_trim isn't available (extremely unlikely), the code still works - it just won't have that extra optimization.

## Configuration

Configure memory cleanup behavior using environment variables in your `.env` file:

```bash
# Enable/disable memory cleanup (default: enabled)
MEMORY_CLEANUP=1

# Cleanup interval in seconds (default: 60)
MEMORY_CLEANUP_INTERVAL=60
```

### Examples

```bash
# Run every 1 minute (current setting)
MEMORY_CLEANUP_INTERVAL=60

# Run every 2 minutes (less aggressive)
MEMORY_CLEANUP_INTERVAL=120

# Run every 30 seconds (more aggressive)
MEMORY_CLEANUP_INTERVAL=30

# Disable cleanup entirely
MEMORY_CLEANUP=0
```

**Note:** No code changes needed - just update your `.env` file and restart the app.

## Expected Results

### Before
```
Memory after spike: 400MB
Memory when idle:   390MB (stays high!)
```

### After
```
Memory after spike: 400MB
Memory when idle:   150MB (released back to OS)
```

The periodic cleanup will:
- ✅ Reduce idle memory consumption
- ✅ Prevent hitting Railway's 500MB limit when idle
- ✅ Allow memory to scale up during usage
- ✅ Return memory to OS when not needed

## Monitoring

You'll see logs like this:

```
[memory-cleanup] Background cleanup task running every 60s
[memory-cleanup] Running periodic cleanup (RSS: 380.5MB)
🧹 Memory Release: 180.3MB freed | RSS: 380.5MB → 200.2MB | GC collected: 450 objects | malloc_trim: ✓
[memory-cleanup] Completed - Cache entries cleaned: 12, Memory freed: 180.3MB
```

**Note:** The "60s" in the log will match your `MEMORY_CLEANUP_INTERVAL` setting.

## Additional Optimization Options

If you still experience memory issues, consider:

1. **Alternative Memory Allocators**
   - **jemalloc**: Better memory release behavior
   - **tcmalloc**: Google's malloc with better fragmentation
   
   Add to railway.toml:
   ```toml
   RAILPACK_DEPLOY_APT_PACKAGES = "libsqlite3-0 libjemalloc2"
   LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
   ```

2. **Environment Variables** (tune glibc behavior)
   ```toml
   MALLOC_TRIM_THRESHOLD_ = "128000"     # Trim when 128KB available
   MALLOC_MMAP_THRESHOLD_ = "131072"     # Use mmap for large allocations
   ```

3. **More Aggressive Cleanup**
   ```bash
   # In your .env file
   MEMORY_CLEANUP_INTERVAL=30  # Every 30 seconds
   ```

## Testing

To verify it's working:
1. Make several API requests to increase memory
2. Wait 60+ seconds without requests
3. Check your memory profiler output
4. You should see RSS drop significantly

## References

- [malloc_trim documentation](https://man7.org/linux/man-pages/man3/malloc_trim.3.html)
- [Python Memory Management](https://docs.python.org/3/c-api/memory.html)
- [glibc malloc tunables](https://www.gnu.org/software/libc/manual/html_node/Memory-Allocation-Tunables.html)
