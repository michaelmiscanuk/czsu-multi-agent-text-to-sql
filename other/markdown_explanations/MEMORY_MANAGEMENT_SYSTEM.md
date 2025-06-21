# Memory Management & Recovery System

## Overview

This document explains the comprehensive memory management and recovery system implemented to address memory leaks, fragmentation issues, and frontend timeout problems in the CZSU Multi-Agent Text-to-SQL application.

## 🚨 Problem Statement

The application was experiencing several critical issues:

1. **Memory Leak (309MB per analysis)** - Each analysis request was consuming excessive memory that wasn't being released
2. **Memory Fragmentation** - VMS/RSS ratios > 2.0 indicating fragmented memory allocation
3. **Frontend Timeouts** - Backend completing successfully but HTTP responses not reaching the frontend
4. **Platform Instability** - Memory pressure causing application restarts

## 🔧 Solution Architecture

### 1. Memory Leak Prevention System

#### Pre-Analysis Cleanup
```python
# MEMORY LEAK PREVENTION: Pre-analysis cleanup
print__memory_monitoring("🧹 PRE-ANALYSIS: Running memory cleanup to prevent leaks")
aggressive_garbage_collection("pre_analysis")

# Check memory before analysis
pre_analysis_fragmentation = detect_memory_fragmentation()
if pre_analysis_fragmentation.get("potential_fragmentation"):
    print__memory_monitoring(f"⚠ PRE-ANALYSIS fragmentation detected - running handler")
    handle_memory_fragmentation()
```

#### Memory Growth Monitoring
```python
# POST-ANALYSIS MEMORY MONITORING: Check for memory growth patterns
post_analysis_memory = process.memory_info().rss / 1024 / 1024
memory_growth = post_analysis_memory - pre_analysis_memory

if memory_growth > 50:  # More than 50MB growth per analysis
    print__memory_monitoring(
        f"🚨 SUSPICIOUS MEMORY GROWTH: {memory_growth:.1f}MB per analysis!"
    )
```

### 2. Memory Fragmentation Detection & Handling

#### Detection Algorithm
```python
def detect_memory_fragmentation() -> dict:
    """Detect potential memory fragmentation issues."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    rss_mb = memory_info.rss / 1024 / 1024  # Physical RAM
    vms_mb = memory_info.vms / 1024 / 1024  # Virtual memory
    
    # High VMS to RSS ratio indicates fragmentation
    fragmentation_ratio = vms_mb / rss_mb if rss_mb > 0 else 0
    
    return {
        "potential_fragmentation": fragmentation_ratio > 2.0,
        "fragmentation_ratio": fragmentation_ratio,
        "significant_growth": memory_growth > 100
    }
```

#### Fragmentation Handler
```python
def handle_memory_fragmentation():
    """Handle detected memory fragmentation with specific countermeasures."""
    
    # 1. Comprehensive garbage collection across all generations
    for generation in range(3):
        collected = gc.collect(generation)
    
    # 2. Clear weak references
    weakref.finalize_registry()
    
    # 3. Reset connection pool to prevent pool fragmentation
    if hasattr(checkpointer, '_fragmentation_reset_needed'):
        checkpointer._fragmentation_reset_needed = True
    
    # 4. Clear rate limiting storage aggressively
    # 5. Force Python to return memory to OS (Windows-specific)
    # 6. Validate if fragmentation was resolved
```

### 3. Frontend Recovery System

#### Timeout Monitoring with PostgreSQL Recovery
```typescript
// MEMORY PRESSURE HANDLING: Promise.race with timeout monitor for automatic recovery
const timeoutMonitor = new Promise<AnalyzeResponse>((resolve, reject) => {
  const timeoutId = setTimeout(async () => {
    console.log('[TimeoutMonitor] ⏰ 30 second timeout reached - checking PostgreSQL');
    
    try {
      // Check if backend completed and saved to PostgreSQL
      const recoverySuccessful = await checkForNewMessagesAfterTimeout(
        currentThreadId, 
        messageCountBeforeRequest
      );
      
      if (recoverySuccessful) {
        console.log('[TimeoutMonitor] ✅ Recovery successful from PostgreSQL');
        resolve({
          // Synthetic response indicating recovery
          prompt: messageText,
          result: "Response recovered from database",
          // ... other required fields
          recovery_mode: true
        });
      } else {
        reject(new Error('Timeout: No response after 30 seconds'));
      }
    } catch (error) {
      reject(error);
    }
  }, 30000); // 30 second timeout
});

// Race between API call and timeout monitor
const data = await Promise.race([apiCall, timeoutMonitor]);
```

#### PostgreSQL-Based Message Recovery
```typescript
const checkForNewMessagesAfterTimeout = async (
  threadId: string, 
  beforeMessageCount: number
): Promise<boolean> => {
  try {
    const response = await authApiFetch<{ messages: ChatMessage[] }>(
      `/chat/${threadId}/messages`, 
      session.id_token
    );
    
    // Check if new messages appeared (backend completed)
    if (response.messages.length > beforeMessageCount) {
      console.log('[Recovery] ✅ New messages found - backend completed successfully');
      
      // Update UI with recovered messages
      setMessages(response.messages);
      return true;
    }
    
    return false;
  } catch (error) {
    console.log('[Recovery] ❌ Recovery check failed:', error);
    return false;
  }
};
```

### 4. Backend Response Guarantee System

#### Memory Pressure Response Handling
```python
# MEMORY PRESSURE RECOVERY: Save response state before attempting HTTP response
response_data = {"response": result, "thread_id": request.thread_id, "run_id": run_id}

# Final memory pressure check before HTTP response
current_memory = process.memory_info().rss / 1024 / 1024

if current_memory > 450:  # Close to 512MB limit
    print__memory_monitoring(f"⚠ HIGH MEMORY PRESSURE: {current_memory:.1f}MB")
    print__memory_monitoring(f"🔄 Response saved to PostgreSQL via checkpoints")
    print__memory_monitoring(f"🔄 Frontend can recover using thread_id: {request.thread_id}")
    
    # Emergency cleanup if very close to limit
    if current_memory > 480:
        # Emergency garbage collection
        # Force memory return to OS
        # Log emergency actions

# CRITICAL: Always attempt to return response regardless of memory state
return response_data
```

#### Fallback Response System
```python
except Exception as response_error:
    # CRITICAL: If response preparation fails, still try to return something
    print__memory_monitoring(f"❌ Error preparing HTTP response: {response_error}")
    
    # Fallback response to prevent frontend hanging
    fallback_response = {
        "response": result if 'result' in locals() else "Analysis completed",
        "thread_id": request.thread_id,
        "run_id": run_id,
        "error": f"Response preparation error: {str(response_error)}",
        "recovery_note": "Check PostgreSQL checkpoints for full results",
        "analysis_completed": True,
        "server_status": "completed_with_response_error"
    }
    
    return fallback_response
```

## 🔄 How It Works Together

### Normal Flow
1. **Pre-Analysis**: Memory cleanup and fragmentation check
2. **Analysis**: LangGraph execution with memory monitoring
3. **Post-Analysis**: Memory growth detection and cleanup
4. **Response**: HTTP response sent to frontend

### Memory Pressure Flow
1. **Detection**: Fragmentation ratio > 2.0 or memory > 450MB
2. **Handler**: Comprehensive cleanup and pool reset
3. **Recovery**: PostgreSQL state preservation
4. **Response**: Guaranteed response delivery or fallback

### Frontend Timeout Flow
1. **Timeout**: 30-second monitor triggers
2. **Check**: Query PostgreSQL for new messages
3. **Recovery**: Update UI with recovered messages
4. **Continuation**: User experience uninterrupted

### Emergency Flow
1. **Critical Memory**: > 480MB detected
2. **Emergency**: Immediate GC and OS memory return
3. **Logging**: Detailed diagnostics for debugging
4. **Restart Recommendation**: If fragmentation persists

## 📊 Memory Monitoring Metrics

### Key Indicators
- **RSS Memory**: Physical RAM usage (main leak indicator)
- **VMS Memory**: Virtual memory allocation
- **Fragmentation Ratio**: VMS/RSS (threshold: 2.0)
- **Memory Growth**: Per-request memory increase
- **GC Effectiveness**: Memory freed by garbage collection

### Thresholds
- **Normal**: < 200MB RSS, ratio < 2.0
- **Warning**: 200-400MB RSS, ratio 2.0-2.5
- **Critical**: > 400MB RSS, ratio > 2.5
- **Emergency**: > 480MB RSS, ratio > 3.0

## 🛡️ Recovery Mechanisms

### 1. Automatic Fragmentation Handling
- Triggered when fragmentation ratio > 2.0
- Comprehensive garbage collection
- Connection pool recreation
- Rate limiting storage cleanup

### 2. Frontend Timeout Recovery
- 30-second timeout monitor
- PostgreSQL message checking
- Automatic UI updates
- Seamless user experience

### 3. Emergency Memory Management
- Emergency GC at > 480MB
- OS memory return (Windows)
- Fallback response system
- Detailed diagnostic logging

### 4. Connection Pool Health
- Automatic pool recreation
- Health checks with timeouts
- Fragmentation-triggered resets
- Fallback to InMemorySaver

## 🔍 Debugging & Monitoring

### Debug Flags
```bash
export DEBUG=1  # Enable detailed logging
```

### Key Log Messages
- `🚨 FRAGMENTATION DETECTED` - Fragmentation handler triggered
- `🧹 MEMORY LEAK PREVENTION` - Cleanup operations
- `⏰ Timeout Monitor` - Frontend recovery triggered
- `✅ Recovery successful` - PostgreSQL recovery worked
- `🚨 EMERGENCY` - Critical memory pressure

### Health Endpoints
- `/health/memory` - Memory status and fragmentation
- `/health/database` - PostgreSQL connection health
- `/debug/pool-status` - Connection pool diagnostics

## 🎯 Performance Benefits

### Before Implementation
- **Memory Growth**: 309MB per analysis
- **Fragmentation**: Ratios > 3.0 common
- **Timeouts**: Frontend hanging frequently
- **Stability**: Platform restarts due to memory

### After Implementation
- **Memory Growth**: < 50MB per analysis (target)
- **Fragmentation**: Automatic detection and handling
- **Timeouts**: Seamless recovery via PostgreSQL
- **Stability**: Self-healing memory management

## 🔧 Configuration

### Environment Variables
```bash
MAX_CONCURRENT_ANALYSES=3    # Limit concurrent requests
MEMORY_LIMIT=536870912       # Memory limit in bytes (512MB)
DEBUG=1            # Enable debug logging
```

### Garbage Collection Tuning
```python
gc.set_threshold(500, 8, 8)  # Aggressive GC thresholds
```

### Connection Pool Settings
- **Timeout**: 120 seconds for creation
- **Health Check**: 5 seconds
- **Retry Logic**: 3 attempts with exponential backoff

## 🚀 Future Enhancements

1. **Predictive Memory Management**: ML-based memory usage prediction
2. **Dynamic Scaling**: Automatic resource adjustment
3. **Advanced Metrics**: Prometheus/Grafana integration
4. **Memory Profiling**: Detailed allocation tracking
5. **Smart Caching**: Intelligent result caching to reduce analysis load

## 📝 Maintenance

### Regular Monitoring
- Check memory health endpoints daily
- Monitor fragmentation ratios
- Review emergency log occurrences
- Validate recovery system effectiveness

### Troubleshooting
1. **High Memory**: Check for fragmentation, run manual cleanup
2. **Timeouts**: Verify PostgreSQL connectivity, check recovery logs
3. **Errors**: Review comprehensive error logs with context
4. **Performance**: Monitor memory growth patterns per request

This system provides a robust, self-healing architecture that handles memory pressure gracefully while ensuring users never lose their work or experience hanging interfaces. 