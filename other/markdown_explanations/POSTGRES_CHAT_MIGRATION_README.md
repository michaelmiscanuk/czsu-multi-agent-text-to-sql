# PostgreSQL Chat System Migration

This document explains the migration from IndexedDB to PostgreSQL for chat management in the czsu-multi-agent-text-to-sql application.

## 🎯 Overview

The chat system has been migrated from client-side IndexedDB storage to server-side PostgreSQL storage to solve several critical issues:

1. **Data Persistence**: Chat history now persists across browser sessions, devices, and server restarts
2. **Reliability**: No more lost chats due to browser storage limitations or clearing
3. **User Isolation**: Proper server-side user isolation with authentication
4. **Performance**: Better performance for large chat histories

## 🗄️ Database Schema

### New Table: `users_threads_runs`

```sql
CREATE TABLE users_threads_runs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    email VARCHAR(255) NOT NULL,
    thread_id VARCHAR(255) NOT NULL,
    run_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(email, thread_id, run_id)
);

-- Indexes for performance
CREATE INDEX idx_users_threads_runs_email ON users_threads_runs(email);
CREATE INDEX idx_users_threads_runs_thread_id ON users_threads_runs(thread_id);
CREATE INDEX idx_users_threads_runs_email_timestamp ON users_threads_runs(email, timestamp DESC);
```

### Schema Explanation

- **timestamp**: When the thread/run was last updated (used for sorting)
- **email**: User's email address for isolation
- **thread_id**: Unique identifier for chat conversation (same as before)
- **run_id**: Unique identifier for each question/analysis within a thread
- **created_at/updated_at**: Standard audit fields

## 🔄 Key Changes

### Before (IndexedDB)
```typescript
// Frontend-only storage
await listThreads(userEmail);
await saveThread(threadMeta);
await deleteThread(userEmail, threadId);
```

### After (PostgreSQL)
```typescript
// Server-side API calls
const response = await fetch('/api/chat-threads');
// Thread creation happens automatically on first message
await fetch(`/api/chat/${threadId}`, { method: 'DELETE' });
```

## 🚀 New API Endpoints

### 1. GET `/chat-threads`
Returns all chat threads for the authenticated user.

**Response:**
```json
[
  {
    "thread_id": "thread_123",
    "latest_timestamp": "2024-01-15T10:30:00Z",
    "run_count": 5
  }
]
```

### 2. DELETE `/chat/{thread_id}`
Deletes all data for a specific thread (both checkpoints and thread entries).

**Response:**
```json
{
  "message": "Checkpoint records and thread entries deleted",
  "deleted_counts": { "checkpoints": 3, "checkpoint_writes": 5 },
  "thread_entries_deleted": { "deleted_count": 5 },
  "thread_id": "thread_123",
  "user_email": "user@example.com"
}
```

### 3. POST `/analyze` (Enhanced)
The existing analyze endpoint now automatically creates thread run entries.

**Request:**
```json
{
  "prompt": "Your question",
  "thread_id": "thread_123"
}
```

**Response:** (includes new `run_id` field)
```json
{
  "prompt": "Your question",
  "result": "Analysis result",
  "thread_id": "thread_123",
  "run_id": "uuid-generated-run-id",
  ...
}
```

## 🔧 Implementation Details

### Backend Functions

#### `create_thread_run_entry(email, thread_id, run_id=None)`
Creates a new entry in the `users_threads_runs` table.
- Auto-generates `run_id` if not provided
- Handles duplicates gracefully with `ON CONFLICT`

#### `get_user_chat_threads(email)`
Retrieves all threads for a user, sorted by latest timestamp.
- Groups by `thread_id`
- Calculates `run_count` and `latest_timestamp`
- Orders by most recent first

#### `delete_user_thread_entries(email, thread_id)`
Deletes all entries for a specific user's thread.
- Used when user deletes a chat
- Returns count of deleted entries

### Integration Points

1. **Checkpointer Initialization**: `get_postgres_checkpointer()` now also sets up the new table
2. **Analysis Endpoint**: Creates thread run entry before processing
3. **Delete Endpoint**: Deletes both checkpoints and thread entries
4. **Authentication**: All endpoints require valid JWT token

## 🧪 Testing

### Running Tests

```bash
# Test PostgreSQL functionality
python test_postgres_chat_system.py

# Test API endpoints
python test_api_endpoints.py
```

### Test Coverage

- ✅ Thread run entry creation (auto-generated and custom run_ids)
- ✅ User thread retrieval (empty and with data)
- ✅ User isolation (users only see their own threads)
- ✅ Thread deletion
- ✅ Timestamp ordering
- ✅ Checkpointer integration
- ✅ API endpoint authentication
- ✅ Error handling

## 🔄 Migration Guide

### Step 1: Update Frontend Components

Replace IndexedDB calls with API calls:

```typescript
// OLD: IndexedDB
import { listThreads, deleteThread } from '@/components/utils';
const threads = await listThreads(userEmail);
await deleteThread(userEmail, threadId);

// NEW: PostgreSQL API
const response = await fetch('/api/chat-threads', {
  headers: { 'Authorization': `Bearer ${token}` }
});
const threads = await response.json();

await fetch(`/api/chat/${threadId}`, { 
  method: 'DELETE',
  headers: { 'Authorization': `Bearer ${token}` }
});
```

### Step 2: Remove IndexedDB Dependencies

1. Remove `idb` package dependency
2. Delete `frontend/src/components/utils.ts` (IndexedDB functions)
3. Update imports in chat components

### Step 3: Update State Management

```typescript
// OLD: Load from IndexedDB on component mount
useEffect(() => {
  listThreads(userEmail).then(setThreads);
}, [userEmail]);

// NEW: Load from PostgreSQL API
useEffect(() => {
  if (!session?.user?.email) return;
  
  fetch('/api/chat-threads', {
    headers: { 'Authorization': `Bearer ${session.id_token}` }
  })
  .then(r => r.json())
  .then(setThreads);
}, [session]);
```

### Step 4: Update Thread Creation

Thread creation is now automatic - no need to manually save thread metadata. The first message sent to a `thread_id` will create the thread entry.

### Step 5: Handle Edge Cases

- **Network errors**: Implement proper error handling for API calls
- **Authentication**: Handle token refresh and re-authentication
- **Loading states**: Show loading indicators for API operations

## 🔒 Security

### Authentication
- All endpoints require valid JWT tokens
- User isolation enforced at database level
- Email extracted from JWT payload

### Row Level Security (RLS)
The `users_threads_runs` table has RLS enabled with policies for Supabase compatibility.

### Data Validation
- Email and thread_id validation
- UUID format validation for run_ids
- SQL injection prevention with parameterized queries

## 🚀 Benefits

### 1. **Cross-Device Sync**
Chats are now available on any device where the user logs in.

### 2. **Server Restart Resilience**
No more lost conversations when the server restarts.

### 3. **Better Performance**
- Database queries are optimized with proper indexes
- No browser storage limitations
- Efficient pagination possible for large chat histories

### 4. **Audit Trail**
Complete audit trail of when threads and runs were created/updated.

### 5. **Analytics Potential**
Can now analyze user behavior patterns, popular thread topics, etc.

## ⚠️ Known Issues & Considerations

### 1. **Network Dependency**
The frontend now requires network connectivity to load chat history.

### 2. **API Rate Limits**
Consider implementing rate limiting for the new endpoints in production.

### 3. **Data Migration**
Existing IndexedDB data will be lost. Consider implementing a one-time migration script if needed.

### 4. **Caching Strategy**
Consider implementing client-side caching for better UX when network is slow.

## 🔧 Configuration

### Environment Variables

Ensure these PostgreSQL connection variables are set:

```bash
user=your_db_user
password=your_db_password
host=your_db_host
port=5432
dbname=your_db_name
GOOGLE_CLIENT_ID=your_google_client_id
```

### Connection Pool Settings

The system uses connection pooling with these defaults:
- `max_size=3`
- `min_size=1`
- `autocommit=True`

## 📊 Monitoring

### Health Checks

The system includes health check functions:
- `test_connection_health()`: Verifies database connectivity
- Connection pool monitoring with statistics

### Logging

All operations include detailed logging:
- Thread creation/deletion events
- User isolation verification
- Error conditions with full context

## 🎯 Future Enhancements

1. **Thread Metadata**: Store thread titles and descriptions
2. **Message Search**: Full-text search across chat history
3. **Export/Import**: Bulk operations for chat data
4. **Analytics Dashboard**: Usage patterns and popular topics
5. **Real-time Updates**: WebSocket support for live chat updates
6. **Conversation Sharing**: Share specific threads with other users

## 📞 Support

If you encounter issues with the migration:

1. Check PostgreSQL connection settings
2. Verify JWT token authentication
3. Run the test suite to validate functionality
4. Check server logs for detailed error messages

The new system has been thoroughly tested and should provide a much more robust chat experience! 