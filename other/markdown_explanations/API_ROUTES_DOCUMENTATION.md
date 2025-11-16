# API Routes & Endpoints Documentation

**Generated:** November 16, 2025  
**Application:** CZSU Multi-Agent Text-to-SQL API  
**Version:** 1.0.0

This document provides comprehensive documentation of all API routes and their endpoints in the CZSU (Czech Statistical Office) Multi-Agent Text-to-SQL application.

---

## Table of Contents
- [Core Analysis Routes](#core-analysis-routes)
- [Chat & Messaging Routes](#chat--messaging-routes)
- [Data Catalog Routes](#data-catalog-routes)
- [Feedback & Sentiment Routes](#feedback--sentiment-routes)
- [Health & Monitoring Routes](#health--monitoring-routes)
- [Debug & Administration Routes](#debug--administration-routes)
- [Utility Routes](#utility-routes)
- [Root & Documentation Routes](#root--documentation-routes)

---

## Core Analysis Routes

### `/analyze`

**Purpose:** Main AI-powered text-to-SQL analysis engine that converts natural language queries into SQL and executes them against the CZSU database.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/analyze` | POST | Execute natural language to SQL analysis | Converts user's natural language question into SQL using a multi-agent AI system. The endpoint:<br>• Parses natural language queries<br>• Identifies relevant CZSU datasets using semantic search<br>• Generates appropriate SQL queries<br>• Executes queries against the database<br>• Returns formatted natural language answers with metadata<br>• Supports iterative query refinement (max 2 iterations)<br>• Implements cancellation tracking for long-running queries<br>• Provides fallback to InMemorySaver if database connection fails<br>• Rate-limited to prevent resource exhaustion<br>• Enforces 4-minute timeout per analysis<br>• Returns comprehensive metadata including datasets used, SQL queries, query results, PDF chunks, and follow-up prompts |

**Key Features:**
- Multi-user support with JWT authentication
- Automatic metadata extraction including datasets used, SQL queries, and follow-up suggestions
- Concurrent execution limiting (controlled by `MAX_CONCURRENT_ANALYSES`)
- Comprehensive error handling with prepared statement error detection
- Memory monitoring and garbage collection
- Cancellation support for user-initiated stops
- Thread and run ID tracking for each analysis session

---

### `/stop-execution`

**Purpose:** User cancellation control for running analysis operations.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/stop-execution` | POST | Cancel a running analysis execution | Allows authenticated users to cancel their own running analysis operations. The endpoint:<br>• Validates user authentication and ownership<br>• Registers cancellation request by thread_id and run_id<br>• Ensures multi-user safety (users can only stop their own executions)<br>• Tracks active execution count for monitoring<br>• Returns success even if execution already completed<br>• Prevents errors when stopping already-finished operations<br>• Provides graceful cancellation without data corruption |

**Key Features:**
- Multi-user safety with ownership verification
- Graceful handling of already-completed executions
- Real-time execution tracking
- No exceptions thrown for completed executions

---

## Chat & Messaging Routes

### `/chat-threads`

**Purpose:** Thread management for conversational interactions with the AI system.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat-threads` | GET | Retrieve paginated list of user's chat threads | Returns all chat conversation threads for the authenticated user with comprehensive metadata. The endpoint:<br>• Fetches threads ordered by most recent activity<br>• Provides pagination support (configurable page size 1-50)<br>• Returns thread metadata including:<br>&nbsp;&nbsp;- Thread ID for reference in other endpoints<br>&nbsp;&nbsp;- Latest activity timestamp<br>&nbsp;&nbsp;- Number of query runs in the thread<br>&nbsp;&nbsp;- Title derived from first query<br>&nbsp;&nbsp;- Full first prompt for context<br>• Calculates total count for pagination UI<br>• Indicates if more pages are available<br>• Handles database errors gracefully with empty results |

**Key Features:**
- Efficient pagination with has_more indicator
- Thread title auto-generation from first prompt
- Timestamp tracking for activity sorting
- Comprehensive error handling

---

### `/chat/{thread_id}`

**Purpose:** Individual thread management and deletion.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat/{thread_id}` | DELETE | Delete all checkpoint records and thread entries for a specific thread | Permanently removes all data associated with a chat thread including:<br>• All PostgreSQL checkpoint records for the thread<br>• All thread run entries from users_threads_runs table<br>• User ownership verification before deletion<br>• Cascade deletion of related sentiment data<br>• Memory cleanup operations<br>• Graceful handling of connection errors<br>• Returns detailed deletion statistics<br>• Security enforcement - users can only delete their own threads |

**Key Features:**
- Complete data removal including checkpoints and metadata
- Multi-user security with ownership verification
- Graceful degradation on database unavailability
- Detailed deletion reporting

---

### `/chat/{thread_id}/messages`

**Purpose:** Message history retrieval for chat threads.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat/{thread_id}/messages` | GET | Load conversation message history from PostgreSQL checkpoints | Retrieves complete message history for a specific thread, preserving original user prompts and AI responses. The endpoint:<br>• Loads messages from PostgreSQL checkpoint storage<br>• Preserves original user message text (not regenerated prompts)<br>• Extracts comprehensive metadata from checkpoints:<br>&nbsp;&nbsp;- User prompts and AI final answers<br>&nbsp;&nbsp;- SQL queries executed<br>&nbsp;&nbsp;- Datasets used in responses<br>&nbsp;&nbsp;- PDF documentation chunks referenced<br>&nbsp;&nbsp;- Follow-up prompt suggestions<br>&nbsp;&nbsp;- Query results and intermediate data<br>• Matches run_ids to messages for feedback tracking<br>• Security verification of thread ownership<br>• Handles checkpoint extraction from LangGraph state<br>• Gracefully degrades on database connection issues |

**Key Features:**
- Complete message history with all metadata
- Original user prompt preservation
- Per-message metadata extraction
- Thread ownership security
- Fallback handling for connection errors

---

### `/chat/{thread_id}/sentiments`

**Purpose:** Sentiment tracking for user satisfaction monitoring.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat/{thread_id}/sentiments` | GET | Retrieve sentiment values for all messages in a thread | Returns all user-submitted sentiment ratings for a specific chat thread. The endpoint:<br>• Fetches sentiment values (positive/negative/neutral) for each run_id<br>• Maps run_ids to their corresponding sentiment ratings<br>• Used by frontend to display thumbs up/down states<br>• Enables sentiment trend analysis<br>• Returns empty object if no sentiments recorded<br>• Verifies user ownership of thread<br>• Provides data for quality monitoring and UX improvements |

**Key Features:**
- Complete sentiment mapping by run_id
- User ownership verification
- Support for sentiment trend analysis
- Empty result handling for new threads

---

### `/chat/{thread_id}/run-ids`

**Purpose:** Run ID mapping for feedback and sentiment tracking.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat/{thread_id}/run-ids` | GET | Get run_ids for messages to enable feedback submission | Retrieves all run_ids associated with a thread's messages for feedback tracking. The endpoint:<br>• Returns list of run_ids with associated prompts and timestamps<br>• Enables frontend to map messages to run_ids for feedback<br>• Validates UUID format of run_ids<br>• Filters out invalid or null run_ids<br>• Orders by timestamp for chronological matching<br>• Required for connecting user feedback to specific AI responses<br>• Supports LangSmith feedback integration<br>• Provides data for quality tracking and debugging |

**Key Features:**
- UUID validation for data integrity
- Chronological ordering for accurate matching
- Prompt and timestamp metadata
- Null/invalid run_id filtering

---

### `/chat/all-messages-for-one-thread/{thread_id}`

**Purpose:** Comprehensive single-thread data retrieval with all metadata.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat/all-messages-for-one-thread/{thread_id}` | GET | Get complete thread data including messages, run_ids, and sentiments | Retrieves comprehensive data package for a single thread including:<br>• All chat messages with full metadata (prompts, answers, datasets, SQL, chunks)<br>• Complete list of run_ids with prompts and timestamps<br>• All sentiment values for the thread<br>• Per-message metadata extraction from checkpoints<br>• Automatic run_id matching to AI messages<br>• Security verification of thread ownership<br>• Used by analysis endpoint for metadata extraction<br>• Supports frontend message display with all context<br>• Single-call efficiency for complete thread state |

**Key Features:**
- Complete thread state in single response
- Automatic run_id to message matching
- Full metadata extraction per interaction
- Security enforcement with ownership check
- Efficient single-call design

---

### `/chat/all-messages-for-all-threads`

**Purpose:** Bulk loading of all user threads with intelligent caching.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/chat/all-messages-for-all-threads` | GET | Bulk load all chat messages for authenticated user with caching | Efficiently loads all threads and messages for a user with sophisticated caching. The endpoint:<br>• Retrieves all threads, messages, run_ids, and sentiments in optimized queries<br>• Implements intelligent caching with configurable timeout (default 60s)<br>• Uses per-user locking to prevent duplicate processing<br>• Processes threads with limited concurrency (max 3 concurrent)<br>• Returns structured data with messages, runIds, and sentiments dictionaries<br>• Cache hit returns instant response with appropriate headers<br>• Double-check locking pattern to prevent race conditions<br>• Graceful degradation on database errors<br>• Memory-efficient processing with semaphore-based throttling<br>• ETag and Cache-Control headers for HTTP caching |

**Key Features:**
- Intelligent multi-level caching
- Concurrency limiting to prevent resource exhaustion
- Cache hit optimization (sub-second responses)
- Per-user locking for consistency
- Comprehensive metadata in single call
- ETag-based cache validation

---

## Data Catalog Routes

### `/catalog`

**Purpose:** CZSU statistical data catalog browsing and search.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/catalog` | GET | Search and browse CZSU data catalog | Provides paginated access to the Czech Statistical Office dataset catalog. The endpoint:<br>• Returns list of available statistical datasets with descriptions<br>• Supports search filtering by selection code or description text<br>• Pagination with configurable page size (1-10000 records)<br>• Returns selection codes and extended descriptions<br>• Enables dataset discovery before querying<br>• Connects to SQLite catalog database<br>• Provides total count for pagination UI<br>• Supports wildcard text search across codes and descriptions<br>• Used by frontend for dataset exploration<br>• Helps users understand available data sources |

**Key Features:**
- Flexible text search across codes and descriptions
- High-volume pagination support (up to 10,000 per page)
- Total count for UI pagination
- Extended descriptions for dataset understanding

---

### `/data-tables`

**Purpose:** Database table listing with descriptions.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/data-tables` | GET | Get list of all available data tables with descriptions | Returns complete list of database tables with short descriptions. The endpoint:<br>• Lists all tables from czsu_data.db SQLite database<br>• Fetches short descriptions from selection_descriptions.db<br>• Filters system tables (sqlite_* excluded)<br>• Supports query parameter for filtering tables by name<br>• Maps selection codes to their descriptions<br>• Returns table name (selection_code) and short_description pairs<br>• Used for table browser and data exploration<br>• Enables users to see available datasets quickly<br>• Graceful handling of missing descriptions |

**Key Features:**
- Dual database query (data + descriptions)
- System table filtering
- Optional text-based filtering
- Description mapping with fallback

---

### `/data-table`

**Purpose:** Individual table data preview and schema inspection.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/data-table` | GET | Get column schema and data preview for a specific table | Returns detailed table structure and sample data for inspection. The endpoint:<br>• Retrieves column names and data types<br>• Returns up to 10,000 rows of sample data<br>• Enables frontend table preview and exploration<br>• Provides schema information for SQL query building<br>• Returns empty result for invalid/missing tables<br>• Sanitizes table names to prevent injection<br>• Used by data exploration UI components<br>• Helps users understand table structure before querying<br>• Supports debugging and data quality checks |

**Key Features:**
- Schema extraction (column names and types)
- Large sample data retrieval (10,000 rows)
- SQL injection protection
- Graceful error handling

---

## Feedback & Sentiment Routes

### `/feedback`

**Purpose:** User feedback collection for AI quality improvement.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/feedback` | POST | Submit user feedback to LangSmith for quality tracking | Collects user feedback on AI responses for quality monitoring and model improvement. The endpoint:<br>• Accepts binary rating (thumbs up/down) and/or text comment<br>• Validates run_id is valid UUID format<br>• Verifies user ownership of run_id before accepting feedback<br>• Submits feedback to LangSmith for tracking and analysis<br>• Supports score-only, comment-only, or combined feedback<br>• Used for identifying problematic responses<br>• Enables continuous quality improvement<br>• Prevents unauthorized feedback submission<br>• Provides data for model fine-tuning and prompt engineering<br>• Tracks user satisfaction metrics |

**Key Features:**
- Flexible feedback types (score/comment/both)
- UUID validation and ownership verification
- LangSmith integration for quality tracking
- Multi-user security enforcement
- Comprehensive error logging

---

### `/sentiment`

**Purpose:** Real-time sentiment tracking in the database.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/sentiment` | POST | Update sentiment rating for a specific run_id | Updates sentiment value in database for immediate UI reflection. The endpoint:<br>• Accepts positive/negative/neutral sentiment values<br>• Validates run_id UUID format<br>• Updates users_threads_runs table with sentiment<br>• Verifies user ownership before update<br>• Provides instant feedback to user<br>• Used for thumbs up/down UI state management<br>• Enables sentiment trend analysis<br>• Faster than feedback endpoint (no LangSmith call)<br>• Supports A/B testing and quality monitoring<br>• Returns success/failure status |

**Key Features:**
- Instant database update for UI responsiveness
- UUID validation and ownership check
- Sentiment value validation
- Fast response time (database-only operation)

---

## Health & Monitoring Routes

### `/health`

**Purpose:** Comprehensive system health monitoring.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/health` | GET | Overall system health check with memory and database status | Provides comprehensive health status for monitoring and alerting. The endpoint:<br>• Checks memory usage (RSS, VMS, percentage)<br>• Verifies database connection and checkpointer functionality<br>• Tests AsyncPostgresSaver with basic read operation<br>• Returns healthy/degraded status based on checks<br>• Includes uptime tracking<br>• Runs garbage collection and reports objects collected<br>• Returns 503 status if database unhealthy<br>• Used by monitoring systems and health checks<br>• Provides version information<br>• Includes timestamp for time-series monitoring |

**Key Features:**
- Multi-component health checking
- Memory usage reporting
- Database connectivity verification
- Automatic garbage collection
- HTTP status code-based alerting

---

### `/health/database`

**Purpose:** Detailed database connection health monitoring.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/health/database` | GET | Detailed database health with latency metrics | Provides in-depth database health information with performance metrics. The endpoint:<br>• Tests database read operations with timing<br>• Reports read latency in milliseconds<br>• Identifies checkpointer type (AsyncPostgresSaver vs fallback)<br>• Indicates if using memory fallback mode<br>• Returns 503 status on database errors<br>• Used for database-specific troubleshooting<br>• Helps identify slow database queries<br>• Monitors connection pool health<br>• Supports performance optimization<br>• Provides diagnostic information for debugging |

**Key Features:**
- Read latency measurement
- Checkpointer type identification
- Fallback mode detection
- Performance metrics collection

---

### `/health/memory`

**Purpose:** Memory usage monitoring with cache management.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/health/memory` | GET | Memory health with cache statistics and scaling guidance | Provides detailed memory usage analysis and capacity planning data. The endpoint:<br>• Reports current memory usage (RSS) in MB<br>• Compares against configured threshold (GC_MEMORY_THRESHOLD)<br>• Cleans up expired bulk cache entries automatically<br>• Reports active cache entries and cleanup count<br>• Calculates memory usage per thread for scaling<br>• Estimates maximum threads before threshold<br>• Returns healthy/warning/high_memory status<br>• Provides cache timeout configuration info<br>• Used for capacity planning and scaling decisions<br>• Helps identify memory leaks or excessive caching<br>• Supports autoscaling configuration |

**Key Features:**
- Automatic cache cleanup on check
- Memory threshold monitoring
- Per-thread memory estimation
- Scaling capacity calculation
- Status-based alerting

---

### `/health/rate-limits`

**Purpose:** Rate limiting status and configuration monitoring.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/health/rate-limits` | GET | Rate limiting statistics and configuration | Reports current rate limiting state and configuration. The endpoint:<br>• Shows total number of tracked client IPs<br>• Reports active clients with recent requests<br>• Displays rate limit window duration<br>• Shows maximum requests per window<br>• Reports burst capacity<br>• Used for load monitoring and capacity planning<br>• Helps identify potential DDoS or abuse<br>• Provides data for rate limit tuning<br>• Supports security monitoring<br>• Always returns healthy status (informational only) |

**Key Features:**
- Client tracking statistics
- Configuration visibility
- Active vs total client reporting
- Load pattern analysis support

---

### `/health/prepared-statements`

**Purpose:** Database prepared statement monitoring and health.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/health/prepared-statements` | GET | Check prepared statement count and database connection | Monitors prepared statement accumulation and database health. The endpoint:<br>• Counts prepared statements in PostgreSQL (\_pg3\_\*, \_pg\_\*)<br>• Lists prepared statement names<br>• Reports checkpointer status and type<br>• Shows connection configuration (prepared statement settings)<br>• Helps identify prepared statement leaks<br>• Used for diagnosing "prepared statement does not exist" errors<br>• Returns degraded status on database errors<br>• Provides diagnostic data for connection tuning<br>• Supports troubleshooting connection pool issues<br>• Shows effect of prepare_threshold settings |

**Key Features:**
- Prepared statement counting and listing
- Checkpointer health verification
- Connection configuration reporting
- Leak detection support

---

## Debug & Administration Routes

### `/debug/chat/{thread_id}/checkpoints`

**Purpose:** Raw checkpoint data inspection for debugging.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/debug/chat/{thread_id}/checkpoints` | GET | Inspect raw checkpoint data structure and contents | Provides detailed view of LangGraph checkpoint internals for debugging. The endpoint:<br>• Retrieves all checkpoints for a thread<br>• Shows checkpoint structure and metadata<br>• Displays metadata.writes for each checkpoint<br>• Previews message content and types<br>• Shows channel_values structure<br>• Reports total checkpoint count<br>• Used for debugging checkpoint processing issues<br>• Helps understand state progression<br>• Supports troubleshooting metadata extraction<br>• Shows raw LangGraph state<br>• Requires authentication (user ownership check) |

**Key Features:**
- Complete checkpoint structure visualization
- Message content preview
- Metadata inspection
- Channel values display

---

### `/debug/pool-status`

**Purpose:** Database connection pool health monitoring.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/debug/pool-status` | GET | Check database connection pool status and connectivity | Tests database connection functionality with timing metrics. The endpoint:<br>• Verifies global checkpointer existence<br>• Identifies checkpointer type<br>• Tests AsyncPostgresSaver basic operations<br>• Measures test operation latency<br>• Reports operational/error status<br>• Shows connection test results<br>• Used for debugging connection issues<br>• Helps identify pool exhaustion<br>• Supports performance troubleshooting<br>• Returns 500 status on critical errors<br>• Includes timestamp for monitoring |

**Key Features:**
- Connection functionality testing
- Operation latency measurement
- Checkpointer type reporting
- Error state detection

---

### `/debug/run-id/{run_id}`

**Purpose:** Run ID validation and ownership verification.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/debug/run-id/{run_id}` | GET | Validate run_id format, existence, and ownership | Comprehensive run_id diagnostic tool for debugging. The endpoint:<br>• Validates UUID format of run_id<br>• Checks if run_id exists in database<br>• Verifies user ownership of run_id<br>• Returns email, thread_id, prompt, timestamp if found<br>• Reports if run_id exists but belongs to different user<br>• Shows run_id type and length<br>• Provides UUID parsing diagnostics<br>• Used for debugging feedback/sentiment issues<br>• Helps identify UUID format problems<br>• Supports troubleshooting ownership errors<br>• Returns detailed diagnostic data |

**Key Features:**
- UUID format validation with detailed errors
- Database existence check
- Ownership verification
- Complete metadata retrieval
- Security enforcement (user-scoped)

---

### `/admin/clear-cache`

**Purpose:** Manual cache clearing for administration.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/admin/clear-cache` | POST | Clear bulk loading cache manually | Administrative endpoint for cache management. The endpoint:<br>• Clears all bulk loading cache entries<br>• Reports number of entries cleared<br>• Checks memory status after clearing<br>• Returns current memory usage in MB<br>• Indicates memory status (normal/high)<br>• Records who cleared cache and when<br>• Used for troubleshooting cache-related issues<br>• Helps free memory immediately<br>• Requires authentication (any authenticated user)<br>• Provides immediate feedback on memory impact<br>• Includes timestamp for audit trail |

**Key Features:**
- Complete cache purge
- Memory usage reporting
- Audit trail (user and timestamp)
- Memory status indication

---

### `/admin/clear-prepared-statements`

**Purpose:** Prepared statement cleanup for memory management.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/admin/clear-prepared-statements` | POST | Clear database prepared statements to free memory | Administrative tool for managing prepared statement accumulation. The endpoint:<br>• Clears accumulated prepared statements<br>• Helps resolve "prepared statement does not exist" errors<br>• Frees database memory<br>• Requires PostgreSQL checkpointer<br>• Returns success/error status<br>• Includes timestamp for tracking<br>• Used for troubleshooting prepared statement issues<br>• Supports connection pool health maintenance<br>• Requires authentication<br>• Returns error if PostgreSQL unavailable |

**Key Features:**
- Prepared statement cleanup
- Memory recovery
- Error state resolution
- PostgreSQL-specific operation

---

### `/debug/set-env`

**Purpose:** Dynamic environment variable modification for testing.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/debug/set-env` | POST | Dynamically set environment variables for debug control | Development tool for runtime configuration changes. The endpoint:<br>• Sets environment variables in running server process<br>• Accepts dictionary of key-value pairs<br>• Updates os.environ for immediate effect<br>• Used for enabling/disabling debug flags without restart<br>• Supports testing different configurations<br>• Returns list of variables changed<br>• Includes timestamp for tracking<br>• Requires authentication<br>• Changes persist until server restart<br>• Used for debugging and testing only |

**Key Features:**
- Runtime environment variable modification
- No server restart required
- Multiple variable updates in single call
- Audit trail with timestamp

---

### `/debug/reset-env`

**Purpose:** Restore environment variables to original values.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/debug/reset-env` | POST | Reset environment variables to original .env values | Reverses debug environment changes to restore original configuration. The endpoint:<br>• Reads original values from .env file<br>• Resets specified variables to .env values<br>• Defaults to "0" if variable not in .env<br>• Used to undo debug configuration changes<br>• Returns dictionary of reset variables and values<br>• Includes timestamp for tracking<br>• Requires authentication<br>• Ensures clean state after testing<br>• Supports safe experimentation with configs<br>• Provides rollback capability |

**Key Features:**
- .env file value restoration
- Safe rollback mechanism
- Default value handling
- Multiple variable reset

---

## Utility Routes

### `/placeholder/{width}/{height}`

**Purpose:** Dynamic placeholder image generation for UI development.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/placeholder/{width}/{height}` | GET | Generate SVG placeholder images with specified dimensions | Utility endpoint for generating placeholder images during development. The endpoint:<br>• Generates SVG images with specified width and height<br>• Limits dimensions to 1-2000 pixels for safety<br>• Shows dimensions as text in center of image<br>• Returns image/svg+xml content type<br>• Includes cache headers for performance (1 hour)<br>• Supports CORS for cross-origin requests<br>• Used by frontend during development<br>• Helps with layout testing without real images<br>• Lightweight (SVG) for fast loading<br>• Graceful error handling with error SVG |

**Key Features:**
- Dynamic dimension specification
- Safety limits on dimensions
- Cache headers for performance
- CORS support
- SVG format (lightweight)

---

### `/initial-followup-prompts`

**Purpose:** AI-generated starter suggestions for new conversations.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/initial-followup-prompts` | GET | Generate AI-powered initial follow-up prompt suggestions | Provides starter suggestions for new chat sessions. The endpoint:<br>• Uses AI to generate contextual starter prompts<br>• Returns list of suggested questions users can ask<br>• Helps users understand system capabilities<br>• Reduces initial friction in starting conversations<br>• Based on CZSU data availability and common queries<br>• Returns empty array on errors (graceful degradation)<br>• Used by frontend to show suggestion chips<br>• Improves user onboarding experience<br>• Requires authentication<br>• Error logging for debugging failed generation |

**Key Features:**
- AI-powered suggestion generation
- Graceful error handling
- Empty array fallback
- User onboarding support

---

## Root & Documentation Routes

### `/`

**Purpose:** API information and navigation hub.

| Endpoint | Method | Purpose | Description |
|----------|--------|---------|-------------|
| `/` | GET | Welcome page with comprehensive API information | Provides complete overview of API capabilities and navigation. The endpoint:<br>• Returns API name, version, and description<br>• Shows operational status and current timestamp<br>• Lists all major endpoint categories with descriptions<br>• Provides links to documentation (Swagger, ReDoc, OpenAPI)<br>• Explains core functionality and features<br>• Includes getting started guide<br>• Lists support resources<br>• Describes multi-agent architecture<br>• Shows rate limiting and monitoring features<br>• Provides complete endpoint reference<br>• Used as landing page for API exploration<br>• Helps developers understand API structure |

**Key Features:**
- Complete API overview
- Categorized endpoint listing
- Documentation links
- Getting started guide
- Feature list
- Support information

---

## Route Summary Statistics

| Category | Number of Routes | Number of Endpoints |
|----------|-----------------|---------------------|
| Core Analysis | 2 | 2 |
| Chat & Messaging | 6 | 6 |
| Data Catalog | 3 | 3 |
| Feedback & Sentiment | 2 | 2 |
| Health & Monitoring | 5 | 5 |
| Debug & Administration | 5 | 5 |
| Utility | 2 | 2 |
| Root & Documentation | 1 | 1 |
| **TOTAL** | **26** | **26** |

---

## Authentication & Security

All endpoints except `/` (root) require JWT authentication via the `get_current_user` dependency. Key security features:

- **JWT Token Validation**: All requests must include valid JWT token
- **User Ownership Verification**: Users can only access/modify their own data
- **Thread Isolation**: Multi-user safety ensures thread data segregation
- **Run ID Validation**: UUID format validation on all run_id parameters
- **Rate Limiting**: Per-IP rate limiting to prevent abuse
- **SQL Injection Protection**: Parameterized queries and input sanitization
- **CORS Configuration**: Controlled cross-origin access

---

## Performance Considerations

### Caching Strategy
- **Bulk Messages**: 60-second cache timeout with per-user locking
- **Health Checks**: No caching (real-time status)
- **Catalog/Data**: No caching (static data, fast SQLite reads)

### Concurrency Limits
- **Analysis**: `MAX_CONCURRENT_ANALYSES` (default 3) concurrent requests
- **Bulk Loading**: 3 concurrent thread processing operations
- **Rate Limiting**: Configurable per-IP request limits

### Timeout Settings
- **Analysis**: 240 seconds (4 minutes) per query
- **Database Operations**: Varies by operation type
- **Cache Cleanup**: Automatic on memory health checks

---

## Error Handling

All endpoints implement comprehensive error handling:

1. **HTTPException**: For client errors (400, 401, 404)
2. **Database Errors**: Graceful degradation with fallback modes
3. **Prepared Statement Errors**: Automatic retry with fresh connections
4. **Timeout Errors**: 408 status with clear timeout messages
5. **Cancellation Errors**: 499 status for user-cancelled operations
6. **Generic Errors**: 500 status with traceback logging

---

## Environment Variables

Key configuration via environment variables:

- `MAX_CONCURRENT_ANALYSES`: Concurrent analysis limit
- `MAX_CONCURRENT_BULK_THREADS`: Bulk loading concurrency
- `BULK_CACHE_TIMEOUT`: Cache timeout in seconds
- `GC_MEMORY_THRESHOLD`: Memory threshold for garbage collection (MB)
- `RATE_LIMIT_WINDOW`: Rate limit time window
- `RATE_LIMIT_REQUESTS`: Max requests per window
- `RATE_LIMIT_BURST`: Burst capacity for rate limiting
- `INMEMORY_FALLBACK_ENABLED`: Enable InMemorySaver fallback

---

## Database Tables Referenced

### Primary Tables
- `users_threads_runs`: Thread and run tracking with user association
- `checkpoints`: LangGraph checkpoint storage (via AsyncPostgresSaver)
- `selection_descriptions`: CZSU dataset catalog and descriptions

### SQLite Databases
- `data/czsu_data.db`: Main statistical data
- `metadata/llm_selection_descriptions/selection_descriptions.db`: Dataset descriptions

---

## API Versioning

**Current Version**: 1.0.0

Version information available at:
- Root endpoint: `/`
- Health check: `/health`

---

## Related Documentation

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Spec**: `/openapi.json`

---

*This documentation was generated through comprehensive code analysis on November 16, 2025. For the most up-to-date information, please refer to the Swagger documentation at `/docs`.*
