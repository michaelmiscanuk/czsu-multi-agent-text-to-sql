# Implementation Core Topics

## Overview - Implementation Architecture

### High-Level System Architecture
- **Multi-tier Architecture**: Frontend (Vercel), Backend (Railway), AI Layer (Azure/Cohere), Data Storage (Supabase/Turso/Chroma Cloud)
  - **Purpose:** Distributes responsibilities across specialized layers so the UI, compute, and storage can scale and evolve independently.
- **Deployment Strategy**: N-tier client-server architecture with cloud-hosted components
  - **Purpose:** Provides clear separation of concerns between presentation, application, and data layers while maintaining loose coupling for scalability and independent deployment.
- **Technology Stack**: React/Next.js frontend, FastAPI backend, LangGraph agent workflow
  - **Purpose:** Combines a modern SPA framework, high-performance Python API, and agent orchestration engine to deliver responsive conversational analytics.
- **Service Integration**: External APIs (Google OAuth, Azure OpenAI, Azure Translator, Cohere, LlamaParse)
  - **Purpose:** Leverages best-in-class hosted capabilities for authentication, language intelligence, reranking, and document parsing without rebuilding them in-house.
- **Data Flow**: User query → Backend API → LangGraph agent → Multiple AI services → Response synthesis
  - **Purpose:** Provides a deterministic processing pipeline that traces how questions are transformed into validated answers and ensures observability at each hop.

### Core Components Overview
- **Frontend Layer**: Next.js 15 with React 19, deployed on Vercel
  - **Purpose:** Delivers a performant, server-rendered user experience with automatic scaling and global CDN distribution.
- **Backend Layer**: FastAPI application with LangGraph, deployed on Railway
  - **Purpose:** Hosts the core business logic and agent execution in a managed environment that supports async workloads and observability.
- **AI Components**: LangGraph agent workflow, LangSmith tracing, Azure OpenAI models, translation services
  - **Purpose:** Coordinates multi-step reasoning, traces execution for debugging, and provides multilingual LLM capabilities for richer responses.
- **Data Storage**: PostgreSQL (Supabase), SQLite (Turso), Vector DB (Chroma Cloud)
  - **Purpose:** Combines relational persistence, analytical datasets, and semantic retrieval to satisfy conversational SQL and knowledge search needs.
- **External Services**: Google OAuth 2.0, CZSU API, LlamaParse, Cohere reranking
  - **Purpose:** Adds secure user login, authoritative statistical data, structured PDF ingestion, and relevance optimization to the overall solution.

### Key Architectural Decisions
- **Agentic Workflow**: LangGraph for orchestrated multi-step reasoning
  - **Purpose:** Gives the system a controllable execution graph that makes complex querying and validation reproducible and inspectable.
- **Dual Retrieval Sources**: Database metadata + PDF documentation for comprehensive answers
  - **Purpose:** Ensures answers can reference both structured tables and contextual narratives, improving completeness and explainability.
- **Hybrid Search**: Semantic + BM25 retrieval with reranking
  - **Purpose:** Balances lexical precision with semantic understanding so that both keywords and intent influence the final ranking.
- **Stateful Conversations**: PostgreSQL checkpointing for conversation persistence
  - **Purpose:** Allows users to resume or audit long-running analyses while giving operators insight into historical interactions.
- **Token Optimization**: Message summarization at key workflow points
  - **Purpose:** Keeps prompts within LLM limits, reduces latency, and lowers inference cost without losing conversational context.
- **Bilingual Support**: Czech and English with automatic language detection
  - **Purpose:** Makes the tool accessible to native users and the broader community while preserving data integrity across languages.

---

## Backend - Core Functionality

### 1. LangGraph Agent Workflow

#### Workflow Architecture
- **Graph-based Orchestration**: StateGraph with controlled execution flow
  - **Purpose:** Provides deterministic node transitions and allows us to visualize, debug, and tune each stage of the pipeline.
- **Multi-phase Processing**: Query rewriting → Parallel retrieval → SQL generation → Answer synthesis
  - **Purpose:** Decomposes the problem into manageable phases so each step can specialize in one responsibility and be monitored separately.
- **Conditional Routing**: Smart decision points based on data availability
  - **Purpose:** Lets the workflow adapt dynamically when data sources are missing or when early results already satisfy the query.
- **Iteration Control**: Maximum iteration limits with reflection loops
  - **Purpose:** Prevents runaway loops, keeps costs predictable, and still gives the agent a chance to refine its reasoning when necessary.

#### Agent Phases

##### Phase 1: Query Preprocessing
- **Prompt Rewriting**: Conversational context resolution, pronoun replacement, topic change detection
  - **Purpose:** Normalizes user input into a standalone query that the retrieval and SQL components can understand consistently.
- **Memory Summarization**: Token-efficient context preservation (summary + last message pattern)
  - **Purpose:** Keeps the agent informed about conversation history while respecting token budgets and latency targets.
- **Optimization**: Search query enhancement with synonyms
  - **Purpose:** Improves recall in downstream retrieval systems by broadening the vocabulary used to describe the topic.

##### Phase 2: Parallel Retrieval (Dual Branches)
- **Branch A - Database Selections**:
  - Hybrid search (semantic + BM25) on dataset descriptions
    - **Purpose:** Captures both intent and exact terminology so the agent can locate the most relevant statistical selections.
  - Cohere reranking for relevance
    - **Purpose:** Reorders candidates using a stronger semantic model to push the most contextually relevant datasets to the top.
  - Top-k filtering with similarity thresholds
    - **Purpose:** Limits the working set to the most promising results, reducing noise and token usage in later phases.
- **Branch B - PDF Documentation**:
  - Translation to English for vector search
    - **Purpose:** Harmonizes multilingual content, making the embeddings space consistent regardless of the query language.
  - Hybrid search on parsed PDF content
    - **Purpose:** Surfaces methodological explanations and definitions that may not exist in structured tables.
  - Cohere reranking with relevance thresholds
    - **Purpose:** Ensures only the strongest supporting passages are kept for answer synthesis and justification.

##### Phase 3: SQL Generation & Execution
- **Schema Loading**: Dynamic schema retrieval from metadata database
  - **Purpose:** Supplies the agent with up-to-date table structures and descriptions to avoid hallucinated column names.
- **Agentic Tool Calling**: LLM autonomously decides when to execute SQL queries
  - **Purpose:** Mimics an analyst’s workflow by letting the model gather evidence iteratively before finalizing an answer.
- **MCP Integration**: Model Context Protocol with remote/local fallback
  - **Purpose:** Abstracts database access behind a standardized interface so we can swap execution environments without code churn.
- **Iterative Data Gathering**: Multiple query execution until sufficient information collected
  - **Purpose:** Allows the agent to refine its approach, compare alternative queries, and stop once it has enough evidence.

##### Phase 4: Reflection & Self-Correction
- **Result Analysis**: LLM evaluates query results for completeness
  - **Purpose:** Guards against premature answers by having the agent verify whether the retrieved data actually addresses the question.
- **Decision Making**: "improve" (better query needed) or "answer" (sufficient data)
  - **Purpose:** Provides a simple control mechanism for looping that keeps conversations responsive.
- **Iteration Limits**: Controlled loops with MAX_ITERATIONS
  - **Purpose:** Maintains predictability for cost and latency while still allowing limited self-improvement cycles.

##### Phase 5: Answer Synthesis
- **Multi-source Integration**: Combines SQL results + PDF chunks
  - **Purpose:** Weaves structured facts with contextual explanations to build answers that are both accurate and informative.
- **Bilingual Response**: Matches query language automatically
  - **Purpose:** Respects the user’s language preference and preserves cultural nuances in the delivered insights.
- **Markdown Formatting**: Structured answer presentation
  - **Purpose:** Produces readable output with headings, tables, and emphasis that users can easily copy into reports.
- **Follow-up Generation**: Contextual suggestions for continued conversation
  - **Purpose:** Encourages exploratory analysis by hinting at natural next questions derived from the current result.

#### State Management
- **DataAnalysisState TypedDict**: 15 fields tracking workflow state
  - **Purpose:** Gives us a typed contract for what information each node can rely on, improving maintainability.
- **Key State Fields**: prompt, rewritten_prompt, messages, iteration, queries_and_results, top_selection_codes, top_chunks, final_answer
  - **Purpose:** Stores the minimal yet sufficient context needed to resume or inspect executions at any checkpoint.
- **Reducers**: Limited queries reducer for memory efficiency
  - **Purpose:** Prevents unbounded growth of query history while preserving the most relevant attempts.
- **Checkpointing**: PostgreSQL persistence for conversation resumption
  - **Purpose:** Enables resilience against crashes and lets users or operators replay past runs for auditing.

#### Key Design Principles
- **Controlled Iteration**: Cycle prevention with iteration counters
  - **Purpose:** Keeps the agent from exceeding time or budget constraints during complex tasks.
- **Resource Cleanup**: Explicit ChromaDB client closure and garbage collection
  - **Purpose:** Avoids memory leaks and keeps long-running services stable under load.
- **Token Management**: Automatic summarization at 4 workflow points
  - **Purpose:** Maintains a lean prompt footprint so that we can use higher-quality models without hitting token ceilings.
- **Modular Architecture**: Separated routing logic in dedicated modules
  - **Purpose:** Makes it easier to test, reason about, and extend individual decision points without touching the whole graph.

### 2. FastAPI Application Structure

#### Configuration & Lifecycle
- **Environment Bootstrapping**: Absolute imports, dotenv loading, event loop policy
  - **Purpose:** Standardizes runtime setup across platforms and ensures environment variables are available early.
- **Runtime Settings**: Checkpointer globals, concurrency limits, memory monitoring
  - **Purpose:** Centralizes operational knobs so we can tune performance and stability without code changes.
- **Lifespan Management**: Startup sequence (checkpointer init, memory baseline), shutdown cleanup
  - **Purpose:** Guarantees resources are ready before traffic arrives and are released cleanly during redeployments.
- **Memory Controls**: GC thresholds, profiling when enabled
  - **Purpose:** Keeps the API responsive by proactively managing memory usage and collecting diagnostics when needed.

#### Middleware Stack
- **CORS**: Permissive policy for cross-origin requests
  - **Purpose:** Allows the Vercel-hosted frontend to call the Railway backend securely from the browser.
- **GZip Compression**: Response compression for JSON payloads over 1000 bytes
  - **Purpose:** Reduces bandwidth usage and speeds up perception of large analysis responses.
- **Throttling**: IP-based semaphore limits (max 8 concurrent requests per IP)
  - **Purpose:** Protects backend resources from noisy clients while still permitting legitimate parallel usage.
- **Rate Limiting**: Request tracking with retry logic and wait mechanisms
  - **Purpose:** Smooths traffic spikes and signals to clients when to back off instead of failing hard.
- **Memory Monitoring**: Request counting and logging for heavy endpoints
  - **Purpose:** Gives early warnings of memory pressure so operators can investigate before incidents occur.

#### Route Architecture

##### Core Routes
- **Analysis** (`/analyze`):
  - Main LangGraph execution endpoint
    - **Purpose:** Entrypoint where user prompts trigger the full agent workflow and final answers are returned.
  - Concurrency control via analysis_semaphore
    - **Purpose:** Prevents the agent from overloading shared compute when multiple users analyze simultaneously.
  - Execution registration for cancellation support
    - **Purpose:** Tracks in-flight runs so stop requests can locate and terminate the correct execution.
  - InMemorySaver fallback on database failures
    - **Purpose:** Keeps the service available even if PostgreSQL is temporarily unreachable.
  
- **Chat Management** (`/chat-threads`, `/chat/{thread_id}`):
  - Paginated thread listing
    - **Purpose:** Lets the frontend load conversation history incrementally for better performance.
  - Thread deletion with cleanup
    - **Purpose:** Allows users to manage their workspace and free storage from old analyses.
  - Message retrieval per thread
    - **Purpose:** Provides focused access to conversation logs for rendering in the chat UI.
  - Bulk message loading optimization
    - **Purpose:** Reduces API chatter by delivering all messages in one call when the cache needs a refresh.

- **Catalog & Data** (`/catalog`, `/data-tables`, `/data-table`):
  - Dataset browsing with pagination
    - **Purpose:** Helps users discover available statistical selections without overloading the API.
  - Table listing with descriptions
    - **Purpose:** Supplies schema metadata so users understand what each table contains before querying.
  - Dynamic table data retrieval with columns/rows
    - **Purpose:** Serves real data snapshots to support manual exploration and charting.

- **Feedback System** (`/feedback`, `/sentiment`):
  - LangSmith feedback submission
    - **Purpose:** Captures structured quality signals that feed back into evaluation workflows.
  - Database sentiment tracking
    - **Purpose:** Persists quick thumbs-up/down reactions for trend monitoring.
  - Run ID-based feedback association
    - **Purpose:** Ties human feedback directly to specific executions for traceability.

##### Support Routes
- **Health Checks**: Database, memory, rate limits, prepared statements
  - **Purpose:** Enables observability dashboards and uptime monitors to confirm subsystem health.
- **Debug Endpoints**: Checkpoint inspection, pool status, run ID debugging
  - **Purpose:** Gives engineers deep introspection tools without touching production databases directly.
- **Stop Execution**: Cancellation support with thread_id + run_id
  - **Purpose:** Allows users to halt expensive analyses, conserving resources and improving UX.

#### Authentication
- **Google OAuth Integration**: JWT token verification with google-auth library
  - **Purpose:** Ensures only authenticated users can access protected endpoints using Google-issued credentials.
- **JWK Retrieval**: Signature verification via Google's public keys
  - **Purpose:** Validates tokens securely and automatically rotates keys when Google updates them.
- **Token Extraction**: Bearer token from Authorization header
  - **Purpose:** Provides a simple, standards-based way for the frontend to pass credentials.
- **User Info**: Email extraction for user tracking
  - **Purpose:** Identifies who initiated requests for auditing, personalization, and rate-limiting.

#### Error Handling & Recovery
- **Global Exception Handlers**: RequestValidationError, HTTPException, ValueError, generic Exception
  - **Purpose:** Delivers consistent JSON error responses and prevents unhandled crashes.
- **Comprehensive Logging**: log_comprehensive_error utility for debugging
  - **Purpose:** Captures structured diagnostics that speed up troubleshooting.
- **Graceful Degradation**: InMemorySaver fallback when PostgreSQL unavailable
  - **Purpose:** Keeps core functionality online even when persistent storage is degraded.
- **Prepared Statement Recovery**: Detection and cleanup of database errors
  - **Purpose:** Automatically clears problematic prepared statements to avoid recurring query failures.

### 3. Checkpointer System

#### PostgreSQL Checkpointer
- **Async Implementation**: AsyncPostgresSaver with connection pooling
  - **Purpose:** Provides scalable persistence that matches the async nature of the agent workflow.
- **Connection Management**: Pool lifecycle with min/max sizes, timeouts, keepalives
  - **Purpose:** Maintains healthy database connections and avoids exhausting server limits.
- **Retry Logic**: Decorator-based retry for transient failures
  - **Purpose:** Smooths over temporary outages without surfacing errors to the user.
- **State Persistence**: Conversation state storage and retrieval
  - **Purpose:** Allows analysis sessions to resume after restarts or cancellations.

#### Factory Pattern
- **initialize_checkpointer()**: Bootstrap with retry logic
  - **Purpose:** Ensures the saver is ready before requests hit the system.
- **get_global_checkpointer()**: Singleton pattern for shared instance
  - **Purpose:** Avoids re-initializing pools and keeps resource usage predictable.
- **cleanup_checkpointer()**: Pool closure and cleanup
  - **Purpose:** Releases database resources gracefully during shutdowns.
- **Fallback Strategy**: InMemorySaver when PostgreSQL unavailable
  - **Purpose:** Preserves functionality in development or outage scenarios without data loss guarantees.

#### User & Thread Management
- **Thread Operations**: Thread creation, retrieval, deletion
  - **Purpose:** Manages the lifecycle of conversations in persistent storage.
- **Sentiment Tracking**: Run-based sentiment storage and retrieval
  - **Purpose:** Stores user feedback alongside execution metadata for later analysis.
- **Message Metadata**: Previews, timestamps, run counts
  - **Purpose:** Supports UI summaries and audit reports without fetching full message bodies.

#### Health & Monitoring
- **Pool Status**: Connection pool metrics
  - **Purpose:** Alerts operators to saturation or idle connections that might indicate configuration issues.
- **Prepared Statement Detection**: Error pattern recognition and cleanup
  - **Purpose:** Prevents corrupt prepared statements from cascading into widespread failures.
- **Rate Limiting Integration**: Request tracking and throttling
  - **Purpose:** Shares rate-limit state with throttling middleware to keep policies consistent.

### 4. Data Preparation Pipeline

#### CZSU Data Ingestion
- **API Integration**: Fetch datasets and selections from CZSU API
  - **Purpose:** Pulls the authoritative statistical content straight from the national source.
- **JSON-stat Processing**: Convert statistical data to DataFrames
  - **Purpose:** Normalizes returned payloads into tabular structures suitable for analysis.
- **CSV Export**: Save to structured CSV files
  - **Purpose:** Creates portable snapshots that feed downstream loading jobs.
- **SQLite Database**: Import CSVs to local SQLite database
  - **Purpose:** Provides a lightweight relational store for testing and preprocessing.
- **Cloud Upload**: Deploy to Turso (SQLite cloud) via CLI
  - **Purpose:** Publishes the prepared data to a managed environment accessible to the MCP server and backend.

#### Metadata Extraction
- **Schema Retrieval**: Get metadata for all selections via CZSU API
  - **Purpose:** Collects schema context required for accurate SQL generation.
- **LLM Enhancement**: Generate extended descriptions using GPT-4o
  - **Purpose:** Produces human-friendly explanations that improve UX and retrieval quality.
- **Selection Descriptions**: Create human-readable dataset descriptions
  - **Purpose:** Equips the frontend and agent with narrative summaries for each selection.
- **Metadata Database**: Store in separate SQLite DB for schema lookups
  - **Purpose:** Keeps lookup operations fast and decoupled from the main statistical data store.

#### PDF Documentation Processing
- **Parsing Strategies**: LlamaParse (premium LLM-based), Azure Document Intelligence
  - **Purpose:** Extracts text accurately from complex methodology PDFs.
- **Text Extraction**: Extract text from CZSU methodology PDFs
  - **Purpose:** Converts reference material into searchable content.
- **Chunking**: Split documents into semantic chunks
  - **Purpose:** Creates retrieval-friendly segments that align with user queries.
- **Vector Embeddings**: Generate embeddings using Azure OpenAI
  - **Purpose:** Encodes chunks into vector space for semantic similarity search.
- **ChromaDB Storage**: Store in local or cloud vector database
  - **Purpose:** Provides a scalable retrieval layer that serves both backend and agent needs.

#### Vector Database Setup
- **Local ChromaDB**: Development environment with persistent storage
  - **Purpose:** Allows offline experimentation and debugging without cloud dependencies.
- **Chroma Cloud**: Production deployment with remote access
  - **Purpose:** Delivers managed reliability and global availability for embedding search.
- **Dual Collections**: Separate collections for dataset descriptions and PDF chunks
  - **Purpose:** Keeps retrieval domains isolated so relevance scoring stays accurate.
- **Hybrid Search Support**: BM25 + semantic search capabilities
  - **Purpose:** Enables combined lexical-semantic queries for higher quality results.

### 5. MCP (Model Context Protocol) Integration

#### Architecture
- **Optional Component**: Remote MCP server or local SQLite fallback
  - **Purpose:** Gives deployment flexibility depending on connectivity and security requirements.
- **FastMCP Implementation**: Python-based MCP server for SQLite queries
  - **Purpose:** Provides a lightweight, standards-compliant bridge between LangGraph and the database.
- **LangChain Adapter**: Integration via langchain-mcp-adapters
  - **Purpose:** Simplifies tool invocation so the agent can call MCP endpoints like any other tool.
- **Tool Calling**: `sqlite_query` tool for agentic SQL execution
  - **Purpose:** Allows LLM-driven querying while enforcing a safe execution interface.

#### Deployment Modes
- **Local Fallback**: Direct SQLite access (default, `USE_LOCAL_SQLITE_FALLBACK=1`)
  - **Purpose:** Keeps local development simple and resilient when remote services are unavailable.
- **Remote MCP**: FastMCP Cloud deployment for centralized access
  - **Purpose:** Supports shared environments where multiple clients need consistent data access.
- **Dual Support**: Automatic fallback logic
  - **Purpose:** Ensures continuity by switching seamlessly between modes without downtime.

#### Benefits
- **Isolation**: Separate database access from main application
  - **Purpose:** Limits blast radius of database issues and improves security posture.
- **Reusability**: MCP server can serve multiple clients
  - **Purpose:** Encourages sharing the same query capabilities across different tools and services.
- **Standardization**: Protocol-based communication
  - **Purpose:** Avoids bespoke integrations and makes it easier to adopt future MCP-compatible tools.

---

## Frontend - Core Functionality

### 1. Next.js Application Architecture

#### Configuration
- **Framework**: Next.js 15 with React 19
  - **Purpose:** Provides modern SSR/SSG features and the latest React capabilities for interactivity.
- **TypeScript**: Strict type checking with path aliases (@/*)
  - **Purpose:** Improves developer productivity and reduces runtime errors through static analysis.
- **Styling**: TailwindCSS 4 with custom utility classes
  - **Purpose:** Enables rapid UI iteration with consistent design tokens.
- **Authentication**: NextAuth with Google OAuth provider
  - **Purpose:** Simplifies auth flows while leaning on Google for identity management.
- **Deployment**: Vercel with API proxying to Railway backend
  - **Purpose:** Co-locates hosting and CDN while securely forwarding API requests to the backend.

#### App Router Structure
- **Dual Layout Pattern**: Server component for metadata, client component for auth logic
  - **Purpose:** Combines SEO-friendly rendering with client-side session awareness.
- **Route Protection**: Public vs protected routes with automatic redirects
  - **Purpose:** Keeps sensitive pages guarded without manual checks in every component.
- **Session Management**: NextAuth SessionProvider wrapper
  - **Purpose:** Supplies session context to the entire component tree for consistent behavior.

### 2. Pages & Features

#### Chat Interface (`/chat`)
- **Thread Management**: Create, rename, delete conversations
  - **Purpose:** Mirrors chat tool expectations and lets users organize analyses by topic.
- **Message Flow**: User input → backend analysis → response display
  - **Purpose:** Delivers real-time conversational analytics with minimal friction.
- **Optimistic UI**: Immediate loading states with backend sync
  - **Purpose:** Provides responsive feedback while longer analyses complete server-side.
- **Recovery Mechanism**: Fallback to PostgreSQL on errors
  - **Purpose:** Prevents data loss when the frontend request times out but the backend succeeded.
- **Cancellation**: Stop execution support with instant UI feedback
  - **Purpose:** Gives users control over long-running jobs and improves perceived performance.
- **Persistence**: localStorage for drafts, active thread, full cache
  - **Purpose:** Preserves user progress across reloads and network blips.

#### Catalog (`/catalog`)
- **Dataset Browser**: Paginated table of CZSU datasets
  - **Purpose:** Provides an overview of available data assets for discovery.
- **Search & Filter**: Client-side filtering with diacritics removal
  - **Purpose:** Makes it easy to find datasets even when names use Czech characters.
- **Navigation**: Click-through to data view
  - **Purpose:** Streamlines workflow from discovery to detailed inspection.

#### Data Explorer (`/data`)
- **Smart Search**: Auto-complete with prefix matching (* for codes only)
  - **Purpose:** Speeds up access to specific tables and supports power users who know codes.
- **Advanced Filtering**: Column-specific filters with numeric operators (>, <, >=, <=, !=, =)
  - **Purpose:** Enables granular analysis directly in the browser without writing SQL.
- **Sortable Columns**: Multi-state sorting (asc/desc/none)
  - **Purpose:** Lets users rank results quickly to spot trends or anomalies.
- **Cross-Navigation**: Links to catalog with pre-filled filters
  - **Purpose:** Keeps context when switching between high-level and detailed views.
- **State Persistence**: localStorage sync for all filters
  - **Purpose:** Retains user adjustments across sessions for a smoother experience.

#### Login (`/login`)
- **Google OAuth**: One-click sign-in
  - **Purpose:** Lowers entry barriers by using familiar credentials.
- **Auto-redirect**: Authenticated users sent to /chat
  - **Purpose:** Gets returning users back into productive workflows immediately.

### 3. State Management

#### ChatCacheContext
- **Centralized Cache**: Threads, messages, run IDs, sentiments
  - **Purpose:** Offers a single source of truth that multiple components can share without prop drilling.
- **localStorage Persistence**: 48-hour cache with automatic save/load
  - **Purpose:** Provides offline-friendly resilience and faster reloads.
- **Cross-tab Sync**: Storage event listener for multi-tab coordination
  - **Purpose:** Prevents conflicting state when the app is open in multiple browser tabs.
- **Pagination**: Server-side with automatic message loading
  - **Purpose:** Keeps memory usage manageable while still offering infinite scroll UX.
- **Bulk Loading**: Single API call for all threads' messages (N+1 problem solution)
  - **Purpose:** Minimizes network overhead during cache warm-up.

#### Key Features
- **Page Refresh Detection**: F5 forces API refresh, navigation uses cache
  - **Purpose:** Balances freshness with speed by detecting true reloads.
- **Cross-tab Loading Prevention**: Prevents concurrent requests across tabs
  - **Purpose:** Avoids duplicate analysis jobs that waste backend resources.
- **Invalidation**: Manual cache refresh controls
  - **Purpose:** Gives users a way to force sync when they suspect stale data.
- **Optimistic Updates**: UI reflects changes before backend confirmation
  - **Purpose:** Keeps the experience snappy even when the network is slow.

### 4. Components Architecture

#### Layout Components
- **Header**: Sticky navigation with active route highlighting
  - **Purpose:** Anchors navigation and keeps page context visible.
- **AuthButton**: Dual-mode (compact/full) with user avatar
  - **Purpose:** Provides quick auth actions and personal context in different layouts.
- **SessionProviderWrapper**: NextAuth context provider
  - **Purpose:** Ensures all client components can access session data without boilerplate.
- **AuthGuard**: Route protection based on auth status
  - **Purpose:** Centralizes access control logic for protected pages.

#### Chat Components
- **MessageArea**:
  - Markdown rendering with custom overrides
    - **Purpose:** Displays rich-formatted answers and tables clearly.
  - Copy functionality (rich text + plain text)
    - **Purpose:** Lets users reuse insights in other tools effortlessly.
  - Datasets badges with navigation
    - **Purpose:** Provides quick jumps from answers to underlying datasets.
  - SQL/PDF modals
    - **Purpose:** Exposes supporting queries and documents for transparency.
  - Follow-up prompts (clickable badges)
    - **Purpose:** Encourages guided exploration based on previous outputs.
  - Progress tracking with 8-minute estimates
    - **Purpose:** Sets user expectations for long-running analyses.
  
- **FeedbackComponent**:
  - Thumbs up/down sentiment tracking
    - **Purpose:** Captures lightweight quality signals from users.
  - Comment textarea dropdown
    - **Purpose:** Allows detailed feedback when users have more context to share.
  - Dual submission (LangSmith + database)
    - **Purpose:** Syncs quality data with both evaluation tooling and internal metrics.
  - localStorage persistence
    - **Purpose:** Remembers feedback state per run so users don’t lose inputs on refresh.
  - Visual confirmation
    - **Purpose:** Provides immediate acknowledgement that feedback was saved.

#### Data Components
- **DatasetsTable**: Hybrid pagination (backend when no filter, client when filtered)
  - **Purpose:** Keeps the interface responsive while handling large dataset catalogs.
- **DataTableView**: Auto-complete search, multi-column filtering, sorting
  - **Purpose:** Offers power-user data browsing without leaving the browser.

### 5. API Integration

#### Configuration
- **Base URL**: Environment-aware (`/api` in production via Vercel rewrites)
  - **Purpose:** Ensures API calls work seamlessly in both local and hosted environments.
- **Timeout**: 10-minute limit for long-running analyses
  - **Purpose:** Matches backend processing windows and avoids frozen requests.
- **Authentication**: Bearer token in Authorization header
  - **Purpose:** Secures API calls with minimal implementation overhead.

#### Fetch Utilities
- **apiFetch()**: Generic fetch with timeout, logging, error handling
  - **Purpose:** Centralizes network behavior and diagnostics for easier maintenance.
- **authApiFetch()**: Auto-refresh on 401 errors
  - **Purpose:** Keeps sessions alive transparently when tokens expire.
- **Token Refresh**: Automatic retry with new token from getSession()
  - **Purpose:** Reduces friction by avoiding forced logouts during long sessions.

#### Backend Routes
- **Chat**: Thread CRUD, message retrieval, bulk loading
  - **Purpose:** Powers the chat interface’s data needs through consistent endpoints.
- **Analysis**: `/analyze` for question answering, `/stop-execution` for cancellation
  - **Purpose:** Connects conversational UI controls to backend processing.
- **Feedback**: LangSmith feedback + sentiment tracking
  - **Purpose:** Routes user responses back to monitoring systems.
- **Data**: Catalog pagination, table listing, table data
  - **Purpose:** Feeds the catalog and data explorer with up-to-date information.

### 6. Authentication Flow

#### NextAuth Setup
- **Google Provider**: OAuth with offline access for token refresh
  - **Purpose:** Enables long-lived sessions without repeatedly prompting the user.
- **JWT Callback**: Stores access token, refresh token, id_token, expiry
  - **Purpose:** Keeps all necessary credentials in one place for client and server use.
- **Session Callback**: Exposes tokens and user info to client
  - **Purpose:** Supplies UI components with identity and token data for API calls.
- **Token Refresh**: Automatic refresh mechanism via Google OAuth API
  - **Purpose:** Maintains uninterrupted service during long analytical sessions.

#### Client-side Auth
- **useSession Hook**: Session state access
  - **Purpose:** Simplifies retrieving auth state in React components.
- **signIn/signOut**: Authentication actions
  - **Purpose:** Provides standard entry and exit points for user sessions.
- **Cache Cleanup**: clearCacheForUserChange() on logout
  - **Purpose:** Prevents data leakage between different authenticated users on the same device.

### 7. Advanced Features

#### Infinite Scroll
- **useInfiniteScroll Hook**: IntersectionObserver-based pagination
  - **Purpose:** Offers smooth, automatic loading for long lists of threads or messages.
- **Auto-loading**: Triggers when sentinel element enters viewport
  - **Purpose:** Removes manual pagination steps for users.
- **Duplicate Prevention**: isLoadingRef flag
  - **Purpose:** Stops concurrent fetches that could corrupt state.

#### Sentiment System
- **useSentiment Hook**: Per-run thumbs up/down tracking
  - **Purpose:** Encapsulates feedback logic so multiple components can record sentiment consistently.
- **Optimistic Updates**: Immediate UI reflection
  - **Purpose:** Makes feedback interactions feel instant regardless of network speed.
- **Database Sync**: Background POST to /sentiment
  - **Purpose:** Persists user reactions for later analysis without blocking the UI.

#### Markdown Rendering
- **markdown-to-jsx**: Custom component overrides
  - **Purpose:** Allows rich formatting while keeping styling consistent with the app.
- **Code Blocks**: Syntax highlighting support
  - **Purpose:** Displays SQL snippets clearly for technical users.
- **Tables**: Full markdown table rendering
  - **Purpose:** Presents data in a readable tabular layout.
- **containsMarkdown()**: Detection logic for rendering mode
  - **Purpose:** Chooses between plain text and markdown rendering automatically.

#### Text Processing
- **removeDiacritics()**: Czech character normalization for search
  - **Purpose:** Ensures filters work even when users omit diacritics.
- **NFD Normalization**: Consistent text comparison
  - **Purpose:** Produces reliable matching for multilingual content.

---

## Deployment - Production Environment

### Frontend Deployment (Vercel)

#### Configuration
- **Platform**: Vercel cloud hosting
  - **Purpose:** Provides globally distributed infrastructure optimized for Next.js apps.
- **Framework**: Next.js 15 (automatic optimization)
  - **Purpose:** Enables zero-config builds and edge caching for fast responses.
- **API Proxying**: 17 rewrite rules to Railway backend
  - **Purpose:** Keeps API traffic on the same domain while routing to the backend securely.
- **Rewrites**: `/api/*` → Railway backend URL
  - **Purpose:** Abstracts backend addresses so the frontend code stays environment-agnostic.
- **Exclusions**: NextAuth routes bypass proxy
  - **Purpose:** Allows authentication callbacks to be handled locally by Next.js APIs.

#### Key Routes Proxied
- Analysis: `/analyze`, `/stop-execution`
  - **Purpose:** Connects chat actions to backend processing endpoints.
- Chat: `/chat-threads`, `/chat/:path*`
  - **Purpose:** Supports thread CRUD operations via the backend.
- Data: `/catalog`, `/data-tables`, `/data-table`
  - **Purpose:** Supplies catalog and explorer pages with up-to-date data.
- Feedback: `/feedback`, `/sentiment`
  - **Purpose:** Routes user feedback securely to the backend services.
- Debug: `/debug/:path*`
  - **Purpose:** Grants engineers access to backend diagnostics through the same domain.

#### Environment Variables
- **NextAuth Config**: Google OAuth credentials, NextAuth secret
  - **Purpose:** Secures authentication flows and token signing.
- **Backend URL**: Railway backend for API rewrites
  - **Purpose:** Points Vercel rewrites to the correct backend instance per environment.
- **Public URL**: Base URL for OAuth callbacks
  - **Purpose:** Ensures Google redirects users to the correct hosted domain.

### Backend Deployment (Railway)

#### Configuration
- **Platform**: Railway cloud hosting
  - **Purpose:** Offers managed Postgres, environment variables, and deployments with minimal DevOps overhead.
- **Builder**: RAILPACK (custom buildpacks)
  - **Purpose:** Installs required system packages and handles Python builds efficiently.
- **Build Command**: uv installation → pip install → unzip files → cleanup
  - **Purpose:** Prepares the runtime environment with dependencies and data assets before startup.
- **Start Command**: uvicorn with dynamic PORT variable
  - **Purpose:** Launches the ASGI server with the port assigned by Railway.

#### Runtime Settings
- **Memory Limit**: 4GB override
  - **Purpose:** Provides headroom for concurrent agent runs and vector operations.
- **Restart Policy**: ON_FAILURE with 5 max retries
  - **Purpose:** Automatically recovers from transient issues without human intervention.
- **Replicas**: 1 (multi-region: europe-west4)
  - **Purpose:** Keeps costs predictable while deploying close to the target audience.
- **Sleep Mode**: Enabled for cost optimization
  - **Purpose:** Reduces spend during periods of inactivity.
- **System Packages**: libsqlite3-0 via RAILPACK_DEPLOY_APT_PACKAGES
  - **Purpose:** Ensures the Python sqlite3 module works in the container.

#### Build Process
1. Install uv (fast Python package installer)
   - **Purpose:** Speeds up dependency installation and reduces build time.
2. Install dependencies via `uv pip install .`
   - **Purpose:** Brings in core backend requirements for runtime.
3. Install dev dependencies via `uv pip install .[dev]`
   - **Purpose:** Makes optional tooling available for diagnostics and notebooks.
4. Unzip compressed data files (SQLite, ChromaDB)
   - **Purpose:** Restores packaged datasets required for runtime operations.
5. Cleanup zip files
   - **Purpose:** Keeps the container lightweight by removing temporary archives.
6. Start uvicorn server on dynamic port
   - **Purpose:** Boots the API entrypoint with the port assigned by Railway.

### Database Services

#### Supabase (PostgreSQL)
- **Purpose**: Checkpointing, conversation state, user management
  - **Usage:** Stores persistent state for LangGraph runs and user metadata to support resumable workflows.
- **Connection**: asyncpg with connection pooling
  - **Usage:** Maintains efficient, asynchronous database access with automatic connection reuse.
- **Tables**: checkpoints, writes, threads, sentiments
  - **Usage:** Organizes persisted data by function to simplify queries and maintenance.
- **Monitoring**: Health checks, prepared statement detection
  - **Usage:** Detects operational issues quickly so they can be addressed before impacting users.

#### Turso (SQLite Cloud)
- **Purpose**: CZSU statistical data storage
  - **Usage:** Hosts the normalized CZSU datasets that agent SQL queries operate on.
- **Connection**: libsql-client or MCP server
  - **Usage:** Provides secure, remote access for both direct API calls and MCP-mediated queries.
- **Data**: Imported CSV tables from CZSU API
  - **Usage:** Supplies authoritative statistical facts for analysis answers.
- **Access**: Remote via HTTP or local fallback
  - **Usage:** Keeps the system running whether the cloud service or local development environment is used.

#### Chroma Cloud (Vector DB)
- **Purpose**: Vector embeddings for hybrid search
  - **Usage:** Stores semantic representations of datasets and documents for retrieval.
- **Collections**: Dataset descriptions, PDF chunks
  - **Usage:** Separates retrieval domains to tailor search strategies for each content type.
- **Embeddings**: Azure OpenAI text-embedding-3-large
  - **Usage:** Produces high-quality vectors that capture multilingual semantics.
- **Search**: Hybrid (semantic + BM25) with Cohere reranking
  - **Usage:** Delivers precise retrieval results for the agent and frontend features.

### External Services Integration

#### Azure OpenAI
- **Models**: GPT-4o (analysis), GPT-4o-mini (summarization), text-embedding-3-large
  - **Purpose:** Covers the range of tasks from heavy-duty reasoning to lightweight summarization and embedding creation.
- **Purpose**: Query rewriting, SQL generation, answer formatting, embeddings
  - **Usage:** Powers the intelligent behaviors that convert natural language into actionable SQL and polished responses.
- **Configuration**: API key, endpoint, deployment names
  - **Usage:** Aligns the application with provisioned Azure deployments for stable operations.

#### Azure AI Services
- **Translation API**: Query translation for PDF search
  - **Purpose:** Normalizes multilingual input so the vector index remains language-agnostic.
- **Language Detection**: Automatic language identification
  - **Purpose:** Chooses the correct translation and response language without user input.
- **Purpose**: Bilingual support (Czech/English)
  - **Usage:** Ensures all components understand user queries and produce localized answers.

#### Cohere
- **Model**: multilingual rerank
  - **Purpose:** Adds a stronger semantic pass to sort retrieval results accurately.
- **Purpose**: Reranking retrieval results (datasets + PDFs)
  - **Usage:** Improves answer quality by prioritizing contextually relevant evidence.
- **Integration**: Two-step retrieval (hybrid search → rerank)
  - **Usage:** Combines fast initial retrieval with precision ranking for best-in-class relevancy.

#### LlamaParse
- **Purpose**: Premium PDF parsing (LLM-based)
  - **Usage:** Handles complex table and layout extractions that simpler parsers miss.
- **Alternative**: Azure Document Intelligence
  - **Usage:** Provides a fallback parsing option when LlamaParse is unavailable or cost-sensitive.
- **Output**: Structured text extraction from methodology PDFs
  - **Usage:** Feeds the PDF retrieval pipeline with clean, chunkable content.

#### Google OAuth
- **Purpose**: User authentication
  - **Usage:** Grants secure access control using familiar Google accounts.
- **Flow**: OAuth 2.0 with offline access
  - **Usage:** Supports long-lived sessions that require token refresh.
- **Integration**: NextAuth provider with token refresh
  - **Usage:** Simplifies frontend implementation by offloading auth complexity to NextAuth.

### LangSmith Integration

#### Tracing & Monitoring
- **Purpose**: Agent workflow observability
  - **Usage:** Captures execution traces for debugging, performance tuning, and compliance.
- **Traces**: Complete execution graphs with node-level details
  - **Usage:** Helps pinpoint bottlenecks or errors inside the LangGraph pipeline.
- **Metrics**: Latency, token usage, error rates
  - **Usage:** Provides quantitative insights that guide optimization efforts.

#### Evaluation
- **Datasets**: Golden datasets for retrieval testing
  - **Purpose:** Supplies controlled scenarios to measure retrieval accuracy.
- **Evaluators**:
  - `selection_correct`: Top-1 accuracy
    - **Purpose:** Measures precision of the first result returned by the retrieval pipeline.
  - `selection_in_top_n`: Top-N recall
    - **Purpose:** Assesses whether relevant results appear within the inspected window.
- **Experiments**: Hybrid search vs full pipeline comparison
  - **Purpose:** Validates the benefit of reranking and other enhancements before production rollout.

#### Feedback Collection
- **In-app Feedback**: Thumbs up/down + comments
  - **Purpose:** Gathers qualitative signals directly from users.
- **API Integration**: `/feedback` endpoint to LangSmith
  - **Purpose:** Channels detailed feedback into evaluation tooling.
- **Run Association**: Feedback linked to specific LangGraph executions
  - **Purpose:** Enables root-cause analysis when users report issues.

### Environment Variables Management

#### Security
- **Secret Storage**: Environment variables in hosting platforms
  - **Purpose:** Keeps credentials out of source control while supporting automated deployments.
- **No Hardcoding**: All credentials via .env files
  - **Purpose:** Allows safe rotation of secrets without code updates.
- **Separation**: Different .env for frontend and backend
  - **Purpose:** Limits blast radius and enforces least-privilege access per service.

#### Key Variables
- **Database URLs**: PostgreSQL, SQLite cloud connections
  - **Purpose:** Directs services to the appropriate persistence layers per environment.
- **API Keys**: Azure OpenAI, Cohere, LlamaParse, Google OAuth
  - **Purpose:** Authorizes access to external services required for the workflow.
- **Service URLs**: ChromaDB cloud, MCP server (if used)
  - **Purpose:** Configures integrations to reach the correct endpoints.
- **Feature Flags**: PDF functionality, MCP usage, fallback modes
  - **Purpose:** Toggles experimental or optional capabilities without redeploying code.

### Monitoring & Debugging

#### Health Checks
- **Endpoints**: `/health`, `/health/database`, `/health/memory`, `/health/rate-limits`
  - **Purpose:** Provides quick diagnostics for uptime monitoring tools.
- **Metrics**: Memory usage, pool status, rate limit state
  - **Purpose:** Surfaces performance indicators that can warn of impending issues.

#### Debug Endpoints
- **Checkpoints**: `/debug/chat/{thread_id}/checkpoints`
  - **Purpose:** Allows inspection of saved conversation states for troubleshooting.
- **Pool Status**: `/debug/pool-status`
  - **Purpose:** Displays connection pool health to spot leaks or saturation.
- **Run Inspection**: `/debug/run-id/{run_id}`
  - **Purpose:** Fetches detailed logs of specific executions for analysis.

#### Admin Controls
- **Cache Clearing**: `/admin/clear-cache`
  - **Purpose:** Provides operators a quick way to invalidate cached data after updates.
- **Statement Cleanup**: `/admin/clear-prepared-statements`
  - **Purpose:** Resolves persistent prepared-statement issues without restarting the service.
- **Environment Toggle**: `/debug/set-env`, `/debug/reset-env`
  - **Purpose:** Helps simulate configuration changes or reset overrides during debugging.

### Performance Optimization

#### Caching Strategies
- **Frontend**: localStorage (48-hour cache), IndexedDB (legacy)
  - **Purpose:** Improves perceived speed by avoiding repeated API calls for recent data.
- **Backend**: Bulk loading cache (5-minute TTL), prepared statement cache
  - **Purpose:** Reduces repeated computations and database overhead for common queries.
- **Cross-tab**: Storage event sync
  - **Purpose:** Keeps cache coherence when multiple tabs are active.

#### Memory Management
- **GC Triggers**: Threshold-based garbage collection (1900MB default)
  - **Purpose:** Prevents the process from exceeding platform limits under heavy load.
- **Profiling**: Optional memory profiler with configurable intervals
  - **Purpose:** Collects insights for tuning memory usage when issues are suspected.
- **Cleanup**: Explicit resource cleanup after retrieval operations
  - **Purpose:** Releases transient resources quickly to keep the service lightweight.

#### Concurrency Control
- **Analysis Semaphore**: Limit concurrent analysis requests
  - **Purpose:** Protects CPU-intensive operations from overwhelming the server.
- **IP Throttling**: Max 8 concurrent per IP
  - **Purpose:** Prevents a single client from consuming disproportionate resources.
- **Rate Limiting**: Request counting with exponential backoff
  - **Purpose:** Encourages clients to spread out requests during high load while keeping the system responsive.
