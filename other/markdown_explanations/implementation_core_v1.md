# Implementation Core Topics

## Overview - Implementation Architecture

### High-Level System Architecture
- **Multi-tier Architecture**: Frontend (Vercel), Backend (Railway), AI Layer (Azure/Cohere), Data Storage (Supabase/Turso/Chroma Cloud)
- **Deployment Strategy**: Microservices with cloud-hosted components
- **Technology Stack**: React/Next.js frontend, FastAPI backend, LangGraph agent workflow
- **Service Integration**: External APIs (Google OAuth, Azure OpenAI, Azure Translator, Cohere, LlamaParse)
- **Data Flow**: User query → Backend API → LangGraph agent → Multiple AI services → Response synthesis

### Core Components Overview
- **Frontend Layer**: Next.js 15 with React 19, deployed on Vercel
- **Backend Layer**: FastAPI application with LangGraph, deployed on Railway
- **AI Components**: LangGraph agent workflow, LangSmith tracing, Azure OpenAI models, translation services
- **Data Storage**: PostgreSQL (Supabase), SQLite (Turso), Vector DB (Chroma Cloud)
- **External Services**: Google OAuth 2.0, CZSU API, LlamaParse, Cohere reranking

### Key Architectural Decisions
- **Agentic Workflow**: LangGraph for orchestrated multi-step reasoning
- **Dual Retrieval Sources**: Database metadata + PDF documentation for comprehensive answers
- **Hybrid Search**: Semantic + BM25 retrieval with reranking
- **Stateful Conversations**: PostgreSQL checkpointing for conversation persistence
- **Token Optimization**: Message summarization at key workflow points
- **Bilingual Support**: Czech and English with automatic language detection

---

## Backend - Core Functionality

### 1. LangGraph Agent Workflow

#### Workflow Architecture
- **Graph-based Orchestration**: StateGraph with controlled execution flow
- **Multi-phase Processing**: Query rewriting → Parallel retrieval → SQL generation → Answer synthesis
- **Conditional Routing**: Smart decision points based on data availability
- **Iteration Control**: Maximum iteration limits with reflection loops

#### Agent Phases

##### Phase 1: Query Preprocessing
- **Prompt Rewriting**: Conversational context resolution, pronoun replacement, topic change detection
- **Memory Summarization**: Token-efficient context preservation (summary + last message pattern)
- **Optimization**: Search query enhancement with synonyms

##### Phase 2: Parallel Retrieval (Dual Branches)
- **Branch A - Database Selections**:
  - Hybrid search (semantic + BM25) on dataset descriptions
  - Cohere reranking for relevance
  - Top-k filtering with similarity thresholds
- **Branch B - PDF Documentation**:
  - Translation to English for vector search
  - Hybrid search on parsed PDF content
  - Cohere reranking with relevance thresholds

##### Phase 3: SQL Generation & Execution
- **Schema Loading**: Dynamic schema retrieval from metadata database
- **Agentic Tool Calling**: LLM autonomously decides when to execute SQL queries
- **MCP Integration**: Model Context Protocol with remote/local fallback
- **Iterative Data Gathering**: Multiple query execution until sufficient information collected

##### Phase 4: Reflection & Self-Correction
- **Result Analysis**: LLM evaluates query results for completeness
- **Decision Making**: "improve" (better query needed) or "answer" (sufficient data)
- **Iteration Limits**: Controlled loops with MAX_ITERATIONS

##### Phase 5: Answer Synthesis
- **Multi-source Integration**: Combines SQL results + PDF chunks
- **Bilingual Response**: Matches query language automatically
- **Markdown Formatting**: Structured answer presentation
- **Follow-up Generation**: Contextual suggestions for continued conversation

#### State Management
- **DataAnalysisState TypedDict**: 15 fields tracking workflow state
- **Key State Fields**: prompt, rewritten_prompt, messages, iteration, queries_and_results, top_selection_codes, top_chunks, final_answer
- **Reducers**: Limited queries reducer for memory efficiency
- **Checkpointing**: PostgreSQL persistence for conversation resumption

#### Key Design Principles
- **Controlled Iteration**: Cycle prevention with iteration counters
- **Resource Cleanup**: Explicit ChromaDB client closure and garbage collection
- **Token Management**: Automatic summarization at 4 workflow points
- **Modular Architecture**: Separated routing logic in dedicated modules

### 2. FastAPI Application Structure

#### Configuration & Lifecycle
- **Environment Bootstrapping**: Absolute imports, dotenv loading, event loop policy
- **Runtime Settings**: Checkpointer globals, concurrency limits, memory monitoring
- **Lifespan Management**: Startup sequence (checkpointer init, memory baseline), shutdown cleanup
- **Memory Controls**: GC thresholds, profiling when enabled

#### Middleware Stack
- **CORS**: Permissive policy for cross-origin requests
- **GZip Compression**: Response compression for JSON payloads over 1000 bytes
- **Throttling**: IP-based semaphore limits (max 8 concurrent requests per IP)
- **Rate Limiting**: Request tracking with retry logic and wait mechanisms
- **Memory Monitoring**: Request counting and logging for heavy endpoints

#### Route Architecture

##### Core Routes
- **Analysis** (`/analyze`):
  - Main LangGraph execution endpoint
  - Concurrency control via analysis_semaphore
  - Execution registration for cancellation support
  - InMemorySaver fallback on database failures
  
- **Chat Management** (`/chat-threads`, `/chat/{thread_id}`):
  - Paginated thread listing
  - Thread deletion with cleanup
  - Message retrieval per thread
  - Bulk message loading optimization

- **Catalog & Data** (`/catalog`, `/data-tables`, `/data-table`):
  - Dataset browsing with pagination
  - Table listing with descriptions
  - Dynamic table data retrieval with columns/rows

- **Feedback System** (`/feedback`, `/sentiment`):
  - LangSmith feedback submission
  - Database sentiment tracking
  - Run ID-based feedback association

##### Support Routes
- **Health Checks**: Database, memory, rate limits, prepared statements
- **Debug Endpoints**: Checkpoint inspection, pool status, run ID debugging
- **Stop Execution**: Cancellation support with thread_id + run_id

#### Authentication
- **Google OAuth Integration**: JWT token verification with google-auth library
- **JWK Retrieval**: Signature verification via Google's public keys
- **Token Extraction**: Bearer token from Authorization header
- **User Info**: Email extraction for user tracking

#### Error Handling & Recovery
- **Global Exception Handlers**: RequestValidationError, HTTPException, ValueError, generic Exception
- **Comprehensive Logging**: log_comprehensive_error utility for debugging
- **Graceful Degradation**: InMemorySaver fallback when PostgreSQL unavailable
- **Prepared Statement Recovery**: Detection and cleanup of database errors

### 3. Checkpointer System

#### PostgreSQL Checkpointer
- **Async Implementation**: AsyncPostgresSaver with connection pooling
- **Connection Management**: Pool lifecycle with min/max sizes, timeouts, keepalives
- **Retry Logic**: Decorator-based retry for transient failures
- **State Persistence**: Conversation state storage and retrieval

#### Factory Pattern
- **initialize_checkpointer()**: Bootstrap with retry logic
- **get_global_checkpointer()**: Singleton pattern for shared instance
- **cleanup_checkpointer()**: Pool closure and cleanup
- **Fallback Strategy**: InMemorySaver when PostgreSQL unavailable

#### User & Thread Management
- **Thread Operations**: Thread creation, retrieval, deletion
- **Sentiment Tracking**: Run-based sentiment storage and retrieval
- **Message Metadata**: Previews, timestamps, run counts

#### Health & Monitoring
- **Pool Status**: Connection pool metrics
- **Prepared Statement Detection**: Error pattern recognition and cleanup
- **Rate Limiting Integration**: Request tracking and throttling

### 4. Data Preparation Pipeline

#### CZSU Data Ingestion
- **API Integration**: Fetch datasets and selections from CZSU API
- **JSON-stat Processing**: Convert statistical data to DataFrames
- **CSV Export**: Save to structured CSV files
- **SQLite Database**: Import CSVs to local SQLite database
- **Cloud Upload**: Deploy to Turso (SQLite cloud) via CLI

#### Metadata Extraction
- **Schema Retrieval**: Get metadata for all selections via CZSU API
- **LLM Enhancement**: Generate extended descriptions using GPT-4o
- **Selection Descriptions**: Create human-readable dataset descriptions
- **Metadata Database**: Store in separate SQLite DB for schema lookups

#### PDF Documentation Processing
- **Parsing Strategies**: LlamaParse (premium LLM-based), Azure Document Intelligence
- **Text Extraction**: Extract text from CZSU methodology PDFs
- **Chunking**: Split documents into semantic chunks
- **Vector Embeddings**: Generate embeddings using Azure OpenAI
- **ChromaDB Storage**: Store in local or cloud vector database

#### Vector Database Setup
- **Local ChromaDB**: Development environment with persistent storage
- **Chroma Cloud**: Production deployment with remote access
- **Dual Collections**: Separate collections for dataset descriptions and PDF chunks
- **Hybrid Search Support**: BM25 + semantic search capabilities

### 5. MCP (Model Context Protocol) Integration

#### Architecture
- **Optional Component**: Remote MCP server or local SQLite fallback
- **FastMCP Implementation**: Python-based MCP server for SQLite queries
- **LangChain Adapter**: Integration via langchain-mcp-adapters
- **Tool Calling**: `sqlite_query` tool for agentic SQL execution

#### Deployment Modes
- **Local Fallback**: Direct SQLite access (default, `USE_LOCAL_SQLITE_FALLBACK=1`)
- **Remote MCP**: FastMCP Cloud deployment for centralized access
- **Dual Support**: Automatic fallback logic

#### Benefits
- **Isolation**: Separate database access from main application
- **Reusability**: MCP server can serve multiple clients
- **Standardization**: Protocol-based communication

---

## Frontend - Core Functionality

### 1. Next.js Application Architecture

#### Configuration
- **Framework**: Next.js 15 with React 19
- **TypeScript**: Strict type checking with path aliases (@/*)
- **Styling**: TailwindCSS 4 with custom utility classes
- **Authentication**: NextAuth with Google OAuth provider
- **Deployment**: Vercel with API proxying to Railway backend

#### App Router Structure
- **Dual Layout Pattern**: Server component for metadata, client component for auth logic
- **Route Protection**: Public vs protected routes with automatic redirects
- **Session Management**: NextAuth SessionProvider wrapper

### 2. Pages & Features

#### Chat Interface (`/chat`)
- **Thread Management**: Create, rename, delete conversations
- **Message Flow**: User input → backend analysis → response display
- **Optimistic UI**: Immediate loading states with backend sync
- **Recovery Mechanism**: Fallback to PostgreSQL on errors
- **Cancellation**: Stop execution support with instant UI feedback
- **Persistence**: localStorage for drafts, active thread, full cache

#### Catalog (`/catalog`)
- **Dataset Browser**: Paginated table of CZSU datasets
- **Search & Filter**: Client-side filtering with diacritics removal
- **Navigation**: Click-through to data view

#### Data Explorer (`/data`)
- **Smart Search**: Auto-complete with prefix matching (* for codes only)
- **Advanced Filtering**: Column-specific filters with numeric operators (>, <, >=, <=, !=, =)
- **Sortable Columns**: Multi-state sorting (asc/desc/none)
- **Cross-Navigation**: Links to catalog with pre-filled filters
- **State Persistence**: localStorage sync for all filters

#### Login (`/login`)
- **Google OAuth**: One-click sign-in
- **Auto-redirect**: Authenticated users sent to /chat

### 3. State Management

#### ChatCacheContext
- **Centralized Cache**: Threads, messages, run IDs, sentiments
- **localStorage Persistence**: 48-hour cache with automatic save/load
- **Cross-tab Sync**: Storage event listener for multi-tab coordination
- **Pagination**: Server-side with automatic message loading
- **Bulk Loading**: Single API call for all threads' messages (N+1 problem solution)

#### Key Features
- **Page Refresh Detection**: F5 forces API refresh, navigation uses cache
- **Cross-tab Loading Prevention**: Prevents concurrent requests across tabs
- **Invalidation**: Manual cache refresh controls
- **Optimistic Updates**: UI reflects changes before backend confirmation

### 4. Components Architecture

#### Layout Components
- **Header**: Sticky navigation with active route highlighting
- **AuthButton**: Dual-mode (compact/full) with user avatar
- **SessionProviderWrapper**: NextAuth context provider
- **AuthGuard**: Route protection based on auth status

#### Chat Components
- **MessageArea**: 
  - Markdown rendering with custom overrides
  - Copy functionality (rich text + plain text)
  - Datasets badges with navigation
  - SQL/PDF modals
  - Follow-up prompts (clickable badges)
  - Progress tracking with 8-minute estimates
  
- **FeedbackComponent**:
  - Thumbs up/down sentiment tracking
  - Comment textarea dropdown
  - Dual submission (LangSmith + database)
  - localStorage persistence
  - Visual confirmation

#### Data Components
- **DatasetsTable**: Hybrid pagination (backend when no filter, client when filtered)
- **DataTableView**: Auto-complete search, multi-column filtering, sorting

### 5. API Integration

#### Configuration
- **Base URL**: Environment-aware (`/api` in production via Vercel rewrites)
- **Timeout**: 10-minute limit for long-running analyses
- **Authentication**: Bearer token in Authorization header

#### Fetch Utilities
- **apiFetch()**: Generic fetch with timeout, logging, error handling
- **authApiFetch()**: Auto-refresh on 401 errors
- **Token Refresh**: Automatic retry with new token from getSession()

#### Backend Routes
- **Chat**: Thread CRUD, message retrieval, bulk loading
- **Analysis**: `/analyze` for question answering, `/stop-execution` for cancellation
- **Feedback**: LangSmith feedback + sentiment tracking
- **Data**: Catalog pagination, table listing, table data

### 6. Authentication Flow

#### NextAuth Setup
- **Google Provider**: OAuth with offline access for token refresh
- **JWT Callback**: Stores access token, refresh token, id_token, expiry
- **Session Callback**: Exposes tokens and user info to client
- **Token Refresh**: Automatic refresh mechanism via Google OAuth API

#### Client-side Auth
- **useSession Hook**: Session state access
- **signIn/signOut**: Authentication actions
- **Cache Cleanup**: clearCacheForUserChange() on logout

### 7. Advanced Features

#### Infinite Scroll
- **useInfiniteScroll Hook**: IntersectionObserver-based pagination
- **Auto-loading**: Triggers when sentinel element enters viewport
- **Duplicate Prevention**: isLoadingRef flag

#### Sentiment System
- **useSentiment Hook**: Per-run thumbs up/down tracking
- **Optimistic Updates**: Immediate UI reflection
- **Database Sync**: Background POST to /sentiment

#### Markdown Rendering
- **markdown-to-jsx**: Custom component overrides
- **Code Blocks**: Syntax highlighting support
- **Tables**: Full markdown table rendering
- **containsMarkdown()**: Detection logic for rendering mode

#### Text Processing
- **removeDiacritics()**: Czech character normalization for search
- **NFD Normalization**: Consistent text comparison

---

## Deployment - Production Environment

### Frontend Deployment (Vercel)

#### Configuration
- **Platform**: Vercel cloud hosting
- **Framework**: Next.js 15 (automatic optimization)
- **API Proxying**: 17 rewrite rules to Railway backend
- **Rewrites**: `/api/*` → Railway backend URL
- **Exclusions**: NextAuth routes bypass proxy

#### Key Routes Proxied
- Analysis: `/analyze`, `/stop-execution`
- Chat: `/chat-threads`, `/chat/:path*`
- Data: `/catalog`, `/data-tables`, `/data-table`
- Feedback: `/feedback`, `/sentiment`
- Debug: `/debug/:path*`

#### Environment Variables
- **NextAuth Config**: Google OAuth credentials, NextAuth secret
- **Backend URL**: Railway backend for API rewrites
- **Public URL**: Base URL for OAuth callbacks

### Backend Deployment (Railway)

#### Configuration
- **Platform**: Railway cloud hosting
- **Builder**: RAILPACK (custom buildpacks)
- **Build Command**: uv installation → pip install → unzip files → cleanup
- **Start Command**: uvicorn with dynamic PORT variable

#### Runtime Settings
- **Memory Limit**: 4GB override
- **Restart Policy**: ON_FAILURE with 5 max retries
- **Replicas**: 1 (multi-region: europe-west4)
- **Sleep Mode**: Enabled for cost optimization
- **System Packages**: libsqlite3-0 via RAILPACK_DEPLOY_APT_PACKAGES

#### Build Process
1. Install uv (fast Python package installer)
2. Install dependencies via `uv pip install .`
3. Install dev dependencies via `uv pip install .[dev]`
4. Unzip compressed data files (SQLite, ChromaDB)
5. Cleanup zip files
6. Start uvicorn server on dynamic port

### Database Services

#### Supabase (PostgreSQL)
- **Purpose**: Checkpointing, conversation state, user management
- **Connection**: asyncpg with connection pooling
- **Tables**: checkpoints, writes, threads, sentiments
- **Monitoring**: Health checks, prepared statement detection

#### Turso (SQLite Cloud)
- **Purpose**: CZSU statistical data storage
- **Connection**: libsql-client or MCP server
- **Data**: Imported CSV tables from CZSU API
- **Access**: Remote via HTTP or local fallback

#### Chroma Cloud (Vector DB)
- **Purpose**: Vector embeddings for hybrid search
- **Collections**: Dataset descriptions, PDF chunks
- **Embeddings**: Azure OpenAI text-embedding-3-large
- **Search**: Hybrid (semantic + BM25) with Cohere reranking

### External Services Integration

#### Azure OpenAI
- **Models**: GPT-4o (analysis), GPT-4o-mini (summarization), text-embedding-3-large
- **Purpose**: Query rewriting, SQL generation, answer formatting, embeddings
- **Configuration**: API key, endpoint, deployment names

#### Azure AI Services
- **Translation API**: Query translation for PDF search
- **Language Detection**: Automatic language identification
- **Purpose**: Bilingual support (Czech/English)

#### Cohere
- **Model**: multilingual rerank
- **Purpose**: Reranking retrieval results (datasets + PDFs)
- **Integration**: Two-step retrieval (hybrid search → rerank)

#### LlamaParse
- **Purpose**: Premium PDF parsing (LLM-based)
- **Alternative**: Azure Document Intelligence
- **Output**: Structured text extraction from methodology PDFs

#### Google OAuth
- **Purpose**: User authentication
- **Flow**: OAuth 2.0 with offline access
- **Integration**: NextAuth provider with token refresh

### LangSmith Integration

#### Tracing & Monitoring
- **Purpose**: Agent workflow observability
- **Traces**: Complete execution graphs with node-level details
- **Metrics**: Latency, token usage, error rates

#### Evaluation
- **Datasets**: Golden datasets for retrieval testing
- **Evaluators**: 
  - `selection_correct`: Top-1 accuracy
  - `selection_in_top_n`: Top-N recall
- **Experiments**: Hybrid search vs full pipeline comparison

#### Feedback Collection
- **In-app Feedback**: Thumbs up/down + comments
- **API Integration**: `/feedback` endpoint to LangSmith
- **Run Association**: Feedback linked to specific LangGraph executions

### Environment Variables Management

#### Security
- **Secret Storage**: Environment variables in hosting platforms
- **No Hardcoding**: All credentials via .env files
- **Separation**: Different .env for frontend and backend

#### Key Variables
- **Database URLs**: PostgreSQL, SQLite cloud connections
- **API Keys**: Azure OpenAI, Cohere, LlamaParse, Google OAuth
- **Service URLs**: ChromaDB cloud, MCP server (if used)
- **Feature Flags**: PDF functionality, MCP usage, fallback modes

### Monitoring & Debugging

#### Health Checks
- **Endpoints**: `/health`, `/health/database`, `/health/memory`, `/health/rate-limits`
- **Metrics**: Memory usage, pool status, rate limit state

#### Debug Endpoints
- **Checkpoints**: `/debug/chat/{thread_id}/checkpoints`
- **Pool Status**: `/debug/pool-status`
- **Run Inspection**: `/debug/run-id/{run_id}`

#### Admin Controls
- **Cache Clearing**: `/admin/clear-cache`
- **Statement Cleanup**: `/admin/clear-prepared-statements`
- **Environment Toggle**: `/debug/set-env`, `/debug/reset-env`

### Performance Optimization

#### Caching Strategies
- **Frontend**: localStorage (48-hour cache), IndexedDB (legacy)
- **Backend**: Bulk loading cache (5-minute TTL), prepared statement cache
- **Cross-tab**: Storage event sync

#### Memory Management
- **GC Triggers**: Threshold-based garbage collection (1900MB default)
- **Profiling**: Optional memory profiler with configurable intervals
- **Cleanup**: Explicit resource cleanup after retrieval operations

#### Concurrency Control
- **Analysis Semaphore**: Limit concurrent analysis requests
- **IP Throttling**: Max 8 concurrent per IP
- **Rate Limiting**: Request counting with exponential backoff
