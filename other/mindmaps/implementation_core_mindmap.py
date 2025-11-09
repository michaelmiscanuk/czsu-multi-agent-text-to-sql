import os

from graphviz import Digraph

# Comprehensive mindmap structure based on implementation_core_v3.md
mindmap = {
    "CZSU Multi-Agent Text-to-SQL Implementation": {
        "1. High-Level System Architecture": {
            "Overview of System Layers and Services": {
                "Presentation Layer (Vercel-hosted Next.js UI)": [
                    "Browser app with React Server Components + client hooks",
                    "Renders chat, catalog, data explorer screens",
                    "Separate tier for global scaling and UI updates",
                    "Files: frontend/src/app/(authenticated)/chat/page.tsx",
                    "Files: frontend/src/app/layout.tsx",
                ],
                "Application Layer (Railway-hosted FastAPI services)": [
                    "Python backend for routing, validation, orchestration",
                    "Owns security checks before agent execution",
                    "Concentrates business logic server-side",
                    "Files: api/main.py, api/routes/analysis.py",
                ],
                "AI Orchestration Layer (LangGraph + Azure OpenAI)": [
                    "StateGraph coordinates prompt rewriting, retrieval, SQL, synthesis",
                    "LLM tool orchestration across workflow phases",
                    "Transparent reasoning steps, controllable retries",
                    "Files: api/routes/analysis.py, checkpointer/globals.py",
                    "Files: my_agent/langgraph/graph_builder.py",
                ],
                "Data Persistence Layer (Polyglot Storage)": {
                    "PostgreSQL (Supabase)": [
                        "Chat history and conversation state",
                        "Durable checkpoint storage",
                    ],
                    "Turso SQLite": [
                        "CZSU statistical datasets",
                        "Low-latency analytical reads",
                    ],
                    "Chroma Cloud": [
                        "Vector embeddings for semantic search",
                        "PDF and metadata chunks",
                    ],
                    "Implementation": [
                        "Files: checkpointer/postgres_checkpointer.py",
                        "Files: metadata/chromadb_client_factory.py",
                        "Files: data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py",
                    ],
                },
                "Integration & Identity Services": {
                    "Google OAuth": [
                        "Federated identity and sign-in",
                        "JWT token management",
                    ],
                    "CZSU API": [
                        "Authoritative statistical datasets",
                        "JSON-stat format ingestion",
                    ],
                    "LlamaParse": [
                        "High-fidelity PDF parsing",
                        "Table structure preservation",
                    ],
                    "Cohere Rerank": [
                        "Semantic reranking of search results",
                        "BM25 + vector result fusion",
                    ],
                    "Implementation": [
                        "Files: frontend/src/app/api/auth/[...nextauth]/route.ts",
                        "Files: api/dependencies/auth.py",
                        "Files: data/pdf_to_chromadb__llamaparse_parsed.py",
                        "Files: api/utils/retrieval.py",
                    ],
                },
                "Observability & Experimentation": [
                    "LangSmith trace ingestion and evaluation",
                    "Custom health endpoints and diagnostics",
                    "Golden datasets for quality testing",
                    "Files: api/routes/debug.py",
                    "Files: Evaluations/LangSmith_Evaluation/",
                    "Files: api/utils/debug.py",
                ],
            },
            "Technology Stack and Rationale": {
                "Next.js 15 with React 19": [
                    "Server rendering + interactive components",
                    "Fast load times and edge-ready deployments",
                    "Clean data fetching patterns",
                    "Files: frontend/next.config.ts",
                    "Files: frontend/src/app/page.tsx",
                ],
                "FastAPI + Uvicorn": [
                    "Async Python framework with type hints",
                    "Auto-generated OpenAPI documentation",
                    "High throughput ASGI server",
                    "Files: api/main.py, uvicorn_start.py",
                ],
                "LangGraph (LangChain Agents)": [
                    "Graph-based workflow engine",
                    "Reusable, debuggable agent steps",
                    "Supervised agent orchestration",
                    "Files: my_agent/langgraph/graph_builder.py",
                    "Files: api/routes/analysis.py",
                ],
                "Azure OpenAI + Cohere Rerank": {
                    "Azure OpenAI": [
                        "GPT-4o for reasoning and analysis",
                        "GPT-4o-mini for summaries",
                        "text-embedding-3-large for vectors",
                        "Enterprise controls and multilingual support",
                    ],
                    "Cohere": [
                        "Multilingual reranking API",
                        "BM25 and vector result fusion",
                    ],
                    "Implementation": [
                        "Files: api/utils/llm_clients.py",
                        "Files: api/utils/retrieval.py",
                    ],
                },
                "Database Stack (Polyglot Persistence)": {
                    "Supabase PostgreSQL": [
                        "Durable conversation state",
                        "Managed relational database",
                    ],
                    "Turso SQLite": [
                        "Low-latency analytics",
                        "Edge-replicated datasets",
                    ],
                    "Chroma Vector DB": [
                        "Semantic search embeddings",
                        "Cloud-hosted vector store",
                    ],
                    "Implementation": [
                        "Files: checkpointer/postgres_checkpointer.py",
                        "Files: data/sqllite_to_csvs.py",
                        "Files: metadata/chromadb_client_factory.py",
                    ],
                },
                "Deployment Pipeline (Vercel + Railway)": [
                    "Vercel: Frontend CDN hosting",
                    "Railway: Backend managed containers",
                    "Independent scaling and releases",
                    "Files: vercel.json, railway.toml, start_backend.bat",
                ],
                "NextAuth (Google OAuth 2.0)": [
                    "JWT session management",
                    "Automatic token refresh",
                    "Secure credential handling",
                    "Files: frontend/src/app/api/auth/[...nextauth]/route.ts",
                    "Files: api/dependencies/auth.py",
                ],
            },
            "Data Flow and Architectural Decisions": {
                "User Query Ingestion via /api/analyze": [
                    "Single backend entry point for chat turns",
                    "Authentication, throttling, logging",
                    "API gateway pattern for secure cloud apps",
                    "Files: frontend/vercel.json",
                    "Files: api/routes/analysis.py",
                ],
                "Agentic Pipeline Phases": {
                    "Phase 1: Rewrite": [
                        "Clarify ambiguous user prompts",
                        "Normalize query structure",
                    ],
                    "Phase 2: Retrieval": [
                        "Query vector and metadata stores",
                        "Gather evidence from multiple sources",
                    ],
                    "Phase 3: SQL Execution": [
                        "Generate and run SQL via MCP",
                        "Validate query results",
                    ],
                    "Phase 4: Synthesis": [
                        "Double-check output accuracy",
                        "Generate narrative answer",
                    ],
                    "Implementation": [
                        "Files: api/routes/analysis.py",
                        "Files: my_agent/langgraph/nodes/*.py",
                    ],
                },
                "Hybrid Retrieval Strategy": [
                    "Vector embeddings (Chroma) + structured metadata",
                    "Cohere reranking for relevance",
                    "BM25 + semantic fusion",
                    "Files: metadata/chromadb_client_factory.py",
                    "Files: api/utils/retrieval.py",
                ],
                "Model Context Protocol (MCP) SQL Execution": [
                    "SQL wrapper for Turso/local SQLite",
                    "Schema enforcement and query safety",
                    "No direct database access for agent",
                    "Files: metadata/chromadb_client_factory.py",
                    "Files: data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py",
                    "Files: my_agent/mcp/*.py",
                ],
                "Stateful Conversation Checkpointing": [
                    "Persist agent state in Supabase",
                    "Resume and audit conversations",
                    "Compliance with governance patterns",
                    "Files: checkpointer/postgres_checkpointer.py",
                    "Files: checkpointer/globals.py",
                ],
                "Bilingual Response Pipeline (Czech ↔ English)": [
                    "Language detection and translation",
                    "Retrieve in English, answer in Czech",
                    "Azure AI Translator integration",
                    "Files: api/utils/language_detection.py",
                    "Files: api/utils/retrieval.py",
                ],
            },
            "System Diagram Components": {
                "Request Routing and Hosting Separation": [
                    "Vercel forwards REST to Railway backend",
                    "OAuth callbacks and static assets at edge",
                    "Independent UI and API scaling",
                    "Files: frontend/vercel.json, railway.toml",
                ],
                "Agent Workflow Orchestration": [
                    "LangGraph coordinates Azure OpenAI, Cohere, MCP",
                    "Explicit reasoning pipeline visualization",
                    "Files: other/diagrams/ANALYZE_FLOW_DIAGRAM_V1.md",
                    "Files: my_agent/langgraph/graph_builder.py",
                ],
                "Data Access Zones": [
                    "Supabase, Turso, Chroma as distinct stores",
                    "Backend-only Postgres access",
                    "Agent reaches SQLite through MCP (defence-in-depth)",
                    "Files: checkpointer/postgres_checkpointer.py",
                    "Files: metadata/chromadb_client_factory.py",
                ],
                "Document Ingestion and Embedding Flow": [
                    "CZSU API → LlamaParse → Azure OpenAI → Chroma",
                    "Curated ingestion for trustworthy retrieval",
                    "Files: data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py",
                    "Files: metadata/chromadb_client_factory.py",
                ],
                "Feedback and Observability Loop": [
                    "LangSmith and diagnostic endpoints",
                    "Quality signals feed operations",
                    "Files: Evaluations/LangSmith_Evaluation/",
                    "Files: api/routes/debug.py",
                ],
            },
        },
        "2. Backend Implementation": {
            "Backend Architecture and Technologies": {
                "FastAPI Application Container": [
                    "Central ASGI app with routing, DI, middleware",
                    "Startup/shutdown lifecycle hooks",
                    "Type hints and OpenAPI generation",
                    "Files: api/main.py (app = FastAPI(...))",
                ],
                "Uvicorn Event Loop & Windows Policy": [
                    "ASGI server for async request handling",
                    "Platform-specific event loop policies",
                    "Windows selector policy for psycopg compatibility",
                    "Files: api/main.py (Windows policy), uvicorn_start.py",
                ],
                "Modular Routing Packages": [
                    "Dedicated route modules: analysis, chat, catalog, feedback, debug",
                    "Clear ownership boundaries",
                    "Files: api/routes/analysis.py",
                    "Files: api/routes/chat.py",
                    "Files: api/routes/catalog.py",
                ],
                "Configuration & Lifespan Management": [
                    "Env var loading via .env",
                    "Memory baselines and graceful shutdown",
                    "Checkpointer initialization on startup",
                    "Files: api/main.py (lifespan context)",
                    "Files: api.utils.memory",
                ],
            },
            "Agent Workflow and Orchestration": {
                "LangGraph StateGraph": [
                    "Deterministic, inspectable agent pipeline",
                    "Prompt rewriting node",
                    "Dual retrieval (vector + metadata) node",
                    "SQL execution via MCP node",
                    "Reflection and validation loops",
                    "Final answer formatting node",
                    "Explicit retry and cancellation control",
                ],
            },
            "API Design and Endpoint Purposes": {
                "Analysis Endpoint (POST /analyze)": [
                    "Accepts chat turn, triggers agent",
                    "Streams checkpoints to frontend",
                    "Returns synthesized answer + artefacts",
                    "Enforces auth, rate limits, structured responses",
                    "Frontend: ChatForm.tsx, useChatActions.ts",
                    "Files: api/routes/analysis.py (analyze handler)",
                ],
                "Chat Threads & Messages": {
                    "GET /chat-threads": [
                        "Paginated thread listing",
                        "Sync UI cache with server state",
                    ],
                    "GET /chat/{thread_id}": [
                        "Fetch single thread metadata",
                    ],
                    "GET /chat/all-messages-for-one-thread": [
                        "Paginated message history",
                    ],
                    "DELETE /chat/{thread_id}": [
                        "Delete thread and messages",
                    ],
                    "POST operations": [
                        "Create, rename threads",
                    ],
                    "Frontend": [
                        "ChatSidebar.tsx",
                        "ChatCacheProvider.tsx",
                    ],
                    "Implementation": [
                        "Files: api/routes/chat.py",
                        "Files: api/routes/messages.py",
                    ],
                },
                "Catalog Navigation (GET /catalog)": [
                    "Paginated dataset metadata",
                    "Selection descriptions and filters",
                    "Backend-for-frontend pattern",
                    "Frontend: catalog/page.tsx, DatasetsTable.tsx",
                    "Files: api/routes/catalog.py",
                ],
                "Data Explorer": {
                    "GET /data-tables": [
                        "List all SQLite tables",
                        "Power user exploration",
                    ],
                    "GET /data-table": [
                        "Return data slices with column metadata",
                        "Controlled query scope with caching/limits",
                    ],
                    "Frontend": [
                        "data/page.tsx",
                        "DataTableView.tsx",
                    ],
                    "Implementation": [
                        "Files: api/routes/bulk.py",
                        "Files: api/routes/catalog.py (get_data_table)",
                    ],
                },
                "Feedback & Sentiment": {
                    "POST /feedback": [
                        "Long-form LangSmith feedback",
                        "Tied to run IDs",
                    ],
                    "POST /sentiment": [
                        "Thumbs up/down reactions",
                        "Quick quality signals",
                    ],
                    "Frontend": [
                        "FeedbackPanel.tsx",
                        "useSentiment.ts",
                    ],
                    "Implementation": [
                        "Files: api/routes/feedback.py",
                    ],
                },
                "Operational & Debug Endpoints": {
                    "GET /health": [
                        "Database, memory, rate-limit status",
                    ],
                    "GET /debug/*": [
                        "Checkpoint history, pool metrics",
                        "Run ID diagnostics",
                    ],
                    "POST /stop-execution": [
                        "Cancel long-running agents",
                        "Frontend: ChatMessageActions.tsx",
                    ],
                    "Implementation": [
                        "Files: api/routes/health.py",
                        "Files: api/routes/debug.py",
                        "Files: api/routes/stop.py",
                    ],
                },
            },
            "Data Management and Persistence": {
                "LangGraph Checkpointer (Supabase PostgreSQL)": {
                    "AsyncPostgresSaver": [
                        "Thread metadata and conversation state",
                        "Sentiment tracking",
                        "Connection pooling with retry logic",
                    ],
                    "Benefits": [
                        "Strong durability for regulated workloads",
                        "Easy auditing with SQL tooling",
                        "Outperforms file storage",
                    ],
                    "Implementation": [
                        "Files: checkpointer/postgres_checkpointer.py",
                        "Files: checkpointer/globals.py",
                    ],
                },
                "Lifecycle Coordination with Fallbacks": [
                    "Startup: Initialize Postgres pool",
                    "Fallback: Switch to in-memory saver on failure",
                    "Graceful degradation during outages",
                    "Files: api/main.py (initialize_checkpointer)",
                    "Files: checkpointer/factory.py",
                ],
                "Schema-Aware State Objects": [
                    "Typed dictionaries (DataAnalysisState)",
                    "Track: prompts, rewrites, chunks, SQL, answers",
                    "Self-describing checkpoints for compliance",
                    "Files: checkpointer/globals.py (DataAnalysisState)",
                ],
                "ChromaDB Vector Store": [
                    "Embeddings for dataset descriptions + PDF chunks",
                    "Semantic retrieval with similarity thresholds",
                    "Cohere reranking integration",
                    "High-recall evidence for SQL justification",
                ],
                "Turso SQLite Analytics": [
                    "Normalized CZSU selections",
                    "Managed SQLite close to users",
                    "Low-latency analytical reads via MCP",
                    "No direct Postgres exposure",
                ],
            },
            "External Service Integration": {
                "Azure OpenAI Deployments": {
                    "GPT-4o": [
                        "Primary reasoning and analysis",
                    ],
                    "GPT-4o-mini": [
                        "Summaries and lightweight tasks",
                    ],
                    "text-embedding-3-large": [
                        "Vector embeddings",
                    ],
                    "Benefits": [
                        "Enterprise governance and regional availability",
                        "Multilingual models",
                        "Compliance certifications",
                    ],
                    "Implementation": [
                        "Files: api/utils/llm_clients.py",
                        "Env vars: api/config/settings.py",
                    ],
                },
                "Azure AI Translator & Language Detection": [
                    "Translate Czech PDFs to English",
                    "Detect user language for responses",
                    "Bilingual experience without manual toggles",
                    "Files: api/utils/retrieval.py",
                    "Files: api/utils/language_detection.py",
                ],
                "Cohere Rerank API": [
                    "Rerank BM25 + vector search results",
                    "Database and PDF retrieval pipelines",
                    "Pushes most semantically aligned passages to top",
                    "Files: api/utils/retrieval.py",
                ],
                "Google OAuth Verification": [
                    "Verify Google-issued ID tokens",
                    "Audience and signature checks",
                    "Federated identity, no password management",
                    "Files: api/dependencies/auth.py (get_current_user)",
                ],
                "CZSU API Ingestion Jobs": [
                    "Pull JSON-stat datasets and metadata",
                    "Convert to CSV/SQLite",
                    "Push to Turso and Chroma",
                    "Automated data freshness and traceability",
                    "Files: data/datasets_selections_get_csvs_01.py",
                    "Files: data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py",
                ],
                "LlamaParse PDF Processing": [
                    "Premium LLM-based parser",
                    "Extract structured text from methodology PDFs",
                    "Preserves tables and narrative structure",
                    "Better than basic OCR for retrieval relevance",
                    "Files: data/pdf_to_chromadb__llamaparse_parsed.py",
                ],
            },
            "Error Handling, Authentication, Middleware": {
                "Global Exception Handlers": [
                    "Consistent JSON error payloads",
                    "FastAPI exception handling middleware",
                ],
                "Throttling Middleware": [
                    "Per-IP concurrency limits",
                    "Semaphore-based request gating",
                ],
                "Memory Monitoring": [
                    "Log heavy requests",
                    "Prevent noisy-neighbor effects",
                ],
                "Stability": [
                    "Keep service stable under load",
                    "Railway resource limits compliance",
                ],
            },
            "Performance Optimizations": {
                "Semaphore-Based Throttling": [
                    "Limit concurrent analyses per IP",
                ],
                "Retry-Friendly Rate Limiting": [
                    "Exponential backoff for transient errors",
                ],
                "Gzip Compression": [
                    "Reduce response payload sizes",
                ],
                "Memory Cleanup Tasks": [
                    "Periodic garbage collection",
                    "Keep ASGI worker within limits",
                ],
                "Benefits": [
                    "Reduce perceived latency",
                    "Prevent resource exhaustion",
                ],
            },
        },
        "3. Frontend Implementation": {
            "Frontend Architecture and Technologies": {
                "Next.js 15 App Router": [
                    "File-system routing with server/client separation",
                    "Built-in metadata handling",
                    "Edge runtime support",
                    "Fast catalog pages (static) + interactive chat (client)",
                    "Files: frontend/src/app/layout.tsx",
                    "Files: frontend/src/app/chat/page.tsx",
                ],
                "React 19 Client Components": [
                    "Interactive UI: chat forms, message areas, filters",
                    "Hooks and state management",
                    "Strict client boundaries",
                    "Rich interactions without blocking server renders",
                    "Files: frontend/src/app/chat/page.tsx",
                    "Files: frontend/src/components/MessageArea.tsx",
                ],
                "TypeScript with Strict Mode": [
                    "Full type coverage across components, contexts, API clients",
                    "Catch interface mismatches early",
                    "Reduce runtime errors, improve refactoring confidence",
                    "Files: frontend/tsconfig.json",
                    "Files: frontend/src/types/index.ts",
                ],
                "TailwindCSS Utility-First Styling": [
                    "Atomic CSS classes for entire UI",
                    "No external stylesheets",
                    "Constraint system for consistent spacing/colors",
                    "Tree-shaking keeps bundle sizes small",
                    "Files: frontend/tailwind.config.ts",
                ],
                "NextAuth Session Management": [
                    "React context for session state",
                    "Sign-in/sign-out actions",
                    "Centralizes auth without prop drilling",
                    "Files: frontend/src/app/api/auth/[...nextauth]/route.ts",
                    "Files: frontend/src/app/layout.tsx",
                ],
            },
            "Main Pages and Features": {
                "Chat Interface (/chat)": {
                    "Components": [
                        "Thread sidebar with infinite scroll",
                        "Message area with streaming",
                        "Input bar with autocomplete",
                        "Feedback panels (thumbs, comments)",
                    ],
                    "Features": [
                        "Orchestrates agent calls",
                        "Displays streamed progress",
                        "Preserves context across reloads",
                        "Conversational UX patterns",
                    ],
                    "Implementation": [
                        "Files: frontend/src/app/chat/page.tsx",
                        "Files: frontend/src/components/MessageArea.tsx",
                    ],
                },
                "Catalog Browser (/catalog)": [
                    "Paginated table of CZSU datasets",
                    "Click-through to data explorer",
                    "Browseable overview for discovery",
                    "Supports exploratory workflows",
                    "Files: frontend/src/app/catalog/page.tsx",
                    "Files: frontend/src/components/DatasetsTable.tsx",
                ],
                "Data Explorer (/data)": {
                    "Features": [
                        "Autocomplete table search",
                        "Column filtering with numeric operators",
                        "Sortable grids",
                        "Inspect datasets without writing SQL",
                    ],
                    "Benefits": [
                        "Reduce friction for ad-hoc analysis",
                        "Backend controls query scope",
                    ],
                    "Implementation": [
                        "Files: frontend/src/app/data/page.tsx",
                        "Files: frontend/src/components/DataTableView.tsx",
                    ],
                },
                "Login Screen (/login)": [
                    "One-click Google OAuth button",
                    "Automatic redirect to chat on success",
                    "Low authentication barrier",
                    "Google-managed credentials",
                    "Files: frontend/src/app/login/page.tsx",
                ],
                "Thread Management Sidebar": [
                    "Scrollable conversation list",
                    "Create, rename, delete actions",
                    "Infinite-scroll pagination",
                    "Resume past analyses",
                    "Organize by topic",
                    "Files: frontend/src/app/chat/page.tsx (sidebar)",
                    "Files: frontend/src/contexts/ChatCacheContext.tsx",
                ],
                "Feedback & Sentiment Controls": [
                    "Thumbs up/down buttons per answer",
                    "Comment dropdown for detailed feedback",
                    "Inline quality signals at moment of judgement",
                    "Routes to LangSmith for evaluation",
                    "Files: frontend/src/components/MessageArea.tsx",
                ],
            },
            "State Management and Component Architecture": {
                "ChatCacheContext Provider": {
                    "Manages": [
                        "Threads, messages, active thread ID",
                        "Run IDs and sentiments",
                        "Pagination state",
                    ],
                    "Features": [
                        "localStorage persistence",
                        "Avoids prop drilling",
                        "Single source of truth for cache invalidation",
                    ],
                    "Implementation": [
                        "Files: frontend/src/contexts/ChatCacheContext.tsx",
                    ],
                },
                "localStorage 48-Hour Cache": [
                    "Serialized threads, messages, metadata",
                    "Timestamp-based expiry",
                    "Reduces API calls, speeds up reloads",
                    "Responsive UI when offline",
                    "Progressive web app strategy",
                    "Files: ChatCacheContext.tsx (CACHE_DURATION, loadFromLocalStorage)",
                ],
                "Cross-Tab Sync via Storage Events": [
                    "Listeners detect localStorage changes",
                    "Update state when another tab modifies cache",
                    "Prevents conflicting states",
                    "Avoids accidental duplicate requests",
                    "Files: ChatCacheContext.tsx (storage event listener)",
                ],
                "Optimistic UI Updates": [
                    "New messages appear immediately",
                    "Backend confirmation happens async",
                    "Silent corrections on error",
                    "Makes interactions feel instant",
                    "Critical for perceived performance",
                    "Files: frontend/src/app/chat/page.tsx",
                ],
                "Infinite Scroll Pagination Hook": [
                    "IntersectionObserver detects viewport entry",
                    "Triggers page loads automatically",
                    "Eliminates manual pagination",
                    "Lazy loading reduces memory footprint",
                    "Files: frontend/src/lib/hooks/useInfiniteScroll.ts",
                    "Files: frontend/src/app/chat/page.tsx",
                ],
                "Component Composition Pattern": [
                    "Shared components: MessageArea, InputBar, DatasetsTable, DataTableView",
                    "Accept props and callbacks for reuse",
                    "Promotes DRY code",
                    "Simplifies testing",
                    "Consistent UI updates across features",
                    "Files: frontend/src/components/*",
                    "Files: frontend/src/app/*/page.tsx",
                ],
            },
            "API Integration": [
                "Centralized apiFetch and authApiFetch utilities",
                "Token injection and timeout control",
                "Automatic retry on 401 errors",
                "Uniform API calls across frontend",
                "RESTful resource modeling",
            ],
            "Authentication Flow": [
                "NextAuth manages Google OAuth flows",
                "JWT callbacks enrich tokens with refresh",
                "Session callbacks expose user identity to components",
                "UI gates features and displays avatars",
                "Secure credential handling",
            ],
            "Advanced Features": {
                "Markdown Rendering": [
                    "Rich text display for agent answers",
                ],
                "Dataset Badge Navigation": [
                    "Quick access to related datasets",
                ],
                "SQL/PDF Modals": [
                    "Detailed inspection of queries and sources",
                ],
                "Progress Indicators": [
                    "Estimated times for long-running operations",
                ],
                "Diacritics-Normalized Search": [
                    "Czech language support",
                ],
                "Philosophy": [
                    "Polish UX without cluttering core workflows",
                    "Progressive enhancement principles",
                ],
            },
        },
        "4. Deployment Implementation": {
            "Deployment Strategies and Platforms": {
                "Vercel Edge Hosting (Frontend)": {
                    "Features": [
                        "Global CDN deployment",
                        "Automatic builds from Git commits",
                        "Environment-specific rewrites",
                    ],
                    "Benefits": [
                        "Static assets close to users worldwide",
                        "Reduced latency",
                        "Automatic traffic spike handling",
                        "Serverless deployment best practices",
                    ],
                    "Implementation": [
                        "Files: frontend/vercel.json",
                        "Vercel dashboard project config",
                    ],
                },
                "Railway Managed Containers (Backend)": {
                    "Features": [
                        "Buildpack-based deployment",
                        "Automated rollouts and health checks",
                        "Environment secrets injection",
                    ],
                    "Benefits": [
                        "Abstracts infrastructure complexity",
                        "One-click rollbacks",
                        "Usage-based billing",
                        "Better than self-managed Kubernetes for early-stage",
                    ],
                    "Implementation": [
                        "Files: railway.toml",
                        "Railway project dashboard",
                    ],
                },
                "API Proxying via Vercel Rewrites": {
                    "Mechanism": [
                        "Frontend routes /api/* to Railway backend",
                        "OAuth callbacks stay local",
                    ],
                    "Benefits": [
                        "CORS simplicity with consolidated domains",
                        "Hides backend URLs from clients",
                        "Backend swaps without frontend code changes",
                        "API gateway offloading pattern",
                    ],
                    "Implementation": [
                        "Files: frontend/vercel.json (17 rewrite rules)",
                    ],
                },
                "Multi-Region Backend Deployment": {
                    "Configuration": [
                        "Railway replicas in europe-west4",
                        "Automatic failover and load balancing",
                    ],
                    "Benefits": [
                        "Reduced latency for European users",
                        "GDPR data residency compliance",
                        "Geographic redundancy for higher availability",
                    ],
                    "Implementation": [
                        "Files: railway.toml (multiRegionConfig)",
                    ],
                },
                "Blue-Green Deployment Support": {
                    "Mechanism": [
                        "Railway overlaps new/old deployments",
                        "Configurable draining periods",
                    ],
                    "Benefits": [
                        "Zero-downtime updates",
                        "Time for health checks to pass",
                        "Instant rollbacks if new version fails",
                        "Continuous delivery principles",
                    ],
                    "Implementation": [
                        "Files: railway.toml (overlapSeconds, drainingSeconds)",
                    ],
                },
                "Automatic SSL/TLS Provisioning": {
                    "Providers": [
                        "Vercel: Let's Encrypt certificates",
                        "Railway: Let's Encrypt certificates",
                    ],
                    "Benefits": [
                        "Removes certificate administration overhead",
                        "Connections encrypted by default",
                        "Automatic renewal",
                        "Compliance with security baselines",
                    ],
                    "Implementation": [
                        "Platform-managed, no manual config",
                    ],
                },
            },
            "Build and Runtime Configuration": {
                "Vercel Build": [
                    "Auto-detects Next.js projects",
                    "Optimized production builds",
                    "Edge runtime compilation",
                ],
                "Railway Build (RAILPACK)": [
                    "Install uv package manager",
                    "Install Python dependencies",
                    "Unzip data files",
                    "Launch Uvicorn ASGI server",
                ],
                "Benefits": [
                    "Reproducible environments across deployments",
                    "Config as code practices",
                ],
            },
            "Database and External Service Setup": {
                "Supabase PostgreSQL (Managed Relational DB)": {
                    "Features": [
                        "Connection pooling",
                        "Automated backups",
                        "Real-time replication",
                        "Point-in-time recovery",
                        "Observability dashboards",
                    ],
                    "Benefits": [
                        "Enterprise-grade durability for checkpoints",
                        "No self-managing backups",
                        "Better operational simplicity than self-hosted",
                    ],
                    "Implementation": [
                        "Files: checkpointer/postgres_checkpointer.py",
                        "get_connection_string with Supabase credentials",
                    ],
                },
                "AsyncPostgresSaver Connection Pool": {
                    "Features": [
                        "Min/max pool sizes",
                        "Keepalive pings",
                        "Retry decorators for transient failures",
                    ],
                    "Benefits": [
                        "Prevents connection exhaustion under load",
                        "Detects stale connections early",
                        "Smooths over cloud network blips",
                        "Cloud database resilience patterns",
                    ],
                    "Implementation": [
                        "Files: checkpointer/postgres_checkpointer.py",
                        "get_connection_kwargs, retry_on_prepared_statement_error",
                    ],
                },
                "Turso SQLite Cloud (Analytical Dataset Storage)": {
                    "Features": [
                        "Managed libSQL service",
                        "Edge replicas for global access",
                        "HTTP API access",
                        "Branching for schema testing",
                    ],
                    "Benefits": [
                        "Fast analytical queries via global replication",
                        "Per-read pricing (not instance time)",
                        "Better for read-heavy analytics than traditional DBs",
                    ],
                    "Implementation": [
                        "Files: data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py",
                        "Turso CLI import workflow",
                    ],
                },
                "Chroma Cloud Vector Database": {
                    "Features": [
                        "Hosted embedding store",
                        "Multi-tenant isolation",
                        "Persistent collections",
                        "API-based access",
                    ],
                    "Benefits": [
                        "Offloads vector indexing maintenance",
                        "Auto-scales retrieval throughput",
                        "Secure with tenant boundaries",
                        "Less operational burden than self-hosted Qdrant/Weaviate",
                    ],
                    "Implementation": [
                        "Files: metadata/chromadb_client_factory.py",
                        "CloudClient instantiation with API keys",
                    ],
                },
                "Azure OpenAI Service Endpoints": {
                    "Features": [
                        "Regional deployments",
                        "Rate limits and abuse monitoring",
                        "Compliance certifications",
                        "Content filtering policies",
                    ],
                    "Benefits": [
                        "Meets enterprise privacy requirements",
                        "Predictable latency through regional endpoints",
                        "Critical for regulated analytics",
                    ],
                    "Implementation": [
                        "Env vars: api/config/settings.py",
                        "Endpoint URLs: api/utils/llm_clients.py",
                    ],
                },
                "Environment Secrets Management": {
                    "Platforms": [
                        "Vercel: Encrypted secrets at rest",
                        "Railway: Encrypted secrets at rest",
                    ],
                    "Mechanism": [
                        "Inject as env vars at runtime",
                    ],
                    "Benefits": [
                        "Avoid committing credentials to Git",
                        "Per-environment configs",
                        "Support rotation without code deploys",
                        "Twelve-factor app methodology",
                    ],
                    "Implementation": [
                        "Vercel/Railway dashboards",
                        ".env templates in repo root",
                    ],
                },
                "LangSmith Cloud Integration": {
                    "Features": [
                        "Trace ingestion endpoint",
                        "Project-based isolation",
                        "Evaluation dataset storage",
                    ],
                    "Benefits": [
                        "Long-term queryable traces",
                        "Offline evaluation runs",
                        "Team collaboration on quality metrics",
                        "No self-hosting infrastructure",
                    ],
                    "Implementation": [
                        "LANGCHAIN_API_KEY env var",
                        "Trace submission: api/routes/analysis.py",
                    ],
                },
            },
            "Monitoring and Debugging": {
                "Health Endpoints": [
                    "Database pool status",
                    "Memory usage metrics",
                    "Rate-limit state",
                ],
                "Debug Routes": [
                    "Checkpoint history",
                    "Run ID diagnostics",
                ],
                "Platform Dashboards": [
                    "Vercel: Request volumes, error rates, deployment logs",
                    "Railway: Container metrics, build logs, health checks",
                ],
                "Benefits": [
                    "Operational awareness",
                    "Rapid incident response",
                ],
            },
            "Performance Optimization": {
                "Gzip Compression": [
                    "Reduce response bandwidth",
                ],
                "48-Hour Browser Caching": [
                    "Reduce API call frequency",
                ],
                "Connection Pooling": [
                    "Reuse database connections",
                ],
                "Semaphore-Based Throttling": [
                    "Limit concurrent requests per IP",
                ],
                "Memory Cleanup Tasks": [
                    "Periodic garbage collection",
                ],
                "Benefits": [
                    "Reduce bandwidth usage",
                    "Improve responsiveness",
                    "Prevent resource exhaustion under sustained load",
                ],
            },
        },
        "Cross-Cutting Concerns": {
            "Security & Authentication": {
                "Google OAuth 2.0": [
                    "Federated identity via NextAuth",
                    "JWT token management",
                    "Automatic token refresh",
                ],
                "Backend Token Verification": [
                    "ID token signature checks",
                    "Audience validation",
                ],
                "Secrets Management": [
                    "Encrypted env vars on Vercel/Railway",
                    "No credentials in Git",
                ],
                "SSL/TLS": [
                    "Automatic Let's Encrypt certificates",
                    "All traffic encrypted",
                ],
            },
            "Observability & Quality": {
                "LangSmith Tracing": [
                    "Agent run instrumentation",
                    "Evaluation datasets",
                    "Quality metric tracking",
                ],
                "Health Endpoints": [
                    "Database status",
                    "Memory metrics",
                    "Rate limits",
                ],
                "Debug Routes": [
                    "Checkpoint inspection",
                    "Run ID diagnostics",
                ],
                "Feedback Collection": [
                    "Inline sentiment (thumbs)",
                    "Detailed comments",
                    "LangSmith integration",
                ],
            },
            "Performance & Scalability": {
                "Frontend": [
                    "Edge CDN deployment",
                    "Static asset caching",
                    "Optimistic UI updates",
                    "Infinite scroll pagination",
                ],
                "Backend": [
                    "Async request handling (Uvicorn)",
                    "Connection pooling (PostgreSQL)",
                    "Semaphore throttling",
                    "Gzip compression",
                    "Memory cleanup tasks",
                ],
                "Database": [
                    "Polyglot persistence (workload-specific stores)",
                    "Edge replication (Turso)",
                    "Managed scaling (Supabase, Chroma Cloud)",
                ],
            },
            "Reliability & Resilience": {
                "Graceful Degradation": [
                    "Fallback to in-memory checkpointer",
                    "Retry logic for transient failures",
                ],
                "Zero-Downtime Deploys": [
                    "Blue-green deployment (Railway)",
                    "Health check validation",
                ],
                "Multi-Region": [
                    "Geographic redundancy",
                    "Automatic failover",
                ],
                "Error Handling": [
                    "Global exception handlers",
                    "Consistent JSON error responses",
                ],
            },
            "Data Quality & Compliance": {
                "Authoritative Data Sources": [
                    "CZSU API ingestion",
                    "Automated freshness",
                ],
                "Audit Trail": [
                    "Persistent checkpoints in PostgreSQL",
                    "Run ID tracking",
                    "LangSmith trace history",
                ],
                "GDPR Compliance": [
                    "European data residency (europe-west4)",
                    "User data deletion (thread deletion)",
                ],
                "Quality Assurance": [
                    "LlamaParse for PDF accuracy",
                    "Cohere reranking for relevance",
                    "Reflection loops in agent workflow",
                ],
            },
        },
    }
}


def create_mindmap_graph(mindmap_dict, graph=None, parent=None, level=0):
    """Recursively create a Graphviz graph from the mindmap dictionary."""
    if graph is None:
        graph = Digraph(comment="CZSU Multi-Agent Text-to-SQL Implementation Mindmap")
        graph.attr(rankdir="LR")  # Left to right layout
        graph.attr("node", fontname="Arial", fontsize="10")
        graph.attr("edge", fontname="Arial", fontsize="8")

    # Color scheme for different levels
    colors = [
        "lightblue",
        "lightgreen",
        "lightyellow",
        "lightpink",
        "lightcyan",
        "lavender",
        "peachpuff",
        "thistle",
    ]

    for key, value in mindmap_dict.items():
        node_id = f"{parent}_{key}" if parent else key
        node_id = (
            node_id.replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        node_id = node_id.replace("/", "_").replace(".", "_").replace(",", "")

        # Set node color based on level
        color = colors[min(level, len(colors) - 1)]

        if isinstance(value, dict):
            # Branch node with sub-structure
            graph.node(
                node_id,
                key,
                shape="box",
                style="filled,rounded",
                fillcolor=color,
                penwidth="2",
            )
            if parent:
                graph.edge(parent, node_id)
            create_mindmap_graph(value, graph, node_id, level + 1)
        elif isinstance(value, list):
            # Leaf node with multiple items
            graph.node(
                node_id,
                key,
                shape="ellipse",
                style="filled",
                fillcolor=color,
                penwidth="1.5",
            )
            if parent:
                graph.edge(parent, node_id)
            for i, item in enumerate(value):
                item_id = f"{node_id}_item_{i}"
                # Truncate very long items for readability
                display_item = item if len(item) <= 60 else item[:57] + "..."
                graph.node(
                    item_id,
                    display_item,
                    shape="note",
                    style="filled",
                    fillcolor="white",
                )
                graph.edge(node_id, item_id, style="dotted")
        else:
            # Single leaf node
            graph.node(node_id, str(value), shape="plaintext")
            if parent:
                graph.edge(parent, node_id)

    return graph


def print_mindmap_text(mindmap_dict, prefix="", is_last=True):
    """Print a text-based representation of the mindmap in a vertical tree format."""
    keys = list(mindmap_dict.keys())
    for i, key in enumerate(keys):
        current_is_last = i == len(keys) - 1
        connector = "└── " if current_is_last else "├── "
        print(prefix + connector + key)

        value = mindmap_dict[key]
        extension = "    " if current_is_last else "│   "

        if isinstance(value, dict):
            print_mindmap_text(value, prefix + extension, current_is_last)
        elif isinstance(value, list):
            for j, item in enumerate(value):
                is_last_item = j == len(value) - 1
                item_connector = "└── " if is_last_item else "├── "
                print(prefix + extension + item_connector + item)


def main():
    """Generate and save the mindmap visualization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    print("Generating comprehensive implementation mindmap...")
    print("This may take a moment due to the large structure...")

    graph = create_mindmap_graph(mindmap)

    # Save as PNG
    png_path = os.path.join(script_dir, script_name)
    try:
        graph.render(png_path, format="png", cleanup=True)
        print(f"\n✓ Mindmap saved as '{png_path}.png'")
    except Exception as e:
        print(f"\n✗ Error saving PNG: {e}")

    # Save as PDF for better quality
    pdf_path = os.path.join(script_dir, script_name)
    try:
        graph.render(pdf_path, format="pdf", cleanup=True)
        print(f"✓ Mindmap saved as '{pdf_path}.pdf'")
    except Exception as e:
        print(f"✗ Error saving PDF: {e}")

    # Save as SVG for scalable vector graphics
    svg_path = os.path.join(script_dir, script_name)
    try:
        graph.render(svg_path, format="svg", cleanup=True)
        print(f"✓ Mindmap saved as '{svg_path}.svg'")
    except Exception as e:
        print(f"✗ Error saving SVG: {e}")

    # Print text representation
    print("\n" + "=" * 80)
    print("TEXT-BASED MINDMAP STRUCTURE:")
    print("=" * 80 + "\n")
    print_mindmap_text(mindmap)
    print("\n" + "=" * 80)
    print("MINDMAP GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
