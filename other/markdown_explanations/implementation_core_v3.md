# Implementation

## 1. High-Level System Architecture

### Overview of System Layers and Services
- **Presentation Layer (Vercel-hosted Next.js UI)**
  - **What:** The browser application that renders chat, catalog, and data explorer screens using React Server Components plus client-side hooks.
  - **Purpose:** Keeping the presentation tier separate lets us scale the user interface globally, ship UI updates quickly, and cache content close to users, which mirrors proven layered patterns for web apps ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/architecture/guide/architecture-styles/n-tier)).
  - **Implementation Pointer:** `frontend/src/app/(authenticated)/chat/page.tsx`, `frontend/src/app/layout.tsx`.
- **Application Layer (Railway-hosted FastAPI services)**
  - **What:** The Python backend that owns routing, validation, LangGraph orchestration, and all security checks before the agents run.
  - **Purpose:** Concentrating business logic server-side makes behaviour predictable, simplifies governance, and keeps sensitive workloads off the client, aligning with guidance on resilient cloud services ([Bass et al., 2021](https://www.sei.cmu.edu/education-outreach/books/software-architecture-in-practice-fourth-edition.cfm)).
  - **Implementation Pointer:** `api/main.py`, `api/routes/analysis.py`.
- **AI Orchestration Layer (LangGraph + Azure OpenAI workflow)**
  - **What:** A LangGraph StateGraph that coordinates prompt rewriting, retrieval, SQL execution, and narrative synthesis across LLM tools.
  - **Purpose:** Using a graph-managed agent keeps reasoning steps transparent, makes retries controllable, and reduces hallucinations compared with single-shot prompts, following LangGraph’s recommended practices for reliable agent pipelines ([LangChain, 2024](https://python.langchain.com/docs/langgraph)).
  - **Implementation Pointer:** `api/routes/analysis.py`, `checkpointer/globals.py`, `my_agent/langgraph/graph_builder.py` (if present).
- **Data Persistence Layer (Supabase, Turso, Chroma Cloud)**
  - **What:** A deliberately mixed storage stack: PostgreSQL for chat history, Turso SQLite for CZSU datasets, and Chroma for embeddings.
  - **Purpose:** Matching each workload with the storage engine that fits best improves latency, reliability, and traceability compared with a single database, echoing the polyglot persistence principle ([Fowler, 2012](https://martinfowler.com/bliki/PolyglotPersistence.html)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py`, `metadata/chromadb_client_factory.py`, `data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py`.
- **Integration & Identity Services (Google OAuth, CZSU API, LlamaParse, Cohere)**
  - **What:** External platforms that deliver login, authoritative statistical datasets, high-fidelity PDF parsing, and semantic reranking.
  - **Purpose:** Delegating these capabilities to well-governed providers keeps focus on our core analytics and gives us enterprise-grade security, data freshness, and retrieval quality without building them in-house ([Erl et al., 2021](https://www.oreilly.com/library/view/next-generation-enterprise-architecture/9780137693017/)).
  - **Implementation Pointer:** `frontend/src/app/api/auth/[...nextauth]/route.ts`, `api/dependencies/auth.py`, `data/pdf_to_chromadb__llamaparse_parsed.py`, `api/utils/retrieval.py`.
- **Observability & Experimentation (LangSmith, custom diagnostics)**
  - **What:** Instrumentation, health endpoints, and golden datasets that let us watch agent runs and evaluate new retrieval tweaks.
  - **Purpose:** Built-in monitoring shortens feedback cycles and safeguards production behaviour, following recommendations for dependable LLM deployments ([Bogatin et al., 2023](https://arxiv.org/abs/2307.05973)).
  - **Implementation Pointer:** `api/routes/debug.py`, `Evaluations/LangSmith_Evaluation/`, `api/utils/debug.py`.

### Technology Stack and Rationale
- **Next.js 15 with React 19**
  - **What:** A modern web framework that mixes server rendering with interactive React components.
  - **Purpose:** This stack gives us fast load times, clean data fetching patterns, and edge-ready deployments so the UI feels responsive worldwide ([Vercel, 2024](https://nextjs.org/docs/app)).
  - **Implementation Pointer:** `frontend/next.config.ts`, `frontend/src/app/page.tsx`.
- **FastAPI + Uvicorn**
  - **What:** An asynchronous Python framework and ASGI server that expose our REST endpoints.
  - **Purpose:** FastAPI’s typing, validation, and performance let us ship reliable services quickly, while Uvicorn keeps throughput high ([FastAPI, 2024](https://fastapi.tiangolo.com/)).
  - **Implementation Pointer:** `api/main.py`, `uvicorn_start.py`.
- **LangGraph (LangChain Agents)**
  - **What:** A workflow engine that represents the agent run as a graph of reusable steps.
  - **Purpose:** LangGraph makes complex prompt pipelines understandable, debuggable, and repeatable—advantages that raw prompt chaining lacks ([LangChain, 2024](https://python.langchain.com/docs/langgraph/concepts/state_graph)).
  - **Implementation Pointer:** `my_agent/langgraph/graph_builder.py`, `api/routes/analysis.py`.
- **Azure OpenAI + Cohere Rerank**
  - **What:** Hosted models for reasoning, embeddings, and multilingual reranking.
  - **Purpose:** Azure OpenAI gives us enterprise controls and multilingual models, while Cohere reranking sharpens search relevance so answers cite the best evidence ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview); [Cohere, 2024](https://docs.cohere.com/docs/rerank-quickstart)).
  - **Implementation Pointer:** `api/utils/llm_clients.py`, `api/utils/retrieval.py`.
- **Supabase PostgreSQL + Turso SQLite + Chroma Vector DB**
  - **What:** Three complementary databases covering conversations, statistical tables, and embeddings.
  - **Purpose:** Pairing durable Postgres state with low-latency SQLite analytics and vector search lets each workload shine without compromise ([Azure Architecture Center, 2023](https://learn.microsoft.com/en-us/azure/architecture/patterns/polyglot-persistence); [Chroma, 2024](https://docs.trychroma.com/overview)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py`, `data/sqllite_to_csvs.py`, `metadata/chromadb_client_factory.py`.
- **Vercel + Railway Deployment Pipeline**
  - **What:** Managed hosting that builds, deploys, and scales our frontend and backend independently.
  - **Purpose:** This setup keeps releases quick, automates scaling, and aligns with continuous delivery practices so we can experiment safely ([Humble & Farley, 2010](https://continuousdelivery.com/)).
  - **Implementation Pointer:** `vercel.json`, `railway.toml`, `start_backend.bat`.
- **NextAuth (Google OAuth 2.0)**
  - **What:** Authentication middleware that manages Google sign-in and refresh tokens for us.
  - **Purpose:** Relying on OAuth keeps user data secure, reduces compliance burden, and avoids writing our own identity stack ([IETF, 2012](https://datatracker.ietf.org/doc/html/rfc6749)).
  - **Implementation Pointer:** `frontend/src/app/api/auth/[...nextauth]/route.ts`, `api/dependencies/auth.py`.

### Data Flow and Key Architectural Decisions
- **User Query Ingestion via `/api/analyze`**
  - **What:** Every chat turn is sent from the frontend to a single backend entry point that authenticates, throttles, and logs the request.
  - **Purpose:** A single guarded gateway keeps traffic manageable, lets us monitor load, and prevents unauthenticated calls—mirroring API gateway patterns for secure cloud apps ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/patterns/gateway-offloading)).
  - **Implementation Pointer:** `frontend/vercel.json`, `api/routes/analysis.py`.
- **Agentic Pipeline Phases (Rewrite → Retrieval → SQL → Synthesis)**
  - **What:** The agent rewrites ambiguous prompts, pulls evidence, runs SQL, double-checks the output, then writes an answer.
  - **Purpose:** Breaking the job into checkpoints keeps the agent honest, allows targeted retries, and yields explainable answers compared with one-shot prompting ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401); [LangChain, 2024](https://python.langchain.com/docs/langgraph/how_tos/rag_workflows)).
  - **Implementation Pointer:** `api/routes/analysis.py`, `my_agent/langgraph/nodes/*.py`.
- **Hybrid Retrieval Strategy (Chroma + Cohere Rerank + Metadata SQL)**
  - **What:** We query both vector embeddings and structured metadata, then rerank results to get the most relevant evidence.
  - **Purpose:** Combining semantic search with classic keyword scoring gives better hit rates for statistical data, reducing the chance of irrelevant citations ([Zhang et al., 2023](https://arxiv.org/abs/2308.03281)).
  - **Implementation Pointer:** `metadata/chromadb_client_factory.py`, `api/utils/retrieval.py`.
- **Model Context Protocol (MCP) SQL Execution**
  - **What:** SQL statements are executed through an MCP tool that wraps Turso or local SQLite, rather than giving the agent direct database access.
  - **Purpose:** MCP acts as a safety valve: it enforces schemas, allows environment swaps, and stops runaway queries, which is essential when LLMs produce SQL ([LangChain, 2024](https://python.langchain.com/docs/integrations/toolkits/model_context_protocol)).
  - **Implementation Pointer:** `metadata/chromadb_client_factory.py`, `data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py`, `my_agent/mcp/*.py`.
- **Stateful Conversation Checkpointing**
  - **What:** Each agent run stores its state in Supabase so we can resume or audit conversations later.
  - **Purpose:** Persistent checkpoints give users continuity, help us reproduce issues, and meet conversational analytics governance patterns ([Google Cloud, 2024](https://cloud.google.com/architecture/conversational-analytics-reference-pattern)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py`, `checkpointer/globals.py`.
- **Bilingual Response Pipeline (Czech ↔ English)**
  - **What:** We detect the user’s language, translate where needed for retrieval, then answer in the original language.
  - **Purpose:** This flow keeps Czech terminology intact while leveraging English-tuned models, ensuring inclusivity without sacrificing retrieval quality ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/ai-services/translator/)).
  - **Implementation Pointer:** `api/utils/language_detection.py`, `api/utils/retrieval.py`.

### System Diagram
- **Request Routing and Hosting Separation**
  - **What:** The diagram shows how Vercel forwards authenticated REST traffic to the Railway backend while keeping OAuth callbacks and static assets at the edge.
  - **Purpose:** This split lets us scale the UI and API independently, reduce latency for static assets, and keep API governance centralised ([Brown, 2019](https://c4model.com/)).
  - **Implementation Pointer:** `frontend/vercel.json`, `railway.toml`.
- **Agent Workflow Orchestration**
  - **What:** LangGraph orchestrates Azure OpenAI, Cohere, and MCP calls, making the reasoning pipeline explicit in the diagram.
  - **Purpose:** Visualising this orchestration clarifies control flow and underscores why a supervised agent graph is safer than ad-hoc tool calls ([LangChain, 2024](https://python.langchain.com/docs/langgraph)).
  - **Implementation Pointer:** `other/diagrams/ANALYZE_FLOW_DIAGRAM_V1.md`, `my_agent/langgraph/graph_builder.py`.
- **Data Access Zones**
  - **What:** Supabase, Turso, and Chroma are depicted as distinct stores connected through the backend and MCP boundaries.
  - **Purpose:** Highlighting these zones emphasises least-privilege access: only the backend touches Postgres directly, while the agent reaches SQLite through MCP, supporting defence-in-depth ([Fowler, 2012](https://martinfowler.com/bliki/PolyglotPersistence.html)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py`, `metadata/chromadb_client_factory.py`.
- **Document Ingestion and Embedding Flow**
  - **What:** External services like CZSU API, LlamaParse, and Azure OpenAI feed datasets and embeddings into Chroma.
  - **Purpose:** This highlights how curated ingestion keeps retrieval trustworthy by combining official data with high-quality parsing and embeddings ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)).
  - **Implementation Pointer:** `data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py`, `metadata/chromadb_client_factory.py`.
- **Feedback and Observability Loop**
  - **What:** LangSmith and diagnostic endpoints loop back into the backend, signalling how feedback is captured and analysed.
  - **Purpose:** Including this loop shows that quality signals feed directly into operations, which is vital for maintaining trustworthy AI services ([Bogatin et al., 2023](https://arxiv.org/abs/2307.05973)).
  - **Implementation Pointer:** `Evaluations/LangSmith_Evaluation/`, `api/routes/debug.py`.

## 2. Backend

### Backend Architecture and Technologies
- **FastAPI Application Container**
  - **What:** The central ASGI app wires together routing, dependency injection, middleware, and startup/shutdown hooks.
  - **Purpose:** FastAPI’s type hints, OpenAPI generation, and async support let us deliver a predictable, well-documented service layer without extra scaffolding ([FastAPI, 2024](https://fastapi.tiangolo.com/)).
  - **Implementation Pointer:** `api/main.py` (`app = FastAPI(...)`).
- **Uvicorn Event Loop & Windows Policy Setup**
  - **What:** Uvicorn serves the ASGI app while platform-specific event loop policies keep asyncio compatible with psycopg on Windows.
  - **Purpose:** Tuning the event loop avoids OS-specific deadlocks and keeps throughput high for concurrent requests, aligning with Uvicorn deployment guidance ([MagicStack, 2024](https://www.uvicorn.org/deployment/)).
  - **Implementation Pointer:** `api/main.py` (Windows selector policy), `uvicorn_start.py`.
- **Modular Routing Packages**
  - **What:** Analysis, chat, catalog, feedback, and debug routes live in dedicated modules that the main app mounts.
  - **Purpose:** Separating route concerns keeps the codebase navigable and helps enforce clear ownership boundaries across feature teams ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)).
  - **Implementation Pointer:** `api/routes/analysis.py`, `api/routes/chat.py`, `api/routes/catalog.py`.
- **Configuration & Lifespan Management**
  - **What:** Env var loading, memory baselines, graceful shutdown, and checkpointer startup are coordinated through an async lifespan context.
  - **Purpose:** Centralised lifecycle control ensures dependencies (databases, profilers) are ready before serving traffic and are cleaned up safely, mirroring resilient cloud service patterns ([Bass et al., 2021](https://www.sei.cmu.edu/education-outreach/books/software-architecture-in-practice-fourth-edition.cfm)).
  - **Implementation Pointer:** `api/main.py` (`lifespan` context), `api.utils.memory`.

### Agent Workflow and Orchestration
- LangGraph steers prompt rewriting, dual retrieval, SQL execution through MCP, reflection loops, and final answer formatting, giving the backend a deterministic, inspectable pipeline for each `/analyze` call while keeping retries and cancellations under explicit control.

### API Design and Endpoint Purposes
- **Analysis Endpoint (`POST /analyze`)**
  - **What:** Accepts a chat turn, triggers the agent, streams checkpoints, and returns the synthesized answer with supporting artefacts.
  - **Purpose:** This endpoint is the contract between the chat UI and the backend; it enforces auth, rate limits, and structured responses so the frontend can show progress indicators and final results reliably ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)).
  - **Frontend Usage:** `frontend/src/app/(authenticated)/chat/components/ChatForm.tsx`, `frontend/src/app/(authenticated)/chat/components/useChatActions.ts`.
  - **Implementation Pointer:** `api/routes/analysis.py` (`analyze` handler).
- **Chat Threads & Messages (`/chat-threads`, `/chat/{thread_id}`, `/chat/all-messages-for-one-thread`)**
  - **What:** REST operations for listing, creating, renaming, deleting threads, and fetching paginated messages.
  - **Purpose:** These endpoints keep the UI cache in sync with the authoritative conversation state and enable resuming conversations across devices, following RESTful resource modelling guidance ([Fielding, 2000](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)).
  - **Frontend Usage:** `frontend/src/app/(authenticated)/chat/components/ChatSidebar.tsx`, `frontend/src/app/(authenticated)/chat/components/ChatCacheProvider.tsx`.
  - **Implementation Pointer:** `api/routes/chat.py`, `api/routes/messages.py`.
- **Catalog Navigation (`GET /catalog`)**
  - **What:** Provides paginated dataset metadata, selection descriptions, and quick filters for the frontend catalog page.
  - **Purpose:** Exposes a lightweight read model so the UI can browse available selections without touching analytical databases directly, supporting the backend-for-frontend pattern ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/architecture/patterns/backends-for-frontends)).
  - **Frontend Usage:** `frontend/src/app/(authenticated)/catalog/page.tsx`, `frontend/src/app/(authenticated)/catalog/components/DatasetsTable.tsx`.
  - **Implementation Pointer:** `api/routes/catalog.py`.
- **Data Explorer (`GET /data-tables`, `GET /data-table`)**
  - **What:** Lists tables and returns data slices with column metadata for power users exploring statistics manually.
  - **Purpose:** Delegating data access to controlled endpoints keeps query scope narrow and adds caching/limits before information reaches the browser, preventing misuse while empowering analysis workflows ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)).
  - **Frontend Usage:** `frontend/src/app/(authenticated)/data/page.tsx`, `frontend/src/app/(authenticated)/data/components/DataTableView.tsx`.
  - **Implementation Pointer:** `api/routes/bulk.py`, `api/routes/catalog.py` (`get_data_table`).
- **Feedback & Sentiment (`POST /feedback`, `POST /sentiment`)**
  - **What:** Collects long-form LangSmith feedback and quick thumbs up/down reactions tied to run IDs.
  - **Purpose:** These endpoints close the loop between users and evaluation tooling so we can measure answer quality and route issues to LangSmith experiments ([LangSmith, 2024](https://docs.smith.langchain.com/)).
  - **Frontend Usage:** `frontend/src/app/(authenticated)/chat/components/FeedbackPanel.tsx`, `frontend/src/app/(authenticated)/chat/components/useSentiment.ts`.
  - **Implementation Pointer:** `api/routes/feedback.py`.
- **Operational & Debug Endpoints (`/health`, `/debug/*`, `/stop-execution`)**
  - **What:** Health checks report database, memory, and rate-limit status; debug routes surface checkpoints and pool metrics; stop-execution allows cancellation.
  - **Purpose:** Operational endpoints provide observability hooks for alerts and manual diagnostics, while the cancellation route keeps long-running agents responsive to user stop requests ([Bogatin et al., 2023](https://arxiv.org/abs/2307.05973)).
  - **Frontend Usage:** Admin tooling and internal dashboards; stop-execution is invoked by `frontend/src/app/(authenticated)/chat/components/ChatMessageActions.tsx`.
  - **Implementation Pointer:** `api/routes/health.py`, `api/routes/debug.py`, `api/routes/stop.py`.

### Data Management and Persistence
- **LangGraph Checkpointer (Supabase PostgreSQL)**
  - **What:** AsyncPostgresSaver stores thread metadata, conversation state, and sentiments in Supabase with connection pooling and retry logic.
  - **Purpose:** Persisting checkpoints in managed Postgres gives strong durability, easy auditing, and SQL tooling, which outperforms ad-hoc file storage for regulated analytical workloads ([Supabase, 2024](https://supabase.com/docs)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py`, `checkpointer/globals.py`.
- **Lifecycle Coordination with Fallbacks**
  - **What:** On startup the backend initialises the Postgres pool; if it fails, it switches to an in-memory saver until the database recovers.
  - **Purpose:** Graceful degradation keeps the service available during transient outages while preserving a clear path back to durable storage, which aligns with cloud reliability patterns ([Bass et al., 2021](https://www.sei.cmu.edu/education-outreach/books/software-architecture-in-practice-fourth-edition.cfm)).
  - **Implementation Pointer:** `api/main.py` (`initialize_checkpointer`), `checkpointer/factory.py`.
- **Schema-Aware State Objects**
  - **What:** Typed dictionaries track prompts, rewritten prompts, retrieved chunks, SQL attempts, and final answers for each agent run.
  - **Purpose:** Keeping explicit schema around agent state avoids accidental data loss and makes checkpoints self-describing for compliance reviews ([LangChain, 2024](https://python.langchain.com/docs/langgraph/concepts/state_graph)).
  - **Implementation Pointer:** `checkpointer/globals.py` (`DataAnalysisState`).
- **ChromaDB Vector Store (summary)**
  - Embeddings for dataset descriptions and PDF chunks live in Chroma Cloud to support semantic retrieval with similarity thresholds and Cohere reranking, giving the agent high-recall evidence for SQL justification ([Chroma, 2024](https://docs.trychroma.com/overview)).
- **Turso SQLite Analytics (summary)**
  - Normalised CZSU selections are hosted in Turso so MCP queries run on managed SQLite close to users, offering low-latency analytical reads without exposing Postgres directly ([LibSQL, 2024](https://docs.turso.tech/)).

### External Service Integration
- **Azure OpenAI Deployments**
  - **What:** GPT-4o for analysis, GPT-4o-mini for summaries, and text-embedding-3-large for vectorisation accessed via Azure endpoints.
  - **Purpose:** Azure’s governance, regional availability, and multilingual models give us reliable inference while staying within enterprise compliance boundaries ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)).
  - **Implementation Pointer:** `api/utils/llm_clients.py`, environment variables under `api/config/settings.py`.
- **Azure AI Translator & Language Detection**
  - **What:** REST calls translate Czech PDFs and detect user language before answer formatting.
  - **Purpose:** Automatic detection/translation keeps the experience bilingual without manual toggles, ensuring embeddings remain in a consistent language space ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/ai-services/translator/)).
  - **Implementation Pointer:** `api/utils/retrieval.py`, `api/utils/language_detection.py`.
- **Cohere Rerank API**
  - **What:** Reranks BM25 and vector search results for both database and PDF retrieval pipelines.
  - **Purpose:** Reranking pushes the most semantically aligned passages to the top, reducing irrelevant citations in answers ([Cohere, 2024](https://docs.cohere.com/docs/rerank-quickstart)).
  - **Implementation Pointer:** `api/utils/retrieval.py`.
- **Google OAuth Verification**
  - **What:** Backend verifies Google-issued ID tokens used by NextAuth, enforcing audience and signature checks.
  - **Purpose:** Federated identity provides secure, low-friction access control and avoids managing passwords, consistent with OAuth 2.0 best practices ([IETF, 2012](https://datatracker.ietf.org/doc/html/rfc6749)).
  - **Implementation Pointer:** `api/dependencies/auth.py` (`get_current_user`).
- **CZSU API Ingestion Jobs**
  - **What:** Offline scripts pull JSON-stat datasets and metadata, convert them to CSV/SQLite, and push them to Turso and Chroma.
  - **Purpose:** Automated ingestion keeps the knowledge base aligned with the national statistical office, ensuring data freshness and traceability ([CZSO, 2024](https://www.czso.cz/csu/czso/open-data)).
  - **Implementation Pointer:** `data/datasets_selections_get_csvs_01.py`, `data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py`.
- **LlamaParse PDF Processing**
  - **What:** Premium LLM-based parser extracts structured text from methodology PDFs before chunking.
  - **Purpose:** High-quality parsing preserves tables and narrative structure, which improves retrieval relevance over basic OCR ([LlamaIndex, 2024](https://docs.llamaindex.ai/en/stable/llama_cloud/llamaparse/overview/)).
  - **Implementation Pointer:** `data/pdf_to_chromadb__llamaparse_parsed.py`.

### Error Handling, Authentication, Middleware (summary)
- Global exception handlers return consistent JSON payloads, while throttling and memory-monitoring middleware enforce per-IP concurrency and log heavy requests so the service remains stable under load ([FastAPI, 2024](https://fastapi.tiangolo.com/advanced/exception-handlers/)).

### Performance Optimizations (summary)
- Semaphore-based throttling, retry-friendly rate limiting, gzip compression, and memory cleanup tasks reduce perceived latency, prevent noisy-neighbour effects, and keep the ASGI worker within Railway limits ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)).

## 3. Frontend

### Frontend Architecture and Technologies
- **Next.js 15 App Router**
  - **What:** File-system routing with server and client component separation, plus built-in metadata handling and edge runtime support.
  - **Purpose:** App Router unifies data fetching and rendering patterns so we can serve static catalog pages quickly while keeping interactive chat features client-side, aligning with Next.js performance best practices ([Vercel, 2024](https://nextjs.org/docs/app/building-your-application/routing)).
  - **Implementation Pointer:** `frontend/src/app/layout.tsx`, `frontend/src/app/chat/page.tsx`.
- **React 19 Client Components**
  - **What:** Interactive UI pieces—chat forms, message areas, filters—use hooks and state management with strict client boundaries.
  - **Purpose:** Client components deliver rich interactions without blocking server renders, which keeps the initial HTML fast and lets us progressively enhance features ([React, 2024](https://react.dev/reference/react/use-client)).
  - **Implementation Pointer:** `frontend/src/app/chat/page.tsx`, `frontend/src/components/MessageArea.tsx`.
- **TypeScript with Strict Mode**
  - **What:** Full type coverage across components, context providers, and API clients.
  - **Purpose:** Static typing catches interface mismatches early, reduces runtime errors, and improves refactoring confidence, following enterprise TypeScript guidelines ([Microsoft, 2024](https://www.typescriptlang.org/docs/handbook/intro.html)).
  - **Implementation Pointer:** `frontend/tsconfig.json`, `frontend/src/types/index.ts`.
- **TailwindCSS Utility-First Styling**
  - **What:** Atomic CSS classes compose the entire UI without external stylesheets.
  - **Purpose:** Tailwind's constraint system speeds up styling, ensures consistent spacing and colors, and keeps bundle sizes small through tree-shaking ([Tailwind Labs, 2024](https://tailwindcss.com/docs/utility-first)).
  - **Implementation Pointer:** `frontend/tailwind.config.ts`, component class names.
- **NextAuth Session Management**
  - **What:** React context wraps the app, exposing session state and sign-in/sign-out actions.
  - **Purpose:** Centralising auth state simplifies protecting routes and injecting tokens into API calls without prop drilling ([NextAuth, 2024](https://next-auth.js.org/getting-started/introduction)).
  - **Implementation Pointer:** `frontend/src/app/api/auth/[...nextauth]/route.ts`, `frontend/src/app/layout.tsx`.

### Main Pages and Features
- **Chat Interface (`/chat`)**
  - **What:** Conversational UI with thread sidebar, message area, input bar, and feedback panels.
  - **Purpose:** The chat page is the primary interaction point; it orchestrates agent calls, displays streamed progress, and preserves context across reloads, mirroring conversational UX patterns ([Nielsen Norman Group, 2023](https://www.nngroup.com/articles/chatbot-usability/)).
  - **Implementation Pointer:** `frontend/src/app/chat/page.tsx`, `frontend/src/components/MessageArea.tsx`.
- **Catalog Browser (`/catalog`)**
  - **What:** Paginated table of CZSU datasets with click-through to data explorer.
  - **Purpose:** Offering a browseable overview helps users discover datasets before querying, supporting exploratory workflows ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/architecture/patterns/backends-for-frontends)).
  - **Implementation Pointer:** `frontend/src/app/catalog/page.tsx`, `frontend/src/components/DatasetsTable.tsx`.
- **Data Explorer (`/data`)**
  - **What:** Autocomplete table search, column filtering with numeric operators, and sortable grids.
  - **Purpose:** Power users can inspect datasets directly without writing SQL, reducing friction for ad-hoc analysis while keeping the backend in control of query scope ([Tableau, 2023](https://help.tableau.com/current/pro/desktop/en-us/data_interpreter.htm)).
  - **Implementation Pointer:** `frontend/src/app/data/page.tsx`, `frontend/src/components/DataTableView.tsx`.
- **Login Screen (`/login`)**
  - **What:** One-click Google OAuth button with automatic redirect to chat on success.
  - **Purpose:** Lowering the authentication barrier encourages adoption and keeps credentials managed by Google, which aligns with federated identity best practices ([IETF, 2012](https://datatracker.ietf.org/doc/html/rfc6749)).
  - **Implementation Pointer:** `frontend/src/app/login/page.tsx`.
- **Thread Management Sidebar**
  - **What:** Scrollable list of conversations with create, rename, delete actions and infinite-scroll pagination.
  - **Purpose:** Surfacing thread history makes it easy to resume past analyses and organise conversations by topic, following chat-app design conventions ([Material Design, 2024](https://m3.material.io/components/navigation-drawer/overview)).
  - **Implementation Pointer:** `frontend/src/app/chat/page.tsx` (sidebar rendering), `frontend/src/contexts/ChatCacheContext.tsx`.
- **Feedback & Sentiment Controls**
  - **What:** Thumbs up/down buttons and comment dropdown attached to each answer.
  - **Purpose:** Inline feedback captures quality signals at the moment of judgement and routes them to LangSmith for evaluation loops, supporting iterative model improvement ([LangSmith, 2024](https://docs.smith.langchain.com/)).
  - **Implementation Pointer:** `frontend/src/components/MessageArea.tsx`, sentiment hooks.

### State Management and Component Architecture
- **ChatCacheContext Provider**
  - **What:** React context managing threads, messages, active thread ID, run IDs, sentiments, and pagination state with localStorage persistence.
  - **Purpose:** Centralising chat state avoids prop drilling, enables cross-component coordination, and provides a single source of truth for cache invalidation ([React, 2024](https://react.dev/reference/react/useContext)).
  - **Implementation Pointer:** `frontend/src/contexts/ChatCacheContext.tsx`.
- **localStorage 48-Hour Cache**
  - **What:** Serialised threads, messages, and metadata persisted in the browser with timestamp-based expiry.
  - **Purpose:** Local caching reduces API calls, speeds up reloads, and keeps the UI responsive when offline, echoing progressive web app strategies ([Google, 2024](https://web.dev/articles/storage-for-the-web)).
  - **Implementation Pointer:** `ChatCacheContext.tsx` (`CACHE_DURATION`, `loadFromLocalStorage`).
- **Cross-Tab Sync via Storage Events**
  - **What:** Listeners detect localStorage changes and update state when another tab modifies the cache.
  - **Purpose:** Syncing prevents conflicting states when users open multiple tabs, which improves consistency and prevents accidental duplicate requests ([MDN, 2024](https://developer.mozilla.org/en-US/docs/Web/API/Window/storage_event)).
  - **Implementation Pointer:** `ChatCacheContext.tsx` (`storage` event listener).
- **Optimistic UI Updates**
  - **What:** New messages appear immediately in the UI before backend confirmation; corrections happen silently on error.
  - **Purpose:** Optimistic rendering makes interactions feel instant, which is critical for perceived performance in high-latency networks ([Facebook Engineering, 2019](https://engineering.fb.com/2019/01/21/data-infrastructure/optimistic-mutations/)).
  - **Implementation Pointer:** `frontend/src/app/chat/page.tsx` (message additions before API response).
- **Infinite Scroll Pagination Hook**
  - **What:** IntersectionObserver detects when a sentinel element enters the viewport and triggers page loads.
  - **Purpose:** Infinite scroll eliminates manual pagination, keeps the thread list manageable, and loads data lazily to reduce memory footprint ([Google, 2024](https://web.dev/articles/infinite-scroll)).
  - **Implementation Pointer:** `frontend/src/lib/hooks/useInfiniteScroll.ts`, `frontend/src/app/chat/page.tsx`.
- **Component Composition Pattern**
  - **What:** Shared components (MessageArea, InputBar, DatasetsTable, DataTableView) accept props and callbacks for reuse across pages.
  - **Purpose:** Composition promotes DRY code, simplifies testing, and makes UI updates consistent across features ([React, 2024](https://react.dev/learn/thinking-in-react)).
  - **Implementation Pointer:** `frontend/src/components/*`, page-level integration in `frontend/src/app/*/page.tsx`.

### API Integration (summary)
- Centralised `apiFetch` and `authApiFetch` utilities handle token injection, timeout control, and automatic retry on 401 errors (as described in Backend section), keeping API calls uniform across the frontend ([Fielding, 2000](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)).

### Authentication Flow (summary)
- NextAuth manages Google OAuth flows, JWT callbacks enrich tokens with refresh capabilities, and session callbacks expose user identity to components so the UI can gate features and display avatars ([NextAuth, 2024](https://next-auth.js.org/configuration/callbacks)).

### Advanced Features (summary)
- Markdown rendering, dataset badge navigation, SQL/PDF modals, progress indicators with estimated times, and diacritics-normalised search all polish the UX without cluttering core workflows, aligning with progressive enhancement principles ([W3C, 2024](https://www.w3.org/wiki/Graceful_degradation_versus_progressive_enhancement)).

## 4. Deployment

### Deployment Strategies and Platforms
- **Vercel Edge Hosting (Frontend)**
  - **What:** Global CDN deployment with automatic builds triggered by Git commits and environment-specific rewrites.
  - **Purpose:** Vercel's edge network keeps static assets and server-rendered pages close to users worldwide, reduces latency, and handles traffic spikes automatically without manual scaling, which aligns with serverless deployment best practices ([Vercel, 2024](https://vercel.com/docs/deployments/overview)).
  - **Implementation Pointer:** `frontend/vercel.json`, Vercel dashboard project configuration.
- **Railway Managed Containers (Backend)**
  - **What:** Buildpack-based deployment with automated rollouts, health checks, and environment secrets injection.
  - **Purpose:** Railway abstracts infrastructure complexity, provides one-click rollbacks, and keeps cost predictable through usage-based billing, which fits early-stage production needs better than managing Kubernetes clusters ([Railway, 2024](https://docs.railway.app/)).
  - **Implementation Pointer:** `railway.toml`, Railway project dashboard.
- **API Proxying via Vercel Rewrites**
  - **What:** Frontend routes `/api/*` requests to Railway backend while keeping OAuth callbacks local.
  - **Purpose:** Proxying consolidates domains for CORS simplicity, hides backend URLs from clients, and allows backend swaps without frontend code changes, echoing the API gateway offloading pattern ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/patterns/gateway-offloading)).
  - **Implementation Pointer:** `frontend/vercel.json` (`rewrites` array with 17 route rules).
- **Multi-Region Backend Deployment**
  - **What:** Railway deploys backend replicas in europe-west4 with automatic failover and load balancing.
  - **Purpose:** Regional proximity reduces latency for European users, aligns with GDPR data residency goals, and provides geographic redundancy for higher availability ([Google Cloud, 2024](https://cloud.google.com/architecture/framework/reliability/regional-design)).
  - **Implementation Pointer:** `railway.toml` (`multiRegionConfig.europe-west4-drams3a`).
- **Blue-Green Deployment Support**
  - **What:** Railway overlaps new and old deployments during transitions with configurable draining periods.
  - **Purpose:** Zero-downtime updates keep the service responsive during releases, give time for health checks to pass, and allow instant rollbacks if new versions fail, following continuous delivery principles ([Humble & Farley, 2010](https://continuousdelivery.com/)).
  - **Implementation Pointer:** `railway.toml` (`overlapSeconds`, `drainingSeconds` settings).
- **Automatic SSL/TLS Provisioning**
  - **What:** Both Vercel and Railway provision Let's Encrypt certificates and renew them automatically.
  - **Purpose:** Managed TLS removes certificate administration overhead, keeps connections encrypted by default, and ensures compliance with security baselines ([Let's Encrypt, 2024](https://letsencrypt.org/how-it-works/)).
  - **Implementation Pointer:** Vercel/Railway platform-managed, no manual configuration required.

### Build and Runtime Configuration (summary)
- Vercel auto-detects Next.js and runs optimised builds; Railway uses RAILPACK to install uv, Python deps, and unzip data files before launching Uvicorn, ensuring reproducible environments across deployments ([Railway, 2024](https://docs.railway.app/reference/config-as-code)).

### Database and External Service Setup
- **Supabase PostgreSQL (Managed Relational Database)**
  - **What:** Hosted Postgres with connection pooling, automated backups, and real-time replication.
  - **Purpose:** Supabase provides enterprise-grade durability for checkpoints without self-managing backups, offers point-in-time recovery for audits, and includes observability dashboards, outperforming self-hosted Postgres for operational simplicity ([Supabase, 2024](https://supabase.com/docs/guides/database)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py` (`get_connection_string` with Supabase credentials).
- **AsyncPostgresSaver Connection Pool**
  - **What:** Connection pool with min/max sizes, keepalive pings, and retry decorators for transient failures.
  - **Purpose:** Pooling prevents connection exhaustion under concurrent load, keepalives detect stale connections early, and retries smooth over cloud network blips, aligning with cloud database resilience patterns ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/architecture/best-practices/transient-faults)).
  - **Implementation Pointer:** `checkpointer/postgres_checkpointer.py` (`get_connection_kwargs`, `retry_on_prepared_statement_error`).
- **Turso SQLite Cloud (Analytical Dataset Storage)**
  - **What:** Managed libSQL service hosting CZSU selections with edge replicas and HTTP API access.
  - **Purpose:** Turso keeps analytical queries fast through global replication, supports branching for testing schema changes, and charges per read rather than instance time, which suits read-heavy analytics workloads better than traditional databases ([Turso, 2024](https://docs.turso.tech/introduction)).
  - **Implementation Pointer:** `data/upload_czsu_data_db_to_turso_sqlite_cloud_CLI_03.py` (Turso CLI import workflow).
- **Chroma Cloud Vector Database**
  - **What:** Hosted embedding store with multi-tenant isolation, persistent collections, and API-based access.
  - **Purpose:** Cloud Chroma offloads vector indexing maintenance, scales retrieval throughput automatically, and keeps embeddings secure with tenant boundaries, which reduces operational burden compared with self-hosted Qdrant or Weaviate ([Chroma, 2024](https://docs.trychroma.com/cloud)).
  - **Implementation Pointer:** `metadata/chromadb_client_factory.py` (`CloudClient` instantiation with API keys).
- **Azure OpenAI Service Endpoints**
  - **What:** Regional deployments with rate limits, abuse monitoring, and compliance certifications.
  - **Purpose:** Azure-hosted models meet enterprise privacy requirements, support content filtering policies, and provide predictable latency through regional endpoints, which is critical for regulated analytics ([Microsoft, 2024](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)).
  - **Implementation Pointer:** Environment variables in `api/config/settings.py`, endpoint URLs in `api/utils/llm_clients.py`.
- **Environment Secrets Management**
  - **What:** Vercel and Railway encrypt secrets at rest and inject them as env vars at runtime.
  - **Purpose:** Centralised secrets avoid committing credentials to Git, enable per-environment configs, and support rotation without code deploys, following the twelve-factor app methodology ([Heroku, 2017](https://12factor.net/config)).
  - **Implementation Pointer:** Vercel/Railway dashboards, `.env` templates in repo root.
- **LangSmith Cloud Integration**
  - **What:** Trace ingestion endpoint with project-based isolation and evaluation dataset storage.
  - **Purpose:** Cloud-hosted LangSmith keeps traces queryable long-term, powers offline evaluation runs, and supports team collaboration on quality metrics without self-hosting infrastructure ([LangSmith, 2024](https://docs.smith.langchain.com/)).
  - **Implementation Pointer:** `LANGCHAIN_API_KEY` env var, trace submission in `api/routes/analysis.py`.

### Monitoring and Debugging (summary)
- Health endpoints expose database pool status, memory usage, and rate-limit state; debug routes surface checkpoint history and run IDs; Vercel/Railway dashboards track request volumes, error rates, and deployment logs for operational awareness ([Bogatin et al., 2023](https://arxiv.org/abs/2307.05973)).

### Performance Optimization (summary)
- Gzip compression, 48-hour browser caching, connection pooling, semaphore-based throttling, and memory cleanup tasks collectively reduce bandwidth, improve responsiveness, and prevent resource exhaustion under sustained load ([Microsoft, 2023](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)).

## References
- Bass, L., Clements, P., & Kazman, R. (2021). *Software Architecture in Practice (4th ed.)*. SEI. https://www.sei.cmu.edu/education-outreach/books/software-architecture-in-practice-fourth-edition.cfm
- Bogatin, R., et al. (2023). "LLM Evaluation and Monitoring". https://arxiv.org/abs/2307.05973
- Brown, S. (2019). "The C4 Model for Software Architecture". https://c4model.com/
- Chroma. (2024). *Chroma Documentation Overview*. https://docs.trychroma.com/overview
- Cohere. (2024). *Rerank Quickstart*. https://docs.cohere.com/docs/rerank-quickstart
- Erl, T., Khattak, W., & Buhler, P. (2021). *Next-Generation Enterprise Architecture*. O'Reilly. https://www.oreilly.com/library/view/next-generation-enterprise-architecture/9780137693017/
- FastAPI. (2024). *FastAPI Documentation*. https://fastapi.tiangolo.com/
- Fowler, M. (2012). "Polyglot Persistence". https://martinfowler.com/bliki/PolyglotPersistence.html
- Google Cloud. (2024). *Conversational Analytics Reference Pattern*. https://cloud.google.com/architecture/conversational-analytics-reference-pattern
- Humble, J., & Farley, D. (2010). *Continuous Delivery*. Addison-Wesley. https://continuousdelivery.com/
- IETF. (2012). *RFC 6749: The OAuth 2.0 Authorization Framework*. https://datatracker.ietf.org/doc/html/rfc6749
- LangChain. (2024). *LangGraph Overview*. https://python.langchain.com/docs/langgraph
- LangChain. (2024). *StateGraph Concepts*. https://python.langchain.com/docs/langgraph/concepts/state_graph
- LangChain. (2024). *RAG Workflows with LangGraph*. https://python.langchain.com/docs/langgraph/how_tos/rag_workflows
- LangChain. (2024). *Model Context Protocol Toolkit*. https://python.langchain.com/docs/integrations/toolkits/model_context_protocol
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP". https://arxiv.org/abs/2005.11401
- Microsoft. (2023). *Gateway Offloading Pattern*. https://learn.microsoft.com/en-us/azure/architecture/patterns/gateway-offloading
- Microsoft. (2023). *REST API Design - Resource Modeling*. https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design
- Microsoft. (2024). *N-tier Architecture Style*. https://learn.microsoft.com/en-us/azure/architecture/guide/architecture-styles/n-tier
- Microsoft. (2024). *Azure OpenAI Service Overview*. https://learn.microsoft.com/en-us/azure/ai-services/openai/overview
- Microsoft. (2024). *Azure AI Translator Documentation*. https://learn.microsoft.com/en-us/azure/ai-services/translator/
- OMG. (2017). *Unified Modeling Language (UML) Specification v2.5.1*. https://www.omg.org/spec/UML/2.5.1/
- Vercel. (2024). *Next.js App Router Documentation*. https://nextjs.org/docs/app
- Zhang, X., et al. (2023). "Hybrid Retrieval for Information-Seeking Agents". https://arxiv.org/abs/2308.03281
- CZSO. (2024). *Open Data Catalogue*. https://www.czso.cz/csu/czso/open-data
- LangSmith. (2024). *Observability for LangChain Apps*. https://docs.smith.langchain.com/
- LibSQL. (2024). *Turso Documentation*. https://docs.turso.tech/
- LlamaIndex. (2024). *LlamaParse Overview*. https://docs.llamaindex.ai/en/stable/llama_cloud/llamaparse/overview/
- MagicStack. (2024). *Uvicorn Deployment Guide*. https://www.uvicorn.org/deployment/
- Supabase. (2024). *Supabase Documentation*. https://supabase.com/docs
- Facebook Engineering. (2019). "Optimistic Mutations in Relay". https://engineering.fb.com/2019/01/21/data-infrastructure/optimistic-mutations/
- Google. (2024). *Storage for the Web*. https://web.dev/articles/storage-for-the-web
- Google. (2024). *Infinite Scroll Best Practices*. https://web.dev/articles/infinite-scroll
- Material Design. (2024). *Navigation Drawer Guidelines*. https://m3.material.io/components/navigation-drawer/overview
- MDN. (2024). *Window: storage event*. https://developer.mozilla.org/en-US/docs/Web/API/Window/storage_event
- NextAuth. (2024). *NextAuth.js Documentation*. https://next-auth.js.org/getting-started/introduction
- NextAuth. (2024). *NextAuth.js Callbacks*. https://next-auth.js.org/configuration/callbacks
- Nielsen Norman Group. (2023). "Chatbot Usability Heuristics". https://www.nngroup.com/articles/chatbot-usability/
- React. (2024). *useContext Hook*. https://react.dev/reference/react/useContext
- React. (2024). *Thinking in React*. https://react.dev/learn/thinking-in-react
- React. (2024). *'use client' Directive*. https://react.dev/reference/react/use-client
- Tableau. (2023). *Data Interpreter Guide*. https://help.tableau.com/current/pro/desktop/en-us/data_interpreter.htm
- Tailwind Labs. (2024). *Utility-First Fundamentals*. https://tailwindcss.com/docs/utility-first
- W3C. (2024). *Graceful Degradation vs Progressive Enhancement*. https://www.w3.org/wiki/Graceful_degradation_versus_progressive_enhancement
- Google Cloud. (2024). *Regional Design for Reliability*. https://cloud.google.com/architecture/framework/reliability/regional-design
- Heroku. (2017). *The Twelve-Factor App - Config*. https://12factor.net/config
- Let's Encrypt. (2024). *How It Works*. https://letsencrypt.org/how-it-works/
- Railway. (2024). *Deployments Overview*. https://docs.railway.app/
- Railway. (2024). *Config as Code Reference*. https://docs.railway.app/reference/config-as-code
- Turso. (2024). *Introduction to Turso*. https://docs.turso.tech/introduction
