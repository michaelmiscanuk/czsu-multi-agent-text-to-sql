# CZSU Multi-Agent Text-to-SQL - Simplified Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(Vercel)
    participant Backend as Backend<br/>(Railway)
    participant MultiAgent as Multi-Agent<br/>(LangGraph)
    participant LLM as LLM<br/>(Azure OpenAI)
    participant VectorDB as Vector DB<br/>(Chroma)
    participant Reranker as Reranker<br/>(Cohere)
    participant MCP as MCP Server
    participant Turso as SQLite<br/>(Turso)
    participant Supabase as PostgreSQL Checkpointer<br/>(Supabase)

    User->>Frontend: Query
    Frontend->>Backend: POST /analyze

    Backend->>Supabase: Create thread
    Backend->>MultiAgent: Invoke Graph

    MultiAgent->>LLM: Embeddings
    MultiAgent->>VectorDB: Vector search
    VectorDB-->>MultiAgent: SQL Metadata, PDF Data

    MultiAgent->>Reranker: Rerank
    Reranker-->>MultiAgent: Results

    MultiAgent->>LLM: Generate SQL
    MultiAgent->>MCP: Execute SQL
    MCP->>Turso: Query data
    Turso-->>MCP: Results
    MCP-->>MultiAgent: Data

    MultiAgent->>LLM: Format answer
    MultiAgent-->>Backend: Response

    Backend->>Supabase: Save state
    Backend-->>Frontend: HTTP 200

    Frontend->>Backend: GET messages
    Backend->>Supabase: Fetch state
    Supabase-->>Backend: Messages
    Backend-->>Frontend: State
    Frontend->>User: Display
```

## Key Services
- **Frontend (Vercel)**: User interface and API proxy (Next.js)
- **Backend (Railway)**: REST API and orchestration (FastAPI)
- **Multi-Agent (LangGraph)**: Workflow orchestration and agent coordination
- **LLM (Azure OpenAI)**: Language model for embeddings, SQL generation, and answer formatting
- **Vector DB (Chroma)**: Semantic search and dataset discovery
- **Reranker (Cohere)**: Result relevance ranking and filtering
- **MCP Server**: Safe SQL execution against databases
- **SQLite (Turso)**: CZSU statistical data storage
- **PostgreSQL Checkpointer (Supabase)**: Conversation persistence and user sessions

## Flow Description
- **User Input**: User types natural language query in frontend
- **API Request**: Frontend forwards POST /analyze to backend
- **Thread Creation**: Backend creates conversation thread in Supabase
- **Multi-Agent Processing**: Multi-agent system orchestrates the analysis workflow
- **Vector Search**: LLM generates embeddings, vector DB performs semantic search
- **Result Reranking**: Reranker improves search result relevance
- **SQL Generation**: LLM generates SQL query from natural language
- **Data Execution**: MCP server runs query against Turso with CZSU data
- **Answer Formatting**: LLM formats results into natural language response
- **State Persistence**: Backend saves conversation state to Supabase
- **Response Delivery**: Results flow back through backend → frontend → user
- **UI Synchronization**: Frontend fetches authoritative state from backend/Supabase
- **Display**: Formatted answer with datasets and follow-ups shown to user