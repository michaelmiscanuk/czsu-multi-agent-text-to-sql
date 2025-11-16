from graphviz import Digraph

# Concise mindmap structure based on SUMMARY section of implementation_core_v3.md
mindmap = {
    "CZSU Multi-Agent Text-to-SQL\nImplementation Summary": {
        "1. High-Level System Architecture": {
            "Overview of System Layers and Services": {
                "Presentation Layer": "Vercel Next.js UI",
                "Application Layer": "Railway FastAPI",
                "AI Orchestration Layer": "LangGraph StateGraph + Azure OpenAI",
                "Data Persistence Layer": "Polyglot Persistence (Supabase PostgreSQL, Turso SQLite, Chroma Cloud)",
                "Integration & Identity Services": "Google OAuth, CZSU API, LlamaParse, Cohere",
                "Observability & Experimentation": "LangSmith, diagnostics",
            },
            "Technology Stack and Rationale": {
                "Frontend": "Next.js 15 + React 19",
                "Backend": "FastAPI + Uvicorn (async ASGI)",
                "AI Orchestration": "LangGraph (graph-based agents)",
                "LLM Services": "Azure OpenAI (GPT-4o, GPT-4o-mini, text-embedding-3-large) + Cohere Rerank",
                "Databases": "Polyglot (PostgreSQL, SQLite, Vector DB)",
                "Deployment": "Vercel + Railway (independent)",
                "Authentication": "NextAuth (Google OAuth 2.0)",
            },
            "Data Flow and Key Architectural Decisions": {
                "User Query Ingestion": "/api/analyze",
                "Agentic Pipeline Phases": "Rewrite → Retrieval → SQL → Synthesis",
                "Hybrid Retrieval Strategy": "Chroma + Cohere Rerank + Metadata SQL",
                "SQL Execution": "Model Context Protocol (MCP)",
                "Conversation State": "Stateful Conversation Checkpointing",
                "Bilingual Support": "Czech ↔ English Response Pipeline",
            },
            "System Diagram": {
                "Request Routing": "Hosting Separation",
                "Agent Workflow": "Orchestration",
                "Data Access": "Defense-in-depth Zones",
                "Document Ingestion": "Embedding Flow",
                "Feedback Loop": "Observability",
            },
        },
        "2. Backend": {
            "Backend Architecture and Technologies": {
                "Application Container": "FastAPI Application Container",
                "Event Loop": "Uvicorn Event Loop & Windows Policy",
                "Routing": "Modular Routing Packages",
                "Lifecycle": "Configuration & Lifespan Management",
            },
            "Agent Workflow and Orchestration": {
                "Orchestration Engine": "LangGraph",
                "Pipeline Steps": "Prompt rewriting, Dual retrieval, MCP SQL execution, Reflection loops, Answer formatting",
                "Control": "Deterministic pipeline, Explicit retry & cancellation",
            },
            "API Design and Endpoint Purposes": {
                "Analysis": "POST /analyze",
                "Chat": "Chat Threads & Messages (/chat-threads, /chat/{thread_id})",
                "Catalog": "GET /catalog",
                "Data Explorer": "/data-tables, /data-table",
                "Feedback": "POST /feedback, POST /sentiment",
                "Operations": "/health, /debug/*, /stop-execution",
            },
            "Data Management and Persistence": {
                "LangGraph Checkpointer": "Supabase PostgreSQL, AsyncPostgresSaver, connection pooling, retry logic",
                "Lifecycle Coordination": "Graceful degradation, in-memory fallback",
                "State Objects": "Schema-Aware (DataAnalysisState)",
                "Vector Store": "ChromaDB",
                "Analytics": "Turso SQLite",
            },
            "External Service Integration": {
                "Azure OpenAI": "GPT-4o, GPT-4o-mini, embeddings",
                "Azure AI": "Translator & Language Detection",
                "Cohere": "Rerank API",
                "Google": "OAuth Verification",
                "CZSU": "API Ingestion Jobs",
                "LlamaParse": "PDF Processing",
            },
            "Error Handling, Authentication, Middleware": {
                "Error Handling": "Global exception handlers",
                "Throttling": "Throttling middleware",
                "Monitoring": "Memory-monitoring middleware",
                "CORS & Compression": "CORS, Brotli compression",
                "Authentication": "JWT verification",
            },
            "Performance Optimizations": {
                "Concurrency": "Semaphore-based throttling",
                "Rate Limiting": "Retry-friendly rate limiting",
                "Compression": "Brotli compression",
                "Resource Management": "Memory cleanup tasks",
            },
        },
        "3. Frontend": {
            "Frontend Architecture and Technologies": {
                "Routing": "Next.js 15 App Router (server/client separation)",
                "UI Framework": "React 19 Client Components",
                "Type Safety": "TypeScript Strict Mode",
                "Styling": "TailwindCSS (utility-first)",
                "Session Management": "NextAuth",
            },
            "Main Pages and Features": {
                "Chat Interface": "/chat (thread sidebar, message area, input bar, feedback panels)",
                "Catalog Browser": "/catalog",
                "Data Explorer": "/data (autocomplete, filtering, sorting)",
                "Login": "/login (Google OAuth)",
                "Thread Management": "Sidebar (infinite scroll)",
                "Feedback": "Feedback & Sentiment Controls",
            },
            "State Management and Component Architecture": {
                "State Provider": "ChatCacheContext Provider",
                "Persistence": "localStorage 48-Hour Cache",
                "Cross-Tab Sync": "Storage events",
                "UI Updates": "Optimistic UI Updates",
                "Pagination": "Infinite Scroll (IntersectionObserver)",
                "Components": "Component Composition Pattern",
            },
            "API Integration": {
                "Fetch Utilities": "apiFetch & authApiFetch (centralized)",
                "Security": "Token injection",
                "Timeouts": "Timeout control",
                "Retry": "Automatic retry on 401",
            },
            "Authentication Flow": {
                "OAuth": "NextAuth OAuth flows",
                "Callbacks": "JWT callbacks, Session callbacks",
                "Refresh": "Token refresh",
            },
            "Advanced Features": {
                "Content": "Markdown rendering, Dataset badge navigation",
                "Modals": "SQL/PDF modals",
                "UI Feedback": "Progress indicators",
                "Normalization": "Diacritics normalization",
                "Enhancement": "Progressive enhancement",
            },
        },
        "4. Deployment": {
            "Deployment Strategies and Platforms": {
                "Frontend Hosting": "Vercel Edge Hosting (CDN, automatic builds)",
                "Backend Hosting": "Railway Managed Containers (buildpacks, automated rollouts)",
                "API Routing": "API Proxying (17 Vercel rewrite rules)",
                "Geography": "Multi-Region Deployment (europe-west4)",
                "Zero-Downtime": "Blue-Green Deployments",
                "SSL/TLS": "Automatic Provisioning (Let's Encrypt)",
            },
            "Build and Runtime Configuration": {
                "Vercel": "Auto-detection (Next.js)",
                "Railway": "RAILPACK (uv, Python deps, Uvicorn)",
                "Environments": "Reproducible environments",
            },
            "Database and External Service Setup": {
                "Supabase PostgreSQL": "Managed, connection pooling, automated backups, point-in-time recovery",
                "AsyncPostgresSaver": "Connection Pool (min/max sizes, keepalive pings, retry decorators)",
                "Turso SQLite Cloud": "Edge replicas, HTTP API, branching, per-read pricing",
                "Chroma Cloud": "Multi-tenant Vector Database, auto-scaling",
                "Azure OpenAI": "Service Endpoints (regional, rate limits, compliance)",
                "Secrets": "Environment Secrets Management (encrypted, runtime injection)",
                "LangSmith": "Cloud Integration (trace ingestion, evaluation datasets)",
            },
            "Monitoring and Debugging": {
                "Health Endpoints": "Database, memory, rate-limits",
                "Debug Routes": "Checkpoints, run IDs, pool status",
                "Dashboards": "Platform dashboards (Vercel, Railway)",
            },
            "Performance Optimization": {
                "Compression": "Brotli compression",
                "Caching": "48-hour browser caching",
                "Pooling": "Connection pooling",
                "Throttling": "Semaphore throttling",
                "Cleanup": "Memory cleanup tasks",
            },
        },
    }
}


def create_mindmap_graph(data, graph=None, parent=None, level=0):
    """
    Recursively creates a Graphviz directed graph from nested dictionary.

    Args:
        data: Dictionary representing the mindmap structure
        graph: Graphviz Digraph object
        parent: Parent node name
        level: Current nesting level (for styling)

    Returns:
        Graphviz Digraph object
    """
    if graph is None:
        graph = Digraph(comment="CZSU Implementation Summary Mindmap")
        graph.attr(rankdir="LR", size="20,15")
        graph.attr("node", shape="box", style="rounded,filled", fontname="Arial")

    # Color scheme for different levels
    colors = [
        "#E8F4F8",
        "#D1E7F0",
        "#B8D9E8",
        "#9FCBE0",
        "#86BDD8",
        "#6DAFD0",
        "#54A1C8",
        "#3B93C0",
    ]

    for key, value in data.items():
        # Create unique node ID
        node_id = f"{parent}_{key}" if parent else key

        # Truncate long labels for readability
        display_key = key if len(key) <= 60 else key[:57] + "..."

        # Node styling based on level
        color = colors[min(level, len(colors) - 1)]
        if level == 0:
            graph.node(
                node_id, display_key, fillcolor=color, fontsize="16", shape="ellipse"
            )
        elif level == 1:
            graph.node(
                node_id, display_key, fillcolor=color, fontsize="14", shape="box"
            )
        else:
            graph.node(node_id, display_key, fillcolor=color, fontsize="12")

        # Create edge from parent
        if parent:
            graph.edge(parent, node_id)

        # Recursively process children
        if isinstance(value, dict):
            create_mindmap_graph(value, graph, node_id, level + 1)
        elif isinstance(value, str):
            # Leaf node
            leaf_id = f"{node_id}_leaf"
            leaf_label = value if len(value) <= 60 else value[:57] + "..."
            graph.node(
                leaf_id, leaf_label, fillcolor="#FFFACD", fontsize="11", shape="note"
            )
            graph.edge(node_id, leaf_id, style="dotted")

    return graph


def print_mindmap_text(data, indent=0):
    """
    Prints the mindmap structure as indented text.

    Args:
        data: Dictionary representing the mindmap structure
        indent: Current indentation level
    """
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}├── {key}")
            print_mindmap_text(value, indent + 1)
        else:
            print(f"{prefix}├── {key}: {value}")


def main():
    """
    Main function to generate and save the mindmap.
    """
    import os

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # Generate graph with very high DPI for superior quality
    graph = create_mindmap_graph(mindmap)
    graph.attr(dpi="600")  # Very high resolution for PNG
    graph.attr(resolution="600")  # Additional quality setting

    # Output files
    output_base = os.path.join(output_dir, "implementation_core_mindmap_summary")

    # Render in multiple formats
    graph.render(output_base, format="png", cleanup=True)
    print(f"✓ Very high-quality PNG mindmap saved to: {output_base}.png")

    graph.render(output_base, format="pdf", cleanup=False)
    print(f"✓ PDF mindmap saved to: {output_base}.pdf")

    # Clean up any unwanted files (like the extensionless file)
    import glob

    for file in glob.glob(
        os.path.join(output_dir, "implementation_core_mindmap_summary")
    ):
        if os.path.isfile(file) and not file.endswith((".png", ".pdf", ".py")):
            try:
                os.remove(file)
                print(f"✓ Cleaned up unwanted file: {os.path.basename(file)}")
            except OSError:
                pass

    # Print text version
    print("\n" + "=" * 80)
    print("MINDMAP TEXT STRUCTURE")
    print("=" * 80 + "\n")
    print_mindmap_text(mindmap)
    print("\n" + "=" * 80)
    print(f"Mindmap generation complete! Files saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
