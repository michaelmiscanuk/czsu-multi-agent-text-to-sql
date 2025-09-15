from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/", tags=["root"])
async def api_root():
    """
    Welcome to CZSU Multi-Agent Text-to-SQL API
    This endpoint provides comprehensive information about available API endpoints,
    their purposes, and how to interact with the system.
    """
    return {
        "name": "CZSU Multi-Agent Text-to-SQL API",
        "version": "1.0.0",
        "description": "AI-powered text-to-SQL conversion using multi-agent architecture",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json",
        },
        "core_endpoints": {
            "text_to_sql": {
                "endpoint": "/analyze",
                "method": "POST",
                "description": "Convert natural language questions to SQL queries",
                "purpose": "Main AI analysis functionality",
            },
            "data_catalog": {
                "endpoint": "/catalog",
                "method": "GET",
                "description": "Get available database schema and table information",
                "purpose": "Explore available data sources",
            },
            "chat_interface": {
                "threads": "/chat-threads",
                "messages": "/chat/{thread_id}/messages",
                "sentiments": "/chat/{thread_id}/sentiments",
                "delete_thread": "/chat/{thread_id}",
                "description": "Interactive chat interface for conversational SQL queries",
            },
        },
        "data_endpoints": {
            "tables": {
                "all_tables": "/data-tables",
                "specific_table": "/data-table",
                "description": "Access table schemas and metadata",
            },
            "bulk_operations": {
                "all_messages": "/chat/all-messages-for-all-threads",
                "thread_messages": "/chat/all-messages-for-one-thread/{thread_id}",
                "description": "Bulk data retrieval operations",
            },
        },
        "system_endpoints": {
            "health": {
                "basic": "/health",
                "database": "/health/database",
                "memory": "/health/memory",
                "rate_limits": "/health/rate-limits",
                "prepared_statements": "/health/prepared-statements",
                "description": "System health and monitoring",
            },
            "debug": {
                "pool_status": "/debug/pool-status",
                "checkpoints": "/debug/chat/{thread_id}/checkpoints",
                "run_details": "/debug/run-id/{run_id}",
                "description": "Development and debugging tools",
            },
            "admin": {
                "clear_cache": "/admin/clear-cache",
                "clear_statements": "/admin/clear-prepared-statements",
                "set_env": "/debug/set-env",
                "reset_env": "/debug/reset-env",
                "description": "Administrative operations",
            },
        },
        "feedback_endpoints": {
            "feedback": "/feedback",
            "sentiment": "/sentiment",
            "description": "User feedback and sentiment analysis",
        },
        "utility_endpoints": {
            "placeholder_image": "/placeholder/{width}/{height}",
            "run_ids": "/chat/{thread_id}/run-ids",
            "description": "Utility and helper functions",
        },
        "getting_started": {
            "step_1": "Visit /docs for interactive API documentation",
            "step_2": "Check /health to verify system status",
            "step_3": "Explore /catalog to see available data",
            "step_4": "Use /analyze to convert text to SQL",
            "step_5": "Try /chat-threads for conversational interface",
        },
        "features": [
            "Multi-agent AI architecture for robust SQL generation",
            "Natural language to SQL conversion",
            "Interactive chat interface",
            "Comprehensive data catalog",
            "Real-time health monitoring",
            "Sentiment analysis and feedback collection",
            "Debug and administrative tools",
            "Rate limiting and throttling protection",
        ],
        "support": {
            "documentation": "/docs",
            "health_check": "/health",
            "system_status": "/debug/pool-status",
        },
    }
