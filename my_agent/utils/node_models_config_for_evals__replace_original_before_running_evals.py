"""Node-specific model configurations for the multi-agent text-to-SQL workflow.

This module defines which model each node uses. Each node can have different:
- model_provider: "azureopenai", "anthropic", "gemini", "ollama", "xai", "mistral", "github"
- model_name: specific model for that provider
- deployment_name: Azure OpenAI deployment name (only for azureopenai)
- temperature: 0.0 = deterministic, 1.0 = creative, 2.0 = very random
- streaming: True for streaming responses, False otherwise
- openai_api_version: Azure OpenAI API version (only for azureopenai)
- base_url: Ollama server URL (only for ollama)

To change a node's model, simply edit the configuration below.
"""

# Node-specific model configurations
NODE_MODELS_CONFIG = {
    "nodes": {
        # Rewrites user prompts for better retrieval and query understanding
        "rewrite_prompt_node": {
            "id": "mistral_mistral-large-2512",
            "model_provider": "mistral",
            "model_name": "mistral-large-2512",
            "temperature": 0.0,
            "streaming": False,
        },
        # Summarizes conversation history to maintain context
        "summarize_messages_node": {
            "id": "mistral_mistral-large-2512",
            "model_provider": "mistral",
            "model_name": "mistral-large-2512",
            "temperature": 0.0,
            "streaming": False,
        },
        # Generates SQL queries using MCP tools (agentic SQL generation)
        "generate_query_node": {
            "id": "mistral_devstral-2512",
            "model_provider": "mistral",
            "model_name": "devstral-2512",
            "temperature": 0.0,
            "streaming": False,
        },
        # Reflects on query results and decides whether to improve or answer
        "reflect_node": {
            "id": "mistral_mistral-large-2512",
            "model_provider": "mistral",
            "model_name": "mistral-large-2512",
            "temperature": 0.0,
            "streaming": False,
        },
        # Formats the final answer from SQL results and PDF chunks
        "format_answer_node": {
            "id": "github_openai_gpt-4.1",
            "model_provider": "github",
            "model_name": "openai/gpt-4.1",
            "temperature": 0.0,
            "streaming": False,
        },
        # Non-streaming fallback for format_answer_node
        "format_answer_node_non_streaming": {
            "id": "github_openai_gpt-4.1",
            "model_provider": "github",
            "model_name": "openai/gpt-4.1",
            "temperature": 0.0,
            "streaming": False,
        },
        # Generates follow-up prompt suggestions for the user
        "followup_prompts_node": {
            "id": "mistral_mistral-large-2512",
            "model_provider": "mistral",
            "model_name": "mistral-large-2512",
            "temperature": 0.0,
            "streaming": False,
        },
    },
    # Default configuration used when node-specific config is not found
    "defaults": {
        "model_provider": "azureopenai",
        "model_name": "gpt-4o-mini",
        "deployment_name": "gpt-4o-mini-mimi2",
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-05-01-preview",
        "base_url": "http://localhost:11434",
    },
}

# Configuration parameters:
# - temperature: 0.0 = deterministic, 1.0 = creative, 2.0 = very random
# - streaming: True for streaming responses (format_answer_node), False otherwise
# - deployment_name: Only required for azureopenai provider, leave empty for others
# - base_url: Only used for ollama provider, ignored for others
# - openai_api_version: Only used for azureopenai provider
