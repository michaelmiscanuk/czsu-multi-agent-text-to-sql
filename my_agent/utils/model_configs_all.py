"""All available model configurations for the multi-agent text-to-SQL workflow.

This module contains a comprehensive list of all tested and available models.
When adding a new model provider or model:
1. Add the configuration to MODEL_CONFIGS_ALL below
2. Test the model using my_agent/utils/models.py
3. Copy the desired configuration to node_models_config.py

DO NOT import this directly in production code - it's a reference/catalog only.
Use node_models_config.py to select which models each node uses.
"""

# All available model configurations
MODEL_CONFIGS_ALL = [
    # ============================================================================
    # AZURE OPENAI MODELS
    # ============================================================================
    # Azure OpenAI GPT-4o - Premium model for complex reasoning and evaluations
    {
        "id": "azureopenai_gpt-4o",
        "model_provider": "azureopenai",
        "model_name": "gpt-4o",
        "deployment_name": "gpt-4o__test1",
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-05-01-preview",
    },
    # Azure OpenAI GPT-4o-mini - Lightweight, cost-effective model
    {
        "id": "azureopenai_gpt-4o-mini",
        "model_provider": "azureopenai",
        "model_name": "gpt-4o-mini",
        "deployment_name": "gpt-4o-mini-mimi2",
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-05-01-preview",
    },
    # Azure OpenAI GPT-4.1 - Evaluation testing model
    {
        "id": "azureopenai_gpt-4.1",
        "model_provider": "azureopenai",
        "model_name": "gpt-4.1",
        "deployment_name": "gpt-4.1___test1",
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-05-01-preview",
    },
    # Azure OpenAI GPT-5-nano - Next-gen lightweight model
    {
        "id": "azureopenai_gpt-5-nano",
        "model_provider": "azureopenai",
        "model_name": "gpt-5-nano",
        "deployment_name": "gpt-5-nano_mimi_test",
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-12-01-preview",
    },
    # Azure OpenAI GPT-5.2-chat - Next-gen chat model
    {
        "id": "azureopenai_gpt-5.2-chat",
        "model_provider": "azureopenai",
        "model_name": "gpt-5.2-chat",
        "deployment_name": "gpt-5.2-chat-mimi-test",
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-12-01-preview",
    },
    # Azure OpenAI GPT-3.2-chat - Evaluation testing model
    {
        "id": "azureopenai_gpt-3.2-chat",
        "model_provider": "azureopenai",
        "model_name": "gpt-3.2-chat",
        "deployment_name": "",  # No deployment name provided in docs
        "temperature": 0.0,
        "streaming": False,
        "openai_api_version": "2024-05-01-preview",
    },
    # ============================================================================
    # ANTHROPIC MODELS
    # ============================================================================
    # Anthropic Claude Sonnet 4.5 - Premium reasoning and analysis model
    {
        "id": "anthropic_claude-sonnet-4-5-20250929",
        "model_provider": "anthropic",
        "model_name": "claude-sonnet-4-5-20250929",
        "temperature": 0.0,
        "streaming": False,
    },
    # ============================================================================
    # GOOGLE GEMINI MODELS
    # ============================================================================
    # Google Gemini 3 Pro Preview - Advanced preview model
    {
        "id": "gemini_gemini-3-pro-preview",
        "model_provider": "gemini",
        "model_name": "gemini-3-pro-preview",
        "temperature": 0.0,
        "streaming": False,
    },
    # Google Gemini 2.0 Flash Exp - Fast experimental model
    {
        "id": "gemini_gemini-2.0-flash-exp",
        "model_provider": "gemini",
        "model_name": "gemini-2.0-flash-exp",
        "temperature": 0.0,
        "streaming": False,
    },
    # Google Gemini 1.5 Pro - Production stable model
    {
        "id": "gemini_gemini-1.5-pro",
        "model_provider": "gemini",
        "model_name": "gemini-1.5-pro",
        "temperature": 0.0,
        "streaming": False,
    },
    # ============================================================================
    # XAI (GROK) MODELS
    # ============================================================================
    # xAI Grok 4.1 Fast Reasoning - Reasoning-optimized model
    {
        "id": "xai_grok-4-1-fast-reasoning",
        "model_provider": "xai",
        "model_name": "grok-4-1-fast-reasoning",
        "temperature": 0.0,
        "streaming": False,
    },
    # xAI Grok 4.1 Fast Non-Reasoning - Fast response model
    {
        "id": "xai_grok-4-1-fast-non-reasoning",
        "model_provider": "xai",
        "model_name": "grok-4-1-fast-non-reasoning",
        "temperature": 0.0,
        "streaming": False,
    },
    # xAI Grok Code Fast 1 - Code generation optimized
    {
        "id": "xai_grok-code-fast-1",
        "model_provider": "xai",
        "model_name": "grok-code-fast-1",
        "temperature": 0.0,
        "streaming": False,
    },
    # ============================================================================
    # MISTRAL AI MODELS
    # ============================================================================
    # Mistral Open Mistral Nemo - Open source option
    {
        "id": "mistral_open-mistral-nemo",
        "model_provider": "mistral",
        "model_name": "open-mistral-nemo",
        "temperature": 0.0,
        "streaming": False,
    },
    # Mistral Large 2512 - Large scale model
    {
        "id": "mistral_mistral-large-2512",
        "model_provider": "mistral",
        "model_name": "mistral-large-2512",
        "temperature": 0.0,
        "streaming": False,
    },
    # Mistral Devstral 2512 - Development optimized
    {
        "id": "mistral_devstral-2512",
        "model_provider": "mistral",
        "model_name": "devstral-2512",
        "temperature": 0.0,
        "streaming": False,
    },
    # Mistral Codestral 2508 - Code generation specialized
    {
        "id": "mistral_codestral-2508",
        "model_provider": "mistral",
        "model_name": "codestral-2508",
        "temperature": 0.0,
        "streaming": False,
    },
    # Mistral Small Latest - Lightweight option
    {
        "id": "mistral_mistral-small-latest",
        "model_provider": "mistral",
        "model_name": "mistral-small-latest",
        "temperature": 0.0,
        "streaming": False,
    },
    # ============================================================================
    # OLLAMA (LOCAL) MODELS
    # ============================================================================
    # Ollama Llama 3.2 1B - Local lightweight model (poor tool calling)
    {
        "id": "ollama_llama3.2:1b",
        "model_provider": "ollama",
        "model_name": "llama3.2:1b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # Ollama Llama 3.2 3B - Local standard model (recommended minimum)
    {
        "id": "ollama_llama3.2:3b",
        "model_provider": "ollama",
        "model_name": "llama3.2:3b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # Ollama Llama 3.1 8B - Local larger model (good tool calling)
    {
        "id": "ollama_llama3.1:8b",
        "model_provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # Ollama Qwen 2.5 Coder 30B - Local code specialist
    {
        "id": "ollama_qwen2.5-coder:30b",
        "model_provider": "ollama",
        "model_name": "qwen2.5-coder:30b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # Ollama Qwen 2.5 Coder Tools 14B - Tool calling optimized
    {
        "id": "ollama_hhao/qwen2.5-coder-tools:14b",
        "model_provider": "ollama",
        "model_name": "hhao/qwen2.5-coder-tools:14b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # Ollama DeepSeek R1 Coder Tools 14B - DeepSeek code specialist
    {
        "id": "ollama_deepseek-r1-coder-tools:14b",
        "model_provider": "ollama",
        "model_name": "deepseek-r1-coder-tools:14b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # Ollama Mistral 7B - Local Mistral variant
    {
        "id": "ollama_mistral:7b",
        "model_provider": "ollama",
        "model_name": "mistral:7b",
        "temperature": 0.0,
        "streaming": False,
        "base_url": "http://localhost:11434",
    },
    # ============================================================================
    # GITHUB MODELS
    # ============================================================================
    # GitHub OpenAI GPT-4o - Premium model via GitHub
    {
        "id": "github_openai_gpt-4.1",
        "model_provider": "github",
        "model_name": "openai/gpt-4.1",
        "temperature": 0.0,
        "streaming": False,
    },
    # GitHub OpenAI GPT-4o-mini - Lightweight model via GitHub
    {
        "id": "github_openai_gpt-4.1-mini",
        "model_provider": "github",
        "model_name": "openai/gpt-4.1-mini",
        "temperature": 0.0,
        "streaming": False,
    },
]

# Configuration parameter reference:
# - id: Unique identifier in format "model_provider_model_name" (for evaluation tracking)
# - model_provider: "azureopenai", "anthropic", "gemini", "ollama", "xai", "mistral", "github"
# - model_name: Specific model identifier for that provider
# - deployment_name: Azure deployment name (ONLY for azureopenai, leave empty for others)
# - temperature: 0.0 = deterministic, 1.0 = creative, 2.0 = very random
# - streaming: True for streaming responses, False for batch
# - openai_api_version: Azure API version (ONLY for azureopenai, leave empty for others)
# - base_url: Server URL (ONLY for ollama, leave empty for others)
