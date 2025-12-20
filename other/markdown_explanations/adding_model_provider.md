# Adding a New Model Provider to CZSU Multi-Agent Text-to-SQL

This guide documents the process of adding a new LLM provider to the CZSU Multi-Agent Text-to-SQL system. Use this as a reference when adding new model providers like Mistral, Grok, or others.

## Model Configuration Overview

The application uses a **centralized Python configuration** for all node-specific model settings. Each node in the workflow (e.g., `rewrite_prompt_node`, `generate_query_node`, `reflect_node`) can use a different model configured in `my_agent/utils/node_models_config.py`.

**Key Benefits:**
- Configure all models in one place
- Different nodes can use different models (e.g., use GPT-4o for query generation but GPT-4o-mini for formatting)
- Easy to switch models without changing code
- Clear visibility of which models are used where

## Prerequisites

### 1. API Key Setup
- Obtain an API key from the model provider (e.g., Mistral AI, xAI, etc.)
- Add a placeholder entry to `.env` file:
  ```bash
  # MISTRAL
  MISTRAL_API_KEY=your-mistral-api-key-here
  ```

### 2. Package Installation
Install the LangChain integration package using UV:
```bash
uv pip install langchain-mistralai
```

### 3. Update Dependencies
Add the package to `pyproject.toml` in the dependencies section:
```toml
"langchain-mistralai>=0.2.10",
```

**Note**: Use Context7 MCP tools to check for the latest package version before adding to `pyproject.toml`.

### 4. Context7 MCP Usage
When researching package versions or documentation, use the Context7 MCP tools:
- `mcp_context7_get-library-docs` - Get documentation for specific libraries
- `mcp_context7_resolve-library-id` - Resolve package names to Context7-compatible IDs

## Files to Modify

### 1. `my_agent/utils/models.py`

**Location**: `my_agent/utils/models.py`

**What to do**:
- Add import for the LangChain model class at the top
- Add a new function `get_<provider>_llm()` following the existing pattern

**Example** (for Mistral):
```python
# Add to imports at top:
from langchain_mistralai import ChatMistralAI

# Add new function:
def get_mistral_llm(
    model_name: str = "mistral-small-latest",
    temperature: Optional[float] = 0.0,
) -> ChatMistralAI:
    """Get an instance of Mistral AI Chat LLM with configurable parameters.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        model_name (str): Mistral model name (e.g., "mistral-small-latest", "mistral-large-latest")
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatMistralAI: Configured LLM instance with async support
    """
    return ChatMistralAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
```

**Testing Example** (for Mistral):
```python
# #####################################################
# llm = get_mistral_llm(
#     model_name="mistral-small-latest",
#     temperature=0.0,
# )
# response = llm.invoke("Hi")
# print(f"Model Type: {type(llm).__name__}, Model Name: {llm.model}")
# print(f"Response: {response.content}")
```

### 2. `my_agent/utils/helpers.py`

**Location**: `my_agent/utils/helpers.py`

**What to do**:
- Add import for the new model function
- Add elif clause in `get_configured_llm()` function
- Update the error message to include the new provider

**Example** (for Mistral):
```python
# Add to imports:
from my_agent.utils.models import (
    # ... existing imports ...
    get_mistral_llm,
)

# Add to get_configured_llm() function (in the provider selection logic):
elif model_provider == "mistral":
    llm = get_mistral_llm(
        model_name=model_name,
        temperature=temperature,
    )
    use_bind_tools = True  # Mistral uses OpenAI-compatible API, requires bind_tools()
```

**Update error message**:
```python
raise ValueError(
    f"Unknown model_provider: {model_provider}. Options: 'azureopenai', 'anthropic', 'gemini', 'ollama', 'xai', 'mistral', 'github'"
)
```

### 3. `my_agent/utils/model_configs_all.py`

**Location**: `my_agent/utils/model_configs_all.py`

**What to do**:
- Add the new model configuration to the `MODEL_CONFIGS_ALL` list
- This is the central catalog of all available models

**Example** (adding Mistral models):
```python
# Add to MODEL_CONFIGS_ALL list:
{
    "model_provider": "mistral",
    "model_name": "mistral-large-latest",
    "temperature": 0.0,
    "streaming": False,
    "description": "Mistral Large Latest - Large scale model for complex tasks",
},
{
    "model_provider": "mistral",
    "model_name": "mistral-small-latest",
    "temperature": 0.0,
    "streaming": False,
    "description": "Mistral Small Latest - Lightweight option for faster responses",
},
```

**Note**: 
- The `deployment_name` field is only required for Azure OpenAI. Leave it empty for other providers.
- The `base_url` field is only required for Ollama. Leave it empty for other providers.
- `my_agent/utils/node_models_config.py` should NOT be modified when adding new providers - it's only for selecting which models to use in production.

### 4. `tests/models/test_models_with_tool.py`

**Location**: `tests/models/test_models_with_tool.py`

**What to do**:
- Add the new provider to the `models_to_test` list
- Update the print statement that lists available models

**Example** (for Mistral):
```python
models_to_test = [
    # "azureopenai",
    # "anthropic",
    # "gemini",
    # "ollama",
    "xai",
    "mistral",  # Add this line
]

# Update the print statement:
print("Models to test: azureopenai, anthropic, gemini, ollama, xai, mistral")
```

## Environment Variables

Add the API key to `.env`:

```bash
# MISTRAL
MISTRAL_API_KEY=your-mistral-api-key-here
```

## Configuration Structure
py` file has the following structure:

```python
NODE_MODELS_CONFIG = {
    "nodes": {
        "node_name_here": {
            "model_provider": "provider_name",
            "model_name": "model_name",
            "deployment_name": "",
            "temperature": 0.0,
            "streaming": False,
            "openai_api_version": "2024-05-01-preview",
            "base_url": "http://localhost:11434",
            "description": "Description of what this node does",
        },
        # ... more nodes ...
    },
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
```

**Configuration Parameters:**
- `model_provider`: Provider name (e.g., "azureopenai", "anthropic", "mistral", "ollama")
- `model_name`: Model name for the selected provider
- `deployment_name`: Only for Azure OpenAI, leave empty for others
- `temperature`: 0.0 = deterministic, 1.0 = creative, 2.0 = very random
- `streaming`: True for streaming responses (only format_answer_node), False otherwise
- `openai_api_version`: Only used for Azure OpenAI
- `base_url`: Only used for Ollama
- `description`: Human-readable description of the node's purpose

## Testing

1. **Test the model provider integration:**
   ```bash
   python tests/models/test_models_with_tool.py
   ```
   In `models_to_test`, uncomment only the new provider (comment others).

2. **Test the model function directly:**
   ```bash
   python my_agent/utils/models.py
   ```
   Uncomment only the section in the main block for the new provider (comment others).

3. **Add to model catalog:**
   Add the model configuration to `model_configs_all.py` so it's available for future use.
   **Note**: Do NOT modify `node_models_config.py` when adding new providers.

## Configuration Examples

### Example 1: Use different models for different nodes
```python
NODE_MODELS_CONFIG = {
    "nodes": {
        "rewrite_prompt_node": {
            "model_provider": "azureopenai",
            "model_name": "gpt-4o",
            "deployment_name": "gpt-4o__test1",
        },
        "generate_query_node": {
            "model_provider": "ollama",
            "model_name": "qwen3-coder:30b",
        },
        "format_answer_node": {
            "model_provider": "anthropic",
            "model_name": "claude-sonnet-4-5-20250929",
        },
    }
}
```

### Example 2: Override config for specific node programmatically
```python
# In your code, you can still override settings:
llm, _ = get_configured_llm(
    node_name="generate_query_node",
    temperature=0.5  # Override just temperature
)
```

## Notes

- All model functions should follow the same pattern: `get_<provider>_llm()`
- Use `use_bind_tools = True` for OpenAI-compatible APIs, `False` for Gemini or other non-compatible APIs
- The Python configuration is loaded once and cached for performance
- Each node gets its config from the Python dict, with fallback to defaults if not specified
- Use Python syntax (True/False, not true/false; comments with #, etc.)
- Ensure the model supports tool calling if used in agentic workflows (especially for generate_query_node)

## Provider-Specific Considerations

- **Azure OpenAI**: Requires `deployment_name` and `model_name`
- **Anthropic/Gemini/xAI/Mistral**: Only `model_name`
- **Ollama**: Local models, specify `base_url` if not default
- **GitHub Models**: Some models only support default temperature
- **Tool Binding**: Most providers use `bind_tools()`, Gemini uses tools in `ainvoke()`