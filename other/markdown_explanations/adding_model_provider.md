# Adding a New Model Provider to CZSU Multi-Agent Text-to-SQL

This guide documents the process of adding a new LLM provider to the CZSU Multi-Agent Text-to-SQL system. Use this as a reference when adding new model providers like Mistral, Grok, or others.

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

# Add to get_configured_llm() function:
elif model_provider == "mistral":
    llm = get_mistral_llm(
        model_name="mistral-small-latest",
        temperature=0.0,
    )
    use_bind_tools = True  # Mistral uses OpenAI-compatible API, requires bind_tools()
```

**Update error message**:
```python
raise ValueError(
    f"Unknown model_provider: {model_provider}. Options: 'azureopenai', 'anthropic', 'gemini', 'ollama', 'xai', 'mistral'"
)
```

### 3. `tests/models/test_models_with_tool.py`

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

## Testing

1. In models_to_test, uncomment only new provider
    Run the model test script:
   ```bash
   python tests/models/test_models_with_tool.py
   BUT - In models_to_test, uncomment only new provider  (comment others)
   ```

2. Test the model function directly:
   ```bash
   python my_agent/utils/models.py
    BUT - uncomment only section in main block - for the new provider (comment others)
   ```

## Notes

- All model functions should follow the same pattern: `get_<provider>_llm()`
- Use `use_bind_tools = True` for OpenAI-compatible APIs, `False` for Gemini or other OpenAI-non-compatible APIs that pass tools inside the invoke function
- Update MODEL_PROVIDER options in `.env` comments if needed

- Ensure the model supports tool calling if used in agent workflows

## Provider-Specific Considerations

- **Azure OpenAI**: Requires `deployment_name` and `model_name`
- **Anthropic/Gemini/xAI/Mistral**: Only `model_name`
- **Ollama**: Local models, specify `base_url` if not default
- **Tool Binding**: Most providers use `bind_tools()`, Gemini uses tools in `ainvoke()`