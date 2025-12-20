"""Helper functions for my_agent module.

This module provides utility functions for schema loading, language translation,
and other operations needed by the agent.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import re
import sqlite3
import requests
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Import model configuration functions (required for get_configured_llm)
from my_agent.utils.models import (
    get_azure_openai_chat_llm,
    get_anthropic_llm,
    get_gemini_llm,
    get_ollama_llm,
    get_xai_llm,
    get_mistral_llm,
    get_github_llm,
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Static IDs for easier debug‑tracking
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
    print(f"🔍 BASE_DIR calculated from __file__: {BASE_DIR}")
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
    print(f"🔍 BASE_DIR calculated from cwd: {BASE_DIR}")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
async def load_schema(state=None):
    """Helper function that loads database schema metadata for selected datasets from SQLite.

    This function retrieves extended schema descriptions from the selection_descriptions.db database
    for the dataset selection codes identified by the retrieval process. The schema includes table
    names, column names, data types, distinct categorical values, and metadata descriptions that are
    essential for accurate SQL query generation.

    The function handles missing or invalid selection codes gracefully by returning appropriate
    error messages, ensuring the SQL generation node always receives usable schema context.

    Args:
        state (dict, optional): State dictionary containing 'top_selection_codes' list.
                               If None or empty, returns fallback message.

    Returns:
        str: Concatenated schema descriptions separated by '**************' delimiter.
             Each schema includes dataset identifier and extended description.
             Returns error message if database access fails or codes are invalid.

    Key Steps:
        1. Extract top_selection_codes from state
        2. Connect to selection_descriptions.db SQLite database
        3. Query extended_description for each selection code
        4. Format schemas with dataset identifier prefix
        5. Join multiple schemas with delimiter
        6. Return concatenated schema string or error message

    Database Schema:
        - Table: selection_descriptions
        - Key columns: selection_code (TEXT), extended_description (TEXT)
        - Location: metadata/llm_selection_descriptions/selection_descriptions.db
    """
    if state and state.get("top_selection_codes"):
        selection_codes = state["top_selection_codes"]
        db_path = (
            BASE_DIR
            / "metadata"
            / "llm_selection_descriptions"
            / "selection_descriptions.db"
        )
        schemas = []
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            for selection_code in selection_codes:
                cursor.execute(
                    """
                    SELECT extended_description FROM selection_descriptions
                    WHERE selection_code = ? AND extended_description IS NOT NULL AND extended_description != ''
                    """,
                    (selection_code,),
                )
                row = cursor.fetchone()
                if row:
                    schemas.append(f"Dataset: {selection_code}.\n" + row[0])
                else:
                    schemas.append(
                        f"No schema found for selection_code " f"{selection_code}."
                    )
        except Exception as exc:
            schemas.append(f"Error loading schema from DB: {exc}")
        finally:
            if "conn" in locals():
                conn.close()
        return "\n**************\n".join(schemas)
    # fallback
    return "No selection_code provided in state."


NEWLINE_SPLIT_PATTERN = re.compile(r"(\r?\n+)")


async def translate_text(text, target_language="en"):
    """Helper function that translates text to a target language using Azure Translator API.

    This function provides language translation for PDF chunk retrieval, where queries may be in
    Czech but PDF documentation is in English. It uses Azure Cognitive Services Translator API
    with asynchronous execution to avoid blocking the event loop.

    The function runs the synchronous HTTP request in a thread pool executor to maintain async
    compatibility while using the requests library. It generates a unique trace ID for each
    request to support debugging and monitoring.

    Args:
        text (str): Text to translate (any language supported by Azure Translator).
        target_language (str): Target language code (e.g., 'en', 'cs', 'de', 'fr'). Defaults to 'en'.

    Returns:
        str: Translated text in the target language.

    Key Steps:
        1. Load Azure Translator credentials from environment
        2. Construct translation endpoint URL with API version and target language
        3. Build HTTP headers with subscription key, region, and trace ID
        4. Create request body with input text
        5. Execute POST request in thread pool (async-safe)
        6. Parse JSON response and extract translated text
        7. Return translated text in target language

    Environment Variables Required:
        - TRANSLATOR_TEXT_SUBSCRIPTION_KEY: Azure Translator API key
        - TRANSLATOR_TEXT_REGION: Azure region (e.g., 'westeurope')
        - TRANSLATOR_TEXT_ENDPOINT: API endpoint URL

    API Details:
        - Endpoint: /translate?api-version=3.0&to={target_language}
        - Method: POST
        - Content-Type: application/json
        - Headers: Ocp-Apim-Subscription-Key, Ocp-Apim-Subscription-Region, X-ClientTraceId
    """

    if not text:
        return ""
    load_dotenv()
    subscription_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
    region = os.environ["TRANSLATOR_TEXT_REGION"]
    endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]

    path = "/translate?api-version=3.0"
    params = f"&to={target_language}"
    constructed_url = endpoint + path + params

    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    parts = NEWLINE_SPLIT_PATTERN.split(text)
    translate_indices = []
    translate_segments = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        if NEWLINE_SPLIT_PATTERN.fullmatch(part):
            continue
        translate_indices.append(idx)
        translate_segments.append(part)

    if not translate_segments:
        return text

    loop = asyncio.get_event_loop()

    async def translate_batch(batch):
        body = [{"text": chunk} for chunk in batch]
        response = await loop.run_in_executor(
            None, lambda: requests.post(constructed_url, headers=headers, json=body)
        )
        response.raise_for_status()
        data = response.json()
        return [item["translations"][0]["text"] for item in data]

    translated_parts = []
    batch_size = 50
    for start in range(0, len(translate_segments), batch_size):
        batch = translate_segments[start : start + batch_size]
        translated_parts.extend(await translate_batch(batch))

    translated_iter = iter(translated_parts)
    for idx in translate_indices:
        parts[idx] = next(translated_iter)

    return "".join(parts)


async def detect_language(text: str) -> str:
    """Helper function that detects the language of text using Azure Translator API.

    This function provides language detection for user prompts to ensure responses match
    the original question's language. It uses Azure Cognitive Services Translator API
    with asynchronous execution to avoid blocking the event loop.

    The function runs the synchronous HTTP request in a thread pool executor to maintain async
    compatibility while using the requests library. It generates a unique trace ID for each
    request to support debugging and monitoring.

    Args:
        text (str): Text to detect language for (any language supported by Azure Translator).

    Returns:
        str: Detected language code (e.g., 'en', 'cs', 'de', 'fr').

    Key Steps:
        1. Load Azure Translator credentials from environment
        2. Construct detection endpoint URL with API version
        3. Build HTTP headers with subscription key, region, and trace ID
        4. Create request body with input text
        5. Execute POST request in thread pool (async-safe)
        6. Parse JSON response and extract language code
        7. Return detected language code

    Environment Variables Required:
        - TRANSLATOR_TEXT_SUBSCRIPTION_KEY: Azure Translator API key
        - TRANSLATOR_TEXT_REGION: Azure region (e.g., 'westeurope')
        - TRANSLATOR_TEXT_ENDPOINT: API endpoint URL

    API Details:
        - Endpoint: /detect?api-version=3.0
        - Method: POST
        - Content-Type: application/json
        - Headers: Ocp-Apim-Subscription-Key, Ocp-Apim-Subscription-Region, X-ClientTraceId
    """
    load_dotenv()
    subscription_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
    region = os.environ["TRANSLATOR_TEXT_REGION"]
    endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]

    path = "/detect?api-version=3.0"
    constructed_url = endpoint + path

    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    body = [{"text": text}]

    # Run the synchronous request in a thread
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: requests.post(constructed_url, headers=headers, json=body)
    )
    result = response.json()
    return result[0]["language"]


# ==============================================================================
# LLM CONFIGURATION HELPER
# ==============================================================================
def load_node_models_config():
    """Load node-specific model configurations from Python config file.

    Returns:
        dict: Configuration dictionary with node-specific settings and defaults
    """
    try:
        from my_agent.utils.node_models_config import NODE_MODELS_CONFIG

        return NODE_MODELS_CONFIG
    except ImportError as e:
        raise ImportError(
            f"Could not import node models configuration: {e}. "
            "Please ensure my_agent/utils/node_models_config.py exists with NODE_MODELS_CONFIG defined."
        )


def get_configured_llm(
    node_name: str = None,
    model_provider: str = None,
    model_name: str = None,
    deployment_name: str = None,
    tools: list = None,
    temperature: float = None,
    streaming: bool = None,
    openai_api_version: str = None,
    base_url: str = None,
):
    """Get configured LLM instance with optional tool binding based on model type.

    This function centralizes model configuration and handles tool binding according to
    each model's API requirements:
    - OpenAI/Anthropic/OLLAMA: Tools are bound using llm.bind_tools()
    - Gemini: Tools are passed separately (caller must pass to ainvoke())

    Configuration Priority (highest to lowest):
    1. Explicit parameters passed to this function
    2. Node-specific configuration from node_models_config.py (if node_name provided)
    3. Default configuration from node_models_config.py

    Args:
        node_name (str, optional): Name of the node (e.g., "rewrite_prompt_node").
                         If provided, loads configuration from node_models_config.py.
                         If None, uses explicit parameters or defaults.
        model_provider (str, optional): Model type - "azureopenai", "gemini", "ollama", "anthropic", "xai", "mistral", "github"
                         If None and node_name provided, reads from config file.
                         If None and no node_name, uses defaults from config file.
        model_name (str, optional): Model name for the selected provider.
                         If None and node_name provided, reads from config file.
        deployment_name (str, optional): Azure OpenAI deployment name.
                         Only used when model_provider is "azureopenai".
                         If None and node_name provided, reads from config file.
        tools (list, optional): List of tool objects to bind/configure with the LLM.
                               If None, no tools are bound.
        temperature (float, optional): Temperature setting for LLM (0.0-2.0).
                         Controls randomness - lower is more deterministic.
                         If None, reads from config file or uses 0.0.
        streaming (bool, optional): Enable streaming responses (if supported by provider).
                         If None, reads from config file or uses False.
        openai_api_version (str, optional): Azure OpenAI API version string.
                         Only used when model_provider is "azureopenai".
                         If None, reads from config file or uses "2024-05-01-preview".
        base_url (str, optional): Base URL for OLLAMA server.
                         Only used when model_provider is "ollama".
                         If None, reads from config file or uses "http://localhost:11434".

    Returns:
        tuple: (llm_configured, use_bind_tools_flag)
               - llm_configured: LLM instance with tools bound (if use_bind_tools=True and tools provided)
                                or base LLM instance (if use_bind_tools=False or no tools)
               - use_bind_tools_flag: Boolean indicating tool binding strategy
                                     True for OpenAI/Anthropic/OLLAMA, False for Gemini

    Raises:
        ValueError: If unknown model_provider is provided
        ValueError: If required parameters are missing
        FileNotFoundError: If node_models_config.py is not found

    Example:
        # Using node-specific configuration
        llm, use_bind_tools = get_configured_llm(node_name="rewrite_prompt_node")

        # Override specific parameters from node config
        llm, use_bind_tools = get_configured_llm(node_name="generate_query_node", temperature=0.5)

        # Legacy usage with explicit parameters (still supported)
        llm, use_bind_tools = get_configured_llm(
            model_provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.7
        )

        # With tools for OpenAI (auto-binds)
        llm, use_bind_tools = get_configured_llm(node_name="generate_query_node", tools=[my_tool])
    """
    load_dotenv()

    # Load configuration from JSON file
    config = load_node_models_config()

    # Determine configuration source based on node_name
    node_config = {}
    if node_name:
        if node_name in config.get("nodes", {}):
            node_config = config["nodes"][node_name]
        else:
            # Use defaults if node not found
            node_config = config.get("defaults", {})
    else:
        # Use defaults if no node_name provided
        node_config = config.get("defaults", {})

    # Apply configuration priority: explicit params > node config > defaults
    model_provider = model_provider or node_config.get("model_provider")
    model_name = model_name or node_config.get("model_name")
    deployment_name = deployment_name or node_config.get("deployment_name", "")
    temperature = (
        temperature if temperature is not None else node_config.get("temperature", 0.0)
    )
    streaming = (
        streaming if streaming is not None else node_config.get("streaming", False)
    )
    openai_api_version = openai_api_version or node_config.get(
        "openai_api_version", "2024-05-01-preview"
    )
    base_url = base_url or node_config.get("base_url", "http://localhost:11434")

    # Validate required parameters
    if not model_provider:
        raise ValueError(
            "model_provider is required. Either provide it as a parameter, "
            "specify node_name, or ensure defaults are set in node_models_config.py"
        )

    if not model_name:
        raise ValueError(
            "model_name is required. Either provide it as a parameter, "
            "specify node_name, or ensure defaults are set in node_models_config.py"
        )

    if model_provider == "azureopenai":
        if not deployment_name:
            raise ValueError(
                "deployment_name is required for model_provider='azureopenai'. "
                "Set it in node_models_config.py or pass as parameter."
            )

        llm = get_azure_openai_chat_llm(
            deployment_name=deployment_name,
            model_name=model_name,
            openai_api_version=openai_api_version,
            temperature=temperature,
            streaming=streaming,
        )
        use_bind_tools = True  # OpenAI requires bind_tools()

    elif model_provider == "anthropic":
        llm = get_anthropic_llm(
            model_name=model_name,
            temperature=temperature,
        )
        use_bind_tools = True  # Anthropic requires bind_tools()

    elif model_provider == "gemini":
        llm = get_gemini_llm(model_name=model_name, temperature=temperature)
        use_bind_tools = False  # Gemini accepts tools directly in ainvoke()

    elif model_provider == "ollama":
        # IMPORTANT: Use models with native tool calling support
        # Recommended models: llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b
        # Specialized: llama3-groq-tool-use:8b (fine-tuned for tool calling)
        # For tool-enabled qwen2.5-coder, use: hhao/qwen2.5-coder-tools
        # Small models (0.5b, 1b) have very poor tool calling support - avoid them!
        llm = get_ollama_llm(
            model_name=model_name, base_url=base_url, temperature=temperature
        )
        use_bind_tools = (
            True  # OLLAMA uses OpenAI-compatible API, requires bind_tools()
        )

    elif model_provider == "xai":
        llm = get_xai_llm(model_name=model_name, temperature=temperature)
        use_bind_tools = True  # xAI uses OpenAI-compatible API, requires bind_tools()
    elif model_provider == "mistral":
        llm = get_mistral_llm(
            model_name=model_name,
            temperature=temperature,
        )
        use_bind_tools = (
            True  # Mistral uses OpenAI-compatible API, requires bind_tools()
        )
    elif model_provider == "github":
        # GitHub Models: Some models (e.g., openai/o3) only support default temperature
        # Only pass temperature if explicitly set to non-default value
        github_kwargs = {"model_name": model_name}
        if temperature != 0.0:  # Only pass if not default
            github_kwargs["temperature"] = temperature

        llm = get_github_llm(**github_kwargs)
        use_bind_tools = (
            True  # GitHub uses OpenAI-compatible API, requires bind_tools()
        )
    else:
        raise ValueError(
            f"Unknown model_provider: {model_provider}. Options: 'azureopenai', 'anthropic', 'gemini', 'ollama', 'xai', 'mistral', 'github'"
        )

    # Bind tools if needed and provided
    # Note: LLM creation functions return RunnableRetry objects
    # We need to access the underlying bound runnable to bind tools
    if tools and use_bind_tools:
        # Extract the underlying LLM from RunnableRetry if needed
        if hasattr(llm, "bound"):
            # llm is a RunnableRetry wrapper, access the underlying model
            base_llm = llm.bound
            llm_configured = base_llm.bind_tools(tools).with_retry(
                stop_after_attempt=30
            )
        else:
            # Direct LLM object (shouldn't happen with current setup)
            llm_configured = llm.bind_tools(tools)
    else:
        llm_configured = llm

    return llm_configured, use_bind_tools
