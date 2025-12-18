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
def get_configured_llm(model_type: str = None, tools: list = None):
    """Get configured LLM instance with optional tool binding based on model type.

    This function centralizes model configuration and handles tool binding according to
    each model's API requirements:
    - OpenAI/Anthropic/OLLAMA: Tools are bound using llm.bind_tools()
    - Gemini: Tools are passed separately (caller must pass to ainvoke())

    Args:
        model_type (str, optional): Model type - "azureopenai", "gemini", "ollama", or "anthropic"
                         If None (default), reads from MODEL_TYPE environment variable,
                         falling back to "azureopenai" if not set
        tools (list, optional): List of tool objects to bind/configure with the LLM.
                               If None, no tools are bound.

    Returns:
        tuple: (llm_configured, use_bind_tools_flag)
               - llm_configured: LLM instance with tools bound (if use_bind_tools=True and tools provided)
                                or base LLM instance (if use_bind_tools=False or no tools)
               - use_bind_tools_flag: Boolean indicating tool binding strategy
                                     True for OpenAI/Anthropic/OLLAMA, False for Gemini

    Raises:
        ValueError: If unknown model_type is provided
        ValueError: If DEPLOYMENT_NAME is required but not set (for azureopenai)

    Environment Variables:
        MODEL_TYPE: LLM provider - "azureopenai", "anthropic", "gemini", "ollama", "xai"
        MODEL_NAME: Model name for the selected provider
        DEPLOYMENT_NAME: Azure OpenAI deployment name (required only for azureopenai)

        Required API Keys by Provider:
        - azureopenai: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
        - anthropic: ANTHROPIC_API_KEY
        - gemini: GOOGLE_API_KEY
        - ollama: No API key required (runs locally)
        - xai: XAI_API_KEY

    Example:
        # Without tools
        llm, use_bind_tools = get_configured_llm()

        # With tools for OpenAI (auto-binds)
        llm, use_bind_tools = get_configured_llm(tools=[my_tool])
        # llm already has tools bound, just call: llm.ainvoke(messages)

        # With tools for Gemini (tools returned separately)
        llm, use_bind_tools = get_configured_llm("gemini", tools=[my_tool])
        # Must pass tools to ainvoke: llm.ainvoke(messages, tools=tools)
    """
    load_dotenv()

    if model_type is None:
        model_type = os.environ.get("MODEL_TYPE")
        if not model_type:
            raise ValueError(
                "MODEL_TYPE environment variable is required. "
                "Set it in .env file to one of: 'azureopenai', 'anthropic', 'gemini', 'ollama', 'xai'"
            )

    # Get model name from environment variable (required for all providers)
    model_name = os.environ.get("MODEL_NAME")
    if not model_name:
        raise ValueError(
            "MODEL_NAME environment variable is required. "
            "Set it in .env file to the appropriate model name for your MODEL_TYPE."
        )

    if model_type == "azureopenai":
        # DEPLOYMENT_NAME is required for Azure OpenAI
        deployment_name = os.environ.get("DEPLOYMENT_NAME")
        if not deployment_name:
            raise ValueError(
                "DEPLOYMENT_NAME environment variable is required for MODEL_TYPE='azureopenai'. "
                "Set it in .env file to one of: 'gpt-4.1___test1', 'gpt-5.2-chat-mimi-test', 'gpt-4o-mini-mimi2'"
            )

        llm = get_azure_openai_chat_llm(
            deployment_name=deployment_name,
            model_name=model_name,
            openai_api_version="2024-05-01-preview",
            temperature=0.0,
        )
        use_bind_tools = True  # OpenAI requires bind_tools()

    elif model_type == "anthropic":
        llm = get_anthropic_llm(
            model_name=model_name,
            temperature=0.0,
        )
        use_bind_tools = True  # Anthropic requires bind_tools()

    elif model_type == "gemini":
        llm = get_gemini_llm(model_name=model_name, temperature=0.0)
        use_bind_tools = False  # Gemini accepts tools directly in ainvoke()

    elif model_type == "ollama":
        # IMPORTANT: Use models with native tool calling support
        # Recommended models: llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b
        # Specialized: llama3-groq-tool-use:8b (fine-tuned for tool calling)
        # For tool-enabled qwen2.5-coder, use: hhao/qwen2.5-coder-tools
        # Small models (0.5b, 1b) have very poor tool calling support - avoid them!
        llm = get_ollama_llm(model_name=model_name, temperature=0.0)
        use_bind_tools = (
            True  # OLLAMA uses OpenAI-compatible API, requires bind_tools()
        )

    elif model_type == "xai":
        llm = get_xai_llm(model_name=model_name, temperature=0.0)
        use_bind_tools = True  # xAI uses OpenAI-compatible API, requires bind_tools()

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Options: 'azureopenai', 'anthropic', 'gemini', 'ollama', 'xai'"
        )

    # Bind tools if needed and provided
    if tools and use_bind_tools:
        llm_configured = llm.bind_tools(tools)
    else:
        llm_configured = llm

    return llm_configured, use_bind_tools
