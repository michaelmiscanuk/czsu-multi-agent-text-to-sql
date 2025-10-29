# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sqlite3
import requests
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

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
                        f"No schema found for selection_code {selection_code}."
                    )
        except Exception as e:
            schemas.append(f"Error loading schema from DB: {e}")
        finally:
            if "conn" in locals():
                conn.close()
        return "\n**************\n".join(schemas)
    # fallback
    return "No selection_code provided in state."


async def translate_to_english(text):
    """Helper function that translates text to English using Azure Translator API.

    This function provides language translation for PDF chunk retrieval, where queries may be in
    Czech but PDF documentation is in English. It uses Azure Cognitive Services Translator API
    with asynchronous execution to avoid blocking the event loop.

    The function runs the synchronous HTTP request in a thread pool executor to maintain async
    compatibility while using the requests library. It generates a unique trace ID for each
    request to support debugging and monitoring.

    Args:
        text (str): Text to translate (any language supported by Azure Translator).

    Returns:
        str: Translated text in English.

    Key Steps:
        1. Load Azure Translator credentials from environment
        2. Construct translation endpoint URL with API version and target language
        3. Build HTTP headers with subscription key, region, and trace ID
        4. Create request body with input text
        5. Execute POST request in thread pool (async-safe)
        6. Parse JSON response and extract translated text
        7. Return English translation

    Environment Variables Required:
        - TRANSLATOR_TEXT_SUBSCRIPTION_KEY: Azure Translator API key
        - TRANSLATOR_TEXT_REGION: Azure region (e.g., 'westeurope')
        - TRANSLATOR_TEXT_ENDPOINT: API endpoint URL

    API Details:
        - Endpoint: /translate?api-version=3.0&to=en
        - Method: POST
        - Content-Type: application/json
        - Headers: Ocp-Apim-Subscription-Key, Ocp-Apim-Subscription-Region, X-ClientTraceId
    """
    load_dotenv()
    subscription_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
    region = os.environ["TRANSLATOR_TEXT_REGION"]
    endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]

    path = "/translate?api-version=3.0"
    params = "&to=en"
    constructed_url = endpoint + path + params

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
    return result[0]["translations"][0]["text"]


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
