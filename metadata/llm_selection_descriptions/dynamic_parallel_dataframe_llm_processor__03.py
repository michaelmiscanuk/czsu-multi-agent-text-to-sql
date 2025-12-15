module_description = r"""Parallel DataFrame Processing with Azure OpenAI for Selection Descriptions

This module provides functionality to process selection descriptions in parallel using Azure OpenAI,
with support for incremental processing, dual storage (CSV + SQLite), and robust error handling.

Key Features:
-------------
1. Incremental Processing: Skips already processed selection codes using SQLite tracking.
2. Parallel Processing: Uses ThreadPoolExecutor for concurrent API calls with configurable workers.
3. Rate Limiting: Implements request rate limiting to respect API constraints.
4. Dual Storage: Saves results to both CSV (human-readable) and SQLite (structured querying).
5. Error Handling: 
   - Comprehensive error handling with detailed logging
   - Both console and file logging
   - Error metrics tracking
   - Graceful continuation on individual record failures
6. Progress Tracking: 
   - Real-time progress bar
   - Detailed processing statistics
   - Visual separators in output
7. External Template Support: Loads prompt templates from external text files with UTF-8 support.

Processing Flow:
--------------
1. Initialization:
   - Loads environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)
   - Loads prompt template from file
   - Validates configuration parameters
   - Sets up console logging and metrics tracking

2. Data Loading:
   - Loads input DataFrame from CSV file
   - Joins with corresponding JSON schema files
   - Validates required columns (selection_code, short_description)
   - Tracks missing schema files for reporting

3. Processing Preparation:
   - Checks each selection_code against SQLite database
   - Skips already processed codes
   - Creates list of records to process
   - Calculates rate limiting parameters
   - Pre-allocates results list for order preservation

4. Parallel Processing:
   - Uses ThreadPoolExecutor for concurrent processing
   - Implements rate limiting between requests
   - Processes each record:
     a. Formats prompt with selection data
     b. Calls Azure OpenAI API (single attempt, no retries)
     c. Handles responses and errors
     d. Saves results incrementally
   - Implements 5-minute timeout per record

5. Result Storage:
   - Saves each processed record to:
     a. CSV file with visual separators (100 '=' characters × 3)
     b. SQLite database with timestamps
   - Updates progress and metrics
   - Handles errors without stopping processing
   - Maintains UTF-8 encoding and proper newlines

6. Completion:
   - Calculates and displays processing statistics:
     - Total processing time
     - Success/failure counts
     - Average time per record
     - Success rate percentage
   - Reports any failed records
   - Lists missing metadata files
   - Returns updated DataFrame

Usage Example:
-------------
# Load and process selection descriptions
from dynamic_parallel_dataframe_llm_processor import process_dataframe_parallel

# Process the DataFrame
result_df = process_dataframe_parallel(
    input_df,
    output_column='extended_description',
    max_workers=3,
    requests_per_minute=30
)

Required Files:
-------------
1. PROMPT_TEMPLATE.txt: Contains the prompt template with placeholders
2. selection_descriptions.csv: Input CSV with selection codes and descriptions
3. schemas/*_schema.json: JSON schema files for each selection code

Output Files:
------------
1. output.csv: Human-readable results with visual separators
2. selection_descriptions.db: SQLite database with processed records

Error Handling:
-------------
- Individual record failures don't stop processing
- All errors are logged to console
- Failed records are tracked in metrics
- Missing schema files are reported at completion
- Database connection errors are handled gracefully
- API timeouts are handled with 5-minute limit per record"""

import csv
import json
import logging

# ===============================================================================
# IMPORTS
# ===============================================================================
# Standard library imports
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import pandas as pd
import tqdm as tqdm_module
from dotenv import load_dotenv

# Retry logic imports
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

# Import the get_azure_openai_chat_llm function
from my_agent.utils.models import get_azure_openai_chat_llm


# ===============================================================================
# CUSTOM EXCEPTIONS
# ===============================================================================
class ConfigurationError(Exception):
    """Raised when there's a configuration error."""

    pass


class ProcessingError(Exception):
    """Raised when processing fails."""

    pass


# ===============================================================================
# CONFIGURATION AND SETUP
# ===============================================================================
# Configure logging with minimal output
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load and validate environment variables
load_dotenv()
required_env_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# Consolidated configuration with rate limiting and processing parameters
CONFIG = {
    "MAX_WORKERS": 15,  # Number of parallel processing threads
    "REQUESTS_PER_MINUTE": {
        "DEFAULT": 5,  # Default rate limit for API calls
        "MIN": 1,  # Minimum allowed rate
        "MAX": 100,  # Maximum allowed rate
    },
    "CSV_SEPARATOR": ";",  # Semicolon separator for CSV files
    "PROMPT_TEMPLATE": "",  # Will be loaded from file
    "OUTPUT_COLUMN": "extended_description",  # Default output column name
    "PROCESS_ALL_SELECTIONS": 0,  # Set to False to process only specific selection
    "SPECIFIC_SELECTION_CODE": [
        "CENYLES104T01",
        "CEN0205ET02",
        "CEN0205ET03",
        "CENYLES104T02",
        "CEN0202AT02",
        "OBY02GT01",
        "FIN02QT3",
        "FIN02QT5",
        "CEN0203CT01",
        "CEN0203DT01",
        "CEN0203DT02",
        "CEN0203DT03",
        "CRU06T4",
        "OBY04BTOR",
        "OBY02PKT03",
        "CEN0305T02",
        "CEN0306T01",
        "CEN0307T01",
        "PRU01BT7",
        "PRU10BT5",
        "PRU01BT9",
        "PRU01CT4",
        "PRU01AT6",
    ],  # Only used when PROCESS_ALL_SELECTIONS is False. Can be a string or list of strings.
    "MAX_RETRIES": 6,  # Maximum number of retry attempts for API calls
    "RETRY_WAIT_MIN": 2,  # Minimum wait time between retries (seconds)
    "RETRY_WAIT_MAX": 60,  # Maximum wait time between retries (seconds)
}

# Track missing metadata files for reporting
missing_metadata_files = []

# Global counter for tracking retry attempts across function calls
_llm_attempt_counter = 0


def validate_config():
    """Validate configuration parameters."""
    if (
        not CONFIG["REQUESTS_PER_MINUTE"]["MIN"]
        <= CONFIG["REQUESTS_PER_MINUTE"]["DEFAULT"]
        <= CONFIG["REQUESTS_PER_MINUTE"]["MAX"]
    ):
        raise ConfigurationError(
            f"Invalid requests_per_minute: {CONFIG['REQUESTS_PER_MINUTE']['DEFAULT']}"
        )
    if CONFIG["MAX_WORKERS"] < 1:
        raise ConfigurationError(f"Invalid max_workers: {CONFIG['MAX_WORKERS']}")


# Validate configuration
validate_config()


# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================
def load_prompt_template_from_txt(txt_path: str = "PROMPT_TEMPLATE.txt") -> str:
    """Load prompt template from a text file.

    Args:
        txt_path (str): Path to the text file containing the prompt template.

    Returns:
        str: The prompt template loaded from the file.

    Raises:
        ConfigurationError: If the file cannot be loaded or is empty.
    """
    try:
        template_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / txt_path
        with open(template_path, "r", encoding="utf-8") as file:
            template = file.read()
            if not template:
                raise ConfigurationError("Template file is empty")
            return template
    except Exception as e:
        raise ConfigurationError(f"Failed to load prompt template from file: {e}")


def load_dataframe_from_csv_and_jsons(csv_path: str) -> pd.DataFrame:
    """Load DataFrame from CSV file and join with corresponding JSON schema files.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame loaded from the CSV file with joined JSON schemas.

    Raises:
        ConfigurationError: If the file cannot be loaded or is empty.
    """
    try:
        csv_file_path = BASE_DIR / csv_path
        with open(csv_file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=CONFIG["CSV_SEPARATOR"])
            header = next(reader)
            data = list(reader)

        # Create DataFrame with renamed columns
        df = pd.DataFrame(data, columns=["selection_code", "short_description"])
        if df.empty:
            raise ConfigurationError("DataFrame is empty")

        # Add JSON schema column with correct name
        df["selection_schema_json"] = df["selection_code"].apply(
            lambda code: load_json_schema(code)
        )
        return df
    except Exception as e:
        raise ConfigurationError(f"Failed to load DataFrame from CSV: {e}")


def load_json_schema(selection_code: str) -> str:
    """Load JSON schema file for a given selection code.

    Args:
        selection_code (str): The selection code to load schema for.

    Returns:
        str: The contents of the JSON schema file.

    Raises:
        ConfigurationError: If the schema file cannot be loaded.
    """
    try:
        schema_path = (
            BASE_DIR / "metadata" / "schemas" / f"{selection_code}_schema.json"
        )
        if not schema_path.exists():
            missing_metadata_files.append(selection_code)
            return ""  # Return empty string for missing schemas
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        missing_metadata_files.append(selection_code)
        logger.warning(f"Failed to load JSON schema for {selection_code}: {e}")
        return ""  # Return empty string for failed loads


def format_system_prompt(**kwargs: Dict[str, Any]) -> str:
    """Format the prompt template using DataFrame column values.

    Args:
        **kwargs: Keyword arguments representing DataFrame column values.

    Returns:
        str: The formatted prompt.

    Raises:
        ValueError: If there is an error formatting the prompt template.
    """
    try:
        return CONFIG["PROMPT_TEMPLATE"].format(**kwargs)
    except Exception as e:
        raise ValueError(f"Error formatting prompt template: {e}")


def save_to_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV file with visual separators between rows.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The filename to save the DataFrame to.
    """
    try:
        absolute_path = BASE_DIR / filename

        # Create a prominent separator line (repeated 3 times)
        separator = "\n" + "=" * 100 + "\n" + "=" * 100 + "\n" + "=" * 100 + "\n"

        # Write with visual separators
        with open(absolute_path, "w", encoding="utf-8", newline="") as f:
            # Write header
            f.write(CONFIG["CSV_SEPARATOR"].join(df.columns) + "\n")

            # Write each row with a prominent separator
            for _, row in df.iterrows():
                f.write(CONFIG["CSV_SEPARATOR"].join(str(val) for val in row) + "\n")
                f.write(separator)  # Add prominent separator after each row

        logger.info(f"DataFrame successfully saved to {absolute_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV: {e}")
        raise


def save_to_sqlite(df: pd.DataFrame, db_name: str = "selection_descriptions.db"):
    """Save DataFrame to SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        db_name (str): The name of the SQLite database file.

    Raises:
        Exception: If there is an error saving to the database.
    """
    try:
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / db_name
        conn = sqlite3.connect(str(db_path))

        # Add timestamp column to DataFrame before saving
        df["processed_at"] = datetime.now().isoformat()

        # Create table with all columns including timestamp
        df.to_sql("selection_descriptions", conn, if_exists="replace", index=False)

        conn.close()
        logger.info(f"DataFrame successfully saved to SQLite database: {db_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to SQLite: {e}")
        raise


# ===============================================================================
# MONITORING AND METRICS
# ===============================================================================
@dataclass
class Metrics:
    """Simple metrics collection for tracking processing statistics.

    This class tracks various metrics during the processing of records:
    - Processing time
    - Success/failure counts
    - Performance statistics
    - Failed records with reasons

    Attributes:
        start_time (float): Timestamp when processing started.
        processed_rows (int): Number of successfully processed rows.
        failed_rows (int): Number of rows that failed processing.
        total_processing_time (float): Total time taken for processing.
        failed_records (list): List of tuples containing (selection_code, error_message) for failed records.
    """

    start_time: float = field(default_factory=time.time)
    processed_rows: int = 0
    failed_rows: int = 0
    total_processing_time: float = 0
    failed_records: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the metrics with formatted timestamps
                           and calculated averages.
        """
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "processed_rows": self.processed_rows,
            "failed_rows": self.failed_rows,
            "total_processing_time": self.total_processing_time,
            "average_time_per_row": self.total_processing_time
            / max(1, self.processed_rows),
            "success_rate": (
                self.processed_rows / max(1, self.processed_rows + self.failed_rows)
            )
            * 100,
            "failed_records": self.failed_records,
        }

    def update_processing_time(self) -> None:
        """Update the total processing time based on the current time."""
        self.total_processing_time = time.time() - self.start_time


def handle_processing_error(
    error: Exception, selection_code: str, metrics: Metrics
) -> None:
    """Handle processing errors consistently.

    This function provides consistent error handling by:
    1. Formatting error messages uniformly
    2. Logging to both console and file
    3. Updating metrics
    4. Ensuring proper error propagation

    Args:
        error (Exception): The error that occurred.
        selection_code (str): The selection code being processed.
        metrics (Metrics): The metrics object to update.
    """
    error_msg = f"Error processing selection code {selection_code}: {str(error)}"
    print(f"\n{error_msg}")
    logger.error(error_msg)
    metrics.failed_rows += 1
    metrics.failed_records.append((selection_code, str(error)))


# ===============================================================================
# CORE PROCESSING FUNCTIONS
# ===============================================================================
# @retry(
#     stop=stop_after_attempt(20),  # Try 20 times
#     wait=wait_exponential(multiplier=10, min=10, max=60),  # Wait between 10-60 seconds, increasing exponentially
#     reraise=True
# )
def get_azure_llm_response(**kwargs: Dict[str, Any]) -> str:
    """Get response from Azure OpenAI using the configured LLM with retry logic.

    This function provides robust API communication with Azure OpenAI,
    including automatic retry logic for transient failures, proper error handling,
    and exponential backoff to ensure reliable data retrieval.

    The function uses the tenacity library for retry logic with exponential backoff,
    automatically retrying failed requests up to MAX_RETRIES times with increasing
    wait times between attempts.

    Args:
        **kwargs: Keyword arguments representing DataFrame column values for prompt formatting.

    Returns:
        str: The cleaned response content from the LLM.

    Raises:
        RetryError: If all retry attempts are exhausted.
    """
    global _llm_attempt_counter

    # Inner function with retry logic
    @retry(
        retry=retry_if_exception_type(Exception),  # Retry on any exception
        stop=stop_after_attempt(CONFIG["MAX_RETRIES"]),
        wait=wait_exponential(
            min=CONFIG["RETRY_WAIT_MIN"], max=CONFIG["RETRY_WAIT_MAX"]
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=False,
    )
    def _get_llm_response_with_retry():
        # Increment counter for each attempt
        global _llm_attempt_counter
        _llm_attempt_counter += 1
        attempt = _llm_attempt_counter

        request_id = f"req_{int(time.time()*1000)}"
        formatted_prompt = format_system_prompt(**kwargs)

        print(
            f"[Attempt {attempt}/{CONFIG['MAX_RETRIES']}] Making LLM API call for request {request_id}"
        )

        try:
            print(f"\nInitializing LLM connection...")
            llm = get_azure_openai_chat_llm(
                deployment_name="gpt-4.1___test1",
                model_name="gpt-4.1",
                openai_api_version="2024-05-01-preview",
            )
            print("LLM initialized successfully")

            messages = [{"role": "user", "content": formatted_prompt}]
            print("Messages prepared")

            print(f"\nMaking API call...")
            response = llm.invoke(messages)
            print(f"\nReceived response from API")
            print("Processing response content...")

            # Clean the response content
            result = response.content.strip()
            # Remove any markdown code block markers
            result = result.replace("```", "")
            # Remove any leading/trailing whitespace
            result = result.strip()

            print("Response content processed successfully")
            return result

        except Exception as e:
            error_msg = f"Error in LLM request {request_id}: {str(e)}"
            print(f"\n{error_msg}")
            print(f"Attempting retry with prompt: {formatted_prompt[:200]}...")
            raise  # Re-raise to trigger retry mechanism

    # Reset counter before starting new request
    _llm_attempt_counter = 0

    # Call inner function with retry logic
    try:
        result = _get_llm_response_with_retry()
    except RetryError as e:
        # All retries exhausted - log and re-raise
        print(f"❌ All {CONFIG['MAX_RETRIES']} retry attempts failed for LLM request")
        print(f"   Last error: {e}")
        raise

    # Reset counter after completion (success or failure)
    _llm_attempt_counter = 0

    return result


def check_selection_code_exists(
    selection_code: str, db_name: str = "selection_descriptions.db"
) -> bool:
    """Check if a selection code already exists in the database.

    Args:
        selection_code (str): The selection code to check.
        db_name (str): The name of the SQLite database file.

    Returns:
        bool: True if the selection code exists and has a non-empty extended_description,
              False otherwise.
    """
    try:
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / db_name
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='selection_descriptions'
            """
            )
            if not cursor.fetchone():
                return False

            # Check if selection code exists and has a non-empty extended_description
            cursor.execute(
                """
                SELECT 1 FROM selection_descriptions 
                WHERE selection_code = ? AND extended_description IS NOT NULL AND extended_description != ''
            """,
                (selection_code,),
            )

            return cursor.fetchone() is not None
    except Exception as e:
        logger.error(f"Error checking selection code in database: {e}")
        return False


def save_single_record_to_csv(
    row: pd.Series, filename: str, is_first_record: bool = False
):
    """Save a single record to CSV file with visual separators.

    This function saves a record to CSV with:
    - Header row for first record only
    - Visual separator (100 '=' characters repeated 3 times)
    - UTF-8 encoding
    - Proper newline handling

    Args:
        row (pd.Series): The row to save.
        filename (str): The filename to save to.
        is_first_record (bool): Whether this is the first record being saved.
                              If True, writes the header row.
    """
    try:
        csv_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / filename
        mode = "w" if is_first_record else "a"

        with open(csv_path, mode, encoding="utf-8", newline="") as f:
            if is_first_record:
                f.write(CONFIG["CSV_SEPARATOR"].join(row.index) + "\n")

            f.write(CONFIG["CSV_SEPARATOR"].join(str(val) for val in row) + "\n")
            f.write("\n" + "=" * 100 + "\n" + "=" * 100 + "\n" + "=" * 100 + "\n")

    except Exception as e:
        error_msg = f"Error saving single record to CSV: {str(e)}"
        logger.error(error_msg)
        raise


def save_single_record_to_sqlite(
    row: pd.Series, db_name: str = "selection_descriptions.db"
):
    """Save a single record to SQLite database with upsert (merge) logic."""
    try:
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / db_name
        print(f"\nAttempting to save to database: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            # Enable foreign keys and set isolation level
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None  # Enable autocommit mode

            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='selection_descriptions'
            """
            )
            table_exists = cursor.fetchone() is not None
            print(f"Table exists: {table_exists}")

            if not table_exists:
                # Create new table with proper constraints
                conn.execute(
                    f"""
                    CREATE TABLE selection_descriptions (
                        selection_code TEXT PRIMARY KEY,
                        short_description TEXT,
                        selection_schema_json TEXT,
                        extended_description TEXT,
                        processed_at TEXT
                    )
                """
                )
                print("Created new selection_descriptions table")
            else:
                # Check if selection_code is already a PRIMARY KEY
                cursor.execute("PRAGMA table_info(selection_descriptions)")
                columns = cursor.fetchall()
                selection_code_is_pk = any(
                    col[1] == "selection_code" and col[5] == 1 for col in columns
                )
                print(f"Selection code is PRIMARY KEY: {selection_code_is_pk}")

                if not selection_code_is_pk:
                    print("Recreating table with proper PRIMARY KEY constraint...")
                    # Drop temporary table if it exists
                    conn.execute("DROP TABLE IF EXISTS selection_descriptions_new")
                    print("Dropped temporary table if it existed")

                    # Create temporary table with correct structure
                    conn.execute(
                        """
                        CREATE TABLE selection_descriptions_new (
                            selection_code TEXT PRIMARY KEY,
                            short_description TEXT,
                            selection_schema_json TEXT,
                            extended_description TEXT,
                            processed_at TEXT
                        )
                    """
                    )
                    print("Created new temporary table")

                    # Copy data from old table to new table, keeping only the most recent record for each selection_code
                    conn.execute(
                        """
                        INSERT INTO selection_descriptions_new 
                        SELECT selection_code, short_description, selection_schema_json, 
                               extended_description, processed_at 
                        FROM (
                            SELECT *,
                                   ROW_NUMBER() OVER (PARTITION BY selection_code ORDER BY processed_at DESC) as rn
                            FROM selection_descriptions
                        ) ranked
                        WHERE rn = 1
                    """
                    )
                    print("Copied data to temporary table")

                    # Drop old table and rename new one
                    conn.execute("DROP TABLE selection_descriptions")
                    conn.execute(
                        "ALTER TABLE selection_descriptions_new RENAME TO selection_descriptions"
                    )
                    print("Table recreated with proper PRIMARY KEY constraint")

            # Now proceed with the upsert operation
            row_dict = row.to_dict()
            row_dict["processed_at"] = datetime.now().isoformat()
            selection_code = row_dict["selection_code"]

            print(f"\nProcessing selection_code: {selection_code}")

            # Start transaction
            conn.execute("BEGIN TRANSACTION")

            try:
                # First check if record exists
                cursor.execute(
                    "SELECT selection_code FROM selection_descriptions WHERE selection_code = ?",
                    (selection_code,),
                )
                existing = cursor.fetchone()
                print(f"Existing record check: {'Found' if existing else 'Not found'}")

                # Prepare the data with proper parameter binding
                placeholders = ", ".join(["?" for _ in row_dict])
                columns = ", ".join(row_dict.keys())

                # Use parameterized query for safety
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO selection_descriptions ({columns})
                    VALUES ({placeholders})
                """,
                    list(row_dict.values()),
                )

                # Verify the operation
                cursor.execute(
                    "SELECT selection_code FROM selection_descriptions WHERE selection_code = ?",
                    (selection_code,),
                )
                if cursor.fetchone():
                    print(f"Successfully saved record for {selection_code}")
                else:
                    print(
                        f"Failed to save record for {selection_code} - record not found after insert"
                    )

                # Commit transaction
                conn.execute("COMMIT")
                print("Transaction committed")

            except sqlite3.IntegrityError as e:
                # Rollback on integrity error
                conn.execute("ROLLBACK")
                print(f"Integrity error: {str(e)}")
                # Try to get more information about the error
                cursor.execute(
                    "SELECT * FROM selection_descriptions WHERE selection_code = ?",
                    (selection_code,),
                )
                if cursor.fetchone():
                    print(f"Record for {selection_code} already exists in database")
                raise
            except Exception as e:
                # Rollback on any other error
                conn.execute("ROLLBACK")
                print(f"Error: {str(e)}")
                raise

    except Exception as e:
        print(f"Error saving single record to SQLite: {e}")
        raise


def process_row(
    index: int,
    row: pd.Series,
    output_column: str,
    delay_between_requests: float,
    metrics: Metrics,
) -> tuple[int, str | None]:
    """Process a single row of the DataFrame.

    This function processes a single row by:
    1. Displaying a visual separator with selection code
    2. Implementing rate limiting between requests
    3. Preparing data for LLM processing
    4. Making the API call
    5. Handling any errors that occur

    Args:
        index (int): The index of the row.
        row (pd.Series): The row to process.
        output_column (str): The name of the output column.
        delay_between_requests (float): Delay between requests in seconds.
        metrics (Metrics): The metrics object to update with processing results.

    Returns:
        tuple[int, str | None]: The index of the row and the result, or None if an error occurred.
    """
    try:
        # Extract selection code and display processing header
        selection_code = row.get("selection_code")
        print(f"\n{'=' * 100}")
        print(f"Starting processing for selection code: {selection_code}")
        print(f"{'=' * 100}\n")

        # Implement rate limiting between requests
        if index > 0:
            print(
                f"\nWaiting {delay_between_requests:.2f} seconds before processing {selection_code}..."
            )
            time.sleep(delay_between_requests)

        # Prepare row data for LLM processing
        row_dict = row.to_dict()
        if output_column in row_dict:
            del row_dict[output_column]  # Remove existing output to avoid conflicts

        # Call LLM and get response
        print(f"\nCalling LLM for {selection_code}...")
        result = get_azure_llm_response(**row_dict)
        print(f"\nSuccessfully got response for {selection_code}")
        return index, result

    except Exception as e:
        # Handle and log any errors during processing
        handle_processing_error(e, row.get("selection_code", f"Row {index}"), metrics)
        return index, None


def process_dataframe_parallel(
    df: pd.DataFrame,
    output_column: str = CONFIG["OUTPUT_COLUMN"],
    max_workers: int = CONFIG["MAX_WORKERS"],
    requests_per_minute: int = CONFIG["REQUESTS_PER_MINUTE"]["DEFAULT"],
) -> pd.DataFrame:
    """Process a DataFrame in parallel using Azure OpenAI.

    This function processes records in parallel while:
    - Skipping already processed records
    - Implementing rate limiting
    - Saving results incrementally
    - Providing detailed progress tracking
    - Handling errors gracefully

    Args:
        df (pd.DataFrame): The DataFrame to process.
        output_column (str): The name of the column to store the results in.
        max_workers (int): The maximum number of workers to use for parallel processing.
        requests_per_minute (int): The maximum number of requests to make per minute.

    Returns:
        pd.DataFrame: The DataFrame with the results added to the specified output column.

    Raises:
        ProcessingError: If there's an error in the overall processing flow.
    """
    metrics = Metrics()

    try:
        # Initial check to identify which records need processing
        print("\nStarting initial check of records...")
        records_to_process = []
        skipped_count = 0

        # First pass: identify records to process and skip already processed ones
        for index, row in df.iterrows():
            selection_code = row.get("selection_code")
            if not selection_code:
                print(f"Warning: Row {index + 1} has no selection_code, skipping")
                continue

            # Skip if this selection code has already been processed
            if check_selection_code_exists(selection_code):
                print(f"Skipping already processed selection code: {selection_code}")
                skipped_count += 1
                continue

            # If processing specific selection only, skip others
            if CONFIG["PROCESS_ALL_SELECTIONS"] == 0:
                specific_codes = CONFIG["SPECIFIC_SELECTION_CODE"]
                if isinstance(specific_codes, str):
                    specific_codes = [specific_codes]
                if selection_code not in specific_codes:
                    print(
                        f"Skipping selection code {selection_code} (not in target selections)"
                    )
                    skipped_count += 1
                    continue

            records_to_process.append((index, row))

        # Print initial processing summary
        print(f"\nProcessing Summary:")
        print(f"- Total records: {len(df)}")
        print(f"- Already processed: {skipped_count}")
        print(f"- To process: {len(records_to_process)}")
        if CONFIG["PROCESS_ALL_SELECTIONS"] == 0:
            specific_codes = CONFIG["SPECIFIC_SELECTION_CODE"]
            if isinstance(specific_codes, str):
                specific_codes = [specific_codes]
            print(f"- Processing only selection codes: {specific_codes}")

        # Early return if nothing to process
        if not records_to_process:
            print("\nNo new records to process!")
            return df

        print("\nStarting processing of new records...")
        # Pre-allocate results list to maintain order of processed records
        results = [None] * len(df)
        # Calculate delay between requests to respect rate limiting
        delay_between_requests = 60 / requests_per_minute
        processed_count = 0

        # Process records in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor, maintaining index for result mapping
            futures = {
                executor.submit(
                    process_row,
                    index,
                    row,
                    output_column,
                    delay_between_requests,
                    metrics,
                ): index
                for index, row in records_to_process
            }

            # Monitor progress with tqdm
            with tqdm_module.tqdm(
                total=len(records_to_process),
                desc="Processing",
                leave=True,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                file=sys.stdout,
            ) as pbar:
                # Process completed futures as they come in
                for future in as_completed(futures):
                    try:
                        # Get result with timeout to prevent hanging on unresponsive tasks
                        # 5-minute timeout per record to handle potential API delays
                        index, result = future.result(timeout=300)
                        results[index] = result

                        if result is not None:
                            # Successfully processed record
                            metrics.processed_rows += 1
                            processed_count += 1

                            # Update DataFrame with the new result
                            df.at[index, output_column] = result
                            row_to_save = df.iloc[index]

                            # Save to both CSV and SQLite for redundancy and backup
                            # CSV provides human-readable format, SQLite for structured querying
                            save_single_record_to_csv(
                                row_to_save,
                                "output.csv",
                                is_first_record=(processed_count == 1),
                            )
                            save_single_record_to_sqlite(row_to_save)
                            print(f"\nSuccessfully saved record {processed_count}")
                        else:
                            # Record processing failed - log and continue
                            print(
                                f"\nFailed to process selection code: {df.iloc[index]['selection_code']}"
                            )

                    except TimeoutError:
                        # Handle timeout errors - task took too long to complete
                        selection_code = df.iloc[index]["selection_code"]
                        error_msg = (
                            f"Timeout while processing selection code: {selection_code}"
                        )
                        print(f"\n{error_msg}")
                        handle_processing_error(
                            TimeoutError(error_msg), selection_code, metrics
                        )
                    except Exception as e:
                        # Handle unexpected errors - log and continue processing
                        selection_code = df.iloc[index]["selection_code"]
                        error_msg = f"Unexpected error in future: {str(e)}"
                        print(f"\n{error_msg}")
                        handle_processing_error(e, selection_code, metrics)
                    pbar.update(1)

        # Calculate and display final processing statistics
        metrics.update_processing_time()

        print(f"\nProcessing completed in {metrics.total_processing_time:.2f} seconds:")
        print(f"- Total records: {len(df)}")
        print(f"- Already processed: {skipped_count}")
        print(f"- Newly processed: {processed_count}")
        print(f"- Failed: {metrics.failed_rows}")
        print(
            f"- Average time per new record: {metrics.total_processing_time/max(1,processed_count):.2f} seconds"
        )
        print(f"- Success rate: {metrics.to_dict()['success_rate']:.1f}%")

        # Display failed records if any
        if metrics.failed_rows > 0:
            print("\nFailed Records:")
            print("=" * 50)
            for selection_code, error in metrics.failed_records:
                print(f"- {selection_code}: {error}")
            print("=" * 50)
            print(f"Total failed records: {metrics.failed_rows}")

        return df

    except Exception as e:
        error_msg = f"Error in process_dataframe_parallel: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg)
        raise ProcessingError(error_msg) from e


# ===============================================================================
# EXECUTION BLOCK
# ===============================================================================
if __name__ == "__main__":
    try:
        # Load prompt template from CSV
        CONFIG["PROMPT_TEMPLATE"] = load_prompt_template_from_txt()
        print(f"Loaded prompt template: {CONFIG['PROMPT_TEMPLATE'][:1000]}...")

        # Load DataFrame from CSV and join with JSON schemas
        input_csv_path = "metadata/selection_descriptions.csv"
        test_df = load_dataframe_from_csv_and_jsons(input_csv_path)

        print("Starting parallel DataFrame processing...")
        processed_df = process_dataframe_parallel(
            test_df,
            output_column=CONFIG["OUTPUT_COLUMN"],
            max_workers=CONFIG["MAX_WORKERS"],
            requests_per_minute=CONFIG["REQUESTS_PER_MINUTE"]["DEFAULT"],
        )

        # Print missing metadata files if any
        if missing_metadata_files:
            print("\nMissing metadata files:")
            print("=" * 50)
            for code in sorted(missing_metadata_files):
                print(f"- {code}")
            print("=" * 50)
            print(f"Total missing files: {len(missing_metadata_files)}")

    except Exception as e:
        print(f"Application error: {e}")
        raise
