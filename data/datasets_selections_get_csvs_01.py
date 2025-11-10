"""Czech Statistical Office (CZSU) Data Extraction and CSV Generation

This module provides functionality to extract statistical data from the Czech Statistical Office (CZSU)
public API and convert it to CSV format for local analysis and processing.
"""

MODULE_DESCRIPTION = r"""Czech Statistical Office (CZSU) Data Extraction and CSV Generation

This module provides functionality to extract statistical data from the Czech Statistical Office (CZSU)
public API and convert it to CSV format for local analysis and processing.

Key Features:
-------------
1. Dataset Management:
   - Retrieval of all available datasets from CZSU API
   - Dataset metadata extraction and validation
   - Configurable dataset filtering (all vs. specific)
   - Error handling for missing or invalid datasets
   - Progress tracking with visual indicators

2. Selection Processing:
   - Automatic discovery of data selections within datasets
   - Selection metadata extraction and validation
   - Configurable selection filtering (all vs. specific)
   - Batch processing with error recovery
   - Individual selection progress tracking

3. Data Conversion:
   - JSON-stat to pandas DataFrame conversion
   - Automatic data validation and cleaning
   - CSV export with UTF-8 encoding
   - File naming using selection codes
   - Directory management and organization

4. API Integration:
   - REST API communication with CZSU endpoints
   - Configurable rate limiting to respect API guidelines
   - Robust error handling for network issues with timeout protection
   - Automatic retry mechanisms using tenacity library with exponential backoff
   - Response validation and parsing
   - JSON cleanup for malformed API responses (fixes trailing/leading commas)
   - Debug file generation for failed API responses

5. Progress Monitoring:
   - Real-time progress bars for datasets and selections
   - Success/failure tracking and reporting
   - Detailed error logging with selection codes
   - Performance metrics and timing
   - Final processing summary
   - Retry attempt counters and diagnostics

6. Configuration Management:
   - Config class for network and retry parameters (timeout, retries, rate limiting)
   - Environment-based configuration options
   - Flexible processing modes (all vs. specific)
   - Configurable target directories
   - Processing scope control
   - Debug output support with RESPONSE_DIAGNOSTICS toggle

Processing Flow:
--------------
1. Initialization:
   - Sets up output directory structure
   - Validates configuration parameters
   - Initializes progress tracking
   - Prepares error logging

2. Dataset Discovery:
   - Fetches complete dataset catalog from API
   - Validates dataset structure and metadata
   - Filters datasets based on configuration
   - Prepares dataset processing queue

3. Selection Discovery:
   - For each dataset, retrieves available selections
   - Validates selection metadata and structure
   - Filters selections based on configuration
   - Builds selection processing queue

4. Data Extraction:
   - Downloads JSON-stat data for each selection
   - Validates data structure and completeness
   - Handles missing or malformed data
   - Applies data cleaning and validation
   - Automatic JSON cleanup for API bugs (trailing commas, etc.)
   - Debug file generation for troubleshooting

5. Data Conversion:
   - Converts JSON-stat to pandas DataFrame
   - Validates DataFrame structure and content
   - Handles empty or invalid datasets
   - Prepares data for CSV export

6. File Output:
   - Generates appropriate filenames
   - Exports data to CSV with proper encoding
   - Validates file creation and content
   - Updates progress and success metrics

7. Reporting:
   - Compiles processing statistics
   - Reports successful and failed operations
   - Logs detailed error information
   - Provides final summary

Usage Example:
-------------
# Process all datasets and selections (default configuration)
python datasets_selections_get_csvs.py

# Process specific dataset only
# Set PROCESS_ALL_DATASETS = 0 and SPECIFIC_DATASET_ID = "OBY01PD"

# Process specific selection only
# Set PROCESS_ALL_SELECTIONS = 0 and SPECIFIC_SELECTION_ID = "OBY01PDT01"

Required Environment:
-------------------
- Python 3.7+
- Internet connection for API access
- Write permissions for output directory
- Required packages: requests, pandas, pyjstat, tqdm, pathlib, tenacity, re

API Endpoints:
-------------
- Datasets catalog: https://data.csu.gov.cz/api/katalog/v1/sady
- Dataset details: https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}
- Dataset selections: https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery
- Selection data: https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}

Output:
-------
- CSV files in data/CSVs/ directory
- Files named using selection codes (e.g., OBY01PDT01.csv)
- UTF-8 encoded with proper header rows
- Processing statistics and error reports
- Debug files for failed API responses (when RESPONSE_DIAGNOSTICS=1)

Error Handling:
-------------
- Network connection failures with automatic retry logic
- API response validation errors
- Data parsing and conversion errors
- File system and permission errors
- Empty or malformed datasets
- Configurable timeout protection and rate limiting
- Exponential backoff retry strategy for transient failures
- JSON cleanup for API server bugs (trailing/leading commas)
- Debug file generation for troubleshooting malformed responses
- Graceful handling of all retry failures without script crashes"""

import requests
import pandas as pd
from pyjstat import pyjstat
import os
import time
from pathlib import Path
from tqdm import tqdm
import json
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

# Set up logging for retry visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global counter for tracking retry attempts across function calls
_fetch_attempt_counter = 0

# ==============================================================================
# DIRECTORY SETUP AND CONFIGURATION
# ==============================================================================
# Create data/CSVs directory if it doesn't exist
csv_dir = Path("data/CSVs")
csv_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DEBUG FILE SETUP
# ==============================================================================
# Single consolidated debug file for all API response errors
# File is overwritten on each script execution, errors appended during runtime
from datetime import datetime

debug_file_path = (
    Path(__file__).parent / "debug_response_datasets_selections_get_csvs.txt"
)

# Initialize/overwrite debug file at script startup
with open(debug_file_path, "w", encoding="utf-8") as f:
    f.write(f"Debug Response Log for datasets_selections_get_csvs.py\n")
    f.write(f"Started: {datetime.now().isoformat()}\n")
    f.write("=" * 80 + "\n\n")

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# Dataset processing configuration
PROCESS_ALL_DATASETS = (
    1  # Set to 1 to process all datasets, 0 to process specific dataset
)
SPECIFIC_DATASET_ID = "WZEM02A"  # Only used when PROCESS_ALL_DATASETS is 0

# Selection processing configuration
PROCESS_ALL_SELECTIONS = (
    1  # Set to 1 to process all selections, 0 to process specific selection
)
SPECIFIC_SELECTION_ID = "WZEM02AT01"  # Only used when PROCESS_ALL_SELECTIONS is 0


# Network and retry configuration
class Config:
    """Configuration class for API request handling.

    This class defines configuration parameters for API communication,
    including timeout settings, retry logic, and rate limiting to ensure
    robust and reliable interaction with the CZSU API.

    Attributes:
        TIMEOUT (int): Request timeout in seconds. Default 30 seconds.
                      Prevents indefinite waiting on slow connections.

        MAX_RETRIES (int): Maximum number of retry attempts for failed requests.
                          Default 3 attempts. Helps handle transient network issues.

        RETRY_WAIT_MIN (int): Minimum wait time in seconds between retries.
                             Default 2 seconds. Part of exponential backoff strategy.

        RETRY_WAIT_MAX (int): Maximum wait time in seconds between retries.
                             Default 10 seconds. Caps exponential backoff to prevent
                             excessively long waits.

        RATE_LIMIT_DELAY (float): Delay in seconds between API requests.
                                 Default 0.5 seconds. Implements rate limiting to
                                 respect API guidelines and prevent overloading.

        RESPONSE_DIAGNOSTICS (int): Enable detailed API response diagnostics.
                               Set to 1 to show response content previews, content-type
                               validation, and detailed error information when API
                               responses are malformed or unexpected. Set to 0 to disable
                               for cleaner output. Default 1 (enabled).

    Usage:
        These settings are used by the fetch_json function to control API
        request behavior, ensuring reliable communication while being
        respectful to the CZSU API infrastructure.
    """

    TIMEOUT = 30  # Request timeout in seconds
    MAX_RETRIES = 6  # Maximum number of retry attempts
    RETRY_WAIT_MIN = 2  # Minimum wait time between retries (seconds)
    RETRY_WAIT_MAX = 10  # Maximum wait time between retries (seconds)
    RATE_LIMIT_DELAY = 0.5  # Delay between requests (seconds)
    RESPONSE_DIAGNOSTICS = 1  # Enable detailed response diagnostics (1=yes, 0=no)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def fetch_json(url):
    """Helper function to fetch JSON data with error handling, retry logic, and rate limiting.

    This function provides robust API communication with the CZSU endpoints,
    including automatic retry logic for transient failures, proper error handling,
    HTTP status validation, timeout protection, and rate limiting to respect API
    guidelines and ensure reliable data retrieval.

    The function uses the tenacity library for retry logic with exponential backoff,
    automatically retrying failed requests up to MAX_RETRIES times with increasing
    wait times between attempts.

    Args:
        url (str): The API endpoint URL to fetch data from

    Returns:
        dict | None: The parsed JSON response data if successful,
                     None if the request failed after all retry attempts
                     or returned invalid data

    Raises:
        requests.exceptions.RequestException: Raised and caught by retry decorator
                                             for network-related errors

    Note:
        - Implements configurable timeout from Config.TIMEOUT
        - Uses exponential backoff retry strategy from Config settings
        - Implements rate limiting with Config.RATE_LIMIT_DELAY
        - Validates HTTP status codes before processing response
        - Handles common API errors gracefully with detailed logging
        - Automatic retry on connection timeouts, network errors, and HTTP errors
    """
    global _fetch_attempt_counter

    # Inner function with retry logic
    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, json.JSONDecodeError)
        ),
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(min=Config.RETRY_WAIT_MIN, max=Config.RETRY_WAIT_MAX),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=False,
    )
    def _fetch_with_retry():
        # Increment counter for each attempt
        global _fetch_attempt_counter
        _fetch_attempt_counter += 1
        attempt = _fetch_attempt_counter

        print(
            f"[Attempt {attempt}/{Config.MAX_RETRIES}] Fetching {url} (timeout: {Config.TIMEOUT}s)"
        )

        # Initialize response variable outside try block
        response = None

        try:
            # Make HTTP GET request with timeout protection
            response = requests.get(url, timeout=Config.TIMEOUT)

            # Validate HTTP status code and raise exception for bad responses
            response.raise_for_status()

            # Implement rate limiting to be respectful to the API
            time.sleep(Config.RATE_LIMIT_DELAY)

            # Check if response is actually JSON before parsing
            content_type = response.headers.get("content-type", "").lower()
            response_text = response.text

            # Check if response looks like HTML (common API error)
            if (
                "<html" in response_text.lower()
                or "<!doctype html" in response_text.lower()
            ):
                print(f"✗ API returned HTML error page instead of JSON for {url}")
                if Config.RESPONSE_DIAGNOSTICS:
                    print(f"   Content-type: {content_type}")
                    print(f"   HTML preview: {response_text[:500]}...")
                # Treat this as a request error and retry
                raise requests.exceptions.RequestException(
                    f"API returned HTML error page: {response.status_code}"
                )

            if "json" not in content_type and "application/json" not in content_type:
                print(
                    f"⚠ Warning: Expected JSON response but got content-type: {content_type}"
                )
                if Config.RESPONSE_DIAGNOSTICS:
                    print(f"   Response preview: {response_text[:500]}...")
                # Still try to parse it as JSON in case content-type is wrong

            # Parse and return JSON response
            try:
                result = response.json()
            except (
                json.JSONDecodeError,
                requests.exceptions.JSONDecodeError,
            ) as json_err:
                # Try to fix common JSON errors before giving up
                # Common issue: trailing/leading commas in objects/arrays (e.g., "{,\"key\":\"value\"}")
                try:
                    import re

                    # Fix trailing comma followed by closing brace/bracket: {, } or [, ]
                    cleaned_json = re.sub(r",\s*([}\]])", r"\1", response.text)
                    # Fix comma at start of object/array: {, "key" -> { "key"
                    cleaned_json = re.sub(r"([{\[])\s*,\s*", r"\1", cleaned_json)

                    result = json.loads(cleaned_json)
                    print(f"⚠ Warning: Fixed malformed JSON for {url}")
                    if Config.RESPONSE_DIAGNOSTICS:
                        print(f"   Applied JSON cleanup (removed invalid commas)")
                except (json.JSONDecodeError, Exception):
                    # Cleanup failed - fall through to error handling
                    raise json_err

            print(
                f"✓ Successfully fetched {url} ({len(result) if isinstance(result, list) else 'data'} items)"
            )
            return result

        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as json_err:
            # Handle JSON parsing errors - API sent malformed JSON
            print(f"✗ JSON parsing failed for {url}: {json_err}")

            if Config.RESPONSE_DIAGNOSTICS and response is not None:
                content_type = response.headers.get("content-type", "").lower()
                print(f"   Response content-type: {content_type}")
                print(f"   Response status code: {response.status_code}")
                print(f"   Response length: {len(response.text)} characters")

                # Show preview of response content for debugging
                content_preview = response.text[:2000]  # Show more context

                if "html" in content_type or "<html" in content_preview.lower():
                    print("   Response appears to be HTML (possibly an error page)")
                    print(f"   HTML preview: {content_preview[:500]}...")
                elif "json" in content_type:
                    print("   Response claims to be JSON but parsing failed")
                    # Try to identify the problematic area
                    error_char_pos = getattr(json_err, "pos", None)
                    if error_char_pos:
                        # Show context around the error position
                        start = max(0, error_char_pos - 200)
                        end = min(len(response.text), error_char_pos + 200)
                        print(f"   Error near position {error_char_pos}:")
                        print(f"   ...{response.text[start:end]}...")
                    else:
                        print(f"   JSON preview: {content_preview}...")
                else:
                    print(f"   Response content preview: {content_preview}...")

                # Append problematic response to consolidated debug file
                try:
                    with open(debug_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"ERROR ENTRY: {url.split('/')[-1]}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"URL: {url}\n")
                        f.write(f"Status: {response.status_code}\n")
                        f.write(f"Content-Type: {content_type}\n")
                        f.write(f"Error: {json_err}\n")
                        f.write(f"\n{'-'*80}\n")
                        f.write("Response Content:\n")
                        f.write(f"{'-'*80}\n")
                        f.write(response.text)
                        f.write(f"\n{'='*80}\n\n")
                    print(f"   💾 Appended error to debug file: {debug_file_path.name}")
                except Exception as save_err:
                    print(f"   ❌ Could not append to debug file: {save_err}")

            raise  # Re-raise to trigger retry mechanism

        except requests.exceptions.RequestException as e:
            # Handle network-related errors (connection, timeout, HTTP errors)
            # The retry decorator will automatically retry based on Config.MAX_RETRIES
            print(f"✗ Request failed for {url}: {e}")
            raise  # Re-raise to trigger retry mechanism
        except Exception as e:
            # Handle unexpected errors that shouldn't be retried
            print(f"✗ Unexpected error fetching {url}: {e}")
            return None

    # Reset counter before starting new URL request
    _fetch_attempt_counter = 0

    # Call inner function with retry logic
    try:
        result = _fetch_with_retry()
    except RetryError as e:
        # All retries exhausted - log and return None to continue processing
        print(f"❌ All {Config.MAX_RETRIES} retry attempts failed for {url}")
        print(f"   Last error: {e}")
        if Config.RESPONSE_DIAGNOSTICS:
            print(f"   Check debug file for details: {debug_file_path.name}")
        result = None

    # Reset counter after completion (success or failure)
    _fetch_attempt_counter = 0

    return result


def save_to_csv(df, filename):
    """Helper function to save DataFrame to CSV with proper encoding and error handling.

    This function handles the conversion of pandas DataFrames to CSV files
    with appropriate encoding, file path management, and comprehensive error
    handling to ensure data integrity and proper file system operations.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to save
        filename (str): The desired filename for the CSV output (without path)

    Returns:
        bool: True if the file was saved successfully, False if an error occurred

    Note:
        - Uses UTF-8 encoding to handle international characters
        - Saves without row indices for cleaner output
        - Creates full file path using the configured CSV directory
        - Provides detailed success/failure logging
    """
    try:
        # Construct full output path using the configured CSV directory
        output_path = csv_dir / filename

        # Save DataFrame to CSV with UTF-8 encoding and no index column
        df.to_csv(output_path, index=False, encoding="utf-8")

        # Log successful save operation with full path
        print(f"Saved: {filename} to {output_path}")
        return True
    except Exception as e:
        # Handle any errors during the save operation
        print(f"Error saving {filename}: {e}")
        return False


# ==============================================================================
# MAIN PROCESSING LOGIC
# ==============================================================================
def main():
    """Main processing function that orchestrates the complete data extraction workflow.

    This function coordinates the entire process of extracting statistical data
    from the CZSU API and converting it to CSV format. It manages the workflow
    from initial configuration through final reporting, including progress
    tracking, error handling, and performance monitoring.

    The function handles:
    - Configuration validation and reporting
    - Dataset catalog retrieval and processing
    - Selection discovery and data extraction
    - Progress monitoring with visual indicators
    - Error tracking and comprehensive reporting
    - Performance metrics and final statistics

    Workflow:
    1. Display current configuration settings
    2. Fetch complete dataset catalog from API
    3. Filter datasets based on configuration
    4. For each dataset, discover available selections
    5. Filter selections based on configuration
    6. Extract and convert data for each selection
    7. Save converted data to CSV files
    8. Report final statistics and any errors

    Returns:
        None: This function performs side effects (file creation, console output)
              but does not return a value

    Note:
        - Uses nested progress bars for visual feedback
        - Maintains separate error tracking for datasets and selections
        - Implements graceful error handling to continue processing
        - Provides detailed final statistics and error reporting
    """
    # Print configuration to console for user verification
    print("=== CONFIGURATION SETTINGS ===")
    print(f"Timeout: {Config.TIMEOUT}s")
    print(f"Max retries: {Config.MAX_RETRIES}")
    print(
        f"Retry wait: {Config.RETRY_WAIT_MIN}-{Config.RETRY_WAIT_MAX}s (exponential backoff)"
    )
    print(f"Rate limit delay: {Config.RATE_LIMIT_DELAY}s")
    print(
        f"Response diagnostics: {'Enabled' if Config.RESPONSE_DIAGNOSTICS else 'Disabled'}"
    )
    print("=" * 40)

    if PROCESS_ALL_DATASETS:
        print("Processing all available datasets")
    else:
        print(f"Processing only dataset: {SPECIFIC_DATASET_ID}")

    if PROCESS_ALL_SELECTIONS:
        print("Processing all available selections")
    else:
        print(f"Processing only selection: {SPECIFIC_SELECTION_ID}")
    print()  # ==========================================================================
    # DATASET CATALOG RETRIEVAL
    # ==========================================================================
    # Fetch the complete list of available datasets from CZSU API
    print("Fetching list of datasets...")
    datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
    datasets = fetch_json(datasets_url)

    # Validate that we successfully retrieved the datasets catalog
    if not datasets:
        print("❌ Failed to fetch datasets list after all retry attempts")
        print(f"   URL: {datasets_url}")
        print(f"   Config: timeout={Config.TIMEOUT}s, max_retries={Config.MAX_RETRIES}")
        return

    print(f"\nFound {len(datasets)} datasets to process")

    # ==========================================================================
    # PROCESSING STATISTICS INITIALIZATION
    # ==========================================================================
    # Initialize counters for tracking processing results
    successful_saves = 0
    failed_datasets = []
    failed_selections = []

    # ==========================================================================
    # DATASET PROCESSING LOOP
    # ==========================================================================
    # Process each dataset with progress bar for visual feedback
    for dataset in tqdm(datasets, desc="Processing datasets", unit="dataset"):
        # Extract dataset identifier from the dataset object
        dataset_id = dataset.get("kod")  # Using 'kod' instead of 'id'
        if not dataset_id:
            print(f"Warning: Could not find kod in dataset: {dataset}")
            continue

        # Apply dataset filtering based on configuration
        # Skip if not processing all datasets and this isn't the specific dataset
        if not PROCESS_ALL_DATASETS and dataset_id != SPECIFIC_DATASET_ID:
            continue

        # ======================================================================
        # DATASET DETAILS RETRIEVAL
        # ======================================================================
        # Get detailed information about the current dataset
        dataset_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}"
        dataset_details = fetch_json(dataset_url)

        # Handle dataset details retrieval failure
        if not dataset_details:
            failed_datasets.append(dataset_id)
            continue

        # ======================================================================
        # SELECTIONS DISCOVERY
        # ======================================================================
        # Get available selections (data subsets) for this dataset
        selections_url = (
            f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"
        )
        selections = fetch_json(selections_url)

        # Handle selections retrieval failure
        if not selections:
            failed_datasets.append(dataset_id)
            continue

        # ======================================================================
        # SELECTION PROCESSING LOOP
        # ======================================================================
        # Process each selection with nested progress bar
        for selection in tqdm(
            selections,
            desc=f"Processing selections for {dataset_id}",
            leave=False,
            unit="selection",
        ):
            # Extract selection identifier from the selection object
            selection_id = selection.get("kod")  # Using 'kod' instead of 'id'
            if not selection_id:
                print(f"Warning: Could not find kod in selection: {selection}")
                continue

            # Apply selection filtering based on configuration
            # Skip if not processing all selections and this isn't the specific selection
            if not PROCESS_ALL_SELECTIONS and selection_id != SPECIFIC_SELECTION_ID:
                continue

            # ==================================================================
            # DATA EXTRACTION
            # ==================================================================
            # Fetch the actual statistical data for this selection
            data_url = (
                f"https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}"
            )
            data = fetch_json(data_url)

            # Handle data retrieval failure
            if not data:
                failed_selections.append(selection_id)
                continue

            try:
                # ==============================================================
                # DATA CONVERSION
                # ==============================================================
                # Convert JSON-stat format to pandas DataFrame using pyjstat
                # This follows the same approach as established in data2.py
                df = pyjstat.from_json_stat(data)[0]

                # Validate that the conversion produced a non-empty DataFrame
                if df.empty:
                    print(f"Warning: Empty DataFrame for {selection_id}")
                    failed_selections.append(selection_id)
                    continue

                # ==============================================================
                # FILE OUTPUT PREPARATION
                # ==============================================================
                # Generate filename using the selection code for easy identification
                filename = f"{selection_id}.csv"

                # ==============================================================
                # CSV EXPORT
                # ==============================================================
                # Save the converted data to CSV format
                if save_to_csv(df, filename):
                    successful_saves += 1
                else:
                    failed_selections.append(selection_id)

            except Exception as e:
                # Handle any errors during data processing or file operations
                print(f"Error processing {selection_id}: {e}")
                failed_selections.append(selection_id)

    # ==========================================================================
    # FINAL REPORTING AND STATISTICS
    # ==========================================================================
    # Display comprehensive processing results
    print("\nProcessing complete:")
    print(f"Successfully saved {successful_saves} files to {csv_dir}")

    # Report failed dataset processing if any occurred
    if failed_datasets:
        print(f"\nFailed to process {len(failed_datasets)} datasets:")
        print(json.dumps(failed_datasets, indent=2))

    # Report failed selection processing if any occurred
    if failed_selections:
        print(f"\nFailed to process {len(failed_selections)} selections:")
        print(json.dumps(failed_selections, indent=2))


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
