module_description = r"""Metadata Schema Extraction for Selection Descriptions

This module provides functionality to extract and save metadata schemas for selection descriptions
from the CSU data catalog API, with support for parallel processing and robust error handling.

Key Features:
-------------
1. Parallel Processing: Uses ThreadPoolExecutor for concurrent API calls with configurable workers.
2. Rate Limiting: Implements request rate limiting to respect API constraints.
3. Error Handling: 
   - Comprehensive error handling with detailed logging
   - Both console and file logging
   - Error type categorization
   - Graceful continuation on individual failures
4. Progress Tracking: 
   - Real-time progress bar
   - Detailed processing statistics
   - Success/failure rate calculation
5. Retry Logic: Implements exponential backoff for failed requests
6. Result Tracking: Maintains detailed processing results in CSV format

Processing Flow:
--------------
1. Initialization:
   - Sets up logging (both file and console)
   - Validates configuration parameters
   - Creates output directories
   - Initializes metrics tracking

2. Dataset Collection:
   - Fetches all datasets from CSU API
   - Validates dataset responses
   - Collects selection codes for each dataset
   - Handles API errors gracefully

3. Processing Preparation:
   - Creates list of selections to process
   - Calculates rate limiting parameters
   - Prepares for parallel processing
   - Initializes progress tracking

4. Parallel Processing:
   - Uses ThreadPoolExecutor for concurrent processing
   - Implements rate limiting between requests
   - Processes each selection:
     a. Checks if schema already exists
     b. Fetches metadata from API
     c. Extracts and validates metadata
     d. Saves to JSON file
   - Implements timeout handling

5. Result Storage:
   - Saves each processed schema to JSON file
   - Maintains UTF-8 encoding
   - Handles file system errors
   - Tracks processing status

6. Completion:
   - Calculates and displays processing statistics:
     - Total selections processed
     - Success/failure counts
     - Skipped (already existing) counts
     - Success rate percentage
   - Reports error distribution
   - Saves detailed results to CSV
   - Lists failed selections

Usage Example:
-------------
# Run the metadata extraction process
python get_metadata_schemas_all_selections.py

Required Environment:
-------------------
- Python 3.7+
- Internet connection to CSU API
- Write permissions for output directory

Output Files:
------------
1. metadata/schemas/*_schema.json: Individual schema files
2. metadata_extraction_results.csv: Detailed processing results
3. metadata_extraction.log: Detailed processing log with timestamps

Error Handling:
-------------
- Individual selection failures don't stop processing
- All errors are logged to both console and file
- Failed selections are tracked in results
- API errors are handled with retry logic
- File system errors are handled gracefully
- Network timeouts are handled with configurable timeout"""

# ===============================================================================
# IMPORTS
# ===============================================================================
# Standard library imports
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd
import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ===============================================================================
# CONFIGURATION AND SETUP
# ===============================================================================
# Set up logging for retry visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global counter for tracking retry attempts across function calls
_fetch_attempt_counter = 0


# Configuration
@dataclass
class Config:
    """Configuration class for metadata extraction process.

    This class defines the configuration parameters for the metadata extraction process,
    including parallel processing settings, rate limiting, and retry behavior to ensure
    robust and reliable interaction with the CZSU API.

    Attributes:
        max_workers (int): Number of parallel processing threads. Default 30.
                          Controls concurrent API requests for faster processing.

        requests_per_minute (int): Maximum API requests per minute. Default 60.
                                  Implements rate limiting to respect API guidelines.

        retry_attempts (int): Number of retry attempts for failed requests. Default 2.
                            Helps handle transient network issues.

        retry_min_wait (int): Minimum wait time between retries in seconds. Default 1.
                            Part of exponential backoff strategy.

        retry_max_wait (int): Maximum wait time between retries in seconds. Default 3.
                            Caps exponential backoff to prevent excessively long waits.

        timeout (int): Request timeout in seconds. Default 15.
                      Prevents indefinite waiting on slow connections.

        selection_batch_size (int): Number of selections to process in each batch. Default 50.
                                   Controls memory usage and progress tracking granularity.

        RESPONSE_DIAGNOSTICS (int): Enable detailed API response diagnostics. Default 1.
                                   Set to 1 to show response content previews, content-type
                                   validation, and detailed error information when API
                                   responses are malformed or unexpected. Set to 0 to disable
                                   for cleaner output.

    Usage:
        These settings are used by the fetch_json function to control API
        request behavior, ensuring reliable communication while being
        respectful to the CZSU API infrastructure.
    """

    max_workers: int = 30
    requests_per_minute: int = 60
    retry_attempts: int = 6
    retry_min_wait: int = 1
    retry_max_wait: int = 3
    timeout: int = 15
    selection_batch_size: int = 50
    RESPONSE_DIAGNOSTICS: int = 1  # Enable detailed response diagnostics (1=yes, 0=no)

    def __post_init__(self):
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if self.RESPONSE_DIAGNOSTICS not in (0, 1):
            raise ValueError("RESPONSE_DIAGNOSTICS must be 0 or 1")


CONFIG = Config()

# ==============================================================================
# DIRECTORY SETUP
# ==============================================================================
# Create schemas directory if it doesn't exist
schemas_dir = Path("metadata/schemas")
schemas_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DEBUG FILE SETUP
# ==============================================================================
# Single consolidated debug file for all API response errors
# File is overwritten on each script execution, errors appended during runtime
debug_file_path = (
    Path(__file__).parent / "debug_response_get_metadata_schemas_all_selections.txt"
)

# Initialize/overwrite debug file at script startup
with open(debug_file_path, "w", encoding="utf-8") as f:
    f.write(f"Debug Response Log for get_metadata_schemas_all_selections.py\n")
    f.write(f"Started: {datetime.now().isoformat()}\n")
    f.write("=" * 80 + "\n\n")


# ===============================================================================
# CORE FUNCTIONS
# ===============================================================================
def fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON data from URL with retry logic, error handling, and response diagnostics.

    This function provides robust API communication with the CZSU endpoints,
    including automatic retry logic for transient failures, proper error handling,
    HTTP status validation, timeout protection, rate limiting, JSON cleanup for
    malformed responses, and debug file generation for troubleshooting.

    The function uses the tenacity library for retry logic with exponential backoff,
    automatically retrying failed requests up to retry_attempts times with increasing
    wait times between attempts.

    Args:
        url (str): The API endpoint URL to fetch data from

    Returns:
        Optional[Dict[str, Any]]: The parsed JSON response data if successful,
                                  None if the request failed after all retry attempts
                                  or returned invalid data

    Raises:
        requests.exceptions.RequestException: Raised and caught by retry decorator
                                             for network-related errors

    Note:
        - Implements configurable timeout from CONFIG.timeout
        - Uses exponential backoff retry strategy from CONFIG settings
        - Implements rate limiting with CONFIG.requests_per_minute
        - Validates HTTP status codes before processing response
        - Handles common API errors gracefully with detailed logging
        - Automatic retry on connection timeouts, network errors, and HTTP errors
        - JSON cleanup for malformed API responses (trailing/leading commas)
        - Debug file generation when RESPONSE_DIAGNOSTICS is enabled
    """
    global _fetch_attempt_counter

    # Inner function with retry logic
    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, json.JSONDecodeError)
        ),
        stop=stop_after_attempt(CONFIG.retry_attempts),
        wait=wait_exponential(min=CONFIG.retry_min_wait, max=CONFIG.retry_max_wait),
        before_sleep=lambda retry_state: print(
            f"⚠ Retrying after error. Attempt {retry_state.attempt_number} of {CONFIG.retry_attempts}"
        ),
        reraise=False,
    )
    def _fetch_with_retry():
        # Increment counter for each attempt
        global _fetch_attempt_counter
        _fetch_attempt_counter += 1
        attempt = _fetch_attempt_counter

        print(
            f"[Attempt {attempt}/{CONFIG.retry_attempts}] Fetching {url} (timeout: {CONFIG.timeout}s)"
        )

        # Initialize response variable outside try block
        response = None

        try:
            # Make HTTP GET request with timeout protection
            response = requests.get(url, timeout=CONFIG.timeout)

            # Validate HTTP status code and raise exception for bad responses
            response.raise_for_status()

            # Implement rate limiting to be respectful to the API
            time.sleep(0.01)

            # Check if response is actually JSON before parsing
            content_type = response.headers.get("content-type", "").lower()
            response_text = response.text

            # Check if response looks like HTML (common API error)
            if (
                "<html" in response_text.lower()
                or "<!doctype html" in response_text.lower()
            ):
                print(f"✗ API returned HTML error page instead of JSON for {url}")
                if CONFIG.RESPONSE_DIAGNOSTICS:
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
                if CONFIG.RESPONSE_DIAGNOSTICS:
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
                    if CONFIG.RESPONSE_DIAGNOSTICS:
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

            if CONFIG.RESPONSE_DIAGNOSTICS and response is not None:
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
            # The retry decorator will automatically retry based on CONFIG.retry_attempts
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
        print(f"❌ All {CONFIG.retry_attempts} retry attempts failed for {url}")
        print(f"   Last error: {e}")
        if CONFIG.RESPONSE_DIAGNOSTICS:
            print(f"   Check debug file for details: {debug_file_path.name}")
        result = None

    # Reset counter after completion (success or failure)
    _fetch_attempt_counter = 0

    return result


def safe_fetch_json(
    url: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Wrapper around fetch_json that handles all types of errors gracefully.

    This function provides a safe way to fetch JSON data by handling all possible
    error types and returning them in a structured way.

    Args:
        url (str): The URL to fetch JSON data from.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]: A tuple containing:
            - The JSON data if successful, None otherwise
            - The error type if an error occurred, None otherwise
            - The error message if an error occurred, None otherwise
    """
    try:
        return fetch_json(url), None, None
    except RetryError as e:
        return None, "RetryError", str(e)
    except requests.exceptions.RequestException as e:
        return None, "RequestError", str(e)
    except json.JSONDecodeError as e:
        return None, "JSONDecodeError", str(e)
    except Exception as e:
        return None, "UnexpectedError", str(e)


def extract_metadata(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract and validate metadata schema from the data.

    This function extracts metadata from the API response and validates
    that all required fields are present and non-empty.

    Args:
        data (Dict[str, Any]): The data to extract metadata from.

    Returns:
        Optional[Dict[str, Any]]: The extracted metadata if valid, None otherwise.
    """
    try:
        # Define required fields
        required_fields = ["version", "class", "href", "label", "source", "id"]

        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            return None

        # Extract metadata with fallback values for optional fields
        metadata = {
            "version": data.get("version"),
            "class": data.get("class"),
            "href": data.get("href"),
            "label": data.get("label"),
            "source": data.get("source"),
            "note": data.get("note", ""),
            "updated": data.get("updated", ""),
            "id": data.get("id"),
            "size": data.get("size", 0),
            "role": data.get("role", ""),
            "dimension": data.get("dimension", {}),
        }

        # Validate the extracted metadata
        if not all(metadata[field] for field in required_fields):
            logger.warning(
                f"Invalid metadata for ID {data.get('id')}: missing required values"
            )
            return None

        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return None


def save_metadata(metadata: Dict[str, Any], output_file: Path) -> bool:
    """Save metadata to JSON file with proper formatting.

    This function saves metadata to a JSON file with:
    - UTF-8 encoding
    - Proper indentation
    - Error handling
    - Logging

    Args:
        metadata (Dict[str, Any]): The metadata to save.
        output_file (Path): The file to save the metadata to.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(output_file, "w", encoding="utf-8", newline="\n") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving metadata to {output_file}: {str(e)}")
        return False


def fetch_selections_for_dataset(dataset_id: str) -> List[Tuple[str, str]]:
    """Fetch selections for a single dataset with error handling.

    This function fetches all selections for a given dataset and handles
    various types of errors that might occur during the process.

    Args:
        dataset_id (str): The ID of the dataset to fetch selections for.

    Returns:
        List[Tuple[str, str]]: List of tuples containing (selection_id, dataset_id).
    """
    selections = []
    selections_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"

    try:
        logger.debug(f"Fetching selections for dataset {dataset_id}")
        data, error_type, error_message = safe_fetch_json(selections_url)

        if data:
            for selection in data:
                selection_id = selection.get("kod")
                if selection_id:
                    selections.append((selection_id, dataset_id))
        else:
            logger.warning(
                f"Failed to fetch selections for dataset {dataset_id}: {error_message}"
            )

    except Exception as e:
        logger.error(f"Error processing dataset {dataset_id}: {str(e)}")

    return selections


def process_selection(selection_id: str, dataset_id: str) -> Dict[str, Any]:
    """Process a single selection and return its status.

    This function processes a selection by:
    1. Checking if the schema already exists
    2. Fetching metadata from the API
    3. Extracting and validating metadata
    4. Saving to a JSON file

    Args:
        selection_id (str): The ID of the selection to process.
        dataset_id (str): The ID of the dataset the selection belongs to.

    Returns:
        Dict[str, Any]: A dictionary containing the processing status and details.
    """
    metadata_url = f"https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}"
    output_file = Path("metadata/schemas") / f"{selection_id}_schema.json"

    # Skip if file already exists
    if output_file.exists():
        return {
            "selection_id": selection_id,
            "dataset_id": dataset_id,
            "url": metadata_url,
            "status": "skipped",
            "error_type": None,
            "error_message": None,
            "processed_at": datetime.now().isoformat(),
        }

    result = {
        "selection_id": selection_id,
        "dataset_id": dataset_id,
        "url": metadata_url,
        "status": "pending",
        "error_type": None,
        "error_message": None,
        "processed_at": None,
    }

    try:
        data, error_type, error_message = safe_fetch_json(metadata_url)

        if data is None:
            result.update(
                {
                    "status": "failed",
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            return result

        metadata = extract_metadata(data)
        if metadata is None:
            result.update(
                {
                    "status": "failed",
                    "error_type": "MetadataExtractionError",
                    "error_message": "Failed to extract valid metadata",
                }
            )
            return result

        if save_metadata(metadata, output_file):
            result.update(
                {"status": "success", "processed_at": datetime.now().isoformat()}
            )
        else:
            result.update(
                {
                    "status": "failed",
                    "error_type": "SaveError",
                    "error_message": "Failed to save metadata to file",
                }
            )

    except Exception as e:
        result.update(
            {
                "status": "failed",
                "error_type": "UnexpectedError",
                "error_message": str(e),
            }
        )

    return result


def main():
    """Main function to orchestrate the metadata extraction process.

    This function:
    1. Sets up the environment
    2. Fetches all datasets
    3. Collects selections
    4. Processes selections in parallel
    5. Generates and saves results
    """
    logger.info("Starting metadata extraction process")

    # ==============================================================================
    # CONFIGURATION DISPLAY
    # ==============================================================================
    # Print configuration to console for user verification
    print("=== CONFIGURATION SETTINGS ===")
    print(f"Max workers: {CONFIG.max_workers}")
    print(f"Requests per minute: {CONFIG.requests_per_minute}")
    print(f"Timeout: {CONFIG.timeout}s")
    print(f"Retry attempts: {CONFIG.retry_attempts}")
    print(
        f"Retry wait: {CONFIG.retry_min_wait}-{CONFIG.retry_max_wait}s (exponential backoff)"
    )
    print(
        f"Response diagnostics: {'Enabled' if CONFIG.RESPONSE_DIAGNOSTICS else 'Disabled'}"
    )
    print("=" * 40)
    print()

    # Verify schemas directory
    logger.info(f"Using schemas directory: {schemas_dir}")

    # ==========================================================================
    # DATASET CATALOG RETRIEVAL
    # ==========================================================================
    # Fetch the complete list of available datasets from CZSU API
    print("Fetching list of datasets...")
    datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
    logger.info(f"Fetching datasets from {datasets_url}")

    try:
        datasets, error_type, error_message = safe_fetch_json(datasets_url)
        if not datasets:
            print(f"❌ Failed to fetch datasets list after all retry attempts")
            print(f"   Error: {error_type} - {error_message}")
            logger.error(f"Failed to fetch datasets: {error_message}")
            return
        print(f"\n✓ Found {len(datasets)} datasets to process")
        logger.info(f"Successfully fetched {len(datasets)} datasets")
    except Exception as e:
        print(f"❌ Unexpected error fetching datasets: {str(e)}")
        logger.error(f"Unexpected error fetching datasets: {str(e)}")
        return

    # ==========================================================================
    # SELECTIONS DISCOVERY
    # ==========================================================================
    # Collect all selections from all datasets for processing
    all_selections = []
    print("\nCollecting selections from datasets...")
    logger.info("Collecting selections from datasets")

    # Process all datasets at once using parallel processing
    with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        futures = {
            executor.submit(
                fetch_selections_for_dataset, dataset.get("kod")
            ): dataset.get("kod")
            for dataset in datasets
            if dataset.get("kod")
        }

        for future in as_completed(futures):
            try:
                selections = future.result()
                all_selections.extend(selections)
            except Exception as e:
                print(
                    f"⚠ Warning: Error processing dataset {futures[future]}: {str(e)}"
                )
                logger.error(f"Error processing dataset {futures[future]}: {str(e)}")

    print(f"✓ Collected {len(all_selections)} selections to process\n")
    logger.info(f"Collected {len(all_selections)} selections to process")

    # ==========================================================================
    # PARALLEL METADATA EXTRACTION
    # ==========================================================================
    # Process all selections in parallel with rate limiting
    results = []
    delay_between_requests = 60 / CONFIG.requests_per_minute

    print(f"Starting parallel processing with {CONFIG.max_workers} workers...")
    print(f"Rate limiting: {CONFIG.requests_per_minute} requests per minute")
    logger.info(f"Starting parallel processing with {CONFIG.max_workers} workers")
    logger.info(f"Rate limiting: {CONFIG.requests_per_minute} requests per minute")

    # Process all selections at once using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        futures = {
            executor.submit(process_selection, selection_id, dataset_id): (
                selection_id,
                dataset_id,
            )
            for selection_id, dataset_id in all_selections
        }

        with tqdm(total=len(all_selections), desc="Processing selections") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    selection_id, dataset_id = futures[future]
                    results.append(
                        {
                            "selection_id": selection_id,
                            "dataset_id": dataset_id,
                            "status": "failed",
                            "error_type": "ProcessingError",
                            "error_message": str(e),
                        }
                    )
                pbar.update(1)
                time.sleep(delay_between_requests)

    # ==========================================================================
    # FINAL REPORTING AND STATISTICS
    # ==========================================================================
    # Create and display results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate comprehensive statistics
    total = len(results_df)
    successful = len(results_df[results_df["status"] == "success"])
    failed = len(results_df[results_df["status"] == "failed"])
    skipped = len(results_df[results_df["status"] == "skipped"])

    # Display comprehensive processing results
    print("\n" + "=" * 70)
    print("METADATA EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total selections processed: {total}")
    print(f"Successfully extracted: {successful}")
    print(f"Already existed (skipped): {skipped}")
    print(f"Failed: {failed}")
    if total > 0:
        print(f"Success rate: {(successful/total)*100:.2f}%")
    print(f"Output directory: {schemas_dir}")
    print("=" * 70)

    # Log final statistics
    logger.info(
        f"Processing complete: {successful}/{total} successful, {failed} failed, {skipped} skipped"
    )

    # Report error distribution if any failures occurred
    if failed > 0:
        print("\nERROR DISTRIBUTION:")
        print("-" * 70)
        error_counts = results_df[results_df["status"] == "failed"][
            "error_type"
        ].value_counts()
        print(error_counts)

        # Save detailed results to CSV
        results_csv = "metadata_extraction_results.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Detailed results saved to {results_csv}")
        print(f"\nDetailed results saved to: {results_csv}")

        # Display sample of failed selections (first 10)
        print("\nSAMPLE FAILED SELECTIONS (first 10):")
        print("-" * 70)
        failed_df = results_df[results_df["status"] == "failed"].head(10)
        for _, row in failed_df.iterrows():
            print(f"Dataset: {row['dataset_id']}, Selection: {row['selection_id']}")
            print(f"  Error: {row['error_type']} - {row['error_message']}")
            print()
    else:
        # Save results even if all successful
        results_csv = "metadata_extraction_results.csv"
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Processing results saved to {results_csv}")
        print(f"\n✓ All processing successful! Results saved to: {results_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
