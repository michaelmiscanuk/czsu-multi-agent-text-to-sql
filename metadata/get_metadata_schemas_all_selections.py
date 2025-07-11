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

#===============================================================================
# IMPORTS
#===============================================================================
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

#===============================================================================
# CONFIGURATION AND SETUP
#===============================================================================
# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_extraction.log'),
        logging.StreamHandler(sys.stdout)  # Ensure output goes to console
    ]
)

logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration class for metadata extraction process.
    
    This class defines the configuration parameters for the metadata extraction process,
    including parallel processing settings, rate limiting, and retry behavior.
    
    Attributes:
        max_workers (int): Number of parallel processing threads
        requests_per_minute (int): Maximum API requests per minute
        retry_attempts (int): Number of retry attempts for failed requests
        retry_min_wait (int): Minimum wait time between retries in seconds
        retry_max_wait (int): Maximum wait time between retries in seconds
        timeout (int): Request timeout in seconds
        selection_batch_size (int): Number of selections to process in each batch
    """
    max_workers: int = 30
    requests_per_minute: int = 60
    retry_attempts: int = 2
    retry_min_wait: int = 1
    retry_max_wait: int = 3
    timeout: int = 15
    selection_batch_size: int = 50
    
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

CONFIG = Config()

#===============================================================================
# CORE FUNCTIONS
#===============================================================================
@retry(
    stop=stop_after_attempt(CONFIG.retry_attempts),
    wait=wait_exponential(multiplier=1, min=CONFIG.retry_min_wait, max=CONFIG.retry_max_wait),
    retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying after error. Attempt {retry_state.attempt_number} of {CONFIG.retry_attempts}")
)
def fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON data from URL with retry logic and error handling.
    
    This function implements retry logic with exponential backoff for failed requests.
    It handles various types of errors and provides detailed logging.
    
    Args:
        url (str): The URL to fetch JSON data from.
        
    Returns:
        Optional[Dict[str, Any]]: The JSON data if successful, None otherwise.
        
    Raises:
        requests.exceptions.RequestException: If the request fails after all retries.
        json.JSONDecodeError: If the response is not valid JSON.
    """
    try:
        logger.debug(f"Fetching URL: {url}")
        response = requests.get(url, timeout=CONFIG.timeout)
        response.raise_for_status()
        time.sleep(0.01)  # Rate limiting
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        raise  # Let tenacity handle the retry
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {url}: {str(e)}")
        raise  # Let tenacity handle the retry
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {str(e)}")
        return None

def safe_fetch_json(url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
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
            "dimension": data.get("dimension", {})
        }

        # Validate the extracted metadata
        if not all(metadata[field] for field in required_fields):
            logger.warning(f"Invalid metadata for ID {data.get('id')}: missing required values")
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
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
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
                selection_id = selection.get('kod')
                if selection_id:
                    selections.append((selection_id, dataset_id))
        else:
            logger.warning(f"Failed to fetch selections for dataset {dataset_id}: {error_message}")
            
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
    output_file = Path('metadata/schemas') / f"{selection_id}_schema.json"
    
    # Skip if file already exists
    if output_file.exists():
        return {
            'selection_id': selection_id,
            'dataset_id': dataset_id,
            'url': metadata_url,
            'status': 'skipped',
            'error_type': None,
            'error_message': None,
            'processed_at': datetime.now().isoformat()
        }
    
    result = {
        'selection_id': selection_id,
        'dataset_id': dataset_id,
        'url': metadata_url,
        'status': 'pending',
        'error_type': None,
        'error_message': None,
        'processed_at': None
    }
    
    try:
        data, error_type, error_message = safe_fetch_json(metadata_url)
        
        if data is None:
            result.update({
                'status': 'failed',
                'error_type': error_type,
                'error_message': error_message
            })
            return result
            
        metadata = extract_metadata(data)
        if metadata is None:
            result.update({
                'status': 'failed',
                'error_type': 'MetadataExtractionError',
                'error_message': 'Failed to extract valid metadata'
            })
            return result
            
        if save_metadata(metadata, output_file):
            result.update({
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            })
        else:
            result.update({
                'status': 'failed',
                'error_type': 'SaveError',
                'error_message': 'Failed to save metadata to file'
            })
            
    except Exception as e:
        result.update({
            'status': 'failed',
            'error_type': 'UnexpectedError',
            'error_message': str(e)
        })
    
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
    
    # Create schemas directory if it doesn't exist
    schemas_dir = Path('metadata/schemas')
    schemas_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified schemas directory at {schemas_dir}")
    
    # 1. Fetch all datasets
    datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
    logger.info(f"Fetching datasets from {datasets_url}")
    
    try:
        datasets, error_type, error_message = safe_fetch_json(datasets_url)
        if not datasets:
            logger.error(f"Failed to fetch datasets: {error_message}")
            return
        logger.info(f"Successfully fetched {len(datasets)} datasets")
    except Exception as e:
        logger.error(f"Unexpected error fetching datasets: {str(e)}")
        return

    # Prepare for parallel processing
    all_selections = []
    logger.info("Collecting selections from datasets")
    
    # Process all datasets at once instead of batches
    with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        futures = {
            executor.submit(fetch_selections_for_dataset, dataset.get('kod')): dataset.get('kod')
            for dataset in datasets if dataset.get('kod')
        }
        
        for future in as_completed(futures):
            try:
                selections = future.result()
                all_selections.extend(selections)
            except Exception as e:
                logger.error(f"Error processing dataset {futures[future]}: {str(e)}")

    logger.info(f"Collected {len(all_selections)} selections to process")

    # Process selections in parallel
    results = []
    delay_between_requests = 60 / CONFIG.requests_per_minute
    
    logger.info(f"Starting parallel processing with {CONFIG.max_workers} workers")
    logger.info(f"Rate limiting: {CONFIG.requests_per_minute} requests per minute")
    
    # Process all selections at once instead of batches
    with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        futures = {
            executor.submit(process_selection, selection_id, dataset_id): (selection_id, dataset_id)
            for selection_id, dataset_id in all_selections
        }
        
        with tqdm(total=len(all_selections), desc="Processing selections") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    selection_id, dataset_id = futures[future]
                    results.append({
                        'selection_id': selection_id,
                        'dataset_id': dataset_id,
                        'status': 'failed',
                        'error_type': 'ProcessingError',
                        'error_message': str(e)
                    })
                pbar.update(1)
                time.sleep(delay_between_requests)

    # Create and display results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    total = len(results_df)
    successful = len(results_df[results_df['status'] == 'success'])
    failed = len(results_df[results_df['status'] == 'failed'])
    skipped = len(results_df[results_df['status'] == 'skipped'])
    
    # Print summary
    print("\nProcessing Summary:")
    print("=" * 50)
    print(f"Total selections: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Already existed (skipped): {skipped}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/total)*100:.2f}%")
    print("=" * 50)
    
    # Print error distribution
    if failed > 0:
        print("\nError Distribution:")
        print("-" * 50)
        error_counts = results_df[results_df['status'] == 'failed']['error_type'].value_counts()
        print(error_counts)
        
        # Save detailed results to CSV
        results_df.to_csv('metadata_extraction_results.csv', index=False)
        print("\nDetailed results saved to 'metadata_extraction_results.csv'")
        
        # Display failed URLs in a more compact format
        print("\nFailed Selections:")
        print("-" * 50)
        failed_df = results_df[results_df['status'] == 'failed']
        for _, row in failed_df.iterrows():
            print(f"{row['selection_id']}: {row['error_type']} - {row['error_message']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)