import json
import requests
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from datetime import datetime
import sys

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
    """Configuration class for processing."""
    max_workers: int = 10
    requests_per_minute: int = 60
    retry_attempts: int = 2
    retry_min_wait: int = 1
    retry_max_wait: int = 3
    timeout: int = 15
    selection_batch_size: int = 50
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")

CONFIG = Config()

@retry(
    stop=stop_after_attempt(CONFIG.retry_attempts),
    wait=wait_exponential(multiplier=1, min=CONFIG.retry_min_wait, max=CONFIG.retry_max_wait),
    retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying after error. Attempt {retry_state.attempt_number} of {CONFIG.retry_attempts}")
)
def fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON data with retry logic and better error handling."""
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
    """Wrapper around fetch_json that handles RetryError gracefully."""
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
    """Extract metadata schema from the data with validation."""
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
    """Save metadata to file with error handling."""
    try:
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving metadata to {output_file}: {str(e)}")
        return False

def fetch_selections_for_dataset(dataset_id: str) -> List[Tuple[str, str]]:
    """Fetch selections for a single dataset with error handling."""
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
    """Process a single selection and return its status."""
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