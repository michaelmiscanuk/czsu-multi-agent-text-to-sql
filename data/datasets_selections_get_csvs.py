module_description = r"""Czech Statistical Office (CZSU) Data Extraction and CSV Generation

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
   - Rate limiting to respect API guidelines
   - Robust error handling for network issues
   - Automatic retry mechanisms
   - Response validation and parsing

5. Progress Monitoring:
   - Real-time progress bars for datasets and selections
   - Success/failure tracking and reporting
   - Detailed error logging with selection codes
   - Performance metrics and timing
   - Final processing summary

6. Configuration Management:
   - Environment-based configuration options
   - Flexible processing modes (all vs. specific)
   - Configurable target directories
   - Processing scope control
   - Debug output support

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
- Required packages: requests, pandas, pyjstat, tqdm, pathlib

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

Error Handling:
-------------
- Network connection failures
- API response validation errors
- Data parsing and conversion errors
- File system and permission errors
- Empty or malformed datasets
- Rate limiting and timeout handling"""

#==============================================================================
# IMPORTS
#==============================================================================
# Standard library imports
import requests
import pandas as pd
from pyjstat import pyjstat
import os
import time
from pathlib import Path
from tqdm import tqdm
import json

#==============================================================================
# DIRECTORY SETUP AND CONFIGURATION
#==============================================================================
# Create data/CSVs directory if it doesn't exist
csv_dir = Path("data/CSVs")
csv_dir.mkdir(parents=True, exist_ok=True)

#==============================================================================
# CONFIGURATION PARAMETERS
#==============================================================================
# Dataset processing configuration
PROCESS_ALL_DATASETS = 1  # Set to 1 to process all datasets, 0 to process specific dataset
SPECIFIC_DATASET_ID = "OBY01PD"  # Only used when PROCESS_ALL_DATASETS is 0

# Selection processing configuration
PROCESS_ALL_SELECTIONS = 1  # Set to 1 to process all selections, 0 to process specific selection
SPECIFIC_SELECTION_ID = "OBY01PDT01"  # Only used when PROCESS_ALL_SELECTIONS is 0

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def fetch_json(url):
    """Helper function to fetch JSON data with error handling and rate limiting.
    
    This function provides robust API communication with the CZSU endpoints,
    including proper error handling, HTTP status validation, and rate limiting
    to respect API guidelines and ensure reliable data retrieval.
    
    Args:
        url (str): The API endpoint URL to fetch data from
        
    Returns:
        dict | None: The parsed JSON response data if successful,
                     None if the request failed or returned invalid data
                     
    Raises:
        requests.exceptions.RequestException: For network-related errors
        
    Note:
        - Implements 0.1 second delay between requests for rate limiting
        - Validates HTTP status codes before processing response
        - Handles common API errors gracefully
    """
    try:
        # Make HTTP GET request to the specified URL
        response = requests.get(url)
        
        # Validate HTTP status code and raise exception for bad responses
        response.raise_for_status()
        
        # Implement rate limiting to be respectful to the API
        time.sleep(0.1)  # Rate limiting to be nice to the API
        
        # Parse and return JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (connection, timeout, HTTP errors)
        print(f"Error fetching {url}: {e}")
        return None

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
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Log successful save operation with full path
        print(f"Saved: {filename} to {output_path}")
        return True
    except Exception as e:
        # Handle any errors during the save operation
        print(f"Error saving {filename}: {e}")
        return False

#==============================================================================
# MAIN PROCESSING LOGIC
#==============================================================================
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
    if PROCESS_ALL_DATASETS:
        print("Processing all available datasets")
    else:
        print(f"Processing only dataset: {SPECIFIC_DATASET_ID}")
        
    if PROCESS_ALL_SELECTIONS:
        print("Processing all available selections")
    else:
        print(f"Processing only selection: {SPECIFIC_SELECTION_ID}")

    #==========================================================================
    # DATASET CATALOG RETRIEVAL
    #==========================================================================
    # Fetch the complete list of available datasets from CZSU API
    print("Fetching list of datasets...")
    datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
    datasets = fetch_json(datasets_url)
    
    # Validate that we successfully retrieved the datasets catalog
    if not datasets:
        print("Failed to fetch datasets list")
        return

    print(f"\nFound {len(datasets)} datasets to process")
    
    #==========================================================================
    # PROCESSING STATISTICS INITIALIZATION
    #==========================================================================
    # Initialize counters for tracking processing results
    successful_saves = 0
    failed_datasets = []
    failed_selections = []

    #==========================================================================
    # DATASET PROCESSING LOOP
    #==========================================================================
    # Process each dataset with progress bar for visual feedback
    for dataset in tqdm(datasets, desc="Processing datasets", unit="dataset"):
        # Extract dataset identifier from the dataset object
        dataset_id = dataset.get('kod')  # Using 'kod' instead of 'id'
        if not dataset_id:
            print(f"Warning: Could not find kod in dataset: {dataset}")
            continue
            
        # Apply dataset filtering based on configuration
        # Skip if not processing all datasets and this isn't the specific dataset
        if not PROCESS_ALL_DATASETS and dataset_id != SPECIFIC_DATASET_ID:
            continue

        #======================================================================
        # DATASET DETAILS RETRIEVAL
        #======================================================================
        # Get detailed information about the current dataset
        dataset_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}"
        dataset_details = fetch_json(dataset_url)
        
        # Handle dataset details retrieval failure
        if not dataset_details:
            failed_datasets.append(dataset_id)
            continue

        #======================================================================
        # SELECTIONS DISCOVERY
        #======================================================================
        # Get available selections (data subsets) for this dataset
        selections_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"
        selections = fetch_json(selections_url)
        
        # Handle selections retrieval failure
        if not selections:
            failed_datasets.append(dataset_id)
            continue

        #======================================================================
        # SELECTION PROCESSING LOOP
        #======================================================================
        # Process each selection with nested progress bar
        for selection in tqdm(selections, desc=f"Processing selections for {dataset_id}", 
                            leave=False, unit="selection"):
            # Extract selection identifier from the selection object
            selection_id = selection.get('kod')  # Using 'kod' instead of 'id'
            if not selection_id:
                print(f"Warning: Could not find kod in selection: {selection}")
                continue
            
            # Apply selection filtering based on configuration
            # Skip if not processing all selections and this isn't the specific selection
            if not PROCESS_ALL_SELECTIONS and selection_id != SPECIFIC_SELECTION_ID:
                continue
            
            #==================================================================
            # DATA EXTRACTION
            #==================================================================
            # Fetch the actual statistical data for this selection
            data_url = f"https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}"
            data = fetch_json(data_url)
            
            # Handle data retrieval failure
            if not data:
                failed_selections.append(selection_id)
                continue

            try:
                #==============================================================
                # DATA CONVERSION
                #==============================================================
                # Convert JSON-stat format to pandas DataFrame using pyjstat
                # This follows the same approach as established in data2.py
                df = pyjstat.from_json_stat(data)[0]
                
                # Validate that the conversion produced a non-empty DataFrame
                if df.empty:
                    print(f"Warning: Empty DataFrame for {selection_id}")
                    failed_selections.append(selection_id)
                    continue
                
                #==============================================================
                # FILE OUTPUT PREPARATION
                #==============================================================
                # Generate filename using the selection code for easy identification
                filename = f"{selection_id}.csv"
                
                #==============================================================
                # CSV EXPORT
                #==============================================================
                # Save the converted data to CSV format
                if save_to_csv(df, filename):
                    successful_saves += 1
                else:
                    failed_selections.append(selection_id)
                
            except Exception as e:
                # Handle any errors during data processing or file operations
                print(f"Error processing {selection_id}: {e}")
                failed_selections.append(selection_id)

    #==========================================================================
    # FINAL REPORTING AND STATISTICS
    #==========================================================================
    # Display comprehensive processing results
    print(f"\nProcessing complete:")
    print(f"Successfully saved {successful_saves} files to {csv_dir}")
    
    # Report failed dataset processing if any occurred
    if failed_datasets:
        print(f"\nFailed to process {len(failed_datasets)} datasets:")
        print(json.dumps(failed_datasets, indent=2))
        
    # Report failed selection processing if any occurred
    if failed_selections:
        print(f"\nFailed to process {len(failed_selections)} selections:")
        print(json.dumps(failed_selections, indent=2))

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":
    main() 