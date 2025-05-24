module_description = r"""Parallel DataFrame Processing with Azure OpenAI

This module provides functionality to process pandas DataFrames in parallel using Azure OpenAI.

Key Features:
-------------
1. Fully Dynamic Column Handling: All DataFrame columns are passed as parameters to the prompt template.
2. Parallel Processing: Uses ThreadPoolExecutor for concurrent API calls.
3. Rate Limiting: Implements request rate limiting to respect API constraints.
4. Error Handling: Retries failed requests and captures errors.
5. Metrics Collection: Tracks processing time and success/failure rates.
6. External Template Loading: Loads prompt templates from external text files with UTF-8 support.

How it works:
------------
1. Input: Takes a DataFrame where column names match placeholders in the prompt template.
2. Configuration: Uses environment variables for Azure OpenAI setup.
3. Processing:
   - Each row is processed as a separate API call.
   - All columns are passed dynamically to the prompt template.
   - Results are collected and merged back into the DataFrame.
4. Output: Returns original DataFrame with new response column.

Usage Example:
-------------
# Create a template file with placeholders
# File: PROMPT_TEMPLATE.txt
# Content:
# You are a helpful assistant.
# Topic: {topic}
# Task: {task}
# Style: {style}
# Additional Context: {context}

# Create DataFrame with matching column names
test_df = pd.DataFrame({
    'topic': ['Quantum Computing', 'AI'],
    'task': ['Explain basics', 'Compare with humans'],
    'style': ['beginner-friendly', 'technical'],
    'context': ['high school', 'graduates']
})

# Process the DataFrame
result_df = process_dataframe_parallel(
    test_df,
    output_column='response',
    max_workers=3,
    requests_per_minute=30
)"""

#===============================================================================
# IMPORTS
#===============================================================================
import pandas as pd
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import tqdm as tqdm_module
import sys
import csv
from pathlib import Path
import sqlite3

# Setup base directory for imports
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Add the base directory to sys.path for local imports
sys.path.append(str(BASE_DIR))

# Import the get_azure_llm function
from my_agent.utils.models import get_azure_llm

#===============================================================================
# CUSTOM EXCEPTIONS
#===============================================================================
class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass

class ProcessingError(Exception):
    """Raised when processing fails.""" 
    pass

#===============================================================================
# CONFIGURATION AND SETUP
#===============================================================================
# Simplified logging setup
logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
logger = logging.getLogger(__name__)

# Load and validate environment in one step
load_dotenv()
for var in ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY']:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Consolidated constants
CONFIG = {
    'MAX_WORKERS': 30,
    'REQUESTS_PER_MINUTE': {
        'DEFAULT': 60,
        'MIN': 1,
        'MAX': 100
    },
    'CSV_SEPARATOR': ';',  # Semicolon separator for CSV files
    'PROMPT_TEMPLATE': ""  # Will be loaded from file
}

# Track missing metadata files
missing_metadata_files = []

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================
def get_script_directory() -> str:
    """Get the absolute path of the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path based on script location."""
    return os.path.join(get_script_directory(), relative_path)

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
        absolute_path = get_absolute_path(txt_path)
        with open(absolute_path, 'r', encoding='utf-8') as file:
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
        absolute_path = BASE_DIR / csv_path
        with open(absolute_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=CONFIG['CSV_SEPARATOR'])
            header = next(reader)
            data = list(reader)
        
        # Create DataFrame with renamed columns
        df = pd.DataFrame(data, columns=['selection_code', 'short_description'])
        if df.empty:
            raise ConfigurationError("DataFrame is empty")

        # Add JSON schema column with correct name
        df['selection_schema_json'] = df['selection_code'].apply(lambda code: load_json_schema(code))
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
        schema_path = BASE_DIR / "metadata" / "schemas" / f"{selection_code}_schema.json"
        if not schema_path.exists():
            missing_metadata_files.append(selection_code)
            return ""  # Return empty string for missing schemas
        with open(schema_path, 'r', encoding='utf-8') as f:
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
        return CONFIG['PROMPT_TEMPLATE'].format(**kwargs)
    except Exception as e:
        raise ValueError(f"Error formatting prompt template: {e}")

def get_progress_bar(iterable, total: int, desc: str):
    """Get a simple progress bar suitable for CLI.

    Args:
        iterable: The iterable to track progress on.
        total (int): The total number of items in the iterable.
        desc (str): Description of the progress.

    Returns:
        tqdm_module.tqdm: A tqdm progress bar instance.
    """
    return tqdm_module.tqdm(
        iterable, 
        total=total, 
        desc=desc,
        leave=True,  # Keep the progress bar
        ncols=100,   # Fixed width
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        file=sys.stdout
    )

def save_to_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV file with visual separators between rows.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The filename to save the DataFrame to.
    """
    try:
        absolute_path = BASE_DIR / filename
        
        # Create a prominent separator line (repeated 3 times)
        separator = '\n' + '=' * 100 + '\n' + '=' * 100 + '\n' + '=' * 100 + '\n'
        
        # Write with visual separators
        with open(absolute_path, 'w', encoding='utf-8', newline='') as f:
            # Write header
            f.write(CONFIG['CSV_SEPARATOR'].join(df.columns) + '\n')
            
            # Write each row with a prominent separator
            for _, row in df.iterrows():
                f.write(CONFIG['CSV_SEPARATOR'].join(str(val) for val in row) + '\n')
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
        db_path = BASE_DIR / db_name
        conn = sqlite3.connect(str(db_path))
        
        # Add timestamp column to DataFrame before saving
        df['processed_at'] = datetime.now().isoformat()
        
        # Create table with all columns including timestamp
        df.to_sql('selection_descriptions', conn, if_exists='replace', index=False)
        
        conn.close()
        logger.info(f"DataFrame successfully saved to SQLite database: {db_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to SQLite: {e}")
        raise

#===============================================================================
# CONFIGURATION
#===============================================================================
@dataclass
class Config:
    """Configuration class for processing."""
    max_workers: int
    requests_per_minute: int
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if not CONFIG['REQUESTS_PER_MINUTE']['MIN'] <= self.requests_per_minute <= CONFIG['REQUESTS_PER_MINUTE']['MAX']:
            raise ConfigurationError(f"Invalid requests_per_minute: {self.requests_per_minute}")
        if self.max_workers < 1:
            raise ConfigurationError(f"Invalid max_workers: {self.max_workers}")

# Create global config with new CONFIG dictionary
CONFIG_INSTANCE = Config(
    max_workers=CONFIG['MAX_WORKERS'],
    requests_per_minute=CONFIG['REQUESTS_PER_MINUTE']['DEFAULT']
)

#===============================================================================
# MONITORING AND METRICS
#===============================================================================
class Metrics:
    """Simple metrics collection."""
    def __init__(self):
        """Initialize metrics."""
        self.start_time = time.time()
        self.processed_rows = 0
        self.failed_rows = 0
        self.total_processing_time = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the metrics.
        """
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "processed_rows": self.processed_rows,
            "failed_rows": self.failed_rows,
            "total_processing_time": self.total_processing_time,
            "average_time_per_row": self.total_processing_time / max(1, self.processed_rows)
        }

def log_execution_time(func):
    """Decorator to log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e: 
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    return wrapper

#===============================================================================
# CORE PROCESSING FUNCTIONS
#===============================================================================
# @retry(
#     stop=stop_after_attempt(20),  # Try 20 times
#     wait=wait_exponential(multiplier=10, min=10, max=60),  # Wait between 10-60 seconds, increasing exponentially
#     reraise=True
# )
def get_azure_llm_response(**kwargs: Dict[str, Any]) -> str:
    """Get response from Azure OpenAI using the configured LLM.

    Args:
        **kwargs: Keyword arguments to pass to the prompt template.

    Returns:
        str: The content of the response from Azure OpenAI.

    Raises:
        Exception: If there is an error during the API call after all retries are exhausted.
    """
    request_id = f"req_{int(time.time()*1000)}"  # Unique request ID
    formatted_prompt = format_system_prompt(**kwargs)  # Format prompt before try block
    
    try:
        print(f"\nInitializing LLM connection...")
        # Use the get_azure_llm function with default temperature
        llm = get_azure_llm()
        
        # Format the message for LangChain
        messages = [{"role": "user", "content": formatted_prompt}]
        
        print(f"\nMaking API call...")
        # Get response using LangChain's invoke method
        response = llm.invoke(messages)
        print(f"\nReceived response from API")
        return response.content
    except Exception as e:
        logger.error(f"Error in LLM request {request_id}: {str(e)}")
        logger.error(f"Attempting retry with prompt: {formatted_prompt[:200]}...")  # Log first 200 chars of prompt
        raise  # Re-raise the exception for the retry decorator to handle

def check_selection_code_exists(selection_code: str, db_name: str = "selection_descriptions.db") -> bool:
    """Check if a selection code already exists in the database.

    Args:
        selection_code (str): The selection code to check.
        db_name (str): The name of the SQLite database file.

    Returns:
        bool: True if the selection code exists, False otherwise.
    """
    try:
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / db_name
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='selection_descriptions'
        """)
        if not cursor.fetchone():
            conn.close()
            return False
            
        # Check if selection code exists and has a non-empty extended_description
        cursor.execute("""
            SELECT 1 FROM selection_descriptions 
            WHERE selection_code = ? AND extended_description IS NOT NULL AND extended_description != ''
        """, (selection_code,))
        
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking selection code in database: {e}")
        return False

def save_single_record_to_csv(row: pd.Series, filename: str, is_first_record: bool = False):
    """Save a single record to CSV file.

    Args:
        row (pd.Series): The row to save.
        filename (str): The filename to save to.
        is_first_record (bool): Whether this is the first record being saved.
    """
    try:
        # Use BASE_DIR with metadata folder
        metadata_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / filename
        mode = 'w' if is_first_record else 'a'
        
        with open(metadata_path, mode, encoding='utf-8', newline='') as f:
            if is_first_record:
                # Write header
                f.write(CONFIG['CSV_SEPARATOR'].join(row.index) + '\n')  
            
            # Write row with separator
            f.write(CONFIG['CSV_SEPARATOR'].join(str(val) for val in row) + '\n')
            f.write('\n' + '=' * 100 + '\n' + '=' * 100 + '\n' + '=' * 100 + '\n')
            
    except Exception as e:
        logger.error(f"Error saving single record to CSV: {e}")
        raise

def save_single_record_to_sqlite(row: pd.Series, db_name: str = "selection_descriptions.db"):
    """Save a single record to SQLite database.

    Args:
        row (pd.Series): The row to save.
        db_name (str): The name of the SQLite database file.
    """
    try:
        # Use BASE_DIR with metadata folder
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / db_name
        conn = sqlite3.connect(str(db_path))
        
        # Add timestamp
        row_dict = row.to_dict()
        row_dict['processed_at'] = datetime.now().isoformat()
        
        # Create DataFrame from single row
        df = pd.DataFrame([row_dict])
        
        # Save to database
        df.to_sql('selection_descriptions', conn, if_exists='append', index=False)
        conn.close()
        
    except Exception as e:
        logger.error(f"Error saving single record to SQLite: {e}")
        raise

def process_dataframe_parallel(
    df: pd.DataFrame, 
    output_column: str = 'extended_description',
    max_workers: int = CONFIG_INSTANCE.max_workers, 
    requests_per_minute: int = CONFIG_INSTANCE.requests_per_minute
) -> pd.DataFrame:
    """Process a DataFrame in parallel using Azure OpenAI.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        output_column (str): The name of the column to store the results in.
        max_workers (int): The maximum number of workers to use for parallel processing.
        requests_per_minute (int): The maximum number of requests to make per minute.

    Returns:
        pd.DataFrame: The DataFrame with the results added to the specified output column.
    """
    metrics = Metrics()
    start_time = time.time()
    
    try:
        print("\nStarting initial check of records...")
        # First, check which records need processing
        records_to_process = []
        skipped_count = 0
        
        for index, row in df.iterrows():
            selection_code = row.get('selection_code')
            if not selection_code:
                print(f"Warning: Row {index + 1} has no selection_code, skipping")
                continue
                
            if check_selection_code_exists(selection_code):
                print(f"Skipping already processed selection code: {selection_code}")
                skipped_count += 1
                continue
                
            records_to_process.append((index, row))
        
        print(f"\nProcessing Summary:")
        print(f"- Total records: {len(df)}")
        print(f"- Already processed: {skipped_count}")
        print(f"- To process: {len(records_to_process)}")
        
        if not records_to_process:
            print("\nNo new records to process!")
            return df
            
        print("\nStarting processing of new records...")
        results = [None] * len(df)
        delay_between_requests = 60 / requests_per_minute
        processed_count = 0
        
        def process_row(index: int, row: pd.Series):
            """Process a single row of the DataFrame.

            Args:
                index (int): The index of the row.
                row (pd.Series): The row to process.

            Returns:
                Tuple[int, Optional[str]]: The index of the row and the result, or None if an error occurred.
            """
            try:
                selection_code = row.get('selection_code')
                print(f"\n{'=' * 100}")
                print(f"Starting processing for selection code: {selection_code}")
                print(f"{'=' * 100}\n")
                
                # Add delay between requests
                if index > 0:
                    print(f"\nWaiting {delay_between_requests:.2f} seconds before processing {selection_code}...")
                    time.sleep(delay_between_requests)
                
                row_dict = row.to_dict()
                if output_column in row_dict:
                    del row_dict[output_column]
                
                print(f"\nCalling LLM for {selection_code}...")
                result = get_azure_llm_response(**row_dict)
                print(f"\nSuccessfully got response for {selection_code}")
                return index, result
                
            except Exception as e:
                error_msg = f"Error processing selection code {selection_code}: {str(e)}"
                print(f"\n{error_msg}")
                logger.error(error_msg)
                logger.error(f"Row data: {row_dict}")
                return index, None
        
        print(f"\nInitializing ThreadPoolExecutor with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print("\nSubmitting tasks to executor...")
            futures = {
                executor.submit(process_row, index, row): index
                for index, row in records_to_process
            }
            
            print("\nStarting progress tracking...")
            with get_progress_bar(total=len(records_to_process), desc="Processing", iterable=as_completed(futures)) as pbar:
                for future in pbar:
                    try:
                        print("\nWaiting for next result...")
                        index, result = future.result(timeout=300)  # 5-minute timeout per record
                        results[index] = result
                        
                        if result is not None:
                            metrics.processed_rows += 1
                            processed_count += 1
                            
                            # Update the DataFrame with the result
                            df.at[index, output_column] = result
                            
                            # Save the processed row immediately
                            row_to_save = df.iloc[index]
                            print(f"\nSaving record {processed_count} to CSV and database...")
                            save_single_record_to_csv(row_to_save, "output.csv", is_first_record=(processed_count == 1))
                            save_single_record_to_sqlite(row_to_save)
                            
                            print(f"\nSuccessfully saved record {processed_count}")
                        else:
                            metrics.failed_rows += 1
                            print(f"\nFailed to process selection code: {df.iloc[index]['selection_code']}")
                            
                    except TimeoutError:
                        print(f"\nTimeout while processing selection code: {df.iloc[index]['selection_code']}")
                        metrics.failed_rows += 1
                    except Exception as e:
                        print(f"\nUnexpected error in future: {str(e)}")
                        metrics.failed_rows += 1
        
        end_time = time.time()
        metrics.total_processing_time = end_time - start_time
        
        print(f"\nProcessing completed in {metrics.total_processing_time:.2f} seconds:")
        print(f"- Total records: {len(df)}")
        print(f"- Already processed: {skipped_count}")
        print(f"- Newly processed: {processed_count}")
        print(f"- Failed: {metrics.failed_rows}")
        print(f"- Average time per new record: {metrics.total_processing_time/max(1,processed_count):.2f} seconds")
        
        if metrics.failed_rows > 0:
            print("\nWARNING: Some records failed to process. Check the logs above for details.")
        
        return df
        
    except Exception as e:
        print(f"\nError in process_dataframe_parallel: {e}")
        raise

#===============================================================================
# EXECUTION BLOCK
#===============================================================================
if __name__ == "__main__":
    try:
        # Load prompt template from CSV
        CONFIG['PROMPT_TEMPLATE'] = load_prompt_template_from_txt()
        print(f"Loaded prompt template: {CONFIG['PROMPT_TEMPLATE'][:1000]}...")
        
        # Load DataFrame from CSV and join with JSON schemas
        input_csv_path = "metadata/selection_descriptions.csv"
        test_df = load_dataframe_from_csv_and_jsons(input_csv_path)
        
        print("Starting parallel DataFrame processing...")
        processed_df = process_dataframe_parallel(
            test_df,
            output_column="extended_description",
            max_workers=3,
            requests_per_minute=30
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