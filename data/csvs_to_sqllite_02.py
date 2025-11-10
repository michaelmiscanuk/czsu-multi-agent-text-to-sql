"""CSV to SQLite Database Converter

This module provides functionality to convert multiple CSV files from a directory
into a single SQLite database, with each CSV file becoming a separate table.
"""
MODULE_DESCRIPTION = r"""CSV to SQLite Database Converter

This module provides functionality to convert multiple CSV files from a directory
into a single SQLite database, with each CSV file becoming a separate table.

Key Features:
-------------
1. Batch CSV Processing:
   - Automatic discovery of CSV files in a directory
   - Support for UTF-8 encoding to handle Czech characters
   - Flexible folder path specification
   - Individual file error handling without stopping the entire process

2. SQLite Database Management:
   - Creates a single SQLite database file (czsu_data.db)
   - Table names derived from CSV filenames (without extension)
   - Table replacement strategy (if_exists='replace')
   - Automatic index exclusion for cleaner tables
   - Database stored in script directory for consistency

3. Error Handling:
   - Individual file error reporting
   - Process continuation despite individual file failures
   - Detailed error messages for debugging
   - Success/failure statistics tracking

4. Progress Reporting:
   - Real-time import status for each file
   - Final summary with database location
   - Import count statistics
   - Clear success/error messaging

Processing Flow:
--------------
1. Path Resolution:
   - Determines script directory location
   - Uses provided folder path or defaults to script directory
   - Sets up SQLite database path in script directory

2. Database Initialization:
   - Creates SQLite connection to czsu_data.db
   - Prepares for batch processing
   - Initializes import counter

3. File Discovery and Processing:
   - Scans directory for .csv files (case-insensitive)
   - Processes each CSV file individually
   - Handles encoding issues automatically

4. Data Import:
   - Reads CSV with pandas using UTF-8 encoding
   - Extracts table name from filename
   - Writes DataFrame to SQLite table
   - Replaces existing tables if present

5. Error Management:
   - Catches and reports individual file errors
   - Continues processing remaining files
   - Maintains error statistics

6. Completion Reporting:
   - Provides final import statistics
   - Reports database location
   - Summarizes successful imports

Usage Example:
-------------
# Import from script's directory
import_all_csv_to_sqlite()

# Import from specific directory
import_all_csv_to_sqlite("data/CSVs")

# Import from absolute path
import_all_csv_to_sqlite("/path/to/csv/files")

Required Environment:
-------------------
- Python 3.7+
- pandas library for CSV reading
- sqlite3 (built-in Python module)
- os (built-in Python module)
- Write permissions in script directory

Output:
-------
- SQLite database file: czsu_data.db
- Tables named after CSV filenames
- UTF-8 encoded data preservation
- Index-free tables for cleaner structure

Error Handling:
-------------
- Individual CSV file processing errors
- Database connection errors
- File system permission errors
- Encoding issues with non-UTF-8 files
- Invalid CSV format handling"""

import os
import sqlite3
import pandas as pd


# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================
def import_all_csv_to_sqlite(folder_path: str = None) -> None:
    """
    Import all CSV files from a folder into SQLite database.

    This function scans a directory for CSV files and imports each one as a separate
    table in a SQLite database. The database is created in the same directory as
    the script for consistency. Each CSV file becomes a table with the same name
    as the file (without the .csv extension).

    The function handles UTF-8 encoding specifically to support Czech characters
    and other international text. If a CSV file fails to import, the error is
    reported but processing continues with the remaining files.

    Args:
        folder_path (str, optional): Path to folder containing CSV files.
                                   If None, uses the script's directory.
                                   Can be relative or absolute path.

    Returns:
        None: Function performs file operations and prints status messages.

    Raises:
         Exception: For any errors during processing (database, file system, or CSV parsing issues).

    Processing Details:
        - Scans for files with .csv extension (case-insensitive)
        - Uses UTF-8 encoding for proper character support
        - Creates table names by removing file extension
        - Replaces existing tables if they already exist
        - Excludes DataFrame index from SQLite tables
        - Maintains connection throughout batch processing

    Output Files:
        - czsu_data.db: SQLite database in script directory
        - Tables: One per CSV file, named after the filename
    """
    try:
        # --- Path Resolution and Setup ---
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Use provided folder path or script directory
        folder_path = folder_path or script_dir

        # --- Database Connection Setup ---
        # Create SQLite database in the script directory
        db_path = os.path.join(script_dir, "czsu_data.db")
        conn = sqlite3.connect(db_path)

        # --- Processing Initialization ---
        # Counter for imported files
        imported_count = 0

        # --- CSV File Discovery and Processing ---
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".csv"):
                try:
                    # --- Individual File Processing ---
                    # Create full path to CSV file
                    csv_file_path = os.path.join(folder_path, filename)

                    # Read CSV file with proper encoding for Czech characters
                    df = pd.read_csv(csv_file_path, encoding="utf-8")

                    # Get the file name without extension to use as table name
                    table_name = os.path.splitext(filename)[0]

                    # --- Database Table Creation ---
                    # Write the dataframe to SQLite
                    df.to_sql(
                        name=table_name,
                        con=conn,
                        if_exists="replace",  # Replace if table exists
                        index=False,
                    )

                    # --- Success Reporting ---
                    print(f"Successfully imported {filename} to table {table_name}")
                    imported_count += 1

                except Exception as e:
                    # --- Individual File Error Handling ---
                    print(f"Error importing {filename}: {str(e)}")

        # --- Database Cleanup ---
        # Close the connection
        conn.close()

        # --- Final Status Reporting ---
      print("\nImport completed:")
        print(f"Database location: {db_path}")
        print(f"Total files imported: {imported_count}")

    except Exception as e:
        # --- Overall Process Error Handling ---
        print(f"Error occurred: {str(e)}")


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
# Example usage
if __name__ == "__main__":
    # --- Configuration Options ---
    # Option 1: Import from script's directory
    # import_all_csv_to_sqlite()

    # Option 2: Import from specific folder
    import_all_csv_to_sqlite("data/CSVs")
