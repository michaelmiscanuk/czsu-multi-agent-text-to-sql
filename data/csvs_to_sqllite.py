import pandas as pd
import sqlite3
import os

def import_all_csv_to_sqlite(folder_path: str = None) -> None:
    """
    Import all CSV files from a folder into SQLite database.
    If folder_path is None, use the same folder as the script.
    
    Args:
        folder_path (str, optional): Path to folder containing CSV files. 
                                   Defaults to script's directory.
    """
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use provided folder path or script directory
        folder_path = folder_path or script_dir
        
        # Create SQLite database in the script directory
        db_path = os.path.join(script_dir, 'czsu_data.db')
        conn = sqlite3.connect(db_path)
        
        # Counter for imported files
        imported_count = 0
        
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.csv'):
                try:
                    # Create full path to CSV file
                    csv_file_path = os.path.join(folder_path, filename)
                    
                    # Read CSV file with proper encoding for Czech characters
                    df = pd.read_csv(csv_file_path, encoding='utf-8')
                    
                    # Get the file name without extension to use as table name
                    table_name = os.path.splitext(filename)[0]
                    
                    # Write the dataframe to SQLite
                    df.to_sql(
                        name=table_name,
                        con=conn,
                        if_exists='replace',  # Replace if table exists
                        index=False
                    )
                    
                    print(f"Successfully imported {filename} to table {table_name}")
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error importing {filename}: {str(e)}")
        
        # Close the connection
        conn.close()
        
        print(f"\nImport completed:")
        print(f"Database location: {db_path}")
        print(f"Total files imported: {imported_count}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Option 1: Import from script's directory
    # import_all_csv_to_sqlite()
    
    # Option 2: Import from specific folder
    import_all_csv_to_sqlite("data/CSVs")