import sqlite3
import os
import gzip

def dump_database(db_path, output_file):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Open a gzip file for writing
    with gzip.open(f"{output_file}.gz", 'wt', encoding='utf-8') as f:
        # Iterate over the dump and write to the gzip file
        for line in conn.iterdump():
            f.write(line + '\n')
    
    conn.close()
    print(f"Database dumped and compressed to {output_file}.gz")
    
    # Get the size of the compressed file
    compressed_size = os.path.getsize(f"{output_file}.gz") / (1024 * 1024)  # Size in MB
    print(f"Compressed file size: {compressed_size:.2f} MB")
    
    return True

def restore_database(db_path, sql_file):
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create new database
    conn = sqlite3.connect(db_path)
    
    try:
        # Read the compressed file
        with gzip.open(f"{sql_file}.gz", 'rt', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Execute the SQL commands
        conn.executescript(sql_content)
        conn.close()
        
        print(f"Database restored to {db_path}")
        return True
    except Exception as e:
        print(f"Error restoring database: {e}")
        return False

if __name__ == "__main__":
    db_path = "data/czsu_data.db"
    output_file = "data/czsu_data_schema.sql"
    
    # Dump the database
    if dump_database(db_path, output_file):
        print("\nTo restore the database, use:")
        print(f"python data/restore_database.py") 