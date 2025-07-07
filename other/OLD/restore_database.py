import sqlite3
import os
import gzip

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
    sql_file = "data/czsu_data_schema.sql"
    
    restore_database(db_path, sql_file) 