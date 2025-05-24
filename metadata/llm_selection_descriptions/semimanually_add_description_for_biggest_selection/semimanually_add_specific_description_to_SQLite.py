import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Set up base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

def add_values_to_sqlite(schema_path: str, column_name: str, add_after_this_text: str) -> None:
    """
    Add values from schema to SQLite database.
    
    Args:
        schema_path (str): Path to the schema JSON file relative to BASE_DIR
        column_name (str): Name of the column in schema to extract values from (will be used as selection_code)
        add_after_this_text (str): Text to include before the values in extended_description
    """
    # Read the schema file
    schema_path = BASE_DIR / schema_path
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    # Extract values from category['label']
    labels = list(schema["dimension"][column_name]["category"]["label"].values())

    # Read the template file
    template_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / "semimanually_add_description_for_biggest_selection" / "template.txt"
    with open(template_path, 'r', encoding='utf-8') as f:
        template_lines = f.readlines()

    # Find the insertion point
    insertion_line = add_after_this_text + '\n'
    insertion_index = template_lines.index(insertion_line) + 1

    # Create extended description by combining template content and values
    extended_description = ''.join(template_lines[:insertion_index])  # Include template content up to insertion point
    
    # Add the values from JSON
    for label in labels:
        extended_description += f'- "{label}"\n'
    
    # Add the rest of the template content
    extended_description += ''.join(template_lines[insertion_index:])

    # Save to SQLite
    try:
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / "selection_descriptions.db"
        print(f"\nAttempting to save to database: {db_path}")
        
        with sqlite3.connect(str(db_path)) as conn:
            # Enable foreign keys and set isolation level
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None  # Enable autocommit mode
            
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='selection_descriptions'
            """)
            table_exists = cursor.fetchone() is not None
            print(f"Table exists: {table_exists}")

            if not table_exists:
                # Create new table with proper constraints
                conn.execute("""
                    CREATE TABLE selection_descriptions (
                        selection_code TEXT PRIMARY KEY,
                        short_description TEXT,
                        selection_schema_json TEXT,
                        extended_description TEXT,
                        processed_at TEXT
                    )
                """)
                print("Created new selection_descriptions table")
            
            # Prepare the data for insertion
            row_dict = {
                'selection_code': column_name,
                'short_description': '',  # Empty as we don't have this information
                'selection_schema_json': json.dumps(schema),  # Store the full schema
                'extended_description': extended_description,
                'processed_at': datetime.now().isoformat()
            }
            
            # Start transaction
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Prepare the data with proper parameter binding
                placeholders = ', '.join(['?' for _ in row_dict])
                columns = ', '.join(row_dict.keys())
                
                # Use parameterized query for safety
                cursor.execute(f"""
                    INSERT OR REPLACE INTO selection_descriptions ({columns})
                    VALUES ({placeholders})
                """, list(row_dict.values()))
                
                # Verify the operation
                cursor.execute("SELECT selection_code FROM selection_descriptions WHERE selection_code = ?", (column_name,))
                if cursor.fetchone():
                    print(f"Successfully saved record for {column_name}")
                else:
                    print(f"Failed to save record for {column_name} - record not found after insert")
                
                # Commit transaction
                conn.execute("COMMIT")
                print("Transaction committed")
                
            except sqlite3.IntegrityError as e:
                # Rollback on integrity error
                conn.execute("ROLLBACK")
                print(f"Integrity error: {str(e)}")
                raise
            except Exception as e:
                # Rollback on any other error
                conn.execute("ROLLBACK")
                print(f"Error: {str(e)}")
                raise
            
    except Exception as e:
        print(f"Error saving to SQLite: {e}")
        raise

if __name__ == "__main__":
    schema_path = "metadata/schemas/PRUM201T1_schema.json"
    column_name = "PRODCOM2"
    add_after_text = 'For "Seznam výrobků", the available values are:'
    
    add_values_to_sqlite(schema_path, column_name, add_after_text) 
    