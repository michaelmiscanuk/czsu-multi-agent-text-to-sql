"""SQLite to CSV Export Utility

Exports all tables from a SQLite database into separate CSV files.
"""

import os
import sqlite3
import pandas as pd


def export_all_sqlite_to_csvs(db_path: str = None, output_folder: str = None) -> None:
    """
    Export all tables from a SQLite database into separate CSV files (semicolon-separated).
    Args:
        db_path (str, optional): Path to the SQLite database. Defaults to 'czsu_data.db' in script's directory.
        output_folder (str, optional): Folder to save CSVs. Defaults to 'CSVs' in script's directory.
    """
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Use provided db_path or default
        db_path = db_path or os.path.join(script_dir, "czsu_data.db")
        output_folder = output_folder or os.path.join(script_dir, "CSVs")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        exported_count = 0
        for table in tables:
            try:
                # Read table into DataFrame
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
                # Write to CSV with semicolon separator
                csv_path = os.path.join(output_folder, f"{table}.csv")
                df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
                print(f"Exported table {table} to {csv_path}")
                exported_count += 1
            except Exception as exc:
                print(f"Error exporting table {table}: {str(exc)}")

        conn.close()
        print("\nExport completed:")
        print(f"Database: {db_path}")
        print(f"Output folder: {output_folder}")
        print(f"Total tables exported: {exported_count}")
    except Exception as exc:
        print(f"Error occurred: {str(exc)}")


# Example usage
if __name__ == "__main__":
    # Option 1: Export from default db to default folder
    export_all_sqlite_to_csvs()

    # Option 2: Export from specific db to specific folder
    # export_all_sqlite_to_csvs("czsu_data.db", "data/CSVs")
