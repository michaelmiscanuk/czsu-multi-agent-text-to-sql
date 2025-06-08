#!/usr/bin/env python3
"""
Clear all checkpoint tables from PostgreSQL.
"""

import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

def clear_checkpoint_tables():
    # Get connection parameters
    user = os.getenv('user')
    password = os.getenv('password') 
    host = os.getenv('host')
    port = os.getenv('port', '5432')
    dbname = os.getenv('dbname')
    
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require'
    
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            # Drop all checkpoint tables
            tables = ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"]
            
            for table in tables:
                try:
                    cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
                    print(f'✓ Dropped table: {table}')
                except Exception as e:
                    print(f'⚠ Could not drop {table}: {e}')
            
            conn.commit()
            print('✅ All checkpoint tables cleared')

if __name__ == "__main__":
    clear_checkpoint_tables() 