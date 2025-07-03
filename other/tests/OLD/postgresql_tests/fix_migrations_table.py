#!/usr/bin/env python3
"""
Fix the checkpoint_migrations table schema to use the correct column name.
"""

import asyncio
import platform
from psycopg_pool import AsyncConnectionPool
import os
from dotenv import load_dotenv

# Fix for Windows ProactorEventLoop issue with psycopg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

async def fix_migrations_table():
    # Get connection parameters
    user = os.getenv('user')
    password = os.getenv('password') 
    host = os.getenv('host')
    port = os.getenv('port', '5432')
    dbname = os.getenv('dbname')
    
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'
    
    connection_kwargs = {
        'sslmode': 'require',
        'autocommit': False,
        'prepare_threshold': 0,
    }
    
    pool = AsyncConnectionPool(
        conninfo=connection_string,
        max_size=1,
        min_size=1,
        kwargs=connection_kwargs,
        open=False
    )
    
    await pool.open()
    
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        # Drop and recreate the migrations table with correct schema
        await conn.execute('DROP TABLE IF EXISTS checkpoint_migrations')
        await conn.execute('''
            CREATE TABLE checkpoint_migrations (
                v INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        print('✓ Fixed checkpoint_migrations table schema')
    
    await pool.close()

if __name__ == "__main__":
    asyncio.run(fix_migrations_table()) 