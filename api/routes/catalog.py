# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
from api.utils.debug import print__catalog_debug, print__data_tables_debug
from api.helpers import traceback_json_response
from api.dependencies.auth import get_current_user
from fastapi import APIRouter, Depends, Query
from typing import Optional
import sqlite3
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Standard imports

# Create router for catalog endpoints
router = APIRouter()
print__catalog_debug("Catalog router initialized successfully")


@router.get("/catalog")
def get_catalog(
    page: int = Query(1, ge=1),
    q: Optional[str] = None,
    page_size: int = Query(10, ge=1, le=10000),
    user=Depends(get_current_user),
):
    print__catalog_debug(
        f"GET /catalog called - page: {page}, page_size: {page_size}, query: '{q}', user: {user.username if hasattr(user, 'username') else 'unknown'}")
    try:
        db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
        print__catalog_debug(f"Connecting to catalog database: {db_path}")
        offset = (page - 1) * page_size
        where_clause = ""
        params = []
        if q:
            print__catalog_debug(f"Building search query for term: '{q}'")
            where_clause = "WHERE selection_code LIKE ? OR extended_description LIKE ?"
            like_q = f"%{q}%"
            params.extend([like_q, like_q])
        else:
            print__catalog_debug(
                "No search query provided, fetching all records")
        query = f"""
            SELECT selection_code, extended_description
            FROM selection_descriptions
            {where_clause}
            ORDER BY selection_code
            LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])
        print__catalog_debug(f"SQL query: {query}")
        print__catalog_debug(f"Query parameters: {params}")
        count_query = f"SELECT COUNT(*) FROM selection_descriptions {where_clause}"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            print__catalog_debug(f"Executing count query: {count_query}")
            cursor.execute(count_query, params[:-2] if q else [])
            total = cursor.fetchone()[0]
            print__catalog_debug(f"Total records found: {total}")
            cursor.execute(query, params)
            rows = cursor.fetchall()
            print__catalog_debug(
                f"Fetched {len(rows)} records for current page")
        results = [
            {"selection_code": row[0], "extended_description": row[1]} for row in rows
        ]
        response_data = {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
        print__catalog_debug(
            f"Returning catalog response with {len(results)} results")
        return response_data
    except Exception as e:
        print__catalog_debug(f"Error in get_catalog: {str(e)}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise


@router.get("/data-tables")
def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
    print__data_tables_debug(
        f"GET /data-tables called - query: '{q}', user: {user.username if hasattr(user, 'username') else 'unknown'}")
    try:
        db_path = "data/czsu_data.db"
        desc_db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
        print__data_tables_debug(f"Connecting to data database: {db_path}")
        print__data_tables_debug(
            f"Connecting to descriptions database: {desc_db_path}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            print__data_tables_debug("Fetching table names from sqlite_master")
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            print__data_tables_debug(f"Found {len(tables)} tables in database")

        if q:
            print__data_tables_debug(f"Filtering tables with query: '{q}'")
            q_lower = q.lower()
            original_count = len(tables)
            tables = [t for t in tables if q_lower in t.lower()]
            print__data_tables_debug(
                f"Filtered from {original_count} to {len(tables)} tables")

        # Fetch short_descriptions from the other DB
        desc_map = {}
        print__data_tables_debug(
            "Fetching short descriptions from descriptions database")
        try:
            with sqlite3.connect(desc_db_path) as desc_conn:
                desc_cursor = desc_conn.cursor()
                desc_cursor.execute(
                    "SELECT selection_code, short_description FROM selection_descriptions"
                )
                descriptions = desc_cursor.fetchall()
                print__data_tables_debug(
                    f"Fetched {len(descriptions)} descriptions")
                for code, short_desc in descriptions:
                    desc_map[code] = short_desc
        except Exception as e:
            print__data_tables_debug(f"Error fetching short_descriptions: {e}")

        # Build result list
        result = [
            {"selection_code": t, "short_description": desc_map.get(t, "")}
            for t in tables
        ]
        print__data_tables_debug(f"Built result list with {len(result)} items")
        return {"tables": result}
    except Exception as e:
        print__data_tables_debug(f"Error in get_data_tables: {str(e)}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise


@router.get("/data-table")
def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
    print__data_tables_debug(
        f"GET /data-table called - table: '{table}', user: {user.username if hasattr(user, 'username') else 'unknown'}")
    try:
        db_path = "data/czsu_data.db"
        print__data_tables_debug(f"Connecting to data database: {db_path}")

        if not table:
            print__data_tables_debug("No table specified")
            return {"columns": [], "rows": []}

        print__data_tables_debug(f"Requested table: {table}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                print__data_tables_debug(
                    f"Executing SELECT query on table '{table}' with LIMIT 10000")
                cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                print__data_tables_debug(
                    f"Successfully fetched data - Columns: {columns}")
                print__data_tables_debug(f"Row count: {len(rows)}")
                if len(rows) > 0:
                    print__data_tables_debug(f"Sample first row: {rows[0]}")

                response_data = {"columns": columns, "rows": rows}
                print__data_tables_debug(
                    f"Returning table data with {len(columns)} columns and {len(rows)} rows")
                return response_data

            except Exception as e:
                print__data_tables_debug(
                    f"Error fetching table '{table}': {e}")
                return {"columns": [], "rows": []}
    except Exception as e:
        print__data_tables_debug(f"Error in get_data_table: {str(e)}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise
