# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

# Import authentication dependencies
from api.dependencies.auth import get_current_user
from api.helpers import traceback_json_response

# Import debug functions
from api.utils.debug import print__debug

# Create router for catalog endpoints
router = APIRouter()


@router.get("/catalog")
def get_catalog(
    page: int = Query(1, ge=1),
    q: Optional[str] = None,
    page_size: int = Query(10, ge=1, le=10000),
    user=Depends(get_current_user),
):
    try:
        db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
        offset = (page - 1) * page_size
        where_clause = ""
        params = []
        if q:
            where_clause = "WHERE selection_code LIKE ? OR extended_description LIKE ?"
            like_q = f"%{q}%"
            params.extend([like_q, like_q])
        query = f"""
            SELECT selection_code, extended_description
            FROM selection_descriptions
            {where_clause}
            ORDER BY selection_code
            LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])
        count_query = f"SELECT COUNT(*) FROM selection_descriptions {where_clause}"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(count_query, params[:-2] if q else [])
            total = cursor.fetchone()[0]
            cursor.execute(query, params)
            rows = cursor.fetchall()
        results = [
            {"selection_code": row[0], "extended_description": row[1]} for row in rows
        ]
        return {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise


@router.get("/data-tables")
def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
    try:
        db_path = "data/czsu_data.db"
        desc_db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
        if q:
            q_lower = q.lower()
            tables = [t for t in tables if q_lower in t.lower()]
        # Fetch short_descriptions from the other DB
        desc_map = {}
        try:
            with sqlite3.connect(desc_db_path) as desc_conn:
                desc_cursor = desc_conn.cursor()
                desc_cursor.execute(
                    "SELECT selection_code, short_description FROM selection_descriptions"
                )
                for code, short_desc in desc_cursor.fetchall():
                    desc_map[code] = short_desc
        except Exception as e:
            print__debug(f"Error fetching short_descriptions: {e}")
        # Build result list
        result = [
            {"selection_code": t, "short_description": desc_map.get(t, "")}
            for t in tables
        ]
        return {"tables": result}
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise


@router.get("/data-table")
def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
    try:
        db_path = "data/czsu_data.db"
        if not table:
            print__debug("No table specified")
            return {"columns": [], "rows": []}
        print__debug(f"Requested table: {table}")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                print__debug(f"Columns: {columns}, Rows count: {len(rows)}")
            except Exception as e:
                print__debug(f"Error fetching table '{table}': {e}")
                return {"columns": [], "rows": []}
        return {"columns": columns, "rows": rows}
    except Exception as e:
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise
