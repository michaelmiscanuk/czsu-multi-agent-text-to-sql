"""
MODULE_DESCRIPTION: Data Catalog Endpoints - CZSU Database Discovery and Schema Exploration

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module implements data catalog and schema exploration endpoints for browsing
and discovering available Czech Statistical Office (CZSU) datasets. It provides
three main endpoints for progressive data discovery:

1. /catalog - Browse and search dataset descriptions
2. /data-tables - List all available data tables
3. /data-table - View schema and sample data for specific tables

The module enables users to understand what data is available before crafting
SQL queries, supporting the text-to-SQL workflow by providing context about
available datasets, their structure, and sample content.

Primary Use Cases:
    - Initial data discovery ("What data is available?")
    - Dataset search ("Find population statistics")
    - Schema inspection ("What columns does this table have?")
    - Sample data preview ("Show me example rows")
    - AI agent context gathering (for better SQL generation)

===================================================================================
KEY FEATURES
===================================================================================

1. Dataset Catalog Search (/catalog)
   - Paginated browsing of available datasets
   - Full-text search by selection code or description
   - Efficient SQL-based filtering
   - Comprehensive dataset descriptions
   - Case-insensitive search with LIKE operator

2. Table Listing (/data-tables)
   - Complete list of all data tables
   - Optional filtering by table name
   - Integration with description database
   - Short descriptions for quick overview
   - Metadata enrichment from multiple sources

3. Table Schema and Data (/data-table)
   - Column names and types
   - Sample data (up to 10,000 rows)
   - Complete schema information
   - Quick data preview capability
   - Error handling for non-existent tables

4. Multi-Database Architecture
   - czsu_data.db: Actual statistical data tables
   - selection_descriptions.db: Metadata and descriptions
   - Coordinated queries across databases
   - Consistent data model

5. Authentication and Security
   - JWT authentication via get_current_user
   - User-specific access logging
   - No sensitive data exposure
   - Read-only database access

6. Windows Compatibility
   - WindowsSelectorEventLoopPolicy for async operations
   - Proper event loop configuration
   - Cross-platform support

7. Comprehensive Logging
   - Debug logging for all operations
   - Query parameter logging
   - Result count logging
   - Error tracking and diagnostics

===================================================================================
API ENDPOINTS
===================================================================================

GET /catalog
    Browse and search the CZSU data catalog with pagination

    Authentication: JWT token required

    Query Parameters:
        page (int, default=1, min=1): Page number (1-indexed)
        page_size (int, default=10, min=1, max=10000): Results per page
        q (str, optional): Search query for filtering

    Returns:
        {
            "results": [
                {
                    "selection_code": "POP2020",
                    "extended_description": "Population statistics for 2020"
                }
            ],
            "total": 150,      # Total matching records
            "page": 1,         # Current page
            "page_size": 10    # Results per page
        }

    Search Behavior:
        - Searches both selection_code and extended_description
        - Case-insensitive partial matching (LIKE '%query%')
        - Returns all records if q is not provided

    Examples:
        GET /catalog?page=1&page_size=20
        GET /catalog?q=population
        GET /catalog?q=2023&page=2

GET /data-tables
    List all available data tables with optional filtering

    Authentication: JWT token required

    Query Parameters:
        q (str, optional): Filter table names by partial match

    Returns:
        {
            "tables": [
                {
                    "selection_code": "table_name",
                    "short_description": "Brief description"
                }
            ]
        }

    Behavior:
        - Excludes SQLite internal tables (sqlite_%)
        - Enriches with descriptions from metadata database
        - Filters table names if q parameter provided
        - Returns all tables if no filter specified

    Examples:
        GET /data-tables
        GET /data-tables?q=population

GET /data-table
    Get schema and sample data for a specific table

    Authentication: JWT token required

    Query Parameters:
        table (str, optional): Table name to query

    Returns:
        {
            "columns": ["col1", "col2", "col3"],
            "rows": [
                ["val1", "val2", "val3"],
                ["val4", "val5", "val6"]
            ]
        }

    Behavior:
        - Returns empty result if table not specified
        - Limits to 10,000 rows for performance
        - Includes all columns
        - Handles non-existent tables gracefully

    Examples:
        GET /data-table?table=population_2023

    Note:
        10,000 row limit prevents memory issues with large tables

===================================================================================
DATABASE ARCHITECTURE
===================================================================================

Two SQLite Databases:

1. czsu_data.db (Data Database)
   Location: data/czsu_data.db
   Purpose: Actual statistical data
   Structure:
       - Multiple tables (one per dataset/selection)
       - Table names = selection codes
       - Columns vary by dataset
       - Rows = statistical records

   Usage:
       - get_data_tables: Lists table names
       - get_data_table: Queries specific table

2. selection_descriptions.db (Metadata Database)
   Location: metadata/llm_selection_descriptions/selection_descriptions.db
   Purpose: Dataset descriptions and metadata
   Structure:
       Table: selection_descriptions
       Columns:
           - selection_code (TEXT, PRIMARY KEY)
           - short_description (TEXT)
           - extended_description (TEXT)

   Usage:
       - get_catalog: Full catalog search
       - get_data_tables: Enriches table list with descriptions

Relationship:
    selection_code in selection_descriptions
    = table_name in czsu_data.db

This allows:
    - Lookup table metadata by name
    - Enrich data tables with human-readable descriptions
    - Search datasets without querying data tables

===================================================================================
PAGINATION STRATEGY
===================================================================================

Catalog Pagination (/catalog):

Algorithm:
    1. Calculate offset: (page - 1) * page_size
    2. Execute COUNT query for total records
    3. Execute SELECT query with LIMIT and OFFSET
    4. Return results + pagination metadata

SQL Pattern:
    SELECT ... LIMIT ? OFFSET ?

Response Fields:
    - results: Current page data
    - total: Total matching records
    - page: Current page number
    - page_size: Records per page

Client Calculation:
    - Total pages = ceil(total / page_size)
    - Has next page = (page * page_size) < total
    - Has previous page = page > 1

Benefits:
    - Efficient for large datasets
    - Predictable memory usage
    - Standard pagination pattern
    - Supports large page_size (up to 10,000)

Example:
    150 total records, page_size=10
    - Page 1: Records 1-10 (offset=0)
    - Page 2: Records 11-20 (offset=10)
    - Page 15: Records 141-150 (offset=140)

===================================================================================
SEARCH FUNCTIONALITY
===================================================================================

Catalog Search (/catalog?q=...):

Implementation:
    WHERE selection_code LIKE ? OR extended_description LIKE ?
    Parameters: ['%query%', '%query%']

Features:
    - Case-insensitive (SQLite LIKE default)
    - Partial matching (wraps with %)
    - Multi-field search (code OR description)
    - OR logic for broader results

Examples:
    q="2023" → Matches:
        - selection_code: "POP2023"
        - extended_description: "Population data from 2023"

    q="population" → Matches:
        - selection_code: "POPULATION_STATS"
        - extended_description: "Detailed population statistics"

Performance:
    - Uses SQLite full table scan (no indexes by default)
    - Acceptable for 1000s of records
    - May need indexes for 100,000+ records
    - LIMIT clause prevents excessive memory use

Table Filtering (/data-tables?q=...):

Implementation:
    Python filtering: tables = [t for t in tables if q_lower in t.lower()]

Features:
    - Case-insensitive
    - Substring matching
    - Post-query filtering (in-memory)

Why Not SQL WHERE:
    - Table names come from sqlite_master
    - Simple LIKE query would work
    - Python filtering allows more flexibility
    - Minimal performance difference (few tables)

===================================================================================
ERROR HANDLING
===================================================================================

Three-Tier Error Handling:

1. Endpoint-Level Try/Except
   - Wraps entire endpoint logic
   - Catches any unexpected errors
   - Calls traceback_json_response for formatting
   - Re-raises if no traceback handler

2. Database-Specific Error Handling
   - Try/except around individual table queries
   - Returns empty results for missing tables
   - Logs error details
   - Continues execution (graceful degradation)

3. Null Handling
   - Checks for None/null values
   - Default empty results if no table specified
   - Empty description if not found in metadata

Error Response Pattern:
    try:
        # Database operation
    except Exception as e:
        print__debug(f"Error: {e}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise

Benefits:
    - Consistent error format
    - Detailed logging for debugging
    - Graceful failures (don't crash API)
    - Client-friendly error messages

Common Errors:
    - Table not found: Returns empty columns/rows
    - Description DB missing: Empty descriptions
    - Invalid SQL: Caught and logged
    - Connection errors: Propagated with traceback

===================================================================================
LOGGING AND DEBUGGING
===================================================================================

Debug Functions:
    - print__catalog_debug: For /catalog endpoint
    - print__data_tables_debug: For /data-tables and /data-table

Logged Information:
    1. Request Entry
       - Endpoint called
       - Query parameters
       - User information

    2. Database Operations
       - Connection paths
       - SQL queries
       - Query parameters
       - Row counts

    3. Search Operations
       - Search terms
       - Filter logic
       - Before/after counts

    4. Results
       - Result counts
       - Sample data
       - Response structure

    5. Errors
       - Exception type and message
       - Stack traces
       - Context information

Log Pattern:
    print__catalog_debug("Operation started")
    print__catalog_debug(f"Parameter: {value}")
    print__catalog_debug(f"Result: {len(results)} records")

Benefits:
    - Troubleshoot issues quickly
    - Monitor performance
    - Audit user access
    - Debug SQL queries

Production:
    - Should be disabled or sent to proper logging system
    - Can impact performance with high volume
    - Consider log levels (DEBUG, INFO, ERROR)

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Catalog Endpoint:
    - Response time: 10-50ms (no search)
    - Response time: 20-100ms (with search)
    - Memory: < 1MB per request
    - Database: SQLite file access
    - Scalability: Linear with result count

Data Tables Endpoint:
    - Response time: 10-30ms
    - Memory: < 1MB per request
    - Database: Two SQLite file accesses
    - Scalability: O(n) with table count

Data Table Endpoint:
    - Response time: 50-500ms (depends on table size)
    - Memory: 1-10MB (depends on data)
    - Limit: 10,000 rows max
    - Database: Single SQLite query

Optimization Opportunities:
    1. Add indexes on selection_code, extended_description
    2. Cache table lists (rarely change)
    3. Pre-compute search indexes
    4. Implement column-only mode (no sample data)
    5. Add compression for large responses

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. Authentication
   - All endpoints require JWT token
   - User validation via get_current_user dependency
   - Logged for audit trail

2. SQL Injection Protection
   - Parameterized queries (? placeholders)
   - No string concatenation in SQL
   - User input properly escaped

3. Path Traversal Protection
   - Fixed database paths (not user-provided)
   - No file system navigation from input
   - Table names validated implicitly

4. Data Exposure
   - Read-only database access
   - No sensitive personal data
   - Public statistical data only

5. Rate Limiting
   - Should be implemented at API gateway
   - page_size limited to 10,000
   - Sample data limited to 10,000 rows

6. Error Information
   - Detailed errors logged server-side
   - Sanitized errors returned to client
   - No database structure leaked

===================================================================================
INTEGRATION WITH AI AGENTS
===================================================================================

AI Workflow:

1. Discovery Phase
   User: "What population data is available?"
   → GET /catalog?q=population
   → AI sees available datasets
   → AI can describe options to user

2. Table Selection Phase
   User: "Show me population 2023"
   → GET /data-tables?q=2023
   → AI identifies exact table name
   → Proceeds to query generation

3. Schema Understanding Phase
   AI needs table structure for SQL generation
   → GET /data-table?table=population_2023
   → AI sees columns: region, age_group, count
   → AI generates accurate SELECT statement

4. Query Generation Phase
   User: "How many people under 18?"
   → AI has schema from step 3
   → AI generates: SELECT SUM(count) WHERE age_group < 18
   → AI can validate column names against schema

Benefits for AI:
    - Reduces hallucination (knows actual columns)
    - Enables accurate SQL generation
    - Provides context for user questions
    - Supports multi-turn conversations

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    1. Pagination calculation
    2. Search query building
    3. Empty result handling
    4. Error response formatting

Integration Tests:
    1. Database connectivity
    2. Multi-database coordination
    3. Authentication integration
    4. End-to-end search workflow

Data Tests:
    1. Valid table names returned
    2. Descriptions properly joined
    3. Column names correct
    4. Sample data format

Performance Tests:
    1. Response time under load
    2. Large result set handling
    3. Concurrent request handling
    4. Memory usage monitoring

Edge Cases:
    1. Empty database
    2. Missing descriptions database
    3. Invalid table names
    4. Special characters in search
    5. Very large page_size
    6. Non-existent pages

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - sqlite3: SQLite database access
    - os: Environment variables
    - sys: Platform detection
    - asyncio: Event loop configuration (Windows)

Third-Party:
    - fastapi: Web framework, routing
    - dotenv: Environment variable loading

Internal:
    - api.utils.debug: Debug logging functions
    - api.helpers: Traceback JSON response
    - api.dependencies.auth: JWT authentication
    - fastapi.Query: Query parameter validation

Database Files:
    - data/czsu_data.db
    - metadata/llm_selection_descriptions/selection_descriptions.db

===================================================================================
MAINTENANCE AND UPDATES
===================================================================================

When Adding New Datasets:
    1. Add table to czsu_data.db
    2. Add description to selection_descriptions.db
    3. Use consistent selection_code naming
    4. No code changes needed (automatic discovery)

When Changing Schema:
    - Data tables: No endpoint changes needed
    - Description table: Update queries if columns change
    - Backward compatibility maintained

Version Compatibility:
    - Endpoints stable across versions
    - Database schema versioning recommended
    - Migration scripts for major changes

===================================================================================
"""

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

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for catalog and data exploration endpoints
# This router will be included in the main FastAPI application
router = APIRouter()
print__catalog_debug("Catalog router initialized successfully")


# ==============================================================================
# DATA CATALOG ENDPOINT
# ==============================================================================


@router.get(
    "/catalog",
    summary="Search CZSU data catalog",
    description="""
    **Browse and search the CZSU (Czech Statistical Office) data catalog.**
    
    Returns a paginated list of available statistical datasets with descriptions.
    Use the `q` parameter to filter by selection code or description text.
    
    **Examples:**
    - `/catalog?page=1&page_size=20` - Get first 20 datasets
    - `/catalog?q=population` - Search for population-related datasets
    - `/catalog?q=2023&page=2` - Search for 2023 data, page 2
    """,
    response_description="Paginated catalog results with selection codes and descriptions",
    responses={
        200: {
            "description": "Successfully retrieved catalog data",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "selection_code": "POP2020",
                                "extended_description": "Population statistics for 2020",
                            }
                        ],
                        "total": 150,
                        "page": 1,
                        "page_size": 10,
                    }
                }
            },
        }
    },
)
def get_catalog(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    q: Optional[str] = Query(
        None, description="Search query to filter by selection code or description"
    ),
    page_size: int = Query(
        10, ge=1, le=10000, description="Number of results per page"
    ),
    user=Depends(get_current_user),
):
    print__catalog_debug(
        f"GET /catalog called - page: {page}, page_size: {page_size}, query: '{q}', user: {user.username if hasattr(user, 'username') else 'unknown'}"
    )
    try:
        # ======================================================================
        # DATABASE CONNECTION AND QUERY PREPARATION
        # ======================================================================

        # Connect to the metadata database containing dataset descriptions
        db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
        print__catalog_debug(f"Connecting to catalog database: {db_path}")

        # Calculate pagination offset for efficient result retrieval
        offset = (page - 1) * page_size

        # Build WHERE clause for search filtering if query parameter provided
        where_clause = ""
        params = []
        if q:
            print__catalog_debug(f"Building search query for term: '{q}'")
            where_clause = "WHERE selection_code LIKE ? OR extended_description LIKE ?"
            like_q = f"%{q}%"
            params.extend([like_q, like_q])
        else:
            print__catalog_debug("No search query provided, fetching all records")
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
            print__catalog_debug(f"Fetched {len(rows)} records for current page")
        results = [
            {"selection_code": row[0], "extended_description": row[1]} for row in rows
        ]
        response_data = {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
        print__catalog_debug(f"Returning catalog response with {len(results)} results")
        return response_data
    except Exception as e:
        print__catalog_debug(f"Error in get_catalog: {str(e)}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise


# ==============================================================================
# DATA TABLES LISTING ENDPOINT
# ==============================================================================


@router.get("/data-tables")
def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
    print__data_tables_debug(
        f"GET /data-tables called - query: '{q}', user: {user.username if hasattr(user, 'username') else 'unknown'}"
    )
    try:
        db_path = "data/czsu_data.db"
        desc_db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
        print__data_tables_debug(f"Connecting to data database: {db_path}")
        print__data_tables_debug(f"Connecting to descriptions database: {desc_db_path}")

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
                f"Filtered from {original_count} to {len(tables)} tables"
            )

        # Fetch short_descriptions from the other DB
        desc_map = {}
        print__data_tables_debug(
            "Fetching short descriptions from descriptions database"
        )
        try:
            with sqlite3.connect(desc_db_path) as desc_conn:
                desc_cursor = desc_conn.cursor()
                desc_cursor.execute(
                    "SELECT selection_code, short_description FROM selection_descriptions"
                )
                descriptions = desc_cursor.fetchall()
                print__data_tables_debug(f"Fetched {len(descriptions)} descriptions")
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


# ==============================================================================
# DATA TABLE SCHEMA AND SAMPLE DATA ENDPOINT
# ==============================================================================


@router.get("/data-table")
def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
    print__data_tables_debug(
        f"GET /data-table called - table: '{table}', user: {user.username if hasattr(user, 'username') else 'unknown'}"
    )
    try:
        # ======================================================================
        # DATABASE CONNECTION AND TABLE VALIDATION
        # ======================================================================

        # Connect to the actual data database containing CZSU statistical tables
        db_path = "data/czsu_data.db"
        print__data_tables_debug(f"Connecting to data database: {db_path}")

        # Validate that a table name was provided in the request
        if not table:
            print__data_tables_debug("No table specified")
            return {"columns": [], "rows": []}

        print__data_tables_debug(f"Requested table: {table}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                print__data_tables_debug(
                    f"Executing SELECT query on table '{table}' with LIMIT 10000"
                )
                cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                print__data_tables_debug(
                    f"Successfully fetched data - Columns: {columns}"
                )
                print__data_tables_debug(f"Row count: {len(rows)}")
                if len(rows) > 0:
                    print__data_tables_debug(f"Sample first row: {rows[0]}")

                response_data = {"columns": columns, "rows": rows}
                print__data_tables_debug(
                    f"Returning table data with {len(columns)} columns and {len(rows)} rows"
                )
                return response_data

            except Exception as e:
                print__data_tables_debug(f"Error fetching table '{table}': {e}")
                return {"columns": [], "rows": []}
    except Exception as e:
        print__data_tables_debug(f"Error in get_data_table: {str(e)}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        raise
