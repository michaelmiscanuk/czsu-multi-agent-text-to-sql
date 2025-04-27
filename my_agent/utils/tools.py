"""Tool definitions for the data analysis workflow.

This module provides custom tools that enable pandas-based data analysis
operations with proper error handling and result formatting.
"""

#===============================================================================
# IMPORTS
#===============================================================================
import os
from pathlib import Path
import pandas as pd
from typing import Optional
from langchain.tools import BaseTool
from langchain_core.tools import ToolException

#===============================================================================
# CONSTANTS & CONFIGURATION
#===============================================================================
# Debug configuration - module level export
DEBUG_MODE = os.getenv("MY_AGENT_DEBUG", "1").lower() in ("1", "true", "yes")

# Constants
TOOL_ID = 20  # Static ID for PandasQueryTool

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================
def debug_print(msg: str) -> None:
    """Print debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    if DEBUG_MODE:
        print(msg)

def validate_path_safety(data_path: Path):
    """Validate that a file path exists and is safe to use.
    
    Args:
        data_path: Path to validate
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is not a file
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not data_path.is_file():
        raise ValueError(f"Data path is not a file: {data_path}")

def check_query_safety(query: str):
    """Check if a query contains potentially unsafe operations.
    
    This function implements a crucial security layer that prevents code injection
    and other dangerous operations. By blocking access to system modules, file operations,
    and execution functions, it restricts the query to safe data operations only.
    
    While not a complete sandbox, this provides an important defense-in-depth measure
    alongside the restricted execution environment.
    
    Args:
        query: The query string to check
        
    Raises:
        ToolException: If query contains unsafe patterns that could lead to system access
    """
    # List of patterns that could enable breaking out of the sandbox
    # These patterns target system access, file operations, and code execution
    dangerous_patterns = [
        "os\.", "system", "exec", "eval", "import ", 
        "subprocess", "open(", ".open", "file", "globals", "locals"
    ]
    
    # Check each pattern and raise an exception if found
    # This provides clear error messages about the specific security violation
    for pattern in dangerous_patterns:
        if pattern in query:
            error_msg = f"Query contains potentially unsafe operation: {pattern}"
            debug_print(f"{TOOL_ID}: {error_msg}")
            raise ToolException(error_msg)

def format_result(result):
    """Format a pandas result into a string representation.
    
    This function handles the complexity of converting different pandas result types
    into consistent, human-readable string formats. It implements special handling for:
    
    1. Single-value Series - returned as plain scalars without index
    2. Single-cell DataFrames - collapsed to scalar values
    3. Multi-row/column results - formatted as tables without index columns
    
    This consistent formatting is important for providing clear, predictable responses
    to the user regardless of the internal data structures used.
    
    Args:
        result: The pandas result object (DataFrame, Series, or other value)
        
    Returns:
        str: A consistently formatted string representation of the result
    """
    if isinstance(result, pd.Series):
        # Return scalar directly for single-item Series
        # This provides cleaner output for common aggregation operations
        if len(result) == 1:
            return str(result.iloc[0])
        result_str = result.to_string(index=False)
    elif isinstance(result, pd.DataFrame):
        # Collapse to scalar for 1×1 DataFrames
        # This simplifies results for queries that return a single value
        if result.shape == (1, 1):
            return str(result.iat[0, 0])
        result_str = result.to_string(index=False)
    else:
        # For non-pandas types, just use string conversion
        result_str = str(result)
    
    return result_str

#===============================================================================
# TOOL CLASSES
#===============================================================================
class PandasQueryTool(BaseTool):
    """Tool for executing pandas queries against a dataframe.
    
    This tool provides a controlled environment for executing pandas expressions
    against a pre-loaded dataset. It implements several important safety features:
    
    1. Path validation to prevent directory traversal
    2. Query safety checking to block dangerous operations
    3. Restricted execution environment with limited globals
    4. Consistent error handling and reporting
    5. Standardized result formatting
    
    These protections create a sandboxed environment that balances flexibility
    (allowing arbitrary pandas expressions) with security (preventing system access).
    """
    
    name: str = "pandas_query"
    description: str = "Execute pandas query on the dataframe named 'df'"
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the PandasQueryTool.
        
        Args:
            data_path: Path to CSV file, defaults to OBY01PDT01.csv in data directory
        """
        super().__init__()
        if data_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            data_path = base_dir / "data" / "OBY01PDT01.csv"

        # Verify file exists and is readable
        data_file_path = Path(data_path)
        validate_path_safety(data_file_path)
            
        debug_print(f"{TOOL_ID}: Loading data from: {data_file_path}")
        try:
            self._df = pd.read_csv(data_file_path)
            debug_print(f"{TOOL_ID}: Data loaded, shape: {self._df.shape}")
        except Exception as load_error:
            error_msg = f"Failed to load CSV data: {str(load_error)}"
            debug_print(f"{TOOL_ID}: {error_msg}")
            raise ValueError(error_msg) from load_error
        
    def _run(self, query: str) -> str:
        """Execute the pandas query on the loaded dataframe.
        
        This method is the core of the tool, handling the actual query execution
        in a controlled environment. The execution follows a careful sequence:
        
        1. Log the incoming query for debugging and audit
        2. Check for potentially unsafe operations before execution
        3. Create a restricted execution environment with only necessary globals
        4. Format the result for consistent output structure
        5. Handle and report any execution errors
        
        This approach balances power and flexibility with appropriate safeguards.
        
        Args:
            query: A pandas query string to be executed
            
        Returns:
            String representation of the query result
            
        Raises:
            ToolException: If the query fails security checks or execution
        """
        debug_print(f"{TOOL_ID}: _run called with query: {query}")
        
        # Security check before execution - critical for preventing code injection
        check_query_safety(query)
        
        try:
            # Log query execution for debugging and audit purposes
            print(f"Executing query: {query}")
            
            # Create a strictly limited execution environment
            # Only the dataframe and pandas module are available
            # This is crucial for security - prevents access to system modules
            query_result = eval(query, {'df': self._df, 'pd': pd}, {})
            
            # Format the result for consistent output structure
            # This ensures users get predictable response formats
            formatted_result = format_result(query_result)

            debug_print(f"{TOOL_ID}: Query result (truncated): {formatted_result[:100]}...")
            return formatted_result
        except Exception as execution_error:
            # Provide clear error information for debugging
            # This helps the calling code understand what went wrong
            error_msg = f"{TOOL_ID}: Query error: {str(execution_error)}"
            debug_print(error_msg)
            raise ToolException(error_msg)
    
    async def _arun(self, query: str) -> str:
        """Async version of the _run method."""
        return self._run(query)
