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
from typing import Optional, Type, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import ToolException
import json  # Import json for structured result formatting

#===============================================================================
# CONSTANTS & CONFIGURATION
#===============================================================================
# Debug configuration - module level export
DEBUG_MODE = os.getenv("MY_AGENT_DEBUG", "1").lower() in ("1", "true", "yes")

# Constants
TOOL_ID = 20  # Static ID for PandasQueryTool

# Load data once at module level
base_dir = Path(__file__).resolve().parents[2]
data_path = base_dir / "data" / "OBY01PDT01.csv"
df = pd.read_csv(data_path)

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

#===============================================================================
# TOOL CLASSES
#===============================================================================

class PandasQueryInput(BaseModel):
    """Schema for pandas query input."""
    query: str = Field(description="pandas query to execute on the dataframe named 'df'")

class PandasQueryTool(BaseTool):
    """Tool for executing pandas queries against a dataframe."""
    name: ClassVar[str] = "pandas_query"
    description: ClassVar[str] = "Execute pandas query on the dataframe named 'df'"
    args_schema: Type[BaseModel] = PandasQueryInput
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            check_query_safety(query)
            
            # Debug print query
            debug_print(f"{TOOL_ID}: =====================================")
            debug_print(f"{TOOL_ID}: Executing query:")
            debug_print(f"{TOOL_ID}: {query}")
            debug_print(f"{TOOL_ID}: =====================================")
            
            # Execute query
            result = eval(query, {'df': df, 'pd': pd}, {})
            
            # Format the result
            if isinstance(result, pd.Series) and len(result) == 1:
                result_value = str(result.iloc[0])
            elif isinstance(result, pd.DataFrame) and result.shape == (1, 1):
                result_value = str(result.iat[0, 0])
            elif isinstance(result, (pd.Series, pd.DataFrame)):
                result_value = result.to_string(index=False)
            else:
                result_value = str(result)
            
            # Debug print result
            debug_print(f"{TOOL_ID}: Query result:")
            debug_print(f"{TOOL_ID}: {result_value}")
            debug_print(f"{TOOL_ID}: =====================================")
            
            # Return both query and result for state tracking
            return json.dumps({
                "query": query,
                "result": result_value
            }, ensure_ascii=False)
            
        except Exception as e:
            raise ToolException(f"Query error: {str(e)}")

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the pandas query asynchronously."""
        return self._run(query, run_manager)
