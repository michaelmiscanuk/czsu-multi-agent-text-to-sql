import pandas as pd
from typing import Dict, Any, Optional, List, Type, ClassVar
import os
from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from pydantic import Field, PrivateAttr

class PandasQueryTool(BaseTool):
    name: str = Field("pandas_query", description="Tool name")
    description: str = Field(
        "Execute pandas query on the dataframe named 'df'",
        description="Tool description"
    )
    
    # Use PrivateAttr for attributes that shouldn't be part of the schema
    _df: pd.DataFrame = PrivateAttr()
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize with the path to the CSV file."""
        # Get the default path if none provided
        if data_path is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            data_path = os.path.join(data_dir, 'OBY01PDT01.csv')
        
        # Initialize the Pydantic model first
        super().__init__()
        
        # Then set private attributes
        self._df = pd.read_csv(data_path)
        
    def _run(self, query: str) -> str:
        """Execute the pandas query."""
        try:
            # Use the private attribute in the evaluation
            result = eval(query, {'df': self._df, 'pd': pd}, {})
            return str(result)
        except Exception as e:
            raise ToolException(f"Query error: {str(e)}")
    
    async def _arun(self, query: str) -> str:
        """Execute the pandas query asynchronously."""
        # For simple tools, we can just call the synchronous version
        return self._run(query)
