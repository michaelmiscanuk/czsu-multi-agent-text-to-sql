import os
import pandas as pd
from typing import Optional
from langchain.tools import BaseTool
from langchain_core.tools import ToolException

class PandasQueryTool(BaseTool):
    name: str = "pandas_query"
    description: str = "Execute pandas query on the dataframe named 'df'"
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize with the path to the CSV file."""
        super().__init__()
        
        if data_path is None:
            # Navigate to project root and then to data directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(base_dir, "data", 'OBY01PDT01.csv')
        
        self._df = pd.read_csv(data_path)
        
    def _run(self, query: str) -> str:
        """Execute the pandas query."""
        try:
            result = eval(query, {'df': self._df, 'pd': pd}, {})
            return str(result)
        except Exception as e:
            raise ToolException(f"Query error: {str(e)}")
    
    async def _arun(self, query: str) -> str:
        """Execute the pandas query asynchronously."""
        return self._run(query)
