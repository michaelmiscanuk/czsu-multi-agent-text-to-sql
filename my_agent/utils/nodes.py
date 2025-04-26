import os
import json
from langchain_core.messages import HumanMessage

from .state import DataAnalysisState

def load_schema():
    """Load the schema metadata from the JSON file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    schema_path = os.path.join(base_dir, "metadata", 'OBY01PDT01_metadata.json')
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def initialize_state(state: DataAnalysisState) -> DataAnalysisState:
    """Initialize the state with the prompt."""    
    # Load schema metadata
    schema = load_schema()
    
    # Add the initial message
    state.messages.append(HumanMessage(content=f"""      
        
        1. Read and analyze the provided inputs:
        - User prompt (in Czech or English)
        - Schema metadata (containing Czech column names and values)
        
        2. Process the prompt by:
        - Identifying key terms in either language
        - Matching terms to their Czech equivalents in schema
        - Handling Czech diacritics and special characters
        - Converting geographical names between languages and similar concepts.

        3. Create pandas query by:
        - Using exact column names from schema (can be Czech or English)
        - Matching user prompt terms to correct data values (in the schema we can 
            find list of unique values in specific column)
        - Ensuring proper string matching for Czech characters
        - Be careful that data can contain records for totals, for example:
            in Column "CZ, Region" we can find "Czech Republic" and "Regions". 
            So you need to be carefully examine dimensional unique values in a schema.

        Examples:
        df[df["Column1"] == "Value1"]["value"]
        df[df["Column1"].isin(["Value1", "Value2"])]["value"].sum()
        df[(df["Column1"] == "Value1") & (df["Column2"] == "Value2")]["value"].mean()
        df.groupby("Column1")["value"].sum()

        4. Execute and validate:
        - Use pandas query tool to execute the constructed query
        - Verify if results match the original query intent
        - Handle any errors by refining the pandas query
        - Return results in YAML format

        5. Post-process results:
        - Ensure all Czech characters are preserved
        - Format output maintaining bilingual clarity

        6. If a user asks for some kind of aggregation, like sum, mean, etc.:
        - Use the appropriate pandas function to perform the aggregation
        - Return the result in a clear and concise format
        
        User prompt to analyze: {state.prompt}
        Schema metadata: {json.dumps(schema, ensure_ascii=False, indent=2)}
    
    """))
    
    return state

def process_query(state: DataAnalysisState, agent_executor) -> DataAnalysisState:
    """Process the query using the agent."""
    # Get latest message
    last_message_content = state.messages[-1].content if state.messages else state.prompt
    
    # Invoke the agent
    result = agent_executor.invoke({
        "input": last_message_content,
        "messages": state.messages
    })
    
    # Update state
    state.messages.extend(result["messages"][-1:])
    
    # Save output - ensure it's a string (JSON serializable)
    output_content = result.get("output", result["messages"][-1].content)
    state.result = str(output_content)
    
    return state

def save_result(state: DataAnalysisState) -> DataAnalysisState:
    """Save the result to a file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    result_path = os.path.join(base_dir, "analysis_results.txt")
    
    # Create a JSON serializable object
    result_obj = {
        "prompt": state.prompt,
        "result": state.result
    }
    
    # Save as text
    with open(result_path, "a", encoding='utf-8') as f:
        f.write(f"Prompt: {state.prompt}\n")
        f.write(f"Result: {state.result}\n")
        f.write(f"----------------------------------------------------------------------------\n")
    
    # Also save as JSON for better interoperability
    json_result_path = os.path.join(base_dir, "analysis_results.json")
    
    # Read existing results if file exists
    existing_results = []
    if os.path.exists(json_result_path):
        try:
            with open(json_result_path, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []
    
    # Append new result
    existing_results.append(result_obj)
    
    # Save updated results
    with open(json_result_path, "w", encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Result saved to {result_path} and {json_result_path}")
    
    return state
