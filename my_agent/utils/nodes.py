import os
import json
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from ..utils.state import DataAnalysisState  # Import the state class

def load_schema() -> Dict[str, Any]:
    """Load the schema metadata from the JSON file."""
    # Get the path to the metadata file
    metadata_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "metadata")
    schema_path = os.path.join(metadata_dir, 'OBY01PDT01_metadata.json')
    
    # Load and return the schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def initialize_state(state: DataAnalysisState) -> DataAnalysisState:
    """Initialize the state with schema and prompt."""    
    # Get the prompt from environment variable or use default
    prompt = os.getenv("ANALYSIS_PROMPT", "What is the amount of men in Prague at the end of Q3 2024?")
    
    # Load schema metadata
    schema = load_schema()
    
    # Update state using attribute assignment
    state.prompt = prompt
    state.data_schema = schema
    state.messages.append(HumanMessage(content=f"""
        I need to analyze this data based on the following prompt: {prompt}
        
        Here is the schema metadata: {json.dumps(schema, ensure_ascii=False, indent=2)}
        
        Please convert this prompt into a pandas query that correctly processes the input.
    """))
    
    return state

def process_query(state: DataAnalysisState, agent_executor) -> DataAnalysisState:
    """Process the query using the agent."""
    # Extract the content of the most recent message to use as input
    if state.messages and isinstance(state.messages[-1], HumanMessage):
        last_message_content = state.messages[-1].content
    else:
        last_message_content = f"Analyze this data based on the prompt: {state.prompt}"
    
    print("\nInvoking agent executor...")
    
    # Invoke the agent with both formats (input and messages)
    result = agent_executor.invoke({
        "input": last_message_content,  # For the prompt template
        "messages": state.messages      # For the agent's conversation history
    })
    
    # Debug output
    print(f"\nAgent result keys: {result.keys()}")
    
    # Update state with the result
    state.messages.extend(result["messages"][-1:])  # Add only the latest message
    
    # Save the complete agent output
    if "output" in result:
        print(f"\nFound output: {result['output']}")
        state.result = result["output"]
    else:
        print("\nNo direct output found, extracting from messages...")
        
        # Extract the result from the AI message
        ai_message = result["messages"][-1]
        if isinstance(ai_message, AIMessage):
            response_content = ai_message.content
            
            # Save AI response
            state.result = response_content
            
            # If there are intermediate steps, they contain the tool outputs
            if "intermediate_steps" in result and result["intermediate_steps"]:
                print(f"\nFound intermediate steps: {len(result['intermediate_steps'])}")
                
                # Extract tool outputs from intermediate steps
                tool_outputs = []
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:  # Each step is (action, output)
                        tool_outputs.append(str(step[1]))
                
                if tool_outputs:
                    state.result += "\n\nTool Results:\n" + "\n".join(tool_outputs)
    
    # Debug the current state result
    print(f"\nCurrent result value (first 100 chars): {str(state.result)[:100]}...")
    
    return state

def save_result(state: DataAnalysisState) -> DataAnalysisState:
    """Save the result to a file."""
    result_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    result_path = os.path.join(result_dir, "analysis_results.txt")
    
    # Debug the state before saving
    print(f"\nState before saving - result exists: {'result' in state.__dict__}")
    print(f"State before saving - result length: {len(state.result) if state.result else 0}")
    
    # Ensure we have content to write
    if not state.result:
        print("\nWARNING: No results found in state object!")
        state.result = "No results were generated."
    
    with open(result_path, "a", encoding='utf-8') as f:
        f.write(f"\nPrompt: {state.prompt}\n")
        f.write(f"Result: {state.result}\n")
        f.write("-" * 50 + "\n")
    
    # Add confirmation message
    print(f"Results saved to {result_path}")
    
    return state
