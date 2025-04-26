import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .utils import DataAnalysisState, PandasQueryTool, initialize_state, process_query, save_result

# Load environment variables - still needed for API keys
load_dotenv()

def create_agent():
    """Create and return the agent with tools."""
    # Initialize tools
    tools = [PandasQueryTool()]
    
    # Initialize the LLM with hardcoded defaults
    llm = AzureChatOpenAI(
        deployment_name='gpt-4o__test1',
        model_name='gpt-4o',
        openai_api_version='2024-05-01-preview',
        temperature=0.7,
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English,
specializing in converting natural language queries into pandas operations.

Your goal is to convert user queries (in Czech or English) into pandas operations by:
- Understanding queries in both Czech and English
- Mapping between Czech/English terms and schema metadata
- Handling bilingual data values and column names
- Constructing accurate pandas queries regardless of input language

IMPORTANT: When presenting any numeric results, always output numbers as plain digits with NO thousands separators, commas, spaces, or formatting (e.g., use '716056' not '716,056' or '716 056').
    """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent with tools
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_graph():
    """Create the workflow graph for data analysis."""
    # Initialize the agent
    agent_executor = create_agent()
    
    # Create the workflow graph
    workflow = StateGraph(DataAnalysisState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("process", lambda state: process_query(state, agent_executor))
    workflow.add_node("save", save_result)
    
    # Define edges - simple linear flow
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "process")
    workflow.add_edge("process", "save")
    workflow.add_edge("save", END)
    
    # Compile the graph with memory saver
    return workflow.compile(checkpointer=MemorySaver())
