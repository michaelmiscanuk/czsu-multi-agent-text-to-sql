import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .utils.state import DataAnalysisState
from .utils.tools import PandasQueryTool
from .utils.nodes import initialize_state, process_query, save_result

# Load environment variables
load_dotenv()

def create_agent_executor():
    """Create and return the agent executor."""
    # Initialize tools
    tools = [PandasQueryTool()]
    
    # Initialize the LLM with direct Azure OpenAI configuration
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o__test1",
        model_name="gpt-4o",
        openai_api_version="2024-05-01-preview",
        temperature=0.7,
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )
    
    # Create the prompt template for the data analysis
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English,
specializing in converting natural language queries into pandas operations.

Your goal is to convert user queries (in Czech or English) into pandas operations by:
- Understanding queries in both Czech and English
- Mapping between Czech/English terms and schema metadata
- Handling bilingual data values and column names
- Constructing accurate pandas queries regardless of input language

Process the prompt by:
1. Identifying key terms in either language
2. Matching terms to their Czech equivalents in schema
3. Handling Czech diacritics and special characters
4. Converting geographical names between languages and similar concepts.

Create pandas query by:
- Using exact column names from schema (can be Czech or English)
- Matching user prompt terms to correct data values
- Ensuring proper string matching for Czech characters
- Be careful that data can contain records for totals

Examples:
df[df["Column1"] == "Value1"]["value"]
df[df["Column1"].isin(["Value1", "Value2"])]["value"].sum()
df[(df["Column1"] == "Value1") & (df["Column2"] == "Value2")]["value"].mean()
df.groupby("Column1")["value"].sum()
    """
    
    # Updated prompt template with the required agent_scratchpad placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent and bind to tools
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create and return the agent executor
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_graph():
    """Create and return the graph for data analysis."""
    # Initialize the agent executor
    agent_executor = create_agent_executor()
    
    # Create the workflow graph
    workflow = StateGraph(DataAnalysisState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("process", lambda state: process_query(state, agent_executor))
    workflow.add_node("save", save_result)
    
    # Define edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "process")
    workflow.add_edge("process", "save")
    workflow.add_edge("save", END)
    
    # Compile the graph with checkpoint support
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
