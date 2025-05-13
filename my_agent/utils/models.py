"""LLM model configuration and initialization.

This module provides functions for creating and configuring language model instances
used throughout the application, with support for both sync and async operations.
"""

import os
from langchain_openai import AzureChatOpenAI
from phoenix.evals import OpenAIModel

def get_azure_llm(temperature=0.0):
    """Get an instance of Azure OpenAI LLM with standard configuration.
    
    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.
    
    Args:
        temperature (float): Temperature setting for generation randomness
        
    Returns:
        AzureChatOpenAI: Configured LLM instance with async support
    """
    return AzureChatOpenAI(
        deployment_name='gpt-4o__test1',
        model_name='gpt-4o',
        openai_api_version='2024-05-01-preview',
        temperature=temperature,
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )
    
    
def get_azure_llm_phoenix_evals(temperature=0.0):
    """Get an instance of Azure OpenAI LLM with standard configuration.
    
    Args:
        temperature (float): Temperature setting for generation randomness
        
    Returns:
        AzureChatOpenAI: Configured LLM instance
    """
    return  OpenAIModel(
        model="gpt-4o__test1",        # For Azure, this is the deployment name
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        temperature=0.0
    )