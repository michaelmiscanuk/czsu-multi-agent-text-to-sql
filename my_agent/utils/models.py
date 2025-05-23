"""LLM model configuration and initialization.

This module provides functions for creating and configuring language model instances
used throughout the application, with support for both sync and async operations.
"""

import os
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

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

def get_azure_embedding_model():
    """Get an instance of Azure OpenAI Embedding model with standard configuration (context7 style).
    
    Returns:
        AzureOpenAI: Configured embedding client instance
    """
    return AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )

if __name__ == "__main__":
    # Test the embedding model with some sample data
    embedding_client = get_azure_embedding_model()
    deployment = "text-embedding-3-large__test1"
    response = embedding_client.embeddings.create(
        input=["first phrase", "second phrase", "third phrase"],
        model=deployment
    )
    for item in response.data:
        length = len(item.embedding)
        print(
            f"data[{item.index}]: length={length}, "
            f"[{item.embedding[0]}, {item.embedding[1]}, ...,{item.embedding[length-2]}, {item.embedding[length-1]}]"
        )
    print(response.usage)