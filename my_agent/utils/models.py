"""LLM model configuration and initialization.

This module provides functions for creating and configuring language model instances
used throughout the application, with support for both sync and async operations.
"""

import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings


# ===============================================================================
# Azure Chat Models
# ===============================================================================
def get_azure_llm_gpt_4o(temperature=0.0):
    """Get an instance of Azure OpenAI LLM with standard configuration.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        temperature (float): Temperature setting for generation randomness

    Returns:
        AzureChatOpenAI: Configured LLM instance with async support
    """
    return AzureChatOpenAI(
        deployment_name="gpt-4o__test1",
        model_name="gpt-4o",
        openai_api_version="2024-05-01-preview",
        temperature=temperature,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def get_azure_llm_gpt_4o_test():
    # Test get_azure_llm_gpt_4o() with a simple message
    print("\nTesting get_azure_llm_gpt_4o()...")
    llm = get_azure_llm_gpt_4o()
    messages = [{"role": "user", "content": "Say hello"}]
    print(f"\nInput message:\n{messages[0]['content']}")
    response = llm.invoke(messages)
    print(f"\nResponse from LLM:\n{response.content}")


def get_azure_llm_gpt_4o_mini(temperature=0.0):
    """Get an instance of Azure OpenAI GPT-4o Mini LLM with standard configuration.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        temperature (float): Temperature setting for generation randomness

    Returns:
        AzureChatOpenAI: Configured LLM instance with async support
    """
    return AzureChatOpenAI(
        deployment_name="gpt-4o-mini-mimi2",
        model_name="gpt-4o-mini",
        openai_api_version="2024-05-01-preview",
        temperature=temperature,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def get_azure_llm_gpt_4o_mini_test():
    # Test get_azure_llm_gpt_4o() with a simple message
    print("\nTesting get_azure_llm_gpt_4o_mini()...")
    llm = get_azure_llm_gpt_4o()
    messages = [{"role": "user", "content": "Say hello"}]
    print(f"\nInput message:\n{messages[0]['content']}")
    response = llm.invoke(messages)
    print(f"\nResponse from LLM:\n{response.content}")


# -------------------------------------------------------------------------------
# def get_azure_llm_gpt_4o_4_1(temperature=0.0):
#     """Get an instance of Azure OpenAI LLM with standard configuration.

#     The returned model instance supports both sync (invoke) and async (ainvoke)
#     operations for flexibility in different execution contexts.

#     Args:
#         temperature (float): Temperature setting for generation randomness

#     Returns:
#         AzureChatOpenAI: Configured LLM instance with async support
#     """
#     return AzureChatOpenAI(
#         deployment_name='gpt-4.1___test1',
#         model_name='gpt-4.1',
#         openai_api_version='2024-05-01-preview',
#         temperature=temperature,
#         azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
#         api_key=os.getenv('AZURE_OPENAI_API_KEY')
#     )

# def get_azure_llm_gpt_4o_4_1_test():
#     # Test get_azure_llm_gpt_4o() with a simple message
#     print("\nTesting get_azure_llm_gpt_4o()...")
#     llm = get_azure_llm_gpt_4o_4_1()
#     messages = [{"role": "user", "content": "What LLM Model version exactly you are?"}]
#     print(f"\nInput message:\n{messages[0]['content']}")
#     response = llm.invoke(messages)
#     print(f"\nResponse from LLM:\n{response.content}")


# ===============================================================================
# Local OLLAMA Models
# ===============================================================================
def get_ollama_llm(
    model_name="llama3.2:1b", base_url="http://localhost:11434/v1", temperature=0.0
):
    """Get an instance of local OLLAMA LLM with standard configuration.

    Uses LangChain's ChatOpenAI with local OLLAMA endpoint for compatibility.

    Args:
        model_name (str): The OLLAMA model name (e.g., "llama3.2:1b", "smollm:latest", "qwen:7b")
        base_url (str): The base URL for the local OLLAMA server (with /v1 endpoint)
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatOpenAI: Configured OLLAMA LLM instance compatible with LangChain
    """
    # Set a dummy API key for local OLLAMA (required by ChatOpenAI but not used)
    os.environ["OPENAI_API_KEY"] = "ollama-local-dummy-key"

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        api_key="ollama-local-dummy-key",  # Required but not used for local OLLAMA
    )


def get_ollama_llm_test():
    """Test the OLLAMA LLM with a simple message"""
    print("\nTesting get_ollama_llm()...")

    try:
        llm = get_ollama_llm()
        print(f"✓ OLLAMA LLM instance created successfully!")
        print(f"Model: {llm.model_name}")
        print(f"Base URL: {llm.openai_api_base}")
        print(f"Temperature: {llm.temperature}")

        # Test with a simple prompt
        print("\nSending test prompt: 'Hi'")
        response = llm.invoke("Hi")
        print(f"\nResponse from OLLAMA:\n{response.content}")
        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error testing OLLAMA LLM: {str(e)}")
        print("Make sure:")
        print("1. OLLAMA is running (ollama serve)")
        print("2. You have a model downloaded (ollama pull llama3.2)")
        print("3. The model name matches what you have installed")


# ===============================================================================
# Azure Embedding Models
# ===============================================================================
def get_azure_embedding_model():
    """Get an instance of Azure OpenAI Embedding model with standard configuration.

    Returns:
        AzureOpenAI: Configured embedding client instance
    """
    return AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def get_azure_embedding_model_test():
    # Test the embedding model with some sample data
    embedding_client = get_azure_embedding_model()
    deployment = "text-embedding-3-large__test1"
    response = embedding_client.embeddings.create(
        input=["first phrase", "second phrase", "third phrase"],
        model=deployment,
    )
    for item in response.data:
        length = len(item.embedding)
        print(
            f"data[{item.index}]: length={length}, "
            f"[{item.embedding[0]}, {item.embedding[1]}, ...,{item.embedding[length-2]}, {item.embedding[length-1]}]"
        )
    print(response.usage)


def get_langchain_azure_embedding_model(model_name="text-embedding-3-large__test1"):
    """Get a LangChain AzureOpenAIEmbeddings instance with standard configuration.

    Args:
        model_name (str): The name of the embedding model deployment

    Returns:
        AzureOpenAIEmbeddings: Configured embedding model instance
    """

    return AzureOpenAIEmbeddings(
        model=model_name,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        deployment=model_name,
    )


def get_langchain_azure_embedding_model_test():
    # Test the LangChain AzureOpenAIEmbeddings with some sample data
    embedding_model = get_langchain_azure_embedding_model()
    phrases = ["first phrase", "second phrase", "third phrase"]
    vectors = embedding_model.embed_documents(phrases)
    for i, vector in enumerate(vectors):
        print(
            f"data[{i}]: length={len(vector)}, [{vector[0]}, {vector[1]}, ..., {vector[-2]}, {vector[-1]}]"
        )


if __name__ == "__main__":
    # get_azure_embedding_model_test()
    # get_azure_llm_gpt_4o_test()
    # get_azure_llm_gpt_4o_mini_test()

    llm = get_ollama_llm("qwen:7b")

    # Then use it like any LangChain model
    response = llm.invoke("Hi")
    print(response.content)
