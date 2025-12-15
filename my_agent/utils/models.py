"""LLM model configuration and initialization.

This module provides functions for creating and configuring language model instances
used throughout the application, with support for both sync and async operations.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings


# ===============================================================================
# Azure Chat Models
# ===============================================================================
def get_azure_openai_chat_llm(
    deployment_name: str,
    model_name: str,
    openai_api_version: str,
    temperature: float = 0.0,
    streaming: bool = False,
) -> AzureChatOpenAI:
    """Get an instance of Azure OpenAI Chat LLM with configurable parameters.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        deployment_name (str): Azure deployment name (e.g., "gpt-4o__test1")
        model_name (str): Model name (e.g., "gpt-4o", "gpt-4o-mini")
        openai_api_version (str): Azure OpenAI API version (e.g., "2024-05-01-preview")
        temperature (float): Temperature setting for generation randomness (default: 0.0)
        streaming (bool): Enable streaming mode (default: False)

    Returns:
        AzureChatOpenAI: Configured LLM instance with async support
    """
    return AzureChatOpenAI(
        deployment_name=deployment_name,
        model_name=model_name,
        openai_api_version=openai_api_version,
        temperature=temperature,
        streaming=streaming,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


# ===============================================================================
# Google Gemini Models
# ===============================================================================
def get_gemini_llm(model_name="gemini-3-pro-preview", temperature=0.0):
    """Get an instance of Google Gemini LLM with standard configuration.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        model_name (str): The Gemini model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance with async support
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        include_thoughts=True,  # CRITICAL: Automatically handles thought signatures in multi-turn conversations (requires langchain-google-genai >= 3.1.0)
    )


def get_gemini_llm_test():
    # Test get_gemini_llm() with a simple message
    print("\nTesting get_gemini_llm()...")
    llm = get_gemini_llm()
    messages = [{"role": "user", "content": "Say hello"}]
    print(f"\nInput message:\n{messages[0]['content']}")
    response = llm.invoke(messages)
    print(f"\nResponse from LLM:\n{response.content}")


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


def get_ollama_llm_test(model_name="llama3.2:1b", prompt="Hi"):
    """Test the OLLAMA LLM with a simple message"""
    print("\nTesting get_ollama_llm()...")

    try:
        llm = get_ollama_llm(model_name=model_name)
        print(f"✓ OLLAMA LLM instance created successfully!")
        print(f"Model: {llm.model_name}")
        print(f"Base URL: {llm.openai_api_base}")
        print(f"Temperature: {llm.temperature}")

        # Test with a simple prompt
        print(f"\nSending test prompt: '{prompt}'")
        response = llm.invoke(prompt)
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
    pass  # No test code to run
