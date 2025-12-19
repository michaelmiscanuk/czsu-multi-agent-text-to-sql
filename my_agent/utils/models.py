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
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_mistralai import ChatMistralAI


# ===============================================================================
# Azure Chat Models
# ===============================================================================
def get_azure_openai_chat_llm(
    deployment_name: str,
    model_name: str,
    openai_api_version: str,
    temperature: Optional[float] = None,
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
# Anthropic Models
# ===============================================================================
def get_anthropic_llm(
    model_name: str = "claude-sonnet-4-5-20250929",
    temperature: Optional[float] = 0.0,
) -> ChatAnthropic:
    """Get an instance of Anthropic Chat LLM with configurable parameters.

    Args:
        model_name (str): Anthropic model name (e.g., "claude-sonnet-4-5-20250929")
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatAnthropic: Configured LLM instance
    """
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
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


# ===============================================================================
# Local OLLAMA Models
# ===============================================================================
def get_ollama_llm(
    model_name="llama3.2:3b", base_url="http://localhost:11434", temperature=0.0
):
    """Get an instance of local OLLAMA LLM with proper tool calling support.

    CRITICAL: Uses ChatOllama from langchain_ollama (NOT ChatOpenAI) for proper tool calling.
    ChatOpenAI wrapper doesn't properly handle OLLAMA's tool calling format - it returns
    tool calls as JSON text instead of actual tool_calls that can be executed.

    IMPORTANT: Smaller models (1b) have very poor tool calling support and often fail to iterate
    in agentic loops. Use at least 3b models or specialized tool-use models for reliable tool calling:
    - Recommended: llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b
    - Specialized: llama3-groq-tool-use:8b (fine-tuned for tool calling)
    - NOT recommended: llama3.2:1b, qwen2.5-coder (unless using hhao/qwen2.5-coder-tools variant)

    Args:
        model_name (str): The OLLAMA model name (e.g., "llama3.2:3b", "llama3.1:8b", "mistral:7b")
                         Note: Use models with tool calling support (llama3.1+, mistral, etc.)
        base_url (str): The base URL for the local OLLAMA server (default: http://localhost:11434)
                       Note: Do NOT include /v1 endpoint for ChatOllama
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatOllama: Configured OLLAMA LLM instance with proper tool calling support
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )


# ===============================================================================
# xAI Models
# ===============================================================================
def get_xai_llm(model_name: str = "", temperature: Optional[float] = 0.0) -> ChatXAI:
    """Get an instance of xAI Chat LLM with configurable parameters.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        model_name (str): xAI model name (e.g., "grok-4")
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatXAI: Configured LLM instance with async support
    """
    return ChatXAI(
        model=model_name,
        temperature=temperature,
        xai_api_key=os.getenv("XAI_API_KEY"),
    )


# ===============================================================================
# Mistral AI Models
# ===============================================================================
def get_mistral_llm(
    model_name: str = "mistral-small-latest",
    temperature: Optional[float] = 0.0,
) -> ChatMistralAI:
    """Get an instance of Mistral AI Chat LLM with configurable parameters.

    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    Args:
        model_name (str): Mistral model name (e.g., "mistral-small-latest", "mistral-large-latest")
        temperature (float): Temperature setting for generation randomness

    Returns:
        ChatMistralAI: Configured LLM instance with async support
    """
    return ChatMistralAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


# ===============================================================================
# GitHub Models (OpenAI-compatible API)
# ===============================================================================
def get_github_llm(
    model_name: str = "openai/o3",
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Get an instance of GitHub Models Chat LLM with configurable parameters.

    GitHub Models provides access to various AI models through an OpenAI-compatible API.
    The returned model instance supports both sync (invoke) and async (ainvoke)
    operations for flexibility in different execution contexts.

    IMPORTANT: Some GitHub models (e.g., openai/o3) only support the default temperature (1)
    and will error if you try to set it to other values. Leave temperature as None to use
    the model's default, or set it only if you know the model supports custom values.

    Args:
        model_name (str): GitHub model name (e.g., "openai/o3", "openai/gpt-4o", "meta-llama/Llama-3.3-70B-Instruct")
        temperature (Optional[float]): Temperature setting for generation randomness.
                                       If None (default), uses the model's default temperature.
                                       Some models only support default values.

    Returns:
        ChatOpenAI: Configured LLM instance with async support
    """
    kwargs = {
        "model": model_name,
        "api_key": os.getenv("GITHUB_TOKEN"),
        "base_url": "https://models.github.ai/inference",
    }

    # Only set temperature if explicitly provided
    if temperature is not None:
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)


if __name__ == "__main__":
    # ######################################################
    # llm = get_azure_openai_chat_llm(
    #     deployment_name="gpt-4o__test1",
    #     model_name="gpt-4o",
    #     openai_api_version="2024-05-01-preview",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(
    #     f"Model Type: {type(llm).__name__}, Deployment Name: {llm.deployment_name}, Model Name: {llm.model_name}"
    # )
    # print(f"Response: {response.content}")

    # ######################################################
    # llm = get_azure_openai_chat_llm(
    #     deployment_name="gpt-4o-mini-mimi2",
    #     model_name="gpt-4o-mini",
    #     openai_api_version="2024-05-01-preview",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(
    #     f"Model Type: {type(llm).__name__}, Deployment Name: {llm.deployment_name}, Model Name: {llm.model_name}"
    # )
    # print(f"Response: {response.content}")

    # ######################################################
    # llm = get_azure_openai_chat_llm(
    #     deployment_name="gpt-4.1___test1",
    #     model_name="gpt-4.1",
    #     openai_api_version="2024-05-01-preview",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(
    #     f"Model Type: {type(llm).__name__}, Deployment Name: {llm.deployment_name}, Model Name: {llm.model_name}"
    # )
    # print(f"Response: {response.content}")

    # #####################################################
    # llm = get_azure_openai_chat_llm(
    #     deployment_name="gpt-5-nano_mimi_test",
    #     model_name="gpt-5-nano",
    #     openai_api_version="2024-12-01-preview",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(
    #     f"Model Type: {type(llm).__name__}, Deployment Name: {llm.deployment_name}, Model Name: {llm.model_name}"
    # )
    # print(f"Response: {response.content}")

    # ######################################################
    # llm = get_azure_openai_chat_llm(
    #     deployment_name="gpt-5.2-chat-mimi-test",
    #     model_name="gpt-5.2-chat",
    #     openai_api_version="2024-12-01-preview",
    # )
    # response = llm.invoke("Hi")
    # print(
    #     f"Model Type: {type(llm).__name__}, Deployment Name: {llm.deployment_name}, Model Name: {llm.model_name}"
    # )
    # print(f"Response: {response.content}")

    # #####################################################
    # llm = get_anthropic_llm(
    #     model_name="claude-sonnet-4-5-20250929",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(f"Model Type: {type(llm).__name__}, Model Name: {llm.model}")
    # print(f"Response: {response.content}")

    # ####################################################
    # llm = get_ollama_llm(
    #     model_name="deepseek-r1-coder-tools:14b",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(f"Model Type: {type(llm).__name__}, Model Name: {llm.model}")
    # print(f"Response: {response.content}")

    # #####################################################
    # llm = get_xai_llm(
    #     model_name="grok-4-1-fast-reasoning-latest",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(f"Model Type: {type(llm).__name__}, Model Name: {llm.model}")
    # print(f"Response: {response.content}")

    # ######################################################
    # llm = get_mistral_llm(
    #     model_name="devstral-latest",
    #     temperature=0.0,
    # )
    # response = llm.invoke("Hi")
    # print(f"Model Type: {type(llm).__name__}, Model Name: {llm.model}")
    # print(f"Response: {response.content}")

    ######################################################
    llm = get_github_llm(
        model_name="openai/gpt-4o-mini",
        # temperature not set - openai/o3 only supports default temperature (1)
    )
    response = llm.invoke("Hi")
    print(f"Model Type: {type(llm).__name__}, Model Name: {llm.model_name}")
    print(f"Response: {response.content}")
