# LLM Models Used in CZSU Multi-Agent Text-to-SQL

This document provides an overview of all language model providers and specific models used throughout the application, including evaluations and agent operations.

## Models Overview

| Provider | Model Name | Deployment Name (Azure only) | API Version (Azure only) | Usage Context |
|----------|-----------|------------------------------|--------------------------|---------------|
| **Azure OpenAI** | gpt-4o | gpt-4o__test1 | 2024-05-01-preview | Judge model in evaluations, primary agent model |
| **Azure OpenAI** | gpt-4o-mini | gpt-4o-mini-mimi2 | 2024-05-01-preview | Alternative lightweight model |
| **Azure OpenAI** | gpt-4.1 | gpt-4.1___test1 | 2024-05-01-preview | Evaluation testing |
| **Azure OpenAI** | gpt-5-nano | gpt-5-nano_mimi_test | 2024-12-01-preview | Evaluation testing |
| **Azure OpenAI** | gpt-5.2-chat | gpt-5.2-chat-mimi-test | 2024-12-01-preview | Evaluation testing |
| **Azure OpenAI** | gpt-3.2-chat | - | 2024-05-01-preview | Evaluation testing |
| **Anthropic** | claude-sonnet-4-5-20250929 | - | - | Alternative premium model |
| **Google Gemini** | gemini-3-pro-preview | - | - | Alternative model provider |
| **Google Gemini** | gemini-2.0-flash-exp | - | - | Fast experimental model |
| **Google Gemini** | gemini-1.5-pro | - | - | Production stable model |
| **xAI (Grok)** | grok-4-1-fast-reasoning | - | - | Reasoning-optimized model |
| **xAI (Grok)** | grok-4-1-fast-non-reasoning | - | - | Fast response model |
| **xAI (Grok)** | grok-code-fast-1 | - | - | Code generation optimized |
| **Mistral AI** | open-mistral-nemo | - | - | Open source option |
| **Mistral AI** | mistral-large-2512 | - | - | Large scale model |
| **Mistral AI** | devstral-2512 | - | - | Development optimized |
| **Mistral AI** | codestral-2508 | - | - | Code generation specialized |
| **Mistral AI** | mistral-small-latest | - | - | Lightweight option |
| **OLLAMA (Local)** | llama3.2:1b | - | - | Local lightweight model |
| **OLLAMA (Local)** | llama3.2:3b | - | - | Local standard model |
| **OLLAMA (Local)** | llama3.1:8b | - | - | Local larger model |
| **OLLAMA (Local)** | qwen2.5-coder:30b | - | - | Local code specialist |
| **OLLAMA (Local)** | hhao/qwen2.5-coder-tools:14b | - | - | Tool calling optimized |
| **OLLAMA (Local)** | deepseek-r1-coder-tools:14b | - | - | DeepSeek code specialist |
| **OLLAMA (Local)** | mistral:7b | - | - | Local Mistral variant |

## Provider Configuration

### Azure OpenAI
- **Endpoint**: Configured via `AZURE_OPENAI_ENDPOINT` environment variable
- **API Key**: Configured via `AZURE_OPENAI_API_KEY` environment variable
- **API Version**: Primarily using `2024-05-01-preview` and `2024-12-01-preview`
- **Async Support**: ✅ Yes (invoke and ainvoke)

### Anthropic
- **API Key**: Configured via `ANTHROPIC_API_KEY` environment variable
- **Async Support**: ✅ Yes

### Google Gemini
- **API Key**: Configured via `GOOGLE_API_KEY` environment variable
- **Special Features**: Includes thought process (`include_thoughts=True`)
- **Async Support**: ✅ Yes

### xAI (Grok)
- **API Key**: Configured via `XAI_API_KEY` environment variable
- **Async Support**: ✅ Yes

### Mistral AI
- **API Key**: Configured via `MISTRAL_API_KEY` environment variable
- **Async Support**: ✅ Yes

### OLLAMA (Local)
- **Base URL**: `http://localhost:11434` (default)
- **Tool Calling**: Uses `ChatOllama` from `langchain_ollama` for proper tool calling support
- **Minimum Recommended Size**: 3B parameters for reliable tool calling
- **Async Support**: ✅ Yes

## Model Selection Criteria

### Judge Model (Evaluations)
- **Primary**: Azure OpenAI GPT-4o
- **Temperature**: 0.0 (deterministic)
- **Purpose**: Evaluating correctness of agent outputs against reference answers

### Query Generation Models
Multiple models tested in evaluations to compare performance:
- Premium models: Claude Sonnet 4.5, GPT-4o, Mistral Large
- Code-specialized: Codestral, DeepSeek, Qwen Coder variants
- Fast models: Grok variants, GPT-4o-mini
- Local models: Various OLLAMA models for offline/cost-effective operation

## Notes

- **Tool Calling Support**: Critical for agentic workflows. Models smaller than 3B parameters often have poor tool calling reliability.
- **Temperature Settings**: Generally set to 0.0 for deterministic, reproducible results in evaluations.
- **Streaming**: Available for all models but typically disabled for batch evaluations.
- **Local vs Cloud**: OLLAMA models provide cost-effective local alternatives, while cloud models offer better performance and reliability.
