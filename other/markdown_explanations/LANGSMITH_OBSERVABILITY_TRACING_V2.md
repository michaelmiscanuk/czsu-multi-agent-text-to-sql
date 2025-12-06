# LangSmith Observability & Tracing Overview

> **High-Level Guide for Diploma Work**

---

## What is LangSmith?

LangSmith is LangChain's **observability platform** for LLM applications. It captures, visualizes, and monitors every step your application takes—from input to output.

---

## Organization Structure

```
Organization → Workspace → Project
```

| Level | Purpose |
|-------|---------|
| **Organization** | Top-level container (users, billing) |
| **Workspace** | Team isolation, shared settings |
| **Project** | Container for all traces of one app/service |

---

## Core Tracing Concepts

### Trace

A **trace** records the complete journey of a single request through your application.

- Starts when input is received
- Ends when output is produced
- Contains one or more **runs**

### Run (Span)

A **run** is a single unit of work within a trace.

- Runs can be **nested** (parent-child hierarchy)
- Each run has a **type** that determines how it's rendered
- Equivalent to an OpenTelemetry **span**

### Thread

A **thread** groups multiple traces from a multi-turn conversation.

- Linked via `session_id` in metadata
- Each conversation turn = one trace
- All traces share the same session identifier

---

## Run Types

| Type | Description |
|------|-------------|
| `llm` | Language model calls (GPT-4, Claude, etc.) |
| `chain` | Sequence of operations |
| `tool` | Tool/function invocations |
| `retriever` | Document retrieval (vector DB queries) |
| `embedding` | Vector embedding generation |
| `parser` | Output parsing |

---

## Key Run Data

Each run captures:

| Category | Fields |
|----------|--------|
| **Core** | id, name, run_type, inputs, outputs, status, error |
| **Timing** | start_time, end_time |
| **Relationships** | trace_id, parent_run_id, session_id |
| **Tokens & Cost** | total_tokens, prompt_tokens, completion_tokens, total_cost |
| **Metadata** | tags[], metadata{} |

---

## Example: RAG Application Trace

```
User Input → chain (root) → retriever → embedding → llm → parser → Output
```

Each step is recorded as a **run** within the **trace**, capturing inputs, outputs, timing, and token usage.

---

## Monitoring

| Feature | Purpose |
|---------|---------|
| **Dashboards** | Trace counts, latency (p50/p99), error rates, costs |
| **Alerts** | Threshold-based notifications (email, webhook) |
| **Automations** | Rule-based actions on trace data |

---

## Integrations

| Framework | Method |
|-----------|--------|
| LangChain / LangGraph | Auto-tracing (built-in) |
| OpenAI | `wrap_openai()` wrapper |
| Custom Code | `@traceable` decorator |

---

## Key Takeaway

> **Trace = Collection of Runs (Spans)**
> 
> Each Run captures inputs, outputs, timing, tokens, and costs.
> Threads link multiple Traces for conversations.

---

*See diagram: `other/diagrams/langsmith_observability_tracing_v2.drawio`*
