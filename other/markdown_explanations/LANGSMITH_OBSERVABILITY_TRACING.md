# LangSmith Observability & Tracing Concepts

> **Comprehensive Guide to Understanding LangSmith's Observability Platform**

This document provides a detailed explanation of LangSmith's observability and tracing architecture, complementing the visual diagram at `other/diagrams/langsmith_observability_tracing.drawio`.

---

## Table of Contents

1. [Overview](#overview)
2. [Organizational Hierarchy](#organizational-hierarchy)
3. [Tracing Hierarchy](#tracing-hierarchy)
4. [Run Data Structure](#run-data-structure)
5. [Feedback & Evaluation](#feedback--evaluation)
6. [Monitoring & Automation](#monitoring--automation)
7. [Datasets & Experiments](#datasets--experiments)
8. [Key Performance Metrics](#key-performance-metrics)
9. [Integrations](#integrations)
10. [Best Practices](#best-practices)

---

## Overview

LangSmith is LangChain's platform for **capturing, debugging, evaluating, and monitoring LLM application behavior**. It provides end-to-end visibility into how applications handle requests, which is critical for LLM applications due to their non-deterministic nature.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Tracing** | Captures every step from input to output |
| **Debugging** | Visualize execution steps and identify issues |
| **Evaluation** | Test performance offline and online |
| **Monitoring** | Dashboards, alerts, and real-time metrics |
| **Deployment** | Package, build, and deploy agents |

---

## Organizational Hierarchy

LangSmith uses a hierarchical structure to organize resources:

```
Organization
    └── Workspace(s)
            └── Project(s)
                    └── Trace(s)
                            └── Run(s)
```

### Organization

The **top-level container** in LangSmith.

- **User Management**: Add/remove users, assign roles
- **Billing & Settings**: Subscription, payment, usage limits
- **RBAC Configuration**: Define custom roles and permissions
- **SSO/SCIM Integration**: Enterprise identity management

### Workspace

**Logical separation for teams or business units** within an organization.

| Feature | Description |
|---------|-------------|
| Team Isolation | Separate resources between different teams |
| Role-Based Access | Manage permissions at workspace level |
| Shared Settings | API keys, configurations |
| Resource Tagging | Organize resources within workspace |

**Key Point**: A workspace can be associated with multiple projects, but each project belongs to only one workspace.

### Project

A **container for traces** related to a single application or service.

- **Prebuilt Dashboards**: Automatically generated monitoring views
- **Automation Rules**: Trigger actions based on trace data
- **Online Evaluators**: Attach evaluators that run on production traces
- **Data Retention Settings**: Configure trace lifecycle
- **Environment Separation**: Use different projects for dev/staging/prod

---

## Tracing Hierarchy

### Trace

A **trace** records the complete sequence of steps your application takes—from receiving an input, through intermediate processing, to producing a final output.

```
Trace = Collection of Runs (Spans)
```

| Attribute | Description |
|-----------|-------------|
| `trace_id` | Unique identifier (UUID) |
| Root Run | First run in the trace |
| Child Runs | Nested operations within the trace |

**OpenTelemetry Analogy**: A LangSmith trace is equivalent to an OpenTelemetry trace (collection of spans).

### Run (Span)

A **run** is a single unit of work or operation within your LLM application. If you're familiar with OpenTelemetry, a run is equivalent to a **span**.

Examples of runs:
- LLM model call
- Chain execution
- Tool invocation
- Document retrieval
- Embedding generation

### Run Types

LangSmith categorizes runs by type to enable specialized rendering and analysis:

| Run Type | Description | Example |
|----------|-------------|---------|
| `llm` | Language model calls | GPT-4, Claude, etc. |
| `chain` | Sequence of operations | LangChain chains |
| `tool` | Tool/function calls | Calculator, search |
| `retriever` | Document retrieval | Vector DB queries |
| `embedding` | Vector embedding generation | text-embedding-3-large |
| `parser` | Output parsing | JSON parsing |
| `prompt` | Prompt formatting | Template rendering |

### Thread

A **thread** represents a multi-turn conversation—a sequence of traces linked together.

- Each turn in a conversation = one trace
- All traces in a conversation share a common identifier
- Linked via metadata keys: `session_id`, `thread_id`, or `conversation_id`

```python
# Example: Linking traces to a thread
with langsmith.trace(
    name="chat_turn",
    metadata={"session_id": "user-123-conv-456"}
):
    # Your chat logic here
    pass
```

---

## Run Data Structure

Each run contains detailed information stored in a standardized format.

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier for the run |
| `name` | string | Name/label of the operation |
| `run_type` | string | Type: llm, chain, tool, etc. |
| `inputs` | object | Input data provided to the run |
| `outputs` | object | Output data generated by the run |
| `start_time` | datetime | When the run started |
| `end_time` | datetime | When the run completed |
| `status` | string | pending, success, or error |
| `error` | string | Error message (if failed) |

### Relationship Fields

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | UUID | Links run to its parent trace |
| `parent_run_id` | UUID | ID of the parent run (for nesting) |
| `child_run_ids` | UUID[] | List of child run IDs |
| `session_id` | UUID | Project/tracing session ID |
| `dotted_order` | string | Hierarchical ordering string |

### Token & Cost Fields

These fields are particularly important for **cost tracking** and **optimization**:

| Field | Type | Description |
|-------|------|-------------|
| `total_tokens` | integer | Total tokens processed |
| `prompt_tokens` | integer | Input tokens |
| `completion_tokens` | integer | Output tokens |
| `total_cost` | decimal | Total cost (USD) |
| `prompt_cost` | decimal | Input token cost |
| `completion_cost` | decimal | Output token cost |
| `first_token_time` | datetime | Time to first token (streaming) |

### Metadata & Tags

Used for **filtering, searching, and grouping** runs:

| Field | Type | Description |
|-------|------|-------------|
| `tags` | string[] | Array of string labels |
| `metadata` | object | Key-value pairs for custom data |
| `extra` | object | Additional information |
| `events` | object[] | Streaming events |
| `feedback_stats` | object | Aggregated feedback scores |

---

## Feedback & Evaluation

### Feedback

**Feedback** allows scoring individual runs based on specific criteria.

Each feedback entry contains:
- **Key**: The criteria being evaluated (e.g., "correctness", "helpfulness")
- **Score**: Numerical or categorical value
- **Comment**: Optional justification
- **Source**: Where the feedback originated

### Feedback Sources

| Source | Description |
|--------|-------------|
| **User Feedback (SDK)** | Collected via `create_feedback()` API |
| **Inline Annotation (UI)** | Manually added in trace view |
| **Annotation Queues** | Structured review workflow |
| **Offline Evaluators** | Generated during experiments |
| **Online Evaluators** | Real-time production evaluation |

### Offline Evaluation

Testing **before deployment** using curated datasets.

**Workflow:**
1. Create a **dataset** with examples (inputs + optional reference outputs)
2. Define **evaluators** (human, code rules, LLM-as-judge, pairwise)
3. Run an **experiment** (execute app on dataset)
4. **Analyze results** and compare experiments
5. Iterate and improve

**Use Cases:**
- Version comparison
- Regression testing
- Benchmarking
- Pre-deployment validation

### Online Evaluation

Monitoring **production quality** in real-time.

**Types of Online Evaluators:**

| Type | Description | Use Case |
|------|-------------|----------|
| **LLM-as-Judge** | Use an LLM to evaluate traces | Toxicity, hallucinations |
| **Custom Code** | Python code running in LangSmith | Format validation, statistics |

**Granularity:**
- **Run Level**: Evaluate individual runs
- **Thread Level**: Evaluate entire conversations

---

## Monitoring & Automation

### Dashboards

LangSmith provides both **prebuilt** and **custom** dashboards.

**Prebuilt Dashboard Sections:**

| Section | Metrics |
|---------|---------|
| **Traces** | Count, latency, error rates |
| **LLM Calls** | Call count, latency |
| **Cost & Tokens** | Total/per-trace tokens, costs |
| **Tools** | Run counts, error rates, latency by tool |
| **Run Types** | Stats for immediate children of root |
| **Feedback Scores** | Aggregate stats for top feedback types |

### Alerts

**Threshold-based alerting** on three core metrics:

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Errored Runs** | Runs with error status | Failure detection |
| **Feedback Score** | Average feedback score | Regression detection |
| **Latency** | Average execution time | Performance monitoring |

**Notification Options:**
- Email
- Webhooks (Slack, PagerDuty, custom)

### Automations

**Rule-based processing** of trace data at scale.

**Structure:** `Filter + Sampling Rate + Action`

**Available Actions:**

| Action | Description |
|--------|-------------|
| Add to Dataset | Capture inputs/outputs for evaluation |
| Add to Annotation Queue | Queue for human review |
| Trigger Webhook | Call external services |
| Extend Data Retention | Keep traces longer |

**Example Rules:**
- Send traces with negative feedback to annotation queue
- Add 10% of traces to dataset for continuous improvement
- Extend retention for all error traces

### Data Retention

| Aspect | Details |
|--------|---------|
| **SaaS Maximum** | 400 days |
| **Extended Retention** | Auto-upgraded for evaluated traces |
| **Datasets** | Persist indefinitely |
| **Deletion** | Via project deletion or support request |

---

## Datasets & Experiments

### Dataset

A **collection of examples** used for repeatable evaluation.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Dataset identifier |
| `name` | string | Dataset name |
| `inputs` | object | Input data for examples |
| `outputs` | object | Optional reference outputs |
| `metadata` | object | Additional information |
| `source_run_id` | UUID | If created from a trace |

**Sources:**
- Manually curated test cases
- Historical production traces
- Synthetic data generation

### Experiment

An **experiment** represents running a specific application version against a dataset.

**Contents:**
- Application outputs
- Evaluator scores
- Execution traces
- Per-example results

**Capabilities:**
- Side-by-side comparison
- Regression detection
- Version benchmarking
- Performance tracking

---

## Key Performance Metrics

LangSmith tracks essential metrics for monitoring application health:

| Metric | Description |
|--------|-------------|
| `latency_p50` | 50th percentile response time |
| `latency_p99` | 99th percentile response time |
| `first_token_p50` | 50th percentile time to first token |
| `first_token_p99` | 99th percentile time to first token |
| `error_rate` | Percentage of failed runs |
| `total_tokens` | Combined prompt + completion tokens |
| `total_cost` | Calculated cost from token usage |
| `feedback_stats` | Aggregated scores by feedback key |

---

## Integrations

LangSmith integrates with multiple frameworks and SDKs:

### Native Integrations

| Framework | Language | Auto-Tracing |
|-----------|----------|--------------|
| LangChain | Python, JS/TS | ✅ |
| LangGraph | Python, JS/TS | ✅ |

### SDK Wrappers

| SDK | Method |
|-----|--------|
| OpenAI | `wrap_openai()` / `wrapOpenAI()` |
| Anthropic | `wrap_anthropic()` |
| Vercel AI SDK | Native integration |

### Other Frameworks

| Framework | Integration Method |
|-----------|-------------------|
| CrewAI | OpenInference instrumentation |
| AutoGen | OpenInference instrumentation |
| Semantic Kernel | OpenTelemetry |
| Custom Code | `@traceable` decorator |

### OpenTelemetry Support

LangSmith can receive traces via OpenTelemetry using the `OtelSpanProcessor`.

---

## Best Practices

### Tracing

1. **Use meaningful names** for runs to aid debugging
2. **Add metadata** for filtering (environment, user ID, version)
3. **Use tags** for categorization and search
4. **Set up threads** for conversational applications
5. **Use appropriate run types** for specialized rendering

### Evaluation

1. **Start with human evaluation** to understand quality dimensions
2. **Build diverse datasets** with both positive and negative examples
3. **Use annotation queues** for structured human feedback
4. **Align LLM-as-judge evaluators** with human feedback
5. **Run regression tests** before deploying changes

### Monitoring

1. **Set up alerts** for error rates and latency spikes
2. **Create custom dashboards** for key metrics
3. **Use automations** to capture interesting traces
4. **Monitor token costs** to control expenses
5. **Review feedback trends** for quality insights

### Organization

1. **Separate projects** by environment (dev/staging/prod)
2. **Use workspaces** for team isolation
3. **Implement RBAC** for access control
4. **Tag resources** for organization

---

## Example Trace Flow (RAG Application)

The diagram includes a visual representation of a typical RAG application trace:

```
User Input → chain (root) → retriever → embedding → llm → Output
```

Each step in this flow is recorded as a **run** within the **trace**, capturing:
- Input/output data
- Timing information
- Token usage
- Any errors

This hierarchical structure makes it easy to:
- Debug where issues occur
- Identify performance bottlenecks
- Track costs per operation
- Evaluate component quality

---

## Related Resources

- [LangSmith Documentation](https://docs.langchain.com/langsmith/)
- [Observability Concepts](https://docs.langchain.com/langsmith/observability-concepts)
- [Tracing Quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
- [Evaluation Guide](https://docs.langchain.com/langsmith/evaluation)
- [Monitoring Dashboards](https://docs.langchain.com/langsmith/dashboards)

---

## Diagram Reference

The accompanying diagram (`langsmith_observability_tracing.drawio`) visualizes:

1. **Organizational Hierarchy** - Organization → Workspace → Project
2. **Tracing Hierarchy** - Trace → Run → Run Types, Thread
3. **Run Data Structure** - Core fields, relationships, tokens, metadata
4. **Feedback & Evaluation** - Sources, offline/online evaluation
5. **Monitoring & Automation** - Dashboards, alerts, automations
6. **Datasets & Experiments** - Evaluation workflow
7. **Example Trace Flow** - RAG application execution
8. **Key Metrics** - Performance indicators
9. **Integrations** - Supported frameworks

---

*Last Updated: December 2024*
*Based on LangSmith Documentation and Official LangChain Resources*
