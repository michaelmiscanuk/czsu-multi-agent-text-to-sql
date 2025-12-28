# Evaluation Core Concepts in AI Applications

This document outlines the fundamental concepts used in the evaluation of AI and Large Language Model (LLM) applications. It focuses on the practical usage of each concept within the development and deployment lifecycle, particularly in the context of frameworks like LangSmith.

## 1. Evaluation

**Definition:**
Evaluation is the systematic process of assessing the quality, performance, and reliability of an AI system's outputs. In the context of LLMs, which are non-deterministic, evaluation provides a structured way to measure "goodness" against defined criteria.

**Usage:**
We use evaluation throughout the entire application lifecycle. During development, it helps us iterate on prompts and model selection. Before deployment, it serves as a gatekeeper to ensure quality standards are met. In production, it allows us to monitor the system's behavior on real-world data. It transforms subjective "feelings" about model performance into objective, actionable data.

## 2. Experiment

**Definition:**
An experiment is a specific execution of an evaluation run. It involves testing a particular version of an application (including its prompts, model parameters, and code) against a dataset.

**Usage:**
We use experiments to compare different configurations side-by-side. For example, we might run Experiment A with GPT-3.5 and Experiment B with GPT-4 using the same dataset. By analyzing the results of these experiments, we can empirically determine which configuration yields better performance, lower latency, or reduced cost. Experiments capture the inputs, outputs, and evaluator scores for every example in the dataset.

## 3. Golden Dataset

**Definition:**
A Golden Dataset (or Ground Truth Dataset) is a curated collection of examples used for evaluation. It typically consists of input data and the corresponding expected "correct" output (reference output).

**Usage:**
We use the Golden Dataset as the standard against which the AI system is measured. It represents the ideal behavior of the application. When running an offline evaluation, the system's outputs are compared to the reference outputs in the Golden Dataset to calculate accuracy and other metrics. Building and maintaining a high-quality Golden Dataset is crucial for reliable evaluation.

## 4. Evaluator

**Definition:**
An evaluator is a component or function that scores the system's outputs. It determines how well the actual output matches the expected outcome or adheres to specific quality criteria.

**Usage:**
We use evaluators to automate the scoring process. There are different types of evaluators:
*   **LLM-as-a-Judge:** Using another LLM to grade the response based on criteria like helpfulness or toxicity.
*   **Heuristic/Rule-based:** Checking for specific string matches, JSON validity, or regex patterns.
*   **Human Review:** Manual scoring by domain experts (often used to validate automated evaluators).
We configure evaluators to run during experiments to generate metrics.

## 5. Metric

**Definition:**
A metric is a quantifiable measure derived from the scores produced by evaluators. It aggregates individual results into high-level statistics.

**Usage:**
We use metrics to get a quick overview of system performance. Common metrics include accuracy, precision, recall, F1 score, latency (p50, p99), and token usage. Metrics allow us to set pass/fail criteria for deployment (e.g., "Accuracy must be above 90%"). They provide the numbers needed for dashboards and reports.

## 6. Offline Evaluation

**Definition:**
Offline evaluation is the process of testing the application against a dataset in a controlled development environment, before the changes are deployed to production.

**Usage:**
We use offline evaluation to validate changes safely. Before shipping a new prompt or code update, we run it against our Golden Dataset. This helps us catch bugs, identify regressions, and ensure that the new version performs at least as well as the previous one. It is the primary method for "Test Driven Development" in AI engineering.

## 7. Online Evaluation

**Definition:**
Online evaluation involves assessing the system's performance on live production data as users interact with it. Unlike offline evaluation, it often lacks reference outputs (ground truth).

**Usage:**
We use online evaluation to monitor the system in the wild. Since we don't always know the "correct" answer for real user queries, we use reference-free evaluators (like checking for toxicity, hallucination, or user sentiment). This helps us detect anomalies, identify drifting performance, and gather failing examples to add to our Golden Dataset for future offline testing.

## 8. Benchmarking

**Definition:**
Benchmarking is the practice of comparing the performance of different models, versions, or systems against a standard set of tests.

**Usage:**
We use benchmarking to make informed decisions about model selection. For instance, we might benchmark a smaller, cheaper model against a larger, expensive one to see if the performance trade-off is acceptable. It helps in ranking different approaches and identifying the "best in class" solution for a specific task.

## 9. Regression Tests and Baseline

**Definition:**
Regression testing ensures that recent changes have not negatively impacted existing functionality. The baseline is the reference point (e.g., the performance of the current production version) used for comparison.

**Usage:**
We use regression tests in our CI/CD pipelines. Every time code is pushed, we run an evaluation and compare the new metrics against the baseline metrics. If the new version performs worse (a regression), the deployment is halted. This prevents "fixing one thing but breaking another."

## 10. Backtesting

**Definition:**
Backtesting is the process of running a new version of the application against historical production data (traces) that occurred in the past.

**Usage:**
We use backtesting to see how a new prompt or logic change *would have* performed on yesterday's or last week's user queries. It validates improvements using real-world diversity and edge cases that might not yet be present in the curated Golden Dataset.

## 11. Sampling

**Definition:**
Sampling is the technique of selecting a representative subset of data (traces or examples) for evaluation rather than analyzing every single instance.

**Usage:**
We use sampling primarily in online evaluation to control costs and latency. Running a complex LLM-as-a-judge evaluator on 100% of production traffic can be prohibitively expensive. Instead, we might sample 5% of traffic to get a statistically significant view of performance without the full overhead.

## 12. Example

**Definition:**
An example is a single unit of data within a dataset, typically consisting of an input and, optionally, a reference output (label).

**Usage:**
We use examples as the building blocks of our datasets. An example represents a single test case or user scenario. When an experiment runs, the application processes each example to generate a result. We often create examples from interesting or failing production traces.

## 13. Task

**Definition:**
The task is the specific function, chain, or agent logic that is being evaluated. It is the "System Under Test."

**Usage:**
We define the task as the input-output function we want to measure. It could be a simple summarization chain, a complex RAG (Retrieval Augmented Generation) pipeline, or an autonomous agent. Defining the task clearly is essential to ensure that the inputs provided by the dataset match what the system expects.
