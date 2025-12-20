"""Subprocess runner for single model evaluation with isolated configuration."""

import sys
import os
import io

# Windows console encoding fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
from pathlib import Path
import importlib.util
import uuid
from langsmith import aevaluate
import logging

os.environ["DEBUG"] = "0"

# Configure logging for retry utilities
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Project root
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

sys.path.insert(0, str(BASE_DIR))

# Import retry utilities
from Evaluations.utils.retry_utils import retry_with_exponential_backoff

# Environment config
NODE_NAME = os.environ.get("EVAL_NODE_NAME", "generate_query_node")
MODEL_ID = os.environ.get("EVAL_MODEL_ID", "")
DATASET_NAME = os.environ.get(
    "EVAL_DATASET_NAME",
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen",
)
MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "2"))
JUDGE_MODEL_ID = os.environ.get("EVAL_JUDGE_MODEL_ID", "azureopenai_gpt-4o")

if not MODEL_ID:
    sys.exit(1)


# Load config modules directly (bypass package __init__.py)
def load_module_directly(module_name: str, file_path: Path):
    """Direct module load from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


model_configs_path = BASE_DIR / "my_agent" / "utils" / "model_configs_all.py"
model_configs_module = load_module_directly(
    "my_agent.utils.model_configs_all", model_configs_path
)
MODEL_CONFIGS_ALL = model_configs_module.MODEL_CONFIGS_ALL

node_models_config_path = BASE_DIR / "my_agent" / "utils" / "node_models_config.py"
node_models_config = load_module_directly(
    "my_agent.utils.node_models_config", node_models_config_path
)


def get_model_config_by_id(model_id: str) -> dict:
    """Get model config by ID."""
    for config in MODEL_CONFIGS_ALL:
        if config.get("id") == model_id:
            return config
    raise ValueError(f"Model ID '{model_id}' not found")


# Set model config before importing my_agent
model_config = get_model_config_by_id(MODEL_ID)

new_config = {
    "model_provider": model_config["model_provider"],
    "model_name": model_config["model_name"],
    "deployment_name": model_config.get("deployment_name", ""),
    "temperature": model_config.get("temperature", 0.0),
    "streaming": model_config.get("streaming", False),
    "openai_api_version": model_config.get("openai_api_version", "2024-05-01-preview"),
    "base_url": model_config.get("base_url", "http://localhost:11434"),
}
node_models_config.NODE_MODELS_CONFIG["nodes"][NODE_NAME] = new_config

# Import my_agent modules after config is set
from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState
from my_agent.utils.models import get_azure_openai_chat_llm


def generate_experiment_name(judge_id: str, node_name: str, model_id: str) -> str:
    """Generate experiment name."""
    return f"judge_{judge_id}__Node_{node_name}__Model_{model_id}"


# Judge model - lookup config from JUDGE_MODEL_ID
judge_config = get_model_config_by_id(JUDGE_MODEL_ID)
judge_llm = get_azure_openai_chat_llm(
    deployment_name=judge_config["deployment_name"],
    model_name=judge_config["model_name"],
    openai_api_version=judge_config.get("openai_api_version", "2024-05-01-preview"),
    temperature=judge_config.get("temperature", 0.0),
)
# Note: judge_llm already has .with_retry(stop_after_attempt=30) from get_azure_openai_chat_llm


@retry_with_exponential_backoff(max_attempts=30, base_delay=1.0, max_delay=300.0)
async def correctness(outputs: dict, reference_outputs: dict) -> bool:
    """LLM judge evaluator."""
    if not outputs or "messages" not in outputs or not outputs["messages"]:
        return False

    actual_answer = outputs["messages"][-1].content
    expected_answer = reference_outputs.get("answers", "[NO EXPECTED ANSWER PROVIDED]")

    instructions = (
        "Given an actual answer and an expected answer, determine whether"
        " the actual answer contains the information in the"
        " expected answer (). Respond with 'CORRECT' if the actual answer (can be rounded)"
        " does contain the expected information and 'INCORRECT'"
        " otherwise. Do not include anything else in your response."
    )
    user_msg = f"ACTUAL ANSWER: {actual_answer}\n\nEXPECTED ANSWER: {expected_answer}"

    response = await judge_llm.ainvoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg},
        ]
    )
    return response.content.upper() == "CORRECT"


def example_to_state(inputs: dict) -> dict:
    """Convert dataset to agent state."""
    return DataAnalysisState(
        prompt=inputs["question"], messages=[], result="", iteration=0
    )


@retry_with_exponential_backoff(max_attempts=30, base_delay=1.0, max_delay=300.0)
async def target_with_config(inputs: dict, graph):
    """Graph invocation wrapper with retry logic for rate limiting."""
    state = example_to_state(inputs)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return await graph.ainvoke(state, config=config)


async def run_evaluation():
    """Run evaluation silently (output captured by parent process)."""
    # Create graph
    graph = create_graph()

    # Generate experiment name
    experiment_name = generate_experiment_name(JUDGE_MODEL_ID, NODE_NAME, MODEL_ID)

    # Create target function
    async def target_fn(inputs: dict):
        return await target_with_config(inputs, graph)

    # Run evaluation
    experiment_results = await aevaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=[correctness],
        max_concurrency=MAX_CONCURRENCY,
        experiment_prefix=experiment_name,
    )

    return experiment_results


if __name__ == "__main__":
    try:
        results = asyncio.run(run_evaluation())
        print("SUCCESS")  # Parent script checks for this
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
