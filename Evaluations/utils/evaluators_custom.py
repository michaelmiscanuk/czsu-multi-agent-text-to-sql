"""Custom evaluators for LangSmith experiments."""

from typing import Any

from Evaluations.utils.retry_utils import retry_with_exponential_backoff


@retry_with_exponential_backoff(max_attempts=30, base_delay=1.0, max_delay=300.0)
async def correctness_evaluator(
    outputs: dict, reference_outputs: dict, judge_llm: Any
) -> bool:
    """LLM judge evaluator for correctness.

    Args:
        outputs: Model outputs containing 'messages' with agent responses
        reference_outputs: Expected outputs with 'answer' field
        judge_llm: LLM instance to use for judging

    Returns:
        bool: True if answer is correct, False otherwise
    """
    if not outputs or "messages" not in outputs or not outputs["messages"]:
        return False

    actual_answer = outputs["messages"][-1].content
    expected_answer = reference_outputs.get("answer", "[NO EXPECTED ANSWER PROVIDED]")

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
