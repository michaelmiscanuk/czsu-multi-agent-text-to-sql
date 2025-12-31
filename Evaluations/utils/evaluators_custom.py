"""Custom evaluators for LangSmith experiments."""

import json
import re
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


@retry_with_exponential_backoff(max_attempts=30, base_delay=1.0, max_delay=300.0)
async def helpfulness_evaluator(inputs: dict, outputs: dict, judge_llm: Any) -> dict:
    """LLM judge evaluator for helpfulness (reference-free).

    Evaluates how helpful the LLM output is in addressing the user's input question.
    This is a reference-free evaluator that only uses the input and output.

    Args:
        inputs: Input dictionary containing the user's question
        outputs: Model outputs containing 'messages' with agent responses
        judge_llm: LLM instance to use for judging

    Returns:
        dict: Evaluation result with 'score' (1-5) and 'reasoning'
    """
    if not outputs or "messages" not in outputs or not outputs["messages"]:
        return {"score": 1, "reasoning": "No output provided"}

    user_question = inputs.get("question", "")
    agent_response = outputs["messages"][-1].content

    instructions = (
        "You are evaluating the helpfulness of an AI assistant's response to a user's question. "
        "Rate the response on a scale of 1-5 based on the following rubric:\n\n"
        "**Helpfulness Rubric:**\n"
        "5 - Exceptionally Helpful: The response directly and comprehensively answers the question, "
        "provides clear and accurate information, is well-structured, and goes beyond basic expectations "
        "by offering additional relevant context or insights.\n\n"
        "4 - Very Helpful: The response directly answers the question with accurate information, "
        "is clear and well-organized, but may lack some additional context or depth that would make it exceptional.\n\n"
        "3 - Moderately Helpful: The response addresses the question and provides relevant information, "
        "but may be incomplete, lack clarity, or miss some important aspects of what was asked.\n\n"
        "2 - Slightly Helpful: The response partially addresses the question but is vague, incomplete, "
        "or contains information that is only tangentially related to what was asked.\n\n"
        "1 - Not Helpful: The response does not address the question, is irrelevant, provides incorrect information, "
        "or is incomprehensible.\n\n"
        "Respond with a JSON object containing:\n"
        "- 'score': An integer from 1 to 5\n"
        "- 'reasoning': A brief explanation (2-3 sentences) justifying your score\n\n"
        "Format your response as valid JSON only, with no additional text."
    )

    user_msg = f"USER QUESTION: {user_question}\n\n" f"AI RESPONSE: {agent_response}"

    response = await judge_llm.ainvoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg},
        ]
    )

    # Parse JSON response with markdown code block handling
    try:
        response_text = response.content.strip()

        # Remove markdown code blocks if present (```json ... ``` or ``` ... ```)
        if "```" in response_text:
            # Extract content between triple backticks
            match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
            )
            if match:
                response_text = match.group(1).strip()

        result = json.loads(response_text)
        score = result.get("score", 3)
        reasoning = result.get("reasoning", "Unable to parse evaluation")
        return {"score": score, "reasoning": reasoning}
    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback if JSON parsing fails
        return {"score": 3, "reasoning": f"JSON parse error: {str(e)[:100]}"}


@retry_with_exponential_backoff(max_attempts=30, base_delay=1.0, max_delay=300.0)
async def grounded_helpfulness_evaluator(
    inputs: dict, outputs: dict, reference_outputs: dict, judge_llm: Any
) -> dict:
    """LLM judge evaluator for grounded helpfulness (uses reference answer).

    Evaluates how helpful the LLM output is while ensuring it contains the correct answer.
    This combines aspects of correctness (grounded in reference answer) and helpfulness.

    Args:
        inputs: Input dictionary containing the user's question
        outputs: Model outputs containing 'messages' with agent responses
        reference_outputs: Expected outputs with 'answer' field (ground truth)
        judge_llm: LLM instance to use for judging

    Returns:
        dict: Evaluation result with 'score' (1-5) and 'reasoning'
    """
    if not outputs or "messages" not in outputs or not outputs["messages"]:
        return {"score": 1, "reasoning": "No output provided"}

    user_question = inputs.get("question", "")
    agent_response = outputs["messages"][-1].content
    expected_answer = reference_outputs.get("answer", "[NO EXPECTED ANSWER PROVIDED]")

    instructions = (
        "You are evaluating the helpfulness of an AI assistant's response to a user's question. "
        "You have access to the expected correct answer (ground truth). "
        "Rate the response on a scale of 1-5 based on the following rubric:\n\n"
        "**Grounded Helpfulness Rubric:**\n"
        "5 - Exceptionally Helpful: The response contains the correct answer (matching the expected answer), "
        "directly and comprehensively addresses the question, provides clear and accurate information, "
        "is well-structured, and goes beyond basic expectations by offering additional relevant context or insights.\n\n"
        "4 - Very Helpful: The response contains the correct answer (matching the expected answer), "
        "directly addresses the question with clear and well-organized information, but may lack some "
        "additional context or depth that would make it exceptional.\n\n"
        "3 - Moderately Helpful: The response contains the correct answer (matching the expected answer) "
        "and provides relevant information, but may be incomplete, lack clarity, or miss some important "
        "aspects of what was asked.\n\n"
        "2 - Slightly Helpful: The response contains the correct answer (matching the expected answer) "
        "but is vague, incomplete, poorly formatted, or the answer is difficult to identify within the response.\n\n"
        "1 - Not Helpful: The response does NOT contain the correct answer, provides an incorrect answer, "
        "is irrelevant, or is incomprehensible.\n\n"
        "CRITICAL: If the expected answer is not present in the AI response (allowing for reasonable rounding), "
        "the score MUST be 1, regardless of how well-written the response is.\n\n"
        "Respond with a JSON object containing:\n"
        "- 'score': An integer from 1 to 5\n"
        "- 'reasoning': A brief explanation (2-3 sentences) justifying your score\n\n"
        "Format your response as valid JSON only, with no additional text."
    )

    user_msg = (
        f"USER QUESTION: {user_question}\n\n"
        f"AI RESPONSE: {agent_response}\n\n"
        f"EXPECTED CORRECT ANSWER: {expected_answer}"
    )

    response = await judge_llm.ainvoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg},
        ]
    )

    # Parse JSON response with markdown code block handling
    try:
        response_text = response.content.strip()

        # Remove markdown code blocks if present (```json ... ``` or ``` ... ```)
        if "```" in response_text:
            # Extract content between triple backticks
            match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
            )
            if match:
                response_text = match.group(1).strip()

        result = json.loads(response_text)
        score = result.get("score", 1)
        reasoning = result.get("reasoning", "Unable to parse evaluation")
        return {"score": score, "reasoning": reasoning}
    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback if JSON parsing fails
        return {"score": 1, "reasoning": f"JSON parse error: {str(e)[:100]}"}
