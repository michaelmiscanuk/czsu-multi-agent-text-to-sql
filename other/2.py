@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_conversation_messages_from_checkpoints(
    checkpointer, thread_id: str, user_email: str = None
) -> List[Dict[str, Any]]:
    """
    Get conversation messages from checkpoints using official AsyncPostgresSaver methods.

    This function extracts messages from LangGraph checkpoints and limits checkpoint processing
    to avoid performance issues with large conversation histories.
    """
    print__checkpointers_debug(
        f"292 - GET CONVERSATION START: Retrieving conversation messages for thread: {thread_id}"
    )
    try:
        # Security check: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__checkpointers_debug(
                f"293 - SECURITY CHECK: Verifying thread ownership for user: {user_email}"
            )

            try:
                async with get_direct_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT COUNT(*) FROM users_threads_runs 
                            WHERE email = %s AND thread_id = %s
                        """,
                            (user_email, thread_id),
                        )
                        result = await cur.fetchone()
                        thread_entries_count = result[0] if result else 0

                    if thread_entries_count == 0:
                        print__checkpointers_debug(
                            f"294 - SECURITY DENIED: User {user_email} does not own thread {thread_id} - access denied"
                        )
                        return []

                    print__checkpointers_debug(
                        f"295 - SECURITY GRANTED: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted"
                    )
            except Exception as e:
                print__checkpointers_debug(
                    f"296 - SECURITY ERROR: Could not verify thread ownership: {e}"
                )
                return []

        config = {"configurable": {"thread_id": thread_id}}

        # Use alist() method with limit to avoid processing too many checkpoints
        checkpoint_tuples = []
        try:
            print__checkpointers_debug(
                "297 - ALIST METHOD: Using official AsyncPostgresSaver.alist() method"
            )

            # Increase limit to capture all checkpoints for complete conversation
            async for checkpoint_tuple in checkpointer.alist(config, limit=200):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__checkpointers_debug(
                f"298 - ALIST ERROR: Error using alist(): {alist_error}"
            )

            # Fallback: use aget_tuple() to get the latest checkpoint only
            if not checkpoint_tuples:
                print__checkpointers_debug(
                    "299 - FALLBACK METHOD: Trying fallback method using aget_tuple()"
                )
                try:
                    state_snapshot = await checkpointer.aget_tuple(config)
                    if state_snapshot:
                        checkpoint_tuples = [state_snapshot]
                        print__checkpointers_debug(
                            "300 - FALLBACK SUCCESS: Using fallback method - got latest checkpoint only"
                        )
                except Exception as fallback_error:
                    print__checkpointers_debug(
                        f"301 - FALLBACK ERROR: Fallback method also failed: {fallback_error}"
                    )
                    return []

        if not checkpoint_tuples:
            print__checkpointers_debug(
                f"302 - NO CHECKPOINTS: No checkpoints found for thread: {thread_id}"
            )
            return []

        print__checkpointers_debug(
            f"303 - CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for verified thread"
        )

        # Sort checkpoints by step number (chronological order)
        checkpoint_tuples.sort(
            key=lambda x: x.metadata.get("step", 0) if x.metadata else 0
        )

        # Extract prompts and answers
        prompts = []
        answers = []

        print__checkpointers_debug(
            f"304 - MESSAGE EXTRACTION: Extracting messages from {len(checkpoint_tuples)} checkpoints"
        )

        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            metadata = checkpoint_tuple.metadata or {}
            step = metadata.get("step", 0)

            # Extract user prompts from checkpoint.channel_values.__start__.prompt
            checkpoint = checkpoint_tuple.checkpoint or {}
            channel_values = checkpoint.get("channel_values", {})
            if isinstance(channel_values, dict) and "__start__" in channel_values:
                start_data = channel_values["__start__"]
                if isinstance(start_data, dict) and "prompt" in start_data:
                    prompt = start_data["prompt"]
                    if prompt and prompt.strip():
                        prompts.append(
                            {
                                "content": prompt.strip(),
                                "step": step,
                                "checkpoint_index": checkpoint_index,
                            }
                        )
                        print__checkpointers_debug(
                            f"305 - USER PROMPT FOUND: Step {step}: {prompt[:50]}..."
                        )

            # Extract AI answers from checkpoint.channel_values.submit_final_answer.final_answer
            if (
                isinstance(channel_values, dict)
                and "submit_final_answer" in channel_values
            ):
                submit_data = channel_values["submit_final_answer"]
                if isinstance(submit_data, dict) and "final_answer" in submit_data:
                    final_answer = submit_data["final_answer"]
                    if final_answer and final_answer.strip():
                        answers.append(
                            {
                                "content": final_answer.strip(),
                                "step": step,
                                "checkpoint_index": checkpoint_index,
                            }
                        )
                        print__checkpointers_debug(
                            f"306 - AI ANSWER FOUND: Step {step}: {final_answer[:50]}..."
                        )

        # Sort prompts and answers by step number
        prompts.sort(key=lambda x: x["step"])
        answers.sort(key=lambda x: x["step"])

        print__checkpointers_debug(
            f"307 - MESSAGE PAIRING: Found {len(prompts)} prompts and {len(answers)} answers"
        )

        # Create conversation messages by pairing prompts and answers
        conversation_messages = []
        message_counter = 0

        # Pair prompts with answers based on order
        for i in range(max(len(prompts), len(answers))):
            # Add user prompt if available
            if i < len(prompts):
                prompt = prompts[i]
                message_counter += 1
                user_message = {
                    "id": f"user_{message_counter}",
                    "content": prompt["content"],
                    "is_user": True,
                    "timestamp": datetime.fromtimestamp(
                        1700000000 + message_counter * 1000
                    ),
                    "checkpoint_order": prompt["checkpoint_index"],
                    "message_order": message_counter,
                    "step": prompt["step"],
                }
                conversation_messages.append(user_message)
                print__checkpointers_debug(
                    f"308 - ADDED USER MESSAGE: Step {prompt['step']}: {prompt['content'][:50]}..."
                )

            # Add AI response if available
            if i < len(answers):
                answer = answers[i]
                message_counter += 1
                ai_message = {
                    "id": f"ai_{message_counter}",
                    "content": answer["content"],
                    "is_user": False,
                    "timestamp": datetime.fromtimestamp(
                        1700000000 + message_counter * 1000
                    ),
                    "checkpoint_order": answer["checkpoint_index"],
                    "message_order": message_counter,
                    "step": answer["step"],
                }
                conversation_messages.append(ai_message)
                print__checkpointers_debug(
                    f"309 - ADDED AI MESSAGE: Step {answer['step']}: {answer['content'][:50]}..."
                )

        print__checkpointers_debug(
            f"310 - CONVERSATION SUCCESS: Created {len(conversation_messages)} conversation messages in proper order"
        )

        # Log first few messages for debugging
        for i, msg in enumerate(conversation_messages[:6]):
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print__checkpointers_debug(
                f"311 - MESSAGE {i+1}: {msg_type} (Step {msg['step']}): {msg['content'][:50]}..."
            )

        if len(conversation_messages) > 6:
            print__checkpointers_debug(
                f"312 - MESSAGE SUMMARY: ...and {len(conversation_messages) - 6} more messages"
            )

        return conversation_messages

    except Exception as e:
        print__checkpointers_debug(
            f"313 - CONVERSATION ERROR: Error retrieving messages from checkpoints: {str(e)}"
        )
        print__checkpointers_debug(
            f"314 - CONVERSATION TRACEBACK: Full traceback: {traceback.format_exc()}"
        )
        return []
