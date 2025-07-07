# Add the missing function back after setup_users_threads_runs_table 
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str):
    """Get queries and results from checkpoints for a thread with retry logic for prepared statement errors."""
    print__checkpointers_debug(f"279 - GET CHECKPOINT START: Getting queries and results from checkpoints for thread: {thread_id}")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints to find all queries_and_results
        checkpoint_tuples = []
        try:
            print__checkpointers_debug("280 - ALIST METHOD: Using official AsyncPostgresSaver.alist() method")
            
            # Get all checkpoints to capture complete queries and results
            async for checkpoint_tuple in checkpointer.alist(config, limit=200):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__checkpointers_debug(f"281 - ALIST ERROR: Error using alist(): {alist_error}")
            
            # Fallback: use aget_tuple() to get the latest checkpoint only
            try:
                print__checkpointers_debug("282 - FALLBACK METHOD: Trying fallback method using aget_tuple()")
                state_snapshot = await checkpointer.aget_tuple(config)
                if state_snapshot:
                    checkpoint_tuples = [state_snapshot]
                    print__checkpointers_debug("283 - FALLBACK SUCCESS: Using fallback method - got latest checkpoint only")
            except Exception as fallback_error:
                print__checkpointers_debug(f"284 - FALLBACK ERROR: Fallback method also failed: {fallback_error}")
                return []
        
        if not checkpoint_tuples:
            print__checkpointers_debug(f"285 - NO CHECKPOINTS: No checkpoints found for thread: {thread_id}")
            return []
        
        print__checkpointers_debug(f"286 - CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for thread")
        
        # Sort checkpoints by step number (chronological order)
        checkpoint_tuples.sort(key=lambda x: x.metadata.get("step", 0) if x.metadata else 0)
        
        # Extract queries_and_results from all checkpoints
        all_queries_and_results = []
        
        print__checkpointers_debug(f"287 - QUERIES EXTRACTION: Extracting queries_and_results from {len(checkpoint_tuples)} checkpoints")
        
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            metadata = checkpoint_tuple.metadata or {}
            step = metadata.get("step", 0)
            
            # Extract queries_and_results from metadata.writes.submit_final_answer.queries_and_results
            writes = metadata.get("writes", {})
            if isinstance(writes, dict) and "submit_final_answer" in writes:
                submit_data = writes["submit_final_answer"]
                if isinstance(submit_data, dict) and "queries_and_results" in submit_data:
                    queries_and_results = submit_data["queries_and_results"]
                    if queries_and_results:
                        # If it's a list, extend; if it's a single item, append
                        if isinstance(queries_and_results, list):
                            all_queries_and_results.extend(queries_and_results)
                            print__checkpointers_debug(f"288 - QUERIES FOUND: Step {step}: Found {len(queries_and_results)} queries and results")
                        else:
                            all_queries_and_results.append(queries_and_results)
                            print__checkpointers_debug(f"289 - QUERIES FOUND: Step {step}: Found 1 query and result")
        
        print__checkpointers_debug(f"290 - GET CHECKPOINT SUCCESS: Found {len(all_queries_and_results)} total queries and results for thread: {thread_id}")
        return all_queries_and_results
        
    except Exception as e:
        print__checkpointers_debug(f"291 - GET CHECKPOINT ERROR: Error getting queries and results from checkpoints: {e}")
        return []