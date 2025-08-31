import asyncio

from my_agent import create_graph
from checkpointer.postgres_checkpointer import get_postgres_checkpointer


async def test_state():
    checkpointer = await get_postgres_checkpointer()
    graph = create_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test_thread_123"}}

    # Check what state exists
    try:
        existing_state = await graph.aget_state(config)
        print(f"State exists: {existing_state is not None}")

        if existing_state:
            print(f"Config: {existing_state.config}")
            print(f"Values exist: {existing_state.values is not None}")

            if existing_state.values:
                print(f"Keys in values: {list(existing_state.values.keys())}")
                messages = existing_state.values.get("messages", [])
                print(f"Messages length: {len(messages)}")
                print(f"Messages: {messages}")

                # Check if the logic would consider this a continuing conversation
                is_continuing = (
                    existing_state
                    and existing_state.values
                    and existing_state.values.get("messages")
                    and len(existing_state.values.get("messages", [])) > 0
                )
                print(f"Would be considered continuing conversation: {is_continuing}")
        else:
            print("No existing state found")

    except Exception as e:
        print(f"Error getting state: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_state())
