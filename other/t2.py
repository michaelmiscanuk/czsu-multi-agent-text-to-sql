import asyncio
import time

# Global counter to show execution order
step_counter = 0


def next_step():
    global step_counter
    step_counter += 1
    return step_counter


async def cook_meal(name, cook_time):
    """Simulate cooking a meal"""
    step = next_step()
    print(f"🍳 Step {step}: Started cooking {name} (takes {cook_time}s)")
    await asyncio.sleep(cook_time)  # This is where we yield control
    step = next_step()
    print(f"✅ Step {step}: {name} is ready!")
    return f"{name} meal"


async def serve_customer(customer_name, meal, cook_time):
    """Simulate serving one customer"""
    step = next_step()
    print(f"👋 Step {step}: Greeting {customer_name}")

    # Take the order (instant)
    step = next_step()
    print(f"📝 Step {step}: Taking order from {customer_name}: {meal}")

    # Send order to kitchen and wait (this is where we "await")
    meal_result = await cook_meal(meal, cook_time)

    # Serve the meal (instant)
    step = next_step()
    print(f"🍽️ Step {step}: Serving {meal_result} to {customer_name}")
    return f"{customer_name} served"


async def simple_restaurant_demo():
    """Simple demo showing event loop switching with numbered steps"""
    global step_counter
    step_counter = 0  # Reset counter

    print("🏪 Restaurant Demo: Watch the Step Numbers!")
    print("=" * 50)

    # Start serving 3 customers concurrently
    # Watch how the step numbers show the switching!
    results = await asyncio.gather(
        serve_customer("Alice", "Pasta", 2),
        serve_customer("Bob", "Burger", 1),
        serve_customer("Carol", "Salad", 0.5),
    )

    print("=" * 50)
    print("🎉 All customers served!")
    print(f"📊 Total steps: {step_counter}")
    return results


# Run the demo
asyncio.run(simple_restaurant_demo())
