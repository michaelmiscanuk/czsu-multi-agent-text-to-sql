# Async/Await Event Loop Architecture

## Overview
This diagram illustrates the architecture of a Python asyncio-based restaurant simulation that demonstrates async/await patterns, event loop management, and concurrent execution.

## Architecture Diagram

```mermaid
graph TB
    %% Entry Point
    Start([Program Start]) --> AsyncRun[asyncio.run]
    AsyncRun --> MainDemo[simple_restaurant_demo]
    
    %% Main Orchestration
    MainDemo --> InitStep[Initialize step_counter = 0]
    InitStep --> GatherTasks[asyncio.gather]
    
    %% Concurrent Customer Tasks
    GatherTasks --> Alice[serve_customer: Alice]
    GatherTasks --> Bob[serve_customer: Bob] 
    GatherTasks --> Carol[serve_customer: Carol]
    
    %% Customer Service Flow
    Alice --> AliceOrder[Order: Burger]
    Bob --> BobOrder[Order: Pizza]
    Carol --> CarolOrder[Order: Salad]
    
    AliceOrder --> AliceCook[cook_meal: Burger, 2s]
    BobOrder --> BobCook[cook_meal: Pizza, 3s]
    CarolOrder --> CarolCook[cook_meal: Salad, 1s]
    
    %% Cooking Process with Event Loop
    AliceCook --> AliceSleep[asyncio.sleep 2s]
    BobCook --> BobSleep[asyncio.sleep 3s]
    CarolCook --> CarolSleep[asyncio.sleep 1s]
    
    AliceSleep --> EventLoop{Event Loop}
    BobSleep --> EventLoop
    CarolSleep --> EventLoop
    
    EventLoop -->|Task Switching| AliceContinue[Alice: Meal Ready]
    EventLoop -->|Task Switching| BobContinue[Bob: Meal Ready]
    EventLoop -->|Task Switching| CarolContinue[Carol: Meal Ready]
    
    %% Completion
    AliceContinue --> AliceServe[Serve Alice]
    BobContinue --> BobServe[Serve Bob]
    CarolContinue --> CarolServe[Serve Carol]
    
    AliceServe --> Complete[All Tasks Complete]
    BobServe --> Complete
    CarolServe --> Complete
    
    Complete --> End([Program End])
    
    %% Utility Functions
    NextStep[next_step] --> StepCounter[(step_counter)]
    AliceCook -.-> NextStep
    BobCook -.-> NextStep
    CarolCook -.-> NextStep
    
    %% Styling
    classDef coroutine fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef eventloop fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef utility fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef async fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class Alice,Bob,Carol,AliceCook,BobCook,CarolCook coroutine
    class EventLoop eventloop
    class NextStep,StepCounter utility
    class AliceSleep,BobSleep,CarolSleep,AsyncRun async
```

## Key Components

### 1. **Entry Point**
- `asyncio.run()` - Creates and manages the event loop
- `simple_restaurant_demo()` - Main coroutine orchestrator

### 2. **Concurrency Management**
- `asyncio.gather()` - Runs multiple coroutines concurrently
- Event loop handles task switching and scheduling

### 3. **Customer Service Pipeline**
- `serve_customer()` - Handles individual customer requests
- `cook_meal()` - Simulates cooking with async delays
- `asyncio.sleep()` - Yields control to allow other tasks to run

### 4. **Utility Functions**
- `next_step()` - Synchronous function for step tracking
- `step_counter` - Global variable for execution tracking

## Event Loop Behavior

1. **Task Creation**: `asyncio.gather()` creates concurrent tasks
2. **Task Execution**: Each `serve_customer()` starts executing
3. **Yielding Control**: `asyncio.sleep()` pauses execution and yields to event loop
4. **Task Switching**: Event loop switches between ready tasks
5. **Resumption**: Tasks resume when their sleep/await completes
6. **Completion**: All tasks complete and results are gathered

## Learning Points

- **Concurrency vs Parallelism**: Tasks run concurrently but not in parallel
- **Event Loop**: Single-threaded cooperative multitasking
- **Await Points**: Where control can be yielded back to the event loop
- **Task Scheduling**: How the event loop manages multiple coroutines
