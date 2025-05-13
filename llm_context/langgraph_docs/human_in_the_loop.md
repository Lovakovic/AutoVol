---
search:
  boost: 2
tags:
  - human-in-the-loop
  - hil
  - overview
hide:
  - tags
---

# Human-in-the-Loop (HITL) in LangGraph

LangGraph provides robust **human-in-the-loop (HITL)** capabilities that enable human intervention at strategic points in your agent workflows. This is particularly valuable for applications where model output requires validation, correction, or additional context before proceeding.

## Table of Contents

1. [Key Capabilities](#key-capabilities)
2. [Implementation Fundamentals](#implementation-fundamentals)
3. [Common Design Patterns](#common-design-patterns)
   - [Approval Workflows](#approval-workflows)
   - [Content Editing](#content-editing)
   - [Tool Call Review](#tool-call-review)
   - [Input Validation](#input-validation)
4. [Advanced Usage](#advanced-usage)
   - [Resuming Multiple Interrupts](#resuming-multiple-interrupts)
   - [Subgraph Interactions](#subgraph-interactions)
5. [Best Practices](#best-practices)
6. [Common Pitfalls](#common-pitfalls)

## Key Capabilities

LangGraph's HITL implementation offers several advantages:

- **Persistent Execution State**: The graph's state is checkpointed after each step, allowing execution to pause indefinitely at defined points. This supports asynchronous human review without time constraints.

- **Flexible Integration Points**: Human intervention can be introduced at any point in the workflow, enabling targeted oversight for sensitive operations.

- **Resumable Workflows**: Execution can be resumed with human input, allowing for a seamless combination of automated and human-guided processes.

- **Rich Context Sharing**: Any JSON-serializable data can be passed to humans for review, providing comprehensive context for decision-making.

## Implementation Fundamentals

### Prerequisites

To implement HITL in LangGraph, you need:

1. A checkpointer to save graph state (required for interrupts to work)
2. The `interrupt` function and `Command` primitive
3. A unique thread ID for tracking conversation state

### Basic Implementation

```python
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    some_text: str

def human_node(state: State):
    # Pause execution and present data to human
    human_input = interrupt({
        "text_to_review": state["some_text"],
        "instructions": "Please review and edit this text."
    })
    
    # Resume with human input
    return {
        "some_text": human_input
    }

# Build and compile graph
graph_builder = StateGraph(State)
graph_builder.add_node("human_node", human_node)
# Add other nodes and edges...

# Required for HITL functionality
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Run graph with a unique thread ID
config = {"configurable": {"thread_id": "unique_thread_id"}}
result = graph.invoke({"some_text": "Original text"}, config=config)

# Access the interrupt information
print(result["__interrupt__"])
# Output: Interrupt(value={"text_to_review": "Original text", "instructions": "Please review..."}, ...)

# Resume execution with human input
graph.invoke(Command(resume="Edited text"), config=config)
```

## Common Design Patterns

LangGraph supports several HITL patterns to address different requirements:

### Approval Workflows

A common pattern is to pause execution for human approval before sensitive operations.

```python
from typing import Literal, TypedDict
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    action: str
    parameters: dict
    approved: bool

def approval_node(state: State) -> Command[Literal["proceed", "reject"]]:
    # Request human approval
    approval = interrupt({
        "action": state["action"],
        "parameters": state["parameters"],
        "message": "Please review this action before proceeding."
    })
    
    # Route based on human decision
    if approval:
        return Command(goto="proceed", update={"approved": True})
    else:
        return Command(goto="reject", update={"approved": False})
```

### Content Editing

When LLM-generated content needs review or enhancement:

```python
def content_editing_node(state: State):
    edited_content = interrupt({
        "original_content": state["generated_content"],
        "instructions": "Review and improve this content if needed."
    })
    
    return {
        "generated_content": edited_content,
        "was_edited": edited_content != state["generated_content"]
    }
```

### Tool Call Review

For reviewing and potentially modifying tool calls before execution:

```python
def tool_review_node(state: State) -> Command[Literal["execute_tool", "revise_plan"]]:
    review_result = interrupt({
        "tool_name": state["tool_call"]["name"],
        "tool_args": state["tool_call"]["args"],
        "context": "Review this API call before execution."
    })
    
    action, data = review_result
    
    if action == "approve":
        # Continue with unchanged tool call
        return Command(goto="execute_tool")
    elif action == "modify":
        # Update tool call with modified arguments
        return Command(goto="execute_tool", update={
            "tool_call": {"name": state["tool_call"]["name"], "args": data}
        })
    else:  # action == "reject"
        # Send back to planning
        return Command(goto="revise_plan", update={"feedback": data})
```

### Input Validation

For validating user input within the graph:

```python
def validated_input_node(state: State):
    prompt = "Please enter your email address:"
    
    while True:
        email = interrupt(prompt)
        
        # Basic email validation
        if "@" in email and "." in email:
            break
        else:
            prompt = f"'{email}' is not a valid email. Please try again:"
    
    return {"user_email": email}
```

## Advanced Usage

### Resuming Multiple Interrupts

If you have multiple interrupts in the task queue, you can resume them all at once:

```python
def resume_multiple_interrupts(graph, thread_id):
    # Get current state with all interrupts
    thread_config = {"configurable": {"thread_id": thread_id}}
    current_state = graph.get_state(thread_config)
    
    # Create a resume map for all interrupts
    resume_map = {
        interrupt.interrupt_id: f"Human input for {i}" 
        for i, interrupt in enumerate(current_state.interrupts)
    }
    
    # Resume all interrupts
    return graph.invoke(Command(resume=resume_map), config=thread_config)
```

### Subgraph Interactions

When using interrupts within subgraphs, understand the execution flow:

```python
def parent_node(state: State):
    # Any code here will re-execute when resuming from an interrupt in the subgraph
    print("This will run again after resume")
    
    # Call subgraph that contains an interrupt
    result = subgraph.invoke(state)
    
    return result
```

## Best Practices

1. **Place interrupts strategically**: Position interrupts at the beginning of nodes or in dedicated nodes to minimize re-executing code unnecessarily.

2. **Handle side effects carefully**: Place side effects (API calls, database operations) after interrupts when possible to avoid duplicating them on resume.

3. **Provide clear context**: Give humans sufficient information to make informed decisions by passing relevant data in the interrupt payload.

4. **Implement timeouts when needed**: For synchronous applications, consider implementing timeouts to handle cases where human input might be delayed.

5. **Use consistent thread IDs**: Ensure consistent thread IDs across invocations to maintain conversation state.

## Common Pitfalls

### Side Effects Before Interrupts

Code before an interrupt will run twice - once before the interrupt and again when resuming:

```python
def problematic_node(state: State):
    # WARNING: This API call will run twice!
    api_call()  # Runs before interrupt and again after resume
    
    value = interrupt("Input needed")
    return {"value": value}
```

**Better approach:**

```python
def better_node(state: State):
    value = interrupt("Input needed")
    
    # Only runs once after resuming
    api_call(value)
    return {"value": value}
```

### Multiple Interrupts in One Node

When using multiple interrupts in a single node, remember that matching is index-based:

```python
def multi_interrupt_node(state: State):
    # First interrupt
    name = interrupt("What's your name?")
    
    # Second interrupt
    age = interrupt("What's your age?")
    
    return {"name": name, "age": age}
```

When resuming, the first value provided goes to the first interrupt, the second to the second interrupt, and so on. Changing the order or number of interrupts between runs can lead to unexpected behavior.

### Changing Node Structure Dynamically

Avoid dynamically changing the structure of nodes containing interrupts:

```python
def dynamic_node(state: State):
    # DON'T DO THIS: conditional interrupts can cause issues
    if state.get("need_name"):
        name = interrupt("What's your name?")
    
    age = interrupt("What's your age?")
    
    return {"name": name if state.get("need_name") else state.get("name"), "age": age}
```

This can cause problems with interrupt index matching when resuming.
