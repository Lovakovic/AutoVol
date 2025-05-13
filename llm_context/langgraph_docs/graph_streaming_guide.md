---
search:
  boost: 3
tags:
  - streaming
  - real-time
  - overview
hide:
  - tags
---

# Streaming in LangGraph: The Definitive Guide

LangGraph provides powerful streaming capabilities that allow you to monitor graph execution in real-time. Streaming is essential for creating responsive AI applications, providing visibility into agent reasoning, and enhancing user experience with intermediate results.

## Table of Contents

1. [Introduction to Streaming](#introduction-to-streaming)
2. [Stream Modes Overview](#stream-modes-overview)
3. [Basic Implementation](#basic-implementation)
4. [Streaming State Updates](#streaming-state-updates)
   - [Values Mode](#values-mode)
   - [Updates Mode](#updates-mode)
5. [Streaming LLM Tokens](#streaming-llm-tokens)
   - [Messages Mode](#messages-mode)
   - [Filtering by LLM Invocation](#filtering-by-llm-invocation)
   - [Filtering by Node](#filtering-by-node)
6. [Custom Data Streaming](#custom-data-streaming)
   - [From Nodes](#from-nodes)
   - [From Tools](#from-tools)
   - [Working with Any LLM](#working-with-any-llm)
7. [Advanced Usage](#advanced-usage)
   - [Multiple Stream Modes](#multiple-stream-modes)
   - [Subgraph Streaming](#subgraph-streaming)
   - [Debug Mode](#debug-mode)
8. [Async Streaming](#async-streaming)
   - [Python 3.11+ Usage](#python-311-usage)
   - [Python < 3.11 Considerations](#python--311-considerations)
9. [Best Practices](#best-practices)

## Introduction to Streaming

LangGraph is built with first-class support for streaming, allowing applications to receive incremental updates as the graph executes. This is particularly important for LLM applications, where operations can take several seconds to complete.

The basic syntax for streaming in LangGraph is:

```python
# Synchronous streaming
for chunk in graph.stream(
    input_data,
    stream_mode="mode_name"
):
    # Process chunk

# Asynchronous streaming
async for chunk in graph.astream(
    input_data,
    stream_mode="mode_name"
):
    # Process chunk
```

## Stream Modes Overview

LangGraph supports five main streaming modes, each serving different purposes:

| Stream Mode | Description | Use Case |
|-------------|-------------|----------|
| `values` | Full state after each step | Complete state visualization |
| `updates` | Changes to state after each step | Bandwidth-efficient state updates |
| `messages` | LLM tokens and metadata | Chat UIs with typing indicators |
| `custom` | User-defined data from inside nodes | Progress reporting, specialized UIs |
| `debug` | Detailed execution information | Troubleshooting complex graphs |

## Basic Implementation

### Prerequisites

To use streaming in LangGraph, you need:

1. A compiled graph
2. Input data
3. Selection of appropriate stream mode

### Example Implementation

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

# Build and compile graph
graph = (
    StateGraph(State)
    .add_node("refine_topic", refine_topic)
    .add_node("generate_joke", generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# Stream execution
for chunk in graph.stream(
    {"topic": "ice cream"}, 
    stream_mode="updates"
):
    print(chunk)
```

## Streaming State Updates

LangGraph provides two modes for streaming state changes: `values` and `updates`.

### Values Mode

The `values` mode streams the complete state after each step, showing the cumulative state.

#### When to use Values Mode

- When you need to know the complete state of the graph at each step
- For visualizing the full state evolution over time
- When building analytics dashboards that show all state fields

#### Example

```python
# Stream the full state after each step
for chunk in graph.stream(
    {"topic": "ice cream"}, 
    stream_mode="values"
):
    print(chunk)
```

Output:
```
{'topic': 'ice cream'}
{'topic': 'ice cream and cats'}
{'topic': 'ice cream and cats', 'joke': 'This is a joke about ice cream and cats'}
```

### Updates Mode

The `updates` mode streams only the changes to the state after each node executes, including the node name that made the changes.

#### When to use Updates Mode

- When you only care about incremental changes to the state
- To reduce bandwidth and processing overhead
- For focusing on specific node outputs rather than the entire state

#### Example

```python
for chunk in graph.stream(
    {"topic": "ice cream"}, 
    stream_mode="updates"
):
    print(chunk)
```

Output:
```
{'refine_topic': {'topic': 'ice cream and cats'}}
{'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
```

## Streaming LLM Tokens

### Messages Mode

The `messages` mode enables streaming LLM outputs token by token, essential for chat interfaces. The output format is a tuple with `(message_chunk, metadata)` where metadata contains details about node execution.

#### When to use Messages Mode

- For chat interfaces where you want to show tokens as they're generated
- When you need to track which node is generating which part of the output
- For creating UI that highlights the currently active reasoning step

#### Example

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START

llm = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    topic: str
    joke: str

def generate_joke(state: State):
    llm_response = llm.invoke([
        {"role": "user", "content": f"Generate a joke about {state['topic']}"}
    ])
    return {"joke": llm_response.content}

graph = (
    StateGraph(State)
    .add_node("generate_joke", generate_joke)
    .add_edge(START, "generate_joke")
    .compile()
)

# Stream the LLM tokens as they're generated
for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
    # metadata contains information about which node is currently active
    # {'langgraph_step': 1, 'langgraph_node': 'generate_joke', ...}
```

Output (streamed token by token):
```
Why did the ice cream go to therapy? Because it had too many sprinkle-related issues!
```

### Filtering by LLM Invocation

You can associate tags with LLM invocations to filter streamed tokens:

```python
from langchain.chat_models import init_chat_model

joke_model = init_chat_model(model="openai:gpt-4o-mini", tags=["joke"])
poem_model = init_chat_model(model="openai:gpt-4o-mini", tags=["poem"])

class State(TypedDict):
    topic: str
    joke: str
    poem: str

async def dual_generation(state, config):
    joke_response = await joke_model.ainvoke(
        [{"role": "user", "content": f"Write a joke about {state['topic']}"}],
        config  # Required for Python < 3.11
    )
    poem_response = await poem_model.ainvoke(
        [{"role": "user", "content": f"Write a poem about {state['topic']}"}],
        config  # Required for Python < 3.11
    )
    return {
        "joke": joke_response.content, 
        "poem": poem_response.content
    }

graph = (
    StateGraph(State)
    .add_node("dual_generation", dual_generation)
    .add_edge(START, "dual_generation")
    .compile()
)

async for message_chunk, metadata in graph.astream(
    {"topic": "cats"},
    stream_mode="messages"
):
    # Only display joke tokens
    if message_chunk.content and metadata["tags"] == ["joke"]:
        print(message_chunk.content, end="", flush=True)
```

### Filtering by Node

You can filter tokens by the executing node:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    topic: str
    joke: str
    poem: str

def write_joke(state: State):
    topic = state["topic"]
    joke_response = model.invoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}]
    )
    return {"joke": joke_response.content}

def write_poem(state: State):
    topic = state["topic"]
    poem_response = model.invoke(
        [{"role": "user", "content": f"Write a short poem about {topic}"}]
    )
    return {"poem": poem_response.content}

graph = (
    StateGraph(State)
    .add_node("write_joke", write_joke)
    .add_node("write_poem", write_poem)
    # write both the joke and the poem concurrently
    .add_edge(START, "write_joke")
    .add_edge(START, "write_poem")
    .compile()
)

for message_chunk, metadata in graph.stream(
    {"topic": "cats"},
    stream_mode="messages"
):
    # Only display tokens from the "write_poem" node
    if message_chunk.content and metadata["langgraph_node"] == "write_poem":
        print(message_chunk.content, end="", flush=True)
```

## Custom Data Streaming

### From Nodes

You can emit custom data from inside nodes using `get_stream_writer()`:

```python
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def process_node(state: State):
    writer = get_stream_writer()
    
    # Emit custom data
    writer({"status": "Starting processing"})
    
    # Do some work
    # ...
    
    writer({"progress": 50, "message": "Halfway done"})
    
    # More work
    # ...
    
    writer({"status": "Processing complete"})
    return {"answer": "Completed"}

graph = (
    StateGraph(State)
    .add_node("process_node", process_node)
    .add_edge(START, "process_node")
    .compile()
)

for chunk in graph.stream(
    {"query": "example"},
    stream_mode="custom"
):
    print(chunk)
```

Output:
```
{'status': 'Starting processing'}
{'progress': 50, 'message': 'Halfway done'}
{'status': 'Processing complete'}
```

### From Tools

You can also emit custom data from inside tools:

```python
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

@tool
async def get_items(place: str) -> str:
    """Use this tool to list items one might find in a place."""
    writer = get_stream_writer()
    
    # This could be replaced with actual streaming logic
    items = ["books", "pencils", "pictures"]
    for chunk in items:
        writer({"custom_tool_data": chunk})
        # In a real application, you might add a delay here
    
    return ", ".join(items)

llm = ChatOpenAI(model_name="gpt-4o-mini")
tools = [get_items]
agent = create_react_agent(llm, tools=tools)

# Stream custom data from inside the tool
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "what items are in the office?"}]},
    stream_mode="custom"
):
    print(chunk)
```

Output:
```
{'custom_tool_data': 'books'}
{'custom_tool_data': 'pencils'}
{'custom_tool_data': 'pictures'}
```

### Working with Any LLM

The `custom` streaming mode works with any LLM API, not just those implementing LangChain interfaces:

```python
from langgraph.config import get_stream_writer

def call_custom_model(state):
    """Call an arbitrary model and stream its output."""
    writer = get_stream_writer()
    
    # Use your custom streaming client
    for token in your_custom_streaming_client(state["prompt"]):
        writer({"llm_token": token})
    
    return {"result": "completed"}
```

**Real-world Example with OpenAI Client:**

```python
import json
from openai import AsyncOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START
from typing import TypedDict

openai_client = AsyncOpenAI()

class State(TypedDict):
    query: str
    result: str

async def stream_tokens(model_name, messages):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )
    role = None
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.role is not None:
            role = delta.role
        if delta.content:
            yield {"role": role, "content": delta.content}

async def get_items(state: State):
    """List items found in a place."""
    writer = get_stream_writer()
    response = ""
    
    async for msg_chunk in stream_tokens(
        "gpt-4o-mini",
        [{"role": "user", "content": f"List items found in {state['query']}"}]
    ):
        response += msg_chunk["content"]
        writer(msg_chunk)
    
    return {"result": response}

graph = (
    StateGraph(State)
    .add_node("get_items", get_items)
    .add_edge(START, "get_items")
    .compile()
)

async for chunk in graph.astream(
    {"query": "kitchen"},
    stream_mode="custom"
):
    if "content" in chunk:
        print(chunk["content"], end="", flush=True)
```

## Advanced Usage

### Multiple Stream Modes

You can combine multiple streaming modes in one call:

```python
from typing import TypedDict
from langgraph.types import StreamWriter

class State(TypedDict):
    topic: str
    joke: str

def generate_joke(state: State, writer: StreamWriter):
    # Stream custom data during node execution
    writer({"custom_key": "Writing custom data while generating a joke"})
    # You can stream multiple items from a single node
    writer({"progress": "50%"})
    writer({"progress": "100%"})
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node("refine_topic", refine_topic)
    .add_node("generate_joke", generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)

for stream_mode, chunk in graph.stream(
    {"topic": "ice cream"}, 
    stream_mode=["updates", "custom"]
):
    print(f"Stream mode: {stream_mode}")
    print(chunk)
    print()
```

Output:
```
Stream mode: updates
{'refine_topic': {'topic': 'ice cream and cats'}}

Stream mode: custom
{'custom_key': 'Writing custom data while generating a joke'}

Stream mode: custom
{'progress': '50%'}

Stream mode: custom
{'progress': '100%'}

Stream mode: updates
{'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
```

### Subgraph Streaming

To include outputs from subgraphs in the streamed data:

```python
from langgraph.graph import START, StateGraph
from typing import TypedDict

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
subgraph_builder.add_node("subgraph_node_2", subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    subgraphs=True
):
    print(chunk)
```

### Debug Mode

The `debug` streaming mode provides the most detailed information about graph execution, including task creation, execution, and results.

#### When to use Debug Mode

- When debugging complex workflows
- For understanding exactly how your graph is executing
- When you need detailed timing information about each step

#### Example

```python
for chunk in graph.stream(
    {"topic": "ice cream"}, 
    stream_mode="debug"
):
    print(chunk)
```

Output:
```
{'type': 'task', 'timestamp': '2025-01-28T22:06:34.789803+00:00', 'step': 1, 'payload': {'id': 'eb305d74-3460-9510-d516-beed71a63414', 'name': 'refine_topic', 'input': {'topic': 'ice cream'}, 'triggers': ['start:refine_topic']}}
{'type': 'task_result', 'timestamp': '2025-01-28T22:06:34.790013+00:00', 'step': 1, 'payload': {'id': 'eb305d74-3460-9510-d516-beed71a63414', 'name': 'refine_topic', 'error': None, 'result': [('topic', 'ice cream and cats')], 'interrupts': []}}
{'type': 'task', 'timestamp': '2025-01-28T22:06:34.790165+00:00', 'step': 2, 'payload': {'id': '74355cb8-6284-25e0-579f-430493c1bdab', 'name': 'generate_joke', 'input': {'topic': 'ice cream and cats'}, 'triggers': ['refine_topic']}}
{'type': 'task_result', 'timestamp': '2025-01-28T22:06:34.790337+00:00', 'step': 2, 'payload': {'id': '74355cb8-6284-25e0-579f-430493c1bdab', 'name': 'generate_joke', 'error': None, 'result': [('joke', 'This is a joke about ice cream and cats')], 'interrupts': []}}
```

## Async Streaming

### Python 3.11+ Usage

For Python 3.11+, async streaming works seamlessly:

```python
async for chunk in graph.astream(
    {"topic": "ice cream"},
    stream_mode="updates"
):
    print(chunk)
```

### Python < 3.11 Considerations

For Python versions below 3.11, two key limitations apply:

#### 1. Explicit Config Passing

You must explicitly pass `config` to async LLM calls:

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

llm = init_chat_model(model="openai:gpt-4o-mini")

class State(TypedDict):
    topic: str
    joke: str

async def call_model(state, config):  # Accept config parameter
    topic = state["topic"]
    print("Generating joke...")
    joke_response = await llm.ainvoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}],
        config  # Pass config explicitly
    )
    return {"joke": joke_response.content}

graph = (
    StateGraph(State)
    .add_node("call_model", call_model)
    .add_edge(START, "call_model")
    .compile()
)

async for chunk, metadata in graph.astream(
    {"topic": "ice cream"},
    stream_mode="messages"
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

#### 2. Writer Parameter for Custom Streaming

You can't use `get_stream_writer()` in async code in Python < 3.11:

```python
from langgraph.types import StreamWriter

# Add writer parameter directly to function signature
async def generate_joke(state: State, writer: StreamWriter):
    writer({"custom_key": "Streaming custom data while generating a joke"})
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node("generate_joke", generate_joke)
    .add_edge(START, "generate_joke")
    .compile()
)

async for chunk in graph.astream(
    {"topic": "ice cream"},
    stream_mode="custom"
):
    print(chunk)
```

## Best Practices

1. **Choose the Right Stream Mode**:
   - Use `values` when you need to track the complete state
   - Use `updates` for bandwidth-efficient state monitoring
   - Use `messages` for chat interfaces with typing indicators
   - Use `custom` for specialized progress reporting
   - Use `debug` only when troubleshooting

2. **Filter Streams When Possible**:
   - Filter by node or LLM tags to reduce processing
   - Use multiple stream modes with appropriate handling

3. **Handle Async Correctly**:
   - Be aware of Python version limitations
   - Pass config explicitly in Python < 3.11
   - Use writer parameter instead of `get_stream_writer()` in async code for Python < 3.11

4. **Design UI for Progressive Updates**:
   - Show typing indicators during LLM generation
   - Display progress bars for tool execution
   - Highlight active parts of your interface based on the current node

5. **Process Streaming Results Efficiently**:
   - Avoid blocking the main thread when processing stream chunks
   - Buffer LLM tokens appropriately for smooth UI updates
   - Use server-sent events or WebSockets to push updates to client applications

6. **Consider the User Experience**:
   - Stream too much information and you may overwhelm users
   - Stream too little and they may think the application is frozen
   - Find the right balance for your specific use case

By leveraging LangGraph's streaming capabilities effectively, you can create more responsive, transparent, and user-friendly AI applications.
