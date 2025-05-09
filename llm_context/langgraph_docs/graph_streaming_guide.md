# LangGraph Streaming: A Comprehensive Guide

LangGraph provides powerful streaming capabilities that allow you to monitor graph execution in real-time. Streaming is essential for creating responsive AI applications, providing visibility into the agent's reasoning process, and enhancing user experience by showing intermediate steps.

This guide covers the five streaming modes available in LangGraph, with examples for each approach.

## Table of Contents

1. [Introduction to Streaming in LangGraph](#introduction-to-streaming-in-langgraph)
2. [Stream Mode: "values"](#stream-mode-values)
3. [Stream Mode: "updates"](#stream-mode-updates)
4. [Stream Mode: "messages"](#stream-mode-messages)
5. [Stream Mode: "custom"](#stream-mode-custom)
6. [Stream Mode: "debug"](#stream-mode-debug)
7. [Using Multiple Streaming Modes](#using-multiple-streaming-modes)
8. [Best Practices](#best-practices)

## Introduction to Streaming in LangGraph

LangGraph is built with first-class support for streaming, allowing applications to receive incremental updates as the graph executes. This is particularly important for LLM applications, where operations can take several seconds to complete. By streaming intermediate results, you can create more responsive user experiences.

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

Let's explore each streaming mode in detail.

## Stream Mode: "values"

The "values" streaming mode returns the complete state of the graph after each step. This gives you a holistic view of the entire state at each point in time.

### When to use

- When you need to know the complete state of the graph at each step
- For visualizing the full state evolution over time
- When building analytics dashboards that show all state fields

### Example

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node("refine_topic", refine_topic)
    .add_node("generate_joke", generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)

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

## Stream Mode: "updates"

The "updates" streaming mode returns only the changes to the state after each node executes. Each update includes the node name and the values returned by that node.

### When to use

- When you only care about incremental changes to the state
- To reduce bandwidth and processing overhead
- For focusing on specific node outputs rather than the entire state

### Example

```python
# Using the same graph as above
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

## Stream Mode: "messages"

The "messages" streaming mode streams LLM tokens and metadata for any LLM invocations inside nodes. This is particularly useful for chat applications, where seeing the model's response character by character improves user experience.

### When to use

- For chat interfaces where you want to show tokens as they're generated
- When you need to track which node is generating which part of the output
- For creating UI that highlights the currently active reasoning step

### Example

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
    .add_node("refine_topic", refine_topic)
    .add_node("generate_joke", generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
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
    # {'langgraph_step': 2, 'langgraph_node': 'generate_joke', ...}
```

Output (streamed token by token):
```
Why did the cat sit on the ice cream cone? Because it wanted to be a "purr-fect" scoop! 🍦🐱
```

## Stream Mode: "custom"

The "custom" streaming mode allows you to emit arbitrary data from inside nodes using a `StreamWriter`. This gives you complete control over what information is streamed out of your graph.

### When to use

- When you want to stream custom progress updates from within a node
- For long-running operations where you want to show intermediate progress
- When implementing specialized visualizations or UI updates

### Example

```python
from langgraph.types import StreamWriter

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

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="custom"
):
    print(chunk)
```

Output:
```
{'custom_key': 'Writing custom data while generating a joke'}
{'progress': '50%'}
{'progress': '100%'}
```

### Advanced Custom Streaming Example

Here's a more complex example showing custom streaming from a tool:

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

## Stream Mode: "debug"

The "debug" streaming mode provides the most detailed information about graph execution, including task creation, execution, and results. This is invaluable for debugging and understanding the internal workings of your graph.

### When to use

- When debugging complex workflows
- For understanding exactly how your graph is executing
- When you need detailed timing information about each step

### Example

```python
# Using the same graph as before
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

## Using Multiple Streaming Modes

LangGraph allows you to combine multiple streaming modes to get the benefits of each. When you specify multiple modes, the streamed outputs will be tuples with the format `(stream_mode, data)`.

### Example

```python
# Combine "updates" and "custom" modes
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
```# Building Custom Agents with LangGraph: A Comprehensive Guide

## Table of Contents
1. [Introduction to LangGraph and Custom Agents](#introduction)
2. [Creating Custom ReAct Agents](#creating-react-agents)
3. [Running Agents](#running-agents)
4. [Streaming Agent Outputs](#streaming)
5. [Working with Tools](#tools)
6. [Context Management](#context)
7. [Memory Management](#memory)
8. [Human-in-the-Loop Integration](#human-in-the-loop)
9. [Example: Building a Custom ReAct Agent](#example)
10. [Tips and Best Practices](#best-practices)

<a id="introduction"></a>
## 1. Introduction to LangGraph and Custom Agents

LangGraph is a framework for building controllable, robust agents using a graph-based approach. Unlike the more rigid prebuilt agents, custom agents in LangGraph give you fine-grained control over the agent's behavior, allowing you to implement specialized reasoning patterns, customize control flow, and integrate advanced features like memory and human oversight.

LangGraph offers several key advantages:

- **Explicit Control Flow**: Represent agent logic as a graph with clear transitions between steps
- **Low-level Primitives**: Highly controllable without hidden prompts or obfuscated cognitive architecture
- **Integration with LangChain**: Full access to LangChain's ecosystem of tools and integrations
- **Production Readiness**: Support for streaming, memory, and human-in-the-loop workflows

While LangGraph provides the prebuilt `create_react_agent()` function, building custom agents opens up more possibilities for specialized behaviors and complex workflows.

<a id="creating-react-agents"></a>
## 2. Creating Custom ReAct Agents

The ReAct (Reasoning and Acting) pattern combines reasoning capabilities with tool use, enabling agents to think through problems step by step. Building a custom ReAct agent gives you control over its reasoning process, tool integration, and state management.

### Basic Agent State Structure

The foundation of a custom ReAct agent is its state schema. At minimum, you need to track messages:

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer that handles appending messages correctly
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

### Defining Key Graph Nodes

A custom ReAct agent typically needs the following nodes:

1. **Model Node (call_model)**: Processes the current state and generates responses or tool calls
2. **Tool Node (handle_tools)**: Executes tool calls made by the LLM
3. **Decision Node (should_continue)**: Determines whether to call more tools or return the final answer

### Building the Graph

With LangGraph, you construct your agent by creating a graph with these nodes:

```python
from langgraph.graph import StateGraph, END

# Define the nodes
def call_model(state: AgentState):
    # LLM processing logic
    pass

def handle_tools(state: AgentState):
    # Tool execution logic
    pass

def should_continue(state: AgentState):
    # Decision logic to continue or finish
    pass

# Create graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", handle_tools)

# Define entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
agent = workflow.compile()
```

<a id="running-agents"></a>
## 3. Running Agents

Once your custom agent is built, you can run it using several methods, depending on your needs.

### Basic Invocation

For simple, synchronous execution:

```python
# Run the agent
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]})
```

### Configuration Options

You can provide runtime configuration to control agent behavior:

```python
# With configuration options
config = {"configurable": {"thread_id": "unique_conversation_id"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config
)
```

### Asynchronous Execution

For non-blocking operation:

```python
# Async execution
result = await agent.ainvoke({"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]})
```

### Setting Recursion Limits

To prevent infinite loops, you can configure recursion limits:

```python
# Set recursion limit to avoid infinite loops
agent_with_limit = agent.with_config({"recursion_limit": 10})
result = agent_with_limit.invoke({"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]})
```

<a id="streaming"></a>
## 4. Streaming Agent Outputs

Streaming is crucial for providing real-time feedback to users, especially in multi-step agent reasoning. LangGraph supports several streaming modes to provide visibility into the agent's progress.

### Streaming Modes

LangGraph offers multiple streaming options:

- **values**: Stream all values in the state after each step
- **updates**: Stream only node names and updates after each step
- **custom**: Stream custom data from inside nodes
- **messages**: Stream LLM tokens with associated metadata
- **debug**: Stream detailed debug information for each step

### Basic Streaming

To stream agent outputs:

```python
# Stream agent execution
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    stream_mode="updates"
):
    print(chunk)
```

### Asynchronous Streaming

For asynchronous streaming:

```python
# Async streaming
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    stream_mode="messages"
):
    print(chunk)
```

### Custom Tool Progress Updates

You can stream custom progress updates from tools using the `get_stream_writer` function:

```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up weather data for {city}...")
    # Perform actual weather lookup
    return f"It's sunny in {city}!"
```

### Multiple Streaming Modes

You can combine multiple streaming modes:

```python
async for stream_mode, chunk in agent.astream(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    stream_mode=["updates", "messages", "custom"]
):
    print(f"Mode: {stream_mode}, Chunk: {chunk}")
```

<a id="tools"></a>
## 5. Working with Tools

Tools are essential components that allow agents to interact with external systems and perform useful actions. LangGraph provides flexible ways to define and integrate tools.

### Defining Tools

You can define tools using several approaches:

**Basic Function Tools**:

```python
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"
```

**Decorated Tools**:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"
```

**Tools with Schema**:

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class WeatherInput(BaseModel):
    """Inputs for weather tool."""
    city: str = Field(description="The city to get weather for")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"
```

### Accessing Agent State in Tools

Tools can access agent state and configuration using additional parameters:

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def get_user_info(
    state: Annotated[AgentState, InjectedState],
    config: RunnableConfig
) -> str:
    """Look up user information."""
    user_id = config["configurable"].get("user_id", "unknown")
    # Use state and config to retrieve user info
    return f"User information for {user_id}: ..."
```

### Special Tool Behaviors

You can configure tools with special behaviors:

**Direct Return Tools**:

```python
@tool(return_direct=True)
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    # This will be returned directly to the user without further agent processing
    return str(eval(expression))
```

**Parallel Tool Execution**:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
# Disable parallel tool calls
model_sequential = model.bind_tools(tools, parallel_tool_calls=False)
```

<a id="context"></a>
## 6. Context Management

Context allows agents to access and manage information beyond the message history, such as user data, session information, or tool outputs.

### Types of Context

LangGraph supports several types of context:

1. **Runtime Configuration**: Immutable data passed at invocation time
2. **State**: Dynamic data that evolves during execution
3. **Long-term Memory**: Persistent data across multiple conversations

### Using Configuration Context

For data that doesn't change during execution:

```python
def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    # Access user information from config
    user_name = config["configurable"].get("user_name", "User")
    system_msg = f"You are a helpful assistant. The user's name is {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

# Use the prompt function
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt=prompt
)

# Provide context at runtime
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi there!"}]},
    config={"configurable": {"user_name": "John Smith"}}
)
```

### Using State Context

For data that changes during agent execution:

```python
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

class CustomState(AgentState):
    user_name: str
    conversation_topic: str = "general"

def prompt(state: CustomState) -> list[AnyMessage]:
    system_msg = f"You are a helpful assistant. User's name is {state['user_name']}. Current topic: {state['conversation_topic']}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    state_schema=CustomState,
    prompt=prompt
)

agent.invoke({
    "messages": [{"role": "user", "content": "Hi there!"}],
    "user_name": "John Smith",
    "conversation_topic": "weather"
})
```

### Dynamic Prompt Generation

You can create context-aware prompts that adapt to the agent's state:

```python
def prompt_generator(state: AgentState) -> list[AnyMessage]:
    # Analyze messages to detect user preferences
    messages = state["messages"]
    # Generate appropriate system message based on conversation
    if any("weather" in msg.content.lower() for msg in messages if hasattr(msg, "content")):
        system_msg = "You are a weather expert assistant. Provide detailed weather information."
    else:
        system_msg = "You are a helpful general assistant."
    return [{"role": "system", "content": system_msg}] + messages
```

<a id="memory"></a>
## 7. Memory Management

Memory enables agents to maintain context across multiple interactions, making conversations more coherent and natural.

### Short-term Memory (Message History)

Short-term memory refers to the conversation history within a single session:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Create checkpointer for short-term memory
checkpointer = InMemorySaver()

# Create agent with memory
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    checkpointer=checkpointer
)

# Run first query with unique thread ID
config = {"configurable": {"thread_id": "conversation_123"}}
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config
)

# Continue conversation with same thread ID
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "And what about New York?"}]},
    config=config  # Same thread_id preserves conversation context
)
```

### Managing Message History Size

For long conversations, you might need to trim the message history:

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def pre_model_hook(state):
    # Trim messages to avoid context window limits
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=4000
    )
    return {"llm_input_messages": trimmed_messages}

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    pre_model_hook=pre_model_hook,
    checkpointer=InMemorySaver()
)
```

### Message Summarization

For more sophisticated memory management, you can use summarization:

```python
from langmem.short_term import SummarizationNode

# Create summarization node
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=ChatOpenAI(model="gpt-3.5-turbo"),
    max_tokens=4000,
    max_summary_tokens=1000,
    output_messages_key="llm_input_messages"
)

# Create agent with summarization
class StateWithContext(AgentState):
    context: dict  # For summarization tracking

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    pre_model_hook=summarization_node,
    state_schema=StateWithContext,
    checkpointer=InMemorySaver()
)
```

### Long-term Memory Storage

For persistence across sessions:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# Short-term memory (conversation state)
checkpointer = InMemorySaver()

# Long-term memory (persistent data)
store = InMemoryStore()

# In production, you would use database-backed storage:
# from langgraph.checkpoint.postgres import PostgresSaver
# checkpointer = PostgresSaver(connection_string="postgresql://...")

# Create graph with both types of memory
graph = workflow.compile(checkpointer=checkpointer, store=store)
```

<a id="human-in-the-loop"></a>
## 8. Human-in-the-Loop Integration

Human-in-the-loop (HITL) functionality allows humans to review, edit, and approve agent actions, particularly important for sensitive operations.

### Basic Tool Review

For human review of tool calls:

```python
from langgraph.types import interrupt

def book_hotel(hotel_name: str, config):
    # Pause execution for human review
    human_decision = interrupt({
        "action_request": {
            "action": "book_hotel",
            "args": {"hotel_name": hotel_name}
        },
        "description": "Please review this hotel booking request"
    })
    
    # If approved, proceed with booking
    if human_decision.get("approved", False):
        # Perform actual booking
        return f"Successfully booked a stay at {hotel_name}."
    else:
        return "Hotel booking was not approved."
```

### Using the Human-in-the-Loop Wrapper

LangGraph provides a helper function for adding HITL to tools:

```python
from langgraph.prebuilt.interrupt import add_human_in_the_loop

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

# Create agent with HITL-wrapped tool
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[
        add_human_in_the_loop(book_hotel)  # Automatically adds interrupt
    ],
    checkpointer=InMemorySaver()
)
```

### Customizing HITL Configuration

You can customize the HITL experience:

```python
from langgraph.prebuilt.interrupt import HumanInterruptConfig

interrupt_config = HumanInterruptConfig(
    allow_accept=True,    # Allow approving without changes
    allow_edit=True,      # Allow editing the args
    allow_respond=True,   # Allow responding with message only
    description="Please review this sensitive operation"
)

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[
        add_human_in_the_loop(book_hotel, interrupt_config=interrupt_config)
    ],
    checkpointer=InMemorySaver()
)
```

### Handling HITL in a Web Application

To implement HITL in a web application:

1. When an interrupt occurs, save the agent state (already done via checkpointer)
2. Present the pending action to the user in your UI
3. When the user responds, resume execution with their input:

```python
# Example server-side code for resuming execution
from langgraph.types import Command

def resume_execution(thread_id, action_id, user_decision):
    # Format response based on user decision
    if user_decision["type"] == "approve":
        response = Command(resume={"approved": True})
    elif user_decision["type"] == "edit":
        response = Command(resume={"approved": True, "args": user_decision["edited_args"]})
    else:  # reject
        response = Command(resume={"approved": False, "reason": user_decision.get("reason", "")})
    
    # Resume agent execution with the human input
    return agent.invoke(
        {"interrupt_response": {action_id: response}},
        config={"configurable": {"thread_id": thread_id}}
    )
```

<a id="example"></a>
## 9. Example: Building a Custom ReAct Agent

Let's put everything together to build a custom ReAct agent with memory, streaming, and human-in-the-loop:

```python
import json
from typing import Annotated, List, Sequence, TypedDict, Literal, Union, Dict, Any
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import add_human_in_the_loop

# 1. Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Dict[str, Any]

# 2. Define tools
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"

@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight from origin to destination on the given date."""
    # Add human review for flight booking
    human_decision = interrupt({
        "action_request": {
            "action": "book_flight",
            "args": {"origin": origin, "destination": destination, "date": date}
        },
        "description": "Please review this flight booking request",
        "config": {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True
        }
    })
    
    if human_decision.get("approved", False):
        return f"Flight from {origin} to {destination} on {date} has been booked."
    else:
        return f"Flight booking was not approved. Reason: {human_decision.get('reason', 'No reason provided')}"

# 3. Define graph nodes
def call_model(state: AgentState):
    """Process messages with the LLM and get a response."""
    # Create system message
    system_message = SystemMessage(content="You are a helpful travel assistant. Use tools when appropriate.")
    
    # Get the messages
    messages = [system_message] + list(state["messages"])
    
    # Call the model
    model = ChatOpenAI(model="gpt-4o")
    response = model.bind_tools([get_weather, book_flight]).invoke(messages)
    
    # Return the updated messages
    return {"messages": [response]}

def handle_tools(state: AgentState):
    """Execute any tool calls from the latest assistant message."""
    # Get latest assistant message
    last_message = state["messages"][-1]
    
    # Skip if no tool calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}
    
    # Initialize tools
    tools = {
        "get_weather": get_weather,
        "book_flight": book_flight
    }
    
    # Execute tools and collect results
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_fn = tools.get(tool_name)
        
        if tool_fn:
            try:
                result = tool_fn(**arguments)
                tool_results.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    )
                )
            except Exception as e:
                tool_results.append(
                    ToolMessage(
                        content=f"Error: {str(e)}",
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    )
                )
    
    # Return tool results
    return {"messages": tool_results}

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if the agent should continue or end."""
    # Get the most recent message
    last_message = state["messages"][-1]
    
    # If it's an AI message with tool calls, continue
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    # Otherwise end
    return "end"

# 4. Build and compile graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", handle_tools)

# Add edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# Add memory management
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 5. Example usage with streaming
config = {"configurable": {"thread_id": "travel_planning_123"}}

# Function to print streaming results nicely
def print_streaming_output(stream):
    for chunk in stream:
        for node_name, update in chunk.items():
            print(f"\n=== Update from {node_name} ===")
            if "messages" in update:
                for message in update["messages"]:
                    if hasattr(message, "pretty_print"):
                        message.pretty_print()
                    else:
                        print(message)
            print("")

# Run the agent with streaming
user_query = "I need to check the weather in Miami and book a flight from New York to Miami on May 15th."
print_streaming_output(
    graph.stream(
        {"messages": [HumanMessage(content=user_query)], "context": {}},
        config=config,
        stream_mode="updates"
    )
)
```

<a id="best-practices"></a>
## 10. Tips and Best Practices

### Agent Design
- **Start Simple**: Begin with a basic agent and gradually add complexity
- **Modular Design**: Break complex agents into smaller, specialized components
- **Clear Node Responsibilities**: Each node should have a single, well-defined purpose
- **Error Handling**: Include robust error handling in all tool implementations

### Performance
- **Message History Management**: Implement message trimming or summarization for long conversations
- **Tool Timeouts**: Add timeouts to tools that might take too long to respond
- **Recursion Limits**: Always set recursion limits to prevent infinite loops

### Human-in-the-Loop
- **Selective Oversight**: Use HITL for sensitive operations like payments or bookings
- **Clear Descriptions**: Provide clear descriptions for human reviewers
- **Fallback Options**: Have fallback paths when HITL operations are rejected

### Deployment
- **Production-Ready Storage**: Use database-backed memory storage for production
- **Testing**: Thoroughly test agent behavior with various inputs
- **Monitoring**: Implement logging and monitoring to track agent performance

### Prompt Engineering
- **Clear Instructions**: Provide clear instructions in system prompts
- **Examples**: Include examples of desired reasoning patterns
- **Context Management**: Be mindful of context window limits

By following these guidelines and utilizing the power of LangGraph, you can build sophisticated, reliable agents that handle complex tasks while maintaining full control over their behavior.

## Best Practices

1. **Choose the right streaming mode for your use case:**
   - Use "values" when you need the complete state
   - Use "updates" for bandwidth efficiency
   - Use "messages" for chat interfaces
   - Use "custom" for specialized progress updates
   - Use "debug" for detailed troubleshooting

2. **Handle streaming asynchronously when possible:**
   - Use `astream()` instead of `stream()` for better performance
   - This is especially important for web applications

3. **Display appropriate feedback based on stream content:**
   - Show typing indicators during LLM generation
   - Display progress bars for tool execution
   - Highlight active parts of your interface based on the current node

4. **Combine streaming modes when needed:**
   - For complex UIs that need both state updates and token streaming
   - Make sure to handle the tuple format `(mode, data)` correctly

5. **Process streaming results efficiently:**
   - Avoid blocking the main thread when processing stream chunks
   - For web applications, use server-sent events or WebSockets to push updates to clients

6. **Consider the user experience:**
   - Stream too much information and you may overwhelm users
   - Stream too little and they may think the application is frozen
   - Find the right balance for your specific use case

By leveraging LangGraph's streaming capabilities, you can create more responsive, more transparent, and more user-friendly AI applications.
