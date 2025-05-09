# Agent Development with LangGraph

LangGraph provides both low-level primitives and high-level prebuilt components for building agent-based applications. This documentation focuses on the prebuilt, reusable components designed to help you construct agentic systems quickly and reliably—without the need to implement orchestration, memory, or human feedback handling from scratch.

## Key Features

LangGraph includes several capabilities essential for building robust, production-ready agentic systems:

- **Memory Integration**: Native support for short-term (session-based) and long-term (persistent across sessions) memory, enabling stateful behaviors in chatbots and assistants.
- **Human-in-the-loop Control**: Execution can pause indefinitely to await human feedback—unlike websocket-based solutions limited to real-time interaction. This enables asynchronous approval, correction, or intervention at any point in the workflow.
- **Streaming Support**: Real-time streaming of agent state, model tokens, tool outputs, or combined streams.
- **Deployment Tooling**: Includes infrastructure-free deployment tools.
- **LangGraph Platform** supports testing, debugging, and deployment.
- **Studio**: A visual IDE for inspecting and debugging workflows.
- Supports multiple **deployment options** for production.

## High-level Building Blocks

LangGraph comes with a set of prebuilt components that implement common agent behaviors and workflows. These abstractions are built on top of the LangGraph framework, offering a faster path to production while remaining flexible for advanced customization.

Using LangGraph for agent development allows you to focus on your application's logic and behavior, instead of building and maintaining the supporting infrastructure for state, memory, and human feedback.

## Package Ecosystem

The high-level components are organized into several packages, each with a specific focus:

| Package | Description | Installation |
|---------|-------------|-------------|
| langgraph-prebuilt (part of langgraph) | Prebuilt components | `pip install -U langgraph langchain` |
| langgraph-supervisor | Supervisor agents | `pip install -U langgraph-supervisor` |
| langgraph-swarm | Swarm multi-agent system | `pip install -U langgraph-swarm` |
| langchain-mcp-adapters | MCP servers for tool and resource integration | `pip install -U langchain-mcp-adapters` |
| langmem | Short-term and long-term memory | `pip install -U langmem` |
| agentevals | Evaluate agent performance | `pip install -U agentevals` |

## Agent Architectures

LangGraph supports various agent architectures and patterns:

### ReAct Agents

ReAct is a popular general-purpose agent architecture that integrates three core concepts:

- **Tool calling**: Allowing the LLM to select and use various tools as needed
- **Memory**: Enabling the agent to retain and use information from previous steps
- **Planning**: Empowering the LLM to create and follow multi-step plans to achieve goals

This architecture allows for more complex and flexible agent behaviors, enabling dynamic problem-solving with multiple steps. You can implement this with `create_react_agent()`.

### Router Agents

Router agents exhibit a relatively limited level of control because the LLM usually focuses on making a single decision and produces a specific output from a limited set of pre-defined options. Routers typically employ structured outputs with LLMs by providing a specific format or schema that the LLM should follow in its response.

### Multi-agent Systems

As you develop agentic systems, they might grow more complex over time, making them harder to manage and scale. You might run into problems such as:

- Agent has too many tools and makes poor decisions about which to call next
- Context grows too complex for a single agent to keep track of
- Need for multiple specialization areas (e.g., planner, researcher, math expert)

To tackle these issues, you can break your application into multiple smaller, independent agents and compose them into a multi-agent system. These agents can be as simple as a prompt and an LLM call, or as complex as a ReAct agent.

In LangGraph, multi-agent workflow can be defined in two ways:

1. **Explicit control flow (normal edges)**: Define the control flow explicitly via graph edges
2. **Dynamic control flow (Command)**: Allow LLMs to decide parts of your application control flow

## Memory Management

LangGraph provides full control over memory implementation:

- **State**: User-defined schema specifying the exact structure of memory to retain
- **Checkpointers**: Mechanism to store state at every step across different interactions

This flexible approach allows you to tailor the memory system to your specific agent architecture needs.

## Human-in-the-Loop

Human involvement can significantly enhance agent reliability, especially for sensitive tasks. LangGraph supports human-in-the-loop patterns where the workflow can pause, await human input, and then resume based on that input.

## Deployment and Infrastructure

LangGraph offers several deployment options:

- **LangGraph Server**: Supports a wide range of agentic application use cases
- **LangGraph Studio**: A specialized IDE for visualization, interaction, and debugging
- **LangGraph CLI**: Command-line interface for local interaction
- **Python/JS SDK**: Programmatic interaction with deployed applications
- **Cloud SaaS**: Connect to GitHub repositories and deploy to LangChain's cloud
- **Self-Hosted options**: For more control over deployment infrastructure

## Real-world Applications

LangGraph is already being used in production by several major companies:

- **Replit, Uber, LinkedIn, GitLab** and others use it as a low-level orchestration framework
- **Klarna** built a customer support bot for 85 million active users
- **Elastic** created a security AI assistant for threat detection

## Advantages Over Other Frameworks

Compared to other agent frameworks, LangGraph offers:

1. **Explicit Control Flow**: Allows representing agent logic as a graph with explicit transitions between nodes
2. **Low-level Primitives**: Provides highly controllable agentic framework without hidden prompts or obfuscated cognitive architecture
3. **Integration with LangChain**: Fully integrated into the LangChain ecosystem for access to integrations and LangSmith observability
4. **Production Readiness**: Designed for production use cases with support for streaming, memory, and human-in-the-loop

## Getting Started

To create a simple ReAct agent with LangGraph:

```python
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Create components
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
search = TavilySearchResults(max_results=2)
tools = [search]

# Create the agent
agent_executor = create_react_agent(model, tools, checkpointer=memory)
```

For more details and advanced us
