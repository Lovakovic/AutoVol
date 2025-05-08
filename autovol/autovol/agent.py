# autovol/agent.py
import os
from typing import List, Optional, Dict, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .volatility_runner import detect_profile, run_volatility_tool_logic, volatility_runner_tool

# --- State Definition ---
class AppState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dump_path: str
    initial_context: str
    profile: Optional[str]
    investigation_log: List[Dict[str, str]]

# --- LLM and Tools ---
# Use the specified Claude 3.7 Sonnet model
llm = ChatAnthropic(
  model="claude-3-7-sonnet-latest",
  max_tokens=64000,
  thinking={"type": "enabled", "budget_tokens": 4000},
)

llm_with_tool = llm.bind_tools([volatility_runner_tool])

# Tool node logic remains the same
def tool_node_logic(state: AppState) -> dict:
    """Executes the volatility tool call using state information."""
    messages = state["messages"]
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": [ToolMessage(content="No tool call found in last AI message.", tool_call_id="error")]}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name != volatility_runner_tool.name:
         return {"messages": [ToolMessage(content=f"Error: Unknown tool '{tool_name}' called.", tool_call_id=tool_call["id"])]}

    dump_path = state.get("dump_path")
    profile = state.get("profile")

    if not dump_path or not profile:
        error_msg = "Error: dump_path or profile not found in agent state for tool execution."
        print(error_msg)
        return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

    tool_output = run_volatility_tool_logic(
        plugin_name=tool_args.get("plugin_name"),
        plugin_args=tool_args.get("plugin_args"),
        dump_path=dump_path,
        profile=profile
    )

    log_entry = {
        "reasoning": last_message.content if isinstance(last_message.content, str) else "Tool call action",
        "command": f"{profile}.{tool_args.get('plugin_name')} {' '.join(tool_args.get('plugin_args') or [])}",
        "output": tool_output
    }
    current_log = state.get("investigation_log", [])
    new_log = current_log + [log_entry]

    # Only return messages and updated log
    return {
        "messages": [ToolMessage(content=tool_output, tool_call_id=tool_call["id"])],
        "investigation_log": new_log
        }

tool_node = tool_node_logic

# --- Agent Nodes ---
def detect_profile_node(state: AppState) -> dict:
    """Detects the profile and updates the state. Does NOT add a system message."""
    print("--- Detecting Profile ---")
    dump_path = state["dump_path"]
    profile = detect_profile(dump_path)
    if not profile:
        error_msg = f"Failed to detect a suitable OS profile for {dump_path}. Cannot proceed."
        print(error_msg)
        # Add an error message to the main message flow for the agent to see
        # It's better to use AIMessage or HumanMessage if the agent should respond to it,
        # or just update the state and let the next node handle the None profile.
        # Let's just update the state and let the agent node decide how to respond.
        return {"profile": None} # Let agent node handle the failure

    print(f"Profile detected: {profile}")
    # --- REMOVED SystemMessage ADDITION ---
    # Just update the profile in the state
    return {"profile": profile}


def agent_node(state: AppState) -> dict:
    """Invokes the LLM to decide the next action or respond."""
    print("--- Calling LLM Agent ---")
    profile = state.get("profile")
    if not profile:
        # Handle the case where profile detection failed in the previous step
        return {"messages": [AIMessage(content="Analysis cannot proceed: Failed to detect a suitable OS profile for the memory dump.")]}

    # Prepare the messages for the LLM
    current_messages = state["messages"]

    # --- CONSTRUCT THE SINGLE SYSTEM PROMPT ---
    # Use f-string to include dynamic info directly
    system_prompt_content = f"""You are a Digital Forensics expert specializing in RAM analysis using Volatility 3.
Your goal is to analyze the provided memory dump based on the user's initial context.

## Analysis Context
Memory Dump Path: {state['dump_path']}
Detected OS Profile Base: {profile}
Use this OS profile base as the prefix for Volatility 3 plugins (e.g., '{profile}.pslist.PsList', '{profile}.netscan.NetScan').
Initial User Context/Suspicion: {state['initial_context']}

## Your Task
1.  Review the user's context and the investigation log so far.
2.  Reason step-by-step about the next logical Volatility 3 plugin to run to advance the investigation.
3.  Call the 'volatility_runner' tool with the correct 'plugin_name' (including the profile prefix) and any necessary 'plugin_args'.
4.  Analyze the tool's output in the subsequent steps to inform your next decision.
5.  If you believe the analysis based on the initial context is complete, provide a concise summary of your findings.

## Tool Available
-   **volatility_runner**: Executes a Volatility 3 plugin.
    -   `plugin_name`: (string, required) Full plugin name including profile prefix (e.g., 'windows.pslist.PsList').
    -   `plugin_args`: (list[string], optional) Arguments for the plugin (e.g., ['--pid', '1234']).

## Investigation Log So Far (Last 5 entries)
{state.get('investigation_log', [])[-5:]}
"""
    system_prompt = SystemMessage(content=system_prompt_content)

    # --- Construct the final message list with the system prompt first ---
    messages_for_llm = [system_prompt] + list(current_messages)

    response = llm_with_tool.invoke(messages_for_llm)

    print(f"LLM Response: {response.content}")
    if response.tool_calls:
        print(f"LLM Tool Calls: {response.tool_calls}")

    # add_messages in the AppState handles appending the response automatically
    # We just need to return the AIMessage itself within the 'messages' key if add_messages wasn't used.
    # Since add_messages *is* used, we just return the dictionary containing the message.
    return {"messages": [response]}


# --- Graph Definition ---
# Graph definition remains the same - it correctly routes based on state updates
graph_builder = StateGraph(AppState)

graph_builder.add_node("detect_profile", detect_profile_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("volatility_tool_node", tool_node)

graph_builder.add_edge(START, "detect_profile")

def post_profile_detection(state: AppState):
    if state.get("profile"):
        return "agent"
    else:
        # Profile detection failed, agent node will handle it
        return "agent" # Route to agent to report the failure

graph_builder.add_conditional_edges(
    "detect_profile",
    post_profile_detection,
    {"agent": "agent"} # Always go to agent after detection attempt
)

def should_continue(state: AppState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    else:
        # If profile is None (failure) or no tool call, end.
        # The agent node should have already produced a message indicating failure if profile is None.
        return "end"

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "volatility_tool_node", "end": END},
)

graph_builder.add_edge("volatility_tool_node", "agent")

app_graph = graph_builder.compile()
