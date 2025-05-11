import os
from typing import List, Optional, Dict, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt, interrupt  # Correct import

from .volatility_runner import detect_profile, run_volatility_tool_logic, volatility_runner_tool

DETECT_PROFILE_CALL_COUNT = 0

# --- State Definition (remains the same) ---
class AppState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  dump_path: str
  initial_context: str
  profile: Optional[str]
  investigation_log: List[Dict[str, str]]


# --- LLM and Tools (remains the same) ---
llm = ChatAnthropic(
  model="claude-3-7-sonnet-latest",
  max_tokens=64000,
  thinking={"type": "enabled", "budget_tokens": 4000},
)
llm_with_tool = llm.bind_tools([volatility_runner_tool])


# --- Utility for Reasoning Extraction (remains the same) ---
def _extract_reasoning_from_ai_message(ai_message: AIMessage) -> str:
  reasoning_parts = []
  if hasattr(ai_message, 'response_metadata') and 'claude-messages-thinking' in ai_message.response_metadata:
    thinking_log = ai_message.response_metadata['claude-messages-thinking']
    if isinstance(thinking_log, list) and thinking_log:
      thinking_content = "\n".join(
        [step.get('thinking', '') if isinstance(step, dict) else str(step) for step in thinking_log if
         (isinstance(step, dict) and step.get('thinking')) or isinstance(step, str)]
      )
      if thinking_content.strip():
        reasoning_parts.append("LLM Thinking Process (from metadata):\n" + thinking_content.strip())
    elif isinstance(thinking_log, str) and thinking_log.strip():
      reasoning_parts.append("LLM Thinking Process (from metadata):\n" + thinking_log.strip())

  thinking_from_content_blocks = []
  text_from_content_blocks = []
  if isinstance(ai_message.content, list):
    for block in ai_message.content:
      if isinstance(block, dict):
        if block.get("type") == "thinking" and block.get("thinking"):
          thinking_from_content_blocks.append(str(block["thinking"]))
        elif block.get("type") == "text" and block.get("text"):
          text_from_content_blocks.append(str(block["text"]))

  if thinking_from_content_blocks:
    reasoning_parts.append("LLM Thinking (from content):\n" + "\n".join(filter(None, thinking_from_content_blocks)))
  if text_from_content_blocks:
    reasoning_parts.append("LLM Reasoning/Action Text:\n" + "\n".join(filter(None, text_from_content_blocks)))

  if not reasoning_parts and isinstance(ai_message.content, str):
    reasoning_parts.append(ai_message.content)

  if not reasoning_parts:
    return "No detailed reasoning extracted."

  return "\n---\n".join(reasoning_parts)


# --- Tool Node (remains the same) ---
def volatility_tool_node(state: AppState) -> dict:
  messages = state["messages"]
  ai_message_with_tool_call = None
  for msg in reversed(messages):
    if isinstance(msg, AIMessage) and msg.tool_calls:
      ai_message_with_tool_call = msg
      break

  if not ai_message_with_tool_call:
    return {"messages": [
      ToolMessage(content="Error: No AIMessage with tool_calls found to execute.", tool_call_id="error_no_aim")]}

  tool_call = ai_message_with_tool_call.tool_calls[0]
  tool_name = tool_call["name"]
  tool_args = tool_call["args"]

  if tool_name != volatility_runner_tool.name:
    return {
      "messages": [ToolMessage(content=f"Error: Unknown tool '{tool_name}' called.", tool_call_id=tool_call["id"])]}

  dump_path = state.get("dump_path")
  profile = state.get("profile")

  if not dump_path or not profile:
    error_msg = "Error: dump_path or profile not found in agent state for tool execution."
    print(error_msg)
    return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

  print(f"--- Executing Volatility Tool: {profile}.{tool_args.get('plugin_name')} ---")
  tool_output_content = run_volatility_tool_logic(
    plugin_name=tool_args.get("plugin_name"),
    plugin_args=tool_args.get("plugin_args"),
    dump_path=dump_path,
    profile=profile
  )
  reasoning_for_log = _extract_reasoning_from_ai_message(ai_message_with_tool_call)
  log_entry = {
    "reasoning": reasoning_for_log,
    "command": f"{profile}.{tool_args.get('plugin_name')} {' '.join(tool_args.get('plugin_args') or [])}",
    "output": tool_output_content
  }
  current_log = state.get("investigation_log", [])
  new_log = current_log + [log_entry]
  return {
    "messages": [ToolMessage(content=tool_output_content, tool_call_id=tool_call["id"])],
    "investigation_log": new_log
  }


# --- Agent Nodes (detect_profile_node and agent_node remain the same) ---
def detect_profile_node(state: AppState) -> dict:
  global DETECT_PROFILE_CALL_COUNT
  DETECT_PROFILE_CALL_COUNT += 1

  print(f"--- Detecting Profile (Call #{DETECT_PROFILE_CALL_COUNT}) ---")
  dump_path = state["dump_path"]
  profile = detect_profile(dump_path)
  if not profile:
    error_msg = f"Failed to detect a suitable OS profile for {dump_path}. Cannot proceed."
    print(error_msg)
    return {"profile": None}
  print(f"Profile detected: {profile}")
  return {"profile": profile}


def agent_node(state: AppState) -> dict:
  print("--- Calling LLM Agent ---")
  profile = state.get("profile")
  # This check is important. If detect_profile is re-run and fails after a resume,
  # we need to handle it.
  if not profile and START not in state.get("next", []):  # Check if not initial run
    # If profile becomes None AFTER the initial detect_profile_node run, it's an issue.
    # This might happen if detect_profile is part of the resume path incorrectly.
    # For now, let's assume profile once set, stays, unless explicitly cleared.
    # The main protection is that the agent_node receives the entire state.
    pass  # Allow agent to proceed if profile was set previously.

  if not profile:  # Catches initial failure or if it was cleared
    return {"messages": [
      AIMessage(content="Analysis cannot proceed: Failed to detect a suitable OS profile for the memory dump.")]}

  current_messages = state["messages"]
  system_prompt_content = f"""You are a Digital Forensics expert specializing in RAM analysis using Volatility 3.
Your goal is to analyze the provided memory dump based on the user's initial context.

## Analysis Context
Memory Dump Path: {state['dump_path']}
Detected OS Profile Base: {profile}
Use this OS profile base as the prefix for Volatility 3 plugins (e.g., '{profile}.pslist.PsList', '{profile}.netscan.NetScan').
Initial User Context/Suspicion: {state['initial_context']}

## Your Task
1.  Review the user's context, any prior user feedback, and the investigation log so far.
2.  Reason step-by-step about the next logical Volatility 3 plugin to run to advance the investigation. Your reasoning should be thorough.
3.  Call the 'volatility_runner' tool with the correct 'plugin_name' (including the profile prefix) and any necessary 'plugin_args'.
4.  If a human reviewer denies your proposed command, take their feedback into account for your next proposal.
5.  Analyze the tool's output in the subsequent steps to inform your next decision.
6.  If you believe the analysis based on the initial context is complete, provide a concise summary of your findings. Do NOT call any tools if you are providing a final summary.

## Tool Available
-   **volatility_runner**: Executes a Volatility 3 plugin.
    -   `plugin_name`: (string, required) Full plugin name including profile prefix (e.g., 'windows.pslist.PsList').
    -   `plugin_args`: (list[string], optional) Arguments for the plugin (e.g., ['--pid', '1234']).

## Investigation Log So Far (Last 5 entries)
{state.get('investigation_log', [])[-5:]}
"""
  system_prompt = SystemMessage(content=system_prompt_content)
  messages_for_llm = [system_prompt] + list(current_messages)

  response = llm_with_tool.invoke(messages_for_llm)
  print(f"LLM Full Response Content: {response.content}")
  if response.tool_calls:
    print(f"LLM Tool Calls: {response.tool_calls}")
  return {"messages": [response]}


# MODIFIED human_tool_review_node
def human_tool_review_node(state: AppState) -> Dict[str, any]:  # Return type for clarity
  """
  Interrupts for human review. The user's decision (resume payload) will be
  the output of this node when the graph resumes.
  This node itself does NOT modify the state directly with messages or logs based on the decision.
  It prepares for the interrupt and then, upon resume, the *graph routing* and subsequent nodes
  will handle the decision.
  """
  print("--- Human Review Node ---")
  last_message = state["messages"][-1]  # The AIMessage with the tool call

  if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
    print("Error: human_tool_review_node reached without a tool call. Passing through.")
    # This is an unexpected state. We'll return a payload that routes back to agent.
    return {"action": "deny", "reason": "Internal error: Review node reached inappropriately."}

  tool_call_to_review = last_message.tool_calls[0]
  extracted_reasoning = _extract_reasoning_from_ai_message(last_message)

  interrupt_payload_for_main = {
    "reasoning_and_thinking": extracted_reasoning,
    "tool_call_args": tool_call_to_review["args"],
    "tool_call_id": tool_call_to_review["id"],
    "profile": state.get("profile")
  }

  print(f"Interrupting for human review. Payload for main.py: {interrupt_payload_for_main}")

  # This call will pause graph execution. user_decision will be the dict from main.py
  # e.g. {"action": "approve"}
  # The returned value from interrupt() IS the output of this node when the graph resumes.
  return interrupt(interrupt_payload_for_main)


# NEW node to process the user's decision from human_tool_review_node
def process_human_review_decision_node(state: AppState, user_decision: Dict) -> dict:
  """
  Processes the decision made by the human during review.
  Updates messages and investigation_log based on the decision.
  The 'user_decision' is the direct output of the 'human_tool_review_node' after resume.
  """
  print(f"--- Processing Human Review Decision: {user_decision} ---")

  action = user_decision.get("action", "error")
  profile = state.get("profile", "unknown_profile")

  # The AIMessage that proposed the tool call is the last one before interrupt.
  # We need to be careful if other messages got added; for now, assume it's state["messages"][-1]
  # A safer way is to pass the ID of the AIMessage that was reviewed, if complex state changes occur.
  # For simplicity, let's assume the relevant AIMessage is the last one in the current state.

  ai_message_that_was_reviewed = None
  for msg in reversed(state["messages"]):  # Find the latest AIMessage with tool_calls
    if isinstance(msg, AIMessage) and msg.tool_calls:
      ai_message_that_was_reviewed = msg
      break

  if not ai_message_that_was_reviewed:
    # This should not happen if routing is correct.
    err_msg = "Critical Error: Could not find the AIMessage that was reviewed in process_human_review_decision_node."
    print(err_msg)
    return {"messages": [HumanMessage(content=err_msg)]}  # Send error to agent

  original_tool_call = ai_message_that_was_reviewed.tool_calls[0]
  original_tool_args = original_tool_call["args"]
  extracted_reasoning = _extract_reasoning_from_ai_message(ai_message_that_was_reviewed)

  current_log = state.get("investigation_log", [])
  log_updates = []
  message_updates = []  # To be processed by add_messages

  if action == "approve":
    print("User approved tool call.")
    log_updates.append({
      "reasoning": f"User approved proposed command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
      "command": f"{profile}.{original_tool_args.get('plugin_name')} {' '.join(original_tool_args.get('plugin_args') or [])}",
      "output": "Command approved by user."
    })
    # The ai_message_that_was_reviewed is already correct. No message update needed.

  elif action == "modify":
    print("User modified tool call.")
    modified_tool_args = user_decision.get("modified_tool_args")
    if not modified_tool_args:
      err_msg = "Error: Modification chosen, but no modified_tool_args provided."
      print(err_msg)
      message_updates.append(HumanMessage(content=err_msg))  # Send to agent
    else:
      new_tool_call_structure = {
        "name": original_tool_call["name"],
        "args": modified_tool_args,
        "id": original_tool_call["id"]
      }
      # Replace the old AIMessage with a new one with modified tool_call
      modified_ai_message = AIMessage(
        content=ai_message_that_was_reviewed.content,
        tool_calls=[new_tool_call_structure],
        id=ai_message_that_was_reviewed.id  # Ensures replacement
      )
      message_updates.append(modified_ai_message)
      log_updates.append({
        "reasoning": f"User modified command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": (
          f"Original: {profile}.{original_tool_args.get('plugin_name')} {' '.join(original_tool_args.get('plugin_args') or [])}\n"
          f"Modified to: {profile}.{modified_tool_args.get('plugin_name')} {' '.join(modified_tool_args.get('plugin_args') or [])}"
        ),
        "output": "Command modified by user."
      })

  elif action == "deny":
    print("User denied tool call.")
    denial_reason = user_decision.get("reason", "No reason provided.")
    # Replace the AI message that had the tool call with one that doesn't
    ai_message_without_tool_call = AIMessage(
      content=ai_message_that_was_reviewed.content,
      tool_calls=[],  # Remove tool call
      id=ai_message_that_was_reviewed.id  # Ensure replacement
    )
    message_updates.append(ai_message_without_tool_call)
    # Add a new HumanMessage with the denial feedback for the LLM
    feedback_to_llm = HumanMessage(
      content=f"User feedback: The proposed tool call was denied. Reason: {denial_reason}. Please reconsider."
    )
    message_updates.append(feedback_to_llm)
    log_updates.append({
      "reasoning": f"User denied command. Reason: {denial_reason}\nOriginal AI Reasoning:\n{extracted_reasoning}",
      "command": f"Denied: {profile}.{original_tool_args.get('plugin_name')} {' '.join(original_tool_args.get('plugin_args') or [])}",
      "output": f"Denied by user. Reason: {denial_reason}"
    })

  else:
    err_msg = f"Error: Invalid action '{action}' in user decision."
    print(err_msg)
    message_updates.append(HumanMessage(content=err_msg))

  return {
    "messages": message_updates,
    "investigation_log": current_log + log_updates if log_updates else current_log
  }


# --- Graph Definition ---
graph_builder = StateGraph(AppState)

# The START node should not re-run profile detection if resuming.
# This means `detect_profile` should only be run once.
# We can achieve this by having START go to detect_profile, and then subsequent cycles
# (after tool execution or denial) go directly to agent.

graph_builder.add_node("detect_profile", detect_profile_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("human_tool_review_interrupt", human_tool_review_node)  # Node that calls interrupt()
graph_builder.add_node("process_review_decision",
                       process_human_review_decision_node)  # Node that handles the resume payload
graph_builder.add_node("volatility_tool_node", volatility_tool_node)

graph_builder.add_edge(START, "detect_profile")


def after_detect_profile(state: AppState):
  # This node is only hit once at the beginning.
  # If profile detection fails, agent_node will handle it and potentially end.
  return "agent"


graph_builder.add_conditional_edges(
  "detect_profile",
  after_detect_profile,
  {"agent": "agent"}
)


def route_after_agent(state: AppState) -> str:
  # This node is hit after profile detection, and after tool execution, and after denial+feedback.
  if not state.get("profile"):  # Profile detection failed initially
    return END  # Agent node would have added a message.

  last_message = state["messages"][-1]
  if isinstance(last_message, AIMessage) and last_message.tool_calls:
    return "human_tool_review_interrupt"
  else:
    # No tool call (final summary, or error from agent, or agent handled profile failure)
    return END


graph_builder.add_conditional_edges(
  "agent",
  route_after_agent,
  {
    "human_tool_review_interrupt": "human_tool_review_interrupt",
    END: END
  },
)

# The output of human_tool_review_interrupt (the user_decision dict) becomes the input
# to process_review_decision node.
# We connect human_tool_review_interrupt to process_review_decision.
graph_builder.add_edge("human_tool_review_interrupt", "process_review_decision")


def route_after_processing_review(state: AppState) -> str:
  """
  Decides where to go after process_review_decision_node.
  This decision is based on the messages *added by* process_review_decision_node.
  Specifically, if the last message is a HumanMessage (denial feedback), go to agent.
  If the last message is an AIMessage with tool_calls (modified/approved), go to tool.
  """
  last_message = state["messages"][-1]  # Check the very last message in the state

  # The AIMessage that was reviewed and potentially modified is now updated in the state.
  # Find the latest AIMessage which might have tool calls.
  relevant_ai_message = None
  for msg in reversed(state["messages"]):
    if isinstance(msg, AIMessage):  # We care about the latest AI's intention
      relevant_ai_message = msg
      break

  if isinstance(last_message, HumanMessage):  # This implies a denial or an error from processing
    return "agent"
  elif relevant_ai_message and relevant_ai_message.tool_calls:  # Approved or Modified
    return "volatility_tool_node"
  else:  # Should not happen if logic is correct, but as a fallback or if AIMessage had no tool calls after mod.
    print("Warning: Unexpected state after processing review. Defaulting to agent.")
    return "agent"


graph_builder.add_conditional_edges(
  "process_review_decision",
  route_after_processing_review,
  {
    "volatility_tool_node": "volatility_tool_node",
    "agent": "agent"
  }
)

graph_builder.add_edge("volatility_tool_node", "agent")

checkpointer = MemorySaver()
app_graph = graph_builder.compile(checkpointer=checkpointer)
