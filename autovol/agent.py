import os
from typing import List, Optional, Dict, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
# Changed import from ChatAnthropic to ChatVertexAI
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt, interrupt

from .volatility_runner import detect_profile, run_volatility_tool_logic, volatility_runner_tool
from .prompts import SYSTEM_PROMPT_TEMPLATE

DETECT_PROFILE_CALL_COUNT = 0


class AppState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  dump_path: str
  initial_context: str
  profile: Optional[str]
  investigation_log: List[Dict[str, str]]
  last_user_review_decision: Optional[Dict]


# Instantiate ChatVertexAI instead of ChatAnthropic
llm = ChatVertexAI(
  model="gemini-2.5-pro-preview-05-06",
  temperature=1,
  max_output_tokens=8000,  # Gemini models use max_output_tokens
)
llm_with_tool = llm.bind_tools([volatility_runner_tool])


def _extract_reasoning_from_ai_message(ai_message: AIMessage) -> str:
  reasoning_parts = []
  # This part for 'claude-messages-thinking' will likely not find anything with Gemini, which is fine.
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

  # This part, extracting 'thinking' or 'text' from content blocks, remains relevant.
  # Gemini might be prompted to structure its reasoning in text blocks.
  thinking_from_content_blocks = []
  text_from_content_blocks = []
  if isinstance(ai_message.content, list):
    for block in ai_message.content:
      if isinstance(block, dict):
        if block.get("type") == "thinking" and block.get("thinking"):  # This type is less common for Gemini
          thinking_from_content_blocks.append(str(block["thinking"]))
        elif block.get("type") == "text" and block.get("text"):
          text_from_content_blocks.append(str(block["text"]))

  if thinking_from_content_blocks:
    reasoning_parts.append(
      "LLM Thinking (from content block):\n" + "\n".join(filter(None, thinking_from_content_blocks)))

  # This will be the primary source of reasoning if the LLM follows the prompt.
  if text_from_content_blocks:
    combined_text = "\n".join(filter(None, text_from_content_blocks))
    if combined_text.strip():
      reasoning_parts.append("LLM Reasoning/Action Text:\n" + combined_text)
  elif isinstance(ai_message.content, str) and ai_message.content.strip() and not reasoning_parts:
    reasoning_parts.append(ai_message.content)  # Fallback to raw string content

  if not reasoning_parts:
    return "No detailed reasoning extracted from AI message. Ensure the LLM provides reasoning in its textual response."
  return "\n---\n".join(reasoning_parts)


def volatility_tool_node(state: AppState) -> dict:
  messages = state["messages"]
  ai_message_with_tool_call = None
  for msg in reversed(messages):
    if isinstance(msg, AIMessage) and msg.tool_calls:
      ai_message_with_tool_call = msg
      break

  if not ai_message_with_tool_call:
    return {"messages": [ToolMessage(content="Error: No AIMessage with tool_calls found to execute.",
                                     tool_call_id="error_no_aim_for_tool_node")]}

  tool_call = ai_message_with_tool_call.tool_calls[0]
  tool_name_called_by_llm = tool_call["name"]
  tool_args = tool_call["args"]

  if tool_name_called_by_llm != volatility_runner_tool.name:
    return {"messages": [ToolMessage(content=f"Error: Unknown tool '{tool_name_called_by_llm}' called by LLM.",
                                     tool_call_id=tool_call["id"])]}

  dump_path = state.get("dump_path")
  profile_context = state.get("profile")

  if not dump_path:
    error_msg = "Error: dump_path not found in agent state for tool execution."
    print(error_msg)
    return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

  plugin_name_from_llm = tool_args.get("plugin_name")
  plugin_args_from_llm = tool_args.get("plugin_args")

  if not plugin_name_from_llm:
    error_msg = "Error: 'plugin_name' missing in tool arguments from LLM."
    print(error_msg)
    return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

  print(f"--- Executing Volatility Tool: {plugin_name_from_llm} {' '.join(plugin_args_from_llm or [])} ---")
  tool_output_content = run_volatility_tool_logic(
    plugin_name=plugin_name_from_llm,
    plugin_args=plugin_args_from_llm,
    dump_path=dump_path,
    profile=profile_context
  )
  # The reasoning extraction should pick up the LLM's textual reasoning if prompted correctly.
  reasoning_for_log = _extract_reasoning_from_ai_message(ai_message_with_tool_call)
  log_entry = {
    "reasoning": reasoning_for_log,
    "command": f"{plugin_name_from_llm} {' '.join(plugin_args_from_llm or [])}",
    "output": tool_output_content
  }
  current_log = state.get("investigation_log", [])
  new_log = current_log + [log_entry]
  return {
    "messages": [ToolMessage(content=tool_output_content, name=plugin_name_from_llm, tool_call_id=tool_call["id"])],
    "investigation_log": new_log
  }


def detect_profile_node(state: AppState) -> dict:
  global DETECT_PROFILE_CALL_COUNT
  DETECT_PROFILE_CALL_COUNT += 1
  print(f"--- Detecting Profile (Call #{DETECT_PROFILE_CALL_COUNT}) ---")
  dump_path = state["dump_path"]
  profile = detect_profile(dump_path)
  if not profile:
    error_msg = f"Failed to detect a suitable OS profile for {dump_path}."
    print(error_msg)
    return {"profile": None}
  print(f"Profile (OS base) detected: {profile}")
  return {"profile": profile}


def agent_node(state: AppState) -> dict:
  print("--- Calling LLM Agent ---")
  profile = state.get("profile")

  if not profile:
    return {"messages": [AIMessage(
      content="Analysis cannot proceed: Failed to detect a suitable OS profile. Please check the dump file or Volatility setup.")]}

  current_messages = state["messages"]

  investigation_log = state.get("investigation_log", [])
  log_summary_parts = []
  for entry in investigation_log[-5:]:
    cmd = entry.get("command", "N/A")
    output_preview = entry.get("output", "N/A")
    if len(output_preview) > 150:
      output_preview = output_preview[:150] + "..."
    log_summary_parts.append(f"- Command: {cmd}\n  Output Preview: {output_preview}")
  investigation_log_summary_str = "\n".join(log_summary_parts) if log_summary_parts else "No commands run yet."

  system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
    dump_path=state['dump_path'],
    profile=profile,
    initial_context=state['initial_context'],
    investigation_log_summary=investigation_log_summary_str
  )

  messages_for_llm = [SystemMessage(content=system_prompt_content)]
  messages_for_llm.extend([m for m in current_messages if not isinstance(m, SystemMessage)])

  response = llm_with_tool.invoke(messages_for_llm)

  print(f"LLM Full Response Content Type: {type(response.content)}")
  if isinstance(response.content, list):
    for item_content in response.content:
      print(f"LLM Content Item: {item_content}")
  else:
    print(f"LLM Full Response Content: {response.content}")

  if response.tool_calls:
    print(f"LLM Tool Calls: {response.tool_calls}")
  return {"messages": [response]}


def human_tool_review_node(state: AppState) -> dict:
  print("--- Human Review Node ---")
  last_ai_message_with_tool_call = None
  for msg in reversed(state["messages"]):
    if isinstance(msg, AIMessage) and msg.tool_calls:
      last_ai_message_with_tool_call = msg
      break

  interrupt_payload_for_main_prompt = {}
  if not last_ai_message_with_tool_call:
    print("Error: human_tool_review_node reached without a preceding tool call from AI.")
    interrupt_payload_for_main_prompt = {
      "reasoning_and_thinking": "Error: No AI tool call to review.",
      "tool_call_args": {"plugin_name": "N/A", "plugin_args": []},
      "tool_call_id": "error_no_tool_call_for_review",
      "error_condition": True
    }
  else:
    tool_call_to_review = last_ai_message_with_tool_call.tool_calls[0]
    extracted_reasoning = _extract_reasoning_from_ai_message(last_ai_message_with_tool_call)
    interrupt_payload_for_main_prompt = {
      "reasoning_and_thinking": extracted_reasoning,
      "tool_call_args": tool_call_to_review["args"],
      "tool_call_id": tool_call_to_review["id"],
    }

  print(f"Interrupting for human review. Payload for main.py prompt: {interrupt_payload_for_main_prompt}")
  user_decision_from_main = interrupt(interrupt_payload_for_main_prompt)
  print(f"Resumed from interrupt. User decision from main.py: {user_decision_from_main}")
  return {"last_user_review_decision": user_decision_from_main}


def process_human_review_decision_node(state: AppState) -> dict:
  user_decision = state.get("last_user_review_decision")

  if user_decision is None:
    err_msg = "Critical Error: last_user_review_decision not found in state for process_human_review_decision_node."
    print(err_msg)
    return {"messages": [HumanMessage(content=err_msg)], "last_user_review_decision": None}

  print(f"--- Processing Human Review Decision (from state): {user_decision} ---")
  action = user_decision.get("action", "error")

  ai_message_that_was_reviewed = None
  for msg in reversed(state.get("messages", [])):
    if isinstance(msg, AIMessage) and msg.tool_calls:
      ai_message_that_was_reviewed = msg
      break

  original_plugin_name = "N/A"
  original_plugin_args_list = []
  original_tool_call_name_attr = "volatility_runner"  # Default, should be tool name from tool call
  original_tool_call_id_attr = "unknown_id"
  extracted_reasoning = "Original AI reasoning not available."
  original_ai_content = [{"type": "text", "text": "Original AI message content not available."}]
  original_ai_message_id = None
  original_ai_response_metadata = {}

  if ai_message_that_was_reviewed:
    original_tool_call = ai_message_that_was_reviewed.tool_calls[0]
    original_tool_args_dict = original_tool_call["args"]
    original_plugin_name = original_tool_args_dict.get('plugin_name', 'plugin_not_specified')
    original_plugin_args_list = original_tool_args_dict.get('plugin_args') or []
    original_tool_call_name_attr = original_tool_call.get("name", "volatility_runner")
    original_tool_call_id_attr = original_tool_call["id"]
    extracted_reasoning = _extract_reasoning_from_ai_message(ai_message_that_was_reviewed)
    original_ai_content = ai_message_that_was_reviewed.content
    original_ai_message_id = ai_message_that_was_reviewed.id
    original_ai_response_metadata = getattr(ai_message_that_was_reviewed, 'response_metadata', {})

  current_log = state.get("investigation_log", [])
  log_updates = []
  message_updates_to_add = []

  if action == "approve":
    print("User approved tool call.")
    # No change to AI message needed if approved, it will proceed to tool node
    log_updates.append({
      "reasoning": f"User approved proposed command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
      "command": f"{original_plugin_name} {' '.join(original_plugin_args_list)}",
      "output": "Command approved by user."
    })

  elif action == "modify":
    print("User modified tool call.")
    modified_tool_args = user_decision.get("modified_tool_args")
    if not modified_tool_args or not modified_tool_args.get("plugin_name"):
      err_msg = "Error: Modification chosen, but no valid modified_tool_args (with plugin_name) provided."
      print(err_msg)
      message_updates_to_add.append(HumanMessage(content=err_msg))
    elif not original_ai_message_id:  # Check if we have an original AI message to modify
      err_msg = "Error: Modification chosen, but original AI message to modify was not found."
      print(err_msg)
      message_updates_to_add.append(HumanMessage(content=err_msg))
    else:
      modified_plugin_name_val = modified_tool_args.get('plugin_name')
      modified_plugin_args_list = modified_tool_args.get('plugin_args') or []
      # Create a new AIMessage with the modified tool call, preserving other attributes
      new_tool_call_structure = {"name": original_tool_call_name_attr, "args": modified_tool_args,
                                 "id": original_tool_call_id_attr}
      modified_ai_message = AIMessage(content=original_ai_content, tool_calls=[new_tool_call_structure],
                                      id=original_ai_message_id, response_metadata=original_ai_response_metadata)
      # Replace the last AI message with this modified one (or add it if more robust handling is needed)
      # For simplicity, we add; the graph's routing will pick up the latest AIMessage with tool_calls.
      message_updates_to_add.append(modified_ai_message)
      log_updates.append({
        "reasoning": f"User modified command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": (f"Original: {original_plugin_name} {' '.join(original_plugin_args_list)}\n"
                    f"Modified to: {modified_plugin_name_val} {' '.join(modified_plugin_args_list)}"),
        "output": "Command modified by user."
      })

  elif action == "deny":
    print("User denied tool call.")
    denial_reason = user_decision.get("reason", "No reason provided.")
    if not original_ai_message_id:  # Check if we have an original AI message
      err_msg = "Error: Deny action chosen, but original AI message to modify was not found."
      print(err_msg)
      message_updates_to_add.append(HumanMessage(content=err_msg))
    else:
      # Create a new AIMessage without the tool call, preserving content and ID
      ai_message_without_tool_call = AIMessage(content=original_ai_content, tool_calls=[], id=original_ai_message_id,
                                               response_metadata=original_ai_response_metadata)
      message_updates_to_add.append(ai_message_without_tool_call)
      # Add a HumanMessage with feedback for the LLM
      feedback_to_llm = HumanMessage(
        content=f"User feedback: The proposed tool call ({original_plugin_name}) was denied. Reason: {denial_reason}. Please reconsider your next step based on this feedback and the existing context.")
      message_updates_to_add.append(feedback_to_llm)
      log_updates.append({
        "reasoning": f"User denied command. Reason: {denial_reason}\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": f"Denied: {original_plugin_name} {' '.join(original_plugin_args_list)}",
        "output": f"Denied by user. Reason: {denial_reason}"
      })

  elif action == "internal_error_at_review":
    err_reason = user_decision.get("reason", "Unknown internal error during review step initiation.")
    print(f"Internal error flagged during review step: {err_reason}")
    # Add a human message to inform the agent of the system error
    message_updates_to_add.append(HumanMessage(
      content=f"System: Internal error occurred before human review could properly start: {err_reason}. Agent needs to reassess."))
    log_updates.append(
      {"reasoning": "Internal error in human review step initiation.", "command": "N/A (Internal Error)",
       "output": err_reason})

  else:  # Fallback for any unhandled action
    err_msg = f"Error: Invalid or unhandled action '{action}' in user decision. Decision data: {user_decision}"
    print(err_msg)
    message_updates_to_add.append(HumanMessage(content=err_msg))  # Inform LLM

  # If messages were added, state.messages will be updated by add_messages reducer
  return_dict = {
    "investigation_log": current_log + log_updates if log_updates else current_log,
    "last_user_review_decision": None  # Clear the decision
  }
  if message_updates_to_add:
    return_dict["messages"] = message_updates_to_add

  return return_dict


graph_builder = StateGraph(AppState)
graph_builder.add_node("detect_profile", detect_profile_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("human_tool_review_interrupt", human_tool_review_node)
graph_builder.add_node("process_review_decision", process_human_review_decision_node)
graph_builder.add_node("volatility_tool_node", volatility_tool_node)

graph_builder.add_edge(START, "detect_profile")
graph_builder.add_conditional_edges("detect_profile", lambda s: "agent", {"agent": "agent"})


def route_after_agent(state: AppState) -> str:
  if not state.get("profile"):
    print("Routing after agent: Profile not set, ending.")
    return END
  last_message = state["messages"][-1] if state.get("messages") else None
  if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print("Routing after agent: AIMessage has tool calls, going to human_tool_review_interrupt.")
    return "human_tool_review_interrupt"
  else:
    print("Routing after agent: No tool calls in last AIMessage (or no messages), ending.")
    return END


graph_builder.add_conditional_edges("agent", route_after_agent,
                                    {"human_tool_review_interrupt": "human_tool_review_interrupt", END: END})

graph_builder.add_edge("human_tool_review_interrupt", "process_review_decision")


def route_after_processing_review(state: AppState) -> str:
  latest_ai_message_with_potential_tool_call = None
  human_feedback_present_for_denial = False
  internal_error_message_present = False

  # Iterate from most recent messages to find the relevant signals
  for msg in reversed(state.get("messages", [])):
    if isinstance(msg, AIMessage):
      # Found the latest AI message, this is key for checking for tool calls.
      # If it has tool calls, it means 'approve' or 'modify' happened and we should execute.
      # If it doesn't have tool calls, it could be due to 'deny' (where we replaced it) or other reasons.
      if latest_ai_message_with_potential_tool_call is None:  # only take the most recent AI message
        latest_ai_message_with_potential_tool_call = msg

    if isinstance(msg, HumanMessage):
      if "User feedback: The proposed tool call" in msg.content and "was denied" in msg.content:
        human_feedback_present_for_denial = True
      if "System: Internal error occurred" in msg.content:  # Check for our specific system error message
        internal_error_message_present = True

    # Optimization: if we found an AI message and a denial/error feedback, we might have enough info.
    # However, the logic is safer to scan relevant parts of recent history.

  if latest_ai_message_with_potential_tool_call and latest_ai_message_with_potential_tool_call.tool_calls:
    # This implies an 'approve' or 'modify' action resulted in an AIMessage with tool_calls.
    print(
      "Routing after processing review: AIMessage has tool calls (approved/modified), going to volatility_tool_node.")
    return "volatility_tool_node"
  elif human_feedback_present_for_denial or internal_error_message_present:
    # A 'deny' action or an internal error explicitly directs back to the agent for reconsideration.
    print(
      "Routing after processing review: Denial feedback or internal error feedback present, going back to agent.")
    return "agent"
  elif latest_ai_message_with_potential_tool_call and not latest_ai_message_with_potential_tool_call.tool_calls:
    # This could be an AI message that was stripped of its tool calls due to denial,
    # or an AI message that legitimately has no tool calls.
    # If no explicit denial/error feedback, but the AI message has no tool calls,
    # it's safer to loop back to the agent to decide the next step (e.g., finish or try something else).
    print(
      "Routing after processing review: Latest AIMessage has no tool calls (and no explicit denial/error feedback), going back to agent.")
    return "agent"
  else:
    # Fallback: if state is unclear (e.g. no recent AI message found after processing review, which is unlikely)
    print(
      "Warning: Unexpected state after processing review. Defaulting to agent.")
    return "agent"


graph_builder.add_conditional_edges("process_review_decision", route_after_processing_review,
                                    {"volatility_tool_node": "volatility_tool_node", "agent": "agent"})
graph_builder.add_edge("volatility_tool_node", "agent")

checkpointer = MemorySaver()
app_graph = graph_builder.compile(checkpointer=checkpointer)
