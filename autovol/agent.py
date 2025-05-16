# autovol/agent.py
import os
import uuid  # For unique filenames if needed
from typing import List, Optional, Dict, Annotated, TypedDict, Sequence, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt, interrupt
from pathlib import Path
# import tempfile # We'll use a session-specific dir for easier user access

from .volatility_runner import detect_profile, run_volatility_tool_logic, volatility_runner_tool, \
  list_all_available_plugins
from .python_interpreter import python_interpreter_tool, run_python_code_logic
from .prompts import SYSTEM_PROMPT_TEMPLATE

DETECT_PROFILE_CALL_COUNT = 0


class AppState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  dump_path: str
  initial_context: str
  profile: Optional[str]
  available_plugins: Optional[List[str]]
  investigation_log: List[Dict[str, Any]]
  last_user_review_decision: Optional[Dict]
  report_session_id: str


llm = ChatVertexAI(
  model="gemini-2.5-pro-preview-05-06",
  temperature=1,
  max_output_tokens=8000,
)
llm_with_tool = llm.bind_tools([volatility_runner_tool, python_interpreter_tool])


def _extract_reasoning_from_ai_message(ai_message: AIMessage) -> str:
  # ... (no changes to this function)
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
  elif isinstance(ai_message.content, str):
    text_from_content_blocks.append(ai_message.content)

  if thinking_from_content_blocks:
    reasoning_parts.append(
      "LLM Thinking (from content block):\n" + "\n".join(filter(None, thinking_from_content_blocks)))

  if text_from_content_blocks:
    combined_text = "\n".join(filter(None, text_from_content_blocks))
    if combined_text.strip():
      reasoning_parts.append(combined_text)

  if not reasoning_parts and isinstance(ai_message.content, str) and ai_message.content.strip():
    reasoning_parts.append(ai_message.content)

  if not reasoning_parts:
    return "No detailed reasoning extracted from AI message. Ensure the LLM provides reasoning in its textual response."
  return "\n---\n".join(reasoning_parts)


# tool_executor_node, detect_profile_node, list_plugins_node, agent_node remain unchanged from the previous step
# ... (paste the unchanged tool_executor_node, detect_profile_node, list_plugins_node, agent_node here) ...
def tool_executor_node(state: AppState) -> dict:
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
  report_session_id = state.get("report_session_id")

  if not report_session_id:
    return {"messages": [ToolMessage(content="Error: report_session_id not found in state.",
                                     tool_call_id=tool_call["id"])]}

  log_entry = {}
  tool_output_for_message = ""
  reasoning_for_log = _extract_reasoning_from_ai_message(ai_message_with_tool_call)

  if tool_name_called_by_llm == volatility_runner_tool.name:
    dump_path = state.get("dump_path")
    profile_context = state.get("profile")

    if not dump_path:
      error_msg = "Error: dump_path not found in agent state for Volatility tool execution."
      return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

    plugin_name_from_llm = tool_args.get("plugin_name")
    plugin_args_from_llm = tool_args.get("plugin_args")

    if not plugin_name_from_llm:
      error_msg = "Error: 'plugin_name' missing in Volatility tool arguments from LLM."
      return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

    print(f"--- Executing Volatility Tool: {plugin_name_from_llm} {' '.join(plugin_args_from_llm or [])} ---")
    tool_output_content = run_volatility_tool_logic(
      plugin_name=plugin_name_from_llm,
      plugin_args=plugin_args_from_llm,
      dump_path=dump_path,
      profile=profile_context
    )
    tool_output_for_message = tool_output_content

    report_session_dir = Path("reports") / report_session_id
    command_outputs_dir = report_session_dir / "command_outputs"
    command_outputs_dir.mkdir(parents=True, exist_ok=True)
    step_number = len(state.get("investigation_log", [])) + 1
    sanitized_plugin_name = "".join(c if c.isalnum() or c in ['_', '.'] else '_' for c in plugin_name_from_llm)
    output_filename = f"step_{step_number}_{sanitized_plugin_name}.txt"
    full_output_file_path_obj = command_outputs_dir / output_filename
    output_file_path_for_log = None
    try:
      with open(full_output_file_path_obj, "w", encoding="utf-8") as f:
        f.write(tool_output_content)
      print(f"Full Volatility output saved to: {full_output_file_path_obj}")
      output_file_path_for_log = str(Path("command_outputs") / output_filename)
    except Exception as e:
      print(f"Error saving full Volatility output to {full_output_file_path_obj}: {e}")

    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"volatility: {plugin_name_from_llm} {' '.join(plugin_args_from_llm or [])}",
      "tool_input": tool_args,
      "output_file_path": output_file_path_for_log,
      "output_details": tool_output_content[:500] + "..." if len(tool_output_content) > 500 else tool_output_content,
      "raw_output_preview_for_prompt": tool_output_content[:250] + "..." if len(
        tool_output_content) > 250 else tool_output_content
    }

  elif tool_name_called_by_llm == python_interpreter_tool.name:
    python_code_to_execute = tool_args.get("code")  # This code is the one approved/modified by user
    if not python_code_to_execute:
      error_msg = "Error: 'code' missing in Python interpreter tool arguments from LLM."
      return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

    print(f"--- Executing Python Code ---")
    execution_result = run_python_code_logic(python_code_to_execute)
    stdout_str = execution_result.get("stdout", "")
    stderr_str = execution_result.get("stderr", "")

    tool_output_for_message = f"Python Code Execution Result:\nStdout:\n{stdout_str}\nStderr:\n{stderr_str}"

    combined_output_for_details = f"Stdout:\n```\n{stdout_str.strip()}\n```\n\nStderr:\n```\n{stderr_str.strip()}\n```"

    preview_limit = 125
    stdout_preview = stdout_str[:preview_limit] + ('...' if len(stdout_str) > preview_limit else '')
    stderr_preview = stderr_str[:preview_limit] + ('...' if len(stderr_str) > preview_limit else '')
    raw_preview = f"Python Stdout: {stdout_preview.strip()}"
    if stderr_str.strip():
      raw_preview += f"\nPython Stderr: {stderr_preview.strip()}"

    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"python_code:\n```python\n{python_code_to_execute}\n```",
      "tool_input": tool_args,
      "output_file_path": None,
      "output_details": combined_output_for_details.strip(),
      "raw_output_preview_for_prompt": raw_preview.strip()
    }

  else:
    error_msg = f"Error: Unknown tool '{tool_name_called_by_llm}' called by LLM."
    return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

  current_log = state.get("investigation_log", [])
  new_log = current_log + [log_entry]

  return {
    "messages": [
      ToolMessage(content=tool_output_for_message, name=tool_name_called_by_llm, tool_call_id=tool_call["id"])],
    "investigation_log": new_log
  }


def detect_profile_node(state: AppState) -> dict:
  global DETECT_PROFILE_CALL_COUNT
  DETECT_PROFILE_CALL_COUNT += 1
  print(f"--- Detecting Profile (Call #{DETECT_PROFILE_CALL_COUNT}) ---")
  dump_path = state["dump_path"]
  profile = detect_profile(dump_path)
  if not profile:
    return {"profile": None}
  print(f"Profile (OS base) detected: {profile}")
  return {"profile": profile}


def list_plugins_node(state: AppState) -> dict:
  print("--- Listing Available Volatility Plugins ---")
  profile = state.get("profile")
  if not profile:
    print("Critical Error: list_plugins_node called but profile is None.")
    return {"available_plugins": None}

  all_plugins = list_all_available_plugins()
  if not all_plugins:
    print("Failed to retrieve any plugins from 'vol.py -h'.")
    return {"available_plugins": None}

  profile_prefix = profile + "."
  relevant_plugins = []
  other_os_prefixes = [p + "." for p in ["windows", "linux", "mac"] if p != profile]

  for p_name in all_plugins:
    is_for_current_profile = p_name.startswith(profile_prefix)
    is_for_other_os = any(p_name.startswith(op) for op in other_os_prefixes)
    if is_for_current_profile or not is_for_other_os:
      relevant_plugins.append(p_name)
  relevant_plugins = sorted(list(set(relevant_plugins)))

  if not relevant_plugins:
    print(f"No plugins found specifically matching profile '{profile}' or general plugins.")
    return {"available_plugins": None}

  print(f"Found {len(relevant_plugins)} relevant plugins for profile '{profile}'. First few: {relevant_plugins[:5]}")
  return {"available_plugins": relevant_plugins}


def agent_node(state: AppState) -> dict:
  print("--- Calling LLM Agent ---")
  profile = state.get("profile")
  available_plugins_list = state.get("available_plugins")

  if not profile:
    return {"messages": [AIMessage(
      content="Analysis cannot proceed: Failed to detect a suitable OS profile.")]}

  current_messages = state["messages"]
  investigation_log = state.get("investigation_log", [])
  log_summary_parts = []

  for entry in investigation_log[-5:]:
    cmd_display = entry.get("command", "N/A")
    entry_type = entry.get("type", "unknown")
    tool_name_entry = entry.get("tool_name", "")

    if entry_type == "tool_execution":
      output_preview = entry.get("raw_output_preview_for_prompt", "Full output processed previously.")
      if tool_name_entry == python_interpreter_tool.name:
        cmd_display = "Python Code Executed (see details in log)"
      elif tool_name_entry == volatility_runner_tool.name:
        cmd_display = cmd_display

      log_summary_parts.append(f"- Command: {cmd_display}\n  Output Preview: {output_preview}")
    elif entry_type == "user_decision" or entry_type == "internal_error":
      details = entry.get("output_details", "Details not available.")
      log_summary_parts.append(f"- Action/Event: {cmd_display}\n  Details: {details}")
    else:
      details = entry.get("output_details", entry.get("output_summary", "N/A"))
      log_summary_parts.append(f"- Command/Action: {cmd_display}\n  Details/Summary: {details}")

  investigation_log_summary_str = "\n".join(log_summary_parts) if log_summary_parts else "No commands run yet."

  available_plugins_str = "Could not be determined. Proceed with caution using common plugins for the '{profile}' OS base."
  if available_plugins_list:
    if len(available_plugins_list) > 40:
      os_specific_examples = [p for p in available_plugins_list if p.startswith(profile + ".")][:2]
      general_examples = [p for p in available_plugins_list if
                          not any(p.startswith(os_base + ".") for os_base in ["windows.", "linux.", "mac."])][:2]
      examples = sorted(list(set(os_specific_examples + general_examples)))
      example_str = f" (e.g., {', '.join(examples)}, ... and others)" if examples else ""
      available_plugins_str = (f"A list of {len(available_plugins_list)} Volatility plugins is available{example_str}. "
                               f"The full list is too long to display here. Focus on plugins starting with '{profile}.' or general purpose ones.")
    else:
      available_plugins_str = "\n- " + "\n- ".join(available_plugins_list)
  elif available_plugins_list == []:
    available_plugins_str = "No specific Volatility plugins were found for the detected profile or general context after filtering. Please use well-known standard plugins for '{profile}' OS base with caution."

  system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
    dump_path=state['dump_path'],
    profile=profile,
    initial_context=state['initial_context'],
    investigation_log_summary=investigation_log_summary_str,
    available_plugins_list_str=available_plugins_str
  )

  messages_for_llm = [SystemMessage(content=system_prompt_content)]
  messages_for_llm.extend([m for m in current_messages if not isinstance(m, SystemMessage)])

  response = llm_with_tool.invoke(messages_for_llm)
  return {"messages": [response]}


def human_tool_review_node(state: AppState) -> dict:
  last_ai_message_with_tool_call = None
  for msg in reversed(state["messages"]):
    if isinstance(msg, AIMessage) and msg.tool_calls:
      last_ai_message_with_tool_call = msg
      break

  interrupt_payload_for_main_prompt = {}
  if not last_ai_message_with_tool_call:
    interrupt_payload_for_main_prompt = {
      "reasoning_and_thinking": "Error: No AI tool call to review.",
      "tool_name": "N/A",
      "tool_call_args": {},  # Keep generic for error
      "tool_call_id": "error_no_tool_call_for_review",
      "error_condition": True
    }
  else:
    tool_call_to_review = last_ai_message_with_tool_call.tool_calls[0]
    tool_name = tool_call_to_review["name"]
    tool_args = tool_call_to_review["args"]
    extracted_reasoning = _extract_reasoning_from_ai_message(last_ai_message_with_tool_call)

    temp_script_path_for_review = None
    if tool_name == python_interpreter_tool.name:
      python_code = tool_args.get("code", "# No code provided by LLM")
      report_session_id = state.get("report_session_id", "unknown_session")
      # Create a dedicated subdirectory for scripts to be reviewed
      review_scripts_dir = Path("reports") / report_session_id / "review_scripts"
      review_scripts_dir.mkdir(parents=True, exist_ok=True)

      # Sanitize tool_call_id for use in filename or use a UUID
      sane_tool_call_id = "".join(c if c.isalnum() else '_' for c in tool_call_to_review['id'])
      script_file_name = f"review_script_{sane_tool_call_id}.py"
      temp_script_path_obj = review_scripts_dir / script_file_name

      try:
        with open(temp_script_path_obj, "w", encoding="utf-8") as f:
          f.write(python_code)
        temp_script_path_for_review = str(temp_script_path_obj.resolve())  # Get absolute path
        print(f"Python script for review saved to: {temp_script_path_for_review}")
      except Exception as e:
        print(f"Error saving Python script for review: {e}")
        # temp_script_path_for_review will remain None, main.py should handle this
        pass

    interrupt_payload_for_main_prompt = {
      "reasoning_and_thinking": extracted_reasoning,
      "tool_name": tool_name,
      "tool_call_args": tool_args,  # Still pass original args (contains code for snippet fallback)
      "tool_call_id": tool_call_to_review["id"],
      "temp_script_path": temp_script_path_for_review  # This will be None if not Python or if save failed
    }
  user_decision_from_main = interrupt(interrupt_payload_for_main_prompt)
  return {"last_user_review_decision": user_decision_from_main}


def process_human_review_decision_node(state: AppState) -> dict:
  # This node doesn't need to change significantly.
  # If the user modified the Python code, main.py will read the edited content
  # from the temp file and pass the *string content* of the new code
  # in `user_decision.get("modified_tool_args")`.
  # This node then constructs the AIMessage with this new code string.
  # ... (paste the unchanged process_human_review_decision_node here) ...
  user_decision = state.get("last_user_review_decision")

  if user_decision is None:
    err_msg = "Critical Error: last_user_review_decision not found."
    return {"messages": [HumanMessage(content=err_msg)], "last_user_review_decision": None}

  action = user_decision.get("action", "error")
  ai_message_that_was_reviewed = None
  for msg in reversed(state.get("messages", [])):
    if isinstance(msg, AIMessage) and msg.tool_calls:
      ai_message_that_was_reviewed = msg
      break

  original_tool_name_called = "unknown_tool"
  original_tool_args_dict_display = {}
  command_display_for_log = "N/A"

  original_tool_call_name_attr = "unknown_tool"
  original_tool_call_id_attr = "unknown_id"
  extracted_reasoning = "Original AI reasoning not available."
  original_ai_content = [{"type": "text", "text": "Original AI content not available."}]
  original_ai_message_id = None
  original_ai_response_metadata = {}

  if ai_message_that_was_reviewed:
    original_tool_call = ai_message_that_was_reviewed.tool_calls[0]
    original_tool_name_called = original_tool_call["name"]
    original_tool_args_dict_display = original_tool_call["args"]
    original_tool_call_name_attr = original_tool_call.get("name", original_tool_name_called)
    original_tool_call_id_attr = original_tool_call["id"]
    extracted_reasoning = _extract_reasoning_from_ai_message(ai_message_that_was_reviewed)
    original_ai_content = ai_message_that_was_reviewed.content
    original_ai_message_id = ai_message_that_was_reviewed.id
    original_ai_response_metadata = getattr(ai_message_that_was_reviewed, 'response_metadata', {})

    if original_tool_name_called == volatility_runner_tool.name:
      p_name = original_tool_args_dict_display.get('plugin_name', 'N/A')
      p_args = original_tool_args_dict_display.get('plugin_args', [])
      command_display_for_log = f"volatility: {p_name} {' '.join(p_args)}"
    elif original_tool_name_called == python_interpreter_tool.name:
      p_code = original_tool_args_dict_display.get('code', '# No code provided')
      command_display_for_log = f"python_code:\n```python\n{p_code[:200]}{'...' if len(p_code) > 200 else ''}\n```"

  current_log = state.get("investigation_log", [])
  log_updates = []
  message_updates_to_add = []

  log_entry_base = {"type": "user_decision", "output_file_path": None, "tool_name": original_tool_name_called}

  if action == "approve":
    log_updates.append({
      **log_entry_base,
      "reasoning": f"User approved proposed command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
      "command": command_display_for_log,
      "tool_input": original_tool_args_dict_display,
      "output_details": "Command approved by user."
    })
    # For 'approve', no new AIMessage is added by this node. The existing one from 'agent_node'
    # (which triggered the interrupt) will proceed to 'tool_executor'.

  elif action == "modify":
    modified_tool_args_from_user = user_decision.get("modified_tool_args")
    valid_modification = False
    modified_command_display_for_log = "N/A (modification error)"

    if original_tool_name_called == volatility_runner_tool.name:
      if modified_tool_args_from_user and modified_tool_args_from_user.get("plugin_name"):
        valid_modification = True
        mod_p_name = modified_tool_args_from_user.get('plugin_name')
        mod_p_args = modified_tool_args_from_user.get('plugin_args', [])
        modified_command_display_for_log = f"volatility: {mod_p_name} {' '.join(mod_p_args)}"
    elif original_tool_name_called == python_interpreter_tool.name:
      # modified_tool_args_from_user for Python will be {"code": "new_code_string"}
      if modified_tool_args_from_user and "code" in modified_tool_args_from_user:
        valid_modification = True
        mod_p_code = modified_tool_args_from_user.get('code', '# No code provided')
        modified_command_display_for_log = f"python_code:\n```python\n{mod_p_code[:200]}{'...' if len(mod_p_code) > 200 else ''}\n```"

    if not valid_modification:
      message_updates_to_add.append(HumanMessage(
        content=f"Error: Modification chosen for {original_tool_name_called}, but new arguments are invalid."))
    elif not original_ai_message_id:
      message_updates_to_add.append(
        HumanMessage(content="Error: Modification chosen, but original AI message context not found."))
    else:
      new_tool_call = {"name": original_tool_call_name_attr, "args": modified_tool_args_from_user,
                       "id": original_tool_call_id_attr}
      modified_ai_message = AIMessage(
        content=original_ai_content,
        tool_calls=[new_tool_call],
        id=original_ai_message_id,
        response_metadata=original_ai_response_metadata
      )
      message_updates_to_add.append(modified_ai_message)
      log_updates.append({
        **log_entry_base,
        "reasoning": f"User modified command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": (f"Original: {command_display_for_log}\n"
                    f"Modified to: {modified_command_display_for_log}"),
        "tool_input": modified_tool_args_from_user,
        "output_details": "Command modified by user."
      })

  elif action == "deny":
    denial_reason = user_decision.get("reason", "No reason provided.")
    if not original_ai_message_id:
      message_updates_to_add.append(
        HumanMessage(content="Error: Deny action chosen, but original AI message context not found."))
    else:
      ai_message_no_tool = AIMessage(
        content=original_ai_content,
        tool_calls=[],
        id=original_ai_message_id,
        response_metadata=original_ai_response_metadata
      )
      message_updates_to_add.append(ai_message_no_tool)
      message_updates_to_add.append(HumanMessage(
        content=f"User feedback: The proposed tool call ({original_tool_name_called}) was denied. Reason: {denial_reason}. Please reconsider."))
      log_updates.append({
        **log_entry_base,
        "reasoning": f"User denied command. Reason: {denial_reason}\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": f"Denied: {command_display_for_log}",
        "tool_input": original_tool_args_dict_display,
        "output_details": f"Denied by user. Reason: {denial_reason}"
      })

  elif action == "internal_error_at_review":
    err_reason = user_decision.get("reason", "Unknown internal error during review.")
    message_updates_to_add.append(HumanMessage(
      content=f"System: Internal error occurred before human review: {err_reason}. Agent needs to reassess."))
    log_updates.append({
      "type": "internal_error",
      "reasoning": "Internal error in human review step initiation.",
      "command": "N/A (Internal Error)",
      "output_details": err_reason,
      "output_file_path": None
    })
  else:
    message_updates_to_add.append(HumanMessage(content=f"Error: Invalid action '{action}' from user decision."))

  return_dict = {
    "investigation_log": current_log + log_updates if log_updates else current_log,
    "last_user_review_decision": None
  }
  if message_updates_to_add:
    return_dict["messages"] = message_updates_to_add

  return return_dict


# Graph definition (no changes to structure)
graph_builder = StateGraph(AppState)
graph_builder.add_node("detect_profile", detect_profile_node)
graph_builder.add_node("list_plugins", list_plugins_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("human_tool_review_interrupt", human_tool_review_node)
graph_builder.add_node("process_review_decision", process_human_review_decision_node)
graph_builder.add_node("tool_executor", tool_executor_node)

graph_builder.add_edge(START, "detect_profile")


def route_after_profile_detection(state: AppState) -> str:
  return "list_plugins" if state.get("profile") else "agent"


graph_builder.add_conditional_edges("detect_profile", route_after_profile_detection,
                                    {"list_plugins": "list_plugins", "agent": "agent"})
graph_builder.add_edge("list_plugins", "agent")


def route_after_agent(state: AppState) -> str:
  if not state.get("profile"): return END
  last_msg = state["messages"][-1] if state.get("messages") else None
  return "human_tool_review_interrupt" if isinstance(last_msg, AIMessage) and last_msg.tool_calls else END


graph_builder.add_conditional_edges("agent", route_after_agent,
                                    {"human_tool_review_interrupt": "human_tool_review_interrupt", END: END})

graph_builder.add_edge("human_tool_review_interrupt", "process_review_decision")


def route_after_processing_review(state: AppState) -> str:
  most_recent_ai_message_with_tool_call = None
  # Iterate backwards through all messages to find the effective state
  for msg_idx in range(len(state.get("messages", [])) - 1, -1, -1):
    msg = state["messages"][msg_idx]
    if isinstance(msg, AIMessage):
      if msg.tool_calls:  # This AIMessage intends to call a tool
        most_recent_ai_message_with_tool_call = msg
      # If it's an AIMessage without tool_calls, it means AI decided not to use a tool OR a previous tool_call was stripped (e.g. by deny)
      # In either case, the presence of this AIMessage means we are past any tool_call decision for *this specific* AIMessage.
      break
    if isinstance(msg, HumanMessage):
      # If a HumanMessage implies a denial or error *after* the last AIMessage we checked,
      # then the tool call is effectively cancelled.
      if ("User feedback: The proposed tool call" in msg.content and "was denied" in msg.content) or \
        ("System: Internal error occurred" in msg.content):
        most_recent_ai_message_with_tool_call = None  # Effectively cancelled
        break
        # Other HumanMessages (like initial context) don't cancel a preceding AI tool call.

  if most_recent_ai_message_with_tool_call:
    return "tool_executor"
  else:
    return "agent"


graph_builder.add_conditional_edges("process_review_decision", route_after_processing_review,
                                    {"tool_executor": "tool_executor", "agent": "agent"})
graph_builder.add_edge("tool_executor", "agent")

checkpointer = MemorySaver()
app_graph = graph_builder.compile(checkpointer=checkpointer)
