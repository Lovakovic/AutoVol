import os
import uuid
from typing import List, Optional, Dict, Annotated, TypedDict, Sequence, Any
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt, interrupt
from pathlib import Path

from .volatility_runner import (
  detect_profile, run_volatility_tool_logic, volatility_runner_tool,
  list_all_available_plugins,
  get_volatility_plugin_help_tool, get_volatility_plugin_help_logic
)
from .python_interpreter import python_interpreter_tool, run_python_code_logic
from .workspace_utils import list_workspace_files_tool, list_workspace_files_logic
from .image_viewer import view_image_file_tool, view_image_file_logic
from .prompts import SYSTEM_PROMPT_TEMPLATE

DETECT_PROFILE_CALL_COUNT = 0
AGENT_NODE_CALL_COUNT = 0  # Can be kept for console call count or removed

# Configuration for message history management
MAX_MESSAGES_IN_CONTEXT = int(os.getenv("AUTOVOL_MAX_MESSAGES", "60"))  # Default to 50 messages
KEEP_LAST_N_MESSAGES = int(os.getenv("AUTOVOL_KEEP_LAST_N", "30"))  # When trimming, keep last 30


class AppState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  dump_path: str
  initial_context: str
  profile: Optional[str]
  available_plugins: Optional[List[str]]
  investigation_log: List[Dict[str, Any]]
  last_user_review_decision: Optional[Dict]
  report_session_id: str
  image_analysis_history: List[Dict[str, Any]]
  multimodal_context: Dict[str, Any]
  token_usage: Dict[str, int]  # Track input_tokens, output_tokens, total_tokens


# Gemini configuration
# llm = ChatVertexAI(
#   model="gemini-2.5-pro-preview-05-06",
#   temperature=0.7,
#   max_output_tokens=8000,
# )

# Claude configuration
llm = ChatAnthropic(
  model="claude-sonnet-4-20250514",
  max_tokens=32000,
  thinking={"type": "enabled", "budget_tokens": 4000},
  api_key=os.getenv("ANTHROPIC_API_KEY")
)


# OpenAI configuration
# reasoning = {
#     "effort": "medium"  # 'low', 'medium', or 'high'
# }
# llm = ChatOpenAI(
#     model="o4-mini", use_responses_api=True, model_kwargs={"reasoning": reasoning}
# )

llm_with_tool = llm.bind_tools([
  volatility_runner_tool,
  python_interpreter_tool,
  list_workspace_files_tool,
  get_volatility_plugin_help_tool,
  view_image_file_tool
])


def _extract_reasoning_from_ai_message(ai_message: AIMessage) -> str:
  reasoning_parts = []
  
  # Check if we're using Anthropic (Claude)
  is_anthropic = isinstance(llm, ChatAnthropic)
  
  # First try to extract thinking from content blocks (Anthropic's format)
  thinking_from_content_blocks = []
  text_from_content_blocks = []
  
  if isinstance(ai_message.content, list):  # Handles models outputting a list of content blocks
    for block in ai_message.content:
      if isinstance(block, dict):
        # Anthropic's thinking blocks have type='thinking' with 'thinking' field
        if block.get("type") == "thinking" and block.get("thinking"):
          thinking_from_content_blocks.append(str(block["thinking"]))
        # Standard text blocks
        elif block.get("type") == "text" and block.get("text"):
          text_from_content_blocks.append(str(block["text"]))
        # Tool use blocks (for completeness)
        elif block.get("type") == "tool_use":
          # Tool calls are handled separately, don't include in reasoning
          pass
      # If a block is just a string (less common for structured outputs but possible)
      elif isinstance(block, str):
        text_from_content_blocks.append(block)
  elif isinstance(ai_message.content, str):  # Handles models outputting a single string content
    text_from_content_blocks.append(ai_message.content)
  
  # If we found thinking blocks (from Anthropic), prioritize those
  if thinking_from_content_blocks:
    reasoning_parts.append(
      "LLM Thinking Process:\n" + "\n".join(filter(None, thinking_from_content_blocks)))
  
  # For non-Anthropic models or when no thinking blocks found, check response_metadata
  if not thinking_from_content_blocks and hasattr(ai_message, 'response_metadata'):
    # Check for any thinking-related keys in metadata
    metadata = ai_message.response_metadata
    
    # Try various possible keys for thinking/reasoning
    thinking_keys = ['claude-messages-thinking', 'thinking', 'reasoning', 'thought_process']
    for key in thinking_keys:
      if key in metadata:
        thinking_log = metadata[key]
        if isinstance(thinking_log, list) and thinking_log:
          thinking_content = "\n".join(
            [step.get('thinking', '') if isinstance(step, dict) else str(step) for step in thinking_log if
             (isinstance(step, dict) and step.get('thinking')) or isinstance(step, str)]
          )
          if thinking_content.strip():
            reasoning_parts.append("LLM Thinking Process (from metadata):\n" + thinking_content.strip())
            break
        elif isinstance(thinking_log, str) and thinking_log.strip():
          reasoning_parts.append("LLM Thinking Process (from metadata):\n" + thinking_log.strip())
          break
  
  # Add primary text content, ensuring it's not just re-adding thinking parts
  if text_from_content_blocks:
    combined_text = "\n".join(filter(None, text_from_content_blocks)).strip()
    if combined_text:
      # For Anthropic, if we already have thinking blocks, only add text if it's substantial
      if thinking_from_content_blocks:
        # Only add text content if it provides additional context beyond thinking
        if len(combined_text) > 50:  # Arbitrary threshold for "substantial"
          reasoning_parts.append("Response:\n" + combined_text)
      else:
        # For non-Anthropic or when no thinking found, include all text
        reasoning_parts.append(combined_text)
  
  # Fallback if no specific reasoning/thinking found, use the main content string
  if not reasoning_parts and isinstance(ai_message.content, str) and ai_message.content.strip():
    reasoning_parts.append(ai_message.content.strip())
  
  if not reasoning_parts:  # If truly nothing, give a default
    return "No detailed reasoning extracted. Main AI response content considered as reasoning if available."
  
  return "\n---\n".join(reasoning_parts)


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

  session_workspace_path = Path("reports") / report_session_id / "workspace"
  try:
    session_workspace_path.mkdir(parents=True, exist_ok=True)
  except Exception as e:
    return {"messages": [ToolMessage(content=f"Error creating session workspace '{session_workspace_path}': {e}",
                                     tool_call_id=tool_call["id"])]}

  log_entry = {}
  tool_output_for_message_content = ""
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
    vol_result_dict = run_volatility_tool_logic(
      plugin_name=plugin_name_from_llm,
      plugin_args=plugin_args_from_llm,
      dump_path=dump_path,
      profile=profile_context,
      session_workspace_dir=str(session_workspace_path)
    )

    stdout_preview = vol_result_dict.get("stdout_preview", "")
    stderr_content = vol_result_dict.get("stderr", "")
    return_code = vol_result_dict.get("return_code", -1)
    saved_workspace_stdout_file = vol_result_dict.get("saved_workspace_stdout_file")
    error_message_from_logic = vol_result_dict.get("error_message")
    info_message_from_logic = vol_result_dict.get("info_message")

    if error_message_from_logic:
      tool_output_for_message_content = error_message_from_logic
    elif saved_workspace_stdout_file:
      tool_output_for_message_content = (
        f"Plugin '{plugin_name_from_llm}' executed (RC={return_code}).\n"
        f"Full standard output saved to workspace file: '{saved_workspace_stdout_file}'\n"
        f"Output Preview:\n```\n{stdout_preview}\n```\n"
      )
      if stderr_content.strip():
        tool_output_for_message_content += f"Stderr (if any):\n```\n{stderr_content}\n```\n"
      tool_output_for_message_content += ("(Other files may also have been created in the workspace "
                                          "if the plugin was instructed to, e.g., with --dump-dir .)")
    elif info_message_from_logic:
      tool_output_for_message_content = info_message_from_logic
    else:
      tool_output_for_message_content = f"Volatility plugin '{plugin_name_from_llm}' execution completed (RC={return_code}). Check logs. Stderr: {stderr_content}"

    report_session_dir = Path("reports") / report_session_id
    command_outputs_logging_dir = report_session_dir / "command_outputs"
    command_outputs_logging_dir.mkdir(parents=True, exist_ok=True)
    step_number = len(state.get("investigation_log", [])) + 1
    sanitized_plugin_name_for_log = "".join(c if c.isalnum() or c in ['_', '.'] else '_' for c in plugin_name_from_llm)
    sane_args_str = "_".join(
      arg.replace('-', '') for arg in (plugin_args_from_llm or []) if arg.isalnum() or arg in ['_', '.'])
    our_log_filename = f"step_{step_number}_{sanitized_plugin_name_for_log}_{sane_args_str[:30]}_fulllog.txt".replace(
      "__", "_").replace("__", "_")  # Handle multiple underscores

    full_our_log_file_path_obj = command_outputs_logging_dir / our_log_filename
    our_log_file_path_for_report_md = None
    try:
      # For our internal log, save the full stdout (not just preview) and stderr
      full_stdout_from_execution = vol_result_dict.get("stdout_preview",
                                                       "")  # This key currently holds the full stdout before truncation for ToolMessage
      # If 'stdout_preview' was already truncated in run_volatility_tool_logic, this needs adjustment
      # Assuming run_volatility_tool_logic's stdout_preview IS the full stdout before it gets truncated for the ToolMessage
      content_for_our_log = f"COMMAND: vol -f {dump_path} {plugin_name_from_llm} {' '.join(plugin_args_from_llm or [])}\n\n"
      content_for_our_log += f"RETURN CODE: {return_code}\n\n"
      content_for_our_log += f"STDOUT (also in workspace/{saved_workspace_stdout_file if saved_workspace_stdout_file else 'N/A'}):\n{full_stdout_from_execution}\n\n"
      content_for_our_log += f"STDERR:\n{stderr_content}\n"
      with open(full_our_log_file_path_obj, "w", encoding="utf-8") as f_log:
        f_log.write(content_for_our_log)
      print(f"Internal log of Volatility output saved to: {full_our_log_file_path_obj}")
      our_log_file_path_for_report_md = str(Path("command_outputs") / our_log_filename)
    except Exception as e_log_save:
      print(f"Error saving internal Volatility log to {full_our_log_file_path_obj}: {e_log_save}")

    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"volatility: {plugin_name_from_llm} {' '.join(plugin_args_from_llm or [])}",
      "tool_input": tool_args,
      "output_file_path": our_log_file_path_for_report_md,
      "workspace_output_file": saved_workspace_stdout_file,
      "output_details": tool_output_for_message_content[:700] + "..." if len(
        tool_output_for_message_content) > 700 else tool_output_for_message_content,
      "raw_output_preview_for_prompt": tool_output_for_message_content[:350] + "..." if len(
        tool_output_for_message_content) > 350 else tool_output_for_message_content
    }

  elif tool_name_called_by_llm == python_interpreter_tool.name:
    python_code_to_execute = tool_args.get("code")
    if not python_code_to_execute:
      error_msg = "Error: 'code' missing in Python interpreter tool arguments from LLM."
      return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

    print(f"--- Executing Python Code (in workspace: {session_workspace_path}) ---")
    execution_result = run_python_code_logic(
      python_code_to_execute,
      session_workspace_dir=str(session_workspace_path)
    )
    stdout_str = execution_result.get("stdout", "")
    stderr_str = execution_result.get("stderr", "")
    tool_output_for_message_content = f"Python Code Execution Result:\nStdout:\n{stdout_str}\nStderr:\n{stderr_str}"
    combined_output_for_details = f"Stdout:\n```\n{stdout_str.strip()}\n```\n\nStderr:\n```\n{stderr_str.strip()}\n```"
    preview_limit = 125  # For log summary
    stdout_preview_log = stdout_str[:preview_limit] + ('...' if len(stdout_str) > preview_limit else '')
    stderr_preview_log = stderr_str[:preview_limit] + ('...' if len(stderr_str) > preview_limit else '')
    raw_preview_for_prompt = f"Python Stdout: {stdout_preview_log.strip()}"  # For next LLM prompt
    if stderr_str.strip():
      raw_preview_for_prompt += f"\nPython Stderr: {stderr_preview_log.strip()}"

    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"python_code (executed in session workspace):\n```python\n{python_code_to_execute}\n```",
      "tool_input": tool_args,
      "output_file_path": None,
      "workspace_output_file": None,
      "output_details": combined_output_for_details.strip(),
      "raw_output_preview_for_prompt": raw_preview_for_prompt.strip()  # Use the potentially truncated one for prompt
    }

  elif tool_name_called_by_llm == list_workspace_files_tool.name:
    relative_path_arg = tool_args.get("relative_path", ".")
    print(f"--- Listing Workspace Files (path: {relative_path_arg} in {session_workspace_path}) ---")
    tool_output_content = list_workspace_files_logic(
      session_workspace_dir=str(session_workspace_path),
      relative_path=relative_path_arg
    )
    tool_output_for_message_content = tool_output_content
    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"list_workspace_files: relative_path='{relative_path_arg}'",
      "tool_input": tool_args,
      "output_file_path": None,
      "workspace_output_file": None,
      "output_details": tool_output_content,
      "raw_output_preview_for_prompt": tool_output_content[:350] + "..." if len(
        tool_output_content) > 350 else tool_output_content
    }

  elif tool_name_called_by_llm == get_volatility_plugin_help_tool.name:
    plugin_name_for_help = tool_args.get("plugin_name")
    if not plugin_name_for_help:
      error_msg = "Error: 'plugin_name' missing for get_volatility_plugin_help tool."
      tool_output_for_message_content = error_msg
    else:
      print(f"--- Getting Help for Volatility Plugin: {plugin_name_for_help} ---")
      help_result_dict = get_volatility_plugin_help_logic(plugin_name_for_help)

      if "error" in help_result_dict:
        tool_output_for_message_content = help_result_dict["error"]
        if "help_text" in help_result_dict:  # Check if partial help text was returned despite error
          tool_output_for_message_content += f"\nPartial Help Text (if any):\n{help_result_dict['help_text'][:1000]}..."
      else:
        tool_output_for_message_content = help_result_dict.get("help_text",
                                                               f"No specific help text found for {plugin_name_for_help}.")

    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"get_volatility_plugin_help: plugin_name='{tool_args.get('plugin_name', 'N/A')}'",
      "tool_input": tool_args,
      "output_file_path": None,
      "workspace_output_file": None,
      "output_details": tool_output_for_message_content[:1500] + "..." if len(
        tool_output_for_message_content) > 1500 else tool_output_for_message_content,
      "raw_output_preview_for_prompt": tool_output_for_message_content[:1000] + "..." if len(
        tool_output_for_message_content) > 1000 else tool_output_for_message_content
    }

  elif tool_name_called_by_llm == view_image_file_tool.name:
    file_path_arg = tool_args.get("file_path")
    analysis_prompt_arg = tool_args.get("analysis_prompt")
    new_image_entry = None
    
    if not file_path_arg:
      error_msg = "Error: 'file_path' missing in image viewer tool arguments from LLM."
      tool_output_for_message_content = error_msg
    else:
      print(f"--- Executing Image Analysis: {file_path_arg} ---")
      analysis_result = view_image_file_logic(
        file_path=file_path_arg,
        session_workspace_dir=str(session_workspace_path),
        analysis_prompt=analysis_prompt_arg,
        llm_instance=llm
      )
      
      if analysis_result["success"]:
        image_info = analysis_result["image_info"]
        analysis_text = analysis_result["analysis_result"]
        image_token_usage = analysis_result.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        
        tool_output_for_message_content = f"Image Analysis Results for '{file_path_arg}':\n\n"
        
        # Add image metadata
        if image_info:
          tool_output_for_message_content += f"**Image Metadata:**\n"
          tool_output_for_message_content += f"- Format: {image_info.get('format', 'Unknown')}\n"
          tool_output_for_message_content += f"- Dimensions: {image_info.get('dimensions', 'Unknown')}\n"
          tool_output_for_message_content += f"- Size: {image_info.get('file_size_mb', 0):.2f} MB\n"
          tool_output_for_message_content += f"- MIME Type: {image_info.get('mime_type', 'Unknown')}\n\n"
        
        # Add analysis results
        tool_output_for_message_content += f"**Forensic Analysis:**\n{analysis_text}\n"
        
        # Prepare image analysis history entry
        new_image_entry = {
          "file_path": file_path_arg,
          "analysis_prompt": analysis_prompt_arg,
          "image_info": image_info,
          "analysis_result": analysis_text,
          "success": True
        }
        
      else:
        tool_output_for_message_content = f"Image Analysis Failed: {analysis_result['error_message']}"
        new_image_entry = {
          "file_path": file_path_arg,
          "analysis_prompt": analysis_prompt_arg,
          "error": analysis_result['error_message'],
          "success": False
        }
    
    # Prepare log entry
    log_entry = {
      "type": "tool_execution",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"view_image_file: '{file_path_arg}'",
      "tool_input": tool_args,
      "output_file_path": None,
      "workspace_output_file": None,
      "output_details": tool_output_for_message_content[:1500] + "..." if len(
        tool_output_for_message_content) > 1500 else tool_output_for_message_content,
      "raw_output_preview_for_prompt": tool_output_for_message_content[:1000] + "..." if len(
        tool_output_for_message_content) > 1000 else tool_output_for_message_content
    }

  else:
    error_msg = f"Error: Unknown tool '{tool_name_called_by_llm}' called by LLM."
    tool_output_for_message_content = error_msg
    log_entry = {
      "type": "tool_execution_error",
      "tool_name": tool_name_called_by_llm,
      "reasoning": reasoning_for_log,
      "command": f"Unknown tool call: {tool_name_called_by_llm} with args {tool_args}",
      "tool_input": tool_args,
      "output_details": error_msg,
      "raw_output_preview_for_prompt": error_msg
    }

  current_log = state.get("investigation_log", [])
  new_log = current_log + [log_entry]
  
  # Prepare return dict
  return_dict = {
    "messages": [
      ToolMessage(content=tool_output_for_message_content, name=tool_name_called_by_llm, tool_call_id=tool_call["id"])],
    "investigation_log": new_log
  }
  
  # Add image analysis history if this was an image analysis
  if tool_name_called_by_llm == view_image_file_tool.name and 'new_image_entry' in locals() and new_image_entry is not None:
    current_image_history = state.get("image_analysis_history", [])
    return_dict["image_analysis_history"] = current_image_history + [new_image_entry]
    
    # Update token usage if image analysis used tokens
    if 'image_token_usage' in locals():
      current_token_usage = state.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
      current_token_usage["input_tokens"] += image_token_usage["input_tokens"]
      current_token_usage["output_tokens"] += image_token_usage["output_tokens"]
      current_token_usage["total_tokens"] += image_token_usage["total_tokens"]
      return_dict["token_usage"] = current_token_usage
      
      print(f"Image analysis token usage - Input: {image_token_usage['input_tokens']}, "
            f"Output: {image_token_usage['output_tokens']}, "
            f"Total: {image_token_usage['total_tokens']}")
  
  return return_dict


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
    # This state should ideally be handled by routing or agent_node if profile detection fails.
    print("Critical Error: list_plugins_node called but profile is None.")
    return {"available_plugins": None}  # Or perhaps trigger an error state/message

  all_plugins = list_all_available_plugins()  # No argument needed, uses default _get_volatility_cmd
  if not all_plugins:
    print("Failed to retrieve any plugins from 'vol -h'.")
    return {"available_plugins": None}  # Keep as None if truly no plugins found

  profile_prefix = profile + "."
  relevant_plugins = []
  # Define other OS prefixes to filter out plugins specific to *other* OSes,
  # but keep general plugins (those not starting with any OS prefix).
  all_os_bases = ["windows", "linux", "mac"]
  other_os_prefixes = [p + "." for p in all_os_bases if p != profile]

  for p_name in all_plugins:
    is_for_current_profile = p_name.startswith(profile_prefix)
    is_for_other_os = any(p_name.startswith(op) for op in other_os_prefixes)

    # Keep if for current profile OR if not specific to any other OS (i.e., general)
    if is_for_current_profile or not is_for_other_os:
      relevant_plugins.append(p_name)

  relevant_plugins = sorted(list(set(relevant_plugins)))  # Deduplicate and sort

  if not relevant_plugins:
    print(f"No plugins found specifically matching profile '{profile}' or general plugins after filtering.")
    # Return empty list instead of None to distinguish from "failed to get list"
    return {"available_plugins": []}

  print(f"Found {len(relevant_plugins)} relevant plugins for profile '{profile}'. First few: {relevant_plugins[:5]}")
  return {"available_plugins": relevant_plugins}


def agent_node(state: AppState) -> dict:
  global AGENT_NODE_CALL_COUNT
  AGENT_NODE_CALL_COUNT += 1
  print(f"--- Calling LLM Agent (Call #{AGENT_NODE_CALL_COUNT}) ---")

  # REMOVED LLM I/O Tracing Directory and File Saving
  # report_session_id = state.get("report_session_id", "unknown_session")
  # llm_io_trace_dir = Path("reports") / report_session_id / "llm_io_trace"
  # try:
  #   llm_io_trace_dir.mkdir(parents=True, exist_ok=True)
  # except Exception as e:
  #   print(f"Warning: Could not create LLM I/O trace directory '{llm_io_trace_dir}': {e}")

  profile = state.get("profile")
  available_plugins_list = state.get("available_plugins")

  if not profile:
    return {"messages": [AIMessage(
      content="Analysis cannot proceed: Failed to detect a suitable OS profile. Ending analysis.")]}

  current_messages = state["messages"]
  investigation_log = state.get("investigation_log", [])
  log_summary_parts = []

  for entry in investigation_log[-5:]:  # Consider if -5 is enough for complex sequences
    cmd_display = entry.get("command", "N/A")
    entry_type = entry.get("type", "unknown")
    tool_name_entry = entry.get("tool_name", "")
    # Use a shorter, more consistent preview for the log summary
    output_preview = entry.get("raw_output_preview_for_prompt", "Details processed by agent.")
    if len(output_preview) > 200:  # Truncate if very long for the summary
      output_preview = output_preview[:200] + "..."

    workspace_file_info = ""
    if entry_type == "tool_execution" and tool_name_entry == volatility_runner_tool.name and entry.get(
      "workspace_output_file"):
      workspace_file_info = f"\n  (Volatility stdout saved to workspace: '{entry['workspace_output_file']}')"
    elif entry_type == "tool_execution" and tool_name_entry == get_volatility_plugin_help_tool.name:
      workspace_file_info = ""  # No file output from help tool itself

    if entry_type == "tool_execution":
      if tool_name_entry == python_interpreter_tool.name:
        cmd_display = "Python Code Executed (see details in main report log)"
      log_summary_parts.append(f"- Command: {cmd_display}\n  Output Preview: {output_preview}{workspace_file_info}")
    elif entry_type == "user_decision" or entry_type == "internal_error":
      details = entry.get("output_details", "Details not available.")
      log_summary_parts.append(f"- Action/Event: {cmd_display}\n  Details: {details}")
    elif entry_type == "tool_execution_error":  # Specific log for unknown tool
      details = entry.get("output_details", "Error details not available.")
      log_summary_parts.append(f"- Tool Error: {cmd_display}\n Details: {details}")
    else:  # Fallback for other types (should be rare)
      details = entry.get("output_details", entry.get("output_summary", "N/A"))
      log_summary_parts.append(f"- Command/Action: {cmd_display}\n  Details/Summary: {details}{workspace_file_info}")

  investigation_log_summary_str = "\n".join(log_summary_parts) if log_summary_parts else "No commands run yet."

  available_plugins_str = "Could not be determined. Proceed with caution using common plugins for the '{profile}' OS base."
  if available_plugins_list is not None:  # Check for None (failed) vs empty list (none found)
    if not available_plugins_list:  # Empty list
      available_plugins_str = "No Volatility plugins were found after filtering for the current OS or general purpose. You may need to use very common, standard plugin names for '{profile}' and the system will attempt to run them."
    elif len(available_plugins_list) > 40:  # Long list
      os_specific_examples = [p for p in available_plugins_list if p.startswith(profile + ".")][:2]
      general_examples = [p for p in available_plugins_list if
                          not any(p.startswith(os_base + ".") for os_base in ["windows.", "linux.", "mac."])][:2]
      examples = sorted(list(set(os_specific_examples + general_examples)))
      example_str = f" (e.g., {', '.join(examples)}, ... and others)" if examples else ""
      available_plugins_str = (f"A list of {len(available_plugins_list)} Volatility plugins is available{example_str}. "
                               f"The full list is too long to display here. Focus on plugins starting with '{profile}.' or general purpose ones.")
    else:  # Short list
      available_plugins_str = "\n- " + "\n- ".join(available_plugins_list)
  # If available_plugins_list is None (initial state or error getting them), the default message is used.

  system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
    dump_path=state['dump_path'],
    profile=profile,
    initial_context=state['initial_context'],
    investigation_log_summary=investigation_log_summary_str,
    available_plugins_list_str=available_plugins_str
  )

  messages_for_llm = [SystemMessage(content=system_prompt_content)]
  
  # Add all non-SystemMessages from the current state to the history
  non_system_messages = [m for m in current_messages if not isinstance(m, SystemMessage)]
  
  # Implement message trimming to prevent context window overflow
  if len(non_system_messages) > MAX_MESSAGES_IN_CONTEXT:
    print(f"WARNING: Message history has {len(non_system_messages)} messages, exceeding limit of {MAX_MESSAGES_IN_CONTEXT}")
    print(f"Trimming to keep last {KEEP_LAST_N_MESSAGES} messages...")
    
    # Always keep the first message (initial context) and the last N messages
    if len(non_system_messages) > 0:
      trimmed_messages = [non_system_messages[0]]  # Keep initial context
      trimmed_messages.extend(non_system_messages[-KEEP_LAST_N_MESSAGES:])
      
      # Add a summary message about what was trimmed
      num_trimmed = len(non_system_messages) - len(trimmed_messages)
      trim_notice = HumanMessage(content=f"[System Notice: {num_trimmed} older messages were trimmed to stay within context limits. The investigation continues from the most recent messages.]")
      trimmed_messages.insert(1, trim_notice)  # Insert after initial context
      
      non_system_messages = trimmed_messages
  
  # Check and truncate individual messages that are too large
  MAX_MESSAGE_LENGTH = 100000  # 100K chars per message
  processed_messages = []
  
  for msg in non_system_messages:
    # Clean content to ensure valid UTF-8
    if isinstance(msg.content, str):
      # Remove invalid UTF-8 characters (surrogates)
      try:
        # Try to encode/decode to catch invalid characters
        clean_content = msg.content.encode('utf-8', errors='replace').decode('utf-8')
        if clean_content != msg.content:
          print(f"WARNING: Cleaned invalid UTF-8 characters from {type(msg).__name__}")
      except Exception:
        # If even that fails, use a more aggressive cleaning
        clean_content = ''.join(char for char in msg.content if ord(char) < 0x10000)
        print(f"WARNING: Aggressively cleaned content in {type(msg).__name__}")
      
      # Now check length on cleaned content
      if len(clean_content) > MAX_MESSAGE_LENGTH:
        print(f"WARNING: {type(msg).__name__} content is {len(clean_content)} chars, truncating to {MAX_MESSAGE_LENGTH}")
        truncated_content = clean_content[:MAX_MESSAGE_LENGTH] + f"\n\n[... {len(clean_content) - MAX_MESSAGE_LENGTH} characters truncated ...]"
      else:
        truncated_content = clean_content
      
      # Create new message with cleaned/truncated content, preserving other attributes
      if isinstance(msg, AIMessage):
        new_msg = AIMessage(content=truncated_content, tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else None)
      elif isinstance(msg, ToolMessage):
        new_msg = ToolMessage(content=truncated_content, tool_call_id=msg.tool_call_id)
      elif isinstance(msg, HumanMessage):
        new_msg = HumanMessage(content=truncated_content)
      else:
        new_msg = msg  # Fallback, shouldn't happen
        
      processed_messages.append(new_msg)
    else:
      processed_messages.append(msg)
  
  messages_for_llm.extend(processed_messages)
  
  # Final safety check - total context size
  total_context_size = sum(len(str(msg.content)) for msg in messages_for_llm)
  MAX_TOTAL_CONTEXT = 1500000  # 1.5MB total context limit (conservative for most models)
  
  if total_context_size > MAX_TOTAL_CONTEXT:
    print(f"ERROR: Total context size ({total_context_size:,} chars) exceeds safety limit ({MAX_TOTAL_CONTEXT:,} chars)")
    print("Attempting aggressive message trimming...")
    
    # Keep only system message, initial context, and last 10 messages
    emergency_messages = [messages_for_llm[0]]  # System message
    if len(processed_messages) > 0:
      emergency_messages.append(processed_messages[0])  # Initial context
      emergency_messages.extend(processed_messages[-10:])  # Last 10 messages
      
    messages_for_llm = emergency_messages
    new_total = sum(len(str(msg.content)) for msg in messages_for_llm)
    print(f"After emergency trimming: {new_total:,} chars with {len(messages_for_llm)} messages")

  # Print context usage statistics
  print(f"Sending {len(messages_for_llm)} messages to LLM (total: {total_context_size:,} chars)")

  # Debug: Save the problematic messages for inspection if error occurs
  debug_file_path = None
  try:
    response = llm_with_tool.invoke(messages_for_llm)
  except Exception as e:
    # Save debug information
    report_session_id = state.get("report_session_id", "unknown_session")
    debug_dir = Path("reports") / report_session_id / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file_path = debug_dir / f"llm_error_{timestamp}.txt"
    
    with open(debug_file_path, "w", encoding="utf-8") as f:
      f.write(f"LLM Invocation Error Debug Information\n")
      f.write(f"=====================================\n\n")
      f.write(f"Error Type: {type(e).__name__}\n")
      f.write(f"Error Message: {str(e)}\n\n")
      
      f.write(f"Request Statistics:\n")
      f.write(f"- Number of messages: {len(messages_for_llm)}\n")
      f.write(f"- System prompt length: {len(system_prompt_content)} characters\n")
      
      total_length = sum(len(str(msg.content)) for msg in messages_for_llm)
      f.write(f"- Total content length: {total_length} characters\n\n")
      
      f.write(f"Messages Details:\n")
      f.write(f"-----------------\n")
      
      for i, msg in enumerate(messages_for_llm):
        f.write(f"\nMessage {i} - {type(msg).__name__}:\n")
        f.write(f"  Content length: {len(str(msg.content))} chars\n")
        
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
          f.write(f"  Tool calls: {len(msg.tool_calls)}\n")
          for j, tc in enumerate(msg.tool_calls):
            f.write(f"    Tool {j}: {tc.get('name', 'unknown')} - Args: {list(tc.get('args', {}).keys())}\n")
        
        if isinstance(msg.content, list):
          f.write(f"  Content type: list with {len(msg.content)} items\n")
          for j, item in enumerate(msg.content):
            if isinstance(item, dict):
              f.write(f"    Item {j}: type={item.get('type', 'unknown')}")
              if item.get('type') == 'text':
                f.write(f" length={len(item.get('text', ''))}")
              f.write("\n")
        else:
          f.write(f"  Content type: {type(msg.content).__name__}\n")
        
        # Write first 500 chars of content (with safe encoding)
        content_str = str(msg.content)
        try:
          # Clean the content for safe writing
          safe_content = content_str.encode('utf-8', errors='replace').decode('utf-8')
          f.write(f"  Content preview: {safe_content[:500]}{'...' if len(safe_content) > 500 else ''}\n")
        except Exception as e:
          f.write(f"  Content preview: [Error displaying content: {str(e)}]\n")
    
    print(f"\n!!! ERROR: LLM invocation failed !!!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print(f"\nDebug information saved to: {debug_file_path}")
    print(f"Total message content length: {total_length} characters")
    print(f"Number of messages: {len(messages_for_llm)}")
    
    # Check for common issues
    if total_length > 1000000:  # 1MB is a rough estimate
      print("\nWARNING: Total content length is very large. This might exceed the model's context window.")
    
    if len(messages_for_llm) > 100:
      print("\nWARNING: Large number of messages. Consider implementing message pruning.")
    
    # Re-raise the exception
    raise

  # Capture token usage from the response
  current_token_usage = state.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
  
  if hasattr(response, 'response_metadata') and response.response_metadata:
    # Gemini uses 'usage_metadata'
    usage_stats = response.response_metadata.get('usage_metadata', {})
    
    # Claude uses 'usage' directly in response_metadata
    if not usage_stats:
      usage_stats = response.response_metadata.get('usage', {})
    
    # OpenAI also uses 'token_usage' or 'usage'
    if not usage_stats:
      usage_stats = response.response_metadata.get('token_usage', {})
    
    if usage_stats:
      # Different providers use different key names
      # Gemini/Claude: input_tokens, output_tokens
      # OpenAI: prompt_tokens, completion_tokens
      input_tokens = usage_stats.get('input_tokens', 0) or usage_stats.get('prompt_tokens', 0)
      output_tokens = usage_stats.get('output_tokens', 0) or usage_stats.get('completion_tokens', 0)
      total_tokens = usage_stats.get('total_tokens', 0) or (input_tokens + output_tokens)
      
      # Update token counts
      current_token_usage["input_tokens"] += input_tokens
      current_token_usage["output_tokens"] += output_tokens
      current_token_usage["total_tokens"] += total_tokens
      
      print(f"Token usage for this call - Input: {input_tokens}, "
            f"Output: {output_tokens}, "
            f"Total: {total_tokens}")

  # REMOVED Saving LLM Response to File

  return {"messages": [response], "token_usage": current_token_usage}


def human_tool_review_node(state: AppState) -> dict:
  # ... (This function remains unchanged from your last working version)
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
      "tool_call_args": {},
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
      review_scripts_dir = Path("reports") / report_session_id / "review_scripts"
      review_scripts_dir.mkdir(parents=True, exist_ok=True)
      sane_tool_call_id = "".join(c if c.isalnum() else '_' for c in tool_call_to_review['id'])
      script_file_name = f"review_script_{sane_tool_call_id}.py"
      temp_script_path_obj = review_scripts_dir / script_file_name
      try:
        with open(temp_script_path_obj, "w", encoding="utf-8") as f:
          f.write(python_code)
        temp_script_path_for_review = str(temp_script_path_obj.resolve())
        print(f"Python script for review saved to: {temp_script_path_for_review}")
      except Exception as e:
        print(f"Error saving Python script for review: {e}")
        pass

    interrupt_payload_for_main_prompt = {
      "reasoning_and_thinking": extracted_reasoning,
      "tool_name": tool_name,
      "tool_call_args": tool_args,
      "tool_call_id": tool_call_to_review["id"],
      "temp_script_path": temp_script_path_for_review
    }
  user_decision_from_main = interrupt(interrupt_payload_for_main_prompt)
  return {"last_user_review_decision": user_decision_from_main}


def process_human_review_decision_node(state: AppState) -> dict:
  # ... (This function remains unchanged from your last working version)
  user_decision = state.get("last_user_review_decision")
  if user_decision is None:
    err_msg = "Critical Error: last_user_review_decision not found."
    # It's better to let the agent try to recover from this if possible.
    # Adding a HumanMessage here will signal the agent.
    return {
      "messages": [
        HumanMessage(content=f"SYSTEM ERROR: {err_msg} Human review decision was lost. Please re-evaluate.")],
      "last_user_review_decision": None  # Clear it
    }

  action = user_decision.get("action", "error")  # Default to 'error' if action key is missing
  ai_message_that_was_reviewed = None
  # Find the most recent AIMessage that actually had tool_calls (the one that triggered the review)
  for msg_idx in range(len(state.get("messages", [])) - 1, -1, -1):
    msg = state["messages"][msg_idx]
    if isinstance(msg, AIMessage) and msg.tool_calls:  # Check specifically for tool_calls
      ai_message_that_was_reviewed = msg
      break

  # Initialize defaults
  original_tool_name_called = "unknown_tool (review context lost)"
  original_tool_args_dict_display = {}
  command_display_for_log = "N/A (review context lost)"
  original_tool_call_id_attr = "unknown_id_at_review"
  extracted_reasoning = "Original AI reasoning not available (review context lost)."
  original_ai_content = "Original AI content not available (review context lost)."
  original_ai_message_id = None  # Crucial for replacing the right message
  original_ai_response_metadata = {}

  if ai_message_that_was_reviewed:
    if ai_message_that_was_reviewed.tool_calls:  # Should always be true if review was triggered
      original_tool_call = ai_message_that_was_reviewed.tool_calls[0]
      original_tool_name_called = original_tool_call["name"]
      original_tool_args_dict_display = original_tool_call["args"]
      original_tool_call_id_attr = original_tool_call["id"]
    # else:
    # This case means human_tool_review_node was called on an AIMessage without tool_calls, which is a graph logic error.
    # The interrupt payload should have error_condition:True in that case.

    extracted_reasoning = _extract_reasoning_from_ai_message(ai_message_that_was_reviewed)
    original_ai_content = ai_message_that_was_reviewed.content
    original_ai_message_id = ai_message_that_was_reviewed.id
    original_ai_response_metadata = getattr(ai_message_that_was_reviewed, 'response_metadata', {})

    # Generate command_display_for_log based on the tool type
    if original_tool_name_called == volatility_runner_tool.name:
      p_name = original_tool_args_dict_display.get('plugin_name', 'N/A')
      p_args = original_tool_args_dict_display.get('plugin_args', [])
      command_display_for_log = f"volatility: {p_name} {' '.join(p_args)}"
    elif original_tool_name_called == python_interpreter_tool.name:
      p_code = original_tool_args_dict_display.get('code', '# No code provided')
      command_display_for_log = f"python_code:\n```python\n{p_code[:200]}{'...' if len(p_code) > 200 else ''}\n```"
    elif original_tool_name_called == list_workspace_files_tool.name:
      rel_path = original_tool_args_dict_display.get('relative_path', '.')
      command_display_for_log = f"list_workspace_files: relative_path='{rel_path}'"
    elif original_tool_name_called == get_volatility_plugin_help_tool.name:
      p_name_help = original_tool_args_dict_display.get('plugin_name', 'N/A')
      command_display_for_log = f"get_volatility_plugin_help: {p_name_help}"

  current_log = state.get("investigation_log", [])
  log_updates = []
  messages_to_replace_or_add = []
  log_entry_base = {"type": "user_decision", "output_file_path": None, "tool_name": original_tool_name_called}

  if action == "approve":
    log_updates.append({
      **log_entry_base,
      "reasoning": f"User approved proposed command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
      "command": command_display_for_log,
      "tool_input": original_tool_args_dict_display,
      "output_details": "Command approved by user."
    })
    # No message changes needed; existing AIMessage with tool_call is used by tool_executor.

  elif action == "modify":
    modified_tool_args_from_user = user_decision.get("modified_tool_args")
    valid_modification = False
    modified_command_display_for_log = "N/A (modification error)"

    if original_tool_name_called == volatility_runner_tool.name:
      if isinstance(modified_tool_args_from_user, dict) and modified_tool_args_from_user.get("plugin_name"):
        valid_modification = True
        mod_p_name = modified_tool_args_from_user.get('plugin_name')
        mod_p_args = modified_tool_args_from_user.get('plugin_args', [])
        modified_command_display_for_log = f"volatility: {mod_p_name} {' '.join(mod_p_args)}"
    elif original_tool_name_called == python_interpreter_tool.name:
      if isinstance(modified_tool_args_from_user, dict) and "code" in modified_tool_args_from_user:
        valid_modification = True
        mod_p_code = modified_tool_args_from_user.get('code', '# No code provided')
        modified_command_display_for_log = f"python_code:\n```python\n{mod_p_code[:200]}{'...' if len(mod_p_code) > 200 else ''}\n```"
    elif original_tool_name_called == list_workspace_files_tool.name:
      if isinstance(modified_tool_args_from_user, dict):
        valid_modification = True
        mod_rel_path = modified_tool_args_from_user.get('relative_path', '.')
        modified_command_display_for_log = f"list_workspace_files: relative_path='{mod_rel_path}'"
    elif original_tool_name_called == get_volatility_plugin_help_tool.name:
      if isinstance(modified_tool_args_from_user, dict) and modified_tool_args_from_user.get("plugin_name"):
        valid_modification = True
        mod_p_name_help = modified_tool_args_from_user.get('plugin_name')
        modified_command_display_for_log = f"get_volatility_plugin_help: {mod_p_name_help}"

    if not valid_modification:
      messages_to_replace_or_add.append(HumanMessage(
        content=f"User attempted to modify the '{original_tool_name_called}' call, but the new arguments were invalid or incomplete. The original proposal will be re-evaluated by the agent."))
      log_updates.append({
        **log_entry_base,
        "reasoning": f"User modification failed validation.\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": f"Attempted Modify Original: {command_display_for_log}",
        "tool_input": modified_tool_args_from_user,
        "output_details": "User modification was invalid. Agent to re-evaluate original proposal with this feedback."
      })
      # To force re-evaluation of the original (now unmodified by user) tool call by agent,
      # we might need to strip tool_calls from the original AIMessage if we don't want tool_executor to run it.
      # Or, the HumanMessage above might be enough for the agent to adjust.
      # For now, let's assume the HumanMessage is enough for the agent to re-plan.
      # If the agent re-proposes the same thing, the user can deny.
      # To be safer and force agent re-plan:
      if ai_message_that_was_reviewed:
        # Clean content for Anthropic format - remove tool_use blocks
        cleaned_content = original_ai_content
        if isinstance(original_ai_content, list):
          # For Anthropic's list format, filter out tool_use blocks
          cleaned_content = []
          for block in original_ai_content:
            if isinstance(block, dict) and block.get("type") in ["text", "thinking"]:
              # Keep only text and thinking blocks
              cleaned_content.append(block)
          # If no text/thinking blocks remain, provide a default
          if not cleaned_content:
            cleaned_content = "User modification invalid. Re-evaluating."
        elif not original_ai_content:
          cleaned_content = "User modification invalid. Re-evaluating."
          
        ai_message_no_tool = AIMessage(
          content=cleaned_content,
          tool_calls=[],  # Strip tool calls
          id=original_ai_message_id,
          response_metadata=original_ai_response_metadata
        )
        messages_to_replace_or_add.insert(0, ai_message_no_tool)  # Replace original AI message


    elif not original_ai_message_id or not ai_message_that_was_reviewed:
      messages_to_replace_or_add.append(
        HumanMessage(
          content="System Error: Modification chosen, but original AI message context not found. Agent to re-evaluate."))
      log_updates.append({**log_entry_base, "output_details": "System error during modification."})
    else:
      new_tool_call_dict = {
        "name": original_tool_name_called,
        "args": modified_tool_args_from_user,
        "id": original_tool_call_id_attr,
        "type": "tool_call"
      }
      modified_ai_message = AIMessage(
        content=original_ai_content if original_ai_content else f"Proceeding with user-modified call to {original_tool_name_called}.",
        tool_calls=[new_tool_call_dict],
        id=original_ai_message_id,
        response_metadata=original_ai_response_metadata
      )
      messages_to_replace_or_add.append(modified_ai_message)
      log_updates.append({
        **log_entry_base,
        "reasoning": f"User modified command.\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": (f"Original: {command_display_for_log}\nModified to: {modified_command_display_for_log}"),
        "tool_input": modified_tool_args_from_user,
        "output_details": "Command modified by user."
      })

  elif action == "deny":
    denial_reason = user_decision.get("reason", "No reason provided.")
    if not original_ai_message_id or not ai_message_that_was_reviewed:
      messages_to_replace_or_add.append(
        HumanMessage(content="System Error: Deny action chosen, original AI context lost. Agent to re-evaluate."))
      log_updates.append({**log_entry_base, "output_details": "System error during denial."})
    else:
      # Clean content for Anthropic format - remove tool_use blocks
      cleaned_content = original_ai_content
      if isinstance(original_ai_content, list):
        # For Anthropic's list format, filter out tool_use blocks
        cleaned_content = []
        for block in original_ai_content:
          if isinstance(block, dict) and block.get("type") in ["text", "thinking"]:
            # Keep only text and thinking blocks
            cleaned_content.append(block)
        # If no text/thinking blocks remain, provide a default
        if not cleaned_content:
          cleaned_content = "Tool call denied by user. Reconsidering."
      elif not original_ai_content:
        cleaned_content = "Tool call denied by user. Reconsidering."
      
      ai_message_no_tool = AIMessage(
        content=cleaned_content,
        tool_calls=[],
        id=original_ai_message_id,
        response_metadata=original_ai_response_metadata
      )
      messages_to_replace_or_add.append(ai_message_no_tool)
      messages_to_replace_or_add.append(HumanMessage(
        content=f"User feedback: The proposed tool call ({original_tool_name_called}) was denied. Reason: {denial_reason}. Please reconsider or propose a different analysis step."))
      log_updates.append({
        **log_entry_base,
        "reasoning": f"User denied command. Reason: {denial_reason}\nOriginal AI Reasoning:\n{extracted_reasoning}",
        "command": f"Denied: {command_display_for_log}",
        "tool_input": original_tool_args_dict_display,
        "output_details": f"Denied by user. Reason: {denial_reason}"
      })

  elif action == "internal_error_at_review":
    err_reason = user_decision.get("reason", "Unknown internal error during review.")
    messages_to_replace_or_add.append(HumanMessage(
      content=f"System: Internal error occurred before human review: {err_reason}. Agent needs to reassess."))
    log_updates.append({
      "type": "internal_error",
      "reasoning": "Internal error in human review step initiation.",
      "command": "N/A (Internal Error)", "output_details": err_reason, "output_file_path": None
    })
  else:  # Unknown action
    unknown_action_reason = f"Error: Invalid action '{action}' received from user decision process."
    messages_to_replace_or_add.append(HumanMessage(content=unknown_action_reason))
    log_updates.append({
      **log_entry_base,
      "reasoning": f"Invalid action '{action}' from user review decision processor.",
      "command": f"Invalid Action during review of: {command_display_for_log}",
      "tool_input": original_tool_args_dict_display, "output_details": unknown_action_reason
    })

  return_dict = {
    "investigation_log": current_log + log_updates,
    "last_user_review_decision": None
  }

  if messages_to_replace_or_add:
    current_msgs_list = list(state.get("messages", []))
    final_message_list = []
    if action == "approve":
      final_message_list = current_msgs_list  # No changes
    elif (action == "modify" or action == "deny") and ai_message_that_was_reviewed and original_ai_message_id:
      temp_message_list = []
      replaced = False
      for msg in current_msgs_list:
        if msg.id == original_ai_message_id:
          temp_message_list.extend(messages_to_replace_or_add)
          replaced = True
        else:
          temp_message_list.append(msg)
      if not replaced:  # Fallback if ID not found
        print(f"Warning: Could not find AIMessage ID {original_ai_message_id} to replace. Appending new messages.")
        final_message_list = current_msgs_list + messages_to_replace_or_add
      else:
        final_message_list = temp_message_list
    else:  # For internal_error, unknown_action, or if context for replace was lost
      final_message_list = current_msgs_list + messages_to_replace_or_add

    return_dict["messages"] = final_message_list

  return return_dict


# Graph definition
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
  if not state.get("profile"): return END  # Should have been handled by agent providing final message
  messages = state.get("messages", [])
  if not messages: return END  # Should not happen

  last_msg = messages[-1]
  if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
    return "human_tool_review_interrupt"
  # If no tool_calls from AI, it implies final summary or unrecoverable error message from AI
  return END


graph_builder.add_conditional_edges("agent", route_after_agent,
                                    {"human_tool_review_interrupt": "human_tool_review_interrupt", END: END})

graph_builder.add_edge("human_tool_review_interrupt", "process_review_decision")


def route_after_processing_review(state: AppState) -> str:
  messages = state.get("messages", [])
  if not messages: return "agent"  # Should ideally not happen; go to agent to re-plan

  # Check the last AIMessage in the potentially modified list of messages.
  # process_human_review_decision_node might have replaced/added messages.
  last_ai_message_with_tool_call = None
  for msg in reversed(messages):
    if isinstance(msg, AIMessage):
      if msg.tool_calls:  # Does this AIMessage *now* have tool calls?
        last_ai_message_with_tool_call = msg
      break  # Found the last AIMessage.
    if isinstance(msg,
                  HumanMessage) and "User feedback" in msg.content or "System Error" in msg.content or "User attempted to modify" in msg.content:
      # If the most recent relevant message is a feedback/error message for the agent, agent should run
      return "agent"

  if last_ai_message_with_tool_call:
    # This implies an 'approve' or a successful 'modify' occurred,
    # and the AIMessage (either original or modified) has tool_calls.
    return "tool_executor"
  else:
    # This implies a 'deny', or an invalid 'modify' where tool_calls were stripped,
    # or an internal error that added a HumanMessage. Agent needs to re-plan.
    return "agent"


graph_builder.add_conditional_edges("process_review_decision", route_after_processing_review,
                                    {"tool_executor": "tool_executor", "agent": "agent"})
graph_builder.add_edge("tool_executor", "agent")  # After tool execution, always go back to agent

checkpointer = MemorySaver()
app_graph = graph_builder.compile(checkpointer=checkpointer)
