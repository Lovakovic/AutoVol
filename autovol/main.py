import os
import uuid
from typing import Optional, cast, Any, List
from pathlib import Path
from datetime import datetime

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Interrupt, Command

load_dotenv()

VERTEX_KEY_FILENAME = "vertex_key.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERTEX_KEY_PATH = PROJECT_ROOT / VERTEX_KEY_FILENAME

if VERTEX_KEY_PATH.exists():
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VERTEX_KEY_PATH)
  print(f"INFO: Set GOOGLE_APPLICATION_CREDENTIALS to: {VERTEX_KEY_PATH}")
elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
  print(
    f"Warning: '{VERTEX_KEY_FILENAME}' not found at '{PROJECT_ROOT}' and GOOGLE_APPLICATION_CREDENTIALS is not otherwise set.")
  print("         The application will likely fail to authenticate with Google Cloud Vertex AI.")

from .agent import app_graph, AppState  # Keep after env setup
from .reporting import generate_report

app = typer.Typer()


def display_script_snippet(script_content: str, n_lines: int = 5) -> None:
  lines = script_content.splitlines()
  if not lines:
    typer.secho("(Script is empty)", fg=typer.colors.YELLOW)
    return

  if len(lines) <= 2 * n_lines:
    typer.secho("--- Full Script (short) ---", fg=typer.colors.CYAN)
    for line in lines:
      typer.secho(line, fg=typer.colors.GREEN)
  else:
    typer.secho("--- First N lines ---", fg=typer.colors.CYAN)
    for line in lines[:n_lines]:
      typer.secho(line, fg=typer.colors.GREEN)
    typer.secho("...", fg=typer.colors.CYAN)
    typer.secho("--- Last N lines ---", fg=typer.colors.CYAN)
    for line in lines[-n_lines:]:
      typer.secho(line, fg=typer.colors.GREEN)
  typer.secho("--- End Snippet ---", fg=typer.colors.CYAN)


@app.command()
def analyze(
  dump_path: str = typer.Argument(..., help="Path to the memory dump file."),
  context: str = typer.Option(..., "--context", "-c", help="Initial context or suspicion for the analysis."),
  snippet_lines: int = typer.Option(5, "--snippet-lines", "-sl",
                                    help="Number of lines for top/bottom snippet of Python scripts."),
):
  print(f"--- Starting AutoVol Analysis ---")
  print(f"Dump File: {dump_path}")
  print(f"Initial Context: {context}")
  print(f"Python script snippet lines: {snippet_lines}")

  if not os.path.exists(dump_path):
    # This check is important, especially for the dump_path coming from user / docker volume
    print(f"\nError: Memory dump file not found at '{dump_path}'")
    raise typer.Exit(code=1)
  
  # Removed VOLATILITY3_PATH check, as volatility_runner.py now defaults to 'vol' from PATH
  # if not os.getenv("VOLATILITY3_PATH"):
  #   print("\nError: VOLATILITY3_PATH environment variable is not set.")
  #   raise typer.Exit(code=1)
  
  if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    # This check is important for Vertex AI authentication
    print("\nError: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    print(f"       Please ensure it points to your Google Cloud service account JSON key file,")
    print(f"       or place '{VERTEX_KEY_FILENAME}' in the project root directory ('{PROJECT_ROOT}').")
    print(f"       In Docker, this should be mounted to '/gcloud_key/vertex_key.json' and ENV var set.")
    raise typer.Exit(code=1)

  timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  safe_dump_stem = "".join(c if c.isalnum() else '_' for c in Path(dump_path).stem)
  report_session_id = f"autovol_report_{safe_dump_stem}_{timestamp_str}"

  report_session_base_dir = Path("reports") / report_session_id
  try:
    report_session_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory created at: {report_session_base_dir}")
    (report_session_base_dir / "workspace").mkdir(parents=True, exist_ok=True)
    print(f"Session workspace directory ensured at: {report_session_base_dir / 'workspace'}")
  except Exception as e:
    print(f"Error creating session directory '{report_session_base_dir}': {e}")
    raise typer.Exit(code=1)

  initial_state_dict: AppState = {
    "messages": [HumanMessage(content=context)],
    "dump_path": dump_path,
    "initial_context": context,
    "profile": None,
    "available_plugins": None,
    "investigation_log": [],
    "last_user_review_decision": None,
    "report_session_id": report_session_id,
    "image_analysis_history": [],
    "multimodal_context": {},
    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
  }

  thread_id = str(uuid.uuid4())
  config_for_stream = {"recursion_limit": 50, "configurable": {"thread_id": thread_id}}

  current_input_for_graph_stream: Any = initial_state_dict
  num_messages_printed_for_cli = len(initial_state_dict.get("messages", []))
  final_full_state_for_report: Optional[AppState] = initial_state_dict

  ai_content_was_streamed_for_current_ai_message = False
  id_of_ai_message_content_streamed: Optional[str] = None

  try:
    while True:
      if not isinstance(current_input_for_graph_stream, Command):
        print(f"\n--- Agent Reasoning Cycle ---")

      should_continue_while_loop_after_this_cycle = False
      input_for_next_graph_cycle = None
      
      async_gen = app_graph.stream(
        current_input_for_graph_stream,
        config=config_for_stream,
        stream_mode=["updates", "messages"]
      )

      for i, raw_chunk_from_graph in enumerate(async_gen):
        current_graph_state_candidate = app_graph.get_state(config_for_stream)
        if current_graph_state_candidate:
          final_full_state_for_report = cast(AppState, current_graph_state_candidate.values)

        is_interrupt_chunk = False
        interrupt_data_payload = None

        if isinstance(raw_chunk_from_graph, dict) and "__interrupt__" in raw_chunk_from_graph:
          is_interrupt_chunk = True
          interrupt_data_payload = raw_chunk_from_graph["__interrupt__"]
        elif isinstance(raw_chunk_from_graph, tuple) and len(raw_chunk_from_graph) == 2:
          stream_mode_name_for_interrupt_check, chunk_data_for_interrupt_check = raw_chunk_from_graph
          if stream_mode_name_for_interrupt_check == "updates" and \
            isinstance(chunk_data_for_interrupt_check, dict) and \
            "__interrupt__" in chunk_data_for_interrupt_check:
            is_interrupt_chunk = True
            interrupt_data_payload = chunk_data_for_interrupt_check["__interrupt__"]

        if is_interrupt_chunk:
          should_continue_while_loop_after_this_cycle = True
          if ai_content_was_streamed_for_current_ai_message:
            print()
            ai_content_was_streamed_for_current_ai_message = False
            id_of_ai_message_content_streamed = None

          if interrupt_data_payload is None:
            print("CRITICAL ERROR: is_interrupt_chunk is true but interrupt_data_payload is None.")
            input_for_next_graph_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Internal error processing interrupt structure."}
            )
            break

          if not isinstance(interrupt_data_payload, tuple) or not interrupt_data_payload:
            print("Error: __interrupt__ value is not a valid tuple.")
            input_for_next_graph_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Malformed interrupt signal from graph"})
            break

          interrupt_object = interrupt_data_payload[0]
          if not isinstance(interrupt_object, Interrupt):
            print(f"Error: __interrupt__ payload is not an Interrupt object, but {type(interrupt_object)}.")
            input_for_next_graph_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Malformed interrupt payload type from graph"})
            break

          interrupt_payload_from_graph_node = interrupt_object.value
          print("\n\n--- Agent Proposes Action: Human Input Required ---")

          tool_name_to_review = interrupt_payload_from_graph_node.get("tool_name", "unknown_tool")
          tool_args_to_review = interrupt_payload_from_graph_node.get("tool_call_args", {})
          temp_script_path_for_review = interrupt_payload_from_graph_node.get("temp_script_path")
          user_decision_payload = {}

          if interrupt_payload_from_graph_node.get("error_condition"):
            error_reasoning_display = interrupt_payload_from_graph_node.get("reasoning_and_thinking",
                                                                            "Error: No specific reason provided for error condition.")
            print(f"Error condition from review node: {error_reasoning_display}")
            user_decision_payload = {"action": "internal_error_at_review", "reason": error_reasoning_display}
          else:
            if tool_name_to_review == "volatility_runner":
              plugin_name = tool_args_to_review.get('plugin_name', 'N/A')
              plugin_args = tool_args_to_review.get('plugin_args', [])
              # Ensure plugin_args is a list, not None
              if plugin_args is None:
                plugin_args = []
              print()
              typer.echo(f"Proposed Volatility command: ", nl=False)
              typer.secho(f"{plugin_name} {' '.join(plugin_args)}", fg=typer.colors.YELLOW)
              typer.echo("(This will run in the session workspace. If it outputs files, they will appear there.)")

            elif tool_name_to_review == "python_interpreter":
              print()
              typer.secho(f"Proposed Python code to execute (will run in session workspace):", fg=typer.colors.YELLOW)
              original_python_code_from_llm = tool_args_to_review.get('code', '# No code provided by LLM')
              if temp_script_path_for_review and Path(temp_script_path_for_review).exists():
                typer.echo(f"Script saved for review at: ", nl=False)
                typer.secho(temp_script_path_for_review, fg=typer.colors.BLUE, underline=True)
                try:
                  with open(temp_script_path_for_review, "r", encoding="utf-8") as f_script:
                    script_content_for_snippet = f_script.read()
                  display_script_snippet(script_content_for_snippet, n_lines=snippet_lines)
                except Exception as e_read:
                  typer.secho(f" (Could not read script from temp file for snippet: {e_read})", fg=typer.colors.RED)
                  display_script_snippet(original_python_code_from_llm, n_lines=snippet_lines)
              else:
                typer.secho(
                  "(Script not saved to temp file or path unavailable, showing inline snippet from LLM proposal)",
                  fg=typer.colors.YELLOW)
                display_script_snippet(original_python_code_from_llm, n_lines=snippet_lines)
            
            elif tool_name_to_review == "list_workspace_files":
                relative_path_arg = tool_args_to_review.get('relative_path', '.')
                print()
                typer.echo(f"Proposed action: List files in session workspace", nl=False)
                if relative_path_arg != '.':
                    typer.echo(f" (specifically path: ", nl=False)
                    typer.secho(relative_path_arg, fg=typer.colors.YELLOW, nl=False)
                    typer.echo(")")
                else:
                    typer.echo(" (root)")
            else:
              typer.echo(f"Proposed action for tool '{tool_name_to_review}': ", nl=False)
              typer.secho(str(tool_args_to_review), fg=typer.colors.YELLOW)

            while True:
              action_choice = typer.prompt("Approve (a), Modify (m), or Deny (d) the action?").lower()
              if action_choice in ['a', 'm', 'd']:
                break
              typer.echo("Invalid choice. Please enter 'a', 'm', or 'd'.", err=True)

            if action_choice == 'a':
              user_decision_payload = {"action": "approve"}
            elif action_choice == 'd':
              denial_reason = typer.prompt("Reason for denial (optional, press Enter for default):",
                                           default="User denied without specific reason.", show_default=False)
              user_decision_payload = {"action": "deny", "reason": denial_reason}
            elif action_choice == 'm':
              modified_args_for_payload = {}
              if tool_name_to_review == "volatility_runner":
                current_plugin_name = tool_args_to_review.get('plugin_name', 'N/A')
                current_plugin_args = tool_args_to_review.get('plugin_args', [])
                modified_plugin_name = typer.prompt(f"New Volatility plugin name (current: {current_plugin_name})",
                                                    default=current_plugin_name)
                current_args_str = ' '.join(current_plugin_args)
                modified_args_str = typer.prompt(f"New plugin args (space-separated, current: '{current_args_str}')",
                                                 default=current_args_str)
                modified_plugin_args_list = modified_args_str.split() if modified_args_str else []
                modified_args_for_payload = {
                  "plugin_name": modified_plugin_name,
                  "plugin_args": modified_plugin_args_list
                }
              elif tool_name_to_review == "python_interpreter":
                code_to_edit = original_python_code_from_llm
                if temp_script_path_for_review and Path(temp_script_path_for_review).exists():
                  try:
                    with open(temp_script_path_for_review, "r", encoding="utf-8") as f_edit:
                      code_to_edit = f_edit.read()
                  except Exception:
                    typer.secho(
                      f"Warning: Could not re-read {temp_script_path_for_review} for editing, using LLM's original code.",
                      fg=typer.colors.YELLOW)
                edited_code_content = typer.edit(text=code_to_edit, extension=".py")
                if edited_code_content is None:
                  typer.secho("Editor closed or failed. Using original code for modification attempt.",
                              fg=typer.colors.YELLOW)
                  edited_code_content = code_to_edit
                elif not edited_code_content.strip() and code_to_edit.strip():
                  typer.secho("Editor returned empty content. Please provide some code or deny.", fg=typer.colors.RED)
                  edited_code_content = code_to_edit
                modified_args_for_payload = {"code": edited_code_content}
              elif tool_name_to_review == "list_workspace_files":
                current_rel_path = tool_args_to_review.get('relative_path', '.')
                modified_rel_path = typer.prompt(f"New relative path for listing (current: '{current_rel_path}')", default=current_rel_path)
                modified_args_for_payload = {"relative_path": modified_rel_path}
              else:
                typer.echo(
                  f"Modification for tool '{tool_name_to_review}' using generic JSON edit (current: {tool_args_to_review}).",
                  err=True)
                new_args_str = typer.prompt("Enter new JSON args string:", default=str(tool_args_to_review))
                try:
                  modified_args_for_payload = eval(new_args_str)
                except:
                  typer.echo("Invalid args format. Reverting to original.", err=True)
                  modified_args_for_payload = tool_args_to_review
              user_decision_payload = {
                "action": "modify",
                "modified_tool_args": modified_args_for_payload
              }
          input_for_next_graph_cycle = Command(resume=user_decision_payload)
          break

        elif isinstance(raw_chunk_from_graph, tuple) and len(raw_chunk_from_graph) == 2:
          stream_mode_name, chunk_data = raw_chunk_from_graph
          if stream_mode_name == "messages":
            message_chunk, metadata = cast(tuple[BaseMessage, dict], chunk_data)
            if isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'content') and message_chunk.content:
              # Extract text content from the message chunk
              text_to_stream = ""
              if isinstance(message_chunk.content, list):
                # Handle Anthropic's list format with content blocks
                for block in message_chunk.content:
                  if isinstance(block, dict):
                    # Only stream text blocks, not thinking blocks
                    if block.get("type") == "text" and block.get("text"):
                      text_to_stream += block.get("text", "")
              elif isinstance(message_chunk.content, str):
                # Handle simple string content (Gemini/OpenAI format)
                text_to_stream = message_chunk.content
              
              if text_to_stream:
                if not ai_content_was_streamed_for_current_ai_message:
                  print(f"AI: ", end="")
                  id_of_ai_message_content_streamed = message_chunk.id
                print(text_to_stream, end="", flush=True)
                ai_content_was_streamed_for_current_ai_message = True

          elif stream_mode_name == "updates":
            if not chunk_data:
              continue
            if final_full_state_for_report and "messages" in final_full_state_for_report:
              all_messages_now = final_full_state_for_report["messages"]
              new_messages_to_print = all_messages_now[num_messages_printed_for_cli:]

              if new_messages_to_print:
                node_name_that_just_ran = list(chunk_data.keys())[0]
                if ai_content_was_streamed_for_current_ai_message and \
                  node_name_that_just_ran == "agent" and \
                  new_messages_to_print and \
                  isinstance(new_messages_to_print[0], AIMessage) and \
                  new_messages_to_print[0].id == id_of_ai_message_content_streamed:
                  print()

                for msg_idx, msg_to_print in enumerate(new_messages_to_print):
                  if isinstance(msg_to_print, AIMessage):
                    content_already_streamed = (
                      ai_content_was_streamed_for_current_ai_message and
                      msg_to_print.id == id_of_ai_message_content_streamed
                    )
                    if content_already_streamed:
                      ai_content_was_streamed_for_current_ai_message = False
                      id_of_ai_message_content_streamed = None
                    else:
                      ai_text_content = ""
                      if isinstance(msg_to_print.content, list):
                        for block in msg_to_print.content:
                          if isinstance(block, dict) and block.get("type") == "text":
                            ai_text_content += block.get("text", "") + "\n"
                      elif isinstance(msg_to_print.content, str):
                        ai_text_content = msg_to_print.content
                      if ai_text_content.strip():
                        print(f"AI: {ai_text_content.strip()}")

                    if msg_to_print.tool_calls:
                      tool_call_summary_parts = []
                      for tc in msg_to_print.tool_calls:
                        tc_tool_name = tc['name']
                        tc_args_dict = tc['args']
                        summary_detail = f"Tool: {tc_tool_name}"
                        if tc_tool_name == "volatility_runner":
                          tc_plugin = tc_args_dict.get('plugin_name', 'N/A')
                          tc_plugin_args_list = tc_args_dict.get('plugin_args', [])
                          tc_plugin_args_str = ' '.join(tc_plugin_args_list) if tc_plugin_args_list else ""
                          summary_detail = f"Volatility: {tc_plugin} {tc_plugin_args_str}".strip()
                        elif tc_tool_name == "python_interpreter":
                          summary_detail = "Python Code (details in log/review)"
                        elif tc_tool_name == "list_workspace_files":
                            rel_path = tc_args_dict.get('relative_path', '.')
                            summary_detail = f"List Workspace Files (path: '{rel_path}')"
                        tool_call_summary_parts.append(summary_detail)

                      if not content_already_streamed or not msg_to_print.content or not str(
                        msg_to_print.content).strip():
                        print()
                      print(f"Tool Call Proposed: {', '.join(tool_call_summary_parts)}")
                      if not content_already_streamed or not msg_to_print.content or not str(
                        msg_to_print.content).strip():
                        print()

                  elif isinstance(msg_to_print, ToolMessage):
                    print()
                    tool_output_for_cli = msg_to_print.content
                    max_tool_output_cli = 700
                    tool_name_for_display = msg_to_print.name if msg_to_print.name else msg_to_print.tool_call_id
                    if len(tool_output_for_cli) > max_tool_output_cli:
                      tool_output_for_cli = tool_output_for_cli[:max_tool_output_cli] + \
                                            f"...\n(Full output for '{tool_name_for_display}' processed by agent)"
                    print(f"Tool Result ({tool_name_for_display}):\n{tool_output_for_cli}")
                    print()
                  elif isinstance(msg_to_print, HumanMessage):
                    if num_messages_printed_for_cli > 0 or msg_to_print is not all_messages_now[0]:
                      print(f"System/User Feedback: {msg_to_print.content}")
                num_messages_printed_for_cli = len(all_messages_now)
          else:
            print(f"Warning: Received chunk for unhandled stream mode: {stream_mode_name} with data: {chunk_data}")
        else:
          print(f"Warning: Stream yielded an unrecognized chunk format: {raw_chunk_from_graph}")

      if should_continue_while_loop_after_this_cycle:
        if input_for_next_graph_cycle is None:
          print("CRITICAL ERROR: Interrupt was signaled, but no input_for_next_graph_cycle was set. Terminating.")
          break
        current_input_for_graph_stream = input_for_next_graph_cycle
        continue
      else:
        if ai_content_was_streamed_for_current_ai_message:
          print()
          ai_content_was_streamed_for_current_ai_message = False
          id_of_ai_message_content_streamed = None
        print("\n--- Analysis Concluded by Agent (or max recursion) ---")
        break

  except Exception as e:
    print(f"\n--- An unexpected error occurred during agent execution ---")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

  finally:
    if ai_content_was_streamed_for_current_ai_message:
      print()
    print("\n--- AutoVol Session Ended ---")
    if final_full_state_for_report and isinstance(final_full_state_for_report, dict):
      print("Generating final report...")
      final_log = final_full_state_for_report.get("investigation_log", [])
      final_dump_path = final_full_state_for_report.get("dump_path", dump_path)
      final_initial_context = final_full_state_for_report.get("initial_context", context)
      final_profile = final_full_state_for_report.get("profile", "Profile Not Detected")
      session_id_for_report = final_full_state_for_report.get("report_session_id", report_session_id)

      final_summary_text = "No final summary message found from AI."
      messages_in_final_state = final_full_state_for_report.get("messages", [])
      if isinstance(messages_in_final_state, list):
        for msg_obj in reversed(messages_in_final_state):
          if isinstance(msg_obj, AIMessage) and not msg_obj.tool_calls:
            current_msg_text_parts = []
            if isinstance(msg_obj.content, str):
              current_msg_text_parts.append(msg_obj.content)
            elif isinstance(msg_obj.content, list):
              for _content_block in msg_obj.content:
                if isinstance(_content_block, dict) and _content_block.get("type") == "text":
                  current_msg_text_parts.append(_content_block.get("text", ""))
            assembled_text = "".join(current_msg_text_parts).strip()
            if assembled_text:
              final_summary_text = assembled_text
              break

      final_token_usage = final_full_state_for_report.get("token_usage", None)
      
      report_msg = generate_report(
        log=final_log,
        dump_path=final_dump_path,
        initial_context=final_initial_context,
        profile=str(final_profile),
        final_summary=final_summary_text,
        report_session_id=session_id_for_report,
        token_usage=final_token_usage
      )
      print(report_msg)
    else:
      print("Warning: Could not generate final report due to missing or invalid final state.")
      if final_full_state_for_report is None:
        print("Reason: final_full_state_for_report was None.")
      elif not isinstance(final_full_state_for_report, dict):
        print(f"Reason: final_full_state_for_report was not a dictionary, type: {type(final_full_state_for_report)}")

if __name__ == "__main__":
  app()
