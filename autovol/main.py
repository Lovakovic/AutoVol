import os
import uuid
from typing import Optional, cast, Any
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Interrupt, Command

# Load .env before other project imports that might need env vars
load_dotenv()

# Attempt to set GOOGLE_APPLICATION_CREDENTIALS from a local key file
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

from .agent import app_graph, AppState
from .reporting import generate_report

app = typer.Typer()


@app.command()
def analyze(
  dump_path: str = typer.Argument(..., help="Path to the memory dump file."),
  context: str = typer.Option(..., "--context", "-c", help="Initial context or suspicion for the analysis."),
):
  print(f"--- Starting AutoVol Analysis ---")
  print(f"Dump File: {dump_path}")
  print(f"Initial Context: {context}")

  if not os.path.exists(dump_path):
    print(f"\nError: Memory dump file not found at '{dump_path}'")
    raise typer.Exit(code=1)
  if not os.getenv("VOLATILITY3_PATH"):
    print("\nError: VOLATILITY3_PATH environment variable is not set.")
    raise typer.Exit(code=1)

  if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("\nError: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    print(f"       Please ensure it points to your Google Cloud service account JSON key file,")
    print(f"       or place '{VERTEX_KEY_FILENAME}' in the project root directory ('{PROJECT_ROOT}').")
    raise typer.Exit(code=1)

  initial_state_dict: AppState = {
    "messages": [HumanMessage(content=context)],
    "dump_path": dump_path,
    "initial_context": context,
    "profile": None,
    "investigation_log": [],
    "last_user_review_decision": None
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

          interrupt_tuple_from_graph = interrupt_data_payload

          if not isinstance(interrupt_tuple_from_graph, tuple) or not interrupt_tuple_from_graph:
            print("Error: __interrupt__ value is not a valid tuple.")
            input_for_next_graph_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Malformed interrupt signal from graph"})
            break

          interrupt_object = interrupt_tuple_from_graph[0]
          if not isinstance(interrupt_object, Interrupt):
            print(f"Error: __interrupt__ payload is not an Interrupt object, but {type(interrupt_object)}.")
            input_for_next_graph_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Malformed interrupt payload type from graph"})
            break

          interrupt_payload_from_graph_value = interrupt_object.value
          # MODIFIED: Changed section header
          print("\n\n--- Agent Proposes Command: Human Input Required ---")

          # REMOVED: Redundant reasoning print, as it was just streamed.
          # reasoning_display = interrupt_payload_from_graph_value.get("reasoning_and_thinking", "No reasoning provided.")
          # print(f"\nLLM Reasoning for this proposal:\n{reasoning_display}\n") # REMOVED

          tool_args_to_review = interrupt_payload_from_graph_value.get("tool_call_args", {})

          user_decision_payload = {}
          if interrupt_payload_from_graph_value.get("error_condition"):
            # If there's an error condition, we might want to display the reasoning/error.
            error_reasoning_display = interrupt_payload_from_graph_value.get("reasoning_and_thinking",
                                                                             "Error: No specific reason provided for error condition.")
            print(f"Error condition from review node: {error_reasoning_display}")
            user_decision_payload = {"action": "internal_error_at_review", "reason": error_reasoning_display}
          else:
            plugin_name = tool_args_to_review.get('plugin_name', 'N/A')
            plugin_args = tool_args_to_review.get('plugin_args', [])
            # Add a newline before the proposed command for clarity
            print()
            typer.echo(f"Proposed command: ", nl=False)
            typer.secho(f"{plugin_name} {' '.join(plugin_args)}", fg=typer.colors.YELLOW)

            while True:
              action_choice = typer.prompt("Approve (a), Modify (m), or Deny (d) the command?").lower()
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
              modified_plugin_name = typer.prompt(f"New plugin name (current: {plugin_name})", default=plugin_name)
              current_args_str = ' '.join(plugin_args)
              modified_args_str = typer.prompt(f"New plugin args (space-separated, current: '{current_args_str}')",
                                               default=current_args_str)
              modified_plugin_args = modified_args_str.split() if modified_args_str else []
              user_decision_payload = {
                "action": "modify",
                "modified_tool_args": {
                  "plugin_name": modified_plugin_name,
                  "plugin_args": modified_plugin_args
                }
              }
          input_for_next_graph_cycle = Command(resume=user_decision_payload)
          break

        elif isinstance(raw_chunk_from_graph, tuple) and len(raw_chunk_from_graph) == 2:
          stream_mode_name, chunk_data = raw_chunk_from_graph

          if stream_mode_name == "messages":
            message_chunk, metadata = cast(tuple[BaseMessage, dict], chunk_data)
            if isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'content') and message_chunk.content:
              if not ai_content_was_streamed_for_current_ai_message:
                print(f"AI: ", end="")
                id_of_ai_message_content_streamed = message_chunk.id
              print(message_chunk.content, end="", flush=True)
              ai_content_was_streamed_for_current_ai_message = True

          elif stream_mode_name == "updates":
            if not chunk_data:
              continue

            node_name_that_ran = list(chunk_data.keys())[0]

            if final_full_state_for_report and "messages" in final_full_state_for_report:
              all_messages_now = final_full_state_for_report["messages"]
              new_messages_to_print = all_messages_now[num_messages_printed_for_cli:]

              if new_messages_to_print:
                if ai_content_was_streamed_for_current_ai_message and \
                  node_name_that_ran == "agent" and \
                  new_messages_to_print and \
                  isinstance(new_messages_to_print[0], AIMessage) and \
                  new_messages_to_print[0].id == id_of_ai_message_content_streamed:
                  print()

                for msg_idx, msg in enumerate(new_messages_to_print):
                  if isinstance(msg, AIMessage):
                    content_already_streamed_for_this_msg = (
                      ai_content_was_streamed_for_current_ai_message and
                      msg.id == id_of_ai_message_content_streamed
                    )

                    if content_already_streamed_for_this_msg:
                      ai_content_was_streamed_for_current_ai_message = False
                      id_of_ai_message_content_streamed = None
                    else:
                      ai_content_print = ""
                      if isinstance(msg.content, list):
                        for block_content in msg.content:
                          if isinstance(block_content, dict) and block_content.get("type") == "text":
                            ai_content_print += block_content.get("text", "") + "\n"
                      elif isinstance(msg.content, str):
                        ai_content_print = msg.content
                      if ai_content_print.strip():
                        print(f"AI: {ai_content_print.strip()}")

                    if msg.tool_calls:
                      tool_call_summary_parts = []
                      for tc in msg.tool_calls:
                        tc_plugin = tc['args'].get('plugin_name', 'N/A')
                        tc_plugin_args_list = tc['args'].get('plugin_args', [])
                        tc_plugin_args_str = ' '.join(tc_plugin_args_list) if tc_plugin_args_list else ""
                        tool_call_summary_parts.append(f"{tc_plugin} {tc_plugin_args_str}".strip())
                      # Added newline before tool call proposal if content wasn't streamed or if it was but content was empty.
                      # If content was streamed, a newline should already be there.
                      if not content_already_streamed_for_this_msg or not msg.content:
                        print()
                      print(f"Tool Call Proposed: {', '.join(tool_call_summary_parts)}")
                      print()  # Add a newline after the tool call proposal

                  elif isinstance(msg, ToolMessage):
                    print()
                    tool_output_summary = msg.content
                    if len(tool_output_summary) > 300:
                      tool_output_summary = tool_output_summary[:300] + "...\n(Full output in report)"
                    print(f"Tool Result ({msg.name if msg.name else msg.tool_call_id}):\n{tool_output_summary}")
                    print()
                  elif isinstance(msg, HumanMessage):
                    if num_messages_printed_for_cli > 0 or msg is not all_messages_now[0]:
                      print(f"System/User Feedback: {msg.content}")
                num_messages_printed_for_cli = len(all_messages_now)
          else:
            print(f"Warning: Received chunk for unhandled stream mode: {stream_mode_name} with data: {chunk_data}")
        else:
          print(f"Warning: Stream yielded an unrecognized chunk format: {raw_chunk_from_graph}")

      if should_continue_while_loop_after_this_cycle:
        if input_for_next_graph_cycle is None:
          print("CRITICAL ERROR: Interrupt was signaled, but no input_for_next_graph_cycle was set. Terminating.")
          raise StopIteration("Interrupt handling logical error")
        current_input_for_graph_stream = input_for_next_graph_cycle
        continue
      else:
        if ai_content_was_streamed_for_current_ai_message:
          print()
          ai_content_was_streamed_for_current_ai_message = False
          id_of_ai_message_content_streamed = None
        print("\n--- Analysis Concluded by Agent ---")
        raise StopIteration

  except StopIteration:
    print("Agent interaction cycle completed.")
  except Exception as e:
    print(f"\n--- An error occurred during graph execution ---")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise typer.Exit(code=1)
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

      report_msg = generate_report(
        log=final_log,
        dump_path=final_dump_path,
        initial_context=final_initial_context,
        profile=str(final_profile),
        final_summary=final_summary_text
      )
      print(report_msg)
    else:
      print("Warning: Could not generate final report due to missing or invalid state at the very end.")
      if final_full_state_for_report is None:
        print("Reason: final_full_state_for_report was None.")
      elif not isinstance(final_full_state_for_report, dict):
        print(f"Reason: final_full_state_for_report was not a dictionary, type: {type(final_full_state_for_report)}")


if __name__ == "__main__":
  app()
