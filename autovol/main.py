import os
import uuid
from typing import Optional
from pathlib import Path  # Added import

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Interrupt, Command

# Load .env before other project imports that might need env vars
load_dotenv()

# Attempt to set GOOGLE_APPLICATION_CREDENTIALS from a local key file
# This should be done early, before 'app_graph' from '.agent' is imported if it initializes the LLM globally
VERTEX_KEY_FILENAME = "vertex_key.json"
# Assuming main.py is in autovol/, so project_root is one level up.
# If your script is run from project root, and main.py is in autovol/, this path is correct.
# If main.py is at project root, then project_root = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERTEX_KEY_PATH = PROJECT_ROOT / VERTEX_KEY_FILENAME

if VERTEX_KEY_PATH.exists():
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VERTEX_KEY_PATH)
  print(f"INFO: Set GOOGLE_APPLICATION_CREDENTIALS to: {VERTEX_KEY_PATH}")
elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
  # Only print warning if it's not set by other means (e.g. user's global env)
  print(
    f"Warning: '{VERTEX_KEY_FILENAME}' not found at '{PROJECT_ROOT}' and GOOGLE_APPLICATION_CREDENTIALS is not otherwise set.")
  print("         The application will likely fail to authenticate with Google Cloud Vertex AI.")

from .agent import app_graph, AppState  # Now import agent stuff
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

  # Updated environment variable check
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

  next_stream_input = initial_state_dict
  num_messages_printed_for_cli = len(initial_state_dict.get("messages", []))
  final_full_state_for_report: Optional[AppState] = initial_state_dict

  try:
    while True:
      print(f"\n--- Graph Cycle (Input type: {type(next_stream_input)}) ---")
      if isinstance(next_stream_input, Command):
        print(f"Resuming with Command object.")

      # Make sure app_graph is defined before this loop; it is via import.
      async_gen = app_graph.stream(  # type: ignore
        next_stream_input,
        config=config_for_stream,
        stream_mode="updates"
      )

      interrupted_this_cycle = False
      next_stream_input_after_cycle = None

      for chunk in async_gen:
        print(f"Raw Stream Chunk: {chunk}")

        if not isinstance(chunk, dict):
          print(f"Warning: Stream yielded non-dict chunk: {chunk}")
          continue

        current_graph_state_candidate = app_graph.get_state(config_for_stream)  # type: ignore
        if current_graph_state_candidate:
          final_full_state_for_report = current_graph_state_candidate.values  # type: ignore

        if "__interrupt__" in chunk:
          interrupted_this_cycle = True
          interrupt_tuple = chunk["__interrupt__"]

          if not isinstance(interrupt_tuple, tuple) or not interrupt_tuple:
            print("Error: __interrupt__ value is not a valid tuple.")
            next_stream_input_after_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Malformed interrupt signal from graph"})
            break

          interrupt_object = interrupt_tuple[0]
          if not isinstance(interrupt_object, Interrupt):
            print(f"Error: __interrupt__ payload is not an Interrupt object, but {type(interrupt_object)}.")
            next_stream_input_after_cycle = Command(
              resume={"action": "internal_error_at_review", "reason": "Malformed interrupt payload type from graph"})
            break

          interrupt_payload_from_graph = interrupt_object.value

          print("\n--- Human Input Required ---")
          reasoning_display = interrupt_payload_from_graph.get("reasoning_and_thinking", "No reasoning provided.")
          tool_args_to_review = interrupt_payload_from_graph.get("tool_call_args", {})

          if interrupt_payload_from_graph.get("error_condition"):
            print(f"Error condition from review node: {reasoning_display}")
            user_decision_payload = {"action": "internal_error_at_review", "reason": reasoning_display}
          else:
            print(f"\nLLM Reasoning & Thinking:\n{reasoning_display}\n")
            plugin_name = tool_args_to_review.get('plugin_name', 'N/A')
            plugin_args = tool_args_to_review.get('plugin_args', [])
            typer.echo(f"Proposed command: ", nl=False)
            typer.secho(f"{plugin_name} {' '.join(plugin_args)}", fg=typer.colors.YELLOW)

            while True:
              action_choice = typer.prompt("Approve (a), Modify (m), or Deny (d) the command?").lower()
              if action_choice in ['a', 'm', 'd']:
                break
              typer.echo("Invalid choice. Please enter 'a', 'm', or 'd'.", err=True)

            user_decision_payload = {}
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
          next_stream_input_after_cycle = Command(resume=user_decision_payload)  # type: ignore
          break
        else:
          if not chunk:
            print("Stream yielded an empty update chunk.")
            continue

          node_name_that_ran = list(chunk.keys())[0]

          # final_full_state_for_report was updated via get_state() at the start of chunk processing
          # Ensure final_full_state_for_report is a dict before accessing "messages"
          if final_full_state_for_report and isinstance(final_full_state_for_report,
                                                        dict) and "messages" in final_full_state_for_report:
            all_messages_now = final_full_state_for_report["messages"]
            new_messages_to_print = all_messages_now[num_messages_printed_for_cli:]

            if new_messages_to_print:
              print(f"\n--- New Streamed Output (from node: {node_name_that_ran}) ---")
              for msg in new_messages_to_print:
                if isinstance(msg, AIMessage):
                  ai_content_print = ""
                  if isinstance(msg.content, list):
                    for block_content in msg.content:
                      if isinstance(block_content, dict) and block_content.get("type") == "text":
                        ai_content_print += block_content.get("text", "") + "\n"
                  elif isinstance(msg.content, str):
                    ai_content_print = msg.content
                  print(f"AI: {ai_content_print.strip()}")
                  if msg.tool_calls:
                    print(f"Tool Call Proposed by AI: {msg.tool_calls}")
                elif isinstance(msg, ToolMessage):
                  tool_output_summary = msg.content
                  if len(tool_output_summary) > 300:
                    tool_output_summary = tool_output_summary[:300] + "...\n(Full output in report)"
                  print(f"Tool Result ({msg.name if msg.name else msg.tool_call_id}):\n{tool_output_summary}")
                elif isinstance(msg, HumanMessage):
                  print(f"User/System: {msg.content}")
              num_messages_printed_for_cli = len(all_messages_now)

      next_stream_input = next_stream_input_after_cycle

      if interrupted_this_cycle:
        continue
      else:
        print("\n--- Graph Execution Fully Finished ---")
        raise StopIteration

  except StopIteration:
    print("Graph execution completed normally.")
  except Exception as e:
    print(f"\n--- An error occurred during graph execution ---")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise typer.Exit(code=1)
  finally:
    print("\n--- Analysis Complete (or Terminated) ---")
    if final_full_state_for_report and isinstance(final_full_state_for_report, dict):
      print("Generating final report...")
      final_log = final_full_state_for_report.get("investigation_log", [])
      final_dump_path = final_full_state_for_report.get("dump_path", dump_path)
      final_initial_context = final_full_state_for_report.get("initial_context", context)
      final_profile = final_full_state_for_report.get("profile", "Profile Not Detected")

      final_summary_text = "No final summary message found from AI."
      # Ensure "messages" key exists and is a list before processing
      messages_in_final_state = final_full_state_for_report.get("messages", [])
      if isinstance(messages_in_final_state, list):
        for msg_obj in reversed(messages_in_final_state):
          if isinstance(msg_obj, AIMessage) and not msg_obj.tool_calls:
            if isinstance(msg_obj.content, str):
              final_summary_text = msg_obj.content
              break
            elif isinstance(msg_obj.content, list):
              text_parts = []
              for _content_block in msg_obj.content:
                if isinstance(_content_block, dict) and _content_block.get("type") == "text":
                  text_parts.append(_content_block.get("text", ""))
              final_summary_text = "".join(text_parts).strip()
              if final_summary_text: break

      report_msg = generate_report(
        log=final_log,
        dump_path=final_dump_path,
        initial_context=final_initial_context,
        profile=final_profile,  # type: ignore
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
