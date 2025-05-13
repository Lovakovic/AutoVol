import os
import uuid
from typing import Optional

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Interrupt, Command

load_dotenv()
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
  if not os.getenv("ANTHROPIC_API_KEY"):
    print("\nError: ANTHROPIC_API_KEY environment variable is not set.")
    raise typer.Exit(code=1)

  initial_state_dict: AppState = {
    "messages": [HumanMessage(content=context)],
    "dump_path": dump_path,
    "initial_context": context,
    "profile": None,
    "investigation_log": [],
    "last_user_review_decision": None  # Initialize the new state field
  }

  thread_id = str(uuid.uuid4())
  config_for_stream = {"recursion_limit": 50, "configurable": {"thread_id": thread_id}}

  next_stream_input = initial_state_dict
  num_messages_printed_for_cli = len(initial_state_dict.get("messages", []))
  final_full_state_for_report: Optional[AppState] = initial_state_dict  # Initialize with initial state

  try:
    while True:
      print(f"\n--- Graph Cycle (Input type: {type(next_stream_input)}) ---")
      if isinstance(next_stream_input, Command):  # For logging resume attempts
        print(f"Resuming with Command object.")

      async_gen = app_graph.stream(
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

        # Always update final_full_state_for_report with the latest state after any node runs or before interrupt
        # This ensures we have the most recent state possible for reporting.
        current_graph_state_candidate = app_graph.get_state(config_for_stream)
        if current_graph_state_candidate:  # Ensure get_state returned something
          final_full_state_for_report = current_graph_state_candidate

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

          interrupt_payload_from_graph = interrupt_object.value  # This is what human_tool_review_node sent

          print("\n--- Human Input Required ---")
          reasoning_display = interrupt_payload_from_graph.get("reasoning_and_thinking", "No reasoning provided.")
          tool_args_to_review = interrupt_payload_from_graph.get("tool_call_args", {})

          if interrupt_payload_from_graph.get("error_condition"):  # Custom flag from human_tool_review_node
            print(f"Error condition from review node: {reasoning_display}")
            # Decide how to proceed, e.g., force a denial or specific error action
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

            user_decision_payload = {}  # This is what process_human_review_decision_node will receive
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

          next_stream_input_after_cycle = Command(resume=user_decision_payload)
          break

        else:  # Regular update chunk (not an interrupt)
          if not chunk:
            print("Stream yielded an empty update chunk.")
            continue

          node_name_that_ran = list(chunk.keys())[0]
          # final_full_state_for_report was already updated at the start of the chunk processing

          if final_full_state_for_report and "messages" in final_full_state_for_report:
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
                  if len(tool_output_summary) > 300:  # Simple truncation for CLI
                    tool_output_summary = tool_output_summary[:300] + "...\n(Full output in report)"
                  print(f"Tool Result ({msg.name if msg.name else msg.tool_call_id}):\n{tool_output_summary}")
                elif isinstance(msg, HumanMessage):
                  print(f"User/System: {msg.content}")
              num_messages_printed_for_cli = len(all_messages_now)

      next_stream_input = next_stream_input_after_cycle

      if interrupted_this_cycle:
        # final_full_state_for_report was updated before breaking
        continue
      else:
        # Stream finished without an interrupt this cycle (next_stream_input is None)
        print("\n--- Graph Execution Fully Finished ---")
        # final_full_state_for_report was updated by the last processed chunk
        raise StopIteration

  except StopIteration:
    print("Graph execution completed normally.")
  except Exception as e:
    print(f"\n--- An error occurred during graph execution ---")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    # final_full_state_for_report should have the state right before the error.
    raise typer.Exit(code=1)
  finally:
    print("\n--- Analysis Complete (or Terminated) ---")
    # final_full_state_for_report should hold the last known good state or initial state.
    if final_full_state_for_report and isinstance(final_full_state_for_report, dict):
      print("Generating final report...")
      final_log = final_full_state_for_report.get("investigation_log", [])
      # Ensure all args for generate_report are correctly fetched or defaulted
      final_dump_path = final_full_state_for_report.get("dump_path", dump_path)  # fallback to initial if not in state
      final_initial_context = final_full_state_for_report.get("initial_context", context)
      final_profile = final_full_state_for_report.get("profile", "Profile Not Detected")

      final_summary_text = "No final summary message found from AI."
      if final_full_state_for_report.get("messages"):
        for msg_obj in reversed(final_full_state_for_report.get("messages", [])):
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
        profile=final_profile,
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
