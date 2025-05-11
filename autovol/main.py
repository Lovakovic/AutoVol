import os
from datetime import datetime
from pathlib import Path
import uuid  # For thread_id
from typing import Optional

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Interrupt

# Import the graph AFTER loading env vars
load_dotenv()
from .agent import app_graph, AppState

# Initialize Typer app
app = typer.Typer()


# --- generate_report function (remains the same) ---
def generate_report(log: list, dump_path: str, initial_context: str, profile: str, final_summary: str = "") -> str:
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  report_filename = f"autovol_report_{Path(dump_path).stem}_{timestamp}.md"
  report_path = Path("reports") / report_filename
  report_path.parent.mkdir(parents=True, exist_ok=True)

  report_content = f"# AutoVol Analysis Report\n\n"
  report_content += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
  report_content += f"**Memory Dump:** `{dump_path}`\n"
  report_content += f"**Detected Profile Base:** `{profile if profile else 'Not Detected'}`\n"
  report_content += f"**Initial Context/Suspicion:**\n```\n{initial_context}\n```\n\n"

  if final_summary:
    report_content += f"## Final LLM Summary:\n\n```text\n{final_summary}\n```\n\n"

  report_content += "---\n\n## Investigation Steps\n\n"

  if not log:
    report_content += "No investigation steps were logged.\n"
  else:
    for i, entry in enumerate(log):
      report_content += f"### Step {i + 1}\n\n"
      reasoning = entry.get('reasoning', 'N/A')
      if not isinstance(reasoning, str):
        reasoning = str(reasoning)

      report_content += f"**Reasoning/Action (LLM/User):**\n```\n{reasoning}\n```\n\n"
      report_content += f"**Executed Command/Action:**\n```bash\n{entry.get('command', 'N/A')}\n```\n\n"
      report_content += f"**Output/Result:**\n"
      report_content += f"<details>\n<summary>Click to view output/result</summary>\n\n```text\n{entry.get('output', 'N/A')}\n```\n\n</details>\n\n"
      report_content += "---\n\n"

  try:
    with open(report_path, "w", encoding='utf-8') as f:
      f.write(report_content)
    return f"Report generated successfully at: {report_path}"
  except Exception as e:
    return f"Error generating report: {e}"


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
    "investigation_log": []
  }

  thread_id = str(uuid.uuid4())
  config_for_stream = {"recursion_limit": 50, "configurable": {"thread_id": thread_id}}

  next_stream_input = initial_state_dict
  num_messages_printed_for_cli = len(initial_state_dict["messages"])
  final_full_state_for_report: Optional[AppState] = None

  try:
    while True:
      print(f"\n--- Graph Cycle (Input type: {type(next_stream_input)}) ---")

      async_gen = app_graph.stream(
        next_stream_input,
        config=config_for_stream,
        stream_mode="updates"
      )

      interrupted_this_cycle = False
      next_stream_input = None

      for chunk in async_gen:
        print(f"Raw Stream Chunk: {chunk}")

        if not isinstance(chunk, dict):
          print(f"Warning: Stream yielded non-dict chunk: {chunk}")
          continue

        if "__interrupt__" in chunk:
          interrupted_this_cycle = True
          interrupt_tuple = chunk["__interrupt__"]
          if not isinstance(interrupt_tuple, tuple) or not interrupt_tuple:
            print("Error: __interrupt__ value is not a valid tuple.")
            next_stream_input = {"action": "error", "reason": "Malformed interrupt signal"}
            break

          interrupt_object = interrupt_tuple[0]
          if not isinstance(interrupt_object, Interrupt):  # Check using the imported Interrupt class
            print(f"Error: __interrupt__ payload is not an Interrupt object, but {type(interrupt_object)}.")
            next_stream_input = {"action": "error", "reason": "Malformed interrupt payload type"}
            break

          interrupt_payload = interrupt_object.value

          print("\n--- Human Input Required ---")

          reasoning_display = interrupt_payload.get("reasoning_and_thinking", "No reasoning provided.")
          tool_args_to_review = interrupt_payload.get("tool_call_args", {})
          current_profile = interrupt_payload.get("profile", "N/A")

          print(f"\nLLM Reasoning & Thinking:\n{reasoning_display}\n")

          plugin_name = tool_args_to_review.get('plugin_name', 'N/A')
          plugin_args = tool_args_to_review.get('plugin_args', [])
          typer.echo(f"Proposed command: ", nl=False)
          typer.secho(f"{current_profile}.{plugin_name} {' '.join(plugin_args)}", fg=typer.colors.YELLOW)

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

          next_stream_input = user_decision_payload
          break

        else:
          if not chunk:
            print("Stream yielded an empty update chunk.")
            continue

          node_name_that_ran = list(chunk.keys())[0]

          final_full_state_for_report = app_graph.get_state(config_for_stream)

          if final_full_state_for_report and "messages" in final_full_state_for_report:
            all_messages_now = final_full_state_for_report["messages"]
            new_messages_to_print = all_messages_now[num_messages_printed_for_cli:]

            if new_messages_to_print:
              print(f"\n--- New Streamed Output (from node: {node_name_that_ran}) ---")
              for msg in new_messages_to_print:
                if isinstance(msg, AIMessage):
                  ai_content_print = ""
                  if isinstance(msg.content, list):
                    for block in msg.content:
                      if block.get("type") == "text":
                        ai_content_print += block.get("text", "") + "\n"
                  elif isinstance(msg.content, str):
                    ai_content_print = msg.content
                  print(f"AI: {ai_content_print.strip()}")
                  if msg.tool_calls:
                    print(f"Tool Call Proposed by AI: {msg.tool_calls}")
                elif isinstance(msg, ToolMessage):
                  tool_output_summary = msg.content
                  if len(tool_output_summary) > 300:
                    tool_output_summary = tool_output_summary[:300] + "...\n(Full output in report)"
                  print(f"Tool Result ({msg.tool_call_id}):\n{tool_output_summary}")
                elif isinstance(msg, HumanMessage):
                  print(f"User/System: {msg.content}")
              num_messages_printed_for_cli = len(all_messages_now)

      if interrupted_this_cycle:
        continue
      else:
        print("\n--- Graph Execution Fully Finished ---")
        final_full_state_for_report = app_graph.get_state(config_for_stream)
        raise StopIteration

  except StopIteration:
    pass
  except Exception as e:
    print(f"\n--- An error occurred during graph execution ---")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

    try:
      final_full_state_for_report = app_graph.get_state(config_for_stream)
    except Exception as get_state_err:
      print(f"Could not retrieve state for partial report after error: {get_state_err}")
      if final_full_state_for_report is None:
        final_full_state_for_report = initial_state_dict

    if final_full_state_for_report and isinstance(final_full_state_for_report, dict):
      print("\nAttempting to generate partial report...")
      report_msg = generate_report(
        final_full_state_for_report.get("investigation_log", []),
        final_full_state_for_report.get("dump_path", dump_path),
        final_full_state_for_report.get("initial_context", context),
        final_full_state_for_report.get("profile", "Unknown")
      )
      print(report_msg)
    else:
      print("Could not generate partial report: State is insufficient or unavailable.")
    raise typer.Exit(code=1)

  print("\n--- Analysis Complete ---")

  if final_full_state_for_report is None:
    try:
      final_full_state_for_report = app_graph.get_state(config_for_stream)
    except Exception:
      print("Warning: Could not retrieve final state for report after graph completion.")
      final_full_state_for_report = initial_state_dict

  if final_full_state_for_report and isinstance(final_full_state_for_report, dict):
    print("Generating final report...")
    final_log = final_full_state_for_report.get("investigation_log", [])
    final_dump_path = final_full_state_for_report.get("dump_path", dump_path)
    final_initial_context = final_full_state_for_report.get("initial_context", context)
    final_profile = final_full_state_for_report.get("profile", "Unknown")

    final_summary_text = ""
    if final_full_state_for_report.get("messages"):
      for msg_obj in reversed(final_full_state_for_report["messages"]):
        if isinstance(msg_obj, AIMessage) and not msg_obj.tool_calls:
          if isinstance(msg_obj.content, str):
            final_summary_text = msg_obj.content
            break
          elif isinstance(msg_obj.content, list):
            text_parts = [block.get("text", "") for block in msg_obj.content if block.get("type") == "text"]
            final_summary_text = "".join(text_parts).strip()
            if final_summary_text: break
      if not final_summary_text:
        final_summary_text = "No final summary message found from AI."

    report_msg = generate_report(
      final_log,
      final_dump_path,
      final_initial_context,
      final_profile,
      final_summary=final_summary_text
    )
    print(report_msg)
  else:
    print("Warning: Could not generate final report due to missing state.")


if __name__ == "__main__":
  app()
