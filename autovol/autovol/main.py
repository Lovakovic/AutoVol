import os
# import json # Not strictly needed anymore
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import the graph AFTER loading env vars
load_dotenv()
from .agent import app_graph, AppState  # Assuming AppState is correctly defined

# from langgraph.graph import END # Not strictly needed for this streaming logic

# Initialize Typer app
app = typer.Typer()


# --- generate_report function remains largely the same ---
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
    report_content += f"## Final LLM Summary:\n\n```text\n{final_summary}\n```\n\n"  # Added code block for summary

  report_content += "---\n\n## Investigation Steps\n\n"

  if not log:
    report_content += "No investigation steps were logged.\n"
  else:
    for i, entry in enumerate(log):
      report_content += f"### Step {i + 1}\n\n"
      reasoning = entry.get('reasoning', 'N/A')
      if isinstance(reasoning, list):
        reasoning_text = ""
        for item in reasoning:
          if isinstance(item, dict) and item.get('type') == 'text':
            reasoning_text += item.get('text', '') + "\n"
          elif isinstance(item, str):
            reasoning_text += item + "\n"
        reasoning = reasoning_text.strip() if reasoning_text.strip() else "Complex reasoning object"
      elif not isinstance(reasoning, str):
        reasoning = str(reasoning)

      report_content += f"**Reasoning/Action (LLM):**\n```\n{reasoning}\n```\n\n"
      report_content += f"**Executed Command:**\n```bash\nvol.py -f <dump_path> {entry.get('command', 'N/A')}\n```\n\n"
      report_content += f"**Output:**\n"
      report_content += f"<details>\n<summary>Click to view output</summary>\n\n```text\n{entry.get('output', 'N/A')}\n```\n\n</details>\n\n"
      report_content += "---\n\n"

  try:
    with open(report_path, "w") as f:
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

  # ... (environment variable checks remain the same) ...
  if not os.path.exists(dump_path):
    print(f"\nError: Memory dump file not found at '{dump_path}'")
    raise typer.Exit(code=1)
  if not os.getenv("VOLATILITY3_PATH"):
    print("\nError: VOLATILITY3_PATH environment variable is not set.")
    raise typer.Exit(code=1)
  if not os.getenv("ANTHROPIC_API_KEY"):
    print("\nError: ANTHROPIC_API_KEY environment variable is not set.")
    raise typer.Exit(code=1)

  initial_state_dict: AppState = {  # Ensure this matches your AppState TypedDict
    "messages": [HumanMessage(content=context)],
    "dump_path": dump_path,
    "initial_context": context,
    "profile": None,
    "investigation_log": []
  }

  config = {"recursion_limit": 50}
  # This will hold the LATEST full state dictionary yielded by the stream
  current_graph_state_snapshot: Optional[AppState] = None
  # Keep track of the number of messages printed to avoid re-printing old ones
  num_messages_printed_for_cli = len(initial_state_dict["messages"])

  try:
    # According to langgraph-streaming.md, for stream_mode="values",
    # `full_state_at_step` IS the AppState dictionary.
    for full_state_at_step in app_graph.stream(
      initial_state_dict,
      config=config,
      stream_mode="values"  # Explicitly using "values"
    ):
      # `full_state_at_step` is the entire state dictionary at this point.
      current_graph_state_snapshot = full_state_at_step

      # --- Enhanced CLI Streaming ---
      # We want to print new AI messages or Tool messages as they appear.
      # The `full_state_at_step` contains all messages up to this point.
      if isinstance(full_state_at_step, dict) and "messages" in full_state_at_step:
        all_messages_now = full_state_at_step["messages"]
        # Print only new messages
        new_messages_to_print = all_messages_now[num_messages_printed_for_cli:]

        if new_messages_to_print:
          print(f"\n--- New Streamed Output ---")  # Indicate a new batch of output
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
                print(f"Tool Call Requested: {msg.tool_calls}")
            elif isinstance(msg, ToolMessage):
              # To avoid overly verbose output for long tool results,
              # you might summarize or truncate here for CLI.
              # The full output is in the investigation_log.
              tool_output_summary = msg.content
              if len(tool_output_summary) > 300:  # Example truncation
                tool_output_summary = tool_output_summary[:300] + "...\n(Full output in report)"
              print(f"Tool Result ({msg.tool_call_id}):\n{tool_output_summary}")

          num_messages_printed_for_cli = len(all_messages_now)  # Update the count

      # The "Event from Node: <key>" prints were because of the old loop.
      # If you want to know which node *just ran* to produce this `full_state_at_step`,
      # `stream_mode="updates"` is better, or you'd have to compare
      # `full_state_at_step` with the *previous* state to see what changed.
      # For "values", we just get the whole state.

  except Exception as e:
    print(f"\n--- An error occurred during graph execution ---")
    print(e)
    # Use current_graph_state_snapshot for partial report
    if current_graph_state_snapshot and isinstance(current_graph_state_snapshot,
                                                   dict) and "investigation_log" in current_graph_state_snapshot:
      print("\nAttempting to generate partial report...")
      report_msg = generate_report(
        current_graph_state_snapshot.get("investigation_log", []),
        current_graph_state_snapshot.get("dump_path", dump_path),
        current_graph_state_snapshot.get("initial_context", context),
        current_graph_state_snapshot.get("profile", "Unknown")
      )
      print(report_msg)
    else:
      # ... (debug prints for error case) ...
      print("Could not generate partial report: current_graph_state_snapshot is insufficient.")
      if current_graph_state_snapshot is not None:
        print(
          f"DEBUG (Exception): current_graph_state_snapshot type: {type(current_graph_state_snapshot)}, keys: {list(current_graph_state_snapshot.keys()) if isinstance(current_graph_state_snapshot, dict) else 'Not a dict'}")
      else:
        print(f"DEBUG (Exception): current_graph_state_snapshot is None.")
    raise typer.Exit(code=1)

  print("\n--- Analysis Complete ---")

  if current_graph_state_snapshot and isinstance(current_graph_state_snapshot,
                                                 dict) and "investigation_log" in current_graph_state_snapshot:
    print("Generating final report...")
    final_log = current_graph_state_snapshot.get("investigation_log", [])
    # ... (rest of the report generation logic using current_graph_state_snapshot) ...
    final_dump_path = current_graph_state_snapshot.get("dump_path", dump_path)
    final_initial_context = current_graph_state_snapshot.get("initial_context", context)
    final_profile = current_graph_state_snapshot.get("profile", "Unknown")

    final_summary_text = ""
    if current_graph_state_snapshot.get("messages"):
      for msg_obj in reversed(current_graph_state_snapshot["messages"]):  # Renamed to avoid conflict
        if isinstance(msg_obj, AIMessage) and not msg_obj.tool_calls:
          if isinstance(msg_obj.content, str):
            final_summary_text = msg_obj.content
            break
          elif isinstance(msg_obj.content, list):
            final_summary_text = "".join(
              [block.get("text", "") for block in msg_obj.content if block.get("type") == "text"])
            break

    report_msg = generate_report(
      final_log,
      final_dump_path,
      final_initial_context,
      final_profile,
      final_summary=final_summary_text
    )
    print(report_msg)

  else:
    # ... (debug prints for final report failure) ...
    print("Warning: Could not retrieve final state to generate report.")
    if current_graph_state_snapshot is not None and isinstance(current_graph_state_snapshot, dict):
      print(
        f"DEBUG (Report Gen Failed): current_graph_state_snapshot keys: {list(current_graph_state_snapshot.keys())}")
      print(
        f"DEBUG (Report Gen Failed): 'investigation_log' present: {'investigation_log' in current_graph_state_snapshot}")
      # print(f"DEBUG (Report Gen Failed): 'investigation_log' content: {current_graph_state_snapshot.get('investigation_log')}") # Can be verbose
    elif current_graph_state_snapshot is None:
      print(f"DEBUG (Report Gen Failed): current_graph_state_snapshot is None.")
    else:
      print(f"DEBUG (Report Gen Failed): current_graph_state_snapshot type: {type(current_graph_state_snapshot)}")


if __name__ == "__main__":
  app()
