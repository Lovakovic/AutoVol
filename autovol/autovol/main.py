import typer
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# --- ADD/MODIFY THIS LINE ---
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
# --- END ADD/MODIFY ---

# Import the graph AFTER loading env vars
load_dotenv()
from .agent import app_graph, AppState

# Initialize Typer app
app = typer.Typer()

# --- generate_report function remains the same ---
def generate_report(log: list, dump_path: str, initial_context: str, profile: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"autovol_report_{Path(dump_path).stem}_{timestamp}.md"
    report_path = Path("reports") / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_content = f"# AutoVol Analysis Report\n\n"
    report_content += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content += f"**Memory Dump:** `{dump_path}`\n"
    report_content += f"**Detected Profile Base:** `{profile}`\n"
    report_content += f"**Initial Context/Suspicion:**\n```\n{initial_context}\n```\n\n"
    report_content += "---\n\n## Investigation Steps\n\n"

    for i, entry in enumerate(log):
        report_content += f"### Step {i+1}\n\n"
        report_content += f"**Reasoning/Action:**\n```\n{entry.get('reasoning', 'N/A')}\n```\n\n"
        report_content += f"**Executed Command:**\n```bash\nvol.py -f <dump_path> {entry.get('command', 'N/A')}\n```\n\n"
        report_content += f"**Output:**\n"
        report_content += f"<details>\n<summary>Click to view output</summary>\n\n```\n{entry.get('output', 'N/A')}\n```\n\n</details>\n\n"
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
    # ... (Initial prints and checks remain the same) ...
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

    # Initial state remains the same
    initial_state: AppState = {
        "messages": [HumanMessage(content=context)], # HumanMessage needed here
        "dump_path": dump_path,
        "initial_context": context,
        "profile": None,
        "investigation_log": []
    }

    config = {"recursion_limit": 50}
    final_state = None
    try:
        for event in app_graph.stream(initial_state, config=config):
            for node_name, output in event.items():
                print(f"\n--- Output from Node: {node_name} ---")
                if isinstance(output, dict) and "messages" in output:
                     last_message = output["messages"][-1]
                     # These checks should now work
                     if isinstance(last_message, AIMessage):
                         print(f"AI: {last_message.content}")
                         if last_message.tool_calls:
                             print(f"Tool Call Requested: {last_message.tool_calls}")
                     elif isinstance(last_message, ToolMessage):
                         print(f"Tool Result ({last_message.tool_call_id}):\n{last_message.content}")
                     elif isinstance(last_message, SystemMessage):
                         print(f"System: {last_message.content}")
                final_state = output

    except Exception as e:
        print(f"\n--- An error occurred during graph execution ---")
        print(e)
        if final_state and "investigation_log" in final_state:
            print("\nAttempting to generate partial report...")
            report_msg = generate_report(
                final_state["investigation_log"],
                final_state.get("dump_path", dump_path),
                final_state.get("initial_context", context),
                final_state.get("profile", "Unknown")
            )
            print(report_msg)
        raise typer.Exit(code=1)

    print("\n--- Analysis Complete ---")

    if final_state and "investigation_log" in final_state:
        print("Generating final report...")
        report_msg = generate_report(
            final_state["investigation_log"],
            final_state.get("dump_path", dump_path),
            final_state.get("initial_context", context),
            final_state.get("profile", "Unknown")
        )
        print(report_msg)
    else:
        print("Warning: Could not retrieve final state to generate report.")

if __name__ == "__main__":
    app()
