from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def generate_report(
  log: List[Dict[str, Any]],
  dump_path: str,
  initial_context: str,
  profile: str,
  final_summary: str = "",
  report_session_id: str = ""
) -> str:
  if not report_session_id:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_dump_stem = "".join(c if c.isalnum() else '_' for c in Path(dump_path).stem)
    report_session_id = f"autovol_report_{safe_dump_stem}_{timestamp}"

  reports_dir_base = Path("reports")
  report_session_dir = reports_dir_base / report_session_id
  report_session_dir.mkdir(parents=True, exist_ok=True)

  # Define filenames for the two reports
  summary_report_filename = "analysis_summary.md"
  details_report_filename = "analysis_details.md"

  summary_report_path = report_session_dir / summary_report_filename
  details_report_path = report_session_dir / details_report_filename

  # --- Generate common metadata header ---
  metadata_header = f"**Session ID:** `{report_session_id}`\n"
  metadata_header += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
  metadata_header += f"**Memory Dump:** `{dump_path}`\n"
  metadata_header += f"**Detected Profile Base:** `{profile if profile else 'Not Detected'}`\n"
  metadata_header += f"**Initial Context/Suspicion:**\n```text\n{initial_context}\n```\n\n"  # Using text for context for safety

  # --- Generate Summary Report ---
  summary_report_content = f"# AutoVol Analysis Summary\n\n"
  summary_report_content += metadata_header

  if final_summary:
    summary_report_content += f"## Final LLM Findings:\n\n"
    # Write the LLM's final summary directly, without triple backticks
    # to allow its Markdown to be rendered.
    summary_report_content += f"{final_summary.strip()}\n\n"
  else:
    summary_report_content += "No final summary was provided by the LLM.\n\n"

  summary_report_content += f"---\n\n"
  summary_report_content += f"For a detailed step-by-step breakdown of the analysis, including all commands and links to full outputs, please see the [Detailed Analysis Report](./{details_report_filename}).\n"
  summary_report_content += f"Raw command outputs are stored in the `command_outputs` subdirectory within this session's report folder.\n"

  # --- Generate Details Report ---
  details_report_content = f"# AutoVol Detailed Analysis Steps\n\n"
  details_report_content += metadata_header
  details_report_content += "---\n\n## Investigation Steps\n\n"

  if not log:
    details_report_content += "No investigation steps were logged.\n"
  else:
    for i, entry in enumerate(log):
      details_report_content += f"### Step {i + 1}\n\n"
      reasoning = entry.get('reasoning', 'N/A')
      if not isinstance(reasoning, str):
        reasoning = str(reasoning)
      details_report_content += f"**Reasoning/Context:**\n```\n{reasoning.strip()}\n```\n\n"

      command_or_action = entry.get('command', 'N/A')
      entry_type = entry.get("type", "unknown")

      if entry_type == "user_decision":
        details_report_content += f"**User Decision & Command Context:**\n```bash\n{command_or_action}\n```\n\n"
      elif entry_type == "tool_execution":
        details_report_content += f"**Executed Command:**\n```bash\n{command_or_action}\n```\n\n"
      else:
        details_report_content += f"**Action/Event:**\n```text\n{command_or_action}\n```\n\n"  # text for safety if not bash

      output_file_path_relative = entry.get('output_file_path')

      if entry_type == "tool_execution" and output_file_path_relative:
        link_text = f"View Full Output ({Path(output_file_path_relative).name})"
        # Ensure the link is relative to the details report file itself
        details_report_content += f"**Output:** [{link_text}](./{output_file_path_relative})\n\n"
      elif entry_type == "user_decision" or entry_type == "internal_error":
        details = entry.get('output_details', 'Details not available.')
        if not isinstance(details, str):
          details = str(details)
        details_report_content += f"**Outcome:**\n```text\n{details.strip()}\n```\n\n"
      else:
        details = entry.get('output_details', entry.get('output_summary', 'N/A'))
        if not isinstance(details, str):
          details = str(details)
        details_report_content += f"**Details/Summary (Fallback):**\n```text\n{details.strip()}\n```\n\n"

      details_report_content += "---\n\n"

  # --- Write Files ---
  files_generated_messages = []
  try:
    with open(summary_report_path, "w", encoding='utf-8') as f:
      f.write(summary_report_content)
    files_generated_messages.append(f"Summary report generated at: {summary_report_path}")
  except Exception as e:
    files_generated_messages.append(f"Error generating summary report: {e}")

  try:
    with open(details_report_path, "w", encoding='utf-8') as f:
      f.write(details_report_content)
    files_generated_messages.append(f"Details report generated at: {details_report_path}")
  except Exception as e:
    files_generated_messages.append(f"Error generating details report: {e}")

  # Add message about the command outputs directory
  command_outputs_dir = report_session_dir / "command_outputs"
  if command_outputs_dir.exists():
    files_generated_messages.append(f"Command outputs saved in: {command_outputs_dir}")

  return "\n".join(files_generated_messages)
