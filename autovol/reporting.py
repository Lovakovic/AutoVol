from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any  # For AppState if needed, but generate_report takes specific args


def generate_report(
  log: List[Dict[str, str]],
  dump_path: str,
  initial_context: str,
  profile: str,
  final_summary: str = ""
) -> str:
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  report_filename = f"autovol_report_{Path(dump_path).stem}_{timestamp}.md"

  # Create 'reports' directory if it doesn't exist relative to the CWD
  # where the main script (autovol/main.py) is executed.
  reports_dir = Path("reports")
  reports_dir.mkdir(parents=True, exist_ok=True)
  report_path = reports_dir / report_filename

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
        reasoning = str(reasoning)  # Ensure string conversion

      report_content += f"**Reasoning/Action (LLM/User):**\n```\n{reasoning}\n```\n\n"
      report_content += f"**Executed Command/Action:**\n```bash\n{entry.get('command', 'N/A')}\n```\n\n"
      report_content += f"**Output/Result:**\n"

      output_text = entry.get('output', 'N/A')
      if not isinstance(output_text, str):
        output_text = str(output_text)

      # For very long outputs, the <details> tag is good.
      # You might also consider truncating output_text for the summary in <details>
      # but keeping the full text inside. For now, keeping it simple.
      report_content += f"<details>\n<summary>Click to view output/result (length: {len(output_text)})</summary>\n\n```text\n{output_text}\n```\n\n</details>\n\n"
      report_content += "---\n\n"

  try:
    with open(report_path, "w", encoding='utf-8') as f:
      f.write(report_content)
    return f"Report generated successfully at: {report_path}"
  except Exception as e:
    return f"Error generating report: {e}"
