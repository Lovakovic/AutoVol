import io
import os 
import contextlib
import re
import sys
import json
import signal
import subprocess
import tempfile
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path 
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class PythonInterpreterInput(BaseModel):
  code: str = Field(description="The Python 3 code to execute.")


def create_execution_script(code: str, workspace_dir: Optional[str]) -> str:
  """Create a standalone Python script that executes the user's code with proper setup."""
  
  script_template = '''
import sys
import os
import io
import re
import json
from datetime import datetime
from pathlib import Path

# Set working directory if provided
workspace_dir = {workspace_dir}
if workspace_dir:
    os.chdir(workspace_dir)

# Set up SESSION_WORKSPACE_DIR
SESSION_WORKSPACE_DIR = str(Path(workspace_dir).resolve()) if workspace_dir else None

# Capture stdout and stderr
import contextlib
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            # User code starts here
{indented_code}
            # User code ends here
    
    print("EXECUTION_SUCCESS")
    print("STDOUT_START")
    print(stdout_capture.getvalue())
    print("STDOUT_END")
    print("STDERR_START")
    print(stderr_capture.getvalue())
    print("STDERR_END")
    
except Exception as e:
    print("EXECUTION_ERROR")
    print("STDOUT_START")
    print(stdout_capture.getvalue())
    print("STDOUT_END")
    print("STDERR_START")
    print(stderr_capture.getvalue())
    print(f"\\nException during execution: {{type(e).__name__}}: {{e}}")
    print("STDERR_END")
finally:
    stdout_capture.close()
    stderr_capture.close()
'''
  
  # Indent the user's code
  indented_code = '\n'.join('            ' + line for line in code.splitlines())
  
  return script_template.format(
    workspace_dir=repr(workspace_dir),
    indented_code=indented_code
  )


def run_python_code_logic(code: str, session_workspace_dir: Optional[str] = None, timeout_seconds: int = 300) -> Dict[str, str]:
  """
  Executes the given Python 3 code and captures its stdout and stderr.
  If session_workspace_dir is provided, changes CWD to it for code execution.
  
  Args:
    code: Python code to execute
    session_workspace_dir: Working directory for execution
    timeout_seconds: Maximum execution time in seconds (default: 300 = 5 minutes)
  """
  
  original_cwd = os.getcwd()
  result_stdout = ""
  result_stderr = ""
  
  try:
    if session_workspace_dir:
      Path(session_workspace_dir).mkdir(parents=True, exist_ok=True)
      print(f"Python interpreter workspace: {session_workspace_dir}")
      print(f"Script execution timeout: {timeout_seconds} seconds")
    
    # Create a temporary file for the script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
      script_content = create_execution_script(code, session_workspace_dir)
      f.write(script_content)
      script_path = f.name
    
    try:
      # Run the script in a subprocess with timeout
      process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=original_cwd  # Run from original directory, script will change if needed
      )
      
      try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        
        # Parse the output
        if "EXECUTION_SUCCESS" in stdout or "EXECUTION_ERROR" in stdout:
          # Extract stdout
          if "STDOUT_START" in stdout and "STDOUT_END" in stdout:
            start = stdout.find("STDOUT_START") + len("STDOUT_START\n")
            end = stdout.find("STDOUT_END")
            result_stdout = stdout[start:end].rstrip('\n')
          
          # Extract stderr
          if "STDERR_START" in stdout and "STDERR_END" in stdout:
            start = stdout.find("STDERR_START") + len("STDERR_START\n")
            end = stdout.find("STDERR_END")
            result_stderr = stdout[start:end].rstrip('\n')
        else:
          # Fallback if markers aren't found
          result_stdout = stdout
          result_stderr = stderr
          
      except subprocess.TimeoutExpired:
        # Kill the process
        process.kill()
        try:
          process.wait(timeout=5)  # Give it 5 seconds to die
        except subprocess.TimeoutExpired:
          # Force kill if needed
          process.terminate()
        
        result_stderr = f"TIMEOUT ERROR: Script execution exceeded {timeout_seconds} seconds limit\n"
        result_stderr += "The script was forcefully terminated.\n"
        result_stderr += "Consider:\n"
        result_stderr += "- Processing data in smaller chunks\n"
        result_stderr += "- Using more efficient algorithms\n"
        result_stderr += "- Saving intermediate results to files\n"
        print(f"Script execution timed out and was killed after {timeout_seconds} seconds")
        
    finally:
      # Clean up the temporary script file
      try:
        os.unlink(script_path)
      except:
        pass
    
  except Exception as e:
    result_stderr = f"Exception during script execution: {type(e).__name__}: {e}"
    import traceback
    result_stderr += f"\n{traceback.format_exc()}"
  
  return {"stdout": result_stdout, "stderr": result_stderr}


@tool("python_interpreter", args_schema=PythonInterpreterInput)
def python_interpreter_tool(code: str) -> str:
  """
  Executes Python 3 code within a dedicated session workspace and returns its standard output and standard error.
  The current working directory for the executed code is set to this session workspace.
  This means you can use relative paths to read files created by Volatility (e.g., 'dumped_file.bin')
  or to create new files (e.g., 'analysis_results.txt', 'my_plot.png').
  
  IMPORTANT: Scripts have a 5-minute execution timeout. Long-running scripts will be terminated.
  For large data processing, consider:
  - Processing data in smaller chunks
  - Using efficient algorithms and data structures
  - Saving intermediate results to files
  
  Available libraries: pandas, numpy, regex, requests, python-dateutil, PyYAML, matplotlib, patoolib, rarfile, py7zr, zipfile.
  Archive support: Can extract/read RAR, 7Z, ZIP files using patoolib, rarfile, py7zr, and zipfile modules.
  If using matplotlib, save plots to a file (e.g., plt.savefig('figure.png')) instead of plt.show().
  A global variable `SESSION_WORKSPACE_DIR` (string) is also available inside your script, 
  containing the absolute path to the session workspace, though using relative paths is often simpler.

  Input is a single string 'code' containing the Python code to run.
  Output is a string containing the captured stdout and stderr.
  """
  return "Python code execution placeholder. Actual execution handled by the agent's tool node."
