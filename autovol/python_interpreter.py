import io
import os 
import contextlib
import re
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path 
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class PythonInterpreterInput(BaseModel):
  code: str = Field(description="The Python 3 code to execute.")


def run_python_code_logic(code: str, session_workspace_dir: Optional[str] = None) -> Dict[str, str]:
  """
  Executes the given Python 3 code and captures its stdout and stderr.
  If session_workspace_dir is provided, changes CWD to it for code execution.
  """
  stdout_capture = io.StringIO()
  stderr_capture = io.StringIO()
  result_stdout = ""
  result_stderr = ""

  original_cwd = os.getcwd()
  try:
    if session_workspace_dir:
      Path(session_workspace_dir).mkdir(parents=True, exist_ok=True) 
      os.chdir(session_workspace_dir)
      print(f"Python interpreter CWD set to: {session_workspace_dir}")
    
    # Import available modules, handling failures gracefully
    available_modules = {'__builtins__': __builtins__}
    
    # Add session workspace directory
    if session_workspace_dir:
        available_modules['SESSION_WORKSPACE_DIR'] = str(Path(session_workspace_dir).resolve())
    else:
        available_modules['SESSION_WORKSPACE_DIR'] = None
    
    # Only pre-import essential modules that need to be available
    # Let users import libraries themselves with their preferred aliases
    essential_modules = {
        'os': os,
        'sys': sys,
        'io': io,
        're': re,
        'json': json,
        'datetime': datetime,
        'pathlib': Path
    }
    
    global_vars: Dict[str, Any] = {
        '__builtins__': __builtins__,
        'SESSION_WORKSPACE_DIR': available_modules.get('SESSION_WORKSPACE_DIR'),
        **essential_modules
    }

    # Use a single namespace for both globals and locals to avoid import issues
    # This allows imports to work properly in all contexts
    exec_namespace = global_vars.copy()
    
    with contextlib.redirect_stdout(stdout_capture):
      with contextlib.redirect_stderr(stderr_capture):
        exec(code, exec_namespace, exec_namespace)

    result_stdout = stdout_capture.getvalue()
    result_stderr = stderr_capture.getvalue()

  except Exception as e:
    result_stderr += f"\nException during execution: {type(e).__name__}: {e}"
  finally:
    stdout_capture.close()
    stderr_capture.close()
    if session_workspace_dir and Path(original_cwd).exists(): 
        os.chdir(original_cwd)
        print(f"Python interpreter CWD restored to: {original_cwd}")

  return {"stdout": result_stdout, "stderr": result_stderr}


@tool("python_interpreter", args_schema=PythonInterpreterInput)
def python_interpreter_tool(code: str) -> str:
  """
  Executes Python 3 code within a dedicated session workspace and returns its standard output and standard error.
  The current working directory for the executed code is set to this session workspace.
  This means you can use relative paths to read files created by Volatility (e.g., 'dumped_file.bin')
  or to create new files (e.g., 'analysis_results.txt', 'my_plot.png').
  
  Available libraries: pandas, numpy, regex, requests, python-dateutil, PyYAML, matplotlib, patoolib, rarfile, py7zr, zipfile.
  Archive support: Can extract/read RAR, 7Z, ZIP files using patoolib, rarfile, py7zr, and zipfile modules.
  If using matplotlib, save plots to a file (e.g., plt.savefig('figure.png')) instead of plt.show().
  A global variable `SESSION_WORKSPACE_DIR` (string) is also available inside your script, 
  containing the absolute path to the session workspace, though using relative paths is often simpler.

  Input is a single string 'code' containing the Python code to run.
  Output is a string containing the captured stdout and stderr.
  """
  return "Python code execution placeholder. Actual execution handled by the agent's tool node."
