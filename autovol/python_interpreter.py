import io
import contextlib
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class PythonInterpreterInput(BaseModel):
  code: str = Field(description="The Python 3 code to execute.")


def run_python_code_logic(code: str) -> Dict[str, str]:
  """
  Executes the given Python 3 code and captures its stdout and stderr.

  Args:
      code: The Python 3 code string to execute.

  Returns:
      A dictionary containing 'stdout' and 'stderr' strings.
  """
  stdout_capture = io.StringIO()
  stderr_capture = io.StringIO()
  result_stdout = ""
  result_stderr = ""

  try:
    # Create a dictionary for the local and global scope of exec
    # This can be expanded with pre-defined variables if needed
    local_vars: Dict[str, Any] = {}
    global_vars: Dict[str, Any] = {'__builtins__': __builtins__}

    with contextlib.redirect_stdout(stdout_capture):
      with contextlib.redirect_stderr(stderr_capture):
        exec(code, global_vars, local_vars)

    result_stdout = stdout_capture.getvalue()
    result_stderr = stderr_capture.getvalue()

  except Exception as e:
    result_stderr += f"\nException during execution: {type(e).__name__}: {e}"
  finally:
    stdout_capture.close()
    stderr_capture.close()

  return {"stdout": result_stdout, "stderr": result_stderr}


@tool("python_interpreter", args_schema=PythonInterpreterInput)
def python_interpreter_tool(code: str) -> str:
  """
  Executes Python 3 code and returns its standard output and standard error.
  Use this tool for data manipulation, calculations, or any task programmable in Python.
  The code is executed in a stateless environment.
  Input is a single string 'code' containing the Python code to run.
  Output is a string containing the captured stdout and stderr.
  """
  # This docstring is what the LLM sees.
  # The actual execution logic is in run_python_code_logic,
  # which will be called by the graph's tool_executor_node.
  # The tool function itself just describes the tool for the LLM.
  # The return here is just a placeholder as graph will call the logic function.
  return "Python code execution placeholder. Actual execution handled by the agent's tool node."
