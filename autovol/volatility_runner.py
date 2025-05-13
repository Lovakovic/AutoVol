import os
import subprocess
import re
from typing import Optional, Tuple, List

# Ensure VOLATILITY3_PATH is set in your .env file or environment
VOLATILITY_PATH = os.getenv("VOLATILITY3_PATH")
PYTHON_EXECUTABLE = "python3"  # Or specify the full path to the python in Volatility's venv if needed


def _get_volatility_path() -> str:
  """Gets the validated path to the Volatility 3 executable."""
  if not VOLATILITY_PATH:
    raise ValueError("VOLATILITY3_PATH environment variable not set or empty.")
  if not os.path.exists(VOLATILITY_PATH) or not VOLATILITY_PATH.endswith("vol.py"):
    raise FileNotFoundError(f"Volatility 3 script not found at: {VOLATILITY_PATH}")
  return VOLATILITY_PATH


def execute_volatility_plugin(
  dump_path: str,
  profile: str,  # Still passed for potential context, but not used in command
  plugin: str,  # Expects full plugin name like 'windows.pslist.PsList'
  plugin_args: Optional[List[str]] = None,
  volatility_path: Optional[str] = None
) -> Tuple[str, str, int]:
  """
  Executes a specified Volatility 3 plugin as a subprocess.

  Args:
      dump_path: Path to the memory dump file.
      profile: The OS profile base (e.g., 'windows').
      plugin: The full plugin name to run (e.g., 'windows.pslist.PsList').
      plugin_args: Optional list of arguments for the plugin.
      volatility_path: Optional override for the path to vol.py.

  Returns:
      A tuple containing (stdout, stderr, return_code).
  """
  vol_path = volatility_path or _get_volatility_path()
  if not os.path.exists(dump_path):
    return "", f"Error: Dump file not found at {dump_path}", 1

  if not re.match(r"^[a-zA-Z0-9._-]+$", plugin):  # Basic validation for plugin name
    return "", f"Error: Invalid plugin name format: {plugin}", 1

  command = [
    PYTHON_EXECUTABLE,
    vol_path,
    "-f",
    dump_path,
    plugin  # Use the plugin name directly
  ]

  if plugin_args:
    command.extend(plugin_args)

  print(f"\nExecuting Volatility command: {' '.join(command)}")

  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,  # Do not raise exception on non-zero exit, handle it manually
      timeout=300  # 5 minutes timeout
    )
    print(f"Volatility exited with code: {result.returncode}")
    return result.stdout, result.stderr, result.returncode
  except FileNotFoundError:
    return "", f"Error: '{PYTHON_EXECUTABLE}' or '{vol_path}' not found.", 1
  except subprocess.TimeoutExpired:
    return "", f"Error: Volatility command timed out after 300 seconds.", 1
  except Exception as e:  # Catch other potential errors
    return "", f"An unexpected error occurred during Volatility execution: {e}", 1


def detect_profile(
  dump_path: str,
  volatility_path: Optional[str] = None
) -> Optional[str]:
  """
  Attempts to detect the OS profile using Volatility's info plugins.
  Returns the OS base (e.g., 'windows', 'linux') or None.
  """
  vol_path = volatility_path or _get_volatility_path()
  if not os.path.exists(dump_path):
    print(f"Error: Dump file not found at {dump_path}")
    return None

  # List of info plugins to try in order
  info_plugins_to_try = ["windows.info.Info", "linux.info.Info", "mac.info.Info"]
  detected_os_base = None

  for plugin_to_try in info_plugins_to_try:
    command = [PYTHON_EXECUTABLE, vol_path, "-f", dump_path, plugin_to_try]
    print(f"\nAttempting profile detection with: {plugin_to_try}")
    try:
      result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
      if result.returncode == 0 and result.stdout:
        # A simple heuristic: if the plugin name is in stdout, it's likely that OS.
        # Volatility 3 info plugins often mention their own name or related OS strings.
        if "windows" in plugin_to_try.lower() and "windows" in result.stdout.lower():
          detected_os_base = "windows"
          break
        elif "linux" in plugin_to_try.lower() and "linux" in result.stdout.lower():
          detected_os_base = "linux"
          break
        elif "mac" in plugin_to_try.lower() and ("mac" in result.stdout.lower() or "darwin" in result.stdout.lower()):
          detected_os_base = "mac"
          break
      else:
        print(
          f"Plugin {plugin_to_try} failed or gave no output. Stderr (if any): {result.stderr[:200]}...")  # Log snippet of stderr
    except FileNotFoundError:
      print(f"Error: '{PYTHON_EXECUTABLE}' or '{vol_path}' not found during profile detection with {plugin_to_try}.")
      return None  # Critical error, can't proceed
    except subprocess.TimeoutExpired:
      print(f"Error: Volatility profile detection with {plugin_to_try} timed out.")
    except Exception as e:
      print(f"An unexpected error occurred during profile detection with {plugin_to_try}: {e}")

  if detected_os_base:
    print(f"Detected OS base: {detected_os_base}")
  else:
    print("Could not reliably determine OS base from info plugins.")
  return detected_os_base


from langchain_core.tools import tool
from pydantic import BaseModel, Field


# from typing import List, Optional # Already imported

class VolatilityPluginInput(BaseModel):
  plugin_name: str = Field(
    description="The full name of the Volatility 3 plugin to run (e.g., 'windows.pslist.PsList', 'linux.pslist.PsList').")
  plugin_args: Optional[List[str]] = Field(default=None,
                                           description="A list of optional arguments to pass to the plugin (e.g., ['--pid', '1234']).")


def run_volatility_tool_logic(plugin_name: str, plugin_args: Optional[List[str]], dump_path: str, profile: str) -> str:
  """Core logic for executing the Volatility tool."""
  if not dump_path:  # Profile is context, dump_path is essential
    return "Error: dump_path is missing for tool execution."
  # Profile here is the OS base (e.g. "windows"), used for context, not directly in command if plugin_name is full.

  stdout, stderr, return_code = execute_volatility_plugin(
    dump_path=dump_path,
    profile=profile,  # Pass for context, though execute_volatility_plugin might not use it if plugin name is full
    plugin=plugin_name,
    plugin_args=plugin_args
  )

  if return_code == 0:
    if not stdout.strip() and not stderr.strip():  # Check both if stdout is empty
      return f"Plugin '{plugin_name}' executed successfully but produced no output on stdout or stderr."
    elif not stdout.strip() and stderr.strip():
      return f"Plugin '{plugin_name}' executed with no stdout output, but stderr contained:\n```\n{stderr}\n```"
    # Truncate extremely long outputs for the ToolMessage, full output in log.
    # For now, returning full output. Agent needs to be robust.
    return f"Success:\n```\n{stdout}\n```"
  else:
    return f"Error executing plugin '{plugin_name}' (code: {return_code}):\nStderr:\n```\n{stderr}\n```\nStdout (if any):\n```\n{stdout}\n```"


@tool("volatility_runner", args_schema=VolatilityPluginInput)
def volatility_runner_tool(plugin_name: str, plugin_args: Optional[List[str]] = None) -> str:
  """
  Runs a Volatility 3 plugin against the memory dump specified in the current context.
  Requires plugin_name (e.g., 'windows.pslist.PsList') and optional plugin_args (e.g., ['--pid', '1234']).
  The dump file path and OS profile base must be available in the agent's state when this tool is invoked by a graph node.
  """
  # This is a placeholder for LangChain's tool binding.
  # The actual logic is in run_volatility_tool_logic, called by the graph node.
  # This docstring is what the LLM sees.
  return "This function is a placeholder for tool binding and should not be called directly. The graph's tool node executes the actual logic."
