# autovol/volatility_runner.py
import os
import subprocess
import re
from typing import Optional, Tuple, List

# Ensure VOLATILITY3_PATH is set in your .env file or environment
VOLATILITY_PATH = os.getenv("VOLATILITY3_PATH")
PYTHON_EXECUTABLE = "python3" # Or specify the full path to the python in Volatility's venv if needed

def _get_volatility_path() -> str:
    """Gets the validated path to the Volatility 3 executable."""
    if not VOLATILITY_PATH:
        raise ValueError(
            "VOLATILITY3_PATH environment variable not set or empty. "
            "Please set it to the full path of your vol.py script."
        )
    if not os.path.exists(VOLATILITY_PATH) or not VOLATILITY_PATH.endswith("vol.py"):
         raise FileNotFoundError(
            f"Volatility 3 script not found at the specified path: {VOLATILITY_PATH}"
        )
    return VOLATILITY_PATH

def execute_volatility_plugin(
    dump_path: str,
    profile: str,
    plugin: str,
    plugin_args: Optional[List[str]] = None,
    volatility_path: Optional[str] = None
) -> Tuple[str, str, int]:
    """
    Executes a specified Volatility 3 plugin as a subprocess.

    Args:
        dump_path: Path to the memory dump file.
        profile: The Volatility profile to use (e.g., 'windows.info').
        plugin: The specific plugin to run (e.g., 'windows.pslist.PsList').
        plugin_args: Optional list of arguments for the plugin.
        volatility_path: Optional override for the path to vol.py.

    Returns:
        A tuple containing (stdout, stderr, return_code).
    """
    vol_path = volatility_path or _get_volatility_path()
    if not os.path.exists(dump_path):
        return "", f"Error: Dump file not found at {dump_path}", 1

    # Basic sanitization - ensure plugin name looks reasonable (alphanumeric, dots, underscores)
    # This is NOT foolproof security but a basic check.
    if not re.match(r"^[a-zA-Z0-9._-]+$", plugin):
         return "", f"Error: Invalid plugin name format: {plugin}", 1

    command = [
        PYTHON_EXECUTABLE,
        vol_path,
        "-f",
        dump_path,
        f"{profile}.{plugin}" # Combine profile and plugin correctly for Vol3
    ]

    # Add plugin arguments carefully
    if plugin_args:
        # Further sanitization could be added here depending on expected args
        command.extend(plugin_args)

    print(f"\nExecuting Volatility command: {' '.join(command)}") # For debugging

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit, handle it below
            timeout=300 # Add a timeout (e.g., 5 minutes)
        )
        print(f"Volatility exited with code: {result.returncode}") # For debugging
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", f"Error: '{PYTHON_EXECUTABLE}' or '{vol_path}' not found. Check paths.", 1
    except subprocess.TimeoutExpired:
         return "", f"Error: Volatility command timed out after 300 seconds.", 1
    except Exception as e:
        return "", f"An unexpected error occurred during Volatility execution: {e}", 1


def detect_profile(
    dump_path: str,
    volatility_path: Optional[str] = None
) -> Optional[str]:
    """
    Attempts to detect the OS profile using Volatility's windows.info.Info plugin.

    Args:
        dump_path: Path to the memory dump file.
        volatility_path: Optional override for the path to vol.py.

    Returns:
        The detected profile string (e.g., 'windows.info') or None if detection fails.
    """
    vol_path = volatility_path or _get_volatility_path()
    if not os.path.exists(dump_path):
        print(f"Error: Dump file not found at {dump_path}")
        return None

    # For Volatility 3, windows.info.Info is a good starting point
    # Linux uses linux.info.Info, Mac uses mac.info.Info
    # We might need to make this smarter later, but start with Windows.
    plugin_to_try = "windows.info.Info" # Adjust if needed for other OS types

    command = [
        PYTHON_EXECUTABLE,
        vol_path,
        "-f",
        dump_path,
        plugin_to_try
    ]

    print(f"\nAttempting profile detection: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120 # Profile detection can sometimes be slow
        )

        if result.returncode != 0:
            print(f"Profile detection failed. Stderr:\n{result.stderr}")
            # Try Linux?
            plugin_to_try = "linux.info.Info"
            command[-1] = plugin_to_try
            print(f"Retrying profile detection with Linux: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
            if result.returncode != 0:
                 print(f"Linux profile detection also failed. Stderr:\n{result.stderr}")
                 return None # Give up for now

        # --- Parsing Logic for Volatility 3 Info Output ---
        # Volatility 3's windows.info.Info provides symbol table suggestions.
        # Example output line:
        # [*] Searching for symbols based on major version 10 minor version 0 build number 19041 architecture AMD64
        # [*] Trying symbol table: windows/symbols/win10-x64-19041.pdb.json - Major version 10, minor version 0, build number 19041 - OK
        # Best Suited Symbol Tables: windows/symbols/win10-x64-19041.pdb.json (Major version 10, minor version 0, build number 19041)

        # Look for the "Best Suited Symbol Tables:" line
        # It might also just suggest based on PE header (e.g. "PE major version 10")
        # A simpler approach for Vol3 might be to just return the OS base, e.g., "windows" or "linux"
        # and let the specific plugins handle finding symbols.

        # Let's try a simpler approach first: just detect the OS base
        if "windows.info.Info" in result.stdout or "windows" in result.stdout.lower():
            print("Detected profile base: windows")
            return "windows" # Use the base OS name as the 'profile' context for the agent
        elif "linux.info.Info" in result.stdout or "linux" in result.stdout.lower():
            print("Detected profile base: linux")
            return "linux"
        # Add mac detection if needed
        # elif "mac.info.Info" in result.stdout or "mac" in result.stdout.lower():
        #     print("Detected profile base: mac")
        #     return "mac"
        else:
            # Fallback if no clear OS detected in output
            print("Could not reliably determine profile base from info plugin.")
            # Attempt parsing specific symbol table line as a last resort
            match = re.search(r"Best Suited Symbol Tables: (\S+)", result.stdout)
            if match:
                 # Extract the base part before the first dot if it looks like windows.plugin.Plugin
                 # This is heuristic and might need refinement.
                 full_plugin = match.group(1).split('.')[0]
                 print(f"Detected profile from symbol table line: {full_plugin}")
                 return full_plugin
            return None

    except FileNotFoundError:
        print(f"Error: '{PYTHON_EXECUTABLE}' or '{vol_path}' not found during profile detection.")
        return None
    except subprocess.TimeoutExpired:
         print("Error: Volatility profile detection timed out.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred during profile detection: {e}")
        return None

# --- Tool Definition for LangChain ---
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

class VolatilityPluginInput(BaseModel):
    plugin_name: str = Field(description="The full name of the Volatility 3 plugin to run (e.g., 'windows.pslist.PsList', 'linux.pslist.PsList').")
    plugin_args: Optional[List[str]] = Field(default=None, description="A list of optional arguments to pass to the plugin (e.g., ['--pid', '1234']).")

# We need a way to pass dump_path and profile to the tool when it's called by LangGraph.
# The tool function itself won't automatically know the state.
# A common pattern is to create a class or use a closure, but for simplicity,
# we'll rely on the agent node passing these in the state, and the tool node function extracting them.
# The @tool decorator makes this slightly tricky directly. Let's define the core logic
# and then wrap it in the LangGraph node later.

def run_volatility_tool_logic(plugin_name: str, plugin_args: Optional[List[str]], dump_path: str, profile: str) -> str:
    """Core logic for executing the Volatility tool."""
    if not dump_path or not profile:
        return "Error: dump_path or profile is missing in the current state."

    stdout, stderr, return_code = execute_volatility_plugin(
        dump_path=dump_path,
        profile=profile,
        plugin=plugin_name,
        plugin_args=plugin_args
    )

    if return_code == 0:
        if not stdout.strip():
            return f"Plugin '{plugin_name}' executed successfully but produced no output."
        # Consider truncating long output here if necessary
        return f"Success:\n```\n{stdout}\n```"
    else:
        return f"Error executing plugin '{plugin_name}' (code: {return_code}):\nStderr:\n```\n{stderr}\n```\nStdout:\n```\n{stdout}\n```"

# The actual tool function used by LangChain/LangGraph needs the specific signature.
# We'll handle passing state later in the agent node definition.
@tool("volatility_runner", args_schema=VolatilityPluginInput)
def volatility_runner_tool(plugin_name: str, plugin_args: Optional[List[str]] = None) -> str:
    """
    Runs a Volatility 3 plugin against the memory dump specified in the current context.
    Requires plugin_name (e.g., 'windows.pslist.PsList') and optional plugin_args (e.g., ['--pid', '1234']).
    The dump file path and profile must be available in the agent's state.
    """
    # This is a placeholder function signature for the @tool decorator.
    # The actual execution logic (run_volatility_tool_logic) will be called
    # within the LangGraph node, which has access to the state (dump_path, profile).
    # This tool definition primarily tells the LLM *how* to structure its request.
    return "Error: This tool function should not be called directly. It's invoked via the LangGraph tool node."
