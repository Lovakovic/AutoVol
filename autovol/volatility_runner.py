import os
import subprocess
import re
from typing import Optional, Tuple, List
from pathlib import Path
import shutil # For checking if 'vol' is on PATH

# VOLATILITY_PATH = os.getenv("VOLATILITY3_PATH") # No longer primary way
PYTHON_EXECUTABLE = "python3"

def _get_volatility_cmd() -> str:
    """Determines the command to run Volatility (e.g., 'vol' or a full path to vol.py)."""
    # Prioritize VOLATILITY3_PATH env var if it's explicitly set and points to vol.py
    env_path = os.getenv("VOLATILITY3_PATH")
    if env_path and env_path.endswith("vol.py") and os.path.exists(env_path):
        # This case is for local dev where user might point to a specific vol.py
        # In Docker, after pip install, 'vol' should be on PATH.
        # So, this path might not be used much in Docker but good for flexibility.
        return env_path # Caller will prepend PYTHON_EXECUTABLE

    # Default to 'vol' assuming it's on PATH (installed by pip install .)
    # We can check if 'vol' is on PATH, but subprocess will fail anyway if it's not.
    if shutil.which("vol"):
        return "vol" # 'vol' is on PATH, caller executes it directly
    elif env_path: # If VOLATILITY3_PATH was set but wasn't vol.py (e.g. just "vol")
        return env_path # Trust the env var
    else: # Fallback if VOLATILITY3_PATH not set and 'vol' not on PATH (should not happen in Docker)
        # This will likely cause failure later, which is fine.
        return "vol"

def _construct_command(base_cmd_path: str, args: List[str]) -> List[str]:
    if base_cmd_path.endswith(".py"):
        return [PYTHON_EXECUTABLE, base_cmd_path] + args
    return [base_cmd_path] + args


def list_all_available_plugins(volatility_path: Optional[str] = None) -> List[str]:
    vol_cmd_base = volatility_path or _get_volatility_cmd()
    command = _construct_command(vol_cmd_base, ["-h"])

    plugin_names = []
    try:
        print(f"\nListing all Volatility plugins with: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        plugin_line_regex = re.compile(r"^\s{4}([a-zA-Z0-9_.]+)(?:\s{2,}.*)?$")
        in_plugins_section = False
        past_plugin_header_line = False
        for line in result.stdout.splitlines():
            stripped_line = line.strip()
            if not in_plugins_section:
                if stripped_line == "Plugins:":
                    in_plugins_section = True
                continue
            if not past_plugin_header_line:
                if line.startswith("  PLUGIN"):
                    past_plugin_header_line = True
                continue
            if stripped_line.startswith("The following plugins could not be loaded"):
                break
            match = plugin_line_regex.match(line)
            if match:
                plugin_names.append(match.group(1))
        if not plugin_names:
            print("Warning: No plugins extracted from help output.")
        else:
            print(f"Successfully extracted {len(plugin_names)} plugin names.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Volatility help (Code: {e.returncode}). Stderr: {e.stderr[:500]}...")
        return []
    except subprocess.TimeoutExpired:
        print(f"Error: Volatility help command timed out.")
        return []
    except FileNotFoundError:
        print(f"Error: Volatility command ('{command[0]}') not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while listing plugins: {e}")
        return []
    return sorted(list(set(plugin_names)))

def execute_volatility_plugin(
  dump_path: str,
  profile: str, # Still passed for context, not always used in command
  plugin: str,
  plugin_args: Optional[List[str]] = None,
  volatility_path: Optional[str] = None,
  session_workspace_dir: Optional[str] = None
) -> Tuple[str, str, int]:
    vol_cmd_base = volatility_path or _get_volatility_cmd()
    if not os.path.exists(dump_path):
        return "", f"Error: Dump file not found at {dump_path}", 1
    if not re.match(r"^[a-zA-Z0-9._-]+$", plugin):
        return "", f"Error: Invalid plugin name format: {plugin}", 1

    cmd_list = ["-f", dump_path, plugin]
    if plugin_args:
        cmd_list.extend(plugin_args)
    
    command = _construct_command(vol_cmd_base, cmd_list)

    print(f"\nExecuting Volatility command: {' '.join(command)}")
    if session_workspace_dir:
        print(f"Setting Volatility CWD to: {session_workspace_dir}")

    original_cwd = os.getcwd()
    try:
        if session_workspace_dir:
            Path(session_workspace_dir).mkdir(parents=True, exist_ok=True)
            os.chdir(session_workspace_dir)
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        print(f"Volatility exited with code: {result.returncode}")
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", f"Error: Volatility command ('{command[0]}') not found.", 1
    except subprocess.TimeoutExpired:
        return "", f"Error: Volatility command timed out.", 1
    except Exception as e:
        return "", f"An unexpected error: {e}", 1
    finally:
        if session_workspace_dir and Path(original_cwd).exists(): # Ensure original_cwd is valid before chdir
            os.chdir(original_cwd)
            print(f"Restored CWD to: {original_cwd}")

def detect_profile(
  dump_path: str,
  volatility_path: Optional[str] = None
) -> Optional[str]:
    vol_cmd_base = volatility_path or _get_volatility_cmd()
    if not os.path.exists(dump_path):
        print(f"Error: Dump file not found at {dump_path}")
        return None

    info_plugins_to_try = ["windows.info.Info", "linux.info.Info", "mac.info.Info"]
    detected_os_base = None
    for plugin_to_try in info_plugins_to_try:
        command = _construct_command(vol_cmd_base, ["-f", dump_path, plugin_to_try])
        print(f"\nAttempting profile detection with: {plugin_to_try}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
            if result.returncode == 0 and result.stdout:
                if "windows" in plugin_to_try.lower() and "windows" in result.stdout.lower():
                    detected_os_base = "windows"
                    break
                # ... (rest of detection logic unchanged)
                elif "linux" in plugin_to_try.lower() and "linux" in result.stdout.lower():
                    detected_os_base = "linux"
                    break
                elif "mac" in plugin_to_try.lower() and ("mac" in result.stdout.lower() or "darwin" in result.stdout.lower()):
                    detected_os_base = "mac"
                    break
            else:
                print(f"Plugin {plugin_to_try} failed or gave no output. Stderr: {result.stderr[:200]}...")
        except FileNotFoundError:
            print(f"Error: Volatility command ('{command[0]}') not found for {plugin_to_try}.")
            return None 
        except subprocess.TimeoutExpired:
            print(f"Error: Volatility profile detection with {plugin_to_try} timed out.")
        except Exception as e:
            print(f"An unexpected error: {e}")
    if detected_os_base:
        print(f"Detected OS base: {detected_os_base}")
    else:
        print("Could not reliably determine OS base.")
    return detected_os_base

from langchain_core.tools import tool
from pydantic import BaseModel, Field

class VolatilityPluginInput(BaseModel):
  plugin_name: str = Field(description="Full plugin name (e.g., 'windows.pslist.PsList').")
  plugin_args: Optional[List[str]] = Field(default=None, description="Optional args (e.g., ['--pid', '1234']). Use ['--dump-dir', '.'] for file output to workspace.")

def run_volatility_tool_logic(
    plugin_name: str, 
    plugin_args: Optional[List[str]], 
    dump_path: str, 
    profile: str, 
    session_workspace_dir: str
) -> str:
    if not dump_path: return "Error: dump_path missing."
    if not session_workspace_dir: return "Error: session_workspace_dir missing."

    stdout, stderr, return_code = execute_volatility_plugin(
        dump_path=dump_path, profile=profile, plugin=plugin_name,
        plugin_args=plugin_args, session_workspace_dir=session_workspace_dir
    )
    # ... (rest of logic unchanged)
    if return_code == 0:
        if not stdout.strip() and not stderr.strip():
            return f"Plugin '{plugin_name}' executed successfully but produced no output. Files may have been written to workspace."
        elif not stdout.strip() and stderr.strip():
            return f"Plugin '{plugin_name}' no stdout, stderr: {stderr}. Files may have been written to workspace."
        return f"Success:\n```\n{stdout}\n```\n(Files may have been written to workspace. Parse stdout for names.)"
    else:
        return f"Error executing '{plugin_name}' (code: {return_code}):\nStderr:\n```\n{stderr}\n```\nStdout:\n```\n{stdout}\n```"

@tool("volatility_runner", args_schema=VolatilityPluginInput)
def volatility_runner_tool(plugin_name: str, plugin_args: Optional[List[str]] = None) -> str:
  """
  Runs Volatility 3 plugin. Dump path/profile from state. Executes in session workspace.
  Args:
    plugin_name: Full plugin name (e.g., 'windows.pslist.PsList').
    plugin_args: Optional args. For file output to workspace, use plugin-specific args like ['--dump-dir', '.']. Parse stdout for filenames. Use 'list_workspace_files' to confirm.
  """
  return "Placeholder. Graph tool node executes logic."

