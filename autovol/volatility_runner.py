import os
import subprocess
import re
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import shutil

# Pydantic and Langchain tool imports at the top
from pydantic import BaseModel, Field
from langchain_core.tools import tool

PYTHON_EXECUTABLE = "python3"

def _get_volatility_cmd() -> str:
    env_path = os.getenv("VOLATILITY3_PATH")
    if env_path and env_path.endswith("vol.py") and os.path.exists(env_path):
        return env_path
    if shutil.which("vol"):
        return "vol"
    elif env_path:
        return env_path
    else:
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
  profile: str,
  plugin: str,
  plugin_args: Optional[List[str]] = None,
  volatility_path: Optional[str] = None,
  session_workspace_dir: Optional[str] = None
) -> Tuple[str, str, int, Optional[str]]:
    vol_cmd_base = volatility_path or _get_volatility_cmd()
    output_file_rel_path = None

    if not os.path.exists(dump_path):
        return "", f"Error: Dump file not found at {dump_path}", 1, None
    if not re.match(r"^[a-zA-Z0-9._-]+$", plugin):
        return "", f"Error: Invalid plugin name format: {plugin}", 1, None

    cmd_list = ["-f", dump_path, plugin]
    if plugin_args:
        cmd_list.extend(plugin_args)

    standardized_output_filename = ""
    if session_workspace_dir:
        sane_plugin_name = "".join(c if c.isalnum() else '_' for c in plugin)
        standardized_output_filename = f"vol_{sane_plugin_name}_output.txt"
        output_file_rel_path = standardized_output_filename

    command = _construct_command(vol_cmd_base, cmd_list)

    print(f"\nExecuting Volatility command: {' '.join(command)}")
    if session_workspace_dir:
        print(f"Setting Volatility CWD to: {session_workspace_dir}")
        if standardized_output_filename:
             print(f"Volatility stdout will also be saved to: {Path(session_workspace_dir) / standardized_output_filename}")


    original_cwd = os.getcwd()
    try:
        if session_workspace_dir:
            Path(session_workspace_dir).mkdir(parents=True, exist_ok=True)
            os.chdir(session_workspace_dir)

        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        print(f"Volatility exited with code: {result.returncode}")

        if session_workspace_dir and standardized_output_filename:
            try:
                with open(standardized_output_filename, "w", encoding="utf-8") as f_out:
                    f_out.write(result.stdout)
                print(f"Successfully saved Volatility stdout to workspace file: {standardized_output_filename}")
            except Exception as e_save:
                print(f"Error saving Volatility stdout to {standardized_output_filename}: {e_save}")
                output_file_rel_path = None # Indicate saving failed

        return result.stdout, result.stderr, result.returncode, output_file_rel_path

    except FileNotFoundError:
        return "", f"Error: Volatility command ('{command[0]}') not found.", 1, None
    except subprocess.TimeoutExpired:
        return "", f"Error: Volatility command timed out.", 1, None
    except Exception as e:
        return "", f"An unexpected error: {e}", 1, None
    finally:
        if session_workspace_dir and Path(original_cwd).exists():
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


class VolatilityPluginInput(BaseModel):
  plugin_name: str = Field(description="Full plugin name (e.g., 'windows.pslist.PsList').")
  plugin_args: Optional[List[str]] = Field(default=None, description="Optional args (e.g., ['--pid', '1234']). If the plugin dumps files (e.g. procdump, dumpfiles), it's best to use its native arguments like ['--dump-dir', '.'] to direct files into the session workspace. The main output of the plugin (stdout) will also be saved to a file in the workspace (e.g., 'vol_windows.pslist.PsList_output.txt') for your Python scripts to read.")

def _create_smart_preview(output: str, max_chars: int = 15000, max_lines: int = 15) -> str:
    """Create a smart preview showing first and last portions of output.
    
    Args:
        output: The full output string
        max_chars: Maximum characters to show from start and end (total will be 2x this)
        max_lines: Maximum non-empty lines to show from start and end (total will be 2x this)
    
    Returns:
        A preview string with first and last portions, or full output if small enough
    """
    # If output is small enough, return it as-is
    if len(output) <= max_chars * 2:
        return output
    
    # Split into lines and filter out empty lines
    lines = output.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    
    # If we have few enough non-empty lines, return all
    if len(non_empty_lines) <= max_lines * 2:
        return output
    
    # Otherwise, create a smart preview
    # First, try line-based approach
    if len(non_empty_lines) > max_lines * 2:
        first_lines = non_empty_lines[:max_lines]
        last_lines = non_empty_lines[-max_lines:]
        
        first_part = '\n'.join(first_lines)
        last_part = '\n'.join(last_lines)
        
        # Check if this fits within character limit
        if len(first_part) + len(last_part) <= max_chars * 2:
            total_lines = len(lines)
            omitted_lines = total_lines - (max_lines * 2)
            return f"{first_part}\n\n... [{omitted_lines} lines omitted] ...\n\n{last_part}"
    
    # Fall back to character-based approach
    first_part = output[:max_chars]
    last_part = output[-max_chars:]
    
    # Try to break at line boundaries for cleaner output
    if '\n' in first_part:
        first_part = first_part[:first_part.rfind('\n')]
    if '\n' in last_part:
        last_part = last_part[last_part.find('\n') + 1:]
    
    total_chars = len(output)
    omitted_chars = total_chars - len(first_part) - len(last_part)
    
    return f"{first_part}\n\n... [{omitted_chars} characters omitted] ...\n\n{last_part}"

def run_volatility_tool_logic(
    plugin_name: str,
    plugin_args: Optional[List[str]],
    dump_path: str,
    profile: str,
    session_workspace_dir: str
) -> Dict[str, Any]:
    if not dump_path: return {"error": "dump_path missing."}
    if not session_workspace_dir: return {"error": "session_workspace_dir missing."}

    stdout, stderr, return_code, saved_stdout_filename = execute_volatility_plugin(
        dump_path=dump_path,
        profile=profile,
        plugin=plugin_name,
        plugin_args=plugin_args,
        session_workspace_dir=session_workspace_dir
    )

    # Create a smart preview for the agent
    stdout_preview = _create_smart_preview(stdout, max_chars=15000, max_lines=15)
    
    result_dict = {
        "stdout_preview": stdout_preview,
        "stderr": stderr,
        "return_code": return_code,
        "saved_workspace_stdout_file": saved_stdout_filename
    }

    if return_code != 0:
        result_dict["error_message"] = (f"Error executing '{plugin_name}' (code: {return_code}):\n"
                                        f"Stderr:\n```\n{stderr}\n```\n"
                                        f"Stdout (if any):\n```\n{stdout[:500]}\n```")
    elif not stdout.strip() and not stderr.strip() and not saved_stdout_filename: # Should not happen if saving is attempted
        result_dict["info_message"] = f"Plugin '{plugin_name}' executed successfully (RC=0) but produced no output and no stdout file was explicitly saved to workspace (this might be an issue or plugin doesn't output to stdout)."
    elif not stdout.strip() and not stderr.strip() and saved_stdout_filename:
         result_dict["info_message"] = f"Plugin '{plugin_name}' executed successfully (RC=0). It produced no direct console output, but its full stdout (which was empty) was saved to workspace file: '{saved_stdout_filename}'. Any files dumped by the plugin itself (e.g. via --dump-dir .) would also be in the workspace."
    elif saved_stdout_filename: # Normal success case
        result_dict["info_message"] = f"Plugin '{plugin_name}' executed successfully (RC=0). Full output saved to workspace file: '{saved_stdout_filename}'. Preview above."
    else: # Fallback if saved_stdout_filename is None but RC=0 (e.g. saving failed)
        result_dict["info_message"] = f"Plugin '{plugin_name}' executed successfully (RC=0), but its stdout could not be saved to a workspace file. Stdout preview above."

    return result_dict

@tool("volatility_runner", args_schema=VolatilityPluginInput)
def volatility_runner_tool(plugin_name: str, plugin_args: Optional[List[str]] = None) -> str:
  """
  Runs a Volatility 3 plugin. The memory dump path and OS profile are taken from the current analysis session state.
  The plugin will be executed with the session's dedicated workspace directory as its Current Working Directory (CWD).

  IMPORTANT:
  1.  The full standard output (stdout) of the Volatility plugin will ALWAYS be saved to a uniquely named text file directly within the session workspace (e.g., 'vol_windows.pslist.PsList_output.txt'). The filename will be provided back to you.
  2.  If the Volatility plugin itself is designed to dump files (e.g., `windows.procdump.ProcDump`, `linux.dumpfiles.DumpFiles`, `memmap`), you should use the plugin's specific arguments (like `--dump-dir .` or `--output-dir .`) to instruct it to save these files into the current directory (which is the session workspace).
  3.  Your subsequent Python scripts (using the `python_interpreter` tool) can then read the main stdout file (e.g., 'vol_windows.pslist.PsList_output.txt') or any other files dumped by the plugin directly from the session workspace using their relative filenames.

  Args:
    plugin_name (str): The full name of the Volatility plugin to run (e.g., 'windows.pslist.PsList', 'linux.bash.Bash').
    plugin_args (Optional[List[str]]): A list of additional command-line arguments for the plugin (e.g., ['--pid', '1234'], or ['--dump-dir', '.']).
  """
  return "Volatility plugin execution placeholder."


# --- New Tool for Getting Plugin Help ---
class GetPluginHelpInput(BaseModel):
    plugin_name: str = Field(description="The full name of the Volatility 3 plugin to get help for (e.g., 'windows.pslist.PsList').")

def get_volatility_plugin_help_logic(plugin_name: str, volatility_path: Optional[str] = None) -> Dict[str, str]:
    vol_cmd_base = volatility_path or _get_volatility_cmd()
    if not re.match(r"^[a-zA-Z0-9._-]+$", plugin_name):
        return {"error": f"Invalid plugin name format: {plugin_name}"}

    # Construct command to get help for a specific plugin
    command = _construct_command(vol_cmd_base, [plugin_name, "-h"])
    print(f"\nFetching help for plugin: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
        # Volatility's help for specific plugins usually returns 0 even if plugin invalid (shows main help)
        # or specific error for plugin args. We need to check stderr too.
        if "unrecognized arguments" in result.stderr.lower() and result.returncode !=0 : # A clear sign of bad args for a *valid* plugin
            return {"error": f"Plugin '{plugin_name}' likely exists, but the '-h' argument caused an error (or was not an option for it). Stderr: {result.stderr}", "help_text": result.stdout}
        elif "invalid choice" in result.stderr.lower() or "unknown plugin" in result.stderr.lower(): # Plugin name itself is bad
             return {"error": f"Plugin '{plugin_name}' not found or invalid. Stderr: {result.stderr}"}
        elif result.returncode == 0 and result.stdout: # Good help text
            return {"help_text": result.stdout}
        elif result.returncode != 0 : # Other errors
            return {"error": f"Error getting help for '{plugin_name}' (RC={result.returncode}). Stderr: {result.stderr}", "help_text": result.stdout}
        else: # RC=0 but no stdout
            return {"error": f"No help text returned for '{plugin_name}', though command ran (RC=0). Stderr: {result.stderr}"}

    except FileNotFoundError:
        return {"error": f"Volatility command ('{command[0]}') not found."}
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout getting help for '{plugin_name}'."}
    except Exception as e:
        return {"error": f"Unexpected error getting help for '{plugin_name}': {e}"}

@tool("get_volatility_plugin_help", args_schema=GetPluginHelpInput)
def get_volatility_plugin_help_tool(plugin_name: str) -> str:
    """
    Fetches the detailed help text for a specific Volatility 3 plugin.
    Use this if a `volatility_runner` command fails due to incorrect arguments for a plugin,
    or if you are unsure about a plugin's specific command-line options.
    The returned help text will show all available arguments and usage for that plugin.
    Args:
        plugin_name (str): The full name of the plugin (e.g., 'windows.pslist.PsList', 'linux.bash.Bash').
    """
    return "Plugin help fetching placeholder."
