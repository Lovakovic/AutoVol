import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from .image_utils import ImageUtils


class ListWorkspaceFilesInput(BaseModel):
  relative_path: Optional[str] = Field(
    default=".",
    description="Optional. A relative path within the session workspace to list. Defaults to the workspace root ('.')."
  )


def list_workspace_files_logic(session_workspace_dir: str, relative_path: Optional[str] = ".") -> str:
  """
  Lists files and directories within a given path inside the session workspace.
  Enhanced with image file detection and metadata.
  """
  if relative_path is None:
    relative_path = "."

  try:
    # Sanitize and resolve the target path
    base_path = Path(session_workspace_dir).resolve()
    target_path_str = (base_path / Path(relative_path)).resolve()

    # Security check: Ensure target_path is still within base_path
    if base_path not in target_path_str.parents and target_path_str != base_path:
      return f"Error: Access denied. Relative path '{relative_path}' attempts to go outside the session workspace."

    if not target_path_str.exists():
      return f"Error: Path does not exist in workspace: '{target_path_str}' (resolved from relative: '{relative_path}')"

    items = []
    image_files = []
    
    for item in target_path_str.iterdir():
      try:
        stat = item.stat()
        item_type = "dir" if item.is_dir() else "file"
        size_str = f"{stat.st_size} bytes" if item.is_file() else ""
        # Getmtime returns timestamp, convert to human-readable
        mtime_str = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if it's an image file
        if item.is_file():
          extension = item.suffix.lower()
          if extension in ImageUtils.SUPPORTED_FORMATS:
            is_valid_image = ImageUtils.validate_image_format(str(item))
            if is_valid_image:
              image_info = ImageUtils.get_image_info(str(item))
              image_meta = f"[IMAGE: {image_info.get('format', '?')} {image_info.get('dimensions', '?')}px]"
              items.append(f"- {item_type}: {item.name} ({size_str.strip()}) (Modified: {mtime_str}) {image_meta}".replace(" ()", ""))
              image_files.append(str(item.relative_to(base_path)))
            else:
              items.append(f"- {item_type}: {item.name} ({size_str.strip()}) (Modified: {mtime_str}) [INVALID IMAGE]".replace(" ()", ""))
          else:
            items.append(f"- {item_type}: {item.name} ({size_str.strip()}) (Modified: {mtime_str})".replace(" ()", ""))
        else:
          items.append(f"- {item_type}: {item.name} ({size_str.strip()}) (Modified: {mtime_str})".replace(" ()", ""))
          
      except Exception as e:
        items.append(f"- ?type?: {item.name} (Error stating: {e})")

    if not items:
      return f"The directory '{target_path_str.name}' (relative: '{relative_path}') is empty."

    result = f"Contents of workspace path '{target_path_str.name}' (relative: '{relative_path}'):\n" + "\n".join(items)
    
    # Add image summary if images were found
    if image_files:
      result += f"\n\n**Image Files Found ({len(image_files)}):**\n"
      result += "These files can be analyzed using the 'view_image_file' tool for forensic insights:\n"
      for img_path in image_files:
        result += f"- {img_path}\n"

    return result

  except Exception as e:
    return f"Error listing workspace files for relative path '{relative_path}': {e}"


def find_images_in_workspace(session_workspace_dir: str, recursive: bool = True) -> List[Dict[str, Any]]:
  """
  Find all image files in the workspace with metadata.
  
  Returns:
    List of dictionaries with image file information
  """
  image_files = []
  
  try:
    base_path = Path(session_workspace_dir)
    if not base_path.exists():
      return image_files
    
    discovered_images = ImageUtils.find_images_in_directory(str(base_path), recursive=recursive)
    
    for img_path in discovered_images:
      try:
        relative_path = Path(img_path).relative_to(base_path)
        image_info = ImageUtils.get_image_info(img_path)
        
        image_files.append({
          'absolute_path': img_path,
          'relative_path': str(relative_path),
          'info': image_info
        })
      except Exception:
        # Skip files that can't be processed
        continue
        
  except Exception:
    pass
    
  return image_files


@tool("list_workspace_files", args_schema=ListWorkspaceFilesInput)
def list_workspace_files_tool(relative_path: Optional[str] = ".") -> str:
  """
  Lists files and directories in the current session's workspace.
  You can specify a 'relative_path' within the workspace to list a subdirectory.
  If 'relative_path' is not provided, it lists the root of the session workspace.
  Output includes item type (file/dir), name, size (for files), and last modified time.
  
  **Enhanced Image Detection**: This tool now automatically identifies image files 
  (JPEG, PNG, BMP, GIF, TIFF, WebP) and displays their metadata. When image files 
  are found, it provides a summary suggesting the use of 'view_image_file' tool 
  for forensic analysis.
  
  Use this to see files created by Volatility (after instructing it to output to '.')
  or files generated by your Python scripts, including any extracted or dumped image files.
  """
  # This docstring is for the LLM.
  # The actual logic is in list_workspace_files_logic and will be called by the graph node,
  # which will supply the session_workspace_dir.
  return "Workspace listing placeholder. Actual execution handled by the agent's tool node."
