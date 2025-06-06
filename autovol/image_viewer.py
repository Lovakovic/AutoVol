import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

from .image_utils import ImageUtils


class ViewImageInput(BaseModel):
    """Input schema for the view_image_file tool."""
    file_path: str = Field(
        description="Path to the image file to view and analyze. Should be relative to the session workspace or absolute path within workspace."
    )
    analysis_prompt: Optional[str] = Field(
        default=None,
        description="Optional specific analysis prompt for the image. If not provided, a general forensic analysis will be performed."
    )


def create_multimodal_message(text_content: str, image_path: str) -> HumanMessage:
    """
    Create a properly formatted HumanMessage with image content for VertexAI.
    
    Args:
        text_content: Text prompt for image analysis
        image_path: Path to the image file
        
    Returns:
        HumanMessage with multimodal content
        
    Raises:
        ValueError: If image cannot be processed
    """
    try:
        base64_data, mime_type = ImageUtils.encode_image_to_base64(image_path)
        
        multimodal_content = [
            {"type": "text", "text": text_content},
            {
                "type": "image",
                "source_type": "base64",
                "data": base64_data,
                "mime_type": mime_type
            }
        ]
        
        return HumanMessage(content=multimodal_content)
        
    except Exception as e:
        raise ValueError(f"Failed to create multimodal message: {str(e)}")


def analyze_image_with_llm(
    image_path: str,
    analysis_prompt: str,
    llm: ChatVertexAI
) -> str:
    """
    Perform direct multimodal LLM invocation for image analysis.
    
    Args:
        image_path: Path to the image file
        analysis_prompt: Text prompt for analysis
        llm: ChatVertexAI instance
        
    Returns:
        Analysis result from the LLM
        
    Raises:
        ValueError: If analysis fails
    """
    try:
        multimodal_message = create_multimodal_message(analysis_prompt, image_path)
        response = llm.invoke([multimodal_message])
        return response.content
        
    except Exception as e:
        raise ValueError(f"Image analysis failed: {str(e)}")


def view_image_file_logic(
    file_path: str,
    session_workspace_dir: str,
    analysis_prompt: Optional[str] = None,
    llm_instance: Optional[ChatVertexAI] = None
) -> Dict[str, Any]:
    """
    Core logic for image viewing and analysis.
    
    Args:
        file_path: Path to the image file (relative to workspace or absolute)
        session_workspace_dir: Session workspace directory
        analysis_prompt: Optional specific analysis prompt
        llm_instance: ChatVertexAI instance for analysis
        
    Returns:
        Dictionary with analysis results, metadata, and status
    """
    result = {
        "success": False,
        "error_message": None,
        "image_info": None,
        "analysis_result": None,
        "analysis_prompt_used": None
    }
    
    try:
        # Resolve file path
        workspace_path = Path(session_workspace_dir)
        
        if os.path.isabs(file_path):
            # Absolute path - ensure it's within workspace for security
            abs_path = Path(file_path)
            try:
                abs_path.relative_to(workspace_path)
                resolved_path = abs_path
            except ValueError:
                result["error_message"] = f"Security error: Image path outside workspace: {file_path}"
                return result
        else:
            # Relative path - resolve relative to workspace
            resolved_path = workspace_path / file_path
        
        if not resolved_path.exists():
            result["error_message"] = f"Image file not found: {resolved_path}"
            return result
        
        # Get image information
        image_info = ImageUtils.get_image_info(str(resolved_path))
        result["image_info"] = image_info
        
        if not image_info["is_valid"]:
            result["error_message"] = f"Invalid or unsupported image format: {resolved_path}"
            return result
        
        if not image_info["size_constraint_ok"]:
            result["error_message"] = f"Image size constraint violation: {image_info['size_constraint_reason']}"
            return result
        
        # Prepare analysis prompt
        if not analysis_prompt:
            analysis_prompt = """Analyze this image from a digital forensics perspective. Please provide:

1. **Visual Content Description**: What is visible in the image? Describe any text, objects, UI elements, or scenes.

2. **Forensic Relevance**: Identify any potentially relevant forensic artifacts such as:
   - File paths, URLs, or system information visible
   - Application interfaces or error messages
   - Timestamps or metadata visible in the image
   - User activity indicators
   - Network or system configuration details

3. **Technical Analysis**: Note any technical details like:
   - Image quality and potential compression artifacts
   - Screenshot characteristics (if applicable)
   - Potential source applications or systems

4. **Security Implications**: Identify any security-relevant information such as:
   - Credentials or sensitive data visible
   - Malware-related content
   - System vulnerabilities or misconfigurations
   - Suspicious activities or anomalies

5. **Investigation Value**: Assess the potential value of this image for digital forensic investigation.

Please be thorough but concise in your analysis."""
        
        result["analysis_prompt_used"] = analysis_prompt
        
        # Perform LLM analysis if instance provided
        if llm_instance:
            try:
                analysis_result = analyze_image_with_llm(
                    str(resolved_path),
                    analysis_prompt,
                    llm_instance
                )
                result["analysis_result"] = analysis_result
            except Exception as e:
                result["error_message"] = f"LLM analysis failed: {str(e)}"
                return result
        else:
            result["analysis_result"] = "Image validated and ready for analysis. No LLM instance provided for analysis."
        
        result["success"] = True
        
    except Exception as e:
        result["error_message"] = f"Unexpected error during image analysis: {str(e)}"
    
    return result


@tool("view_image_file", args_schema=ViewImageInput)
def view_image_file_tool(file_path: str, analysis_prompt: Optional[str] = None) -> str:
    """
    View and analyze image files found in the workspace during forensic analysis.
    
    This tool can analyze various image formats (JPEG, PNG, BMP, GIF, TIFF, WebP) and provide
    forensic insights about their content. It's particularly useful for analyzing:
    - Screenshots extracted from memory dumps
    - Cached images from web browsers
    - Application icons and interface elements
    - Documents or files with embedded images
    - Visual evidence of user activity
    
    The tool performs both technical validation and content analysis of images.
    
    Args:
        file_path: Path to the image file (relative to workspace or absolute path within workspace)
        analysis_prompt: Optional specific analysis prompt. If not provided, performs general forensic analysis.
        
    Returns:
        Analysis results including image metadata and forensic insights.
    """
    return "Image analysis placeholder. Actual execution handled by the agent's tool executor node."