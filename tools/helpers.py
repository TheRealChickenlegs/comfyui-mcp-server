"""Shared helper functions for tool implementations"""

import json
import logging
from typing import Any, Dict, Optional



logger = logging.getLogger("MCP_Server")


def build_markdown_response(response_data: Dict[str, Any], tool_name: Optional[str] = None) -> str:
    """Build a Markdown string suitable for Open WebUI rendering.

    The primary text content contains the image embed as Markdown so OWUI
    renders the image inline.  The JSON metadata dict no longer appears as
    visible text in the chat; metadata is presented as a compact line below
    the image.

    When the response is an error or status handle, falls back to JSON so
    the LLM still receives structured data.
    """
    if "error" in response_data:
        return json.dumps(response_data)

    if response_data.get("status") == "running":
        return json.dumps(response_data)

    image_url = response_data.get("asset_url") or response_data.get("image_url") or ""
    tool_label = (tool_name or response_data.get("tool", "")).replace("_", " ").title()
    width = response_data.get("width")
    height = response_data.get("height")
    mime = response_data.get("mime_type", "")
    asset_id = response_data.get("asset_id", "")
    prompt_id = response_data.get("prompt_id", "")

    lines: list[str] = []

    if tool_label:
        lines.append(f"**{tool_label}**")

    if image_url:
        lines.append("")
        lines.append(f"![Generated Image]({image_url})")

    parts: list[str] = []
    if width and height:
        parts.append(f"{width}×{height}px")
    if mime:
        short_mime = mime.split("/")[-1]
        parts.append(short_mime)
    if asset_id:
        parts.append(f"ID: `{asset_id}`")
    if parts:
        if image_url or tool_label:
            lines.append("")
        lines.append("*" + " | ".join(parts) + "*")

    if prompt_id:
        lines.append(f"")
        lines.append(f"*Prompt: `{prompt_id}`*")

    if not lines:
        return json.dumps(response_data)

    return "\n".join(lines).strip()


def register_and_build_response(
    result: Dict[str, Any],
    workflow_id: str,
    asset_registry,
    tool_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to register asset and build response data.

    Eliminates code duplication between run_workflow() and _register_workflow_tool().

    Args:
        result: Result dict from comfyui_client.run_custom_workflow()
        workflow_id: Workflow ID
        asset_registry: AssetRegistry instance
        tool_name: Optional tool name (for workflow-backed tools)
        session_id: Optional session identifier for conversation filtering

    Returns:
        Response data dict with asset_id, asset_url, metadata, etc.
        If the workflow is still running (timeout), returns a job handle dict instead.
    """
    # If the result is a "still running" job handle, pass it through directly
    if result.get("status") == "running":
        return result

    # Register asset in registry using stable identity
    asset_metadata = result.get("asset_metadata", {})
    metadata = {"workflow_id": workflow_id}
    if tool_name:
        metadata["tool"] = tool_name
    
    asset_record = asset_registry.register_asset(
        filename=result.get("filename", ""),
        subfolder=result.get("subfolder", ""),
        folder_type=result.get("folder_type", "output"),
        workflow_id=workflow_id,
        prompt_id=result.get("prompt_id", ""),
        mime_type=asset_metadata.get("mime_type"),
        width=asset_metadata.get("width"),
        height=asset_metadata.get("height"),
        bytes_size=asset_metadata.get("bytes_size"),
        comfy_history=result.get("comfy_history"),
        submitted_workflow=result.get("submitted_workflow"),
        metadata=metadata,
        session_id=session_id
    )
    
    # Build response data
    # Use asset_record.asset_url (computed from stable identity)
    asset_url = asset_record.asset_url or result.get("asset_url", "")
    response_data = {
        "asset_id": asset_record.asset_id,
        "asset_url": asset_url,
        "image_url": asset_url,  # Backward compatibility
        "filename": asset_record.filename,  # Stable identity
        "subfolder": asset_record.subfolder,  # Stable identity
        "folder_type": asset_record.folder_type,  # Stable identity
        "workflow_id": workflow_id,
        "prompt_id": result.get("prompt_id"),
        "mime_type": asset_record.mime_type,
        "width": asset_record.width,
        "height": asset_record.height,
        "bytes_size": asset_record.bytes_size,
    }
    
    if tool_name:
        response_data["tool"] = tool_name

    # Add markdown image embed hint for the LLM to include in its response
    if asset_url:
        supported_types = ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif")
        if response_data.get("mime_type") in supported_types:
            response_data["image_markdown"] = f"![Generated Image]({asset_url})"
    
    # Include base64 image data if available (legacy)
    if "image_base64" in result:
        response_data["image_base64"] = result["image_base64"]
        response_data["image_mime_type"] = result.get("image_mime_type", "image/png")
    
    return response_data


def build_markdown_response(response_data: Dict[str, Any], tool_name: Optional[str] = None) -> str:
    """Build a Markdown string for OWUI to render the image inline.

    Formats the result as::

        ### {Tool Name}

        ![image]({asset_url})

    Falls back to JSON for error or running-job responses.
    """
    if "error" in response_data:
        return json.dumps(response_data)

    if response_data.get("status") == "running":
        return json.dumps(response_data)

    tool_label = (tool_name or response_data.get("tool", "")).replace("_", " ").title()
    raw_url = response_data.get("asset_url") or response_data.get("image_url") or ""

    lines: list[str] = []
    if tool_label:
        lines.append(f"### {tool_label}")

    # Use the asset URL directly if it's already a public HTTPS URL (ComfyUI public URL).
    # Only rewrite to MCP REST API endpoint if:
    # 1. COMFYUI_OUTPUT_ROOT is set (shared volume available)
    # 2. AND the URL is a ComfyUI internal /view? URL (not already public)
    if raw_url:
        import os, urllib.parse
        comfyui_output_root = os.environ.get("COMFYUI_OUTPUT_ROOT")
        public_mcp_url = os.environ.get("PUBLIC_MCP_URL", "")
        
        # Check if URL is a ComfyUI /view? pattern that needs rewriting
        if "/view?" in raw_url and comfyui_output_root and public_mcp_url:
            parsed = urllib.parse.urlparse(raw_url)
            params = urllib.parse.parse_qs(parsed.query)
            filename = params.get("filename", [""])[0]
            sf = params.get("subfolder", [""])[0]
            if filename:
                clean_url = f"{public_mcp_url}/api/v1/assets/file/{filename}"
                if sf:
                    clean_url += f"?subfolder={sf}"
                raw_url = clean_url
        # Otherwise use raw_url as-is (should be public ComfyUI HTTPS URL)
        
        lines.append("")
        lines.append(f"![image]({raw_url})")

    if not lines:
        return json.dumps(response_data)

    return "\n".join(lines).strip()
