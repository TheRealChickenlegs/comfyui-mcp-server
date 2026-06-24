"""Shared helper functions for tool implementations"""

import json
import logging
from typing import Any, Dict, Optional

from asset_processor import encode_preview_for_mcp, fetch_asset_bytes, get_cache_key

logger = logging.getLogger("MCP_Server")


def register_and_build_response(
    result: Dict[str, Any],
    workflow_id: str,
    asset_registry,
    tool_name: Optional[str] = None,
    return_inline_preview: bool = False,
    session_id: Optional[str] = None,
    preview_fetch_base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Helper function to register asset and build response data.

    Eliminates code duplication between run_workflow() and _register_workflow_tool().

    Args:
        result: Result dict from comfyui_client.run_custom_workflow()
        workflow_id: Workflow ID
        asset_registry: AssetRegistry instance
        tool_name: Optional tool name (for workflow-backed tools)
        return_inline_preview: Whether to include inline preview
        session_id: Optional session identifier for conversation filtering
        preview_fetch_base_url: Internal ComfyUI URL for thumbnail fetch
            (e.g. "http://comfyui:8188"). Uses public asset_url if not provided.

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
    
    # Include inline preview if requested
    if return_inline_preview:
        try:
            # Try reading from local disk first (shared volume)
            import os
            image_bytes = None
            output_root = os.environ.get("COMFYUI_OUTPUT_ROOT")
            if output_root:
                from pathlib import Path
                local_path = Path(output_root) / asset_record.subfolder / asset_record.filename
                if local_path.is_file():
                    image_bytes = local_path.read_bytes()
            # Fall back to HTTP fetch if not available on disk
            if image_bytes is None:
                if preview_fetch_base_url:
                    preview_url = asset_record.get_asset_url(preview_fetch_base_url.rstrip("/"))
                elif asset_url:
                    preview_url = asset_url
                else:
                    preview_url = asset_record.get_asset_url(asset_registry.comfyui_base_url)
                image_bytes = fetch_asset_bytes(preview_url)

            if image_bytes:
                cache_key = get_cache_key(asset_record.asset_id, 256, 70)
                encoded = encode_preview_for_mcp(
                    image_bytes,
                    max_dim=256,
                    max_b64_chars=100_000,
                    quality=70,
                    cache_key=cache_key,
                )
                response_data["inline_preview_base64"] = f"data:{encoded.mime_type};base64,{encoded.b64}"
                response_data["inline_preview_mime_type"] = encoded.mime_type
                response_data["_inline_raw_bytes"] = encoded.raw_bytes
        except Exception as e:
            logger.warning(f"Failed to generate inline preview: {e}")
            # Don't fail the request if preview generation fails
    
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
    has_base64 = bool(response_data.get("inline_preview_base64"))
    raw_url = response_data.get("asset_url") or response_data.get("image_url") or ""

    lines: list[str] = []
    if tool_label:
        lines.append(f"### {tool_label}")

    # Append ![image] line with a clean URL — but skip it when we already have
    # an MCP ImageContent block to avoid browser mixed-content errors
    if raw_url and not has_base64:
        if "/view?" in raw_url:
            import os, urllib.parse
            parsed = urllib.parse.urlparse(raw_url)
            params = urllib.parse.parse_qs(parsed.query)
            filename = params.get("filename", [""])[0]
            sf = params.get("subfolder", [""])[0]
            if filename:
                mcp_url = os.environ.get("PUBLIC_MCP_URL", "http://localhost:3333")
                clean_url = f"{mcp_url}/api/v1/assets/file/{filename}"
                if sf:
                    clean_url += f"?subfolder={sf}"
                raw_url = clean_url
        lines.append("")
        lines.append(f"![image]({raw_url})")

    if not lines:
        return json.dumps(response_data)

    return "\n".join(lines).strip()
