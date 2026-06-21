"""Workflow management tools for ComfyUI MCP Server"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage
from asset_processor import fetch_asset_bytes
from tools.helpers import build_markdown_response, register_and_build_response

logger = logging.getLogger("MCP_Server")

_claude_mode = os.getenv("CLAUDE_CODE", "false").strip().lower() in ("1", "true", "yes")


def register_workflow_tools(
    mcp: FastMCP,
    workflow_manager,
    comfyui_client,
    defaults_manager,
    asset_registry
):
    """Register workflow tools with the MCP server"""
    
    @mcp.tool()
    def list_workflows() -> dict:
        """List all available workflows in the workflow directory.
        
        Returns a catalog of workflows with their IDs, names, descriptions,
        available inputs, and optional metadata.
        """
        catalog = workflow_manager.get_workflow_catalog()
        return {
            "workflows": catalog,
            "count": len(catalog),
            "workflow_dir": str(workflow_manager.workflows_dir)
        }

    @mcp.tool()
    def run_workflow(
        workflow_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        return_inline_preview: bool = False
    ) -> dict:
        """Run a saved ComfyUI workflow with constrained parameter overrides.
        
        Args:
            workflow_id: The workflow ID (filename stem, e.g., "generate_image")
            overrides: Optional dict of parameter overrides (e.g., {"prompt": "a cat", "width": 1024})
            options: Optional dict of execution options (reserved for future use)
            return_inline_preview: If True, include a small thumbnail base64 in response (256px, ~100KB)
        
        Returns:
            Result with asset_url, workflow_id, and execution metadata. If return_inline_preview=True,
            also includes inline_preview_base64 for immediate viewing.
        """
        if overrides is None:
            overrides = {}
        
        # Load workflow
        workflow = workflow_manager.load_workflow(workflow_id)
        if not workflow:
            return {"error": f"Workflow '{workflow_id}' not found"}
        
        # Extract seed from overrides if provided (handled separately by unified method)
        seed = overrides.pop("seed", None)
        
        try:
            # Apply overrides with constraints using unified method
            workflow = workflow_manager.apply_parameter_overrides(
                workflow, workflow_id, overrides, defaults_manager, seed=seed
            )

            # Extract and remove override report before submitting to ComfyUI
            override_report = workflow.pop("__override_report__", None)

            # Determine output preferences
            output_preferences = workflow_manager._guess_output_preferences(workflow)

            # Execute workflow
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=output_preferences,
            )

            # Register asset and build response
            response = register_and_build_response(
                result,
                workflow_id,
                asset_registry,
                tool_name=None,
                return_inline_preview=return_inline_preview,
                session_id=None
            )

            # Include override report so the agent can see what was applied/dropped
            if override_report and override_report.get("overrides_dropped"):
                response["overrides_applied"] = override_report["overrides_applied"]
                response["overrides_dropped"] = override_report["overrides_dropped"]

            # Default: return a plain Markdown string so OWUI renders the image inline.
            # When CLAUDE_CODE=true, also include an MCPImage content item for clients
            # that natively support the image content type (Claude Desktop, Cursor).
            markdown_text = build_markdown_response(response, tool_name=None)
            image_url = response.get("asset_url") or response.get("image_url")

            if _claude_mode:
                try:
                    if image_url:
                        image_bytes = fetch_asset_bytes(image_url, timeout=5)
                        return [
                            markdown_text,
                            MCPImage(data=image_bytes, format=response.get("mime_type", "image/png").split("/")[-1])
                        ]
                except Exception as e:
                    logger.warning(f"Failed to create Image content for tool result: {e}")

            return markdown_text
        except Exception as exc:
            logger.exception("Workflow '%s' failed", workflow_id)
            return {"error": str(exc)}
