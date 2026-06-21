"""Workflow generation tools (auto-registered from workflow files)"""

import copy
import inspect
import logging
import os
import random
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage
from managers.workflow_manager import AUDIO_OUTPUT_KEYS, VIDEO_OUTPUT_KEYS
from models.workflow import WorkflowToolDefinition
from asset_processor import fetch_asset_bytes
from tools.helpers import build_markdown_response, register_and_build_response

_claude_mode = os.getenv("CLAUDE_CODE", "false").strip().lower() in ("1", "true", "yes")

logger = logging.getLogger("MCP_Server")


def register_workflow_generation_tools(
    mcp: FastMCP,
    workflow_manager,
    comfyui_client,
    defaults_manager,
    asset_registry
):
    """Register workflow-backed generation tools (e.g., generate_image, generate_song)"""
    
    def _register_workflow_tool(definition: WorkflowToolDefinition):
        def _tool_impl(*args, **kwargs):
            # Extract return_inline_preview if present (not a workflow parameter)
            return_inline_preview = kwargs.pop("return_inline_preview", False)
            # Session tracking can be added via request context in the future
            session_id = None
            
            # Coerce parameter types before signature binding
            # MCP/JSON-RPC may pass numbers as strings, so we need to convert them
            coerced_kwargs = {}
            param_dict = {p.name: p for p in definition.parameters.values()}
            
            for key, value in kwargs.items():
                if key in param_dict:
                    param = param_dict[key]
                    # Coerce to correct type if needed
                    if value is not None:
                        try:
                            # Handle string representations of numbers
                            if param.annotation is int:
                                if isinstance(value, str) and value.strip().isdigit():
                                    coerced_kwargs[key] = int(value)
                                elif isinstance(value, (int, float)):
                                    coerced_kwargs[key] = int(value)
                                else:
                                    coerced_kwargs[key] = value
                            elif param.annotation is float:
                                if isinstance(value, str):
                                    coerced_kwargs[key] = float(value)
                                elif isinstance(value, (int, float)):
                                    coerced_kwargs[key] = float(value)
                                else:
                                    coerced_kwargs[key] = value
                            else:
                                coerced_kwargs[key] = value
                        except (ValueError, TypeError) as e:
                            # If coercion fails, use original value and let validation handle it
                            logger.warning(f"Failed to coerce {key}={value!r} to {param.annotation.__name__}: {e}")
                            coerced_kwargs[key] = value
                    else:
                        coerced_kwargs[key] = None
                else:
                    # Unknown parameter, pass through
                    coerced_kwargs[key] = value
            
            bound = _tool_impl.__signature__.bind(*args, **coerced_kwargs)
            bound.apply_defaults()
            
            # Determine namespace using workflow manager (content-aware)
            namespace = workflow_manager._determine_namespace(definition.workflow_id)
            # Refine using output preferences (catches custom audio/video workflows)
            if definition.output_preferences == AUDIO_OUTPUT_KEYS:
                namespace = "audio"
            elif definition.output_preferences == VIDEO_OUTPUT_KEYS:
                namespace = "video"

            try:
                # Only validate model if the workflow actually has a 'model' parameter
                has_model_param = "model" in definition.parameters
                if has_model_param:
                    provided_model = dict(bound.arguments).get("model")
                    resolved_model = defaults_manager.get_default(namespace, "model", provided_model)

                    if resolved_model and not defaults_manager.is_model_valid(namespace, resolved_model):
                        is_valid, model_name, source = defaults_manager.validate_default_model(namespace)
                        available_models = list(defaults_manager._available_models_set)
                        sample_models = available_models[:5] if available_models else []

                        error_msg = (
                            f"Default model '{model_name}' (from {source} defaults) not found in ComfyUI checkpoints. "
                            f"Set a valid model via `set_defaults`, config file, or env var. "
                            f"Try `list_models` to see available checkpoints."
                        )
                        if sample_models:
                            error_msg += f" Available models: {sample_models}"

                        return {"error": error_msg}
                
                workflow = workflow_manager.render_workflow(definition, dict(bound.arguments), defaults_manager)
                result = comfyui_client.run_custom_workflow(
                    workflow,
                    preferred_output_keys=definition.output_preferences,
                )
                
                # Register asset and build response
                response_data = register_and_build_response(
                    result,
                    definition.workflow_id,
                    asset_registry,
                    tool_name=definition.tool_name,
                    return_inline_preview=return_inline_preview,
                    session_id=session_id
                )

                # Default: return a plain Markdown string so OWUI renders the image inline.
                # When CLAUDE_CODE=true, also include an MCPImage content item for clients
                # that natively support the image content type (Claude Desktop, Cursor).
                markdown_text = build_markdown_response(response_data, tool_name=definition.tool_name)
                image_url = response_data.get("asset_url") or response_data.get("image_url")

                if _claude_mode:
                    try:
                        if image_url:
                            image_bytes = fetch_asset_bytes(image_url, timeout=5)
                            return [
                                markdown_text,
                                MCPImage(data=image_bytes, format=response_data.get("mime_type", "image/png").split("/")[-1])
                            ]
                    except Exception as e:
                        logger.warning(f"Failed to create Image content for tool result: {e}")

                return markdown_text
                
            except Exception as exc:
                error_str = str(exc).lower()
                # Check if error is related to missing model (only if workflow uses models)
                if has_model_param and ("model" in error_str or "checkpoint" in error_str or "ckpt" in error_str):
                    comfyui_client.refresh_models()
                    defaults_manager.refresh_model_set()

                    provided_model = dict(bound.arguments).get("model")
                    resolved_model = defaults_manager.get_default(namespace, "model", provided_model)

                    if resolved_model and not defaults_manager.is_model_valid(namespace, resolved_model):
                        is_valid, model_name, source = defaults_manager.validate_default_model(namespace)
                        available_models = list(defaults_manager._available_models_set)
                        sample_models = available_models[:5] if available_models else []

                        error_msg = (
                            f"Default model '{model_name}' (from {source} defaults) not found in ComfyUI checkpoints. "
                            f"Set a valid model via `set_defaults`, config file, or env var. "
                            f"Try `list_models` to see available checkpoints."
                        )
                        if sample_models:
                            error_msg += f" Available models: {sample_models}"

                        return {"error": error_msg}

                logger.exception("Workflow '%s' failed", definition.workflow_id)
                return {"error": str(exc)}

        # Separate required and optional parameters to ensure correct ordering
        required_params = []
        optional_params = []
        annotations: Dict[str, Any] = {}
        
        for param in definition.parameters.values():
            annotation_type = param.annotation
            
            if param.required:
                parameter = inspect.Parameter(
                    name=param.name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation_type,
                )
                required_params.append(parameter)
            else:
                parameter = inspect.Parameter(
                    name=param.name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation_type,
                    default=None,
                )
                optional_params.append(parameter)
            annotations[param.name] = param.annotation
        
        # Add return_inline_preview as optional parameter
        optional_params.append(inspect.Parameter(
            name="return_inline_preview",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=bool,
            default=False,
        ))
        annotations["return_inline_preview"] = bool
        
        # Combine: required parameters first, then optional
        parameters = required_params + optional_params
        annotations["return"] = dict
        _tool_impl.__signature__ = inspect.Signature(parameters, return_annotation=dict)
        _tool_impl.__annotations__ = annotations
        _tool_impl.__name__ = f"tool_{definition.tool_name}"
        _tool_impl.__doc__ = definition.description
        mcp.tool(name=definition.tool_name, description=definition.description)(_tool_impl)
        logger.info(
            "Registered MCP tool '%s' for workflow '%s'",
            definition.tool_name,
            definition.workflow_id,
        )
    
    # Register all workflow-backed tools
    if workflow_manager.tool_definitions:
        for tool_definition in workflow_manager.tool_definitions:
            _register_workflow_tool(tool_definition)
    else:
        logger.info(
            "No workflow placeholders found in %s; add %s markers to enable auto tools",
            workflow_manager.workflows_dir,
            "PARAM_",
        )


def register_regenerate_tool(
    mcp: FastMCP,
    comfyui_client,
    asset_registry,
    workflow_manager
):
    """Register the regenerate tool for iterating on existing assets."""
    
    @mcp.tool()
    def regenerate(
        asset_id: str,
        seed: Optional[int] = None,
        return_inline_preview: bool = False,
        param_overrides: Optional[Dict[str, Any]] = None
    ) -> dict:
        """Regenerate an existing asset with optional parameter overrides.
        
        Retrieves the original workflow and parameters from the asset's provenance
        data, applies any overrides, and re-submits to ComfyUI.
        
        Args:
            asset_id: ID of the asset to regenerate
            seed: New random seed (None = generate new random seed, -1 = use original seed)
            return_inline_preview: If True, include a small thumbnail base64 in response
            param_overrides: Dict of workflow parameters to override (e.g., {"steps": 30, "cfg": 8.0, "prompt": "new prompt"})
        
        Returns:
            dict: New asset information with same structure as generate_* tools
        
        Examples:
            # Regenerate with different seed
            regenerate(asset_id="abc123")
            
            # Regenerate with higher quality settings
            regenerate(asset_id="abc123", param_overrides={"steps": 30, "cfg": 10.0})
            
            # Modify the prompt
            regenerate(asset_id="abc123", param_overrides={"prompt": "a beautiful sunset, oil painting style"})
            
            # Use exact same parameters (deterministic)
            regenerate(asset_id="abc123", seed=-1)
        """
        try:
            # Step 1: Retrieve original asset metadata
            asset = asset_registry.get_asset(asset_id)
            if not asset:
                return {"error": f"Asset {asset_id} not found (registry is in-memory and resets on restart). Generate a new asset to regenerate."}
            
            # Extract the stored workflow
            original_workflow = asset.submitted_workflow
            if not original_workflow:
                return {"error": "No workflow data stored for this asset. Cannot regenerate."}
            
            # Step 2: Deep copy workflow to avoid mutating the stored one
            workflow = copy.deepcopy(original_workflow)
            
            # Step 3: Apply parameter overrides and seed using unified method
            if param_overrides is None:
                param_overrides = {}
            workflow = workflow_manager.apply_parameter_overrides(
                workflow,
                asset.workflow_id,
                param_overrides,
                defaults_manager=None,  # No defaults for regeneration - use explicit overrides only
                seed=seed,
            )
            
            # Step 4: Determine output preferences from original workflow
            output_preferences = None
            if asset.workflow_id:
                if "image" in asset.workflow_id.lower():
                    output_preferences = ("images", "image", "gifs", "gif")
                elif "audio" in asset.workflow_id.lower() or "song" in asset.workflow_id.lower():
                    output_preferences = ("audio", "audios", "sound", "files")
                elif "video" in asset.workflow_id.lower():
                    output_preferences = ("videos", "video", "mp4", "mov", "webm")
            
            # Step 5: Submit to ComfyUI
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=output_preferences,
            )
            
            # Step 6: Register and return new asset
            return register_and_build_response(
                result,
                asset.workflow_id,
                asset_registry,
                tool_name="regenerate",
                return_inline_preview=return_inline_preview,
                session_id=asset.session_id  # Preserve original session
            )
        except Exception as e:
            logger.exception(f"Failed to regenerate asset {asset_id}")
            return {"error": f"Failed to regenerate: {str(e)}"}
