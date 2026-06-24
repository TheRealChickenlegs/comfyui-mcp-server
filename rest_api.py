"""FastAPI REST API for ComfyUI — consumed by Open WebUI's OpenAPI tool import."""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from tools.helpers import build_markdown_response, register_and_build_response

logger = logging.getLogger("MCP_Server")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    workflow_id: str = Field(description="Workflow ID (e.g. generate_image_flux)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters (prompt, steps, width, height, …)")


class RegenerateRequest(BaseModel):
    asset_id: str = Field(description="Asset ID to regenerate")
    seed: Optional[int] = Field(None, description="New seed (None=random, -1=keep original)")
    param_overrides: Optional[Dict[str, Any]] = Field(None, description="Parameter overrides")


# ---------------------------------------------------------------------------
# App factory  (fastapi imported lazily so --stdio mode doesn't require it)
# ---------------------------------------------------------------------------

def create_rest_api(comfyui_client, workflow_manager, defaults_manager, asset_registry):
    """Create a configured FastAPI app with all ComfyUI REST endpoints."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="ComfyUI MCP — REST API",
        description="REST endpoints for ComfyUI image generation, workflow management, and asset browsing. "
                    "Import this spec into Open Web UI (Workspace → Tools → + → Import from OpenAPI) "
                    "to make every endpoint available as an LLM-callable tool.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        servers=[{"url": "/api/v1", "description": "ComfyUI REST API"}],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Workflows
    # -----------------------------------------------------------------------

    @app.get(
        "/workflows",
        summary="List available workflows",
        operation_id="list_workflows",
    )
    async def list_workflows():
        catalog = workflow_manager.get_workflow_catalog()
        return {"workflows": catalog, "count": len(catalog)}

    @app.post(
        "/workflows/{workflow_id}",
        summary="Run a specific workflow by ID",
        operation_id="run_workflow",
    )
    async def run_workflow(workflow_id: str, overrides: Optional[Dict[str, Any]] = None):
        if overrides is None:
            overrides = {}
        try:
            workflow = workflow_manager.load_workflow(workflow_id)
            if not workflow:
                raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

            workflow = workflow_manager.apply_workflow_overrides(
                workflow, workflow_id, overrides, defaults_manager
            )
            override_report = workflow.pop("__override_report__", None)
            output_preferences = workflow_manager._guess_output_preferences(workflow)

            result = comfyui_client.run_custom_workflow(
                workflow, preferred_output_keys=output_preferences,
            )
            response = register_and_build_response(result, workflow_id, asset_registry, tool_name=None, return_inline_preview=True, preview_fetch_base_url=comfyui_client.base_url)
            if override_report and override_report.get("overrides_dropped"):
                response["overrides_applied"] = override_report["overrides_applied"]
                response["overrides_dropped"] = override_report["overrides_dropped"]

            response["markdown"] = build_markdown_response(response, tool_name=None)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Workflow '{workflow_id}' failed")
            raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------------------------------------------------
    # Image generation (auto-registered workflows)
    # -----------------------------------------------------------------------

    @app.post(
        "/generate",
        summary="Generate content using an auto-registered workflow",
        operation_id="generate",
    )
    async def generate(body: GenerateRequest):
        definition = next(
            (td for td in (workflow_manager.tool_definitions or []) if td.workflow_id == body.workflow_id),
            None,
        )
        if not definition:
            avail = [td.workflow_id for td in (workflow_manager.tool_definitions or [])]
            raise HTTPException(
                status_code=404,
                detail=f"No auto-registered workflow '{body.workflow_id}'. Available: {avail}",
            )

        # Merge defaults for any missing params
        namespace = workflow_manager._determine_namespace(definition.workflow_id)
        resolved_params = dict(body.params)
        for pname, param in definition.parameters.items():
            if pname not in resolved_params and param.required:
                model_default = defaults_manager.get_default(namespace, pname, None)
                if model_default:
                    resolved_params[pname] = model_default

        try:
            workflow = workflow_manager.render_workflow(definition, resolved_params, defaults_manager)
            result = comfyui_client.run_custom_workflow(
                workflow, preferred_output_keys=definition.output_preferences,
            )
            response_data = register_and_build_response(
                result, definition.workflow_id, asset_registry,
                tool_name=definition.tool_name,
                return_inline_preview=True,
                preview_fetch_base_url=comfyui_client.base_url,
            )
            response_data["markdown"] = build_markdown_response(response_data, tool_name=definition.tool_name)
            return response_data
        except Exception as e:
            logger.exception(f"Generation '{body.workflow_id}' failed")
            raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------------------------------------------------
    # Regenerate
    # -----------------------------------------------------------------------

    @app.post(
        "/regenerate",
        summary="Regenerate an existing asset with optional parameter overrides",
        operation_id="regenerate",
    )
    async def regenerate(body: RegenerateRequest):
        import copy

        asset = asset_registry.get_asset(body.asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset {body.asset_id} not found")

        original_workflow = asset.submitted_workflow
        if not original_workflow:
            raise HTTPException(status_code=400, detail="No workflow data stored for this asset")

        workflow = copy.deepcopy(original_workflow)
        param_overrides = body.param_overrides or {}

        # Apply overrides via the same helper used by the MCP regenerate tool
        from tools.generation import _update_workflow_params, _update_seed
        if param_overrides:
            workflow = _update_workflow_params(workflow, param_overrides)
        workflow = _update_seed(workflow, body.seed)

        output_preferences = None
        if asset.workflow_id:
            if "image" in asset.workflow_id.lower():
                output_preferences = ("images", "image", "gifs", "gif")
            elif "audio" in asset.workflow_id.lower() or "song" in asset.workflow_id.lower():
                output_preferences = ("audio", "audios", "sound", "files")

        result = comfyui_client.run_custom_workflow(
            workflow, preferred_output_keys=output_preferences,
        )
        response_data = register_and_build_response(
            result, asset.workflow_id, asset_registry,
            tool_name="regenerate", session_id=asset.session_id,
            return_inline_preview=True,
            preview_fetch_base_url=comfyui_client.base_url,
        )
        response_data["markdown"] = build_markdown_response(response_data, tool_name="regenerate")
        return response_data

    # -----------------------------------------------------------------------
    # Assets
    # -----------------------------------------------------------------------

    @app.get(
        "/assets",
        summary="List recently generated assets",
        operation_id="list_assets",
    )
    async def list_assets(limit: int = 10, workflow_id: Optional[str] = None):
        assets = asset_registry.list_assets(limit=limit, workflow_id=workflow_id)
        return {
            "assets": [
                {
                    "asset_id": a.asset_id,
                    "asset_url": a.asset_url or a.get_asset_url(asset_registry.comfyui_base_url),
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "width": a.width,
                    "height": a.height,
                    "bytes_size": a.bytes_size,
                    "workflow_id": a.workflow_id,
                    "prompt_id": a.prompt_id,
                    "created_at": a.created_at.isoformat(),
                }
                for a in assets
            ],
            "count": len(assets),
        }

    @app.get(
        "/assets/{asset_id}",
        summary="Get asset metadata",
        operation_id="get_asset_metadata",
    )
    async def get_asset_metadata(asset_id: str):
        asset = asset_registry.get_asset(asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

        asset_url = asset.asset_url or asset.get_asset_url(asset_registry.comfyui_base_url)
        return {
            "asset_id": asset.asset_id,
            "asset_url": asset_url,
            "filename": asset.filename,
            "mime_type": asset.mime_type,
            "width": asset.width,
            "height": asset.height,
            "bytes_size": asset.bytes_size,
            "workflow_id": asset.workflow_id,
            "prompt_id": asset.prompt_id,
            "created_at": asset.created_at.isoformat(),
            "comfy_history": asset.comfy_history,
        }

    # -----------------------------------------------------------------------
    # Jobs / queue
    # -----------------------------------------------------------------------

    @app.get(
        "/queue",
        summary="Get ComfyUI queue status",
        operation_id="get_queue_status",
    )
    async def get_queue_status():
        try:
            queue_data = comfyui_client.get_queue()
            return {
                "queue_running": queue_data.get("queue_running", []),
                "queue_pending": queue_data.get("queue_pending", []),
                "running_count": len(queue_data.get("queue_running", [])),
                "pending_count": len(queue_data.get("queue_pending", [])),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/jobs/{prompt_id}",
        summary="Get job status and history",
        operation_id="get_job",
    )
    async def get_job(prompt_id: str):
        try:
            history = comfyui_client.get_history(prompt_id)
            if prompt_id in history:
                prompt_data = history[prompt_id]
                if "error" in prompt_data:
                    return {"status": "error", "prompt_id": prompt_id, "error": prompt_data["error"]}
                if "outputs" in prompt_data and prompt_data["outputs"]:
                    return {"status": "completed", "prompt_id": prompt_id, "outputs": prompt_data["outputs"]}
                return {"status": "processing", "prompt_id": prompt_id}
            return {"status": "not_found", "prompt_id": prompt_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/jobs/{prompt_id}/cancel",
        summary="Cancel a queued or running job",
        operation_id="cancel_job",
    )
    async def cancel_job(prompt_id: str):
        try:
            comfyui_client.cancel_prompt(prompt_id)
            return {"status": "cancelled", "prompt_id": prompt_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    @app.get(
        "/health",
        summary="Health check",
        operation_id="health",
        include_in_schema=True,
    )
    async def health():
        return {"status": "ok", "service": "comfyui-mcp-server"}

    # -----------------------------------------------------------------------
    # File serving from shared output volume
    # -----------------------------------------------------------------------

    @app.get(
        "/assets/file/{filename}",
        summary="Serve a generated asset file from the shared output volume",
        operation_id="get_asset_file",
    )
    async def get_asset_file(filename: str, subfolder: str = ""):
        import os
        from pathlib import Path
        from fastapi.responses import FileResponse
        import mimetypes

        output_root = os.environ.get("COMFYUI_OUTPUT_ROOT")
        if not output_root:
            raise HTTPException(
                503,
                detail="COMFYUI_OUTPUT_ROOT not set — cannot serve files from disk",
            )
        filepath = Path(output_root) / subfolder / filename
        if not filepath.is_file():
            raise HTTPException(404, detail=f"File not found: {filename}")

        mime, _ = mimetypes.guess_type(str(filepath))
        return FileResponse(str(filepath), media_type=mime or "application/octet-stream")

    return app
