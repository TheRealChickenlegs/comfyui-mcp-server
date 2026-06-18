"""Configuration tools for ComfyUI MCP Server"""

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP


def register_configuration_tools(
    mcp: FastMCP,
    comfyui_client,
    defaults_manager
):
    """Register configuration tools with the MCP server"""
    
    @mcp.tool()
    def list_models() -> dict:
        """List all available models in ComfyUI, organized by model type.
        
        Returns checkpoint models (for SD), UNet models (for Flux), CLIP models,
        and VAE models. This helps AI agents choose appropriate models for
        different generation workflows.
        """
        checkpoints = comfyui_client.available_models
        unets = comfyui_client.available_unets
        clips = comfyui_client.available_clips
        vaes = comfyui_client.available_vaes
        return {
            "checkpoint_models": checkpoints,
            "unet_models": unets,
            "clip_models": clips,
            "vae_models": vaes,
            "counts": {
                "checkpoints": len(checkpoints),
                "unets": len(unets),
                "clips": len(clips),
                "vaes": len(vaes),
            }
        }

    @mcp.tool()
    def get_defaults() -> dict:
        """Get current effective defaults for image, audio, and video generation.
        
        Returns merged defaults from all sources (runtime, config, env, hardcoded).
        Shows what values will be used when parameters are not explicitly provided.
        """
        return defaults_manager.get_all_defaults()

    @mcp.tool()
    def set_defaults(
        image: Optional[Dict[str, Any]] = None,
        audio: Optional[Dict[str, Any]] = None,
        video: Optional[Dict[str, Any]] = None,
        flux: Optional[Dict[str, Any]] = None,
        persist: bool = False
    ) -> dict:
        """Set runtime defaults for image, audio, video, and/or flux generation.
        
        Args:
            image: Optional dict of default values for SD image generation (e.g., {"model": "sd_xl_base_1.0.safetensors", "width": 1024})
            audio: Optional dict of default values for audio generation (e.g., {"model": "ace_step_v1_3.5b.safetensors", "seconds": 30})
            video: Optional dict of default values for video generation (e.g., {"model": "wan2.2_vae.safetensors", "width": 1280, "duration": 5})
            flux: Optional dict of default values for Flux generation (e.g., {"unet": "flux-dev.safetensors", "width": 1024, "steps": 20})
            persist: If True, write defaults to config file (~/.config/comfy-mcp/config.json). Otherwise, changes are ephemeral.
        
        Returns:
            Success status and any validation errors (e.g., invalid model names).
        """
        results = {}
        errors = []
        
        if image:
            result = defaults_manager.set_defaults("image", image, validate_models=True)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["image"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("image", image)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist image defaults: {persist_result['error']}")
        
        if audio:
            result = defaults_manager.set_defaults("audio", audio, validate_models=True)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["audio"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("audio", audio)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist audio defaults: {persist_result['error']}")
        
        if video:
            result = defaults_manager.set_defaults("video", video, validate_models=True)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["video"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("video", video)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist video defaults: {persist_result['error']}")
        
        if flux:
            result = defaults_manager.set_defaults("flux", flux, validate_models=False)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["flux"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("flux", flux)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist flux defaults: {persist_result['error']}")
        
        if errors:
            return {"success": False, "errors": errors}
        
        return {"success": True, "updated": results}
