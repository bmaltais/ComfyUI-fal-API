"""
ComfyUI-fal-API package initialization.

This module dynamically imports node modules and aggregates their class mappings.
"""
import importlib
import importlib.util

node_list = [
    "image_node",
    "video_node",
    "llm_node",
    "vlm_node",
    "trainer_node",
    "upscaler_node",
    "ltx_video_node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
