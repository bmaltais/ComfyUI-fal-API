import folder_paths
from .fal_utils import ApiHandler, ImageUtils


class LTXImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "num_frames": ("INT", {"default": 121, "min": 1, "max": 1024}),
                "video_size": (
                    [
                        "auto",
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                    ],
                    {"default": "auto"},
                ),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "use_multiscale": ("BOOLEAN", {"default": True}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0}),
                "acceleration": (
                    ["none", "regular", "high", "full"],
                    {"default": "none"},
                ),
                "camera_lora": (
                    [
                        "none",
                        "dolly_in",
                        "dolly_out",
                        "dolly_left",
                        "dolly_right",
                        "jib_up",
                        "jib_down",
                        "static",
                    ],
                    {"default": "none"},
                ),
                "camera_lora_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "lora_path": ("STRING", {"default": ""}),
                "lora_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "video_output_type": (
                    ["X264 (.mp4)", "VP9 (.webm)", "PRORES4444 (.mov)", "GIF (.gif)"],
                    {"default": "X264 (.mp4)"},
                ),
                "video_quality": (
                    ["low", "medium", "high", "maximum"],
                    {"default": "high"},
                ),
                "video_write_mode": (
                    ["fast", "balanced", "small"],
                    {"default": "balanced"},
                ),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self,
        prompt,
        image,
        num_frames,
        video_size,
        generate_audio,
        use_multiscale,
        fps,
        acceleration,
        camera_lora,
        camera_lora_scale,
        negative_prompt="",
        seed=-1,
        lora_path="",
        lora_scale=1.0,
        video_output_type="X264 (.mp4)",
        video_quality="high",
        video_write_mode="balanced",
        enable_prompt_expansion=False,
        enable_safety_checker=True,
    ):
        image_url = ImageUtils.upload_image(image)
        if not image_url:
            return ApiHandler.handle_video_generation_error(
                "LTXImageToVideo", "Failed to upload image"
            )

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "num_frames": num_frames,
            "video_size": video_size,
            "generate_audio": generate_audio,
            "use_multiscale": use_multiscale,
            "fps": fps,
            "acceleration": acceleration,
            "camera_lora": camera_lora,
            "camera_lora_scale": camera_lora_scale,
            "negative_prompt": negative_prompt,
            "video_output_type": video_output_type,
            "video_quality": video_quality,
            "video_write_mode": video_write_mode,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed

        endpoint = "fal-ai/ltx-2-19b/distilled/image-to-video"
        if lora_path:
            endpoint += "/lora"
            arguments["loras"] = [{"path": lora_path, "scale": lora_scale}]

        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("LTXImageToVideo", e)


NODE_CLASS_MAPPINGS = {"LTXImageToVideo_fal": LTXImageToVideo}


NODE_DISPLAY_NAME_MAPPINGS = {"LTXImageToVideo_fal": "LTX Image to Video (fal)"}
