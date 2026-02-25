import folder_paths
from .fal_utils import ApiHandler, ImageUtils, ResultProcessor

class Flux2KleinBaseEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "image_size": ([
                    "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"
                ], {"default": "square_hd"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 16}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "FAL/Image"

    def edit_image(
        self,
        prompt,
        image,
        image2=None,
        image3=None,
        image4=None,
        negative_prompt="",
        guidance_scale=5.0,
        seed=-1,
        num_inference_steps=28,
        image_size="square_hd",
        width=1024,
        height=1024,
        num_images=1,
        acceleration="regular",
        enable_safety_checker=True,
        output_format="png",
        sync_mode=False,
    ):
        # Collect all provided images
        images = [img for img in (image, image2, image3, image4) if img is not None]
        image_urls = ImageUtils.prepare_images(images) if images else []
        if not image_urls:
            return ApiHandler.handle_image_generation_error(
                "Flux2KleinBaseEdit", "No image provided."
            )

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
            "acceleration": acceleration,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-2/klein/9b/base/edit", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Flux2KleinBaseEdit", e)
