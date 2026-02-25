import configparser
import io
import os
import tempfile
import zipfile
import asyncio
import concurrent.futures
import hashlib
import json
import logging
from typing import Callable, Any

import numpy as np
import requests
import torch
from fal_client.client import SyncClient
from fal_client import AsyncClient
from PIL import Image


# Configure logging to suppress verbose HTTP request logging
def _configure_logging():
    """Suppress verbose logging from httpx, urllib3, and other libraries."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("fal_client").setLevel(logging.WARNING)


_configure_logging()


class FalPayload:
    """
    Builder for FAL API payloads to reduce duplication in nodes.

    This class provides a fluent interface for constructing the arguments
    dictionary passed to FAL API endpoints, handling common patterns like
    image/video sizes, seeds, and LoRAs.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the payload with base arguments."""
        self._arguments = kwargs

    def set_image_size(self, image_size: str, width: int | None = None, height: int | None = None) -> "FalPayload":
        """
        Add image size to the payload.

        Args:
            image_size: Predefined size name or 'custom'.
            width: Required if image_size is 'custom'.
            height: Required if image_size is 'custom'.

        Returns:
            The FalPayload instance for chaining.
        """
        if image_size == "custom" and width and height:
            self._arguments["image_size"] = {"width": width, "height": height}
        else:
            self._arguments["image_size"] = image_size
        return self

    def set_video_size(self, video_size: str, width: int | None = None, height: int | None = None) -> "FalPayload":
        """
        Add video size to the payload.

        Args:
            video_size: Predefined size name or 'custom'.
            width: Required if video_size is 'custom'.
            height: Required if video_size is 'custom'.

        Returns:
            The FalPayload instance for chaining.
        """
        if video_size == "custom" and width and height:
            self._arguments["video_size"] = {"width": width, "height": height}
        else:
            self._arguments["video_size"] = video_size
        return self

    def set_seed(self, seed: int) -> "FalPayload":
        """
        Add seed to the payload if valid.

        Args:
            seed: The seed value. -1 indicates random seed (omitted).

        Returns:
            The FalPayload instance for chaining.
        """
        if seed != -1:
            self._arguments["seed"] = seed
        return self

    def add_loras(self, loras_input: list[tuple[str, float]]) -> "FalPayload":
        """
        Add LoRAs to the payload.

        Args:
            loras_input: A list of (path_or_url, scale) tuples.

        Returns:
            The FalPayload instance for chaining.
        """
        loras = []
        for path, scale in loras_input:
            if path and path != "None":
                loras.append({"path": path, "scale": scale})

        if loras:
            self._arguments["loras"] = loras
        return self

    def update(self, **kwargs) -> "FalPayload":
        """Update arguments with multiple values."""
        self._arguments.update(kwargs)
        return self

    def build(self) -> dict[str, Any]:
        """Return the constructed arguments dictionary."""
        return self._arguments

    def __repr__(self) -> str:
        return f"FalPayload(args={list(self._arguments.keys())})"


class FileUploadCache:
    """
    Manages file upload caching with SHA-256 hashing and JSON persistence.
    
    This class encapsulates the responsibilities of caching file uploads to FAL,
    persisting the cache to disk, and retrieving cached URLs by file hash.
    
    Attributes:
        _cache_file_path (str): Path to the JSON cache file.
        _cache (dict): In-memory mapping of file hash -> upload URL.
        _loaded (bool): Whether the cache has been loaded from disk.
    
    Example:
        >>> cache = FileUploadCache()
        >>> url = cache.get_cached_url(file_hash)
        >>> cache.set_cached_url(file_hash, url)
        >>> cache.save()
    """

    def __init__(self, cache_file_path: str | None = None) -> None:
        """
        Initialize the FileUploadCache.
        
        Args:
            cache_file_path: Optional path to cache file. Defaults to
                'upload_cache.json' in parent directory of this module.
        """
        if cache_file_path is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            parent_dir = os.path.dirname(current_dir)
            cache_file_path = os.path.join(parent_dir, "upload_cache.json")
        
        self._cache_file_path = cache_file_path
        self._cache: dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Load cache from disk if not already loaded."""
        if self._loaded:
            return

        if os.path.exists(self._cache_file_path):
            try:
                with open(self._cache_file_path, 'r') as f:
                    self._cache = json.load(f)
            except (IOError, json.JSONDecodeError):
                self._cache = {}
        else:
            self._cache = {}

        self._loaded = True

    def save(self) -> None:
        """Save cache to disk."""
        try:
            with open(self._cache_file_path, 'w') as f:
                json.dump(self._cache, f, indent=4)
        except IOError:
            print("Warning: Could not write to upload cache file.")

    def get(self, file_hash: str) -> str | None:
        """
        Retrieve a cached URL by file hash.
        
        Args:
            file_hash: SHA-256 hash of the uploaded file.
            
        Returns:
            The cached URL if found, None otherwise.
        """
        self.load()
        return self._cache.get(file_hash)

    def set(self, file_hash: str, url: str) -> None:
        """
        Cache a file URL by its hash and persist to disk.
        
        Args:
            file_hash: SHA-256 hash of the file.
            url: The FAL upload URL to cache.
        """
        self.load()
        self._cache[file_hash] = url
        self.save()

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        return f"FileUploadCache(path={self._cache_file_path!r}, entries={len(self._cache)})"


class FalConfig:
    """Singleton class to handle FAL configuration and client setup."""

    _instance = None
    _client = None
    _key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if os.environ.get("FAL_KEY") is not None:
                print("FAL_KEY found in environment variables")
                self._key = os.environ["FAL_KEY"]
            else:
                print("FAL_KEY not found in environment variables")
                self._key = config["API"]["FAL_KEY"]
                print("FAL_KEY found in config.ini")
                os.environ["FAL_KEY"] = self._key
                print("FAL_KEY set in environment variables")

            # Check if FAL key is the default placeholder
            if self._key == "<your_fal_api_key_here>":
                print("WARNING: You are using the default FAL API key placeholder!")
                print("Please set your actual FAL API key in either:")
                print("1. The config.ini file under [API] section")
                print("2. Or as an environment variable named FAL_KEY")
                print("Get your API key from: https://fal.ai/dashboard/keys")
        except KeyError:
            print("Error: FAL_KEY not found in config.ini or environment variables")

    def get_client(self):
        """Get or create the FAL client."""
        if self._client is None:
            self._client = SyncClient(key=self._key)
        return self._client

    def get_key(self):
        """Get the FAL API key."""
        return self._key


class ImageUtils:
    """Utility functions for image processing."""
    _file_upload_cache = FileUploadCache()

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def upload_image(image):
        """Upload image tensor to FAL and return URL."""
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            # Upload the temporary file using the caching mechanism
            image_url = ImageUtils.upload_file(temp_file_path)
            return image_url
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)
                
    @staticmethod
    def upload_file(file_path):
        """Upload a file to FAL and return URL, with caching."""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Check if file is already cached
            cached_url = ImageUtils._file_upload_cache.get(file_hash)
            if cached_url:
                return cached_url

            # Upload new file
            client = FalConfig().get_client()
            file_url = client.upload_file(file_path)

            # Cache the new URL
            ImageUtils._file_upload_cache.set(file_hash, file_url)
            return file_url
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None
        
    @staticmethod
    def mask_to_image(mask):
        """Convert mask tensor to image tensor."""
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result
    
    @staticmethod
    def prepare_images(images):
        """Preprocess images for use with FAL."""
        image_urls = []
        if images is not None:
            if isinstance(images, torch.Tensor):
                if images.ndim == 4 and images.shape[0] > 1:
                    for i in range(images.shape[0]):
                        single_img = images[i:i+1]
                        img_url = ImageUtils.upload_image(single_img)
                        if img_url:
                            image_urls.append(img_url)
                else:
                    img_url = ImageUtils.upload_image(images)
                    if img_url:
                        image_urls.append(img_url)

            elif isinstance(images, (list, tuple)):
                for img in images:
                    img_url = ImageUtils.upload_image(img)
                    if img_url:
                        image_urls.append(img_url)
        return image_urls

    @staticmethod
    def create_images_zip(images):
        """
        Create a zip file from a list of images and upload it to FAL.

        Args:
            images: List or batch tensor of images.

        Returns:
            The URL of the uploaded zip file, or None on failure.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
                temp_zip_path = temp_zip.name

            with zipfile.ZipFile(temp_zip_path, "w") as zf:
                # Handle batch tensor
                if isinstance(images, torch.Tensor) and images.ndim == 4:
                    image_list = [images[i] for i in range(images.shape[0])]
                elif isinstance(images, (list, tuple)):
                    image_list = images
                else:
                    image_list = [images]

                for idx, img_input in enumerate(image_list):
                    pil_img = ImageUtils.tensor_to_pil(img_input)
                    if not pil_img:
                        continue

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                        pil_img.save(temp_img, format="PNG")
                        temp_img_path = temp_img.name

                    zf.write(temp_img_path, f"image_{idx}.png")
                    os.unlink(temp_img_path)

            return ImageUtils.upload_file(temp_zip_path)
        except Exception as e:
            print(f"Error creating images zip: {str(e)}")
            return None
        finally:
            if 'temp_zip_path' in locals() and os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            images = []
            for img_info in result["images"]:
                img_url = img_info["url"]
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            # Stack the images along a new first dimension
            stacked_images = np.stack(images, axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)

            return (img_tensor,)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        """Process single image result and return tensor."""
        try:
            img_url = result["image"]["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Stack the images along a new first dimension
            stacked_images = np.stack([img_array], axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing single image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        """Create a blank black image tensor."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class FalClient:
    """
    High-level client for interacting with FAL API.

    Encapsulates configuration, execution, and error handling.
    """

    def __init__(self, config: FalConfig | None = None) -> None:
        """Initialize the client with optional configuration."""
        self._config = config or FalConfig()

    def execute(
        self,
        endpoint: str,
        payload: FalPayload | dict[str, Any],
        model_name: str,
        processor: Callable[[Any], Any] | None = None,
        error_handler: Callable[[str, Exception], Any] | None = None,
    ) -> Any:
        """
        Execute a FAL API call with built-in error handling and processing.

        Args:
            endpoint: The FAL API endpoint (e.g., 'fal-ai/flux-pro').
            payload: The arguments for the API call.
            model_name: Friendly name of the model for error reporting.
            processor: Optional function to process the API result.
            error_handler: Optional function to handle exceptions.

        Returns:
            The processed result or the output of the error handler.
        """
        try:
            arguments = payload.build() if isinstance(payload, FalPayload) else payload
            client = self._config.get_client()
            handler = client.submit(endpoint, arguments=arguments)
            result = handler.get()

            if processor:
                return processor(result)
            return result
        except Exception as e:
            if error_handler:
                return error_handler(model_name, e)
            print(f"Error executing {endpoint} ({model_name}): {str(e)}")
            raise e

    def submit_multiple_and_get_results(self, endpoint, arguments, variations):
        """Submit multiple jobs concurrently to FAL API and get results."""
        try:
            # Run the async code in a thread to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    ApiHandler._submit_multiple_async(endpoint, arguments, variations)
                )
                return future.result()
        except Exception as e:
            print(f"Error in submit_multiple_and_get_results: {str(e)}")
            raise e

    @staticmethod
    async def _submit_multiple_async(endpoint, arguments, variations):
        """Submit multiple jobs concurrently and gather results."""
        client = AsyncClient(key=FalConfig().get_key())

        # Submit all jobs concurrently
        handlers = await asyncio.gather(*[
            client.submit(endpoint, arguments={**arguments, "seed": arguments.get("seed", 0) + i} if "seed" in arguments else arguments)
            for i in range(variations)
        ])

        # Get all results concurrently
        results = await asyncio.gather(*[
            handler.get() for handler in handlers
        ], return_exceptions=True)

        return results

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        print(f"Error generating video with {model_name}: {str(error)}")
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)


class ApiHandler:
    """Legacy utility functions for API interactions, now using FalClient internally."""
    _client = FalClient()

    @staticmethod
    def submit_and_get_result(endpoint, arguments):
        """Submit job to FAL API and get result."""
        return ApiHandler._client.execute(endpoint, arguments, endpoint)

    @staticmethod
    def submit_multiple_and_get_results(endpoint, arguments, variations):
        """Submit multiple jobs concurrently to FAL API and get results."""
        return ApiHandler._client.submit_multiple_and_get_results(endpoint, arguments, variations)

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        return FalClient.handle_video_generation_error(model_name, error)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        return FalClient.handle_image_generation_error(model_name, error)

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        return FalClient.handle_text_generation_error(model_name, error)
