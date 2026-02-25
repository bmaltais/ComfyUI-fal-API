"""
Pytest configuration and fixtures for ComfyUI-fal-API tests.

This module sets up mocking for external dependencies before tests run.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy external dependencies before any imports
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.PngImagePlugin'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()

# Mock fal_client with proper structure
fal_client_mock = MagicMock()
fal_client_mock.client = MagicMock()
fal_client_mock.client.SyncClient = MagicMock()
fal_client_mock.AsyncClient = MagicMock()
sys.modules['fal_client'] = fal_client_mock
sys.modules['fal_client.client'] = fal_client_mock.client

# Mock configparser for config handling
import configparser
sys.modules['configparser'] = configparser
