"""
Unit tests for FalPayload and FalClient classes.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure nodes is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nodes')))

from fal_utils import FalPayload, FalClient, ResultProcessor

class TestFalPayload:
    def test_initialization(self):
        payload = FalPayload(prompt="test")
        assert payload.build() == {"prompt": "test"}

    def test_set_image_size_standard(self):
        payload = FalPayload().set_image_size("square_hd")
        assert payload.build() == {"image_size": "square_hd"}

    def test_set_image_size_custom(self):
        payload = FalPayload().set_image_size("custom", 1024, 768)
        assert payload.build() == {"image_size": {"width": 1024, "height": 768}}

    def test_set_video_size_custom(self):
        payload = FalPayload().set_video_size("custom", 1280, 720)
        assert payload.build() == {"video_size": {"width": 1280, "height": 720}}

    def test_set_seed(self):
        payload = FalPayload().set_seed(1234)
        assert payload.build() == {"seed": 1234}

        payload = FalPayload().set_seed(-1)
        assert "seed" not in payload.build()

    def test_add_loras(self):
        payload = FalPayload().add_loras([("path1", 0.5), ("path2", 0.8), (None, 1.0), ("None", 1.0)])
        assert payload.build()["loras"] == [
            {"path": "path1", "scale": 0.5},
            {"path": "path2", "scale": 0.8}
        ]

    def test_update(self):
        payload = FalPayload(a=1).update(b=2, c=3)
        assert payload.build() == {"a": 1, "b": 2, "c": 3}

class TestFalClient:
    @patch('fal_utils.FalConfig')
    def test_execute_success(self, mock_config_class):
        mock_config = mock_config_class.return_value
        mock_sync_client = MagicMock()
        mock_config.get_client.return_value = mock_sync_client

        mock_handler = MagicMock()
        mock_sync_client.submit.return_value = mock_handler
        mock_handler.get.return_value = {"status": "ok", "images": [{"url": "http://example.com/img.png"}]}

        client = FalClient(config=mock_config)
        payload = FalPayload(prompt="test")

        # Test without processor
        result = client.execute("endpoint", payload, "model")
        assert result["status"] == "ok"

        # Test with processor
        mock_processor = MagicMock(return_value="processed")
        result = client.execute("endpoint", payload, "model", processor=mock_processor)
        assert result == "processed"
        mock_processor.assert_called_once()

    @patch('fal_utils.FalConfig')
    def test_execute_failure(self, mock_config_class):
        mock_config = mock_config_class.return_value
        mock_sync_client = MagicMock()
        mock_config.get_client.return_value = mock_sync_client
        mock_sync_client.submit.side_effect = Exception("API Error")

        client = FalClient(config=mock_config)

        # Test without error handler
        with pytest.raises(Exception) as excinfo:
            client.execute("endpoint", {}, "model")
        assert "API Error" in str(excinfo.value)

        # Test with error handler
        mock_error_handler = MagicMock(return_value="error_handled")
        result = client.execute("endpoint", {}, "model", error_handler=mock_error_handler)
        assert result == "error_handled"

    @patch('fal_utils.FalConfig')
    def test_execute_failure_with_actual_exception(self, mock_config_class):
        mock_config = mock_config_class.return_value
        mock_sync_client = MagicMock()
        mock_config.get_client.return_value = mock_sync_client
        exception_instance = Exception("API Error")
        mock_sync_client.submit.side_effect = exception_instance

        client = FalClient(config=mock_config)
        mock_error_handler = MagicMock(return_value="error_handled")

        result = client.execute("endpoint", {}, "model", error_handler=mock_error_handler)
        assert result == "error_handled"
        mock_error_handler.assert_called_once_with("model", exception_instance)
