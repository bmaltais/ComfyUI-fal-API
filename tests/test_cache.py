"""
Unit tests for FileUploadCache class using pytest.

Tests the cache logic including:
- Cache initialization
- File hashing and caching
- Disk persistence
- Cache retrieval
- Error handling
"""

import os
import json
import tempfile
import hashlib
from unittest.mock import patch, MagicMock

import pytest

# Import after conftest has set up mocks
import sys
_nodes_path = os.path.join(os.path.dirname(__file__), '..', 'nodes')
if _nodes_path not in sys.path:
    sys.path.insert(0, _nodes_path)

from fal_utils import FileUploadCache


class TestFileUploadCacheInitialization:
    """Test FileUploadCache initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default cache file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch the path resolution to use our temp dir
            with patch('os.path.dirname') as mock_dirname:
                mock_dirname.side_effect = [tmpdir, tmpdir]
                cache = FileUploadCache()
                
                assert cache._cache is not None
                assert isinstance(cache._cache, dict)
                assert cache._loaded is False

    def test_init_with_custom_path(self):
        """Test initialization with custom cache file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_cache.json")
            cache = FileUploadCache(cache_file_path=custom_path)
            
            assert cache._cache_file_path == custom_path
            assert cache._cache == {}
            assert cache._loaded is False

    def test_init_cache_dict_is_empty(self):
        """Test that new cache starts with empty dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            assert len(cache._cache) == 0


class TestFileUploadCacheLoadSave:
    """Test cache loading and saving to disk."""

    def test_save_creates_file(self):
        """Test that save() creates the cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            cache._cache["hash1"] = "url1"
            cache.save()
            
            assert os.path.exists(cache_path)

    def test_save_writes_valid_json(self):
        """Test that save() writes valid JSON to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            test_data = {
                "hash1": "url1",
                "hash2": "url2",
            }
            cache._cache = test_data
            cache.save()
            
            # Load and verify
            with open(cache_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded == test_data

    def test_load_from_existing_file(self):
        """Test that load() reads cache from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            # Create cache file with test data
            test_data = {"hash1": "url1", "hash2": "url2"}
            with open(cache_path, 'w') as f:
                json.dump(test_data, f)
            
            # Load it
            cache = FileUploadCache(cache_file_path=cache_path)
            cache.load()
            
            assert cache._cache == test_data
            assert cache._loaded is True

    def test_load_creates_empty_cache_if_file_missing(self):
        """Test that load() creates empty cache if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "nonexistent.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            cache.load()
            
            assert cache._cache == {}
            assert cache._loaded is True

    def test_load_handles_corrupted_json(self):
        """Test that load() handles corrupted JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "bad.json")
            
            # Write invalid JSON
            with open(cache_path, 'w') as f:
                f.write("{invalid json}")
            
            cache = FileUploadCache(cache_file_path=cache_path)
            cache.load()
            
            # Should create empty cache instead of crashing
            assert cache._cache == {}
            assert cache._loaded is True

    def test_load_idempotent(self):
        """Test that load() only loads once (idempotent)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            # Initial data
            initial_data = {"hash1": "url1"}
            with open(cache_path, 'w') as f:
                json.dump(initial_data, f)
            
            cache = FileUploadCache(cache_file_path=cache_path)
            cache.load()
            assert cache._cache == initial_data
            
            # Modify file on disk
            modified_data = {"hash1": "url1", "hash2": "url2"}
            with open(cache_path, 'w') as f:
                json.dump(modified_data, f)
            
            # Load again - should still be initial data (idempotent)
            cache.load()
            assert cache._cache == initial_data


class TestFileUploadCacheGetSet:
    """Test cache get/set operations."""

    def test_set_and_get(self):
        """Test setting and retrieving a cached value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            hash_value = hashlib.sha256(b"test").hexdigest()
            url = "https://example.com/file.png"
            
            cache.set(hash_value, url)
            retrieved = cache.get(hash_value)
            
            assert retrieved == url

    def test_get_nonexistent_returns_none(self):
        """Test that getting a nonexistent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            missing_hash = hashlib.sha256(b"missing").hexdigest()
            result = cache.get(missing_hash)
            
            assert result is None

    def test_set_persists_to_disk(self):
        """Test that set() persists data to disk immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            hash_value = "abc123"
            url = "https://example.com/file.png"
            
            cache.set(hash_value, url)
            
            # Verify file was written
            assert os.path.exists(cache_path)
            
            # Load from disk and verify
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            assert data[hash_value] == url

    def test_set_loads_cache_first(self):
        """Test that set() loads existing cache before writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            # Pre-populate cache file
            existing = {"existing_hash": "existing_url"}
            with open(cache_path, 'w') as f:
                json.dump(existing, f)
            
            # Create new cache instance and set a value
            cache = FileUploadCache(cache_file_path=cache_path)
            cache.set("new_hash", "new_url")
            
            # Verify both old and new values exist
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            assert "existing_hash" in data
            assert "new_hash" in data
            assert data["existing_hash"] == "existing_url"
            assert data["new_hash"] == "new_url"

    def test_get_loads_cache_first(self):
        """Test that get() loads cache from disk if not already loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            # Pre-populate cache file
            test_data = {"test_hash": "test_url"}
            with open(cache_path, 'w') as f:
                json.dump(test_data, f)
            
            # Create cache instance without loading
            cache = FileUploadCache(cache_file_path=cache_path)
            assert cache._loaded is False
            
            # Get should trigger load
            result = cache.get("test_hash")
            
            assert result == "test_url"
            assert cache._loaded is True

    def test_multiple_entries(self):
        """Test managing multiple cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            # Add multiple entries
            entries = {
                "hash1": "url1",
                "hash2": "url2",
                "hash3": "url3",
            }
            
            for hash_val, url in entries.items():
                cache.set(hash_val, url)
            
            # Verify all entries
            for hash_val, url in entries.items():
                assert cache.get(hash_val) == url


class TestFileUploadCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_save_file_write_error(self):
        """Test handling of file write errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid path (directory instead of file)
            cache_path = os.path.join(tmpdir, "subdir", "cache.json")
            # Don't create the subdir, forcing a write error
            
            cache = FileUploadCache(cache_file_path=cache_path)
            cache._cache["test"] = "url"
            
            # This should not raise but print a warning
            cache.save()  # Should handle IOError gracefully

    def test_empty_cache(self):
        """Test operations on an empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            # Operations on empty cache
            assert cache.get("any_hash") is None
            assert len(cache._cache) == 0

    def test_cache_with_special_characters_in_url(self):
        """Test caching URLs with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            special_url = "https://example.com/file?param=value&other=test#anchor"
            cache.set("hash1", special_url)
            
            assert cache.get("hash1") == special_url

    def test_cache_with_unicode_url(self):
        """Test caching URLs with unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            unicode_url = "https://example.com/file_文件_αβγ"
            cache.set("hash_unicode", unicode_url)
            
            assert cache.get("hash_unicode") == unicode_url


class TestFileUploadCacheRepr:
    """Test string representation of cache."""

    def test_repr_empty_cache(self):
        """Test __repr__ for empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            repr_str = repr(cache)
            
            assert "FileUploadCache" in repr_str
            assert "entries=0" in repr_str

    def test_repr_with_entries(self):
        """Test __repr__ for cache with entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            cache.set("hash1", "url1")
            cache.set("hash2", "url2")
            
            repr_str = repr(cache)
            
            assert "FileUploadCache" in repr_str
            assert "entries=2" in repr_str
            assert cache_path in repr_str


class TestFileUploadCacheIntegration:
    """Integration tests for realistic usage scenarios."""

    def test_cache_lifecycle(self):
        """Test complete cache lifecycle: create, set, save, load, get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            # Phase 1: Create and populate cache
            cache1 = FileUploadCache(cache_file_path=cache_path)
            cache1.set("file1_hash", "https://fal.ai/uploads/file1")
            cache1.set("file2_hash", "https://fal.ai/uploads/file2")
            
            # Phase 2: Create new instance and load from disk
            cache2 = FileUploadCache(cache_file_path=cache_path)
            
            # Phase 3: Retrieve values
            assert cache2.get("file1_hash") == "https://fal.ai/uploads/file1"
            assert cache2.get("file2_hash") == "https://fal.ai/uploads/file2"

    def test_cache_with_sha256_hashes(self):
        """Test cache with real SHA256 file hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            # Simulate multiple file uploads
            file_contents = [
                b"image1.png content",
                b"image2.png content",
                b"another image",
            ]
            
            for i, content in enumerate(file_contents):
                file_hash = hashlib.sha256(content).hexdigest()
                url = f"https://fal.ai/uploads/{file_hash}"
                cache.set(file_hash, url)
            
            # Verify all files are cached
            for content in file_contents:
                file_hash = hashlib.sha256(content).hexdigest()
                cached_url = cache.get(file_hash)
                assert cached_url == f"https://fal.ai/uploads/{file_hash}"

    def test_concurrent_operations(self):
        """Test that cache handles multiple set operations correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            cache = FileUploadCache(cache_file_path=cache_path)
            
            # Simulate rapid successive operations
            for i in range(10):
                cache.set(f"hash_{i}", f"url_{i}")
            
            # Verify all entries persisted
            for i in range(10):
                assert cache.get(f"hash_{i}") == f"url_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

