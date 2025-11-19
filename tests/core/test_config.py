"""
Unit tests for Configuration Module.

Tests cover:
- pyproject.toml configuration loading
- Configuration precedence (CLI > env > .env > pyproject.toml > defaults)
- Error handling for missing/invalid pyproject.toml
- Backward compatibility
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from nocp.core.config import ProxyConfig, load_pyproject_defaults, reset_config


class TestPyProjectLoading:
    """Tests for pyproject.toml configuration loading."""

    def test_load_pyproject_defaults_success(self, tmp_path):
        """Test successful loading of [tool.nocp] configuration."""
        # Create a temporary pyproject.toml
        pyproject_content = """
[tool.nocp]
compression_threshold = 8000
enable_semantic_pruning = false
default_output_format = "compact_json"
log_level = "DEBUG"
litellm_default_model = "openai/gpt-4"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        # Change to temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            defaults = load_pyproject_defaults()

            assert defaults["compression_threshold"] == 8000
            assert defaults["enable_semantic_pruning"] is False
            assert defaults["default_output_format"] == "compact_json"
            assert defaults["log_level"] == "DEBUG"
            assert defaults["litellm_default_model"] == "openai/gpt-4"
        finally:
            os.chdir(original_dir)

    def test_load_pyproject_defaults_missing_file(self, tmp_path):
        """Test handling of missing pyproject.toml."""
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            defaults = load_pyproject_defaults()
            assert defaults == {}
        finally:
            os.chdir(original_dir)

    def test_load_pyproject_defaults_missing_tool_section(self, tmp_path):
        """Test handling when [tool.nocp] section is missing."""
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"

[tool.other]
setting = "value"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            defaults = load_pyproject_defaults()
            assert defaults == {}
        finally:
            os.chdir(original_dir)

    def test_load_pyproject_defaults_invalid_toml(self, tmp_path):
        """Test handling of invalid TOML syntax."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text("invalid [[ toml syntax")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Should return empty dict and log warning
            defaults = load_pyproject_defaults()
            assert defaults == {}
        finally:
            os.chdir(original_dir)

    @patch('nocp.core.config.tomllib', None)
    def test_load_pyproject_defaults_no_tomllib(self):
        """Test warning when tomllib/tomli is not available."""
        with pytest.warns(ImportWarning, match="tomli package not installed"):
            defaults = load_pyproject_defaults()
            assert defaults == {}


class TestConfigurationPrecedence:
    """Tests for configuration precedence order."""

    def test_explicit_kwargs_override_pyproject(self, tmp_path, monkeypatch):
        """Test that explicit kwargs override pyproject.toml settings."""
        # Create pyproject.toml with defaults
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 5000
log_level = "INFO"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        # Set minimal required env var
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            # Explicit kwargs should override pyproject.toml
            config = ProxyConfig(
                default_compression_threshold=10000,
                log_level="ERROR"
            )

            assert config.default_compression_threshold == 10000
            assert config.log_level == "ERROR"
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_env_vars_override_pyproject(self, tmp_path, monkeypatch):
        """Test that environment variables override pyproject.toml settings."""
        # Create pyproject.toml with defaults
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 5000
log_level = "INFO"
litellm_default_model = "gemini/gemini-2.0-flash-exp"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        # Set environment variables
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("NOCP_DEFAULT_COMPRESSION_THRESHOLD", "15000")
        monkeypatch.setenv("NOCP_LOG_LEVEL", "DEBUG")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            # Env vars should override pyproject.toml
            assert config.default_compression_threshold == 15000
            assert config.log_level == "DEBUG"
            # But pyproject.toml values are used when env var not set
            assert config.litellm_default_model == "gemini/gemini-2.0-flash-exp"
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_pyproject_overrides_defaults(self, tmp_path, monkeypatch):
        """Test that pyproject.toml settings override hardcoded defaults."""
        # Create pyproject.toml with custom defaults
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 8000
enable_semantic_pruning = false
default_output_format = "compact_json"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            # pyproject.toml should override defaults
            assert config.default_compression_threshold == 8000
            assert config.enable_semantic_pruning is False
            assert config.default_output_format == "compact_json"
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_hardcoded_defaults_used_when_no_overrides(self, monkeypatch):
        """Test that hardcoded defaults are used when no overrides present."""
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        reset_config()

        config = ProxyConfig()

        # Should use hardcoded defaults
        assert config.default_compression_threshold == 5000
        assert config.enable_semantic_pruning is True
        assert config.default_output_format == "toon"
        reset_config()


class TestConfigurationTypes:
    """Tests for type conversion and validation."""

    def test_integer_fields_converted(self, tmp_path, monkeypatch):
        """Test that integer fields are properly converted from pyproject.toml."""
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 7500
max_input_tokens = 500000
litellm_max_retries = 5
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            assert isinstance(config.default_compression_threshold, int)
            assert config.default_compression_threshold == 7500
            assert isinstance(config.max_input_tokens, int)
            assert config.max_input_tokens == 500000
            assert isinstance(config.litellm_max_retries, int)
            assert config.litellm_max_retries == 5
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_boolean_fields_converted(self, tmp_path, monkeypatch):
        """Test that boolean fields are properly converted from pyproject.toml."""
        pyproject_content = """
[tool.nocp]
enable_semantic_pruning = false
enable_knowledge_distillation = true
enable_metrics_logging = false
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            assert isinstance(config.enable_semantic_pruning, bool)
            assert config.enable_semantic_pruning is False
            assert isinstance(config.enable_knowledge_distillation, bool)
            assert config.enable_knowledge_distillation is True
            assert isinstance(config.enable_metrics_logging, bool)
            assert config.enable_metrics_logging is False
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_string_fields_preserved(self, tmp_path, monkeypatch):
        """Test that string fields are properly preserved from pyproject.toml."""
        pyproject_content = """
[tool.nocp]
log_level = "WARNING"
default_output_format = "json"
litellm_default_model = "anthropic/claude-3-sonnet"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            assert config.log_level == "WARNING"
            assert config.default_output_format == "json"
            assert config.litellm_default_model == "anthropic/claude-3-sonnet"
        finally:
            os.chdir(original_dir)
            reset_config()


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing configurations."""

    def test_config_works_without_pyproject(self, monkeypatch):
        """Test that configuration still works without pyproject.toml."""
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        reset_config()

        # Should work fine without pyproject.toml
        config = ProxyConfig()
        assert config.gemini_api_key == "test-key"
        assert config.default_compression_threshold == 5000
        reset_config()

    def test_existing_methods_still_work(self, monkeypatch):
        """Test that existing config methods still function correctly."""
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        reset_config()

        config = ProxyConfig()

        # Test tool threshold registration
        config.register_tool_threshold("test_tool", 8000)
        assert config.get_compression_threshold("test_tool") == 8000
        assert config.get_compression_threshold("unknown_tool") == 5000

        # Test ensure_log_directory
        config.ensure_log_directory()
        assert config.metrics_log_file.parent.exists()

        reset_config()
