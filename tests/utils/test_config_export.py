"""
Unit tests for Configuration Export/Import functionality.

Tests cover:
- Export configuration to YAML
- Import configuration from YAML
- Round-trip export/import
- Secret exclusion/inclusion
- Configuration diff
- Error handling
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from nocp.core.config import ProxyConfig, reset_config
from nocp.utils.config_export import export_config, import_config, print_config_diff


class TestConfigExport:
    """Tests for configuration export functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset global config before each test
        reset_config()

    def teardown_method(self):
        """Clean up after tests."""
        reset_config()

    def test_export_config_default_path(self):
        """Test exporting config to default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("nocp.utils.config_export.Path") as mock_path:
                # Create a temporary config
                config = ProxyConfig(gemini_api_key="test-key-123")

                # Mock the default path to use temp directory
                output_path = Path(tmpdir) / "config.yaml"
                mock_path.return_value = output_path

                result_path = export_config(config, output_path=output_path)

                # Verify file was created
                assert result_path.exists()
                assert result_path == output_path

                # Verify file contents
                with result_path.open() as f:
                    exported_data = yaml.safe_load(f)

                # Secrets should be excluded by default
                assert "gemini_api_key" not in exported_data
                assert "gemini_model" in exported_data

    def test_export_config_include_secrets(self):
        """Test exporting config with secrets included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProxyConfig(gemini_api_key="test-secret-key")
            output_path = Path(tmpdir) / "config_with_secrets.yaml"

            result_path = export_config(
                config,
                output_path=output_path,
                include_secrets=True
            )

            # Verify file contents
            with result_path.open() as f:
                exported_data = yaml.safe_load(f)

            # Secrets should be included
            assert "gemini_api_key" in exported_data
            assert exported_data["gemini_api_key"] == "test-secret-key"

    def test_export_config_exclude_secrets(self):
        """Test that secrets are excluded by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProxyConfig(
                gemini_api_key="gemini-secret",
                openai_api_key="openai-secret",
                anthropic_api_key="anthropic-secret"
            )
            output_path = Path(tmpdir) / "config_no_secrets.yaml"

            export_config(config, output_path=output_path, include_secrets=False)

            # Verify file contents
            with output_path.open() as f:
                exported_data = yaml.safe_load(f)

            # All secrets should be excluded
            assert "gemini_api_key" not in exported_data
            assert "openai_api_key" not in exported_data
            assert "anthropic_api_key" not in exported_data

    def test_export_creates_parent_directory(self):
        """Test that export creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProxyConfig(gemini_api_key="test-key")
            output_path = Path(tmpdir) / "nested" / "path" / "config.yaml"

            # Parent directory doesn't exist yet
            assert not output_path.parent.exists()

            export_config(config, output_path=output_path)

            # Parent directory should be created
            assert output_path.parent.exists()
            assert output_path.exists()

    def test_export_yaml_format(self):
        """Test that exported YAML is properly formatted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProxyConfig(
                gemini_api_key="test-key",
                gemini_model="gemini-2.5-flash",
                log_level="DEBUG",
                default_compression_threshold=5000
            )
            output_path = Path(tmpdir) / "config.yaml"

            export_config(config, output_path=output_path)

            # Read and parse YAML
            with output_path.open() as f:
                content = f.read()
                exported_data = yaml.safe_load(content)

            # Verify YAML is properly formatted (not flow style)
            assert "default_compression_threshold: 5000" in content
            assert "log_level: DEBUG" in content

            # Verify data integrity
            assert exported_data["gemini_model"] == "gemini-2.5-flash"
            assert exported_data["default_compression_threshold"] == 5000


class TestConfigImport:
    """Tests for configuration import functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_config()

    def teardown_method(self):
        """Clean up after tests."""
        reset_config()

    def test_import_config_basic(self):
        """Test importing a basic config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create a YAML config file
            config_data = {
                "gemini_api_key": "imported-key",
                "gemini_model": "gemini-2.0-flash",
                "log_level": "INFO",
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Import the config
            imported_config = import_config(config_path)

            # Verify imported data
            assert isinstance(imported_config, ProxyConfig)
            assert imported_config.gemini_api_key == "imported-key"
            assert imported_config.gemini_model == "gemini-2.0-flash"
            assert imported_config.log_level == "INFO"

    def test_import_config_with_defaults(self):
        """Test that import preserves defaults for missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "minimal_config.yaml"

            # Create minimal config with only required field
            config_data = {
                "gemini_api_key": "minimal-key",
            }

            with config_path.open("w") as f:
                yaml.dump(config_data, f)

            # Import the config
            imported_config = import_config(config_path)

            # Verify required field
            assert imported_config.gemini_api_key == "minimal-key"

            # Verify defaults are applied
            assert imported_config.gemini_model == "gemini-2.5-flash"
            assert imported_config.log_level == "INFO"
            assert imported_config.default_compression_threshold == 5000

    def test_import_nonexistent_file(self):
        """Test that importing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            import_config(Path("/nonexistent/path/config.yaml"))

    def test_import_invalid_yaml(self):
        """Test that importing invalid YAML raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"

            # Create invalid YAML
            with config_path.open("w") as f:
                f.write("invalid: yaml: content: [unclosed")

            with pytest.raises(yaml.YAMLError):
                import_config(config_path)


class TestConfigRoundTrip:
    """Tests for export/import round-trip."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_config()

    def teardown_method(self):
        """Clean up after tests."""
        reset_config()

    def test_round_trip_basic(self):
        """Test that config survives export/import round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original config
            original_config = ProxyConfig(
                gemini_api_key="round-trip-key",
                gemini_model="gemini-2.5-flash",
                log_level="DEBUG",
                default_compression_threshold=8000,
                enable_semantic_pruning=True,
            )

            output_path = Path(tmpdir) / "roundtrip.yaml"

            # Export
            export_config(
                original_config,
                output_path=output_path,
                include_secrets=True  # Include secrets for full round-trip
            )

            # Import
            imported_config = import_config(output_path)

            # Verify all fields match
            assert imported_config.gemini_api_key == original_config.gemini_api_key
            assert imported_config.gemini_model == original_config.gemini_model
            assert imported_config.log_level == original_config.log_level
            assert imported_config.default_compression_threshold == \
                original_config.default_compression_threshold
            assert imported_config.enable_semantic_pruning == \
                original_config.enable_semantic_pruning

    def test_round_trip_without_secrets(self):
        """Test round-trip without secrets (common use case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original config
            original_config = ProxyConfig(
                gemini_api_key="original-secret",
                gemini_model="gemini-2.0-flash",
                log_level="INFO",
            )

            output_path = Path(tmpdir) / "no_secrets.yaml"

            # Export without secrets
            export_config(original_config, output_path=output_path, include_secrets=False)

            # Import (will need to provide required gemini_api_key)
            # Read the YAML and add a key
            with output_path.open() as f:
                config_data = yaml.safe_load(f)

            config_data["gemini_api_key"] = "new-secret"

            with output_path.open("w") as f:
                yaml.dump(config_data, f)

            imported_config = import_config(output_path)

            # Verify non-secret fields match
            assert imported_config.gemini_model == original_config.gemini_model
            assert imported_config.log_level == original_config.log_level

            # Secret was replaced
            assert imported_config.gemini_api_key == "new-secret"


class TestConfigDiff:
    """Tests for configuration diff functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_config()

    def teardown_method(self):
        """Clean up after tests."""
        reset_config()

    def test_print_config_diff(self, capsys):
        """Test printing config diff."""
        config1 = ProxyConfig(
            gemini_api_key="key1",
            gemini_model="gemini-2.0-flash",
            log_level="DEBUG",
        )

        config2 = ProxyConfig(
            gemini_api_key="key2",
            gemini_model="gemini-2.5-flash",
            log_level="INFO",
        )

        # This should print to console
        print_config_diff(config1, config2)

        # Capture output
        captured = capsys.readouterr()

        # Verify output contains differences
        assert "Configuration Diff" in captured.out

    def test_config_diff_identical(self, capsys):
        """Test diff of identical configs."""
        config1 = ProxyConfig(
            gemini_api_key="same-key",
            gemini_model="gemini-2.0-flash",
        )

        config2 = ProxyConfig(
            gemini_api_key="same-key",
            gemini_model="gemini-2.0-flash",
        )

        print_config_diff(config1, config2)

        captured = capsys.readouterr()

        # Should still print table but with no differences
        assert "Configuration Diff" in captured.out
