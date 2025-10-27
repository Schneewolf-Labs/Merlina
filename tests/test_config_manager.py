#!/usr/bin/env python3
"""
Tests for the ConfigManager module
"""

import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager


def test_save_and_load():
    """Test saving and loading a configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=tmpdir)

        # Test config
        config = {
            "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "output_name": "test-model",
            "use_lora": True,
            "lora_r": 16,
            "learning_rate": 5e-5
        }

        # Save config
        filepath = manager.save_config(
            name="test-config",
            config=config,
            description="Test configuration",
            tags=["test", "llama3"]
        )

        print(f"✓ Saved config to: {filepath}")

        # Load config
        loaded = manager.load_config("test-config")

        assert "_metadata" in loaded, "Metadata should be present"
        assert loaded["base_model"] == config["base_model"]
        assert loaded["lora_r"] == config["lora_r"]

        print("✓ Loaded config successfully")

        # Test metadata
        metadata = loaded["_metadata"]
        assert metadata["name"] == "test-config"
        assert metadata["description"] == "Test configuration"
        assert "test" in metadata["tags"]
        assert "llama3" in metadata["tags"]

        print("✓ Metadata verified")


def test_list_configs():
    """Test listing configurations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=tmpdir)

        # Save multiple configs
        for i in range(3):
            manager.save_config(
                name=f"config-{i}",
                config={"test": i},
                tags=["test"] if i % 2 == 0 else []
            )

        # List all configs
        all_configs = manager.list_configs()
        assert len(all_configs) == 3, "Should have 3 configs"

        print(f"✓ Listed {len(all_configs)} configs")

        # Filter by tag
        tagged_configs = manager.list_configs(tag="test")
        assert len(tagged_configs) == 2, "Should have 2 configs with 'test' tag"

        print(f"✓ Filtered configs by tag: {len(tagged_configs)}")


def test_delete_config():
    """Test deleting a configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=tmpdir)

        # Save and delete
        manager.save_config("to-delete", {"test": "value"})
        assert manager.delete_config("to-delete") == True
        assert manager.delete_config("nonexistent") == False

        print("✓ Delete config works")


def test_get_without_metadata():
    """Test getting config without metadata"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=tmpdir)

        config = {"key": "value"}
        manager.save_config("test", config)

        # Get without metadata
        clean_config = manager.get_config_without_metadata("test")
        assert "_metadata" not in clean_config
        assert clean_config["key"] == "value"

        print("✓ Get config without metadata works")


def test_export_import():
    """Test exporting and importing configs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=tmpdir)

        config = {"test": "export"}
        manager.save_config("original", config)

        # Export
        export_path = Path(tmpdir) / "exported.json"
        manager.export_config("original", str(export_path))

        assert export_path.exists()
        print(f"✓ Exported to: {export_path}")

        # Import
        manager.import_config(str(export_path), "imported")

        imported = manager.get_config_without_metadata("imported")
        assert imported["test"] == "export"

        print("✓ Import successful")


def test_sanitize_filename():
    """Test filename sanitization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(config_dir=tmpdir)

        # Test with invalid characters
        unsafe_name = "my/config:with*invalid?chars"
        manager.save_config(unsafe_name, {"test": "value"})

        # Should be saved with sanitized name
        configs = manager.list_configs()
        assert len(configs) == 1

        print("✓ Filename sanitization works")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running ConfigManager Tests")
    print("=" * 60)

    tests = [
        test_save_and_load,
        test_list_configs,
        test_delete_config,
        test_get_without_metadata,
        test_export_import,
        test_sanitize_filename
    ]

    for test in tests:
        print(f"\n▶ Running: {test.__name__}")
        try:
            test()
            print(f"✅ {test.__name__} PASSED\n")
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}\n")
            return False
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}\n")
            return False

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
