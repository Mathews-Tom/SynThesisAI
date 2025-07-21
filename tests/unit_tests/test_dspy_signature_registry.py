"""
Unit tests for DSPy signature registry.

These tests verify the functionality of the signature registry,
including registration, retrieval, compatibility checking, and persistence.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from core.dspy.signature_registry import SignatureRegistry, get_signature_registry
from core.dspy.signatures import SignatureValidationError


class TestSignatureRegistry:
    """Test SignatureRegistry functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary registry file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self.temp_dir.name) / "test_registry.json"

        # Create registry instance
        self.registry = SignatureRegistry(str(self.registry_path))

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test registry initialization."""
        assert self.registry is not None
        assert self.registry.registry_path == self.registry_path
        assert "signatures" in self.registry.registry
        assert "compatibility_matrix" in self.registry.registry
        assert "usage_stats" in self.registry.registry
        assert "version" in self.registry.registry

    def test_register_signature(self):
        """Test registering signatures."""
        # Register a valid signature
        result = self.registry.register_signature(
            "test_signature",
            "input1, input2 -> output1, output2",
            "test_domain",
            "Test signature description",
            "1.0.0",
        )
        assert result is True

        # Verify signature is registered
        assert "test_signature" in self.registry.registry["signatures"]
        assert (
            self.registry.registry["signatures"]["test_signature"]["signature"]
            == "input1, input2 -> output1, output2"
        )
        assert (
            self.registry.registry["signatures"]["test_signature"]["domain"]
            == "test_domain"
        )
        assert (
            self.registry.registry["signatures"]["test_signature"]["description"]
            == "Test signature description"
        )
        assert (
            self.registry.registry["signatures"]["test_signature"]["version"] == "1.0.0"
        )

        # Register an invalid signature
        result = self.registry.register_signature(
            "invalid_signature", "invalid signature", "test_domain"
        )
        assert result is False
        assert "invalid_signature" not in self.registry.registry["signatures"]

    def test_get_signature(self):
        """Test retrieving signatures."""
        # Register a signature
        self.registry.register_signature(
            "test_signature", "input1, input2 -> output1, output2"
        )

        # Get signature
        signature = self.registry.get_signature("test_signature")
        assert signature == "input1, input2 -> output1, output2"

        # Get non-existent signature
        signature = self.registry.get_signature("non_existent")
        assert signature is None

        # Verify usage stats are updated
        assert (
            self.registry.registry["usage_stats"]["test_signature"]["usage_count"] == 1
        )
        assert (
            self.registry.registry["usage_stats"]["test_signature"]["last_used"]
            is not None
        )

    def test_get_signature_info(self):
        """Test retrieving signature information."""
        # Register a signature
        self.registry.register_signature(
            "test_signature",
            "input1, input2 -> output1, output2",
            "test_domain",
            "Test signature description",
        )

        # Get signature info
        info = self.registry.get_signature_info("test_signature")
        assert info is not None
        assert info["signature"] == "input1, input2 -> output1, output2"
        assert info["domain"] == "test_domain"
        assert info["description"] == "Test signature description"
        assert info["inputs"] == ["input1", "input2"]
        assert info["outputs"] == ["output1", "output2"]

        # Get non-existent signature info
        info = self.registry.get_signature_info("non_existent")
        assert info is None

    def test_list_signatures(self):
        """Test listing signatures."""
        # Register signatures
        self.registry.register_signature(
            "test_signature1", "input1, input2 -> output1, output2", "domain1"
        )
        self.registry.register_signature(
            "test_signature2", "input3, input4 -> output3, output4", "domain2"
        )
        self.registry.register_signature(
            "test_signature3", "input5, input6 -> output5, output6", "domain1"
        )

        # List all signatures
        signatures = self.registry.list_signatures()
        assert len(signatures) == 3

        # List signatures by domain
        signatures = self.registry.list_signatures("domain1")
        assert len(signatures) == 2
        assert signatures[0]["domain"] == "domain1"
        assert signatures[1]["domain"] == "domain1"

        signatures = self.registry.list_signatures("domain2")
        assert len(signatures) == 1
        assert signatures[0]["domain"] == "domain2"

        # List signatures for non-existent domain
        signatures = self.registry.list_signatures("non_existent")
        assert len(signatures) == 0

    def test_delete_signature(self):
        """Test deleting signatures."""
        # Register a signature
        self.registry.register_signature(
            "test_signature", "input1, input2 -> output1, output2"
        )

        # Verify signature exists
        assert "test_signature" in self.registry.registry["signatures"]

        # Delete signature
        result = self.registry.delete_signature("test_signature")
        assert result is True

        # Verify signature is deleted
        assert "test_signature" not in self.registry.registry["signatures"]
        assert "test_signature" not in self.registry.registry["usage_stats"]

        # Delete non-existent signature
        result = self.registry.delete_signature("non_existent")
        assert result is False

    def test_update_signature(self):
        """Test updating signatures."""
        # Register a signature
        self.registry.register_signature(
            "test_signature",
            "input1, input2 -> output1, output2",
            "domain1",
            "Original description",
            "1.0.0",
        )

        # Update signature
        result = self.registry.update_signature(
            "test_signature",
            "input1, input2, input3 -> output1, output2, output3",
            "domain2",
            "Updated description",
            "2.0.0",
        )
        assert result is True

        # Verify signature is updated
        info = self.registry.get_signature_info("test_signature")
        assert (
            info["signature"] == "input1, input2, input3 -> output1, output2, output3"
        )
        assert info["domain"] == "domain2"
        assert info["description"] == "Updated description"
        assert info["version"] == "2.0.0"
        assert info["inputs"] == ["input1", "input2", "input3"]
        assert info["outputs"] == ["output1", "output2", "output3"]
        assert "updated_at" in info

        # Update non-existent signature
        result = self.registry.update_signature("non_existent", "a -> b")
        assert result is False

        # Update with invalid signature
        result = self.registry.update_signature("test_signature", "invalid signature")
        assert result is False

        # Update only specific fields
        result = self.registry.update_signature(
            "test_signature", description="Partially updated description"
        )
        assert result is True

        # Verify only specified fields are updated
        info = self.registry.get_signature_info("test_signature")
        assert (
            info["signature"] == "input1, input2, input3 -> output1, output2, output3"
        )
        assert info["domain"] == "domain2"
        assert info["description"] == "Partially updated description"
        assert info["version"] == "2.0.0"

    def test_check_compatibility(self):
        """Test checking signature compatibility."""
        # Register signatures
        self.registry.register_signature("signature1", "a, b, c -> x, y, z")
        self.registry.register_signature("signature2", "a, b -> x, y")
        self.registry.register_signature("signature3", "a, b, d -> x, y, w")

        # Check compatibility by name
        assert self.registry.check_compatibility("signature1", "signature2") is True
        assert self.registry.check_compatibility("signature2", "signature1") is False

        # Check compatibility by string
        assert (
            self.registry.check_compatibility("a, b, c -> x, y, z", "a, b -> x, y")
            is True
        )
        assert (
            self.registry.check_compatibility("a, b -> x, y", "a, b, c -> x, y, z")
            is False
        )

        # Check compatibility with strict mode
        assert (
            self.registry.check_compatibility("signature1", "signature1", strict=True)
            is True
        )
        assert (
            self.registry.check_compatibility("signature1", "signature2", strict=True)
            is False
        )

        # Check compatibility with non-existent signature
        assert self.registry.check_compatibility("signature1", "non_existent") is False
        assert self.registry.check_compatibility("non_existent", "signature1") is False

    def test_build_compatibility_matrix(self):
        """Test building compatibility matrix."""
        # Register signatures
        self.registry.register_signature("signature1", "a, b, c -> x, y, z")
        self.registry.register_signature("signature2", "a, b -> x, y")
        self.registry.register_signature("signature3", "a, b, d -> x, y, w")

        # Build compatibility matrix
        matrix = self.registry.build_compatibility_matrix()

        # Verify matrix
        assert "signature1" in matrix
        assert "signature2" in matrix
        assert "signature3" in matrix

        assert matrix["signature1"]["signature1"] is True
        assert matrix["signature1"]["signature2"] is True
        assert matrix["signature2"]["signature1"] is False
        assert matrix["signature1"]["signature3"] is False
        assert matrix["signature3"]["signature1"] is False

        # Build matrix for specific signatures
        matrix = self.registry.build_compatibility_matrix(["signature1", "signature2"])

        # Verify matrix
        assert "signature1" in matrix
        assert "signature2" in matrix
        assert "signature3" not in matrix

    def test_find_compatible_signatures(self):
        """Test finding compatible signatures."""
        # Register signatures
        self.registry.register_signature("signature1", "a, b, c -> x, y, z")
        self.registry.register_signature("signature2", "a, b -> x, y")
        self.registry.register_signature("signature3", "a, b, d -> x, y, w")
        self.registry.register_signature(
            "signature4",
            "a, b, c -> x, y, z",  # Same as signature1
        )

        # Find compatible signatures by name
        compatible = self.registry.find_compatible_signatures("signature1")
        assert "signature1" in compatible
        assert "signature2" in compatible
        assert "signature4" in compatible
        assert "signature3" not in compatible

        # Find compatible signatures by string
        compatible = self.registry.find_compatible_signatures("a, b, c -> x, y, z")
        assert "signature1" in compatible
        assert "signature2" in compatible
        assert "signature4" in compatible
        assert "signature3" not in compatible

        # Find compatible signatures with strict mode
        compatible = self.registry.find_compatible_signatures("signature1", strict=True)
        assert "signature1" in compatible
        assert "signature4" in compatible
        assert "signature2" not in compatible
        assert "signature3" not in compatible

    def test_import_domain_signatures(self):
        """Test importing domain signatures."""
        # Import mathematics domain signatures
        count = self.registry.import_domain_signatures("mathematics")
        assert count > 0

        # Verify signatures are imported
        signatures = self.registry.list_signatures("mathematics")
        assert len(signatures) > 0

        # Import all domain signatures
        count = self.registry.import_domain_signatures()
        assert count > 0

        # Verify signatures from multiple domains are imported
        domains = set(sig["domain"] for sig in self.registry.list_signatures())
        assert len(domains) > 1
        assert "mathematics" in domains
        assert "science" in domains

    def test_export_import_signatures(self):
        """Test exporting and importing signatures."""
        # Register signatures
        self.registry.register_signature(
            "test_signature1", "input1, input2 -> output1, output2", "domain1"
        )
        self.registry.register_signature(
            "test_signature2", "input3, input4 -> output3, output4", "domain2"
        )

        # Export signatures
        export_path = Path(self.temp_dir.name) / "export.json"
        result = self.registry.export_signatures(str(export_path))
        assert result is True
        assert export_path.exists()

        # Create new registry
        new_registry = SignatureRegistry(str(self.registry_path) + ".new")

        # Import signatures
        count = new_registry.import_signatures(str(export_path))
        assert count == 2

        # Verify signatures are imported
        assert "test_signature1" in new_registry.registry["signatures"]
        assert "test_signature2" in new_registry.registry["signatures"]

    def test_persistence(self):
        """Test registry persistence."""
        # Register signatures
        self.registry.register_signature(
            "test_signature1", "input1, input2 -> output1, output2", "domain1"
        )
        self.registry.register_signature(
            "test_signature2", "input3, input4 -> output3, output4", "domain2"
        )

        # Create new registry instance with same path
        new_registry = SignatureRegistry(str(self.registry_path))

        # Verify signatures are loaded
        assert "test_signature1" in new_registry.registry["signatures"]
        assert "test_signature2" in new_registry.registry["signatures"]

        # Verify signature details are preserved
        assert (
            new_registry.registry["signatures"]["test_signature1"]["signature"]
            == "input1, input2 -> output1, output2"
        )
        assert (
            new_registry.registry["signatures"]["test_signature1"]["domain"]
            == "domain1"
        )
        assert (
            new_registry.registry["signatures"]["test_signature2"]["signature"]
            == "input3, input4 -> output3, output4"
        )
        assert (
            new_registry.registry["signatures"]["test_signature2"]["domain"]
            == "domain2"
        )

    def test_global_registry(self):
        """Test global registry instance."""
        # Get global registry
        registry1 = get_signature_registry()
        registry2 = get_signature_registry()

        # Verify same instance is returned
        assert registry1 is registry2


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
