"""
DSPy Signature Registry

This module provides a centralized registry for managing DSPy signatures
across the application, including version tracking, compatibility checking,
and signature transformation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .exceptions import SignatureValidationError
from .signatures import (
    SignatureManager,
    create_custom_signature,
    get_all_domains,
    get_domain_signature,
    validate_signature,
)

logger = logging.getLogger(__name__)


class SignatureRegistry:
    """
    Centralized registry for DSPy signatures.

    Provides functionality for managing signatures across the application,
    including persistence, version tracking, and compatibility checking.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize signature registry.

        Args:
            registry_path: Optional path to registry file
        """
        self.manager = SignatureManager()
        self.registry_path = Path(
            registry_path or ".cache/dspy/signature_registry.json"
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Registry data structure
        self.registry = {
            "signatures": {},
            "compatibility_matrix": {},
            "usage_stats": {},
            "version": "1.0.0",
        }

        # Load registry if exists
        self._load_registry()

        logger.info("Initialized signature registry at %s", self.registry_path)

    def _load_registry(self) -> bool:
        """
        Load registry from file.

        Returns:
            True if loaded successfully
        """
        if not self.registry_path.exists():
            logger.info(
                "Registry file not found at %s, creating new registry",
                self.registry_path,
            )
            self._save_registry()
            return False

        try:
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
            logger.info(
                "Loaded signature registry with %d signatures",
                len(self.registry["signatures"]),
            )
            return True
        except Exception as e:
            logger.error("Failed to load registry: %s", str(e))
            return False

    def _save_registry(self) -> bool:
        """
        Save registry to file.

        Returns:
            True if saved successfully
        """
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)
            logger.debug("Saved signature registry to %s", self.registry_path)
            return True
        except Exception as e:
            logger.error("Failed to save registry: %s", str(e))
            return False

    def register_signature(
        self,
        name: str,
        signature: str,
        domain: str = None,
        description: str = None,
        version: str = "1.0.0",
    ) -> bool:
        """
        Register a signature in the registry.

        Args:
            name: Unique name for the signature
            signature: DSPy signature string
            domain: Optional domain for the signature
            description: Optional description
            version: Signature version

        Returns:
            True if registered successfully
        """
        try:
            # Validate signature
            validate_signature(signature)

            # Add to registry
            self.registry["signatures"][name] = {
                "signature": signature,
                "domain": domain,
                "description": description or f"Signature for {name}",
                "version": version,
                "created_at": self._get_timestamp(),
                "inputs": self.manager.parse_signature(signature)[0],
                "outputs": self.manager.parse_signature(signature)[1],
            }

            # Initialize usage stats
            if name not in self.registry["usage_stats"]:
                self.registry["usage_stats"][name] = {
                    "usage_count": 0,
                    "last_used": None,
                }

            # Save registry
            self._save_registry()

            logger.info("Registered signature '%s' (v%s)", name, version)
            return True

        except Exception as e:
            logger.error("Failed to register signature '%s': %s", name, str(e))
            return False

    def get_signature(self, name: str) -> Optional[str]:
        """
        Get a signature by name.

        Args:
            name: Signature name

        Returns:
            Signature string or None if not found
        """
        if name in self.registry["signatures"]:
            # Update usage stats
            self.registry["usage_stats"][name]["usage_count"] += 1
            self.registry["usage_stats"][name]["last_used"] = self._get_timestamp()
            self._save_registry()

            return self.registry["signatures"][name]["signature"]

        return None

    def get_signature_info(self, name: str) -> Optional[Dict]:
        """
        Get signature information.

        Args:
            name: Signature name

        Returns:
            Signature information or None if not found
        """
        if name in self.registry["signatures"]:
            return self.registry["signatures"][name]

        return None

    def list_signatures(self, domain: str = None) -> List[Dict]:
        """
        List registered signatures.

        Args:
            domain: Optional domain filter

        Returns:
            List of signature information
        """
        signatures = []

        for name, info in self.registry["signatures"].items():
            if domain is None or info["domain"] == domain:
                signatures.append(
                    {
                        "name": name,
                        **info,
                        "usage": self.registry["usage_stats"].get(
                            name, {"usage_count": 0}
                        ),
                    }
                )

        return signatures

    def delete_signature(self, name: str) -> bool:
        """
        Delete a signature from the registry.

        Args:
            name: Signature name

        Returns:
            True if deleted successfully
        """
        if name in self.registry["signatures"]:
            del self.registry["signatures"][name]

            # Remove from usage stats
            if name in self.registry["usage_stats"]:
                del self.registry["usage_stats"][name]

            # Remove from compatibility matrix
            for sig_name in list(self.registry["compatibility_matrix"].keys()):
                if name in self.registry["compatibility_matrix"][sig_name]:
                    del self.registry["compatibility_matrix"][sig_name][name]

                if (
                    sig_name == name
                    and sig_name in self.registry["compatibility_matrix"]
                ):
                    del self.registry["compatibility_matrix"][sig_name]

            # Save registry
            self._save_registry()

            logger.info("Deleted signature '%s'", name)
            return True

        logger.warning("Signature '%s' not found for deletion", name)
        return False

    def update_signature(
        self,
        name: str,
        signature: str = None,
        domain: str = None,
        description: str = None,
        version: str = None,
    ) -> bool:
        """
        Update a signature in the registry.

        Args:
            name: Signature name
            signature: Optional new signature string
            domain: Optional new domain
            description: Optional new description
            version: Optional new version

        Returns:
            True if updated successfully
        """
        if name not in self.registry["signatures"]:
            logger.warning("Signature '%s' not found for update", name)
            return False

        try:
            # Validate new signature if provided
            if signature is not None:
                validate_signature(signature)
                self.registry["signatures"][name]["signature"] = signature
                self.registry["signatures"][name]["inputs"] = (
                    self.manager.parse_signature(signature)[0]
                )
                self.registry["signatures"][name]["outputs"] = (
                    self.manager.parse_signature(signature)[1]
                )

            # Update other fields if provided
            if domain is not None:
                self.registry["signatures"][name]["domain"] = domain

            if description is not None:
                self.registry["signatures"][name]["description"] = description

            if version is not None:
                self.registry["signatures"][name]["version"] = version

            # Update timestamp
            self.registry["signatures"][name]["updated_at"] = self._get_timestamp()

            # Save registry
            self._save_registry()

            logger.info("Updated signature '%s'", name)
            return True

        except Exception as e:
            logger.error("Failed to update signature '%s': %s", name, str(e))
            return False

    def check_compatibility(
        self, signature1: str, signature2: str, strict: bool = False
    ) -> bool:
        """
        Check if two signatures are compatible.

        Args:
            signature1: First signature name or string
            signature2: Second signature name or string
            strict: Whether to require exact match

        Returns:
            True if signatures are compatible
        """
        # Get signature strings
        sig1 = signature1 if " -> " in signature1 else self.get_signature(signature1)
        sig2 = signature2 if " -> " in signature2 else self.get_signature(signature2)

        if sig1 is None or sig2 is None:
            return False

        # Check compatibility
        return self.manager.is_signature_compatible(sig1, sig2, strict)

    def build_compatibility_matrix(self, signatures: List[str] = None) -> Dict:
        """
        Build compatibility matrix for signatures.

        Args:
            signatures: Optional list of signature names to include

        Returns:
            Compatibility matrix
        """
        # Get signatures to include
        if signatures is None:
            signatures = list(self.registry["signatures"].keys())

        # Build matrix
        matrix = {}

        for sig1 in signatures:
            matrix[sig1] = {}

            for sig2 in signatures:
                if sig1 == sig2:
                    matrix[sig1][sig2] = True
                    continue

                # Check compatibility
                sig1_str = self.get_signature(sig1)
                sig2_str = self.get_signature(sig2)

                if sig1_str is None or sig2_str is None:
                    matrix[sig1][sig2] = False
                    continue

                matrix[sig1][sig2] = self.manager.is_signature_compatible(
                    sig1_str, sig2_str
                )

        # Update registry
        self.registry["compatibility_matrix"] = matrix
        self._save_registry()

        return matrix

    def find_compatible_signatures(
        self, signature: str, strict: bool = False
    ) -> List[str]:
        """
        Find signatures compatible with the given signature.

        Args:
            signature: Signature name or string
            strict: Whether to require exact match

        Returns:
            List of compatible signature names
        """
        # Get signature string
        sig = signature if " -> " in signature else self.get_signature(signature)

        if sig is None:
            return []

        # Find compatible signatures
        compatible = []

        for name, info in self.registry["signatures"].items():
            if self.manager.is_signature_compatible(sig, info["signature"], strict):
                compatible.append(name)

        return compatible

    def import_domain_signatures(self, domain: str = None) -> int:
        """
        Import built-in domain signatures to registry.

        Args:
            domain: Optional domain to import

        Returns:
            Number of signatures imported
        """
        count = 0

        # Get domains to import
        domains = [domain] if domain else get_all_domains()

        for domain in domains:
            # Get signature types for domain
            try:
                signature_types = self.manager.signatures.get(domain, {}).keys()

                for sig_type in signature_types:
                    # Get signature
                    signature = get_domain_signature(domain, sig_type)

                    # Register signature
                    name = f"{domain}_{sig_type}"
                    description = f"{domain.capitalize()} {sig_type} signature"

                    if self.register_signature(name, signature, domain, description):
                        count += 1

            except Exception as e:
                logger.error(
                    "Failed to import signatures for domain '%s': %s", domain, str(e)
                )

        return count

    def export_signatures(self, export_path: str = None) -> bool:
        """
        Export registry to file.

        Args:
            export_path: Optional export path

        Returns:
            True if exported successfully
        """
        export_path = export_path or f"{self.registry_path.stem}_export.json"

        try:
            with open(export_path, "w") as f:
                json.dump(self.registry, f, indent=2)

            logger.info("Exported signature registry to %s", export_path)
            return True

        except Exception as e:
            logger.error("Failed to export registry: %s", str(e))
            return False

    def import_signatures(self, import_path: str) -> int:
        """
        Import signatures from file.

        Args:
            import_path: Path to import file

        Returns:
            Number of signatures imported
        """
        try:
            with open(import_path, "r") as f:
                import_data = json.load(f)

            count = 0

            # Import signatures
            for name, info in import_data.get("signatures", {}).items():
                if self.register_signature(
                    name,
                    info["signature"],
                    info.get("domain"),
                    info.get("description"),
                    info.get("version", "1.0.0"),
                ):
                    count += 1

            logger.info("Imported %d signatures from %s", count, import_path)
            return count

        except Exception as e:
            logger.error("Failed to import signatures: %s", str(e))
            return 0

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.now().isoformat()


# Global registry instance
_signature_registry = None


def get_signature_registry() -> SignatureRegistry:
    """Get the global signature registry instance."""
    global _signature_registry
    if _signature_registry is None:
        _signature_registry = SignatureRegistry()
    return _signature_registry
