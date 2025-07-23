"""
Registry for DSPy signatures.

This module provides a registry system for managing DSPy signatures
across different domains and use cases.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SignatureRegistry:
    """
    Registry for DSPy signatures.

    Manages domain-specific signatures and provides lookup functionality.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """Initialize signature registry.

        Args:
            registry_path: Optional path to registry file (for persistence)
        """
        self.signatures: Dict[str, str] = {}
        self.versions: Dict[str, str] = {}
        from pathlib import Path

        self.registry_path = Path(registry_path) if registry_path else None
        self.logger = logging.getLogger(__name__ + ".SignatureRegistry")

        # Initialize registry structure for tests
        self.registry = {
            "signatures": {},
            "compatibility_matrix": {},
            "usage_stats": {},
            "version": "1.0.0",
        }

        self._initialize_default_signatures()

    def _initialize_default_signatures(self) -> None:
        """Initialize default signatures for STREAM domains."""
        default_signatures = {
            "mathematics": "mathematical_concept, difficulty_level, learning_objectives -> problem_statement, solution, proof, reasoning_trace, pedagogical_hints, misconception_analysis",
            "science": "scientific_concept, difficulty_level, learning_objectives -> problem_statement, solution, experimental_design, evidence_evaluation, reasoning_trace, scientific_principles",
            "technology": "technical_concept, difficulty_level, learning_objectives -> problem_statement, solution, algorithm_explanation, system_design, reasoning_trace, implementation_considerations",
            "reading": "literary_concept, difficulty_level, learning_objectives -> comprehension_question, analysis_prompt, critical_thinking_exercise, reasoning_trace",
            "engineering": "engineering_concept, difficulty_level, learning_objectives -> design_challenge, optimization_problem, constraint_analysis, reasoning_trace",
            "arts": "artistic_concept, difficulty_level, learning_objectives -> creative_prompt, aesthetic_analysis, cultural_context, reasoning_trace",
            "interdisciplinary": "primary_domain, secondary_domain, difficulty_level, learning_objectives -> cross_domain_connections, interdisciplinary_principles, reasoning_trace",
        }

        for domain, signature in default_signatures.items():
            self.register_signature(domain, signature, domain)

    def register_signature(
        self,
        name: str,
        signature: str,
        domain: str = "",
        description: str = "",
        version: str = "1.0.0",
    ) -> bool:
        """
        Register a signature for a domain.

        Args:
            domain: Domain name
            signature: DSPy signature string
            version: Version string
            description: Optional description

        Returns:
            True if registered successfully
        """
        try:
            # Validate signature format
            if " -> " not in signature:
                return False

            self.signatures[name] = signature
            self.versions[name] = version

            # Update registry structure
            self.registry["signatures"][name] = {
                "signature": signature,
                "domain": domain,
                "description": description,
                "version": version,
                "created_at": "2024-01-01T00:00:00",
            }

            # Initialize usage stats
            self.registry["usage_stats"][name] = {"usage_count": 0, "last_used": None}

            self.logger.info("Registered signature: %s", name)
            return True
        except Exception as e:
            self.logger.error("Failed to register signature %s: %s", name, str(e))
            return False

    def get_signature(self, domain: str) -> Optional[str]:
        """
        Get signature for a domain.

        Args:
            domain: Domain name

        Returns:
            Signature string or None if not found
        """
        # Try exact match first
        signature = self.signatures.get(domain)
        if signature:
            # Update usage stats
            if domain in self.registry["usage_stats"]:
                self.registry["usage_stats"][domain]["usage_count"] += 1
                self.registry["usage_stats"][domain]["last_used"] = (
                    "2024-01-01T00:00:00"
                )

            self.logger.debug("Retrieved signature for domain: %s", domain)
            return signature

        # Try case-insensitive match
        domain_lower = domain.lower()
        for key, value in self.signatures.items():
            if key.lower() == domain_lower:
                # Update usage stats
                if key in self.registry["usage_stats"]:
                    self.registry["usage_stats"][key]["usage_count"] += 1
                    self.registry["usage_stats"][key]["last_used"] = (
                        "2024-01-01T00:00:00"
                    )

                self.logger.debug(
                    "Retrieved signature for domain (case-insensitive): %s", domain
                )
                return value

        self.logger.warning("No signature found for domain: %s", domain)
        return None

    def list_domains(self) -> List[str]:
        """
        List all registered domains.

        Returns:
            List of domain names
        """
        return list(self.signatures.keys())

    def remove_signature(self, domain: str) -> bool:
        """
        Remove signature for a domain.

        Args:
            domain: Domain name

        Returns:
            True if removed successfully
        """
        if domain in self.signatures:
            del self.signatures[domain]
            self.logger.info("Removed signature for domain: %s", domain)
            return True
        else:
            self.logger.warning("No signature to remove for domain: %s", domain)
            return False

    def get_all_signatures(self) -> Dict[str, str]:
        """
        Get all registered signatures.

        Returns:
            Dictionary of domain -> signature mappings
        """
        return self.signatures.copy()

    def get_signature_info(self, signature_name: str) -> Optional[Dict]:
        """
        Get detailed information about a signature.

        Args:
            signature_name: Name of the signature

        Returns:
            Dictionary with signature information or None if not found
        """
        if signature_name not in self.signatures:
            return None

        signature = self.signatures[signature_name]

        # Parse signature to get inputs and outputs
        if " -> " in signature:
            inputs_str, outputs_str = signature.split(" -> ", 1)
            inputs = [inp.strip() for inp in inputs_str.split(",")]
            outputs = [out.strip() for out in outputs_str.split(",")]
        else:
            inputs = []
            outputs = []

        # Get domain and description from registry if available
        registry_info = self.registry["signatures"].get(signature_name, {})
        domain = registry_info.get("domain", signature_name)
        description = registry_info.get(
            "description", f"Signature for {signature_name}"
        )

        return {
            "signature": signature,
            "domain": domain,
            "description": description,
            "inputs": inputs,
            "outputs": outputs,
            "version": self.versions.get(signature_name, "1.0.0"),
        }

    def list_signatures(self, domain: Optional[str] = None) -> List[Dict]:
        """
        List signatures, optionally filtered by domain.

        Args:
            domain: Optional domain filter

        Returns:
            List of signature information dictionaries
        """
        signatures = []
        for name, signature in self.signatures.items():
            info = self.get_signature_info(name)
            if info and (domain is None or info["domain"] == domain):
                signatures.append(info)
        return signatures

    def delete_signature(self, signature_name: str) -> bool:
        """
        Delete a signature.

        Args:
            signature_name: Name of the signature to delete

        Returns:
            True if deleted successfully
        """
        if signature_name in self.signatures:
            del self.signatures[signature_name]
            if signature_name in self.versions:
                del self.versions[signature_name]
            if signature_name in self.registry["usage_stats"]:
                del self.registry["usage_stats"][signature_name]
            self.logger.info("Deleted signature: %s", signature_name)
            return True
        return False

    def update_signature(
        self,
        signature_name: str,
        signature: Optional[str] = None,
        domain: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
    ) -> bool:
        """
        Update an existing signature.

        Args:
            signature_name: Name of the signature to update
            signature: New signature string
            domain: New domain
            description: New description
            version: New version

        Returns:
            True if updated successfully
        """
        if signature_name not in self.signatures:
            return False

        if signature is not None:
            # Validate signature format
            if " -> " not in signature:
                return False
            self.signatures[signature_name] = signature

        if version is not None:
            self.versions[signature_name] = version

        # Update registry metadata
        if signature_name not in self.registry["signatures"]:
            self.registry["signatures"][signature_name] = {}

        if domain is not None:
            self.registry["signatures"][signature_name]["domain"] = domain

        if description is not None:
            self.registry["signatures"][signature_name]["description"] = description

        self.registry["signatures"][signature_name]["updated_at"] = (
            "2024-01-01T00:00:00"
        )

        self.logger.info("Updated signature: %s", signature_name)
        return True

    def check_compatibility(self, sig1: str, sig2: str, strict: bool = False) -> bool:
        """
        Check compatibility between two signatures.

        Args:
            sig1: First signature (name or string)
            sig2: Second signature (name or string)
            strict: Whether to use strict compatibility

        Returns:
            True if compatible
        """
        # Get actual signature strings
        signature1 = self.signatures.get(sig1, sig1)
        signature2 = self.signatures.get(sig2, sig2)

        try:
            # Parse signatures
            if " -> " not in signature1 or " -> " not in signature2:
                return False

            inputs1, outputs1 = signature1.split(" -> ", 1)
            inputs2, outputs2 = signature2.split(" -> ", 1)

            inputs1 = [inp.strip() for inp in inputs1.split(",")]
            outputs1 = [out.strip() for out in outputs1.split(",")]
            inputs2 = [inp.strip() for inp in inputs2.split(",")]
            outputs2 = [out.strip() for out in outputs2.split(",")]

            if strict:
                return inputs1 == inputs2 and outputs1 == outputs2

            # Non-strict: sig2 inputs/outputs must be subset of sig1
            return set(inputs2).issubset(set(inputs1)) and set(outputs2).issubset(
                set(outputs1)
            )
        except Exception:
            return False

    def build_compatibility_matrix(
        self, signatures: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, bool]]:
        """
        Build compatibility matrix for signatures.

        Args:
            signatures: Optional list of signature names to include

        Returns:
            Compatibility matrix
        """
        if signatures is None:
            signatures = list(self.signatures.keys())

        matrix = {}
        for sig1 in signatures:
            matrix[sig1] = {}
            for sig2 in signatures:
                matrix[sig1][sig2] = self.check_compatibility(sig1, sig2)

        return matrix

    def find_compatible_signatures(
        self, signature: str, strict: bool = False
    ) -> List[str]:
        """
        Find signatures compatible with the given signature.

        Args:
            signature: Signature name or string
            strict: Whether to use strict compatibility

        Returns:
            List of compatible signature names
        """
        compatible = []
        for name in self.signatures.keys():
            if self.check_compatibility(signature, name, strict):
                compatible.append(name)
        return compatible

    def import_domain_signatures(self, domain: Optional[str] = None) -> int:
        """
        Import domain signatures.

        Args:
            domain: Optional specific domain to import

        Returns:
            Number of signatures imported
        """
        # For now, just return the count of existing signatures
        # In a full implementation, this would load from external sources
        if domain:
            return 1 if domain in self.signatures else 0
        return len(self.signatures)

    def export_signatures(self, export_path: str) -> bool:
        """
        Export signatures to file.

        Args:
            export_path: Path to export file

        Returns:
            True if exported successfully
        """
        try:
            import json
            from pathlib import Path

            export_data = {
                "signatures": self.signatures,
                "versions": self.versions,
                "exported_at": "2024-01-01T00:00:00",
            }

            Path(export_path).write_text(json.dumps(export_data, indent=2))
            return True
        except Exception as e:
            self.logger.error("Failed to export signatures: %s", str(e))
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
            import json
            from pathlib import Path

            data = json.loads(Path(import_path).read_text())
            count = 0

            for name, signature in data.get("signatures", {}).items():
                self.signatures[name] = signature
                if name in data.get("versions", {}):
                    self.versions[name] = data["versions"][name]
                count += 1

            return count
        except Exception as e:
            self.logger.error("Failed to import signatures: %s", str(e))
            return 0


# Global registry instance
_signature_registry = None


def get_signature_registry() -> SignatureRegistry:
    """Get the global signature registry instance."""
    global _signature_registry
    if _signature_registry is None:
        _signature_registry = SignatureRegistry()
    return _signature_registry
