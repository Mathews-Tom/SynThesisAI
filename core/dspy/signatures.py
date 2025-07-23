"""
DSPy signature management and utilities.

This module provides utilities for managing DSPy signatures,
including validation, creation, and domain-specific operations.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from .signature_registry import get_signature_registry

logger = logging.getLogger(__name__)


class SignatureManager:
    """
    Manager for DSPy signatures.

    Provides utilities for signature validation, parsing, and management.
    """

    def __init__(self):
        """Initialize signature manager."""
        self.registry = get_signature_registry()
        self.logger = logging.getLogger(__name__ + ".SignatureManager")

    def validate_signature(self, signature: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a DSPy signature.

        Args:
            signature: Signature string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check basic format: inputs -> outputs
            if " -> " not in signature:
                return False, "Signature must contain ' -> ' separator"

            inputs_str, outputs_str = signature.split(" -> ", 1)

            # Validate inputs
            inputs = [inp.strip() for inp in inputs_str.split(",")]
            if not inputs or any(not inp for inp in inputs):
                return False, "Invalid input specification"

            # Validate outputs
            outputs = [out.strip() for out in outputs_str.split(",")]
            if not outputs or any(not out for out in outputs):
                return False, "Invalid output specification"

            # Check for valid identifiers
            identifier_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
            for inp in inputs:
                if not identifier_pattern.match(inp):
                    return False, f"Invalid input identifier: {inp}"

            for out in outputs:
                if not identifier_pattern.match(out):
                    return False, f"Invalid output identifier: {out}"

            return True, None

        except Exception as e:
            return False, f"Signature validation error: {str(e)}"

    def parse_signature(self, signature: str) -> Tuple[List[str], List[str]]:
        """
        Parse a DSPy signature into inputs and outputs.

        Args:
            signature: Signature string

        Returns:
            Tuple of (inputs, outputs)
        """
        if " -> " not in signature:
            raise ValueError("Invalid signature format")

        inputs_str, outputs_str = signature.split(" -> ", 1)
        inputs = [inp.strip() for inp in inputs_str.split(",")]
        outputs = [out.strip() for out in outputs_str.split(",")]

        return inputs, outputs

    def create_signature(self, inputs: List[str], outputs: List[str]) -> str:
        """
        Create a DSPy signature from inputs and outputs.

        Args:
            inputs: List of input field names
            outputs: List of output field names

        Returns:
            Signature string
        """
        inputs_str = ", ".join(inputs)
        outputs_str = ", ".join(outputs)
        return f"{inputs_str} -> {outputs_str}"

    def get_signature_info(self, signature: str) -> Dict[str, any]:
        """
        Get information about a signature.

        Args:
            signature: Signature string

        Returns:
            Dictionary with signature information
        """
        is_valid, error = self.validate_signature(signature)
        if not is_valid:
            return {"valid": False, "error": error}

        inputs, outputs = self.parse_signature(signature)

        return {
            "valid": True,
            "inputs": inputs,
            "outputs": outputs,
            "input_count": len(inputs),
            "output_count": len(outputs),
            "signature": signature,
        }

    def register_custom_signature(
        self, domain: str, signature_type: str, signature: str, version: str = "1.0.0"
    ) -> bool:
        """
        Register a custom signature.

        Args:
            domain: Domain name
            signature_type: Type of signature
            signature: Signature string
            version: Version string

        Returns:
            True if registered successfully
        """
        # Store the signature with proper parameters
        registry = get_signature_registry()
        signature_name = f"{domain}_{signature_type}"
        return registry.register_signature(
            signature_name, signature, domain, f"Custom signature for {domain}", version
        )

    def get_signature(
        self, domain: str, signature_type: str = "generation"
    ) -> Optional[str]:
        """
        Get signature for domain and type.

        Args:
            domain: Domain name
            signature_type: Type of signature

        Returns:
            Signature string or None if not found
        """
        registry = get_signature_registry()
        signature_name = f"{domain}_{signature_type}"
        return registry.get_signature(signature_name)

    def get_signature_version(
        self, domain: str, signature_type: str = "generation"
    ) -> str:
        """
        Get signature version.

        Args:
            domain: Domain name
            signature_type: Type of signature

        Returns:
            Version string
        """
        registry = get_signature_registry()
        signature_name = f"{domain}_{signature_type}"
        return registry.versions.get(signature_name, "1.0.0")

    def is_signature_compatible(
        self, signature1: str, signature2: str, strict: bool = False
    ) -> bool:
        """
        Check if two signatures are compatible.

        Args:
            signature1: First signature
            signature2: Second signature
            strict: Whether to use strict compatibility checking

        Returns:
            True if compatible
        """
        try:
            inputs1, outputs1 = self.parse_signature(signature1)
            inputs2, outputs2 = self.parse_signature(signature2)

            if strict:
                return inputs1 == inputs2 and outputs1 == outputs2

            # Non-strict: signature2 inputs/outputs must be subset of signature1
            return set(inputs2).issubset(set(inputs1)) and set(outputs2).issubset(
                set(outputs1)
            )
        except Exception:
            return False

    def extend_signature(
        self,
        signature: str,
        additional_inputs: List[str] = None,
        additional_outputs: List[str] = None,
    ) -> str:
        """
        Extend a signature with additional inputs and outputs.

        Args:
            signature: Base signature
            additional_inputs: Additional input fields
            additional_outputs: Additional output fields

        Returns:
            Extended signature
        """
        inputs, outputs = self.parse_signature(signature)

        if additional_inputs:
            for inp in additional_inputs:
                if inp not in inputs:
                    inputs.append(inp)

        if additional_outputs:
            for out in additional_outputs:
                if out not in outputs:
                    outputs.append(out)

        return self.create_signature(inputs, outputs)

    def create_composite_signature(
        self, domain1: str, domain2: str, signature_type: str = "generation"
    ) -> str:
        """
        Create a composite signature from two domains.

        Args:
            domain1: First domain
            domain2: Second domain
            signature_type: Type of signature

        Returns:
            Composite signature
        """
        registry = get_signature_registry()
        sig1 = registry.get_signature(domain1)
        sig2 = registry.get_signature(domain2)

        if not sig1 or not sig2:
            return ""

        inputs1, outputs1 = self.parse_signature(sig1)
        inputs2, outputs2 = self.parse_signature(sig2)

        # Combine inputs and outputs, removing duplicates
        combined_inputs = list(set(inputs1 + inputs2))
        combined_outputs = list(set(outputs1 + outputs2))

        return self.create_signature(combined_inputs, combined_outputs)

    def simplify_signature(
        self, signature: str, required_inputs: List[str], required_outputs: List[str]
    ) -> str:
        """
        Simplify a signature to only include required fields.

        Args:
            signature: Original signature
            required_inputs: Required input fields
            required_outputs: Required output fields

        Returns:
            Simplified signature

        Raises:
            SignatureValidationError: If required fields are not in signature
        """
        from .exceptions import SignatureValidationError

        inputs, outputs = self.parse_signature(signature)

        # Check if all required fields exist
        for inp in required_inputs:
            if inp not in inputs:
                raise SignatureValidationError(
                    f"Required input '{inp}' not in signature", signature=signature
                )

        for out in required_outputs:
            if out not in outputs:
                raise SignatureValidationError(
                    f"Required output '{out}' not in signature", signature=signature
                )

        return self.create_signature(required_inputs, required_outputs)

    def get_domain_signatures_by_type(self, signature_type: str) -> Dict[str, str]:
        """
        Get all domain signatures of a specific type.

        Args:
            signature_type: Type of signature

        Returns:
            Dictionary of domain -> signature mappings
        """
        registry = get_signature_registry()
        return registry.get_all_signatures()

    @property
    def signatures(self) -> Dict[str, str]:
        """Get all signatures."""
        registry = get_signature_registry()
        return registry.get_all_signatures()


def get_domain_signature(
    domain: str, signature_type: str = "generation"
) -> Optional[str]:
    """
    Get signature for a specific domain and type.

    Args:
        domain: Domain name
        signature_type: Type of signature (generation, validation, etc.)

    Returns:
        Signature string or None if not found
    """
    registry = get_signature_registry()
    return registry.get_signature(domain)


def validate_signature(signature: str) -> bool:
    """
    Validate a DSPy signature.

    Args:
        signature: Signature string

    Returns:
        True if valid

    Raises:
        SignatureValidationError: If signature is invalid
    """
    from .exceptions import SignatureValidationError

    manager = SignatureManager()
    is_valid, error_message = manager.validate_signature(signature)
    if not is_valid:
        raise SignatureValidationError(error_message, signature=signature)
    return is_valid


def create_custom_signature(inputs: List[str], outputs: List[str]) -> str:
    """
    Create a custom DSPy signature.

    Args:
        inputs: List of input field names
        outputs: List of output field names

    Returns:
        Signature string

    Raises:
        SignatureValidationError: If inputs or outputs are invalid
    """
    from .exceptions import SignatureValidationError

    if not inputs:
        raise SignatureValidationError("Inputs cannot be empty", signature="")
    if not outputs:
        raise SignatureValidationError("Outputs cannot be empty", signature="")

    # Check for valid identifiers
    import re

    identifier_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    for inp in inputs:
        if not identifier_pattern.match(inp):
            raise SignatureValidationError(
                f"Invalid input identifier: {inp}", signature=""
            )

    for out in outputs:
        if not identifier_pattern.match(out):
            raise SignatureValidationError(
                f"Invalid output identifier: {out}", signature=""
            )

    manager = SignatureManager()
    return manager.create_signature(inputs, outputs)


def get_all_domains() -> List[str]:
    """
    Get all available domains.

    Returns:
        List of domain names
    """
    registry = get_signature_registry()
    return registry.list_domains()


def get_signature_types(domain: str) -> Dict[str, str]:
    """
    Get signature types for a domain.

    Args:
        domain: Domain name

    Returns:
        Dictionary of signature type -> signature mappings

    Raises:
        SignatureValidationError: If domain is invalid
    """
    from .exceptions import SignatureValidationError

    registry = get_signature_registry()
    if domain not in registry.list_domains():
        raise SignatureValidationError(f"Invalid domain: {domain}", signature=domain)

    # For now, return basic signature types
    # In a full implementation, this would be more sophisticated
    return {
        "generation": registry.get_signature(domain),
        "validation": registry.get_signature(domain),
        "equivalence": registry.get_signature(domain),
        "refinement": registry.get_signature(domain),
        "assessment": registry.get_signature(domain),
    }
