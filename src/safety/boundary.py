"""Workspace sandbox boundaries, path sanitization, and untrusted observation shielding."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union


class SecurityError(Exception):
    """Raised when an action violates workspace boundary or safety policy."""
    pass


class WorkspaceSandbox:
    """Enforces strict path containment, safe operations, and untrusted observation tagging."""

    def __init__(
        self,
        workspace_root: Union[str, Path],
        allow_destructive: bool = False,
        approval_callback: Optional[Callable[[str, str], bool]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.allow_destructive = allow_destructive
        self.approval_callback = approval_callback
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def validate_path(self, target_path: Union[str, Path]) -> Path:
        """Resolve path and verify it stays strictly inside the workspace root."""
        # Convert to Path
        path = Path(target_path)
        if not path.is_absolute():
            resolved = (self.workspace_root / path).resolve()
        else:
            resolved = path.resolve()

        # Check path containment
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise SecurityError(
                f"Security Violation: Target path '{target_path}' resolves to '{resolved}' "
                f"which is outside approved workspace '{self.workspace_root}'."
            )

        return resolved

    def check_approval(self, operation: str, details: str) -> bool:
        """Check approval for destructive or side-effecting operations."""
        if self.allow_destructive:
            return True
        if self.approval_callback is not None:
            return self.approval_callback(operation, details)
        return False

    @staticmethod
    def wrap_untrusted_observation(observation: str, source: str = "tool") -> str:
        """Enclose external tool observation in protective untrusted tags.
        
        Prevents prompt injection by marking data as passive observation.
        """
        sanitized = observation.replace("<SYSTEM_DIRECTIVE>", "[STRIPPED]")
        return f"<OBSERVATION source='{source}'>\n{sanitized}\n</OBSERVATION>"
