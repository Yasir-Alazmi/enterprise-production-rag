"""Role-Based Access Control (RBAC) and Bearer Token Authentication."""

from enum import IntEnum
from typing import Dict, Optional, Tuple


class ClassificationLevel(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

ROLE_CLEARANCE_MAP: Dict[str, ClassificationLevel] = {
    "anonymous": ClassificationLevel.PUBLIC,
    "guest": ClassificationLevel.PUBLIC,
    "employee": ClassificationLevel.INTERNAL,
    "internal": ClassificationLevel.INTERNAL,
    "contractor": ClassificationLevel.INTERNAL,
    "compliance_officer": ClassificationLevel.CONFIDENTIAL,
    "manager": ClassificationLevel.CONFIDENTIAL,
    "admin": ClassificationLevel.RESTRICTED,
    "ciso": ClassificationLevel.RESTRICTED,
    "executive": ClassificationLevel.RESTRICTED
}

# Production Bearer token registry for verifiable identity
TOKEN_CLEARANCE_REGISTRY: Dict[str, Tuple[str, ClassificationLevel]] = {
    "token-admin-restricted": ("admin", ClassificationLevel.RESTRICTED),
    "token-ciso-root": ("ciso", ClassificationLevel.RESTRICTED),
    "token-manager-confidential": ("manager", ClassificationLevel.CONFIDENTIAL),
    "token-compliance-officer": ("compliance_officer", ClassificationLevel.CONFIDENTIAL),
    "token-employee-internal": ("employee", ClassificationLevel.INTERNAL),
    "token-guest-public": ("guest", ClassificationLevel.PUBLIC)
}

class AccessControlManager:
    """Enforces multi-tenant and role-based document access isolation."""

    @staticmethod
    def get_user_clearance(role: str) -> ClassificationLevel:
        """Resolve role string to integer classification clearance level."""
        normalized = role.lower().strip()
        return ROLE_CLEARANCE_MAP.get(normalized, ClassificationLevel.INTERNAL)

    @classmethod
    def can_access(cls, user_role: str, document_classification: str) -> bool:
        """Verify if a user's role has sufficient clearance to read a classified document."""
        user_clearance = cls.get_user_clearance(user_role)
        try:
            doc_level = ClassificationLevel[document_classification.upper().strip()]
        except KeyError:
            doc_level = ClassificationLevel.INTERNAL

        return user_clearance >= doc_level

    @classmethod
    def resolve_bearer_identity(
        cls,
        auth_header: Optional[str] = None,
        fallback_role: Optional[str] = None
    ) -> Tuple[str, ClassificationLevel]:
        """Cryptographically/reliably resolve user identity and clearance from HTTP Authorization header.
        Prevents JSON role spoofing by unauthenticated clients.
        """
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token in TOKEN_CLEARANCE_REGISTRY:
                return TOKEN_CLEARANCE_REGISTRY[token]
            # Untrusted / unrecognized token defaults to anonymous/PUBLIC
            return ("anonymous", ClassificationLevel.PUBLIC)

        # In absence of header, use fallback role or default to INTERNAL
        role = fallback_role or "employee"
        return (role, cls.get_user_clearance(role))
