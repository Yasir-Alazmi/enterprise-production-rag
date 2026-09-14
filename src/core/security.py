"""Role-Based Access Control (RBAC) and data classification hierarchy."""

from enum import IntEnum
from typing import Dict


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
    "compliance_officer": ClassificationLevel.CONFIDENTIAL,
    "manager": ClassificationLevel.CONFIDENTIAL,
    "admin": ClassificationLevel.RESTRICTED,
    "ciso": ClassificationLevel.RESTRICTED,
    "executive": ClassificationLevel.RESTRICTED
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
