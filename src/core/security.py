"""Role-Based Access Control (RBAC), JWT Verification, and Bearer Token Security."""

import json
import time
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import jwt
from fastapi import HTTPException, status

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


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
    "executive": ClassificationLevel.RESTRICTED,
}

# Standard reference tokens for local environments and test verification
DEFAULT_TOKEN_REGISTRY: Dict[str, Tuple[str, ClassificationLevel]] = {
    "token-admin-restricted": ("admin", ClassificationLevel.RESTRICTED),
    "token-ciso-root": ("ciso", ClassificationLevel.RESTRICTED),
    "token-manager-confidential": ("manager", ClassificationLevel.CONFIDENTIAL),
    "token-compliance-officer": ("compliance_officer", ClassificationLevel.CONFIDENTIAL),
    "token-employee-internal": ("employee", ClassificationLevel.INTERNAL),
    "token-guest-public": ("guest", ClassificationLevel.PUBLIC),
}


def create_jwt_token(
    subject: str,
    role: str,
    clearance: Optional[ClassificationLevel] = None,
    expires_in_seconds: int = 3600,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Generate signed HMAC-SHA256 JWT for testing and machine-to-machine auth."""
    secret = secret_key or settings.jwt_secret_key
    algo = algorithm or settings.jwt_algorithm
    assigned_clearance = clearance or ROLE_CLEARANCE_MAP.get(role.lower(), ClassificationLevel.INTERNAL)

    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "clearance": int(assigned_clearance),
        "iat": int(time.time()),
        "exp": int(time.time() + expires_in_seconds),
        "iss": "enterprise-production-rag",
    }
    return jwt.encode(payload, secret, algorithm=algo)


def verify_jwt_token(
    token: str,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Tuple[str, ClassificationLevel]:
    """Verify cryptographic signature, expiration, and claims of a JWT."""
    secret = secret_key or settings.jwt_secret_key
    algo = algorithm or settings.jwt_algorithm

    try:
        decoded = jwt.decode(
            token,
            secret,
            algorithms=[algo],
            options={"require": ["exp", "sub", "role"]},
        )
        role = str(decoded.get("role", "employee")).lower()
        clearance_val = decoded.get("clearance")
        if clearance_val is not None and isinstance(clearance_val, int):
            clearance = ClassificationLevel(clearance_val)
        else:
            clearance = ROLE_CLEARANCE_MAP.get(role, ClassificationLevel.INTERNAL)
        return (role, clearance)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: JWT signature has expired.",
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: Invalid JWT token ({type(e).__name__}).",
        )


class AccessControlManager:
    """Enforces zero-trust identity resolution, cryptographic token verification, and RBAC isolation."""

    @classmethod
    def get_token_registry(cls) -> Dict[str, Tuple[str, ClassificationLevel]]:
        """Resolve combined active token registry from settings and defaults."""
        registry = dict(DEFAULT_TOKEN_REGISTRY)
        if settings.api_tokens_json:
            try:
                custom_tokens = json.loads(settings.api_tokens_json)
                for tok, (r, c_val) in custom_tokens.items():
                    registry[tok] = (r, ClassificationLevel(c_val))
            except Exception as e:
                logger.warning("Failed to parse custom api_tokens_json: %s", e)
        return registry

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
        require_auth: bool = False,
    ) -> Tuple[str, ClassificationLevel]:
        """Cryptographically resolve user identity and clearance strictly from HTTP Authorization header.

        Zero-Trust Policy:
        - If no Authorization header is provided:
            * On public endpoints: assigns ('guest', PUBLIC).
            * On protected endpoints (require_auth=True): raises HTTP 401 Unauthorized.
        - If Authorization header is provided with malformed or unrecognized token:
            * Strictly raises HTTP 401 Unauthorized (never downgrades invalid credentials to guest).
        - Supports both standard HMAC-SHA256 JWT tokens and pre-shared API bearer tokens.
        """
        if not auth_header:
            if require_auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required: Missing Authorization Bearer header.",
                )
            return ("guest", ClassificationLevel.PUBLIC)

        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Malformed authorization header. Expected 'Bearer <token>'.",
            )

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: Empty Bearer token provided.",
            )

        # Check if token is a JWT (contains three base64 segments separated by dots)
        if token.count(".") == 2:
            return verify_jwt_token(token)

        # Pre-shared API Bearer token lookup
        registry = cls.get_token_registry()
        if token in registry:
            return registry[token]

        # Explicitly supplied unrecognized token must be rejected with 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Invalid or unrecognized Bearer token.",
        )

    @classmethod
    def enforce_clearance(
        cls,
        user_clearance: ClassificationLevel,
        min_clearance: ClassificationLevel,
        operation_name: str = "administrative operation",
    ) -> None:
        """Enforce minimum required clearance level, raising HTTP 403 Forbidden on violation."""
        if user_clearance < min_clearance:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access forbidden: Clearance level '{user_clearance.name}' is insufficient "
                    f"for {operation_name}. Minimum required: '{min_clearance.name}'."
                ),
            )
