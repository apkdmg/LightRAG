from datetime import datetime, timedelta
import logging
import re
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel

from .config import global_args

logger = logging.getLogger(__name__)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def sanitize_workspace_id(username: str) -> str:
    """
    Convert a username to a valid workspace ID.

    Replaces invalid characters with underscores and ensures the ID
    is safe for use in file paths and database identifiers.

    Args:
        username: The username to sanitize.

    Returns:
        A sanitized workspace ID string.
    """
    # Replace @ and . with underscores, remove other special chars
    workspace_id = re.sub(r"[^a-zA-Z0-9_-]", "_", username.lower())
    # Remove consecutive underscores
    workspace_id = re.sub(r"_+", "_", workspace_id)
    # Remove leading/trailing underscores
    workspace_id = workspace_id.strip("_")
    # Ensure non-empty
    if not workspace_id:
        workspace_id = "default"
    return workspace_id


class TokenPayload(BaseModel):
    sub: str  # Username
    exp: datetime  # Expiration time
    role: str = "user"  # User role, default is regular user
    workspace_id: Optional[str] = None  # Workspace ID derived from username
    metadata: dict = {}  # Additional metadata


class AuthHandler:
    def __init__(self):
        self.secret = global_args.token_secret
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        self.accounts = {}
        auth_accounts = global_args.auth_accounts
        if auth_accounts:
            for account in auth_accounts.split(","):
                username, password = account.split(":", 1)
                self.accounts[username] = password

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: int = None,
        metadata: dict = None,
        workspace_id: str = None,
    ) -> str:
        """
        Create JWT token

        Args:
            username: Username
            role: User role, default is "user", guest is "guest"
            custom_expire_hours: Custom expiration time (hours), if None use default value
            metadata: Additional metadata
            workspace_id: Optional workspace ID, if None will be derived from username

        Returns:
            str: Encoded JWT token
        """
        # Choose default expiration time based on role
        if custom_expire_hours is None:
            if role == "guest":
                expire_hours = self.guest_expire_hours
            else:
                expire_hours = self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = datetime.utcnow() + timedelta(hours=expire_hours)

        # Derive workspace_id from username if not provided
        if workspace_id is None:
            workspace_id = sanitize_workspace_id(username)

        # Create payload
        payload = TokenPayload(
            sub=username,
            exp=expire,
            role=role,
            workspace_id=workspace_id,
            metadata=metadata or {},
        )

        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        """
        Validate JWT token

        Args:
            token: JWT token

        Returns:
            dict: Dictionary containing user information including workspace_id

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            expire_timestamp = payload["exp"]
            expire_time = datetime.utcfromtimestamp(expire_timestamp)

            if datetime.utcnow() > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            username = payload["sub"]
            # Get workspace_id from token or derive from username for backwards compatibility
            workspace_id = payload.get("workspace_id") or sanitize_workspace_id(username)

            # Return complete payload including workspace_id
            return {
                "username": username,
                "role": payload.get("role", "user"),
                "workspace_id": workspace_id,
                "metadata": payload.get("metadata", {}),
                "exp": expire_time,
            }
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


auth_handler = AuthHandler()


def _is_admin_user(username: str) -> bool:
    """
    Check if username is in ADMIN_ACCOUNTS.

    Args:
        username: The username to check

    Returns:
        bool: True if user is an admin
    """
    admin_accounts = global_args.admin_accounts
    if not admin_accounts:
        return False
    admins = [a.strip().lower() for a in admin_accounts.split(",")]
    return username.lower() in admins


def validate_any_token(token: str) -> dict:
    """
    Validate token as LightRAG JWT or Keycloak access token.

    This is a hybrid validator that supports:
    - LightRAG JWT (from /login endpoint)
    - Keycloak user access token (from Authorization Code flow)
    - Keycloak service account token (from Client Credentials flow)

    Returns standardized user info dict with keys:
    - username: The user's username
    - role: User role (admin, user, guest)
    - workspace_id: The user's workspace ID
    - metadata: Additional metadata including auth_mode

    Args:
        token: The JWT token to validate

    Returns:
        dict: Standardized user info

    Raises:
        HTTPException: If all validation methods fail
    """
    from .oauth2 import get_keycloak_client

    # 1. Try LightRAG JWT first (fast, local validation)
    try:
        return auth_handler.validate_token(token)
    except HTTPException:
        pass

    # 2. Try Keycloak access token if OAuth2 is enabled
    keycloak_client = get_keycloak_client()
    if keycloak_client:
        try:
            payload = keycloak_client.validate_access_token(token)

            # Check if this is a service account (Client Credentials)
            if keycloak_client.is_service_account_token(payload):
                # Service accounts get admin role for on-behalf operations
                client_id = payload.get("clientId") or payload.get("azp")
                return {
                    "username": f"service-account-{client_id}",
                    "role": "admin",  # Service accounts have admin privileges
                    "workspace_id": "service_account",
                    "metadata": {
                        "auth_mode": "client_credentials",
                        "client_id": client_id,
                        "scope": payload.get("scope", ""),
                    },
                }

            # Regular user access token
            # Use email for workspace_id derivation to ensure consistency with
            # cookie-based SSO login (which uses email as the username/sub)
            email = payload.get("email")
            preferred_username = payload.get("preferred_username") or payload.get("sub")
            # For display/logging, use preferred_username; for workspace, use email
            username = preferred_username
            # Derive workspace_id from email to match SSO login behavior
            workspace_source = email or preferred_username
            role = "admin" if _is_admin_user(username) else "user"

            logger.info(
                f"OAuth2 user resolved: username={username}, "
                f"workspace_source={workspace_source}, "
                f"workspace_id={sanitize_workspace_id(workspace_source)}"
            )

            return {
                "username": username,
                "role": role,
                "workspace_id": sanitize_workspace_id(workspace_source),
                "metadata": {
                    "auth_mode": "keycloak_direct",
                    "email": email,
                },
            }
        except HTTPException:
            pass

    # All validation failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )
