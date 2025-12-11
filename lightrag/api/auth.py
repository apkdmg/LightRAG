from datetime import datetime, timedelta
import re
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel

from .config import global_args

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
