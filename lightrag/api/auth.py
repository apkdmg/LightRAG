from datetime import datetime, timedelta, timezone
import re
import time
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel

from ..utils import logger
from .config import DEFAULT_TOKEN_SECRET, global_args, is_oauth2_usable
from .passwords import BCRYPT_PASSWORD_PREFIX, hash_password, verify_password

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
    # Session start (epoch seconds). Carried unchanged through auto-renewal so
    # the total session length can be capped (TOKEN_MAX_SESSION_HOURS).
    sst: Optional[int] = None


class AuthHandler:
    def __init__(self):
        auth_accounts = global_args.auth_accounts
        self.secret = global_args.token_secret
        # OAuth2/SSO is "usable" only when enabled AND its client credentials
        # are configured. When usable, the local TOKEN_SECRET signs the session
        # JWT minted after SSO login, so a strong secret is mandatory (fail
        # closed) to prevent admin-token forgery against the public default.
        oauth2_usable = is_oauth2_usable(global_args)
        self.oauth2_usable = oauth2_usable
        if not self.secret:
            if auth_accounts:
                raise ValueError(
                    "TOKEN_SECRET must be explicitly set to a non-default value when AUTH_ACCOUNTS is configured."
                )
            if oauth2_usable:
                raise ValueError(
                    "TOKEN_SECRET must be explicitly set to a non-default value when OAuth2/SSO "
                    "is enabled (OAUTH2_ENABLED=true with OAUTH2_CLIENT_ID and OAUTH2_CLIENT_SECRET set). "
                    "The default secret is public and would allow admin-token forgery."
                )
            self.secret = DEFAULT_TOKEN_SECRET
            logger.warning(
                "TOKEN_SECRET not set and AUTH_ACCOUNTS is not configured. "
                "Falling back to the default guest-mode JWT secret. "
            )
        algorithm = global_args.jwt_algorithm
        if not algorithm or algorithm.lower() == "none":
            raise ValueError(
                "JWT_ALGORITHM must be set to a secure algorithm (e.g. HS256). "
                "The 'none' algorithm is not permitted."
            )
        self.algorithm = algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        self.accounts = {}
        self._dummy_password_hash: Optional[str] = None
        invalid_accounts = []
        if auth_accounts:
            for account in auth_accounts.split(","):
                try:
                    username, password = account.split(":", 1)
                    if not username or not password:
                        raise ValueError
                    self.accounts[username] = password
                except ValueError:
                    invalid_accounts.append(account)
        if invalid_accounts:
            invalid_entries = ", ".join(invalid_accounts)
            logger.error(f"Invalid account format in AUTH_ACCOUNTS: {invalid_entries}")
            raise ValueError(
                "AUTH_ACCOUNTS must use comma-separated user:password pairs."
            )

    @property
    def login_required(self) -> bool:
        """True when users must authenticate: password accounts or usable SSO.

        This — not ``accounts`` alone — decides whether anonymous guest access
        is permitted, so an SSO-only deployment (no AUTH_ACCOUNTS) is not
        treated as "authentication disabled".
        """
        return bool(self.accounts) or self.oauth2_usable

    def verify_password(self, username: str, plain_password: str) -> bool:
        """
        Verify password for a user. Supports explicit bcrypt values and plaintext.

        Args:
            username: Username to verify
            plain_password: Plaintext password to check

        Returns:
            bool: True if password is correct, False otherwise
        """
        if username not in self.accounts:
            # Spend the same time as a real bcrypt check so response timing
            # does not reveal which usernames exist.
            if any(
                p.startswith(BCRYPT_PASSWORD_PREFIX) for p in self.accounts.values()
            ):
                if self._dummy_password_hash is None:
                    self._dummy_password_hash = hash_password("dummy-password")
                verify_password(plain_password, self._dummy_password_hash)
            return False

        stored_password = self.accounts[username]
        return verify_password(plain_password, stored_password)

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: int = None,
        metadata: dict = None,
        workspace_id: str = None,
        session_started_at: Optional[float] = None,
    ) -> str:
        """
        Create JWT token

        Args:
            username: Username
            role: User role, default is "user", guest is "guest"
            custom_expire_hours: Custom expiration time (hours), if None use default value
            metadata: Additional metadata
            workspace_id: Optional workspace ID; if None, derived from username
            session_started_at: Epoch seconds when the session began; defaults
                to now. Pass the original value when renewing a token.

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

        expire = datetime.now(timezone.utc) + timedelta(hours=expire_hours)

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
            sst=int(session_started_at if session_started_at else time.time()),
        )

        return jwt.encode(payload.model_dump(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        """
        Validate JWT token

        Args:
            token: JWT token

        Returns:
            dict: Dictionary containing user information

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Explicitly exclude 'none' to prevent algorithm confusion attacks
            allowed_algorithms = [self.algorithm]
            if "none" in (a.lower() for a in allowed_algorithms):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Insecure JWT algorithm configuration",
                )
            payload = jwt.decode(token, self.secret, algorithms=allowed_algorithms)
            expire_timestamp = payload["exp"]
            expire_time = datetime.fromtimestamp(expire_timestamp, timezone.utc)

            if datetime.now(timezone.utc) > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            username = payload["sub"]
            metadata = payload.get("metadata", {}) or {}

            # A password-login token stops working as soon as its account is
            # removed from AUTH_ACCOUNTS, instead of living until expiry.
            if (
                metadata.get("auth_mode") == "enabled"
                and self.accounts
                and username not in self.accounts
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account no longer exists",
                )

            # Get workspace_id from token, or derive from username for
            # backwards compatibility with tokens issued before this field
            workspace_id = payload.get("workspace_id") or sanitize_workspace_id(
                username
            )

            # Return complete payload including workspace_id
            return {
                "username": username,
                "role": payload.get("role", "user"),
                "workspace_id": workspace_id,
                "metadata": metadata,
                "exp": expire_time,
                "session_started_at": payload.get("sst"),
            }
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


auth_handler = AuthHandler()


def _is_admin_user(*usernames: str) -> bool:
    """
    Check if any of the given identifiers is in ADMIN_ACCOUNTS.

    Matching is case-insensitive and accepts multiple candidate identifiers
    (e.g. email and preferred_username) so the same human is treated as an
    admin regardless of which identifier a given auth path surfaces.

    Args:
        *usernames: One or more candidate identifiers to check. Empty/None
            candidates are ignored.

    Returns:
        bool: True if ANY non-empty candidate matches an ADMIN_ACCOUNTS entry.
    """
    admin_accounts = global_args.admin_accounts
    if not admin_accounts:
        return False
    admins = {a.strip().lower() for a in admin_accounts.split(",") if a.strip()}
    if not admins:
        return False
    return any(u and u.strip().lower() in admins for u in usernames)


def _split_csv(value: Optional[str]) -> set[str]:
    return {v.strip() for v in (value or "").split(",") if v.strip()}


def is_token_issued_for_lightrag(payload: dict) -> bool:
    """
    Check that a Keycloak access token was issued for this server.

    Any client in the same realm can obtain validly signed tokens, so the
    signature and issuer alone do not mean the token is meant for LightRAG.
    Accept it only when:

    - its audience (``aud``) contains OAUTH2_CLIENT_ID or an entry in
      OAUTH2_ALLOWED_AUDIENCES, or
    - its authorized party (``azp``/``clientId``) is OAUTH2_CLIENT_ID, an entry
      in OAUTH2_ALLOWED_CLIENTS, or a client configured in the OBO / admin
      allowlist.
    """
    from .obo_allowlist import is_known_client

    own_client = (getattr(global_args, "oauth2_client_id", "") or "").strip()

    allowed_audiences = _split_csv(getattr(global_args, "oauth2_allowed_audiences", ""))
    if own_client:
        allowed_audiences.add(own_client)
    aud = payload.get("aud")
    audiences = {aud} if isinstance(aud, str) else set(aud or [])
    if audiences & allowed_audiences:
        return True

    allowed_clients = _split_csv(getattr(global_args, "oauth2_allowed_clients", ""))
    if own_client:
        allowed_clients.add(own_client)
    client_id = payload.get("azp") or payload.get("clientId")
    if client_id and (client_id in allowed_clients or is_known_client(client_id)):
        return True

    return False


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

            if not is_token_issued_for_lightrag(payload):
                logger.warning(
                    "Rejected Keycloak access token issued to another client: "
                    f"azp={payload.get('azp') or payload.get('clientId')}"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token was not issued for this service",
                )

            # Check if this is a service account (Client Credentials)
            if keycloak_client.is_service_account_token(payload):
                client_id = payload.get("clientId") or payload.get("azp")
                # Admin is granted ONLY to client IDs explicitly listed in the
                # unified OBO admin allowlist (OBO_ADMIN_CLIENTS in the
                # .obo_allowlist file, hot-reloaded; with a DEPRECATED fallback
                # to the OAUTH2_SERVICE_ACCOUNT_ADMIN_CLIENTS env var). With the
                # safe default (empty allowlist) NO service account becomes
                # admin — any client-credentials token from the same realm
                # authenticates as a normal "user", not a LightRAG admin.
                from .obo_allowlist import check_admin_client

                if check_admin_client(client_id):
                    role = "admin"
                    logger.info(
                        f"Service account granted admin role: client_id={client_id}"
                    )
                else:
                    role = "user"
                    logger.info(
                        f"Service account authenticated as user (not in admin "
                        f"allowlist): client_id={client_id}"
                    )
                return {
                    "username": f"service-account-{client_id}",
                    "role": role,
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
            # The email becomes the workspace identity, so it must be verified.
            if (
                email
                and getattr(global_args, "oauth2_require_verified_email", True)
                and payload.get("email_verified") is not True
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Email address is not verified by the identity provider",
                )
            # Derive workspace_id from email to match SSO login behavior
            workspace_source = email or preferred_username
            role = "admin" if _is_admin_user(preferred_username, email) else "user"

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
