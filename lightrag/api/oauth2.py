"""
OAuth2/Keycloak SSO Authentication Module for LightRAG.

Handles:
- PKCE code verifier/challenge generation
- Authorization URL construction
- Token exchange with Keycloak
- JWKS-based ID token validation
- User info extraction

This module works alongside existing JWT authentication to provide
SSO capabilities while maintaining backward compatibility.
"""

import secrets
import hashlib
import base64
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlencode

import httpx
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, status

logger = logging.getLogger("lightrag.api.oauth2")


@dataclass
class OAuth2Config:
    """OAuth2/Keycloak configuration from environment."""

    enabled: bool
    client_id: str
    client_secret: Optional[str]
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    issuer: str
    redirect_uri: str
    scopes: str = "openid profile email"


class KeycloakClient:
    """
    Handles Keycloak OAuth2 operations.

    Implements the Authorization Code Flow with PKCE for secure
    authentication in public clients (SPAs).
    """

    def __init__(self, config: OAuth2Config):
        self.config = config
        self._jwks_client: Optional[PyJWKClient] = None
        self._state_store: Dict[str, Dict[str, Any]] = {}  # In-memory state storage

    def generate_pkce_pair(self) -> tuple:
        """
        Generate PKCE code_verifier and code_challenge.

        Returns:
            tuple: (code_verifier, code_challenge) pair
        """
        # Generate a cryptographically random code verifier (43-128 chars)
        code_verifier = secrets.token_urlsafe(64)

        # Create S256 code challenge
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        return code_verifier, code_challenge

    def generate_state(self) -> str:
        """
        Generate a cryptographically secure state parameter for CSRF protection.

        Returns:
            str: A URL-safe random string
        """
        return secrets.token_urlsafe(32)

    def store_auth_state(
        self, state: str, code_verifier: str, expires_in: int = 600
    ) -> None:
        """
        Store state and PKCE verifier temporarily.

        Args:
            state: The state parameter to use as key
            code_verifier: The PKCE code verifier to store
            expires_in: Expiration time in seconds (default: 10 minutes)
        """
        # Clean up expired states first
        self._cleanup_expired_states()

        self._state_store[state] = {
            "code_verifier": code_verifier,
            "expires": datetime.utcnow() + timedelta(seconds=expires_in),
        }
        logger.debug(f"Stored auth state: {state[:8]}...")

    def get_auth_state(self, state: str) -> Optional[str]:
        """
        Retrieve and remove stored code_verifier for state.

        Args:
            state: The state parameter to look up

        Returns:
            The code_verifier if found and not expired, None otherwise
        """
        data = self._state_store.pop(state, None)
        if data and data["expires"] > datetime.utcnow():
            logger.debug(f"Retrieved auth state: {state[:8]}...")
            return data["code_verifier"]

        if data:
            logger.warning(f"Auth state expired: {state[:8]}...")
        else:
            logger.warning(f"Auth state not found: {state[:8]}...")
        return None

    def _cleanup_expired_states(self) -> None:
        """Remove expired states from the store."""
        now = datetime.utcnow()
        expired = [
            state
            for state, data in self._state_store.items()
            if data["expires"] <= now
        ]
        for state in expired:
            del self._state_store[state]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired auth states")

    def get_authorization_url(self) -> tuple:
        """
        Build Keycloak authorization URL with PKCE.

        Returns:
            tuple: (authorization_url, state) pair
        """
        state = self.generate_state()
        code_verifier, code_challenge = self.generate_pkce_pair()
        self.store_auth_state(state, code_verifier)

        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "scope": self.config.scopes,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        authorization_url = f"{self.config.authorization_endpoint}?{urlencode(params)}"
        logger.info(f"Generated authorization URL with state: {state[:8]}...")

        return authorization_url, state

    async def exchange_code(self, code: str, state: str) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.

        Args:
            code: The authorization code from Keycloak
            state: The state parameter for validation

        Returns:
            dict: Token response containing access_token, id_token, etc.

        Raises:
            HTTPException: If state is invalid or token exchange fails
        """
        code_verifier = self.get_auth_state(state)
        if not code_verifier:
            logger.error(f"Invalid or expired state parameter: {state[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state parameter. Please try logging in again.",
            )

        # Prepare token request data
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "code_verifier": code_verifier,
        }

        # Add client_secret if configured (for confidential clients)
        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        logger.info("Exchanging authorization code for tokens...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self.config.token_endpoint,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            except httpx.RequestError as e:
                logger.error(f"Token exchange request failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to connect to identity provider",
                )

        if response.status_code != 200:
            error_detail = "Failed to exchange authorization code"
            try:
                error_data = response.json()
                error_detail = error_data.get(
                    "error_description", error_data.get("error", error_detail)
                )
            except Exception:
                pass

            logger.error(
                f"Token exchange failed: {response.status_code} - {error_detail}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=error_detail,
            )

        logger.info("Token exchange successful")
        return response.json()

    @property
    def jwks_client(self) -> PyJWKClient:
        """
        Lazy-load JWKS client for token validation.

        Returns:
            PyJWKClient: Client for fetching and caching JWKS keys
        """
        if self._jwks_client is None:
            self._jwks_client = PyJWKClient(
                self.config.jwks_uri,
                cache_jwk_set=True,
                lifespan=3600,  # Cache keys for 1 hour
            )
            logger.info(f"Initialized JWKS client for: {self.config.jwks_uri}")
        return self._jwks_client

    def validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Validate ID token using Keycloak JWKS.

        Args:
            id_token: The JWT ID token from Keycloak

        Returns:
            dict: Decoded token payload

        Raises:
            HTTPException: If token validation fails
        """
        try:
            # Get the signing key from JWKS
            signing_key = self.jwks_client.get_signing_key_from_jwt(id_token)

            # Decode and validate the token
            payload = jwt.decode(
                id_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.client_id,
                issuer=self.config.issuer,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            logger.info(
                f"ID token validated for user: {payload.get('preferred_username', payload.get('sub'))}"
            )
            return payload

        except jwt.ExpiredSignatureError:
            logger.error("ID token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="ID token has expired",
            )
        except jwt.InvalidAudienceError:
            logger.error("ID token audience mismatch")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="ID token audience mismatch",
            )
        except jwt.InvalidIssuerError:
            logger.error("ID token issuer mismatch")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="ID token issuer mismatch",
            )
        except jwt.PyJWTError as e:
            logger.error(f"ID token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid ID token: {str(e)}",
            )

    def extract_user_info(self, token_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract user information from validated token payload.

        Args:
            token_payload: Decoded JWT payload from Keycloak

        Returns:
            dict: Extracted user information
        """
        return {
            "sub": token_payload.get("sub"),
            "email": token_payload.get("email"),
            "email_verified": token_payload.get("email_verified", False),
            "preferred_username": token_payload.get("preferred_username"),
            "name": token_payload.get("name"),
            "given_name": token_payload.get("given_name"),
            "family_name": token_payload.get("family_name"),
        }

    def validate_access_token(self, access_token: str) -> Dict[str, Any]:
        """
        Validate Keycloak access token using JWKS.

        Supports both:
        - User access tokens (from Authorization Code flow)
        - Service account tokens (from Client Credentials flow)

        Unlike ID tokens, access tokens may not have audience claim for the client.
        We validate: signature, issuer, expiration.

        Args:
            access_token: The JWT access token from Keycloak

        Returns:
            dict: Decoded token payload

        Raises:
            HTTPException: If token validation fails
        """
        try:
            # Get the signing key from JWKS
            signing_key = self.jwks_client.get_signing_key_from_jwt(access_token)

            # Decode and validate the token
            # Note: Access tokens may not have audience claim for the client,
            # so we disable audience verification
            payload = jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.config.issuer,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": False,  # Access tokens may not have client as audience
                    "verify_iss": True,
                },
            )

            logger.info(
                f"Access token validated for: {payload.get('preferred_username', payload.get('sub', 'service-account'))}"
            )
            return payload

        except jwt.ExpiredSignatureError:
            logger.error("Access token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access token has expired",
            )
        except jwt.InvalidIssuerError:
            logger.error("Access token issuer mismatch")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access token issuer mismatch",
            )
        except jwt.PyJWTError as e:
            logger.error(f"Access token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid access token: {str(e)}",
            )

    def is_service_account_token(self, payload: Dict[str, Any]) -> bool:
        """
        Check if the token is from Client Credentials grant (service account).

        Client Credentials tokens have:
        - No 'preferred_username' (or it's the client_id)
        - 'clientId' claim present
        - 'azp' (authorized party) equals the client_id

        Service account usernames typically follow pattern: service-account-<client_id>

        Args:
            payload: Decoded JWT payload

        Returns:
            bool: True if this is a service account token
        """
        # Check for service account indicators
        client_id = payload.get("clientId") or payload.get("azp")
        preferred_username = payload.get("preferred_username", "")

        # Service account usernames typically follow pattern: service-account-<client_id>
        if preferred_username.startswith("service-account-"):
            return True

        # Or no preferred_username but has clientId
        if not preferred_username and client_id:
            return True

        return False


# Module-level singleton
_keycloak_client: Optional[KeycloakClient] = None


def get_keycloak_client() -> Optional[KeycloakClient]:
    """
    Get or create Keycloak client instance.

    Returns:
        KeycloakClient if OAuth2 is enabled, None otherwise
    """
    global _keycloak_client

    # Import here to avoid circular imports
    from .config import global_args

    if _keycloak_client is None and getattr(global_args, "oauth2_enabled", False):
        config = OAuth2Config(
            enabled=global_args.oauth2_enabled,
            client_id=global_args.oauth2_client_id,
            client_secret=global_args.oauth2_client_secret,
            authorization_endpoint=global_args.oauth2_authorization_endpoint,
            token_endpoint=global_args.oauth2_token_endpoint,
            userinfo_endpoint=global_args.oauth2_userinfo_endpoint,
            jwks_uri=global_args.oauth2_jwks_uri,
            issuer=global_args.oauth2_issuer,
            redirect_uri=global_args.oauth2_redirect_uri,
            scopes=global_args.oauth2_scopes,
        )
        _keycloak_client = KeycloakClient(config)
        logger.info("Keycloak client initialized")

    return _keycloak_client


def reset_keycloak_client() -> None:
    """
    Reset the Keycloak client singleton.

    Useful for testing or when configuration changes.
    """
    global _keycloak_client
    _keycloak_client = None
    logger.info("Keycloak client reset")
