"""
OBO (On-Behalf-Of) Client Allowlist Manager.

Controls which OAuth2 clients and API keys can perform OBO operations
(accessing workspaces via X-Target-Workspace header).

Uses a separate config file (.obo_allowlist) with .env-like format for
hot-reload support (changes apply without server restart).

Config file format (.obo_allowlist):
    # Format: [client_id:workspace1,workspace2] or [client_id:*] for all
    OBO_ALLOWED_CLIENTS=[partner-app:tenant_a,tenant_b],[backend-service:*]
    OBO_API_KEY_ALLOWED=true
    OBO_API_KEY_WORKSPACES=*
    OBO_DEFAULT_POLICY=deny
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

logger = logging.getLogger("lightrag.api.obo_allowlist")

# Cache TTL in seconds
DEFAULT_CACHE_TTL_SECONDS = 60


@dataclass
class ClientPermissions:
    """Permissions for a single OAuth2 client."""

    client_id: str
    allowed_workspaces: Set[str]  # Specific workspaces allowed
    allow_all_workspaces: bool = False  # If True, "*" was specified


@dataclass
class OBOAllowlistConfig:
    """Parsed OBO allowlist configuration."""

    default_policy: str = "deny"  # "allow" or "deny"
    clients: Dict[str, ClientPermissions] = field(default_factory=dict)
    api_key_obo_allowed: bool = False
    api_key_allowed_workspaces: Set[str] = field(default_factory=set)
    api_key_allow_all_workspaces: bool = False
    file_mtime: float = 0.0


def _parse_allowed_clients(value: str) -> Dict[str, ClientPermissions]:
    """
    Parse OBO_ALLOWED_CLIENTS format.

    Format: [client_id:workspace1,workspace2],[client_id:*]

    Examples:
        [backend-service:*]
        [partner-app:tenant_a,tenant_b],[backend-service:*]

    Args:
        value: The config value

    Returns:
        Dict of client_id -> ClientPermissions
    """
    clients: Dict[str, ClientPermissions] = {}

    if not value or not value.strip():
        return clients

    # Find all [client_id:workspaces] patterns
    pattern = r"\[([^:\]]+):([^\]]+)\]"
    matches = re.findall(pattern, value)

    for client_id, workspaces_str in matches:
        client_id = client_id.strip()
        workspaces_str = workspaces_str.strip()

        if not client_id:
            continue

        if workspaces_str == "*":
            clients[client_id] = ClientPermissions(
                client_id=client_id,
                allowed_workspaces=set(),
                allow_all_workspaces=True,
            )
        else:
            workspaces = {w.strip() for w in workspaces_str.split(",") if w.strip()}
            clients[client_id] = ClientPermissions(
                client_id=client_id,
                allowed_workspaces=workspaces,
                allow_all_workspaces=False,
            )

    return clients


def _parse_workspaces(value: str) -> tuple[Set[str], bool]:
    """
    Parse workspace list.

    Args:
        value: Comma-separated workspaces or "*" for all

    Returns:
        Tuple of (workspace_set, allow_all_flag)
    """
    if not value or not value.strip():
        return set(), False

    value = value.strip()
    if value == "*":
        return set(), True

    workspaces = {w.strip() for w in value.split(",") if w.strip()}
    return workspaces, False


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower().strip() in ("true", "1", "yes")


def _parse_config_file(file_path: str) -> Dict[str, str]:
    """
    Parse .env-like config file.

    Args:
        file_path: Path to config file

    Returns:
        Dict of key -> value
    """
    config = {}

    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    except Exception as e:
        logger.debug(f"Could not read config file {file_path}: {e}")

    return config


class OBOAllowlistManager:
    """
    Manages OBO client allowlist with hot-reload support.

    Reads from a .obo_allowlist file with .env-like format.
    Uses TTL-based caching - config is reloaded if file changed.
    """

    def __init__(self, config_path: Optional[str] = None, cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        self._config_path = config_path
        self._cache_ttl = cache_ttl
        self._config: Optional[OBOAllowlistConfig] = None
        self._last_check: float = 0.0
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization to get working_dir from global_args."""
        if self._initialized:
            return

        if self._config_path is None:
            # Check env var first, then default to working_dir
            env_path = os.getenv("OBO_ALLOWLIST_PATH")
            if env_path:
                self._config_path = env_path
            else:
                from .config import global_args
                working_dir = getattr(global_args, "working_dir", "./rag_storage")
                self._config_path = os.path.join(working_dir, ".obo_allowlist")

        self._initialized = True

    def _should_reload(self) -> bool:
        """Check if config should be reloaded based on TTL and file mtime."""
        now = time.time()
        if now - self._last_check < self._cache_ttl:
            return False
        self._last_check = now

        try:
            mtime = os.path.getmtime(self._config_path)
            if self._config and mtime == self._config.file_mtime:
                return False
            return True
        except OSError:
            # File doesn't exist
            return self._config is not None or self._config is None

    def _load_config(self) -> OBOAllowlistConfig:
        """Load and parse config from file."""
        config = OBOAllowlistConfig()

        # Default policy from env (fallback)
        config.default_policy = os.getenv("OBO_DEFAULT_POLICY", "deny").lower()

        if not os.path.exists(self._config_path):
            logger.info(
                f"OBO allowlist file not found: {self._config_path}, "
                f"using default_policy={config.default_policy}"
            )
            return config

        try:
            config.file_mtime = os.path.getmtime(self._config_path)
            file_config = _parse_config_file(self._config_path)

            # Parse values
            if "OBO_DEFAULT_POLICY" in file_config:
                config.default_policy = file_config["OBO_DEFAULT_POLICY"].lower()
                if config.default_policy not in ("allow", "deny"):
                    logger.warning(f"Invalid OBO_DEFAULT_POLICY, using 'deny'")
                    config.default_policy = "deny"

            if "OBO_ALLOWED_CLIENTS" in file_config:
                config.clients = _parse_allowed_clients(file_config["OBO_ALLOWED_CLIENTS"])

            if "OBO_API_KEY_ALLOWED" in file_config:
                config.api_key_obo_allowed = _parse_bool(file_config["OBO_API_KEY_ALLOWED"])

            if "OBO_API_KEY_WORKSPACES" in file_config:
                config.api_key_allowed_workspaces, config.api_key_allow_all_workspaces = (
                    _parse_workspaces(file_config["OBO_API_KEY_WORKSPACES"])
                )

            logger.info(
                f"OBO allowlist loaded from {self._config_path}: "
                f"{len(config.clients)} clients, "
                f"api_key_obo={config.api_key_obo_allowed}, "
                f"default_policy={config.default_policy}"
            )

        except Exception as e:
            logger.error(f"Failed to load OBO allowlist: {e}")

        return config

    def get_config(self) -> OBOAllowlistConfig:
        """Get current config, reloading if necessary."""
        self._ensure_initialized()
        if self._config is None or self._should_reload():
            self._config = self._load_config()
        return self._config

    def is_client_allowed(self, client_id: str, target_workspace: str) -> bool:
        """
        Check if a client is allowed to perform OBO for a workspace.

        Args:
            client_id: The OAuth2 client_id or "api_key" for shared API key
            target_workspace: The target workspace being accessed

        Returns:
            True if allowed, False otherwise
        """
        config = self.get_config()

        # Handle shared API key
        if client_id == "api_key":
            if not config.api_key_obo_allowed:
                logger.debug("OBO check: api_key not allowed (OBO_API_KEY_ALLOWED=false)")
                return config.default_policy == "allow"
            if config.api_key_allow_all_workspaces:
                return True
            allowed = target_workspace in config.api_key_allowed_workspaces
            if not allowed:
                logger.debug(f"OBO check: api_key not allowed for workspace '{target_workspace}'")
            return allowed

        # Handle OAuth2 client
        if client_id not in config.clients:
            logger.debug(
                f"OBO check: client '{client_id}' not in allowlist, "
                f"using default_policy={config.default_policy}"
            )
            return config.default_policy == "allow"

        permissions = config.clients[client_id]
        if permissions.allow_all_workspaces:
            return True

        allowed = target_workspace in permissions.allowed_workspaces
        if not allowed:
            logger.debug(
                f"OBO check: client '{client_id}' not allowed for workspace '{target_workspace}'"
            )
        return allowed

    def reload(self) -> None:
        """Force reload config."""
        self._ensure_initialized()
        self._last_check = 0
        self._config = self._load_config()


# Global singleton
_manager: Optional[OBOAllowlistManager] = None


def get_manager() -> OBOAllowlistManager:
    """Get or create the global OBO allowlist manager."""
    global _manager
    if _manager is None:
        _manager = OBOAllowlistManager()
    return _manager


def check_obo_allowed(client_id: str, target_workspace: str) -> bool:
    """
    Check if a client is allowed to perform OBO for a workspace.

    Args:
        client_id: The OAuth2 client_id or "api_key" for shared API key
        target_workspace: The target workspace being accessed

    Returns:
        True if allowed, False otherwise
    """
    return get_manager().is_client_allowed(client_id, target_workspace)


def reload_config() -> None:
    """Force reload the OBO allowlist config."""
    get_manager().reload()
