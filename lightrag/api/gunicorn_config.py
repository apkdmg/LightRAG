# gunicorn_config.py
import os
import logging
import logging.handlers
from lightrag.api.ops.context import (
    CONSOLE_LOG_FORMAT,
    DETAILED_LOG_FORMAT,
    ContextFormatter,
)
from lightrag.kg.shared_storage import finalize_share_data
from lightrag.utils import (
    LightragPathFilter,
    SafeStreamHandler,
    attach_log_hygiene,
    get_env_value,
)
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
)


# Get log directory path from environment variable
log_dir = os.getenv("LOG_DIR", os.getcwd())
log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

# Ensure log directory exists (owner-only when created here)
os.makedirs(os.path.dirname(log_file_path), mode=0o700, exist_ok=True)

# Get log file max size and backup count from environment variables
log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

# These variables will be set by run_with_gunicorn.py
workers = None
bind = None
loglevel = None
certfile = None
keyfile = None

# Enable preload_app option
preload_app = True

# Use Uvicorn worker
worker_class = "uvicorn.workers.UvicornWorker"

# Other Gunicorn configurations

# Logging configuration
errorlog = os.getenv("ERROR_LOG", log_file_path)  # Default write to lightrag.log
accesslog = os.getenv("ACCESS_LOG", log_file_path)  # Default write to lightrag.log


_LEVEL_NAMES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def resolve_log_level(level: str | None = None) -> str:
    """The effective level name: ``level``, else ``LOG_LEVEL``, else INFO."""
    name = (level or os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    if name == "WARN":
        name = "WARNING"
    return name if name in _LEVEL_NAMES else "INFO"


def build_logconfig_dict(level: str | None = None) -> dict:
    """Gunicorn's logging config at ``level``.

    Built as a function because the module is imported before
    run_with_gunicorn.py knows the log level; it calls this again once
    ``loglevel`` is set, so LOG_LEVEL is honoured.
    """
    level_name = resolve_log_level(level)
    handler_filters = ["redact", "context"]
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": "lightrag.api.ops.context.ContextFormatter",
                "fmt": DETAILED_LOG_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
                "filters": handler_filters,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf8",
                "filters": handler_filters,
            },
        },
        "filters": {
            "path_filter": {
                "()": "lightrag.utils.LightragPathFilter",
            },
            "redact": {
                "()": "lightrag.api.ops.redact.RedactFilter",
            },
            "context": {
                "()": "lightrag.api.ops.context.ContextFilter",
            },
        },
        "loggers": {
            "lightrag": {
                "handlers": ["console", "file"],
                "level": level_name,
                "propagate": False,
            },
            "gunicorn": {
                "handlers": ["console", "file"],
                "level": level_name,
                "propagate": False,
            },
            "gunicorn.error": {
                "handlers": ["console", "file"],
                "level": level_name,
                "propagate": False,
            },
            "gunicorn.access": {
                "handlers": ["console", "file"],
                "level": level_name,
                "propagate": False,
                "filters": ["path_filter"],
            },
        },
    }


logconfig_dict = build_logconfig_dict(loglevel)


def configure_worker_logging(level: str | None = None) -> None:
    """Per-worker logging: one console and one file handler for the process.

    ``lightrag``, ``uvicorn`` and ``uvicorn.access`` share the same two
    handlers, and every ``lightrag.*`` child logger propagates to ``lightrag``
    instead of owning its own rotating file handler. ``uvicorn.error`` stays
    enabled at ERROR so unhandled-exception tracebacks are kept.
    """
    level_name = resolve_log_level(level)
    console_handler = SafeStreamHandler()
    console_handler.setFormatter(ContextFormatter(CONSOLE_LOG_FORMAT))
    attach_log_hygiene(console_handler)
    handlers: list[logging.Handler] = [console_handler]
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(ContextFormatter(DETAILED_LOG_FORMAT))
        attach_log_hygiene(file_handler)
        handlers.append(file_handler)
    except OSError as e:
        print(f"Could not open log file {log_file_path}: {e}; console logging only")

    for name, add_filter in (
        ("lightrag", True),
        ("uvicorn", False),
        ("uvicorn.access", True),
    ):
        target = logging.getLogger(name)
        target.handlers = list(handlers)
        target.filters = []
        target.setLevel(level_name)
        target.propagate = False
        if add_filter:
            target.addFilter(LightragPathFilter())

    # Child loggers carry no handlers of their own; records reach the shared
    # pair through the "lightrag" logger.
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("lightrag."):
            child = logging.getLogger(name)
            child.handlers = []
            child.setLevel(logging.NOTSET)
            child.propagate = True

    # Unhandled exceptions in the ASGI app are logged on uvicorn.error. It
    # propagates to the "uvicorn" handlers; its own INFO chatter (per-worker
    # startup lines) stays out.
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers = []
    uvicorn_error_logger.filters = []
    uvicorn_error_logger.setLevel(max(logging.ERROR, logging.getLevelName(level_name)))
    uvicorn_error_logger.propagate = True


def on_starting(server):
    """
    Executed when Gunicorn starts, before forking the first worker processes
    You can use this function to do more initialization tasks for all processes
    """
    print("=" * 80)
    print(f"GUNICORN MASTER PROCESS: on_starting jobs for {workers} worker(s)")
    print(f"Process ID: {os.getpid()}")
    print("=" * 80)

    # Memory usage monitoring
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        msg = (
            f"Memory usage after initialization: {memory_info.rss / 1024 / 1024:.2f} MB"
        )
        print(msg)
    except ImportError:
        print("psutil not installed, skipping memory usage reporting")

    # Log the location of the LightRAG log file
    print(f"LightRAG log file: {log_file_path}\n")

    print("Gunicorn initialization complete, forking workers...\n")


def on_exit(server):
    """
    Executed when Gunicorn is shutting down.
    This is a good place to release shared resources.
    """
    print("=" * 80)
    print("GUNICORN MASTER PROCESS: Shutting down")
    print(f"Process ID: {os.getpid()}")

    print("Finalizing shared storage...")
    finalize_share_data()

    print("Gunicorn shutdown complete")
    print("=" * 80)


def post_fork(server, worker):
    """
    Executed after a worker has been forked.
    This is a good place to set up worker-specific configurations.
    """
    configure_worker_logging(loglevel)
