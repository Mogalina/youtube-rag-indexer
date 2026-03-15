from pathlib import Path
from loguru import logger
from typing import Optional


# Default log file path (relative to project root)
_DEFAULT_LOG_FILE = "logs/tubx.log"


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_type: str = "json"
) -> None:
    """
    Configure the logger with specified settings.

    Logs are written only to a rotating file — never to the console.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to logs/tubx.log)
        rotation: Log file rotation threshold (default 100 MB circular)
        retention: Log retention period
        format_type: Format type ('json' or 'text')
    """
    # Remove all default handlers (suppresses console output)
    logger.remove()

    # JSON format for file logging
    log_format = (
        "{{" 
        '"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
        '"level": "{level}", '
        '"module": "{module}", '
        '"function": "{function}", '
        '"line": {line}, '
        '"message": "{message}"'
        "}}"
    )

    # Resolve log file path
    resolved_log_file = log_file or _DEFAULT_LOG_FILE
    log_path = Path(resolved_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # File-only sink with circular rotation at 100 MB
    logger.add(
        resolved_log_file,
        format=log_format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
    )

    logger.info(
        f"Logger initialized with level={level}, "
        f"log_file={resolved_log_file}, "
        f"rotation={rotation}, "
        f"retention={retention}, "
        f"format_type={format_type}"
    )


def get_logger(name: str):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(logger_name=name)
