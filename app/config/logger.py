import sys
from loguru import logger

def setup_logger():
    """Configure loguru logger for the application with colored output."""

    # Remove default logger
    logger.remove()

    logger.add(
        sys.stderr,
        format="<level>{time:YYYY-MM-DD HH:mm:ss} | {level: <6} | {name}:{function}:{line} - {message}</level>",
        level="DEBUG",
        colorize=True,
    )

    logger.add(
        "logs/app.log",
        rotation="10 MB",       # Rotate when file size reaches 10MB
        retention="1 week",      # Keep logs for 1 week
        compression="zip",       # Compress old logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <6} | {function}:{line} - {message}",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )

    return logger

# Create a pre-configured logger instance
logger = setup_logger()