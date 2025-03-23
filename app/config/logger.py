import sys
from loguru import logger

def setup_logger():
    """Configure loguru logger with custom colors (INFO in green)."""

    # Remove default logger
    logger.remove()

    # Console logger with custom colors
    logger.add(
        sys.stderr,
        format="<level>{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} {message}</level>\n",
        level="DEBUG",
        colorize=True,
    )

    # # Custom colors for specific levels
    # logger.level("INFO", color="<green>")
    # logger.level("SUCCESS", color="<green><bold>")
    # logger.level("DEBUG", color="<italic><cyan>")
    # logger.level("WARNING", color="<fg #FFA500>")
    # logger.level("ERROR", color="<red>")
    # logger.level("CRITICAL", color="<bold><white><bg red>")

    # Custom colors for specific levels
    logger.level("INFO", color="<bold><fg #2ECC71>")         # Emerald Green → Sleek & vibrant with bold + italic
    logger.level("SUCCESS", color="<bold><fg #1ABC9C>")    # Turquoise → Stylish & modern with underline
    logger.level("DEBUG", color="<italic><fg #3498DB>")               # Dodger Blue → Sleek blue with italic
    logger.level("WARNING", color="<bold><fg #F39C12>")       # Sunflower Orange → Bold orange with italic
    logger.level("ERROR", color="<bold><red>")                # Alizarin → Bold red
    logger.level("CRITICAL", color="<bold><fg #FFFFFF><bg #8B0000>")  # Dark Red → Bold white text on dark red background for critical errors

    # File logger with standard formatting
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} {message}\n"
        ),
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )

    return logger

# Create a pre-configured logger instance
logger = setup_logger()
