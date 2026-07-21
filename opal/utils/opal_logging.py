# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import os

__env = None


# ANSI color codes
class LogColors:
    GRAY = "\033[90m"  # Gray for DEBUG
    GREEN = "\033[32m"  # Green for INFO
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log level names and full ERROR messages"""

    LEVEL_COLORS = {
        "DEBUG": LogColors.GRAY,
        "INFO": LogColors.GREEN,
        "WARNING": LogColors.YELLOW,
        "ERROR": LogColors.RED,
        "CRITICAL": LogColors.RED,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if colors should be disabled via environment variable
        self.use_colors = os.environ.get("OPAL_NO_COLOR", "0") != "1"

    def format(self, record):
        # If colors are disabled, just format normally
        if not self.use_colors:
            return super().format(record)

        # Save the original levelname
        original_levelname = record.levelname

        # For ERROR and CRITICAL, color the entire message (don't color just levelname)
        if original_levelname in ("ERROR", "CRITICAL"):
            # Format the message without coloring levelname
            result = super().format(record)
            # Color the entire formatted message
            result = f"{LogColors.RED}{result}{LogColors.RESET}"
        else:
            # For other levels, only color the levelname
            color = self.LEVEL_COLORS.get(record.levelname, LogColors.RESET)
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            # Format the message
            result = super().format(record)
            # Restore original levelname
            record.levelname = original_levelname

        return result


class _CustomFilter(logging.Filter):
    def __init__(self, env):
        logging.Filter.__init__(self)
        self._env = env

    def filter(self, record):
        # Call the specific function for each log record
        record.custom_time = self._env.now
        return True  # Allow the log


formatter_2 = ColoredFormatter(
    "%(levelname)s [%(custom_time)8.3f s] %(name)s " "%(funcName)s (%(filename)s:%(lineno)d): %(message)s"
)

formatter_1 = ColoredFormatter("%(levelname)s [%(custom_time)7.2f s] %(name)s" "(%(filename)s:%(lineno)d): %(message)s")

formatter_0 = ColoredFormatter("%(levelname)s [%(custom_time)6.1f s]" "(%(filename)s:%(lineno)d): %(message)s")

formatter_default = ColoredFormatter("%(levelname)s" "(%(filename)s:%(lineno)d): %(message)s")


def setup_logging(env, log_level="DEBUG", log_file=None):
    if env:
        logging.getLogger("").handlers.clear()

    # Get formatter based on OPAL_LOG_FORMAT environment variable (default: 0)
    log_format = int(os.environ.get("OPAL_LOG_FORMAT", "0"))

    # Select formatter based on environment variable
    if log_format == 2:
        formatter = formatter_2
    elif log_format == 1:
        formatter = formatter_1
    else:  # default to 0
        formatter = formatter_0
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

    # Root logger with filter
    logger = logging.getLogger()
    # logger.addFilter(_CustomFilter(env))  # get simulation time
    logger.addHandler(console_handler)
    if log_file is not None:
        logger.addHandler(file_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # set the remaining log levels
    level_str = log_level.upper()
    level = getattr(logging, level_str, logging.INFO)  # fallback to INFO
    logger.setLevel(level)
    console_handler.setLevel(level)
    if log_file is not None:
        file_handler.setLevel(level)

    for handler in logging.root.handlers:
        handler.addFilter(_CustomFilter(env))


def reset_logging_formatter():
    """Reset all logging handlers to use formatter_default"""
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(formatter_default)
