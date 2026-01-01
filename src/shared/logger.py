import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class Logger:
    """
    Singleton logger with colored console output and rotating file logs.
    """
    _instance = None

    # Mapping of log levels to colors for console output
    COLOR_MAP = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA
    }

    def __new__(cls, log_name="app.log", log_dir="logs", level=logging.INFO):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)

            # Ensure log directory exists
            cls._instance.log_dir = Path(log_dir)
            cls._instance.log_dir.mkdir(parents=True, exist_ok=True)

            # Create logger
            cls._instance.logger = logging.getLogger("AppLogger")
            cls._instance.logger.setLevel(level)
            cls._instance.logger.propagate = False

            # File formatter (plain)
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Rotating file handler
            file_handler = RotatingFileHandler(
                cls._instance.log_dir / log_name,
                maxBytes=5*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            cls._instance.logger.addHandler(file_handler)

            # Console handler with colored output
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(cls.ColoredFormatter(file_formatter))
            cls._instance.logger.addHandler(console_handler)

        return cls._instance

    def get_logger(self):
        return self.logger

    # Custom formatter class for console logs
    class ColoredFormatter(logging.Formatter):
        def __init__(self, base_formatter):
            super().__init__()
            self.base_formatter = base_formatter

        def format(self, record):
            color = Logger.COLOR_MAP.get(record.levelname, "")
            message = self.base_formatter.format(record)
            return f"{color}{message}{Style.RESET_ALL}"
