import logging

from rich.console import Console
from rich.logging import RichHandler


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(console=Console(width=150))]
    )
    logger = logging.getLogger("rich")
    return logger
