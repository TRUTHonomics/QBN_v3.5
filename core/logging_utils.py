"""
Gedeelde logging utility volgens project logregels (KFL/QBN SKILL kfl-logging).

Logregels:
1. Locatie: _log/ in project root
2. Bestandsnaam: {YYMMDD-HH-MM-ss}_{scriptname}.log (timestamp eerst)
3. Archivering: *_{scriptname}.log van _log/ naar _log/archive/
4. Format: timestamp, level, message (komma; datefmt YYYY-MM-DD HH:MM:SS)
5. Console: bij TTY WARNING bold oranje, ERROR bold rood (ANSI)
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_FMT = "%(asctime)s, %(levelname)s, %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

_BOLD_ORANGE = "\033[1;38;5;208m"
_BOLD_RED = "\033[1;31m"
_RESET = "\033[0m"


class _ConsoleFormatter(logging.Formatter):
    """Formatter that adds ANSI colors for WARNING/ERROR when used on console."""

    def format(self, record):
        record.levelname = record.levelname.upper()
        base = super().format(record)
        level_colors = {"WARNING": _BOLD_ORANGE, "ERROR": _BOLD_RED}
        color = level_colors.get(record.levelname, "")
        if color:
            base = base.replace(record.levelname, f"{color}{record.levelname}{_RESET}")
        return base


def setup_logging(
    script_name: str,
    log_level: int = logging.INFO,
    project_root: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup logging volgens KFL/QBN logregels.

    Args:
        script_name: Naam voor het log bestand (zonder extensie)
        log_level: Logging level (default: INFO)
        project_root: Optioneel; anders PROJECT_ROOT (QBN repo root)

    Returns:
        Geconfigureerde logger instance
    """
    root = project_root if project_root is not None else PROJECT_ROOT
    log_dir = root / "_log"
    archive_dir = log_dir / "archive"

    log_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Archiveer bestaande logs van dit script (*_scriptname.log)
    for old_log in log_dir.glob(f"*_{script_name}.log"):
        try:
            if old_log.is_file():
                shutil.move(str(old_log), str(archive_dir / old_log.name))
        except Exception:
            pass

    ts = datetime.now().strftime("%y%m%d-%H-%M-%S")
    log_file = log_dir / f"{ts}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
    logger.addHandler(file_handler)

    if sys.stderr.isatty():
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(_ConsoleFormatter(FILE_FMT, datefmt=DATE_FMT))
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
        logger.addHandler(console_handler)

    logger.info(f"Started. Logging to: {log_file}")

    def exception_handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = exception_handler

    return logger
