"""
Gedeelde logging utility volgens project logregels.

Logregels:
1. Log bestandsnaam: scriptnaam_YYMMDD-HH-MM-ss.log (2-cijferig jaar)
2. Locatie: _log map in project root
3. Maak _log aan als deze niet bestaat
4. Archiveer oude logs van hetzelfde script naar _log/archive
5. Maak _log/archive aan als deze niet bestaat
6. Log regel format: timestamp, level, bericht
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(
    script_name: str,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging volgens project logregels.

    Args:
        script_name: Naam voor het log bestand (zonder extensie)
        log_level: Logging level (default: INFO)

    Returns:
        Geconfigureerde logger instance
    """
    log_dir = PROJECT_ROOT / "_log"
    archive_dir = log_dir / "archive"

    # Regel 3 & 5: Maak directories aan als ze niet bestaan
    log_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Regel 4: Archiveer bestaande oude logs van dit script
    for old_log in log_dir.glob(f"{script_name}_*.log"):
        try:
            shutil.move(str(old_log), str(archive_dir / old_log.name))
        except Exception:
            pass  # Negeer archive fouten

    # Regel 1: Correct timestamp format YYMMDD-HH-MM-ss (2-cijferig jaar!)
    timestamp = datetime.now().strftime("%y%m%d-%H-%M-%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    # Regel 6: Log format - timestamp, level, bericht
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Clear bestaande handlers en configureer
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True
    )

    logger = logging.getLogger(script_name)
    # REASON: Vermijd Unicode/emoji issues op Windows consoles (cp1252).
    logger.info(f"New {script_name} run started. Logging to: {log_file}")

    # REASON: Vang uncaught exceptions op en log ze naar het log bestand
    # Dit voorkomt dat tracebacks alleen naar stderr gaan en niet in de logs terechtkomen
    def exception_handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            # Laat KeyboardInterrupt door naar de default handler
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
    
    sys.excepthook = exception_handler

    return logger
