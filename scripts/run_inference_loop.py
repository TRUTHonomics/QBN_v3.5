#!/usr/bin/env python3
"""
QBN Real-time Inference Loop Runner.
Start de asynchrone inference loop en watchdog.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.inference_loop import InferenceLoopService
from services.watchdog import InferenceLoopWatchdog
from database.db import get_cursor
from core.logging_utils import setup_logging

logger = setup_logging("inference_loop")

def get_active_assets():
    """Haal assets op die gemarkeerd zijn voor verwerking."""
    with get_cursor() as cur:
        cur.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1 ORDER BY id")
        return [row[0] for row in cur.fetchall()]

async def main():
    # 1. Bepaal assets
    asset_ids = get_active_assets()
    if not asset_ids:
        logger.error("❌ Geen actieve assets gevonden in symbols.symbols. Gebruik symbols.selected_in_current_run = 1")
        return

    # 2. Database config voor asyncpg
    # REASON: Gebruik environment variabelen (consistent met database/db.py)
    db_config = {
        'host': os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', '10.10.10.3')),
        'port': int(os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))),
        'database': os.getenv('POSTGRES_DB', os.getenv('DB_NAME', 'kflhyper')),
        'user': os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'qbn')),
        'password': os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASS', '1234'))
    }

    # 3. Initialiseer services
    service = InferenceLoopService(asset_ids, db_config)
    watchdog = InferenceLoopWatchdog(service)

    # 4. Start
    logger.info("🎬 Starten van QBN Real-time Inference Pipeline...")
    
    try:
        # We gebruiken gather om zowel de service als de watchdog te draaien
        await asyncio.gather(
            service.start(),
            watchdog.start()
        )
    except KeyboardInterrupt:
        logger.info("👋 Afsluiten door gebruiker...")
    except Exception as e:
        logger.critical(f"💥 Kritieke fout in main loop: {e}", exc_info=True)
    finally:
        await service.stop()
        await watchdog.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

