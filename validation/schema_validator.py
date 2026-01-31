"""
Schema Validator voor QBN v2 Walk-Forward Validation.

Valideert of de historische data in de split-tables consistent is met de v2 eisen
voor succesvolle inference en training.
"""

import logging
from typing import Dict, List, Tuple
from database.db import get_cursor

logger = logging.getLogger(__name__)

class SchemaValidator:
    """Valideert of historische MTF data aligned is met de v2 requirements."""

    def __init__(self):
        self.required_tables = [
            'kfl.mtf_signals_lead',
            'kfl.mtf_signals_coin',
            'kfl.mtf_signals_conf',
            'qbn.barrier_outcomes'
        ]

    def validate_v2_alignment(self, asset_id: int) -> bool:
        """
        Voer alle validatie-checks uit.
        """
        logger.info(f"Start v2 schema validatie voor asset {asset_id}...")
        
        checks = [
            self._check_tables_exist(),
            self._check_column_consistency(),
            self._check_data_availability(asset_id)
        ]
        
        success = all(checks)
        if success:
            logger.info("✅ Schema validatie PASSED")
        else:
            logger.error("❌ Schema validatie FAILED")
            
        return success

    def _check_tables_exist(self) -> bool:
        """Check of alle benodigde tabellen bestaan."""
        with get_cursor() as cur:
            for table in self.required_tables:
                schema, name = table.split('.')
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = %s AND table_name = %s
                    )
                """, (schema, name))
                if not cur.fetchone()[0]:
                    logger.error(f"Tabel {table} ontbreekt in de database.")
                    return False
        return True

    def _check_column_consistency(self) -> bool:
        """Check of de vereiste kolommen aanwezig zijn."""
        # REASON: Controleer of de 'layer' kolom aanwezig is in signal_weights (v3.3 requirement)
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_schema = 'qbn' AND table_name = 'signal_weights' AND column_name = 'layer'
                    )
                """)
                if not cur.fetchone()[0]:
                    logger.error("Kolom 'layer' ontbreekt in qbn.signal_weights.")
                    return False
            return True
        except Exception as e:
            logger.error(f"Fout bij check column consistency: {e}")
            return False

    def _check_data_availability(self, asset_id: int) -> bool:
        """Check of er data aanwezig is voor het opgegeven asset."""
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM kfl.mtf_signals_lead WHERE asset_id = %s", (asset_id,))
            count = cur.fetchone()[0]
            if count == 0:
                logger.warning(f"Geen data gevonden in kfl.mtf_signals_lead voor asset {asset_id}")
                return False
        return True

