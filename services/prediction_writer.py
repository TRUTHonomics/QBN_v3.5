"""
Prediction Writer Service voor QBN v2.
Schrijft inference resultaten naar de database (cache & hypertable).

DEPRECATED: Dit bestand is deprecated sinds QBN v3.2.
Alle inference output wordt nu geconsolideerd naar qbn.output_entry
via services/signal_writer.py (QBNOutputWriter).

De bayesian_predictions tabellen worden uitgefaseerd.
Dit bestand wordt behouden voor backward compatibility maar
wordt niet meer gebruikt in de inference_loop.

Zie: database/migrations/015_extend_output_entry_full_inference.sql
Zie: database/migrations/016_deprecate_bayesian_predictions.sql
"""

import warnings
warnings.warn(
    "prediction_writer.py is deprecated. Use signal_writer.QBNOutputWriter instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

import asyncpg
from asyncpg import Pool

from inference.trade_aligned_inference import DualInferenceResult

logger = logging.getLogger(__name__)

class PredictionWriter:
    """
    Dual-write service voor predictions:
    1. UPSERT naar qbn.bayesian_predictions_current (real-time cache)
    2. INSERT naar qbn.bayesian_predictions (historische hypertable)
    """
    
    def __init__(self, pool: Optional[Pool] = None):
        self.pool = pool
        # Mapping van state namen naar integer IDs (-3 tot +3)
        self.state_mapping = {
            'Strong_Bearish': -3,
            'Bearish': -2,
            'Slight_Bearish': -1,
            'Neutral': 0,
            'Slight_Bullish': 1,
            'Bullish': 2,
            'Strong_Bullish': 3
        }

    def set_pool(self, pool: Pool):
        """Set de connection pool."""
        self.pool = pool

    async def write_prediction(self, asset_id: int, result: DualInferenceResult):
        """
        Schrijf een inference resultaat naar de database.
        """
        if not self.pool:
            raise RuntimeError("Connection pool niet geïnitialiseerd in PredictionWriter")

        # Voorbereiden van data
        now = datetime.now(timezone.utc)
        
        # Predictions mappen naar ints
        p_1h = self.state_mapping.get(result.predictions.get('1h'), 0)
        p_4h = self.state_mapping.get(result.predictions.get('4h'), 0)
        p_1d = self.state_mapping.get(result.predictions.get('1d'), 0)
        
        # Distributies naar JSON
        d_1h = json.dumps(result.distributions.get('1h'))
        d_4h = json.dumps(result.distributions.get('4h'))
        d_1d = json.dumps(result.distributions.get('1d'))
        et_dist = json.dumps(result.entry_timing_distribution)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. UPSERT naar current cache
                await self._upsert_current(conn, asset_id, result.timestamp, result, p_1h, p_4h, p_1d, d_1h, d_4h, d_1d, et_dist)
                
                # 2. INSERT naar historical archive
                await self._insert_historical(conn, asset_id, result.timestamp, result, p_1h, p_4h, p_1d, d_1h, d_4h, d_1d, et_dist)

    async def _upsert_current(self, conn, asset_id, time, result, p_1h, p_4h, p_1d, d_1h, d_4h, d_1d, et_dist):
        query = """
        INSERT INTO qbn.bayesian_predictions_current (
            asset_id, time,
            prediction_1h, prediction_4h, prediction_1d,
            confidence_1h, confidence_4h, confidence_1d,
            distribution_1h, distribution_4h, distribution_1d,
            expected_atr_1h, expected_atr_4h, expected_atr_1d,
            regime, leading_composite, coincident_composite, confirming_composite,
            entry_timing_distribution,
            inference_time_ms, model_version, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, NOW()
        )
        ON CONFLICT (asset_id) DO UPDATE SET
            time = EXCLUDED.time,
            prediction_1h = EXCLUDED.prediction_1h,
            prediction_4h = EXCLUDED.prediction_4h,
            prediction_1d = EXCLUDED.prediction_1d,
            confidence_1h = EXCLUDED.confidence_1h,
            confidence_4h = EXCLUDED.confidence_4h,
            confidence_1d = EXCLUDED.confidence_1d,
            distribution_1h = EXCLUDED.distribution_1h,
            distribution_4h = EXCLUDED.distribution_4h,
            distribution_1d = EXCLUDED.distribution_1d,
            expected_atr_1h = EXCLUDED.expected_atr_1h,
            expected_atr_4h = EXCLUDED.expected_atr_4h,
            expected_atr_1d = EXCLUDED.expected_atr_1d,
            regime = EXCLUDED.regime,
            leading_composite = EXCLUDED.leading_composite,
            coincident_composite = EXCLUDED.coincident_composite,
            confirming_composite = EXCLUDED.confirming_composite,
            entry_timing_distribution = EXCLUDED.entry_timing_distribution,
            inference_time_ms = EXCLUDED.inference_time_ms,
            model_version = EXCLUDED.model_version,
            updated_at = NOW()
        """
        await conn.execute(query, 
            asset_id, time,
            p_1h, p_4h, p_1d,
            result.confidences.get('1h'), result.confidences.get('4h'), result.confidences.get('1d'),
            d_1h, d_4h, d_1d,
            result.expected_atr_moves.get('1h'), result.expected_atr_moves.get('4h'), result.expected_atr_moves.get('1d'),
            result.regime, result.leading_composite, result.coincident_composite, result.confirming_composite,
            et_dist,
            result.inference_time_ms, result.model_version
        )

    async def _insert_historical(self, conn, asset_id, time, result, p_1h, p_4h, p_1d, d_1h, d_4h, d_1d, et_dist):
        query = """
        INSERT INTO qbn.bayesian_predictions (
            asset_id, time,
            prediction_1h, prediction_4h, prediction_1d,
            confidence_1h, confidence_4h, confidence_1d,
            distribution_1h, distribution_4h, distribution_1d,
            expected_atr_1h, expected_atr_4h, expected_atr_1d,
            regime, leading_composite, coincident_composite, confirming_composite,
            entry_timing_distribution,
            inference_time_ms, model_version
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
        )
        ON CONFLICT (asset_id, time) DO NOTHING
        """
        await conn.execute(query, 
            asset_id, time,
            p_1h, p_4h, p_1d,
            result.confidences.get('1h'), result.confidences.get('4h'), result.confidences.get('1d'),
            d_1h, d_4h, d_1d,
            result.expected_atr_moves.get('1h'), result.expected_atr_moves.get('4h'), result.expected_atr_moves.get('1d'),
            result.regime, result.leading_composite, result.coincident_composite, result.confirming_composite,
            et_dist,
            result.inference_time_ms, result.model_version
        )

