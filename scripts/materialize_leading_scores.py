#!/usr/bin/env python3
"""
materialize_leading_scores.py - Berekent en materialiseert leading_score in barrier_outcomes

Dit script berekent de leading_score voor elke rij in qbn.barrier_outcomes
door de signalen uit kfl.mtf_signals_lead te aggregeren volgens de
signal_classification configuratie.

KRITISCH: Alleen data gebruiken die beschikbaar was op time_1 (geen look-ahead bias).

Usage:
    python scripts/materialize_leading_scores.py --asset-id 1
    python scripts/materialize_leading_scores.py --asset-id 1 --batch-size 50000
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import glob
import shutil
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from config.ida_config import IDAConfig
from core.logging_utils import setup_logging

logger = setup_logging("materialize_leading_scores")


class LeadingScoreMaterializer:
    """
    Materialiseert leading_score in qbn.barrier_outcomes.
    
    De leading_score is een gewogen som van alle LEADING signalen,
    genormaliseerd naar [-1, +1].
    """
    
    # Polarity mapping
    POLARITY_MAP = {
        'bullish': 1,
        'bearish': -1,
        'neutral': 0,
    }
    
    def __init__(self, asset_id: int, batch_size: int = 50000, run_id: str = None):
        self.asset_id = asset_id
        self.batch_size = batch_size
        self.run_id = run_id
        self.config = IDAConfig()
        
        # Laad signal classification
        self.leading_signals = self._load_leading_signals()
        logger.info(f"Loaded {len(self.leading_signals)} LEADING signals")
    
    def _load_leading_signals(self) -> List[Dict]:
        """Laad alle LEADING signalen uit qbn.signal_classification."""
        query = """
            SELECT signal_name, polarity
            FROM qbn.signal_classification
            WHERE semantic_class = 'LEADING'
            ORDER BY signal_name
        """
        
        with get_cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        
        signals = []
        for row in rows:
            signal_name = row[0].lower()  # Lowercase voor column matching
            polarity_str = row[1].lower() if row[1] else 'neutral'
            polarity_val = self.POLARITY_MAP.get(polarity_str, 0)
            
            signals.append({
                'signal_name': signal_name,
                'polarity': polarity_val,
            })
        
        return signals
    
    def _get_signal_columns(self) -> List[str]:
        """
        Bepaal welke signaal kolommen we nodig hebben uit mtf_signals_lead.
        
        We gebruiken de 60-minuten variant (_60 suffix) omdat barrier_outcomes
        gebaseerd is op 60-minuten klines.
        """
        columns = []
        for sig in self.leading_signals:
            # Signaal kolom met _60 suffix
            col_name = f"{sig['signal_name']}_60"
            columns.append(col_name)
        return columns
    
    def _build_score_query(self) -> str:
        """
        Bouw de SQL query voor score berekening.
        
        KRITISCH: Deze query mag GEEN data gebruiken uit qbn tabellen
        (behalve barrier_outcomes voor de UPDATE).
        """
        # Bouw CASE expressions voor elke signal
        case_parts = []
        for sig in self.leading_signals:
            col_name = f"{sig['signal_name']}_60"
            polarity = sig['polarity']
            
            if polarity != 0:
                # COALESCE om NULL te behandelen als 0
                case_parts.append(f"COALESCE({col_name}, 0) * {polarity}")
        
        if not case_parts:
            raise ValueError("Geen LEADING signalen met polarity != 0 gevonden")
        
        # Som van gewogen signalen
        score_expr = " + ".join(case_parts)
        n_signals = len([s for s in self.leading_signals if s['polarity'] != 0])
        
        # Normaliseer naar [-1, +1] door te delen door aantal signalen
        # en te clippen
        normalized_expr = f"GREATEST(-1.0, LEAST(1.0, ({score_expr})::real / {n_signals}))"
        
        return normalized_expr
    
    def get_pending_count(self) -> int:
        """Tel hoeveel rijen nog geen leading_score hebben."""
        query = """
            SELECT COUNT(*) 
            FROM qbn.barrier_outcomes
            WHERE asset_id = %s
              AND leading_score IS NULL
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id,))
            return cur.fetchone()[0]
    
    def materialize(self, overwrite: bool = False) -> int:
        """
        Materialiseer leading_score voor alle barrier_outcomes.
        
        Args:
            overwrite: Als True, herbereken ook bestaande scores
            
        Returns:
            Aantal geüpdatete rijen
        """
        # Valideer query tegen data leakage
        score_expr = self._build_score_query()
        
        # Basis query - JOIN mtf_signals_lead op barrier_outcomes
        base_condition = "bo.asset_id = %s"
        if not overwrite:
            base_condition += " AND bo.leading_score IS NULL"
        
        # Tel totaal
        count_query = f"""
            SELECT COUNT(*)
            FROM qbn.barrier_outcomes bo
            WHERE {base_condition}
        """
        
        with get_cursor() as cur:
            cur.execute(count_query, (self.asset_id,))
            total = cur.fetchone()[0]
        
        if total == 0:
            logger.info("Geen rijen om te updaten")
            return 0
        
        logger.info(f"Materializing leading_score voor {total:,} rijen...")
        
        # Update query met JOIN naar mtf_signals_lead
        # KRITISCH: We joinen op asset_id en time_1, wat correct is
        # omdat time_1 de signaal-timestamp is
        update_query = f"""
            UPDATE qbn.barrier_outcomes bo
            SET leading_score = subq.score,
                run_id = %s,
                updated_at = NOW()
            FROM (
                SELECT 
                    mtf.asset_id,
                    mtf.time_1,
                    {score_expr} as score
                FROM kfl.mtf_signals_lead mtf
                WHERE mtf.asset_id = %s
            ) subq
            WHERE bo.asset_id = subq.asset_id
              AND bo.time_1 = subq.time_1
              AND bo.asset_id = %s
        """
        
        if not overwrite:
            update_query += " AND bo.leading_score IS NULL"
        
        # Voer update uit in batches
        updated = 0
        
        # Voor grote datasets, gebruik een batch-approach
        if total > self.batch_size:
            # Haal timestamps op in batches
            timestamps_query = f"""
                SELECT bo.time_1
                FROM qbn.barrier_outcomes bo
                WHERE {base_condition}
                ORDER BY bo.time_1
            """
            
            with get_cursor() as cur:
                cur.execute(timestamps_query, (self.asset_id,))
                all_timestamps = [row[0] for row in cur.fetchall()]
            
            # Process in batches
            for i in tqdm(range(0, len(all_timestamps), self.batch_size), desc="Batches"):
                batch_timestamps = all_timestamps[i:i + self.batch_size]
                min_ts = min(batch_timestamps)
                max_ts = max(batch_timestamps)
                
                batch_query = f"""
                    UPDATE qbn.barrier_outcomes bo
                    SET leading_score = subq.score,
                        run_id = %s,
                        updated_at = NOW()
                    FROM (
                        SELECT 
                            mtf.asset_id,
                            mtf.time_1,
                            {score_expr} as score
                        FROM kfl.mtf_signals_lead mtf
                        WHERE mtf.asset_id = %s
                          AND mtf.time_1 >= %s
                          AND mtf.time_1 <= %s
                    ) subq
                    WHERE bo.asset_id = subq.asset_id
                      AND bo.time_1 = subq.time_1
                      AND bo.asset_id = %s
                      AND bo.time_1 >= %s
                      AND bo.time_1 <= %s
                """
                
                if not overwrite:
                    batch_query += " AND bo.leading_score IS NULL"
                
                with get_cursor(commit=True) as cur:
                    cur.execute(batch_query, (
                        self.run_id,
                        self.asset_id, min_ts, max_ts,
                        self.asset_id, min_ts, max_ts
                    ))
                    updated += cur.rowcount
        else:
            # Kleine dataset - update in één keer
            with get_cursor(commit=True) as cur:
                cur.execute(update_query, (self.run_id, self.asset_id, self.asset_id))
                updated = cur.rowcount
        
        logger.info(f"Updated {updated:,} rijen met leading_score")
        return updated
    
    def validate_scores(self) -> Dict:
        """Valideer de gematerialiseerde scores."""
        query = """
            SELECT 
                COUNT(*) as total,
                COUNT(leading_score) as with_score,
                AVG(leading_score) as mean_score,
                STDDEV(leading_score) as std_score,
                MIN(leading_score) as min_score,
                MAX(leading_score) as max_score,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY leading_score) as median_score
            FROM qbn.barrier_outcomes
            WHERE asset_id = %s
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id,))
            row = cur.fetchone()
        
        result = {
            'total': row[0],
            'with_score': row[1],
            'coverage': row[1] / row[0] if row[0] > 0 else 0,
            'mean': float(row[2]) if row[2] else None,
            'std': float(row[3]) if row[3] else None,
            'min': float(row[4]) if row[4] else None,
            'max': float(row[5]) if row[5] else None,
            'median': float(row[6]) if row[6] else None,
        }
        
        # Sanity checks
        warnings = []
        if result['coverage'] < 1.0:
            warnings.append(f"Coverage: {result['coverage']:.1%} (some rows missing scores)")
        if result['min'] is not None and result['min'] < -1.0:
            warnings.append(f"Min score {result['min']:.4f} < -1.0")
        if result['max'] is not None and result['max'] > 1.0:
            warnings.append(f"Max score {result['max']:.4f} > 1.0")
        
        result['warnings'] = warnings
        return result


def main():
    parser = argparse.ArgumentParser(description='Materialize leading_score')
    parser.add_argument('--asset-id', type=int, required=True, help='Asset ID')
    parser.add_argument('--batch-size', type=int, default=50000, help='Batch size')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing scores')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, no update')
    parser.add_argument('--run-id', type=str, help='Run identifier for traceability')
    
    args = parser.parse_args()
    
    logger.info(f"Starting leading_score materialization for asset {args.asset_id} (run_id: {args.run_id})")
    
    materializer = LeadingScoreMaterializer(
        asset_id=args.asset_id,
        batch_size=args.batch_size,
        run_id=args.run_id
    )
    
    if args.validate_only:
        result = materializer.validate_scores()
        print("\n📊 Validation Results:")
        print(f"   Total rows:  {result['total']:,}")
        print(f"   With score:  {result['with_score']:,} ({result['coverage']:.1%})")
        if result['mean'] is not None:
            print(f"   Mean:        {result['mean']:.4f}")
            print(f"   Std:         {result['std']:.4f}")
            print(f"   Range:       [{result['min']:.4f}, {result['max']:.4f}]")
            print(f"   Median:      {result['median']:.4f}")
        
        if result['warnings']:
            print("\n⚠️ Warnings:")
            for w in result['warnings']:
                print(f"   - {w}")
        return
    
    # Materialiseer
    updated = materializer.materialize(overwrite=args.overwrite)
    
    # Valideer
    result = materializer.validate_scores()
    
    print("\n✅ Materialization Complete")
    print(f"   Updated:     {updated:,} rows")
    print(f"   Coverage:    {result['coverage']:.1%}")
    if result['mean'] is not None:
        print(f"   Score range: [{result['min']:.4f}, {result['max']:.4f}]")
    
    if result['warnings']:
        print("\n⚠️ Warnings:")
        for w in result['warnings']:
            print(f"   - {w}")


if __name__ == '__main__':
    main()
