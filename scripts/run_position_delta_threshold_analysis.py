#!/usr/bin/env python3
"""
Position Delta Threshold Analysis CLI

MI Grid Search voor optimalisatie van delta thresholds in Position Management.
Delta scores meten verandering in coincident/confirming composites sinds trade entry.

USAGE:
    python scripts/run_position_delta_threshold_analysis.py --asset-id 1
    python scripts/run_position_delta_threshold_analysis.py --asset-id 9889 --lookback 90

VEREISTEN:
    - EventWindowDetector moet al gedraaid zijn (events in qbn.event_windows)
    - barrier_outcomes moet gevuld zijn met event_id labels
    - coincident_score en confirming_score moeten beschikbaar zijn

OUTPUT:
    - Optimale thresholds in qbn.position_delta_threshold_config
    - MI heatmaps in _validation/position_delta_analysis/
    - Markdown report
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Voeg project root toe aan path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging_utils import setup_logging
from analysis.position_delta_optimizer import PositionDeltaThresholdOptimizer
from core.step_validation import validate_step_input, log_handshake_out, StepValidationError
from database.db import get_cursor

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Position Delta Threshold Analysis - MI Grid Search"
    )
    parser.add_argument(
        "--asset-id", 
        type=int, 
        required=True,
        help="Asset ID waarvoor thresholds worden geoptimaliseerd"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Aantal dagen terug voor data (default: alle beschikbare event data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory voor reports/heatmaps"
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Schakel diversity constraints uit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alleen analyseren, niet opslaan in database"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for traceability"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging("run_position_delta_threshold_analysis")
    
    lookback_desc = f"{args.lookback} days" if args.lookback else "all data"
    
    logger.info("=" * 60)
    logger.info("POSITION DELTA THRESHOLD ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Asset ID: {args.asset_id}")
    logger.info(f"Run ID: {args.run_id or 'N/A'}")
    logger.info(f"Lookback: {lookback_desc}")
    logger.info(f"Diversity constraints: {'disabled' if args.no_diversity else 'enabled'}")
    logger.info("=" * 60)
    
    try:
        # Initialiseer optimizer
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        optimizer = PositionDeltaThresholdOptimizer(
            asset_id=args.asset_id,
            lookback_days=args.lookback,  # None = all data
            output_dir=output_dir,
            enforce_diversity=not args.no_diversity
        )
        
        # Laad event data
        logger.info("\n📊 Loading event data...")
        event_data = optimizer.load_event_data()
        
        if event_data.empty:
            logger.error("❌ No event data found. Run EventWindowDetector first.")
            logger.error("   Hint: Draai eerst 'Full Training Run' in het training menu")
            sys.exit(1)
        
        # Validation guard: check upstream event_windows
        if args.run_id:
            try:
                with get_cursor() as cur:
                    validate_step_input(
                        conn=cur.connection,
                        step_name="position_delta_threshold_analysis",
                        upstream_table="qbn.event_windows",
                        asset_id=args.asset_id,
                        run_id=args.run_id,
                        min_rows=10
                    )
            except StepValidationError as e:
                logger.info(f"Upstream validation note: {e}")
            except Exception as e:
                logger.warning(f"Upstream validation failed (DB issue): {e}")
        
        # Voer analyse uit
        logger.info("\n🔬 Running MI Grid Search...")
        results = optimizer.analyze()
        
        if not results:
            logger.error("❌ No valid results from analysis")
            sys.exit(1)
        
        # Log resultaten
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMAL THRESHOLDS")
        logger.info("=" * 60)
        
        for key, result in results.items():
            logger.info(f"\n{key}:")
            logger.info(f"  Threshold: {result.optimal_threshold:.4f}")
            logger.info(f"  MI Score: {result.mi_score:.4f}")
            dist_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(result.distribution.items()))
            logger.info(f"  Distribution: {dist_str}")
        
        # Opslaan
        if not args.dry_run:
            logger.info("\n💾 Saving results to database...")
            optimizer.save_results(results, run_id=args.run_id)
            logger.info("✅ Results saved successfully")
        else:
            logger.info("\n⚠️ Dry run - results NOT saved to database")
        
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
