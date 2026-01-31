#!/usr/bin/env python3
"""
CLI entrypoint voor de QBN v2 Walk-Forward Validation.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation.walk_forward_validator import WalkForwardValidator
from validation.schema_validator import SchemaValidator
from core.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='QBN v2 Walk-Forward Validation CLI')
    parser.add_argument('--asset-id', type=int, default=1, help='Asset ID om te valideren')
    parser.add_argument('--days', type=int, default=30, help='Aantal dagen voor de validatie periode')
    parser.add_argument('--train-window', type=int, default=90, help='Trainingsvenster in dagen')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    logger = setup_logging("walk_forward")

    # 1. Schema Validatie
    schema_v = SchemaValidator()
    if not schema_v.validate_v2_alignment(args.asset_id):
        logger.error("❌ Schema validatie mislukt. Stop validatie.")
        sys.exit(1)

    # 2. Run Validation
    output_dir = Path(args.output_dir) if args.output_dir else None
    validator = WalkForwardValidator(args.asset_id, output_dir=output_dir)
    
    # De validatie periode eindigt nu
    end_date = datetime.now()
    # De validatie periode begint 'days' geleden
    # Maar we hebben 'train_window' aan data VOOR de start nodig
    start_date = end_date - timedelta(days=args.days)
    # We corrigeren de start_date zodat run_validation genoeg historie heeft voor het eerste window
    history_start = start_date - timedelta(days=args.train_window)
    
    validator.run_validation(
        start_date=history_start,
        end_date=end_date,
        train_window_days=args.train_window,
        test_step_days=7 # Wekelijkse stappen
    )

    # 3. Toon Resultaten
    print("\n" + validator.summary_report())

if __name__ == '__main__':
    main()

