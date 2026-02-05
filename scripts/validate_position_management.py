#!/usr/bin/env python3
"""
Validation Script for Position Management (v3.1).
Replays Position Confidence & Prediction logic on historical Event Windows.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from inference.cpt_cache_manager import CPTCacheManager
from inference.position_confidence_generator import PositionConfidenceGenerator
from inference.position_prediction_generator import PositionPredictionGenerator
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
from inference.event_window_detector import EventWindowDetector, EventWindowConfig
from menus.training_menu import _get_optimal_leading_thresholds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_generators(asset_id, run_id=None):
    """Load generators with CPTs from cache."""
    cache = CPTCacheManager()
    
    # Load Position Confidence
    # REASON: CPTCacheManager.get_cpt() ondersteunt geen run_id argument, 
    # maar pakt automatisch de LAATSTE via generated_at DESC.
    conf_cpt = cache.get_cpt(asset_id, "Position_Confidence")
    if not conf_cpt:
        logger.error(f"No Position_Confidence CPT found for asset {asset_id}")
        return None, None
    
    conf_gen = PositionConfidenceGenerator()
    parsed_cpt = {}
    for k, v in conf_cpt.get('conditional_probabilities', {}).items():
        parts = tuple(k.split('|'))
        if len(parts) == 3:
            parsed_cpt[parts] = v
    conf_gen._cpt = parsed_cpt
    
    # Load Position Prediction
    pred_cpt = cache.get_cpt(asset_id, "Position_Prediction")
    if not pred_cpt:
        logger.error(f"No Position_Prediction CPT found for asset {asset_id}")
        return conf_gen, None
        
    pred_gen = PositionPredictionGenerator()
    parsed_pred_cpt = {}
    for k, v in pred_cpt.get('conditional_probabilities', {}).items():
        parts = tuple(k.split('|'))
        if len(parts) == 3:
            parsed_pred_cpt[parts] = v
    pred_gen._cpt = parsed_pred_cpt
    
    return conf_gen, pred_gen

def validate_asset(asset_id, run_id=None, output_dir=None):
    logger.info(f"Starting validation for Asset {asset_id}...")
    
    # 1. Load Generators
    conf_gen, pred_gen = load_generators(asset_id, run_id)
    if not conf_gen or not pred_gen:
        logger.error("Could not load generators. Aborting.")
        return

    # 2. Fetch Data & Detect Events
    logger.info("Fetching data and detecting events...")
    cpt_gen = QBNv3CPTGenerator()
    # Use full history to capture all events
    data = cpt_gen._prepare_merged_dataset([asset_id], lookback_days=None)
    
    if data.empty:
        logger.warning("No data found.")
        return

    # Get thresholds
    opt_strong, opt_delta = _get_optimal_leading_thresholds(asset_id)
    config = EventWindowConfig(
        absolute_threshold=opt_strong,
        delta_threshold=opt_delta,
        max_window_minutes=1440
    )
    
    detector = EventWindowDetector(config)
    events, labeled_data = detector.detect_events(data, asset_id)
    
    # Filter for in-event rows
    event_rows = labeled_data[labeled_data['event_id'].notna()].copy()
    
    if event_rows.empty:
        logger.warning("No events detected.")
        return
        
    logger.info(f"Replaying inference on {len(event_rows)} event rows...")
    
    results = []
    
    for _, row in event_rows.iterrows():
        # Prepare inputs
        coinc = row.get('coincident_composite', 'neutral')
        conf = row.get('confirming_composite', 'neutral')
        time_min = row.get('time_since_entry', 0) # EventWindowDetector adds this
        if pd.isna(time_min): time_min = 0
        
        # 1. Inference: Position Confidence
        conf_state, conf_score, _ = conf_gen.get_confidence(
            coinc, 
            conf, 
            int(time_min)
        )
        
        # 2. Inference: Position Prediction
        # Placeholder for PnL
        pnl = row.get('current_pnl_atr', 0.0) 
        
        pred_result = pred_gen.predict(
            conf_state,
            int(time_min),
            pnl
        )
        
        # 3. Ground Truth
        actual_outcome = row.get('event_outcome', 'neutral')
        if actual_outcome in ['neutral', 'timeout', 'none']:
            truth = 'timeout'
        else:
            is_up = str(actual_outcome).startswith('up_')
            is_long = row.get('event_direction') == 'long'
            aligned = (is_long and is_up) or (not is_long and not is_up)
            truth = 'target_hit' if aligned else 'stoploss_hit'
            
        results.append({
            'event_id': row['event_id'],
            'time_min': time_min,
            'predicted': pred_result.dominant_outcome,
            'confidence': pred_result.confidence,
            'actual': truth,
            'correct': pred_result.dominant_outcome == truth
        })
        
    # Metrics
    res_df = pd.DataFrame(results)
    accuracy = res_df['correct'].mean()
    
    # Prepare Report Output
    report_lines = []
    report_lines.append(f"📊 Validation Results Asset {asset_id}:")
    report_lines.append(f"   Total Rows: {len(res_df)}")
    report_lines.append(f"   Accuracy: {accuracy:.1%} ({res_df['correct'].sum()}/{len(res_df)})")
    
    # Group by Prediction
    report_lines.append(f"\n   Breakdown by Prediction:")
    breakdown = res_df.groupby('predicted')['correct'].agg(['mean', 'count'])
    breakdown['mean'] = breakdown['mean'].map(lambda x: f"{x:.1%}")
    report_lines.append(str(breakdown))
    
    # Time bucket accuracy
    res_df['time_bucket'] = pd.cut(res_df['time_min'], bins=[-1, 60, 240, 720, 1440], labels=['0-1h', '1-4h', '4-12h', '12-24h'])
    report_lines.append(f"\n   Accuracy by Time Bucket:")
    time_acc = res_df.groupby('time_bucket', observed=True)['correct'].agg(['mean', 'count'])
    time_acc['mean'] = time_acc['mean'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "0.0%")
    report_lines.append(str(time_acc))

    # State machine check: entry -> hold -> exit flow per event
    report_lines.append(f"\n   State Machine Check (entry -> hold -> exit):")
    events = event_rows.groupby('event_id')
    n_events = len(events)
    n_with_hold = sum(1 for _, g in events if len(g) >= 2)  # at least entry + 1 hold row
    n_with_exit = sum(1 for _, g in events if g['time_since_entry'].max() >= 0)  # has time progression
    report_lines.append(f"   Events with valid hold phase: {n_with_hold}/{n_events}")
    report_lines.append(f"   Events with time progression (exit path): {n_with_exit}/{n_events}")

    # Console output
    for line in report_lines:
        print(line)

    # Markdown export if output_dir provided
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = out_path / f"pos_accuracy_report_{ts}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Position Management Accuracy Report\n\n")
            f.write(f"**Asset:** {asset_id}\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
            f.write("```text\n")
            f.write("\n".join(report_lines))
            f.write("\n```\n")
        logger.info(f"💾 Report saved to {report_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-id', type=int, required=True)
    parser.add_argument('--run-id', type=str, help='Specific CPT run ID')
    parser.add_argument('--output-dir', type=str, help='Directory to save MD report')
    args = parser.parse_args()
    
    validate_asset(args.asset_id, args.run_id, args.output_dir)
