"""
Prediction Accuracy Validator - Walk-forward validatie van Entry en Position predictions.

Vergelijkt CPT-predictions met werkelijke barrier outcomes om:
- Entry prediction accuracy per horizon (1h/4h/1d) en confidence-niveau te meten
- Position prediction accuracy te evalueren
- Resultaten te rapporteren voor productie-readiness checks

Output: _validation/prediction_accuracy/accuracy_report_*.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_cursor
from core.logging_utils import setup_logging
from core.output_manager import ValidationOutputManager

logger = setup_logging("prediction_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valideer prediction accuracy via walk-forward test"
    )
    parser.add_argument("--asset-id", type=int, required=True, help="Asset ID")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Lookback period voor accuracy check (default: 30)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum aantal samples per check (default: 50)"
    )
    return parser.parse_args()


def fetch_entry_predictions(asset_id: int, lookback_days: int) -> List[Dict]:
    """
    Haal entry predictions op uit qbn.output_entry voor recente periode.
    
    REASON: We gebruiken qbn.output_entry als "ground truth" voor wat de
    inference zou hebben voorspeld op elk moment.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                time,
                trade_hypothesis,
                entry_confidence,
                entry_confidence_score,
                prediction_1h,
                prediction_4h,
                prediction_1d,
                leading_composite
            FROM qbn.output_entry
            WHERE asset_id = %s
              AND time >= NOW() - INTERVAL '%s days'
            ORDER BY time DESC
            LIMIT 10000
        """, (asset_id, lookback_days))
        
        rows = cur.fetchall()
        
    if not rows:
        return []
    
    return [
        {
            'time': row[0],
            'hypothesis': row[1],
            'confidence': row[2],
            'confidence_score': row[3],
            'pred_1h': row[4],
            'pred_4h': row[5],
            'pred_1d': row[6],
            'leading_composite': row[7],
        }
        for row in rows
    ]


def fetch_barrier_outcomes(asset_id: int, lookback_days: int) -> Dict[datetime, Dict]:
    """
    Haal barrier outcomes op voor dezelfde periode.
    
    Returns: Dict[time_1, outcome_dict] voor snelle lookup.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                time_1,
                first_significant_barrier,
                time_to_up_050_atr,
                time_to_down_050_atr,
                time_to_up_100_atr,
                time_to_down_100_atr,
                time_to_up_150_atr,
                time_to_down_150_atr
            FROM qbn.barrier_outcomes
            WHERE asset_id = %s
              AND time_1 >= NOW() - INTERVAL '%s days'
              AND first_significant_barrier != 'none'
        """, (asset_id, lookback_days))
        
        rows = cur.fetchall()
    
    outcomes = {}
    for row in rows:
        outcomes[row[0]] = {
            'first_barrier': row[1],
            'up_050': row[2],
            'down_050': row[3],
            'up_100': row[4],
            'down_100': row[5],
            'up_150': row[6],
            'down_150': row[7],
        }
    
    return outcomes


def classify_barrier_direction(first_barrier: str) -> str:
    """Bepaal richting van first_significant_barrier."""
    if first_barrier.startswith('up_'):
        return 'bullish'
    elif first_barrier.startswith('down_'):
        return 'bearish'
    else:
        return 'neutral'


def evaluate_entry_accuracy(
    predictions: List[Dict],
    outcomes: Dict[datetime, Dict]
) -> Dict:
    """
    Evalueer entry prediction accuracy.
    
    Match predictions met outcomes en bereken:
    - Overall accuracy per horizon
    - Accuracy per confidence niveau
    - Confusion matrix
    """
    results = {
        'by_horizon': {},
        'by_confidence': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'confusion': defaultdict(Counter),
        'n_matched': 0,
        'n_unmatched': 0,
    }
    
    for pred in predictions:
        time = pred['time']
        
        # Zoek matching outcome
        outcome = outcomes.get(time)
        if not outcome:
            results['n_unmatched'] += 1
            continue
        
        results['n_matched'] += 1
        
        # Bepaal werkelijke richting
        actual = classify_barrier_direction(outcome['first_barrier'])
        
        # Evalueer per horizon
        for horizon in ['1h', '4h', '1d']:
            pred_key = f'pred_{horizon}'
            predicted = pred.get(pred_key, 'neutral')
            
            if horizon not in results['by_horizon']:
                results['by_horizon'][horizon] = {'total': 0, 'correct': 0}
            
            results['by_horizon'][horizon]['total'] += 1
            
            if predicted == actual:
                results['by_horizon'][horizon]['correct'] += 1
            
            # Confusion matrix per horizon
            results['confusion'][horizon][f"{predicted}_vs_{actual}"] += 1
        
        # Per confidence niveau
        confidence = pred.get('confidence', 'UNKNOWN')
        hypothesis = pred.get('hypothesis', 'NEUTRAL')
        
        results['by_confidence'][confidence]['total'] += 1
        
        # Simplified: check of hypothesis match met actual
        if (hypothesis == 'LONG' and actual == 'bullish') or \
           (hypothesis == 'SHORT' and actual == 'bearish') or \
           (hypothesis == 'NEUTRAL' and actual == 'neutral'):
            results['by_confidence'][confidence]['correct'] += 1
    
    return results


def generate_report(
    asset_id: int,
    lookback_days: int,
    accuracy_results: Dict,
    output_dir: Path
) -> str:
    """Genereer Markdown rapport."""
    
    lines = [
        f"# Prediction Accuracy Report",
        f"",
        f"**Asset ID:** {asset_id}  ",
        f"**Lookback:** {lookback_days} days  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC  ",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        f"- **Matched predictions:** {accuracy_results['n_matched']}",
        f"- **Unmatched predictions:** {accuracy_results['n_unmatched']}",
        f"",
    ]
    
    # Accuracy per horizon
    lines.extend([
        f"## Entry Prediction Accuracy by Horizon",
        f"",
        f"| Horizon | Accuracy | Correct | Total |",
        f"|---------|----------|---------|-------|",
    ])
    
    for horizon in ['1h', '4h', '1d']:
        stats = accuracy_results['by_horizon'].get(horizon, {'total': 0, 'correct': 0})
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            lines.append(
                f"| {horizon} | {acc:.1f}% | {stats['correct']} | {stats['total']} |"
            )
        else:
            lines.append(f"| {horizon} | N/A | 0 | 0 |")
    
    lines.append(f"")
    
    # Accuracy per confidence
    lines.extend([
        f"## Accuracy by Confidence Level",
        f"",
        f"| Confidence | Accuracy | Correct | Total |",
        f"|------------|----------|---------|-------|",
    ])
    
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        stats = accuracy_results['by_confidence'].get(conf, {'total': 0, 'correct': 0})
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            lines.append(
                f"| {conf} | {acc:.1f}% | {stats['correct']} | {stats['total']} |"
            )
        else:
            lines.append(f"| {conf} | N/A | 0 | 0 |")
    
    lines.extend([
        f"",
        f"---",
        f"",
        f"## Notes",
        f"",
        f"- Entry predictions worden gematcht met `first_significant_barrier` uit `qbn.barrier_outcomes`",
        f"- Accuracy meet % correcte richting-voorspellingen (bullish/bearish/neutral)",
        f"- Unmatched predictions: predictions zonder corresponding barrier outcome (bijv. geen significant movement)",
        f"",
    ])
    
    # Schrijf rapport
    report_path = output_dir / f"accuracy_report_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text("\n".join(lines), encoding='utf-8')
    
    return str(report_path)


def main():
    args = parse_args()
    
    logger.info(f"Starting prediction accuracy validation for asset {args.asset_id}")
    
    # Output directory
    output_mgr = ValidationOutputManager()
    output_dir = output_mgr.create_output_dir(
        script_name="prediction_accuracy",
        asset_id=args.asset_id,
    )
    
    # Fetch data
    logger.info(f"Fetching entry predictions (last {args.lookback_days} days)...")
    predictions = fetch_entry_predictions(args.asset_id, args.lookback_days)
    
    logger.info(f"Fetching barrier outcomes...")
    outcomes = fetch_barrier_outcomes(args.asset_id, args.lookback_days)
    
    if len(predictions) < args.min_samples:
        logger.warning(
            f"Onvoldoende predictions: {len(predictions)} < {args.min_samples}. "
            f"Verhoog lookback of wacht tot meer inference data beschikbaar is."
        )
        print(f"WARN: Insufficient predictions ({len(predictions)} < {args.min_samples})")
        return 0
    
    # Evaluate
    logger.info("Evaluating entry prediction accuracy...")
    accuracy = evaluate_entry_accuracy(predictions, outcomes)
    
    # Report
    report_path = generate_report(args.asset_id, args.lookback_days, accuracy, output_dir)
    logger.info(f"Report saved: {report_path}")
    
    # Print summary voor Dagster logs
    print(f"\nPrediction Accuracy Summary (asset {args.asset_id}):")
    print(f"  Matched predictions: {accuracy['n_matched']}")
    
    for horizon in ['1h', '4h', '1d']:
        stats = accuracy['by_horizon'].get(horizon, {'total': 0, 'correct': 0})
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {horizon} accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    logger.info("Prediction accuracy validation complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
