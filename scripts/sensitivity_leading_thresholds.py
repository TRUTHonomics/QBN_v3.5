"""
Sensitivity Analyse op Leading_Composite thresholds.

Scant verschillende neutral_band en strong_threshold combinaties en meet:
- Aantal trades gegenereerd
- Trade Hypothesis distributie
- Entry predictions variatie

Doel: Vind threshold combinatie die trades genereert zonder te veel noise.

Usage:
    docker exec QBN_v4_Dagster_Webserver python /app/scripts/sensitivity_leading_thresholds.py --asset-id 1
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, '/app')

try:
    from core.logging_utils import setup_logging
    setup_logging('sensitivity_leading_thresholds')
except ModuleNotFoundError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from database.db import get_cursor
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
from inference.gpu.gpu_inference_engine import GPUInferenceEngine
from simulation.data_loader import BacktestDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Sensitivity analyse op Leading Composite thresholds')
    parser.add_argument('--asset-id', type=int, default=1, help='Asset ID')
    parser.add_argument('--lookback-days', type=int, default=30, help='Days to analyze')
    parser.add_argument('--neutral-band-range', type=float, nargs=3, default=[0.02, 0.08, 0.01],
                        help='neutral_band range: min max step (default: 0.02 0.08 0.01)')
    parser.add_argument('--strong-threshold-range', type=float, nargs=3, default=[0.10, 0.20, 0.02],
                        help='strong_threshold range: min max step (default: 0.10 0.20 0.02)')
    return parser.parse_args()


def test_threshold_combination(
    asset_id: int,
    neutral_band: float,
    strong_threshold: float,
    data: pd.DataFrame,
    generator: QBNv3CPTGenerator
) -> Dict:
    """
    Test één threshold combinatie en return metrics.
    
    Returns:
        Dict met: trade_hypothesis_counts, prediction_variation, signal_stats
    """
    # Tijdelijk thresholds in config overschrijven
    # REASON: We moeten de ThresholdLoader class aanpassen om custom thresholds te accepteren
    # Voor nu simuleren we dit door direct de generator's threshold logic te patchen
    
    # Genereer composite scores met de test thresholds
    # Dit is een simplified versie — in productie zou je de volledige ThresholdConfig updaten
    from config.threshold_loader import ThresholdLoader
    
    # Mock threshold config
    mock_thresholds = {
        'leading_composite': {
            'neutral_band': neutral_band,
            'strong_threshold': strong_threshold,
            'bearish_neutral_band': neutral_band,
            'bullish_neutral_band': neutral_band,
            'bearish_strong_threshold': strong_threshold,
            'bullish_strong_threshold': strong_threshold,
        }
    }
    
    # Bereken discretized leading composite
    generator.load_signal_classification(asset_id, "1h")
    from inference.node_types import SemanticClass
    
    try:
        # Genereer leading composite CPT met custom thresholds
        # We moeten de signal_aggregator's threshold logic overschrijven
        aggregator = generator.signal_aggregator
        
        # Bereken raw scores
        leading_signals = [col for col in data.columns if col.endswith('_60') and 
                          aggregator.signal_classification.get(col.replace('_60', '')) == SemanticClass.LEADING]
        
        if not leading_signals:
            return {
                'neutral_band': neutral_band,
                'strong_threshold': strong_threshold,
                'error': 'No leading signals found',
                'trade_hypothesis_counts': {},
                'total_trades': 0
            }
        
        # Bereken weighted sum
        weighted_scores = []
        for sig in leading_signals:
            weight = aggregator.signal_weights.get(sig.replace('_60', ''), 1.0)
            weighted_scores.append(data[sig] * weight)
        
        if not weighted_scores:
            return {
                'neutral_band': neutral_band,
                'strong_threshold': strong_threshold,
                'error': 'No weighted scores',
                'trade_hypothesis_counts': {},
                'total_trades': 0
            }
        
        weighted_sum = sum(weighted_scores)
        total_weight = sum([aggregator.signal_weights.get(s.replace('_60', ''), 1.0) for s in leading_signals])
        
        # Normalize
        raw_score = weighted_sum / (total_weight * 2.0) if total_weight > 0 else 0.0
        
        # Discretize met custom thresholds
        def discretize_composite(score):
            if score <= -strong_threshold:
                return 'strong_bearish'
            elif score <= -neutral_band:
                return 'bearish'
            elif score >= strong_threshold:
                return 'strong_bullish'
            elif score >= neutral_band:
                return 'bullish'
            else:
                return 'neutral'
        
        leading_composite_states = raw_score.apply(discretize_composite)
        
        # Map naar Trade Hypothesis (simplified — actual heeft HTF_Regime als parent)
        # Voor sensitivity analyse assumeren we directe mapping
        def leading_to_hypothesis(state):
            if state == 'strong_bearish':
                return 'strong_short'
            elif state == 'bearish':
                return 'weak_short'
            elif state == 'strong_bullish':
                return 'strong_long'
            elif state == 'bullish':
                return 'weak_long'
            else:
                return 'no_setup'
        
        trade_hypothesis = leading_composite_states.apply(leading_to_hypothesis)
        
        # Count
        hypothesis_counts = trade_hypothesis.value_counts().to_dict()
        
        # Bereken trade count (alles behalve no_setup)
        total_trades = len(trade_hypothesis[trade_hypothesis != 'no_setup'])
        trade_pct = 100.0 * total_trades / len(trade_hypothesis) if len(trade_hypothesis) > 0 else 0.0
        
        # Variatie metrics
        unique_states = len(leading_composite_states.unique())
        entropy = -sum([(c/len(leading_composite_states)) * np.log2(c/len(leading_composite_states)) 
                        for c in leading_composite_states.value_counts() if c > 0])
        
        return {
            'neutral_band': neutral_band,
            'strong_threshold': strong_threshold,
            'trade_hypothesis_counts': hypothesis_counts,
            'leading_composite_counts': leading_composite_states.value_counts().to_dict(),
            'total_trades': total_trades,
            'trade_pct': trade_pct,
            'unique_states': unique_states,
            'entropy': entropy,
            'raw_score_mean': float(raw_score.mean()),
            'raw_score_std': float(raw_score.std()),
            'raw_score_min': float(raw_score.min()),
            'raw_score_max': float(raw_score.max()),
        }
        
    except Exception as e:
        logger.error(f"Error testing thresholds nb={neutral_band} st={strong_threshold}: {e}")
        return {
            'neutral_band': neutral_band,
            'strong_threshold': strong_threshold,
            'error': str(e),
            'trade_hypothesis_counts': {},
            'total_trades': 0
        }


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Leading Composite Threshold Sensitivity Analysis")
    logger.info("="*80)
    logger.info(f"Asset ID: {args.asset_id}")
    logger.info(f"Lookback: {args.lookback_days} days")
    logger.info(f"neutral_band range: {args.neutral_band_range[0]:.3f} to {args.neutral_band_range[1]:.3f} step {args.neutral_band_range[2]:.3f}")
    logger.info(f"strong_threshold range: {args.strong_threshold_range[0]:.3f} to {args.strong_threshold_range[1]:.3f} step {args.strong_threshold_range[2]:.3f}")
    
    # Load data
    logger.info("\n📥 Loading data...")
    loader = BacktestDataLoader(args.asset_id)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.lookback_days)
    
    data = loader.fetch_data(start_date, end_date)
    if data.empty:
        logger.error("No data found")
        return 1
    
    logger.info(f"✅ Loaded {len(data)} rows")
    
    # Initialize generator
    logger.info("\n🧠 Initializing CPT generator...")
    generator = QBNv3CPTGenerator()
    
    # Generate threshold grid
    nb_min, nb_max, nb_step = args.neutral_band_range
    st_min, st_max, st_step = args.strong_threshold_range
    
    neutral_bands = np.arange(nb_min, nb_max + nb_step/2, nb_step)
    strong_thresholds = np.arange(st_min, st_max + st_step/2, st_step)
    
    total_combinations = len(neutral_bands) * len(strong_thresholds)
    logger.info(f"\n🔍 Testing {total_combinations} threshold combinations...")
    logger.info(f"   neutral_band values: {list(neutral_bands)}")
    logger.info(f"   strong_threshold values: {list(strong_thresholds)}")
    
    # Run sensitivity scan
    results = []
    count = 0
    for nb in neutral_bands:
        for st in strong_thresholds:
            count += 1
            logger.info(f"\n[{count}/{total_combinations}] Testing neutral_band={nb:.3f}, strong_threshold={st:.3f}")
            
            result = test_threshold_combination(args.asset_id, nb, st, data, generator)
            results.append(result)
            
            if 'error' not in result:
                logger.info(f"   Trades: {result['total_trades']} ({result['trade_pct']:.1f}%)")
                logger.info(f"   Trade Hypothesis: {result['trade_hypothesis_counts']}")
                logger.info(f"   Entropy: {result['entropy']:.3f}, Unique states: {result['unique_states']}")
    
    # Analyse resultaten
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    # Sort door trade count
    valid_results = [r for r in results if 'error' not in r]
    sorted_results = sorted(valid_results, key=lambda x: x['total_trades'], reverse=True)
    
    logger.info(f"\nTop 10 configurations by trade count:")
    logger.info(f"{'Rank':<5} {'NB':<7} {'ST':<7} {'Trades':<8} {'Trade%':<8} {'Entropy':<9} {'States':<7}")
    logger.info("-" * 65)
    
    for i, r in enumerate(sorted_results[:10], 1):
        logger.info(f"{i:<5} {r['neutral_band']:<7.3f} {r['strong_threshold']:<7.3f} "
                   f"{r['total_trades']:<8} {r['trade_pct']:<7.1f}% {r['entropy']:<8.3f} {r['unique_states']:<7}")
    
    # Huidige production config
    current_nb = 0.05
    current_st = 0.15
    current_result = [r for r in valid_results if r['neutral_band'] == current_nb and r['strong_threshold'] == current_st]
    if current_result:
        r = current_result[0]
        logger.info(f"\nCurrent production config (nb={current_nb}, st={current_st}):")
        logger.info(f"   Trades: {r['total_trades']} ({r['trade_pct']:.1f}%)")
        logger.info(f"   Entropy: {r['entropy']:.3f}, States: {r['unique_states']}")
        
        # Compare met beste
        best = sorted_results[0]
        delta_trades = best['total_trades'] - r['total_trades']
        logger.info(f"\nBest config would generate {delta_trades:+d} more trades ({100*delta_trades/len(data):.1f}% more entries)")
    
    # Save results
    output_dir = Path('_validation/sensitivity_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'leading_thresholds_{timestamp}.json'
    
    output_data = {
        'timestamp': timestamp,
        'asset_id': args.asset_id,
        'lookback_days': args.lookback_days,
        'total_rows': len(data),
        'neutral_band_range': args.neutral_band_range,
        'strong_threshold_range': args.strong_threshold_range,
        'results': results,
        'top_10': sorted_results[:10],
        'current_production': {
            'neutral_band': current_nb,
            'strong_threshold': current_st,
            'result': current_result[0] if current_result else None
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\nSensitivity analysis complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
