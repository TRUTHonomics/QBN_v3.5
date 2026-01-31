#!/usr/bin/env python3
"""
compute_barrier_weights.py - López de Prado Uniqueness + Soft-Attribution Delta (IDA)

Berekent training_weight voor barrier outcomes om seriële correlatie te corrigeren.
Signalen die dezelfde barrier-hit claimen krijgen samen gewicht 1.0, verdeeld op
basis van hun informatiewaarde (delta + absolute score).

Usage:
    python scripts/compute_barrier_weights.py --asset-id 1
    python scripts/compute_barrier_weights.py --asset-id 1 --dry-run
    python scripts/compute_barrier_weights.py --asset-id 1 --config balanced
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import glob
import shutil
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from config.ida_config import IDAConfig, STATIONARITY_DEFAULTS
from core.logging_utils import setup_logging

logger = setup_logging("compute_barrier_weights")


class IDAWeightCalculator:
    """
    López de Prado Uniqueness + Soft-Attribution Delta (IDA) Calculator.
    
    Verdeelt een totaalgewicht van 1.0 over alle signalen die dezelfde 
    barrier-hit claimen, waarbij signalen met hogere "informatiewaarde" 
    (delta + absolute score) meer gewicht krijgen.
    """
    
    def __init__(self, asset_id: int, config: Optional[IDAConfig] = None, run_id: Optional[str] = None):
        self.asset_id = asset_id
        self.config = config or IDAConfig.baseline()
        self.run_id = run_id
        self.stats = {}
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Haal data op voor IDA berekening.
        
        Returns:
            DataFrame met barrier outcomes inclusief leading_score
        """
        query = """
            SELECT 
                asset_id,
                time_1,
                leading_score,
                first_significant_barrier,
                first_significant_time_min,
                training_weight
            FROM qbn.barrier_outcomes
            WHERE asset_id = %s
              AND first_significant_barrier != 'none'
              AND leading_score IS NOT NULL
            ORDER BY time_1
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id,))
            rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=[
            'asset_id', 'time_1', 'leading_score', 
            'first_significant_barrier', 'first_significant_time_min',
            'training_weight'
        ])
        
        logger.info(f"Fetched {len(df):,} rows with barrier hits")
        return df
    
    def validate_stationarity(self, df: pd.DataFrame) -> Dict:
        """
        Pre-flight check: is leading_score stationair genoeg voor IDA?
        
        Tests:
        1. ADF test (Augmented Dickey-Fuller) - p < 0.05 = stationair
        2. Rolling mean stability - geen duidelijke trend
        3. Outlier ratio - niet teveel extreme waarden
        """
        try:
            from scipy.stats import zscore
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.warning("scipy/statsmodels niet beschikbaar, stationariteitscheck overgeslagen")
            return {'overall_pass': True, 'skipped': True}
        
        scores = df['leading_score'].dropna()
        
        if len(scores) < 100:
            logger.warning("Te weinig data voor stationariteitscheck")
            return {'overall_pass': True, 'insufficient_data': True}
        
        defaults = STATIONARITY_DEFAULTS
        
        # Test 1: ADF test
        try:
            adf_result = adfuller(scores, maxlag=24)
            adf_pvalue = adf_result[1]
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            adf_pvalue = 0.0  # Assume stationary
        
        # Test 2: Rolling mean trend
        rolling_mean = scores.rolling(window=min(defaults['rolling_window'], len(scores) // 10)).mean()
        mean_trend = rolling_mean.diff().mean() if len(rolling_mean.dropna()) > 0 else 0
        
        # Test 3: Outlier ratio (|z| > 3)
        z_scores = zscore(scores)
        outlier_ratio = (abs(z_scores) > defaults['outlier_zscore']).mean()
        
        result = {
            'adf_pvalue': float(adf_pvalue),
            'adf_stationary': adf_pvalue < defaults['adf_significance'],
            'mean_trend': float(mean_trend) if not np.isnan(mean_trend) else 0,
            'trend_acceptable': abs(mean_trend) < defaults['trend_threshold'],
            'outlier_ratio': float(outlier_ratio),
            'outliers_acceptable': outlier_ratio < defaults['max_outlier_ratio'],
            'overall_pass': all([
                adf_pvalue < defaults['adf_significance'],
                abs(mean_trend) < defaults['trend_threshold'],
                outlier_ratio < defaults['max_outlier_ratio']
            ])
        }
        
        if not result['overall_pass']:
            logger.warning(f"⚠️ STATIONARITY CHECK: adf_p={adf_pvalue:.4f}, "
                          f"trend={mean_trend:.6f}, outliers={outlier_ratio:.2%}")
        else:
            logger.info("✅ Stationarity check passed")
        
        return result
    
    def assign_cluster_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster signalen die EXACT dezelfde marktbeweging claimen.
        
        Geen fuzzy window - alleen signalen met identieke:
        - asset_id
        - first_significant_barrier
        - hit_timestamp (time_1 + first_significant_time_min)
        """
        # Bereken exacte kalender-tijd van de barrier hit
        df['hit_timestamp'] = pd.to_datetime(df['time_1']) + \
                              pd.to_timedelta(df['first_significant_time_min'], unit='m')
        
        # Cluster op exacte match
        df['cluster_id'] = df.groupby([
            'asset_id', 
            'first_significant_barrier', 
            'hit_timestamp'
        ]).ngroup()
        
        n_clusters = df['cluster_id'].nunique()
        avg_cluster_size = len(df) / n_clusters if n_clusters > 0 else 0
        
        logger.info(f"Created {n_clusters:,} clusters (avg size: {avg_cluster_size:.1f})")
        
        self.stats['n_clusters'] = n_clusters
        self.stats['avg_cluster_size'] = avg_cluster_size
        
        return df
    
    def compute_ida_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IDA Soft-Attribution Delta model.
        
        KRITISCH: Score wordt EERST geflipped voor DOWN barriers,
        DAARNA wordt delta berekend. Dit zorgt dat een beweging van
        +0.2 naar -0.8 (sterker bearish) als positieve delta telt.
        
        Formula per signaal i in cluster:
            effective_score_i = score_i (UP) of -score_i (DOWN)
            Delta_i = max(0, effective_score_i - effective_score_{i-1})
            A_i = (score_weight * |effective_score_i|) + (delta_weight * Delta_i) + epsilon
            Weight_i = A_i / sum(A_j)
        """
        results = []
        neff_warnings = 0
        
        for cluster_id, cluster_df in tqdm(df.groupby('cluster_id'), desc="Computing weights"):
            # Sorteer binnen cluster op tijd
            cluster_df = cluster_df.sort_values('time_1').copy()
            
            # STAP 1: Flip score voor DOWN barriers (EERST!)
            is_down = cluster_df['first_significant_barrier'].str.startswith('down')
            cluster_df['effective_score'] = cluster_df['leading_score'].copy()
            cluster_df.loc[is_down, 'effective_score'] = -cluster_df.loc[is_down, 'leading_score']
            
            # STAP 2: Bereken delta op effective_score (DAARNA!)
            cluster_df['delta'] = cluster_df['effective_score'].diff().clip(lower=0)
            cluster_df['delta'] = cluster_df['delta'].fillna(0)
            
            # STAP 3: Bereken attribution value
            cluster_df['attrib'] = (
                self.config.score_weight * cluster_df['effective_score'].abs() +
                self.config.delta_weight * cluster_df['delta'] +
                self.config.epsilon
            )
            
            # STAP 4: Normaliseer binnen cluster
            total = cluster_df['attrib'].sum()
            cluster_df['new_weight'] = cluster_df['attrib'] / total
            
            # STAP 5: N_eff monitoring en safeguard
            n_raw = len(cluster_df)
            weights = cluster_df['new_weight']
            n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
            
            if n_eff < self.config.neff_warning_ratio * n_raw and n_raw > 2:
                neff_warnings += 1
                # Apply minimum weight floor
                cluster_df['new_weight'] = cluster_df['new_weight'].clip(
                    lower=self.config.min_weight_floor
                )
                # Re-normalize
                cluster_df['new_weight'] = cluster_df['new_weight'] / cluster_df['new_weight'].sum()
            
            results.append(cluster_df)
        
        result_df = pd.concat(results, ignore_index=True)
        
        logger.info(f"Computed weights for {len(result_df):,} rows")
        if neff_warnings > 0:
            logger.warning(f"N_eff safeguard applied to {neff_warnings} clusters")
        
        self.stats['neff_warnings'] = neff_warnings
        
        return result_df
    
    def compute_neff_stats(self, df: pd.DataFrame) -> Dict:
        """
        Monitor effectieve sample size per barrier type.
        
        N_eff = (sum(w))^2 / sum(w^2)
        """
        stats = {}
        
        for barrier in df['first_significant_barrier'].unique():
            mask = df['first_significant_barrier'] == barrier
            weights = df.loc[mask, 'new_weight']
            
            n_raw = len(weights)
            if n_raw == 0:
                continue
                
            n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
            
            stats[barrier] = {
                'n_raw': int(n_raw),
                'n_eff': float(n_eff),
                'ratio': float(n_eff / n_raw) if n_raw > 0 else 0,
                'warning': n_eff < 0.3 * n_raw
            }
        
        return stats
    
    def validate_weights(self, df: pd.DataFrame) -> Dict:
        """
        Valideer de berekende weights.
        
        Checks:
        - Som per cluster = 1.0
        - Geen negatieve weights
        - Geen weights > 1.0
        """
        issues = []
        
        # Check sum per cluster
        cluster_sums = df.groupby('cluster_id')['new_weight'].sum()
        bad_sums = cluster_sums[abs(cluster_sums - 1.0) > 0.001]
        if len(bad_sums) > 0:
            issues.append(f"{len(bad_sums)} clusters met sum != 1.0")
        
        # Check bounds
        if (df['new_weight'] < 0).any():
            issues.append("Negatieve weights gevonden")
        
        if (df['new_weight'] > 1.0).any():
            issues.append("Weights > 1.0 gevonden")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_clusters_checked': len(cluster_sums),
            'weight_stats': {
                'min': float(df['new_weight'].min()),
                'max': float(df['new_weight'].max()),
                'mean': float(df['new_weight'].mean()),
                'std': float(df['new_weight'].std()),
            }
        }
    
    def update_database(self, df: pd.DataFrame) -> int:
        """
        Update training_weight in database.
        
        Returns:
            Aantal geüpdatete rijen
        """
        # Bouw update data
        updates = df[['asset_id', 'time_1', 'new_weight']].values.tolist()
        
        query = """
            UPDATE qbn.barrier_outcomes
            SET training_weight = %s,
                run_id = %s,
                updated_at = NOW()
            WHERE asset_id = %s
              AND time_1 = %s
        """
        
        updated = 0
        batch_size = 10000
        
        for i in tqdm(range(0, len(updates), batch_size), desc="Updating database"):
            batch = updates[i:i + batch_size]
            
            with get_cursor(commit=True) as cur:
                for asset_id, time_1, weight in batch:
                    cur.execute(query, (float(weight), self.run_id, asset_id, time_1))
                updated += len(batch)
        
        logger.info(f"Updated {updated:,} rows in database")
        return updated
    
    def run(self, dry_run: bool = False) -> Dict:
        """
        Run de volledige IDA weight berekening.
        
        Args:
            dry_run: Als True, bereken maar update database niet
            
        Returns:
            Dict met statistieken en resultaten
        """
        logger.info(f"Starting IDA weight calculation for asset {self.asset_id}")
        logger.info(f"Config: delta_weight={self.config.delta_weight}, "
                   f"score_weight={self.config.score_weight}")
        
        # Fetch data
        df = self.fetch_data()
        
        if len(df) == 0:
            logger.warning("Geen data gevonden met barrier hits en leading_score")
            return {'success': False, 'reason': 'no_data'}
        
        # Pre-flight stationarity check
        stationarity = self.validate_stationarity(df)
        
        # Assign clusters
        df = self.assign_cluster_ids(df)
        
        # Compute weights
        df = self.compute_ida_weights(df)
        
        # Validate
        validation = self.validate_weights(df)
        
        if not validation['valid']:
            logger.error(f"Validation failed: {validation['issues']}")
            if not dry_run:
                return {'success': False, 'reason': 'validation_failed', 'issues': validation['issues']}
        
        # N_eff stats
        neff_stats = self.compute_neff_stats(df)
        
        # Update database
        if not dry_run:
            updated = self.update_database(df)
        else:
            updated = 0
            logger.info("DRY RUN - database niet geüpdatet")
        
        result = {
            'success': True,
            'dry_run': dry_run,
            'rows_processed': len(df),
            'rows_updated': updated,
            'n_clusters': self.stats.get('n_clusters', 0),
            'avg_cluster_size': self.stats.get('avg_cluster_size', 0),
            'neff_warnings': self.stats.get('neff_warnings', 0),
            'stationarity': stationarity,
            'validation': validation,
            'neff_stats': neff_stats,
            'config': {
                'delta_weight': self.config.delta_weight,
                'score_weight': self.config.score_weight,
                'epsilon': self.config.epsilon,
            }
        }
        
        return result


def run_ablation_study(asset_id: int) -> Dict:
    """
    Run ablatiestudie met alle configuraties.
    
    Returns:
        Dict met resultaten per configuratie
    """
    configs = IDAConfig.get_ablation_configs()
    results = {}
    
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation: {name}")
        logger.info(f"{'='*60}")
        
        calculator = IDAWeightCalculator(asset_id, config)
        result = calculator.run(dry_run=True)
        results[name] = result
    
    return results


def main():
    parser = argparse.ArgumentParser(description='IDA Training Weight Calculator')
    parser.add_argument('--asset-id', type=int, required=True, help='Asset ID')
    parser.add_argument('--dry-run', action='store_true', help='Calculate but do not update database')
    parser.add_argument('--config', type=str, default='baseline',
                       choices=['baseline', 'balanced', 'delta_only', 'aggressive'],
                       help='IDA configuration to use')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for results')
    parser.add_argument('--run-id', type=str, help='Run identifier for traceability')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / '_validation' / 'ida_weights'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.ablation:
        # Run ablation study
        results = run_ablation_study(args.asset_id)
        
        # Save results
        output_file = output_dir / f'ablation_study_{args.asset_id}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nAblation results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)
        print(f"{'Config':<15} {'delta/score':<12} {'N_eff Warns':<12} {'Avg Weight Std':<15}")
        print("-"*70)
        
        for name, result in results.items():
            if result['success']:
                config = result['config']
                ratio = f"{config['delta_weight']:.1f}/{config['score_weight']:.1f}"
                neff_warns = result['neff_warnings']
                weight_std = result['validation']['weight_stats']['std']
                print(f"{name:<15} {ratio:<12} {neff_warns:<12} {weight_std:<15.4f}")
        
        return
    
    # Single run
    configs = IDAConfig.get_ablation_configs()
    config = configs.get(args.config, IDAConfig.baseline())
    
    calculator = IDAWeightCalculator(args.asset_id, config, run_id=args.run_id)
    result = calculator.run(dry_run=args.dry_run)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'ida_result_{args.asset_id}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("IDA WEIGHT CALCULATION COMPLETE")
    print("="*60)
    
    if result['success']:
        print(f"✅ Success: {'DRY RUN' if args.dry_run else 'Updated'}")
        print(f"   Rows processed:  {result['rows_processed']:,}")
        print(f"   Clusters:        {result['n_clusters']:,}")
        print(f"   Avg cluster size: {result['avg_cluster_size']:.1f}")
        print(f"   N_eff warnings:  {result['neff_warnings']}")
        
        if not args.dry_run:
            print(f"   Rows updated:    {result['rows_updated']:,}")
        
        # N_eff summary
        neff_stats = result['neff_stats']
        warnings = [b for b, s in neff_stats.items() if s.get('warning', False)]
        if warnings:
            print(f"\n⚠️ Low N_eff barriers: {', '.join(warnings)}")
    else:
        print(f"❌ Failed: {result.get('reason', 'unknown')}")
        if 'issues' in result:
            for issue in result['issues']:
                print(f"   - {issue}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
