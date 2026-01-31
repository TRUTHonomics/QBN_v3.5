#!/usr/bin/env python3
"""
validate_ida_weights.py - Dry-run validatie voor IDA training weights

Genereert visualisaties en statistieken voordat training_weight definitief
wordt ge-update. Dit helpt bij het identificeren van problemen voordat
ze de CPT training beïnvloeden.

Output:
- Histogram van weight-distributie per barrier type
- N_eff / N_raw ratio tabel
- Scatter plot: delta vs training_weight
- Tijdreeks plot voor recente periodes

Usage:
    python scripts/validate_ida_weights.py --asset-id 1 --output-dir _validation/asset_1/6_ida_weight_validation
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import glob
import shutil
# REASON: Fix typing imports
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from config.ida_config import IDAConfig
from core.logging_utils import setup_logging

# Plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

logger = setup_logging("validate_ida_weights")


class IDAWeightValidator:
    """Valideert IDA training weights met visualisaties."""
    
    def __init__(self, asset_id: int, output_dir: Optional[Path] = None):
        self.asset_id = asset_id
        self.output_dir = output_dir or (PROJECT_ROOT / '_validation' / 'ida_weights')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = IDAConfig.baseline()
    
    def fetch_data(self) -> pd.DataFrame:
        """Haal data op met berekende weights."""
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
        
        if df.empty:
            return df

        # Bereken hit_timestamp voor clustering
        df['hit_timestamp'] = pd.to_datetime(df['time_1']) + \
                              pd.to_timedelta(df['first_significant_time_min'], unit='m')
        
        # Bereken effective_score en delta voor analyse
        is_down = df['first_significant_barrier'].str.startswith('down')
        df['effective_score'] = df['leading_score'].copy()
        df.loc[is_down, 'effective_score'] = -df.loc[is_down, 'leading_score']
        
        # Cluster assignment
        df['cluster_id'] = df.groupby([
            'asset_id', 
            'first_significant_barrier', 
            'hit_timestamp'
        ]).ngroup()
        
        # Delta per cluster
        df = df.sort_values(['cluster_id', 'time_1'])
        df['delta'] = df.groupby('cluster_id')['effective_score'].diff().clip(lower=0)
        df['delta'] = df['delta'].fillna(0)
        
        return df
    
    def compute_neff_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereken N_eff statistieken per barrier type.
        
        N_eff = (sum(w))^2 / sum(w^2)
        """
        stats = []
        
        for barrier in sorted(df['first_significant_barrier'].unique()):
            mask = df['first_significant_barrier'] == barrier
            weights = df.loc[mask, 'training_weight']
            
            n_raw = len(weights)
            if n_raw == 0:
                continue
            
            n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
            ratio = n_eff / n_raw if n_raw > 0 else 0
            
            stats.append({
                'barrier': barrier,
                'n_raw': n_raw,
                'n_eff': n_eff,
                'ratio': ratio,
                'warning': '⚠️' if ratio < 0.3 else '✓'
            })
        
        return pd.DataFrame(stats)
    
    def validate_cluster_sums(self, df: pd.DataFrame) -> Dict:
        """Valideer dat som van weights per cluster = 1.0."""
        cluster_sums = df.groupby('cluster_id')['training_weight'].sum()
        
        tolerance = 0.001
        valid_clusters = ((cluster_sums - 1.0).abs() <= tolerance).sum()
        invalid_clusters = len(cluster_sums) - valid_clusters
        
        return {
            'total_clusters': len(cluster_sums),
            'valid_clusters': int(valid_clusters),
            'invalid_clusters': int(invalid_clusters),
            'all_valid': invalid_clusters == 0,
            'worst_deviation': float((cluster_sums - 1.0).abs().max()) if not cluster_sums.empty else 0.0,
        }
    
    def plot_weight_distribution(self, df: pd.DataFrame) -> Optional[str]:
        """Plot histogram van weight distributie per barrier type."""
        if not HAS_PLOTTING:
            logger.warning("matplotlib niet beschikbaar, skip plot")
            return None
        
        barriers = sorted(df['first_significant_barrier'].unique())
        n_barriers = len(barriers)
        if n_barriers == 0:
            return None
        
        fig, axes = plt.subplots(
            nrows=(n_barriers + 2) // 3, 
            ncols=min(3, n_barriers),
            figsize=(15, 4 * ((n_barriers + 2) // 3))
        )
        axes = np.array(axes).flatten() if n_barriers > 1 else [axes]
        
        for idx, barrier in enumerate(barriers):
            ax = axes[idx]
            data = df[df['first_significant_barrier'] == barrier]['training_weight']
            
            sns.histplot(data, bins=50, kde=True, ax=ax)
            ax.set_title(f'{barrier} (n={len(data):,})')
            ax.set_xlabel('Training Weight')
            ax.axvline(x=1.0/len(data) if len(data) > 0 else 0.5, 
                      color='red', linestyle='--', label='Uniform')
        
        # Hide unused axes
        for idx in range(len(barriers), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'weight_distribution_{self.asset_id}.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        return str(output_file)
    
    def plot_delta_vs_weight(self, df: pd.DataFrame) -> Optional[str]:
        """Scatter plot van delta vs training_weight."""
        if not HAS_PLOTTING or df.empty:
            return None
        
        # Sample voor performance
        sample_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            sample_df['delta'], 
            sample_df['training_weight'],
            alpha=0.3,
            c=sample_df['effective_score'].abs(),
            cmap='viridis',
            s=10
        )
        
        plt.colorbar(scatter, label='|Effective Score|')
        ax.set_xlabel('Delta (Score Change)')
        ax.set_ylabel('Training Weight')
        ax.set_title(f'Delta vs Training Weight (Asset {self.asset_id})')
        
        # Fit line
        if len(sample_df) > 10:
            try:
                z = np.polyfit(sample_df['delta'], sample_df['training_weight'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(sample_df['delta'].min(), sample_df['delta'].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
                ax.legend()
            except:
                pass
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'delta_vs_weight_{self.asset_id}.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        return str(output_file)
    
    def plot_time_series(self, df: pd.DataFrame, days: int = 7) -> Optional[str]:
        """Tijdreeks plot voor recente periode."""
        if not HAS_PLOTTING or df.empty:
            return None
        
        # Filter naar recente data
        max_time = df['time_1'].max()
        min_time = max_time - timedelta(days=days)
        recent_df = df[df['time_1'] >= min_time].copy()
        
        if len(recent_df) < 10:
            logger.warning(f"Te weinig data voor tijdreeks plot ({len(recent_df)} rows)")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Training weight over time
        axes[0].scatter(recent_df['time_1'], recent_df['training_weight'], 
                       alpha=0.5, s=10, c='blue')
        axes[0].set_ylabel('Training Weight')
        axes[0].set_title(f'IDA Weights Over Time (Last {days} Days)')
        
        # Plot 2: Leading score
        axes[1].scatter(recent_df['time_1'], recent_df['leading_score'], 
                       alpha=0.5, s=10, c='green')
        axes[1].set_ylabel('Leading Score')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 3: Delta
        axes[2].scatter(recent_df['time_1'], recent_df['delta'], 
                       alpha=0.5, s=10, c='orange')
        axes[2].set_ylabel('Delta')
        axes[2].set_xlabel('Time')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'time_series_{self.asset_id}.png'
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        return str(output_file)
    
    def run(self) -> Dict:
        """Run volledige validatie."""
        logger.info(f"Starting IDA weight validation for asset {self.asset_id}")
        
        # Fetch data
        df = self.fetch_data()
        
        if df.empty:
            return {'success': False, 'reason': 'no_data'}
        
        logger.info(f"Loaded {len(df):,} rows")
        
        # Compute stats
        neff_df = self.compute_neff_stats(df)
        cluster_validation = self.validate_cluster_sums(df)
        
        # Generate plots
        plots = {}
        if HAS_PLOTTING:
            plots['weight_distribution'] = self.plot_weight_distribution(df)
            plots['delta_vs_weight'] = self.plot_delta_vs_weight(df)
            plots['time_series'] = self.plot_time_series(df)
        
        # Summary stats
        weight_stats = {
            'min': float(df['training_weight'].min()),
            'max': float(df['training_weight'].max()),
            'mean': float(df['training_weight'].mean()),
            'std': float(df['training_weight'].std()),
            'median': float(df['training_weight'].median()),
        }
        
        # Correlation delta vs weight
        correlation = float(df['delta'].corr(df['training_weight']))
        
        result = {
            'success': True,
            'asset_id': self.asset_id,
            'n_rows': len(df),
            'n_clusters': df['cluster_id'].nunique(),
            'cluster_validation': cluster_validation,
            'neff_stats': neff_df.to_dict('records'),
            'weight_stats': weight_stats,
            'delta_weight_correlation': correlation,
            'plots': plots,
            'output_dir': str(self.output_dir),
        }
        
        # Save results
        output_file = self.output_dir / f'validation_report_{self.asset_id}.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Validate IDA Training Weights')
    parser.add_argument('--asset-id', type=int, required=True, help='Asset ID')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    validator = IDAWeightValidator(args.asset_id, output_dir)
    result = validator.run()
    
    # Print report
    print("\n" + "="*70)
    print("IDA WEIGHT VALIDATION REPORT")
    print("="*70)
    
    if not result['success']:
        print(f"❌ Validation failed: {result.get('reason', 'unknown')}")
        return
    
    print(f"\n📊 Data Summary:")
    print(f"   Total rows:    {result['n_rows']:,}")
    print(f"   Clusters:      {result['n_clusters']:,}")
    
    print(f"\n🔍 Cluster Validation:")
    cv = result['cluster_validation']
    status = "✅" if cv['all_valid'] else "❌"
    print(f"   {status} Valid clusters: {cv['valid_clusters']:,}/{cv['total_clusters']:,}")
    if not cv['all_valid']:
        print(f"   Worst deviation: {cv['worst_deviation']:.6f}")
    
    print(f"\n📈 Weight Statistics:")
    ws = result['weight_stats']
    print(f"   Min:    {ws['min']:.6f}")
    print(f"   Max:    {ws['max']:.6f}")
    print(f"   Mean:   {ws['mean']:.6f}")
    print(f"   Std:    {ws['std']:.6f}")
    print(f"   Median: {ws['median']:.6f}")
    
    print(f"\n📉 Delta-Weight Correlation: {result['delta_weight_correlation']:.4f}")
    if result['delta_weight_correlation'] > 0.3:
        print("   ✅ Positive correlation (expected)")
    else:
        print("   ⚠️ Low/negative correlation (investigate)")
    
    print(f"\n📋 N_eff Statistics by Barrier:")
    print(f"   {'Barrier':<15} {'N_raw':>8} {'N_eff':>10} {'Ratio':>8} {'Status':>8}")
    print("   " + "-"*55)
    
    for stat in result['neff_stats']:
        print(f"   {stat['barrier']:<15} {stat['n_raw']:>8,} {stat['n_eff']:>10.1f} "
              f"{stat['ratio']:>8.2%} {stat['warning']:>8}")
    
    if result['plots']:
        print(f"\n📊 Generated Plots:")
        for name, path in result['plots'].items():
            if path:
                print(f"   - {name}: {path}")
    
    print(f"\n📁 Full report: {result['output_dir']}")


if __name__ == '__main__':
    main()
