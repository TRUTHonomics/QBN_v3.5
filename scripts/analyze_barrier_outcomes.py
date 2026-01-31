#!/usr/bin/env python3
"""
analyze_barrier_outcomes.py - Uitgebreide statistische analyse van barrier outcomes.

Berekent:
1. Hit rates voor ALLE barrier levels (0.25 tot 1.50).
2. Statistieken voor time-to-hit per level.
3. Visuele representatie van tijdsduur (ASCII percentiel diagram).
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import shutil

# REASON: Plotting support voor PNG output
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from core.logging_utils import setup_logging

def setup_output_dirs(custom_dir=None):
    """Setup output directories and archive existing plots."""
    if custom_dir:
        output_dir = Path(custom_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%y%m%d-%H-%M-%S")
    output_dir = PROJECT_ROOT / "_validation" / "outcome_barrier_analysis"
    archive_dir = PROJECT_ROOT / "_validation" / "archive"
    if output_dir.exists():
        archive_subdir = archive_dir / f"outcome_barrier_{timestamp}"
        archive_subdir.mkdir(parents=True, exist_ok=True)
        for png in output_dir.glob("*.png"):
            try:
                shutil.move(str(png), str(archive_subdir / png.name))
            except Exception:
                pass
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

BARRIER_COLS = [
    'time_to_up_025_atr', 'time_to_up_050_atr', 'time_to_up_075_atr',
    'time_to_up_100_atr', 'time_to_up_125_atr', 'time_to_up_150_atr',
    'time_to_up_175_atr', 'time_to_up_200_atr', 'time_to_up_225_atr',
    'time_to_up_250_atr', 'time_to_up_275_atr', 'time_to_up_300_atr',
    'time_to_down_025_atr', 'time_to_down_050_atr', 'time_to_down_075_atr',
    'time_to_down_100_atr', 'time_to_down_125_atr', 'time_to_down_150_atr',
    'time_to_down_175_atr', 'time_to_down_200_atr', 'time_to_down_225_atr',
    'time_to_down_250_atr', 'time_to_down_275_atr', 'time_to_down_300_atr'
]

def fetch_data(asset_id: Optional[int] = None) -> pd.DataFrame:
    """Haal alle relevante barrier data op inclusief IDA weights."""
    cols_str = ", ".join(BARRIER_COLS)
    query = f"""
        SELECT 
            asset_id,
            time_1,
            first_significant_barrier,
            first_significant_time_min,
            training_weight,
            {cols_str}
        FROM qbn.barrier_outcomes
    """
    if asset_id:
        query += f" WHERE asset_id = {asset_id}"
        
    logger.info(f"🔍 Querying qbn.barrier_outcomes (Asset: {asset_id or 'ALL'})...")
    
    with get_cursor() as cur:
        start_fetch = datetime.now()
        cur.execute(query)
        logger.info(f"⏳ Query executed, fetching results...")
        rows = cur.fetchall()
        elapsed = (datetime.now() - start_fetch).total_seconds()
        logger.info(f"✅ Fetched {len(rows)} rows in {elapsed:.2f}s")
        
    return pd.DataFrame(rows, columns=[
        'asset_id', 'time_1', 'first_sig_barrier', 'first_sig_time', 'training_weight'
    ] + BARRIER_COLS)

def draw_histogram(data: pd.Series, title: str, output_dir: Path, weights: Optional[pd.Series] = None, bins: int = 40):
    """Tekent een histogram met optionele weging (IDA)."""
    if data.empty:
        return
        
    # REASON: Window is nu 24 uur (1440 min)
    max_time = 1440
    # Gebruik weights voor np.histogram indien meegegeven
    w_values = weights.values if weights is not None else None
    counts, bin_edges = np.histogram(data, bins=bins, range=(0, max_time), weights=w_values)
    
    max_count = max(counts) if max(counts) > 0 else 1
    chart_width = 50
    
    label_type = "Weighted Info" if weights is not None else "Raw Count"
    print(f"\n📈 TIJD-DISTRIBUTIE HISTOGRAM ({label_type}): {title}")
    print(f"{'Tijd (min)':<12} {'Value':>10} | {'Frequentie':<50}")
    print("-" * 75)
    
    for i in range(len(counts)):
        bin_start = int(bin_edges[i])
        bin_end = int(bin_edges[i+1])
        count = counts[i]
        
        bar_len = int((count / max_count) * chart_width)
        bar = "█" * bar_len
        
        label = f"{bin_start:>4}-{bin_end:<4}"
        print(f"{label:<12} {count:>10.1f} | {bar}")
    print("-" * 75)

    if HAS_PLOTTING:
        try:
            plt.figure(figsize=(12, 6))
            sns.histplot(x=data, weights=w_values, bins=bins, binrange=(0, max_time), kde=True, color='blue' if weights is None else 'purple')
            plt.title(f"{label_type} Time-to-Hit: {title}")
            plt.xlabel("Time (minutes)")
            plt.ylabel(label_type)
            
            # REASON: Markeer X-as per 60 min (uurbasis)
            plt.xticks(np.arange(0, max_time + 1, 60))
            plt.xlim(0, max_time)
            
            plt.grid(True, alpha=0.3)
            
            suffix = "raw" if weights is None else "weighted"
            plt.savefig(output_dir / f"dist_{title.lower()}_{suffix}.png")
            plt.close()
        except Exception as e:
            print(f"Fout bij opslaan PNG voor {title}: {e}")

def analyze_stats(df: pd.DataFrame, output_dir: Path):
    """Voer de statistische analyse uit."""
    total_rows = len(df)
    if total_rows == 0:
        logger.warning("Geen data gevonden om te analyseren.")
        print("Geen data gevonden om te analyseren.")
        return

    # REASON: Bereken Effective Sample Size (N_eff)
    total_weight = df['training_weight'].sum()
    logger.info(f"📊 Analyzing {total_rows} rows, N_eff: {total_weight:.1f}")
    
    # REASON: Bereken onafhankelijkheid / correlatie
    # We voegen een kolom toe met de absolute 'kalender-tijd' van de hit
    logger.info("🕒 Calculating calendar hit times...")
    df['calendar_hit_time'] = pd.to_datetime(df['time_1']) + pd.to_timedelta(df['first_sig_time'], unit='m')
    
    print("\n" + "="*80)
    print(f"📊 BARRIER OUTCOME VOLLEDIGE ANALYSE")
    print(f"   Raw Samples (N): {total_rows:,}")
    print(f"   Information Content (N_eff): {total_weight:,.1f}")
    print(f"   Redundancy Factor: {total_rows / total_weight:.1f}x")
    print("="*80)

    # 1. Hit Rates voor ALLE barriers
    logger.info("📈 Calculating hit rates...")
    print("\n[1] HIT-RATES PER BARRIER LEVEL:")
    print(f"{'Barrier Level':<25} {'Raw Hits':>10} {'Weighted':>10} {'Eff. Rate %':>12}")
    print("-" * 65)
    
    hit_stats = []
    for col in BARRIER_COLS:
        mask = df[col].notna()
        raw_hits = mask.sum()
        weighted_hits = df.loc[mask, 'training_weight'].sum()
        
        # Effective Rate is weighted hits / total information content
        eff_rate = (weighted_hits / total_weight) * 100 if total_weight > 0 else 0
        
        name = col.replace('time_to_', '').replace('_atr', '').upper()
        print(f"{name:<25} {raw_hits:>10,} {weighted_hits:>10.1f} {eff_rate:>11.2f}%")
        
        hit_stats.append({
            'name': name, 
            'col': col, 
            'raw_hits': raw_hits, 
            'weighted_hits': weighted_hits, 
            'eff_rate': eff_rate
        })

    # 2. Correlatie Analyse (López de Prado Redundantie)
    logger.info("🔗 Performing redundancy analysis...")
    print("\n[2] LÓPEZ DE PRADO REDUNDANCY ANALYSIS:")
    print("-" * 65)
    for stat in [s for s in hit_stats if s['raw_hits'] > 100]:
        avg_uniqueness = stat['weighted_hits'] / stat['raw_hits'] if stat['raw_hits'] > 0 else 0
        print(f"{stat['name']:<15}: Avg Uniqueness={avg_uniqueness:.3f} (1/{1/(avg_uniqueness if avg_uniqueness > 0 else 1):.1f} signals per event)")

    # 3. Time-to-Hit Statistieken (Weighted)
    logger.info("⏱️ Calculating time-to-hit statistics...")
    print("\n[3] TIME-TO-HIT STATISTIEKEN (Weighted by Info Content):")
    print(f"{'Barrier':<20} {'W-Mean':>8} {'W-Std':>8} {'Median':>8} {'P2.5':>8} {'P97.5':>8}")
    print("-" * 70)
    
    for stat in hit_stats:
        if stat['raw_hits'] < 2:
            continue
            
        subset = df[df[stat['col']].notna()]
        times = subset[stat['col']]
        weights = subset['training_weight']
        
        # Weighted mean
        w_mean = (times * weights).sum() / weights.sum()
        # Weighted variance
        w_var = (weights * (times - w_mean)**2).sum() / weights.sum()
        w_std = np.sqrt(w_var)
        
        # Median/Quantiles (unweighted for simplicity in terminal, weighted in plots)
        median = times.median()
        p025 = times.quantile(0.025)
        p975 = times.quantile(0.975)
        
        print(f"{stat['name']:<20} {w_mean:>8.0f} {w_std:>8.0f} {median:>8.0f} {p025:>8.0f} {p975:>8.0f}")

    # 4. Visuele Representatie (Weighted Histogrammen)
    logger.info("🖼️ Starting visual analysis...")
    print("\n[4] VISUELE ANALYSE (WEIGHTED):")
    if HAS_PLOTTING:
        # A. Correlation Scatter Plot (Global + Zoom)
        try:
            logger.info("🎨 Generating correlation scatter plots...")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Subplot 1: Global view
            non_none = df[df['first_sig_barrier'] != 'none']
            sample_size = min(5000, len(non_none))
            if sample_size > 0:
                sample_df = non_none.sample(sample_size)
                ax1.scatter(pd.to_datetime(sample_df['time_1']), 
                            pd.to_datetime(sample_df['calendar_hit_time']), 
                            alpha=0.4, s=5, c=pd.factorize(sample_df['first_sig_barrier'])[0], cmap='tab20')
                ax1.set_title("Global Signal vs Hit Time (6 Years)")
                ax1.grid(True, alpha=0.3)

            # Subplot 2: Zoom view (Last 30 days of data)
            recent_threshold = df['time_1'].max() - pd.Timedelta(days=30)
            recent_df = df[(df['time_1'] > recent_threshold) & (df['first_sig_barrier'] != 'none')]
            if not recent_df.empty:
                ax2.scatter(pd.to_datetime(recent_df['time_1']), 
                            pd.to_datetime(recent_df['calendar_hit_time']), 
                            alpha=0.6, s=15, c=pd.factorize(recent_df['first_sig_barrier'])[0], cmap='tab20')
                ax2.set_title("ZOOM: Last 30 Days (Horizontal lines = Serial Correlation/Duplicates)")
                ax2.set_ylabel("Actual Market Hit Time")
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "barrier_correlation_scatter.png")
            plt.close()
            logger.info("✅ Correlation scatter plots generated.")
            print(" -> barrier_correlation_scatter.png (met zoom-sectie) gegenereerd")
        except Exception as e:
            logger.error(f"❌ Error in correlation plot: {e}")
            print(f"Fout bij opslaan correlation plot: {e}")

    # B. Histogrammen voor ALLE barriers (Weighted)
    logger.info("📊 Generating individual distribution plots...")
    print(" -> Genereren van distributie plots (Weighted)...")
    for stat in hit_stats:
        if stat['raw_hits'] >= 10:
            logger.info(f"  - Plotting {stat['name']}...")
            col_time = stat['col']
            subset = df[df[col_time].notna()]
            times = subset[col_time]
            weights = subset['training_weight']
            
            # Teken histogram (zowel Terminal ASCII als PNG)
            draw_histogram(times, stat['name'], output_dir, weights=weights)

    logger.info("✅ Full analysis completed.")
    print("\n" + "="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-id', type=int, help='Asset ID')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    args = parser.parse_args()
    
    logger = setup_logging("analyze_barrier_outcomes")
    logger.info(f"🚀 Starting analysis. Asset: {args.asset_id}, Output: {args.output_dir}")
    
    output_dir = setup_output_dirs(args.output_dir)
    logger.info(f"📂 Output directory prepared: {output_dir}")
    
    logger.info("📡 Fetching data from database...")
    data = fetch_data(args.asset_id)
    logger.info(f"✅ Data fetched: {len(data)} rows.")
    
    analyze_stats(data, output_dir)
