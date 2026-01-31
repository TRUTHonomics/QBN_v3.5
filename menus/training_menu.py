#!/usr/bin/env python3
"""
Training Menu voor QBN v3.1
===========================
Container: QBN_v3.1_Training (ROLE=training)
Doel: Model training, CPT generation, data preparatie
"""

import sys
import os
import subprocess
import time
import logging
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from menus.shared import (
    clear_screen, print_header, show_database_stats, 
    run_gpu_benchmark, run_archive_reports
)
from database.db import get_cursor
from inference.event_window_detector import (
    EventWindowDetector, EventWindowConfig, 
    save_events_to_cache  # save_event_labels_to_db is lokaal gedefinieerd
)
from inference.position_prediction_generator import PositionPredictionGenerator
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator


def show_menu():
    """Toon training menu - QBN v3.2 v260122"""
    print("--- Categorie 1: System Status ---")
    print("  1.  📊 Database Statistieken")
    print("  2.  🚀 GPU Performance Check")
    print()
    print("--- Categorie 2: Data Preparation ---")
    print("  3.  🛡️  Barrier Outcome Backfill")
    print("  4.  📊 Compute IDA Training Weights")
    print("  5.  📋 Outcome Status Check")
    print()
    print("--- Categorie 3a: Leading Composite Tuning (Entry Phase) ---")
    print("  6.  🧬 Entry Hypothesis (Leading) Signal Alpha")
    print("  7.  🔬 Composite Threshold Optimalisatie (Trigger)")
    print("  8.  🎲 Combination Alpha Analysis")
    print()
    print("--- Categorie 3b: Event & Management Context (In-Trade) ---")
    print("  9.  🎯 Event Window Detection & Status (v3.3 - met delta_leading)")
    print("  16. 📈 Position Delta Threshold Analyse (v3.2)")
    print()
    print("--- Categorie 4: CPT Generation ---")
    print("  10. 🧠 CPT Generation (v3.3 - Triple Composite)")
    print("  11. 💾 CPT Cache Status")
    print("  12. 🎯 Position_Prediction CPT Only")
    print()
    print("--- Categorie 5: Full Training ---")
    print("  13. 🚀 VOLLEDIGE TRAINING RUN (v3.3 Triple Composite)")
    print()
    print("--- Categorie 6: Utilities ---")
    print("  14. 📦 Archiveer Rapporten")
    print("  15. 🔄 Sync Config naar KFL Backend")
    print()
    print("  99. 🔄 Refresh status")
    print("  0.  Exit")
    print()
    return input("Keuze: ").strip()


# ==============================================================================
# Data Preparation
# ==============================================================================
def run_outcome_backfill():
    """GPU-accelerated outcome backfill naar qbn.signal_outcomes"""
    print("\n" + "="*60)
    print("🎯 LEGACY OUTCOME BACKFILL (qbn.signal_outcomes)")
    print("="*60 + "\n")

    print("📋 DOELTABEL: qbn.signal_outcomes (genormaliseerd)")
    print("   - Outcomes worden NIET meer opgeslagen in kfl.mtf_signals_*")
    print("   - ATR wordt opgehaald uit kfl.indicators.atr_14")
    print()
    print("📋 SUBMENU:")
    print("  1. Status check")
    print("  2. Run backfill (single asset ID)")
    print("  3. Run backfill (selected assets)")
    print("  4. Run backfill (all assets)")
    print("  5. Validate backfill")
    print("  0. Terug")

    choice = input("\nKeuze: ").strip()

    if choice == '1':
        print("\n🔄 Running status check...")
        # REASON: outcome_backfill.py is vervangen door barrier_backfill.py
        subprocess.run([sys.executable, 'scripts/barrier_backfill.py', '--asset-id', '1', '--status'], cwd=PROJECT_ROOT)

    elif choice == '2':
        asset_id = input("\nAsset ID (bijv. 1): ").strip()
        if not asset_id:
            print("❌ Asset ID is verplicht")
            input("\nDruk op Enter om terug te gaan...")
            return
        print(f"\n🔄 Running barrier backfill voor asset ID {asset_id}...")
        subprocess.run([sys.executable, 'scripts/barrier_backfill.py', '--asset-id', asset_id], cwd=PROJECT_ROOT)

    elif choice == '3':
        print(f"\n🔄 Running barrier backfill voor geselecteerde assets...")
        # REASON: Gebruik qbn_pipeline_runner om over assets te loopen (parallel)
        subprocess.run([sys.executable, 'scripts/qbn_pipeline_runner.py', '--tasks', 'outcome', '--assets', 'selected', '--workers', '4'], cwd=PROJECT_ROOT)

    elif choice == '4':
        confirm = input("\n⚠️  Dit kan lang duren voor ALLE assets. Doorgaan? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Geannuleerd")
            input("\nDruk op Enter om terug te gaan...")
            return
        print("\n🔄 Running full barrier backfill via pipeline runner (parallel)...")
        subprocess.run([sys.executable, 'scripts/qbn_pipeline_runner.py', '--tasks', 'outcome', '--assets', 'all', '--workers', '4'], cwd=PROJECT_ROOT)

    elif choice == '5':
        scope = input("\nValidatie scope (asset ID of 'all') [all]: ").strip() or "all"
        print(f"\n🔄 Running barrier validation voor {scope}...")
        if scope.lower() == 'all':
            subprocess.run([sys.executable, 'scripts/analyze_barrier_outcomes.py'], cwd=PROJECT_ROOT)
        else:
            subprocess.run([sys.executable, 'scripts/analyze_barrier_outcomes.py', '--asset-id', scope], cwd=PROJECT_ROOT)

    elif choice == '0':
        return
    else:
        print("\n⚠️  Ongeldige keuze")

    input("\nDruk op Enter om terug te gaan...")


def run_barrier_backfill():
    """GPU-accelerated barrier outcome backfill naar qbn.barrier_outcomes"""
    print("\n" + "="*60)
    print("🛡️ BARRIER OUTCOME BACKFILL (qbn.barrier_outcomes)")
    print("="*60 + "\n")

    print("📋 DOELTABEL: qbn.barrier_outcomes (First-Touch methodology)")
    print("   - Gebruikt ATR-normalized barriers")
    print("   - GPU versneld via CuPy")
    print()
    
    asset_id = input("Asset ID (bijv. 1): ").strip()
    if not asset_id:
        print("❌ Asset ID is verplicht")
        return
        
    batch_size = input("Batch size [100000]: ").strip() or "100000"
    config_name = input("Barrier Config name [default]: ").strip() or "default"
    
    print("\nStart opties:")
    print("  1. Resume vanaf checkpoint (indien aanwezig)")
    print("  2. Volledige backfill (vanaf het begin)")
    print("  3. Vanaf specifieke datum (YYYY-MM-DD)")
    start_mode = input("Keuze [1]: ").strip() or "1"
    
    cmd = [
        sys.executable, 'scripts/barrier_backfill.py',
        '--asset-id', asset_id,
        '--batch-size', batch_size,
        '--config', config_name
    ]
    
    if start_mode == '2':
        cmd.append('--no-resume')
        cmd.append('--overwrite')
    elif start_mode == '3':
        since = input("Datum (bijv. 2020-01-01): ").strip()
        if since:
            cmd.extend(['--since', since])
    
    print(f"\n🔄 Start barrier backfill voor asset {asset_id}...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    input("\nDruk op Enter om terug te gaan...")


def run_barrier_validation():
    """Valideer barrier data integriteit"""
    print("\n" + "="*60)
    print("✅ BARRIER DATA VALIDATION")
    print("="*60 + "\n")
    
    asset_id = input("Asset ID (leeg voor alle) []: ").strip()
    
    cmd = [sys.executable, 'scripts/validate_barrier_data.py']
    if asset_id:
        cmd.extend(['--asset-id', asset_id])
        
    print(f"\n🔄 Start validatie...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)

    # REASON: Diepe statistische analyse altijd uitvoeren
    print(f"\n📊 Start diepe statistische analyse...")
    cmd_stats = [sys.executable, 'scripts/analyze_barrier_outcomes.py']
    if asset_id:
        cmd_stats.extend(['--asset-id', asset_id])
    subprocess.run(cmd_stats, cwd=PROJECT_ROOT)
        
    input("\nDruk op Enter om terug te gaan...")


def run_outcome_analysis():
    """Run analyses on outcome data"""
    print("\n" + "="*60)
    print("📊 OUTCOME ANALYSE (DISTRIBUTIE/ATR)")
    print("="*60 + "\n")
    
    print("Scope selecteren:")
    print("  1. Specifiek asset")
    print("  2. Actief in huidige run")
    print("  3. Alle assets")
    print("  0. Terug")
    
    choice = input("\nKeuze: ").strip()
    if choice == '0' or not choice:
        return

    cmd = [sys.executable, 'scripts/validate_outcome_backfill.py']

    if choice == '1':
        asset = input("Asset ID of symbool [1]: ").strip() or "1"
        if asset.isdigit():
            cmd.extend(['--asset-id', asset])
        else:
            cmd.extend(['--asset', asset])
    elif choice == '2':
        cmd.append('--selected')
    elif choice == '3':
        cmd.append('--all')
    else:
        print("⚠️ Ongeldige keuze")
        return

    print(f"\n🔄 Start analyse...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    input("\nDruk op Enter om terug te gaan...")


def run_outcome_status():
    """Toon outcome coverage status"""
    print("\n" + "="*60)
    print("📋 OUTCOME STATUS")
    print("="*60 + "\n")
    
    print("Scope selecteren:")
    print("  1. Specifiek asset")
    print("  2. Actief in huidige run")
    print("  3. Alle assets")
    print("  0. Terug")
    
    choice = input("\nKeuze: ").strip()
    if choice == '0' or not choice:
        return

    try:
        from inference.target_generator import create_target_generator
        from database.db import get_cursor
        
        generator = create_target_generator()
        asset_ids = []

        if choice == '1':
            aid = input("Asset ID [1]: ").strip() or "1"
            asset_ids = [int(aid)]
        elif choice == '2':
            with get_cursor() as cur:
                cur.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1 ORDER BY id")
                asset_ids = [row[0] for row in cur.fetchall()]
        elif choice == '3':
            with get_cursor() as cur:
                cur.execute("SELECT DISTINCT asset_id FROM kfl.mtf_signals_lead ORDER BY asset_id")
                asset_ids = [row[0] for row in cur.fetchall()]

        if not asset_ids:
            print("❌ Geen assets gevonden")
            input("\nDruk op Enter om terug te gaan...")
            return

        print(f"\n{'ID':<5} {'Total':>10} {'1h':>8} {'4h':>8} {'1d':>8} {'ATR':>8} {'Ready'}")
        print("-" * 65)
        
        for aid in asset_ids:
            try:
                r = generator.verify_training_data_ready(aid)
                ready_mark = "✅" if r.get('ready') else "❌"
                avg_atr = (r.get('coverage_atr_1h', 0) + r.get('coverage_atr_4h', 0) + r.get('coverage_atr_1d', 0)) / 3
                print(f"{aid:<5} {r.get('total_rows', 0):>10,} {r.get('coverage_1h', 0):>7.0%} {r.get('coverage_4h', 0):>7.0%} {r.get('coverage_1d', 0):>7.0%} {avg_atr:>7.0%} {ready_mark}")
            except Exception as e:
                print(f"{aid:<5} FOUT: {str(e)[:40]}")
    
    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()
    
    input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Helper functions for Option 9 (Event Detection with DB sync)
# ==============================================================================
def _get_optimal_leading_thresholds(asset_id: int) -> Tuple[float, float]:
    """Haal de optimale Leading thresholds uit de database (gezet door Optie 7)."""
    from config.threshold_loader import ThresholdLoader
    from core.config_defaults import DEFAULT_COMPOSITE_STRONG_THRESHOLD, DEFAULT_EVENT_DELTA_THRESHOLD
    from core.config_warnings import warn_fallback_active
    
    try:
        # We proberen de 1h horizon als baseline voor de triggers
        loader = ThresholdLoader(asset_id=asset_id, horizon='1h')
        # We forceren een check of de waarden uit de DB komen of fallback zijn
        strong = float(loader.composite_strong_threshold)
        
        # Voor delta pakken we een default als deze niet in de loader zit
        delta = DEFAULT_EVENT_DELTA_THRESHOLD 
        
        # Check in de database of er een resultaat is voor delta_threshold
        with get_cursor() as cur:
            cur.execute("""
                SELECT param_value FROM qbn.composite_threshold_config 
                WHERE asset_id = %s AND config_type = 'leading_composite' 
                AND param_name = 'delta_threshold'
                ORDER BY updated_at DESC LIMIT 1
            """, (asset_id,))
            row = cur.fetchone()
            if row:
                delta = float(row[0])
                
        return strong, delta
    except Exception as e:
        from core.config_defaults import DEFAULT_COMPOSITE_STRONG_THRESHOLD, DEFAULT_EVENT_DELTA_THRESHOLD
        from core.config_warnings import warn_fallback_active
        
        warn_fallback_active(
            component="training_menu",
            config_name=f"asset_{asset_id}_leading_thresholds",
            fallback_values={'strong': DEFAULT_COMPOSITE_STRONG_THRESHOLD, 'delta': DEFAULT_EVENT_DELTA_THRESHOLD},
            reason=str(e),
            fix_command="Draai menu optie 7 (Threshold Optimalisatie)"
        )
        return DEFAULT_COMPOSITE_STRONG_THRESHOLD, DEFAULT_EVENT_DELTA_THRESHOLD


def run_event_window_detection_logic(asset_id: int, auto_save: bool = False):
    """
    Core logica voor event detection, losgekoppeld van menu-interactie.
    Wordt gebruikt door de 'Volledige Run' om automatisch drempels uit de DB te halen.
    """
    barrier_data = load_barrier_outcomes(asset_id)
    if barrier_data.empty:
        logger.error(f"Geen barrier data gevonden voor asset {asset_id}")
        return False
        
    # Haal drempels uit DB (gezet door vorige stap in de run)
    opt_strong, opt_delta = _get_optimal_leading_thresholds(asset_id)
    
    config = EventWindowConfig(
        absolute_threshold=opt_strong,
        delta_threshold=opt_delta,
        max_window_minutes=1440
    )
    
    detector = EventWindowDetector(config)
    events, labeled_data = detector.detect_events(barrier_data, asset_id)
    
    if auto_save:
        save_event_labels_to_db(asset_id, labeled_data)
        save_events_to_cache(events)  # KRITISCH: sla events op in event_windows tabel
        logger.info(f"✅ {len(events)} events automatisch gedetecteerd en opgeslagen (thr={opt_strong:.2f})")
    
    return True


def run_event_window_detection():
    """Menu optie 9: Event Window Detection."""
    print("\n" + "="*60)
    print("🎯 EVENT WINDOW DETECTION (v3.1)")
    print("="*60)
    
    asset_id_str = input("Asset ID [1]: ") or "1"
    asset_id = int(asset_id_str)
    
    # Load barrier data
    barrier_data = load_barrier_outcomes(asset_id)
    
    if barrier_data.empty:
        print("❌ Geen barrier data gevonden. Draai eerst Barrier Outcome Backfill.")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    # REASON: Automatisch inlezen van optimale drempels uit de database (v3.1 flow)
    print("\n🔍 Zoeken naar optimale drempels in database...")
    opt_strong, opt_delta = _get_optimal_leading_thresholds(asset_id)
    
    print("\nDetectie Parameters:")
    spike_thr = float(input(f"   Spike threshold (abs) [{opt_strong:.2f}]: ") or str(opt_strong))
    delta_thr = float(input(f"   Spike threshold (delta) [{opt_delta:.2f}]: ") or str(opt_delta))
    max_win = int(input("   Max window (min) [1440]: ") or "1440")
    
    config = EventWindowConfig(
        absolute_threshold=spike_thr,
        delta_threshold=delta_thr,
        max_window_minutes=max_win
    )
    
    # Detecteer events
    print("🔄 Bezig met detectie...")
    detector = EventWindowDetector(config)
    events, labeled_data = detector.detect_events(barrier_data, asset_id)
    
    # Toon statistieken
    stats = detector.validate_events(events)
    
    print("\n📊 Event Detection Results:")
    print(f"   Events gevonden: {stats.total_events}")
    print(f"   Gem. duur: {stats.avg_duration_min:.0f} min")
    print(f"   Timeouts: {stats.events_with_timeout} ({100*stats.events_with_timeout/(max(1,stats.total_events)):.1f}%)")
    
    # Optie om naar database te schrijven
    save = input("\nOpslaan naar database? [y/N]: ").lower() == 'y'
    if save:
        save_event_labels_to_db(asset_id, labeled_data)
        save_events_to_cache(events)  # KRITISCH: sla events op in event_windows tabel
        print("✅ Event labels opgeslagen in qbn.barrier_outcomes")
        print(f"✅ {len(events)} events opgeslagen in qbn.event_windows")
    
    input("\nDruk op Enter om terug te gaan...")


def run_position_delta_threshold_analysis():
    """
    Menu optie 16: Position Delta Threshold Analyse (v3.2).
    
    MI Grid Search voor optimalisatie van delta thresholds in Position Management.
    Delta scores meten verandering in coincident/confirming composites sinds trade entry.
    """
    print("\n" + "="*60)
    print("📈 POSITION DELTA THRESHOLD ANALYSE (v3.2)")
    print("="*60)
    
    print("""
    Deze analyse optimaliseert thresholds voor delta scores in Position Management.
    
    Delta scores meten de verandering in coincident/confirming composites sinds
    de entry van een trade. Dit is informatiever dan absolute waarden omdat het
    de richting van de markt relatief aan de entry weergeeft.
    
    VEREISTEN:
    - Event Window Detection moet al gedraaid zijn
    - barrier_outcomes moet event_id labels bevatten
    
    OUTPUT:
    - Optimale thresholds in qbn.position_delta_threshold_config
    - MI heatmaps in _validation/position_delta_analysis/
    """)
    
    print("📋 SUBMENU:")
    print("  1. Status check (beschikbare event data)")
    print("  2. Run analyse (single asset)")
    print("  3. Run analyse (dry-run, niet opslaan)")
    print("  0. Terug")
    
    choice = input("\nKeuze: ").strip()
    
    if choice == '1':
        # Status check
        print("\n🔍 Checking event data availability...")
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT 
                        asset_id,
                        COUNT(*) as total_rows,
                        COUNT(event_id) as event_rows,
                        COUNT(DISTINCT event_id) as n_events
                    FROM qbn.barrier_outcomes
                    WHERE time_1 >= NOW() - INTERVAL '180 days'
                    GROUP BY asset_id
                    ORDER BY asset_id
                """)
                rows = cur.fetchall()
                
                print("\n📊 Event Data per Asset:")
                print(f"{'Asset':<8} {'Total Rows':<12} {'Event Rows':<12} {'Events':<10}")
                print("-" * 42)
                
                for asset_id, total, event_rows, n_events in rows:
                    pct = 100 * event_rows / total if total > 0 else 0
                    print(f"{asset_id:<8} {total:<12,} {event_rows:<12,} {n_events:<10} ({pct:.1f}%)")
                
                # Check delta threshold config
                cur.execute("SELECT COUNT(*) FROM qbn.position_delta_threshold_config")
                config_count = cur.fetchone()[0]
                print(f"\n💾 Delta threshold configs in DB: {config_count}")
                
        except Exception as e:
            print(f"❌ Fout: {e}")
    
    elif choice == '2':
        # Run analyse
        asset_id_str = input("\nAsset ID [1]: ") or "1"
        asset_id = int(asset_id_str)
        
        lookback = input("Lookback days [all]: ") or ""
        
        print(f"\n🔬 Running Position Delta Threshold Analysis for asset {asset_id}...")
        cmd = [
            sys.executable, 
            'scripts/run_position_delta_threshold_analysis.py',
            '--asset-id', str(asset_id)
        ]
        if lookback:
            cmd.extend(['--lookback', lookback])
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    elif choice == '3':
        # Dry run
        asset_id_str = input("\nAsset ID [1]: ") or "1"
        asset_id = int(asset_id_str)
        
        print(f"\n🔬 Running Position Delta Threshold Analysis (DRY RUN) for asset {asset_id}...")
        subprocess.run([
            sys.executable, 
            'scripts/run_position_delta_threshold_analysis.py',
            '--asset-id', str(asset_id),
            '--dry-run'
        ], cwd=PROJECT_ROOT)
    
    elif choice == '0':
        return
    
    input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Training Weights (IDA)
# ==============================================================================
def run_ida_weights():
    """Compute IDA Training Weights (López de Prado + Soft-Attribution Delta)"""
    print("\n" + "="*60)
    print("📊 IDA TRAINING WEIGHTS (López de Prado)")
    print("="*60 + "\n")
    
    print("De IDA-methodiek verdeelt een totaalgewicht van 1.0 over alle")
    print("signalen die dezelfde barrier-hit claimen, waarbij signalen met")
    print("hogere 'informatiewaarde' (delta + score) meer gewicht krijgen.")
    print()
    
    print("📋 SUBMENU:")
    print("  1. Materialiseer leading_score (vereist voor IDA)")
    print("  2. Bereken IDA weights (dry-run)")
    print("  3. Bereken IDA weights (update database)")
    print("  4. Run ablatiestudie (test 4 configuraties)")
    print("  0. Terug")
    
    choice = input("\nKeuze: ").strip()
    
    if choice == '1':
        # Materialiseer leading_score
        asset_id = input("\nAsset ID (bijv. 1): ").strip()
        if not asset_id:
            print("❌ Asset ID is verplicht")
            input("\nDruk op Enter om terug te gaan...")
            return
        
        overwrite = input("Overschrijf bestaande scores? (y/n) [n]: ").strip().lower() == 'y'
        
        cmd = [sys.executable, 'scripts/materialize_leading_scores.py', '--asset-id', asset_id]
        if overwrite:
            cmd.append('--overwrite')
        
        print(f"\n🔄 Start leading_score materialisatie voor asset {asset_id}...")
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    elif choice == '2':
        # Dry-run IDA weights
        asset_id = input("\nAsset ID (bijv. 1): ").strip()
        if not asset_id:
            print("❌ Asset ID is verplicht")
            input("\nDruk op Enter om terug te gaan...")
            return
        
        config = _get_ida_config()
        
        cmd = [
            sys.executable, 'scripts/compute_barrier_weights.py',
            '--asset-id', asset_id,
            '--config', config,
            '--dry-run'
        ]
        
        print(f"\n🔄 Start IDA weight berekening (dry-run) voor asset {asset_id}...")
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    elif choice == '3':
        # Update database
        asset_id = input("\nAsset ID (bijv. 1): ").strip()
        if not asset_id:
            print("❌ Asset ID is verplicht")
            input("\nDruk op Enter om terug te gaan...")
            return
        
        config = _get_ida_config()
        
        confirm = input(f"\n⚠️ Dit zal training_weight updaten in de database. Doorgaan? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Geannuleerd")
            input("\nDruk op Enter om terug te gaan...")
            return
        
        cmd = [
            sys.executable, 'scripts/compute_barrier_weights.py',
            '--asset-id', asset_id,
            '--config', config
        ]
        
        print(f"\n🔄 Start IDA weight berekening voor asset {asset_id}...")
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    elif choice == '4':
        # Ablatiestudie
        asset_id = input("\nAsset ID (bijv. 1): ").strip()
        if not asset_id:
            print("❌ Asset ID is verplicht")
            input("\nDruk op Enter om terug te gaan...")
            return
        
        cmd = [
            sys.executable, 'scripts/compute_barrier_weights.py',
            '--asset-id', asset_id,
            '--ablation'
        ]
        
        print(f"\n🔄 Start ablatiestudie voor asset {asset_id}...")
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    elif choice == '0':
        return
    else:
        print("\n⚠️ Ongeldige keuze")
    
    input("\nDruk op Enter om terug te gaan...")


def run_ida_validation():
    """Valideer IDA Training Weights met visualisaties"""
    print("\n" + "="*60)
    print("🔍 IDA WEIGHT VALIDATION (Dry-Run)")
    print("="*60 + "\n")
    
    print("Genereert validatie-rapporten en visualisaties:")
    print("  - Weight distributie histogrammen")
    print("  - N_eff / N_raw ratio's")
    print("  - Delta vs Weight correlatie")
    print("  - Tijdreeks analyse")
    print()
    
    asset_id = input("Asset ID (bijv. 1): ").strip()
    if not asset_id:
        print("❌ Asset ID is verplicht")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    cmd = [sys.executable, 'scripts/validate_ida_weights.py', '--asset-id', asset_id]
    
    print(f"\n🔄 Start IDA weight validatie voor asset {asset_id}...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    input("\nDruk op Enter om terug te gaan...")


def _get_ida_config():
    """Helper om IDA configuratie te kiezen."""
    print("\nSelecteer IDA Configuratie:")
    print("  1. Baseline (80% delta, 20% score) [Default]")
    print("  2. Balanced (50% delta, 50% score)")
    print("  3. Delta-only (100% delta)")
    print("  4. Aggressive (90% delta, 10% score)")
    
    choice = input("Keuze [1]: ").strip() or "1"
    config_map = {'1': 'baseline', '2': 'balanced', '3': 'delta_only', '4': 'aggressive'}
    return config_map.get(choice, 'baseline')


# ==============================================================================
# Signal Analysis
# ==============================================================================
def run_alpha_analysis(layer='HYPOTHESIS'):
    """Signal Alpha Analysis (Hypothesis or Confidence)"""
    from database.db import get_cursor
    print("\n" + "="*60)
    title = "🧬 ENTRY HYPOTHESIS (Leading) SIGNAL ALPHA ANALYSIS" if layer == 'HYPOTHESIS' else "🛡️ ENTRY CONFIDENCE (Coincident/Confirming) SIGNAL ALPHA ANALYSIS"
    print(title)
    print("="*60 + "\n")
    
    print("Selecteer asset(s):")
    print("  1. Bitcoin (Asset ID 1)")
    print("  2. Ander Asset ID invoeren")
    print("  3. Active in current run")
    print("  0. Terug")
    
    choice = input("\nKeuze: ").strip()
    if choice == '0' or not choice:
        return

    asset_ids = []
    if choice == '1':
        asset_ids = [1]
    elif choice == '2':
        asset_id = input("Asset ID: ").strip()
        if asset_id:
            asset_ids = [int(asset_id)]
    elif choice == '3':
        with get_cursor() as cur:
            cur.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1")
            asset_ids = [row[0] for row in cur.fetchall()]

    if not asset_ids:
        print("⚠️ Geen geldige assets geselecteerd.")
        return

    script_path = 'alpha-analysis/analyze_signal_alpha.py'
    
    for asset_id in asset_ids:
        print(f"\n🔄 Start {layer} analyse voor asset {asset_id}...")
        result = subprocess.run([
            sys.executable, script_path, 
            '--asset', str(asset_id),
            '--layer', layer
        ], cwd=PROJECT_ROOT)
        
        if result.returncode == 0:
            print(f"✅ Asset {asset_id} voltooid")
        else:
            print(f"❌ Asset {asset_id} gefaald")
    
    input("\nDruk op Enter om terug te gaan...")


def run_threshold_analysis():
    """Composite Threshold Optimalisatie Analyse"""
    print("\n" + "="*60)
    print("🔬 COMPOSITE THRESHOLD OPTIMALISATIE")
    print("="*60 + "\n")
    
    print("Bepaalt optimale threshold waarden voor:")
    print("  - COMPOSITE_NEUTRAL_BAND")
    print("  - COMPOSITE_STRONG_THRESHOLD")
    print()
    
    try:
        asset_id = input("Asset ID [1]: ").strip() or "1"
        methods = input("Methoden (mi,cart,logreg) [all]: ").strip() or "mi,cart,logreg"
        horizons = input("Horizons (1h,4h,1d of all) [all]: ").strip() or "all"
        lookback = input("Lookback dagen [365]: ").strip() or "365"
        
        cmd = [
            sys.executable, 'scripts/run_threshold_analysis.py',
            '--asset-id', asset_id,
            '--methods', methods,
            '--horizons', horizons,
            '--lookback-days', lookback,
            '--output-dir', '_validation/threshold_analysis',
            '--apply-results',
            '--per-node'  # REASON: Genereer per-node output voor makkelijkere debugging
        ]
        
        print(f"\n🔄 Start threshold analyse...")
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


def run_combination_alpha_analysis():
    """Combination Alpha Analysis"""
    print("\n" + "="*60)
    print("🎲 COMBINATION ALPHA ANALYSIS (OR, Sens/Spec)")
    print("="*60 + "\n")
    
    print("Analyseert de voorspellende waarde van elke signaalcombinatie.")
    print("Classificeert als: Golden Rule | Promising | Noise")
    print()
    print("Selecteer asset:")
    print("  1. Bitcoin (Asset ID 1) [Default]")
    print("  2. Ander Asset ID invoeren")
    print("  0. Terug")
    
    choice = input("\nKeuze [1]: ").strip() or "1"
    if choice == '0':
        return
    
    asset_id = "1" if choice == '1' else input("Asset ID: ").strip()
    if not asset_id:
        print("⚠️ Geen geldig asset ID")
        return
    
    print("\nTarget type:")
    print("  1. Bullish")
    print("  2. Bearish")
    print("  3. Significant")
    print("  4. Alle targets [Default]")
    
    target_choice = input("\nKeuze [4]: ").strip() or "4"
    
    target_map = {'1': 'bullish', '2': 'bearish', '3': 'significant', '4': 'all'}
    if target_choice == '4':
        targets = ['bullish', 'bearish', 'significant']
    else:
        targets = [target_map.get(target_choice, 'bullish')]
    
    lookback_input = input("Lookback dagen (leeg = alle data) [all]: ").strip()
    lookback_days = None if not lookback_input or lookback_input.lower() == 'all' else int(lookback_input)
    
    bootstrap = input("Run bootstrap CI? [y/n, default y]: ").strip().lower() != 'n'
    save_db = input("Opslaan in database? [y/n, default y]: ").strip().lower() != 'n'
    
    for target in targets:
        print(f"\n🔄 Start analyse voor target '{target}'...")
        
        cmd = [
            sys.executable, 'scripts/run_combination_analysis.py',
            '--asset-id', asset_id,
            '--target', target,
            '--min-samples', '30'
        ]
        
        if lookback_days is not None:
            cmd.extend(['--lookback-days', str(lookback_days)])
        if not bootstrap:
            cmd.append('--no-bootstrap')
        if save_db:
            cmd.append('--save-db')
        cmd.append('--save-json')
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    input("\nDruk op Enter om terug te gaan...")


def run_cpt_generation_v31():
    """Menu optie 11: v3.1 CPT Generation."""
    print("\n" + "="*60)
    print("🧠 CPT GENERATION (v3.1 - DUAL PREDICTION)")
    print("="*60)
    
    print("""
    v3.4 Direct Sub-Predictions Architecture:
    - ✅ Momentum_Prediction (Leading-based prijsrichting)
    - ✅ Volatility_Regime (Coincident-based volatiliteit)
    - ✅ Exit_Timing (Confirming-based exit timing)
    - ✅ Position_Prediction direct gekoppeld aan MP/VR/ET
    """)
    
    asset_id_str = input("Asset ID [1]: ") or "1"
    asset_id = int(asset_id_str)
    run_id = generate_run_id()
    
    generator = QBNv3CPTGenerator(run_id=run_id)
    
    # Check voor event data
    if not has_event_data(asset_id):
        print("⚠️ Geen event data. Draai eerst Event Window Detection (optie 6).")
        proceed = input("Doorgaan zonder Position nodes? [y/N]: ").lower() == 'y'
        if not proceed:
            return
    
    print(f"\n🔄 Genereer CPTs voor asset {asset_id} (run_id: {run_id})...")
    
    # Generate all CPTs
    results = generator.generate_all_cpts(asset_id=asset_id)
    
    print("\n✅ CPT Generation voltooid:")
    for node, stats in results.items():
        if isinstance(stats, dict) and 'observations' in stats:
            cov = stats.get('coverage', 0)
            print(f"   {node}: {stats['observations']} obs, coverage {cov*100:.1f}%")
        else:
            print(f"   {node}: Voltooid")
            
    input("\nDruk op Enter om terug te gaan...")


def run_position_prediction_cpt_only():
    """Menu optie 13: Generate Position_Prediction CPT alleen."""
    print("\n" + "="*60)
    print("🎯 POSITION_PREDICTION CPT GENERATION")
    print("="*60)
    
    asset_id_str = input("Asset ID [1]: ") or "1"
    asset_id = int(asset_id_str)
    
    # Check prerequisites
    if not has_event_data(asset_id):
        print("❌ Geen event data. Draai eerst Event Window Detection.")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    run_id = generate_run_id()
    
    print(f"\n🔄 Genereer Position_Prediction CPT...")
    
    generator = PositionPredictionGenerator()
    
    # Load training data
    training_data = load_barrier_outcomes(asset_id)
    in_event_data = training_data[training_data['event_id'].notna()].copy()
    
    if in_event_data.empty:
        print("❌ Geen rijen binnen events gevonden voor training.")
        input("\nDruk op Enter om terug te gaan...")
        return

    # Mock/Default values for missing columns if necessary
    if 'current_pnl_atr' not in in_event_data.columns:
        in_event_data['current_pnl_atr'] = 0.0

    # Generate CPT
    cpt = generator.generate_cpt(in_event_data)
    
    # Show metrics
    metrics = generator._metrics
    if metrics:
        print(f"\n📊 Training Metrics:")
        print(f"   Observations: {metrics.observations}")
        print(f"   Target hit rate: {metrics.target_hit_rate*100:.1f}%")
        print(f"   Stoploss hit rate: {metrics.stoploss_hit_rate*100:.1f}%")
        print(f"   Timeout rate: {metrics.timeout_rate*100:.1f}%")
        print(f"   Coverage: {metrics.coverage*100:.1f}%")
    
    # Save
    generator.save_cpt(asset_id, run_id)
    print("✅ Position_Prediction CPT opgeslagen")
    
    input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# CPT Generation
# ==============================================================================
def run_cpt_generation():
    """CPT Generation Submenu"""
    from database.db import get_cursor
    print("\n" + "="*60)
    print("📈 CPT GENERATIE v3 (Trade-Aligned & Alpha-Weighted)")
    print("="*60 + "\n")

    print("Selecteer generatie modus:")
    print("  1. Single Asset (specifiek asset ID)")
    print("  2. Selected Assets (per-asset CPTs voor selected_in_current_run)")
    print("  3. Composite Top X (combineert top X volume assets)")
    print("  4. Composite Global (alle beschikbare data)")
    print("  0. Terug")

    mode = input("\nKeuze: ").strip()
    if mode == '0' or not mode:
        return

    if mode == '1':
        _run_single_asset_cpt()
    elif mode == '2':
        _run_selected_assets_cpt()
    elif mode == '3':
        _run_composite_top_x_cpt()
    elif mode == '4':
        _run_composite_global_cpt()
    else:
        print("⚠️ Ongeldige keuze")

    input("\nDruk op Enter om terug te gaan...")


def run_barrier_benchmarks():
    """Run performance benchmarks voor barrier operations"""
    print("\n" + "="*60)
    print("⚡ BARRIER PERFORMANCE BENCHMARKS")
    print("="*60 + "\n")
    
    print("🔄 Start benchmarks (CPU single, GPU batch, Inference)...")
    subprocess.run([sys.executable, 'scripts/benchmark_barriers.py'], cwd=PROJECT_ROOT)
    input("\nDruk op Enter om terug te gaan...")


def _run_single_asset_cpt():
    """Genereer CPT voor een enkel asset."""
    from database.db import get_cursor
    from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
    from config.network_config import MODEL_VERSION

    print("\n--- Single Asset CPT ---")
    print("  1. Bitcoin (Asset ID 1)")
    print("  2. Ander Asset ID")

    choice = input("\nKeuze: ").strip()
    if choice == '1':
        asset_id = 1
    elif choice == '2':
        asset_id = input("Asset ID: ").strip()
        if not asset_id:
            print("⚠️ Geen geldig asset ID")
            return
        asset_id = int(asset_id)
    else:
        return

    outcome_mode = "barrier"
    lookback_days = _get_lookback_days()

    print(f"\n🔄 Start v3 CPT generatie voor asset {asset_id}...")
    print(f"   Model Versie: {MODEL_VERSION}")
    print(f"   Outcome Mode: {outcome_mode}")

    try:
        generator = QBNv3CPTGenerator()
        cpts = generator.generate_all_cpts(
            asset_id, 
            lookback_days=lookback_days, 
            save_to_db=True
        )
        coverages = [c.get('validation', {}).get('coverage', 0) for c in cpts.values() if isinstance(c, dict)]
        avg_cov = sum(coverages) / len(coverages) if coverages else 0
        print(f"✅ {len(cpts)} nodes gegenereerd (Avg Coverage: {avg_cov:.1%})")
    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()


def _run_selected_assets_cpt():
    """Genereer per-asset CPTs voor alle selected_in_current_run assets."""
    from database.db import get_cursor
    from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
    from config.network_config import MODEL_VERSION

    print("\n--- Selected Assets CPT (per asset) ---")

    with get_cursor() as cur:
        cur.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1 ORDER BY id")
        asset_ids = [row[0] for row in cur.fetchall()]

    if not asset_ids:
        print("⚠️ Geen assets geselecteerd in current run")
        return

    print(f"Gevonden: {len(asset_ids)} assets: {asset_ids[:10]}{'...' if len(asset_ids) > 10 else ''}")

    outcome_mode = "barrier"
    lookback_days = _get_lookback_days()

    print(f"\n🔄 Start v3 CPT generatie voor {len(asset_ids)} assets...")
    print(f"   Model Versie: {MODEL_VERSION}")
    print(f"   Outcome Mode: {outcome_mode}")

    try:
        generator = QBNv3CPTGenerator()
        for i, aid in enumerate(asset_ids):
            print(f"   [{i+1}/{len(asset_ids)}] Training Asset {aid}...", end=" ", flush=True)
            try:
                cpts = generator.generate_all_cpts(
                    aid, 
                    lookback_days=lookback_days, 
                    outcome_mode=outcome_mode,
                    save_to_db=True
                )
                coverages = [c.get('validation', {}).get('coverage', 0) for c in cpts.values() if isinstance(c, dict)]
                avg_cov = sum(coverages) / len(coverages) if coverages else 0
                print(f"✅ {len(cpts)} nodes (Cov: {avg_cov:.1%})")
            except Exception as e:
                print(f"❌ {e}")

        print(f"\n✅ Per-asset CPT generatie voltooid!")
    except Exception as e:
        import traceback
        print(f"❌ Kritieke fout: {e}")
        traceback.print_exc()


def _run_composite_top_x_cpt():
    """Genereer composite CPT van top X volume assets."""
    from database.db import get_cursor
    from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
    from config.network_config import MODEL_VERSION

    print("\n--- Composite Top X CPT ---")

    top_x = input("Aantal top assets (bijv. 10): ").strip()
    if not top_x or not top_x.isdigit():
        print("⚠️ Ongeldige invoer")
        return
    top_x = int(top_x)

    # Haal top X assets op basis van volume
    with get_cursor() as cur:
        cur.execute("""
            SELECT id FROM symbols.symbols
            WHERE selected_in_current_run = 1
            ORDER BY volume_24h_usd DESC NULLS LAST
            LIMIT %s
        """, (top_x,))
        asset_ids = [row[0] for row in cur.fetchall()]

    if not asset_ids:
        print("⚠️ Geen assets gevonden")
        return

    if len(asset_ids) < top_x:
        print(f"⚠️ Slechts {len(asset_ids)} assets beschikbaar (gevraagd: {top_x})")

    print(f"Top {len(asset_ids)} assets geselecteerd: {asset_ids}")

    outcome_mode = "barrier"
    lookback_days = _get_lookback_days()
    scope_key = f"top_{len(asset_ids)}"

    print(f"\n🔄 Start composite CPT generatie voor '{scope_key}'...")
    print(f"   Model Versie: {MODEL_VERSION}")
    print(f"   Outcome Mode: {outcome_mode}")
    print(f"   Assets: {asset_ids}")

    try:
        generator = QBNv3CPTGenerator()
        cpts = generator.generate_composite_cpts(
            scope_key=scope_key,
            asset_ids=asset_ids,
            lookback_days=lookback_days,
            outcome_mode=outcome_mode,
            save_to_db=True
        )
        coverages = [c.get('validation', {}).get('coverage', 0) for c in cpts.values() if isinstance(c, dict)]
        avg_cov = sum(coverages) / len(coverages) if coverages else 0
        print(f"\n✅ Composite CPT '{scope_key}' gegenereerd: {len(cpts)} nodes (Avg Coverage: {avg_cov:.1%})")
    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()


def _run_composite_global_cpt():
    """Genereer composite CPT van ALLE beschikbare assets."""
    from database.db import get_cursor
    from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
    from config.network_config import MODEL_VERSION

    print("\n--- Composite Global CPT ---")

    # Haal alle assets met data op
    with get_cursor() as cur:
        cur.execute("""
            SELECT DISTINCT asset_id FROM kfl.mtf_signals_lead
            ORDER BY asset_id
        """)
        asset_ids = [row[0] for row in cur.fetchall()]

    if not asset_ids:
        print("⚠️ Geen assets gevonden met signal data")
        return

    print(f"Gevonden: {len(asset_ids)} assets met signal data")
    print(f"Assets: {asset_ids[:15]}{'...' if len(asset_ids) > 15 else ''}")

    confirm = input(f"\n⚠️ Dit genereert CPTs over {len(asset_ids)} assets. Dit kan lang duren. Doorgaan? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Geannuleerd")
        return

    outcome_mode = "barrier"
    lookback_days = _get_lookback_days()
    scope_key = "all_assets"

    print(f"\n🔄 Start global composite CPT generatie...")
    print(f"   Model Versie: {MODEL_VERSION}")
    print(f"   Outcome Mode: {outcome_mode}")
    print(f"   Scope: {scope_key}")
    print(f"   Assets: {len(asset_ids)}")

    try:
        generator = QBNv3CPTGenerator()
        cpts = generator.generate_composite_cpts(
            scope_key=scope_key,
            asset_ids=asset_ids,
            lookback_days=lookback_days,
            outcome_mode=outcome_mode,
            save_to_db=True
        )
        coverages = [c.get('validation', {}).get('coverage', 0) for c in cpts.values() if isinstance(c, dict)]
        avg_cov = sum(coverages) / len(coverages) if coverages else 0
        print(f"\n✅ Global CPT '{scope_key}' gegenereerd: {len(cpts)} nodes (Avg Coverage: {avg_cov:.1%})")
    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()


def _get_lookback_days():
    """Helper om lookback dagen te vragen."""
    lookback_input = input("\nLookback dagen (leeg = 'all') ['all']: ").strip().lower()
    return None if not lookback_input or lookback_input == 'all' else int(lookback_input)


def run_cpt_cache_status():
    """Toon CPT cache status"""
    print("\n" + "="*60)
    print("💾 CPT CACHE STATUS")
    print("="*60 + "\n")
    
    try:
        from inference.cpt_generator import ConditionalProbabilityTableGenerator
        
        generator = ConditionalProbabilityTableGenerator()
        status = generator.get_cpt_cache_status()
        
        if status.get('error'):
            print(f"❌ Fout: {status['error']}")
        elif status['total_assets'] == 0:
            print("📭 Geen CPT's in cache")
        else:
            print(f"📊 {status['total_assets']} assets in cache:\n")
            print(f"   {'Asset':<10} {'Nodes':<8} {'Obs':<12} {'Gegenereerd'}")
            print("   " + "-"*55)
            
            for asset in status['assets']:
                generated = asset['newest'][:19] if asset['newest'] else 'onbekend'
                obs = asset['total_observations'] or 0
                print(f"   {asset['asset_id']:<10} {asset['node_count']:<8} {obs:<12} {generated}")
        
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


def run_cpt_health_report():
    """CPT Health Report"""
    print("\n" + "="*60)
    print("🛡️ CPT HEALTH REPORT")
    print("="*60 + "\n")
    
    try:
        from database.db import get_cursor
        
        with get_cursor() as cur:
            # REASON: Gebruik DISTINCT ON om alleen de LAATSTE CPT per node te tonen
            # EXPL: Voorkomt verwarring door oude/lege cache entries in het rapport.
            cur.execute("""
                SELECT DISTINCT ON (asset_id, node_name) 
                    asset_id, node_name, coverage, entropy, info_gain, 
                    stability_score, semantic_score, observations, generated_at
                FROM qbn.cpt_cache
                ORDER BY asset_id, node_name, generated_at DESC
            """)
            rows = cur.fetchall()
            
        if not rows:
            print("📭 Geen CPT's in cache")
        else:
            print(f"{'Asset':<6} {'Node':<25} {'Cov':>5} {'Entr':>5} {'Gain':>5} {'Stab':>5} {'Sem':>5} {'Obs':>8}")
            print("-" * 75)
            
            for aid, node, cov, entr, gain, stab, sem, obs in rows:
                f_cov = f"{cov:.0%}" if cov else "-"
                f_entr = f"{entr:.1f}" if entr else "-"
                f_gain = f"{gain:.2f}" if gain else "-"
                f_stab = f"{stab:.2f}" if stab else "-"
                f_sem = f"{sem:.2f}" if sem else "-"
                f_obs = f"{obs:,}" if obs else "0"
                
                status = "✅"
                if cov and cov < 0.5: status = "⚠️"
                if stab and stab < 0.7: status = "🔴"
                
                print(f"{aid:<6} {node:<25} {f_cov:>5} {f_entr:>5} {f_gain:>5} {f_stab:>5} {f_sem:>5} {f_obs:>8} {status}")
            
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Utilities
# ==============================================================================
def run_sync_config():
    """Sync thresholds naar KFL Backend"""
    print("\n" + "="*60)
    print("🔄 SYNC CONFIG NAAR KFL BACKEND")
    print("="*60 + "\n")
    
    try:
        from database.db import get_cursor
        import yaml
        
        kfl_config_path = PROJECT_ROOT / "kfl_backend_config" / "discretization.yaml"
        
        print(f"📄 Doel: {kfl_config_path}")
        print()
        
        # Haal thresholds uit database
        with get_cursor() as cur:
            cur.execute("""
                SELECT asset_id, horizon, param_name, param_value, updated_at
                FROM qbn.composite_threshold_config
                WHERE config_type LIKE '%composite%'
                ORDER BY asset_id, horizon, param_name
            """)
            rows = cur.fetchall()
        
        if not rows:
            print("⚠️ Geen thresholds in database")
            input("\nDruk op Enter om terug te gaan...")
            return
            
        print(f"📊 {len(rows)} threshold entries gevonden in database\n")
        
        # Toon preview
        print(f"{'Asset':<6} {'Horizon':<8} {'Param':<20} {'Value':>10}")
        print("-" * 50)
        for aid, horizon, param, value, updated in rows:
            print(f"{aid:<6} {horizon:<8} {param:<20} {float(value):>10.3f}")
        
        confirm = input("\n\nDeze config naar YAML exporteren? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Geannuleerd")
            input("\nDruk op Enter om terug te gaan...")
            return
            
        # Build YAML structure
        config = {'thresholds': {}}
        for aid, horizon, param, value, _ in rows:
            key = f"asset_{aid}"
            if key not in config['thresholds']:
                config['thresholds'][key] = {}
            if horizon not in config['thresholds'][key]:
                config['thresholds'][key][horizon] = {}
            config['thresholds'][key][horizon][param] = float(value)
        
        # Write YAML
        kfl_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(kfl_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"\n✅ Config geschreven naar {kfl_config_path}")
        
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


def run_full_training_v31():
    """Volledige v3.2 training run (delta-based position management)."""
    print("\n" + "="*60)
    print("🚀 START VOLLEDIGE TRAINING RUN (v3.2)")
    print("="*60 + "\n")
    
    asset_id_str = input("Asset ID [1]: ") or "1"
    asset_id = int(asset_id_str)
    
    print("\n--- Barrier Backfill Opties ---")
    print("  1. Resumeren vanaf checkpoint (alleen ontbrekende data) [default]")
    print("  2. Volledige herberekening (negeer checkpoint, overschrijf alles)")
    backfill_choice = input("\nKeuze [1]: ").strip() or "1"
    
    run_id = generate_run_id()
    
    backfill_cmd = [sys.executable, 'scripts/barrier_backfill.py', '--asset-id', str(asset_id), '--batch-size', '100000', '--run-id', run_id]
    if backfill_choice == "2":
        backfill_cmd.extend(['--no-resume', '--overwrite'])
        backfill_mode_str = "Volledige herberekening (No-Resume, Overwrite)"
    else:
        backfill_mode_str = "Resumeren vanaf checkpoint"
    
    print(f"\n📋 Configuratie:")
    print(f"   - Asset ID:     {asset_id}")
    print(f"   - Run ID:       {run_id}")
    print(f"   - Backfill:     {backfill_mode_str}")
    print(f"   - v3.1 stappen: Alpha Analysis, Threshold Opt, Event Detection, CPT Generation (Dual)")
    print("\n🔄 Starten over 3 seconden... (Ctrl+C om te annuleren)")
    time.sleep(3)
    
    # REASON: De volgorde is nu aangepast om Thresholds EERST te optimaliseren, 
    # en daarna pas de outcomes te genereren die daarop gebaseerd zijn.
    # 1. Threshold Optimalisatie
    # 2. Data Fundament (Signal Backfill via Docker Exec -> KFL GPU)
    # 3. Outcome Backfill (met nieuwe signalen)
    # 4. Leading Tuning (Alpha)
    # 5. Context (Event Windows)
    # 6. Training (CPT Generation)
    
    # --- Stap 1: Thresholds ---
    steps = [
        {
            'name': 'Stap 1: Composite Threshold Optimalisatie (Leading)',
            'cmd': [sys.executable, 'scripts/run_threshold_analysis.py', '--asset-id', str(asset_id), '--targets', 'leading', '--apply-results', '--run-id', run_id]
        }
    ]
    
    # --- Stap 2: Signal Backfill (Cross-Container) ---
    # Vraag gebruiker om signal backfill als we in een interactieve sessie zitten
    # Voor nu nemen we aan dat we dit altijd willen als onderdeel van de full run
    
    # Check of we in Docker zitten en socket hebben
    in_docker = os.path.exists('/.dockerenv')
    has_socket = os.path.exists('/var/run/docker.sock')
    # Check of de docker command beschikbaar is
    has_docker_cmd = shutil.which('docker') is not None
    
    if in_docker and has_socket and has_docker_cmd:
        # Docker-in-Docker orchestratie
        steps.append({
            'name': 'Stap 2: KFL GPU Signal Backfill (Recalculate)',
            'cmd': [
                'docker', 'exec', 
                '-e', 'PYTHONPATH=/app/src',  # REASON: Module pad fix voor KFL GPU container
                'KFL_backend_GPU_v5_3', 
                'python', '-m', 'backfill.cli',
                '--asset_id', str(asset_id),
                '--mode-indicators', 'gaps_only',
                '--mode-signals', 'full',
                '--intervals', '1,60,240,D' # Standaard intervallen
            ]
        })
    else:
        # Fallback of waarschuwing
        print("\n⚠️  Geen toegang tot Docker socket. Kan KFL GPU container niet aansturen.")
        print("   Sla Stap 2 (Signal Backfill) over. Zorg dat signalen handmatig zijn bijgewerkt!")
        time.sleep(2)

    steps.extend([
        # --- Fase 3: Data Fundament (Outcomes) ---
        {
            'name': 'Stap 3: Barrier Outcome Backfill (met nieuwe thresholds)',
            # REASON: Gebruik modern barrier_backfill.py script (v3.1+)
            # --no-resume en --overwrite dwingen een herberekening af met de nieuwe thresholds uit Stap 1
            'cmd': [
                sys.executable, 'scripts/barrier_backfill.py', 
                '--asset-id', str(asset_id), 
                '--batch-size', '100000',
                '--no-resume', 
                '--overwrite',
                '--run-id', run_id
            ]
        },
        # Stap 3.1 (Raw Check) is nu redundant omdat Stap 3 al de volledige backfill doet
        # We halen hem weg om tijd te besparen en verwarring te voorkomen.

        {
            'name': 'Stap 3.5: Materialize Leading Scores (Update weights)',
            'cmd': [sys.executable, 'scripts/materialize_leading_scores.py', '--asset-id', str(asset_id), '--overwrite', '--run-id', run_id]
        },
        
        {
            'name': 'Stap 4: Compute IDA Training Weights',
            'cmd': [sys.executable, 'scripts/compute_barrier_weights.py', '--asset-id', str(asset_id), '--config', 'baseline', '--run-id', run_id]
        },
        
        # --- Fase 4: Leading Tuning ---
        {
            'name': 'Stap 6: Entry Hypothesis Alpha Analysis',
            'cmd': [sys.executable, 'alpha-analysis/analyze_signal_alpha.py', '--asset', str(asset_id), '--layer', 'HYPOTHESIS', '--run-id', run_id]
        },
        {
            'name': 'Stap 8: Combination Alpha Analysis',
            'cmd': [sys.executable, 'scripts/run_combination_analysis.py', '--asset-id', str(asset_id), '--all-targets', '--save-db', '--run-id', run_id]
        },

        # --- Fase 5: Event Context ---
        {
            'name': 'Stap 9: Event Window Detection (v3.2 - met delta scores)',
            # REASON: We roepen een script aan dat de drempels uit de DB haalt (via de helper in training_menu)
            # v3.2: EventWindowDetector berekent nu ook delta scores en uniqueness weights
            'cmd': [sys.executable, '-c', 
                   f"from menus.training_menu import load_barrier_outcomes, run_event_window_detection_logic; "
                   f"run_event_window_detection_logic({asset_id}, auto_save=True)"]
        },
        
        # --- Fase 5.5: Position Delta Thresholds (v3.2 NIEUW) ---
        {
            'name': 'Stap 9.5: Position Delta Threshold Optimalisatie (v3.2)',
            # REASON: Optimaliseer delta thresholds voor Position_Confidence training
            # Geen --lookback = gebruik alle beschikbare event data
            'cmd': [sys.executable, 'scripts/run_position_delta_threshold_analysis.py', 
                   '--asset-id', str(asset_id)]
        },

        # --- Fase 6: Training ---
        {
            'name': 'Stap 10: CPT Generation (v3.2 - delta-based)',
            # REASON: v3.2 CPT generator gebruikt delta scores en uniqueness weighting
            'cmd': [sys.executable, '-c', f"from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator; gen = QBNv3CPTGenerator(run_id='{run_id}'); gen.generate_all_cpts({asset_id})"]
        }
    ])
    
    all_issues = []
    
    for step in steps:
        print(f"\n--- ⏳ Uitvoeren: {step['name']} ---")
        try:
            process = subprocess.Popen(
                step['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                print(clean_line)
                
                upper_line = clean_line.upper()
                if 'ERROR' in upper_line or 'EXCEPTION' in upper_line:
                    all_issues.append(f"[{step['name']}] {clean_line}")

            process.wait()

            if process.returncode != 0:
                msg = f"❌ {step['name']} gefaald met returncode {process.returncode}"
                print(msg)
                all_issues.append(msg)
            else:
                print(f"✅ {step['name']} succesvol voltooid")
                
        except Exception as e:
            msg = f"❌ Fout bij uitvoeren van {step['name']}: {e}"
            print(msg)
            all_issues.append(msg)
            
    print("\n" + "="*60)
    print("🏁 TRAINING RUN SAMENVATTING")
    print(f"   Run ID: {run_id}")
    print("="*60)
    
    if not all_issues:
        print("\n✅ Geen kritieke errors gevonden!")
    else:
        print(f"\n⚠️  Er zijn {len(all_issues)} meldingen gevonden:\n")
        for issue in all_issues:
            print(f"  - {issue}")
            
    print("\n" + "="*60)
    input("\nDruk op Enter om terug te gaan naar het menu...")


# ==============================================================================
# v3.1 Helpers
# ==============================================================================

def has_event_data(asset_id: int) -> bool:
    """Check of er event data is voor asset."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM qbn.barrier_outcomes 
            WHERE asset_id = %s AND event_id IS NOT NULL
        """, (asset_id,))
        return cur.fetchone()[0] > 0

def has_position_confidence_cpt(asset_id: int) -> bool:
    """Check of Position_Confidence CPT bestaat."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM qbn.cpt_cache 
            WHERE scope_key = %s AND node_name = 'Position_Confidence'
              AND model_version >= '3.1'
        """, (f"asset_{asset_id}",))
        return cur.fetchone()[0] > 0

def generate_run_id() -> str:
    """Genereer unieke run ID."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def save_event_labels_to_db(asset_id: int, labeled_data: pd.DataFrame):
    """Sla event labels op naar qbn.barrier_outcomes."""
    from database.db import insert_many
    
    # We updaten alleen de event_id kolom. In PostgreSQL doen we dit meestal via een temp table of batch update.
    # Hier gebruiken we een simpele loop voor demonstratie of een meer efficiente methode als beschikbaar.
    # Gegeven de context van QBNv3, doen we een batch update.
    
    updates = []
    for _, row in labeled_data[labeled_data['event_id'].notna()].iterrows():
        updates.append((row['event_id'], asset_id, row['time_1']))
        
    if not updates:
        return

    with get_cursor(commit=True) as cur:
        # Batch update via temp table
        cur.execute("CREATE TEMP TABLE temp_event_labels (event_id VARCHAR(32), asset_id INT, time_1 TIMESTAMPTZ)")
        from psycopg2.extras import execute_values
        execute_values(cur, "INSERT INTO temp_event_labels VALUES %s", updates)
        
        cur.execute("""
            UPDATE qbn.barrier_outcomes bo
            SET event_id = t.event_id
            FROM temp_event_labels t
            WHERE bo.asset_id = t.asset_id AND bo.time_1 = t.time_1
        """)
    logger.info(f"✅ {len(updates)} event labels opgeslagen")

def load_barrier_outcomes(asset_id: int) -> pd.DataFrame:
    """Laad barrier outcomes voor event detection."""
    with get_cursor() as cur:
        query = "SELECT * FROM qbn.barrier_outcomes WHERE asset_id = %s ORDER BY time_1"
        cur.execute(query, (asset_id,))
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)

# ==============================================================================
# Main Handler
# ==============================================================================
def handle_choice(choice: str):
    """Dispatch voor training container v3.2"""
    handlers = {
        '1': show_database_stats,
        '2': run_gpu_benchmark,
        '3': run_barrier_backfill,
        '4': run_ida_weights,
        '5': run_outcome_status,
        '6': lambda: run_alpha_analysis('HYPOTHESIS'),
        '7': run_threshold_analysis,
        '8': run_combination_alpha_analysis,
        '9': run_event_window_detection,
        '10': run_cpt_generation_v31,
        '11': run_cpt_cache_status,
        '12': run_position_prediction_cpt_only,
        '13': run_full_training_v31,
        '14': run_archive_reports,
        '15': run_sync_config,
        '16': run_position_delta_threshold_analysis,  # v3.2 NIEUW
    }
    
    if choice in handlers:
        handlers[choice]()
        return True
    elif choice == '99':
        return True
    elif choice == '0':
        return False
    else:
        print("\n⚠️  Ongeldige keuze")
        time.sleep(1)
        return True


def run():
    """Main loop voor training menu"""
    from menus.shared import wait_for_database
    
    if not wait_for_database():
        print("⚠️  Doorgaan zonder database connectie...")

    # REASON: Startup check voor threshold config
    try:
        from config.threshold_loader import ThresholdLoader
        from core.config_warnings import warn_fallback_active
        
        # Check Asset 1 als baseline
        status = ThresholdLoader.check_database_availability(asset_id=1)
        if not status.get('available') or status.get('status') == 'incomplete':
            warn_fallback_active(
                component="TrainingContainer",
                config_name="global_threshold_config",
                fallback_values={'status': status.get('status', 'unknown')},
                reason="Geen of incomplete threshold configuratie in database",
                fix_command="Draai 'Threshold Optimalisatie' (Optie 7) voor actieve assets"
            )
    except Exception as e:
        logger.error(f"Fout bij startup config check: {e}")

    # REASON: Pre-flight state vocabulary check om CPT/state mismatches vroeg te detecteren.
    # EXPL: Draait als apart script zodat logging in _log/ wordt afgedwongen.
    try:
        subprocess.run(
            [sys.executable, "scripts/check_state_vocab.py"],
            cwd=PROJECT_ROOT,
            check=False,
        )
    except Exception as e:
        logger.error(f"Fout bij state vocab preflight check: {e}")

    while True:
        clear_screen()
        print_header('training')
        choice = show_menu()
        
        if not handle_choice(choice):
            print("\n👋 Tot ziens!\n")
            break


if __name__ == '__main__':
    run()
