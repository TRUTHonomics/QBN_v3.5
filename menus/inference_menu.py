#!/usr/bin/env python3
"""
Inference Menu voor QBN v3.1
============================
Container: QBN_v3.1 (ROLE=inference)
Doel: Real-time inference en monitoring
"""

import sys
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from menus.shared import (
    clear_screen, print_header, show_database_stats, run_gpu_benchmark
)


def show_menu():
    """Toon inference menu - 8 opties"""
    print("--- Categorie 1: System Status ---")
    print("  1.  📊 Database Connectie Check")
    print("  2.  🚀 GPU Status & Memory")
    print()
    print("--- Categorie 2: Inference Operations ---")
    print("  3.  🔮 Single Inference Test (1 asset)")
    print("  4.  🎯 Single Inference Test (with Position)")
    print("  5.  🔮 Batch Inference Test (all assets)")
    print("  6.  ⚡ Latency Profiling")
    print()
    print("--- Categorie 3: Real-time Service ---")
    print("  7.  ▶️  Start Inference Loop")
    print("  8.  ⏹️  Stop Inference Loop")
    print("  9.  📈 View Loop Status")
    print()
    print("  99. 🔄 Refresh status")
    print("  0.  Exit")
    print()
    return input("Keuze: ").strip()


# ==============================================================================
# Inference Operations
# ==============================================================================
def run_single_inference_test():
    """Test inference op 1 asset"""
    print("\n" + "="*60)
    print("🔮 SINGLE INFERENCE TEST")
    print("="*60 + "\n")

    try:
        from inference.inference_loader import InferenceLoader
        
        asset_id = input("Asset ID [1]: ").strip() or "1"
        asset_id = int(asset_id)

        print(f"\n🔄 Initialiseer Inference Engine voor asset {asset_id}...")
        loader = InferenceLoader()
        engine = loader.load_inference_engine(asset_id)
        print("   ✅ Engine klaar")

        print(f"\n🔄 Haal huidige evidence op...")
        evidence = loader.load_current_evidence(asset_id)
        print(f"   ✅ Evidence geladen (tijd: {evidence.timestamp})")

        print("\n🔄 Run inference...")
        start = time.time()
        result = engine.infer(evidence)
        elapsed = (time.time() - start) * 1000

        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS - Asset {asset_id}")
        print(f"Regime: {result.regime.upper()}")
        print(f"Model:  {getattr(result, 'model_version', '3.2')}")
        print('='*50)

        for horizon in ['1h', '4h', '1d']:
            p = result.predictions.get(horizon)
            if p is None:
                continue
            
            print(f"\n{horizon.upper()}:")
            if hasattr(p, 'p_up_strong'):  # BarrierPrediction
                print(f"   Mode:             BARRIER")
                print(f"   Direction:        {p.expected_direction}")
                print(f"   Confidence:       {p.directional_confidence:+.2%}")
                print(f"   Win Rate:         {p.estimated_win_rate:.1%}")
                print(f"   Dist (U/N/D):     {p.p_up_strong+p.p_up_weak:.1%} / {p.p_neutral:.1%} / {p.p_down_strong+p.p_down_weak:.1%}")
            else:  # Legacy
                prob = getattr(result, 'confidences', {}).get(horizon, 0)
                expected_atr = getattr(result, 'expected_atr_moves', {}).get(horizon, 0)
                print(f"   Mode:             LEGACY")
                print(f"   State:            {p}")
                print(f"   Probability:      {prob:.1%}")
                print(f"   Expected ATR:     {expected_atr:+.2f}")

        print(f"\nInference time: {elapsed:.1f}ms (Target: <25ms)")

    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()

    input("\nDruk op Enter om terug te gaan...")


def run_single_inference_with_position():
    """Test inference met position context (v3.1)."""
    print("\n" + "="*60)
    print("🎯 SINGLE INFERENCE TEST (WITH POSITION)")
    print("="*60 + "\n")

    try:
        from inference.inference_loader import InferenceLoader
        from inference.trade_aligned_inference import PositionContext
        from datetime import datetime, timezone, timedelta
        
        asset_id = int(input("Asset ID [1]: ").strip() or "1")
        
        # Load engine
        print(f"\n🔄 Initialiseer Inference Engine...")
        loader = InferenceLoader()
        engine = loader.load_inference_engine(asset_id)
        
        # Get current evidence
        print(f"🔄 Haal huidige evidence op...")
        evidence = loader.load_current_evidence(asset_id)
        
        # Simulate position context
        print("\n--- Position Context (Simulatie) ---")
        direction = input("Direction [LONG/short]: ").strip().upper() or "LONG"
        time_since_entry = int(input("Minuten sinds entry [60]: ").strip() or "60")
        current_pnl_atr = float(input("Huidige PnL (ATR) [0.5]: ").strip() or "0.5")
        
        context = PositionContext(
            position_id="TEST_MIGRATION",
            direction=direction,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=time_since_entry),
            entry_price=95000, # Dummy
            current_price=95000 * (1.02 if direction == 'LONG' else 0.98), # Dummy
            atr_at_entry=500 # Dummy
        )
        
        # Override computed pnl to use user input for test
        context._pnl_override = current_pnl_atr
        # Note: We overwrite the property behavior for the test
        context.__class__.current_pnl_atr = property(lambda self: getattr(self, '_pnl_override', 0))

        print("\n🔄 Run Dual-Inference...")
        start = time.time()
        result = engine.run_inference(evidence, context)
        elapsed = (time.time() - start) * 1000

        print(f"\n{'='*50}")
        print(f"DUAL PREDICTION RESULTS - Asset {asset_id}")
        print(f"Regime: {result.regime.upper()}")
        print(f"Hypothesis: {result.trade_hypothesis}")
        print(f"Position Confidence: {result.position_confidence}")
        print('='*50)

        print("\nENTRY PREDICTIONS:")
        for horizon in ['1h', '4h', '1d']:
            state = result.entry_predictions.get(horizon)
            conf = result.entry_confidences.get(horizon, 0)
            print(f"   {horizon}: {state:<12} (conf: {conf:.2%})")

        print("\nPOSITION PREDICTION:")
        if result.position_prediction:
            pred = result.position_prediction
            print(f"   Dominant: {pred.dominant_outcome.upper()}")
            print(f"   Confidence: {pred.confidence:.2%}")
            print(f"   Target Hit: {pred.target_hit:.1%}")
            print(f"   SL Hit:     {pred.stoploss_hit:.1%}")
            print(f"   Timeout:    {pred.timeout:.1%}")
        else:
            print("   ⚠️ Geen Position Prediction beschikbaar (check CPT)")

        print(f"\nInference time: {elapsed:.1f}ms")

    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()

    input("\nDruk op Enter om terug te gaan...")




def run_batch_inference_test():
    """Test inference op alle actieve assets"""
    print("\n" + "="*60)
    print("🔮 BATCH INFERENCE TEST")
    print("="*60 + "\n")

    try:
        from database.db import get_cursor
        from inference.inference_loader import InferenceLoader
        
        # Haal actieve assets op
        with get_cursor() as cur:
            cur.execute("""
                SELECT id, bybit_symbol 
                FROM symbols.symbols 
                WHERE selected_in_current_run = 1 
                ORDER BY id
            """)
            assets = cur.fetchall()
        
        if not assets:
            print("⚠️ Geen actieve assets gevonden")
            input("\nDruk op Enter om terug te gaan...")
            return
            
        print(f"🔄 Test inference voor {len(assets)} assets...\n")
        
        loader = InferenceLoader()
        results = []
        
        for asset_id, symbol in assets:
            try:
                engine = loader.load_inference_engine(asset_id)
                evidence = loader.load_current_evidence(asset_id)
                
                start = time.time()
                result = engine.infer(evidence)
                elapsed = (time.time() - start) * 1000
                
                # Bepaal prediction label
                p1h = result.predictions.get('1h')
                if hasattr(p1h, 'expected_direction'):
                    pred_label = p1h.expected_direction
                    mode_label = "BAR"
                else:
                    pred_label = str(p1h)
                    mode_label = "LEG"

                status = "✅" if elapsed < 25 else "⚠️"
                print(f"   {status} {symbol:<12} {pred_label:<10} ({mode_label}) {elapsed:>6.1f}ms")
                results.append((symbol, elapsed, True))
            except Exception as e:
                print(f"   ❌ {symbol:<12} FOUT: {str(e)[:30]}")
                results.append((symbol, 0, False))
        
        # Summary
        successful = [r for r in results if r[2]]
        if successful:
            avg_latency = sum(r[1] for r in successful) / len(successful)
            print(f"\n📊 Gemiddelde latency: {avg_latency:.1f}ms")
            print(f"   Succesvol: {len(successful)}/{len(results)}")

    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        traceback.print_exc()

    input("\nDruk op Enter om terug te gaan...")


def run_latency_profiling():
    """Meet inference latency"""
    print("\n" + "="*60)
    print("⚡ INFERENCE LATENCY PROFILING (Target: <25ms)")
    print("="*60 + "\n")
    
    try:
        asset_id = input("Asset ID [1]: ").strip() or "1"
        iterations = input("Aantal iteraties [1000]: ").strip() or "1000"
            
        print(f"🔄 Start profiling voor asset {asset_id}...")
        
        cmd = [
            sys.executable, 'scripts/profile_inference.py',
            '--asset-id', asset_id,
            '--iterations', iterations
        ]
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
            
    except Exception as e:
        print(f"❌ Fout bij uitvoeren profiling: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Real-time Service
# ==============================================================================
def start_inference_loop():
    """Start de asynchrone real-time inference loop"""
    print("\n" + "="*60)
    print("▶️  START REAL-TIME INFERENCE LOOP")
    print("="*60 + "\n")
    
    print("Dit start de asynchrone service die luistert naar signal updates.")
    print("De loop blijft draaien tot je op Ctrl+C drukt.\n")
    
    try:
        cmd = [sys.executable, 'scripts/run_inference_loop.py']
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\n👋 Loop gestopt.")
    except Exception as e:
        print(f"❌ Fout bij starten loop: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


def stop_inference_loop():
    """Stop de inference loop gracefully"""
    print("\n" + "="*60)
    print("⏹️  STOP INFERENCE LOOP")
    print("="*60 + "\n")
    
    # REASON: Stuur SIGTERM naar het inference loop process
    try:
        import signal
        
        pid_file = PROJECT_ROOT / '.inference_loop.pid'
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            print(f"🔄 Stopping inference loop (PID: {pid})...")
            os.kill(pid, signal.SIGTERM)
            print("✅ Stop signaal verzonden")
            pid_file.unlink()
        else:
            print("⚠️ Geen actieve inference loop gevonden (.inference_loop.pid niet gevonden)")
            print("   Tip: Als de loop nog draait, gebruik Ctrl+C in die terminal")
    except ProcessLookupError:
        print("⚠️ Process niet gevonden (mogelijk al gestopt)")
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


def view_loop_status():
    """Toon status van de inference loop"""
    print("\n" + "="*60)
    print("📈 INFERENCE LOOP STATUS")
    print("="*60 + "\n")
    
    try:
        from database.db import get_cursor
        
        # Check laatste predictions
        with get_cursor() as cur:
            cur.execute("""
                SELECT 
                    outcome_mode,
                    COUNT(*) as total,
                    MAX(time) as last_prediction,
                    COUNT(DISTINCT asset_id) as unique_assets
                FROM qbn.output_entry
                WHERE time > NOW() - INTERVAL '1 hour'
                GROUP BY outcome_mode
            """)
            rows = cur.fetchall()
            
            print("📊 Laatste uur per mode:")
            if not rows:
                print("   Geen data gevonden")
            else:
                for mode, total, last, unique in rows:
                    print(f"   {mode:<12}: {total:>6,} rows, {unique:>3} assets (Laatste: {last})")
        
        # Check PID file
        pid_file = PROJECT_ROOT / '.inference_loop.pid'
        if pid_file.exists():
            pid = pid_file.read_text().strip()
            print(f"\n🟢 Loop status: ACTIEF (PID: {pid})")
        else:
            print(f"\n🔴 Loop status: NIET ACTIEF")
            
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Main Handler
# ==============================================================================
def handle_choice(choice: str):
    """Dispatch voor inference container"""
    handlers = {
        '1': show_database_stats,
        '2': run_gpu_benchmark,
        '3': run_single_inference_test,
        '4': run_single_inference_with_position,
        '5': run_batch_inference_test,
        '6': run_latency_profiling,
        '7': start_inference_loop,
        '8': stop_inference_loop,
        '9': view_loop_status,
    }
    
    if choice in handlers:
        handlers[choice]()
        return True
    elif choice == '99':
        return True  # Refresh
    elif choice == '0':
        return False  # Exit
    else:
        print("\n⚠️  Ongeldige keuze")
        time.sleep(1)
        return True


def run():
    """Main loop voor inference menu"""
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
                component="InferenceContainer",
                config_name="global_threshold_config",
                fallback_values={'status': status.get('status', 'unknown')},
                reason="Geen of incomplete threshold configuratie in database",
                fix_command="Draai 'Threshold Optimalisatie' in de Training Container"
            )
    except Exception as e:
        # Geen logging module import hier, gebruik print of importeer
        print(f"⚠️ Fout bij startup config check: {e}")

    # REASON: Pre-flight state vocabulary check om CPT/state mismatches vroeg te detecteren.
    # EXPL: Draait als apart script zodat logging in _log/ wordt afgedwongen.
    try:
        subprocess.run(
            [sys.executable, "scripts/check_state_vocab.py"],
            cwd=PROJECT_ROOT,
            check=False,
        )
    except Exception as e:
        print(f"⚠️ Fout bij state vocab preflight check: {e}")

    while True:
        clear_screen()
        print_header('inference')
        choice = show_menu()
        
        if not handle_choice(choice):
            print("\n👋 Tot ziens!\n")
            break


if __name__ == '__main__':
    from core.logging_utils import setup_logging
    setup_logging("inference_menu")
    run()
