#!/usr/bin/env python3
"""
Gedeelde functies voor QBN menu's
=================================
Bevat: database checks, GPU checks, utility functies
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def check_gpu_status() -> str:
    """Check GPU beschikbaarheid"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"✅ {name} ({mem:.1f} GB)"
        else:
            return "⚠️  Geen GPU - CPU mode"
    except ImportError:
        return "❌ PyTorch niet geïnstalleerd"
    except Exception as e:
        return f"❌ Fout: {e}"


def check_database_status() -> str:
    """Check database connectie"""
    try:
        from database.db import get_cursor
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            return "✅ Verbonden met KFLhyper"
    except ImportError:
        return "❌ Database module niet gevonden"
    except Exception as e:
        return f"❌ {str(e)[:40]}"


def wait_for_database():
    """Wacht tot database beschikbaar is"""
    print("🔍 Wacht op PostgreSQL...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            from database.db import get_cursor
            with get_cursor() as cur:
                cur.execute("SELECT 1")
                print("✅ Database verbonden!\n")
                return True
        except Exception as e:
            print(f"   Poging {attempt+1}/{max_attempts}: {e}")
            time.sleep(2)
    
    print("❌ Database niet bereikbaar na 60 seconden\n")
    return False


def print_header(role: str):
    """Print menu header met status"""
    gpu_status = check_gpu_status()
    db_status = check_database_status()
    
    role_names = {
        'inference': 'INFERENCE',
        'training': 'TRAINING',
        'validation': 'VALIDATION'
    }
    
    print("="*60)
    print(f"     QUANTBAYES NEXUS v3 - {role_names.get(role, role.upper())}")
    print("="*60)
    print()
    print("[STATUS]")
    print(f"   Database: {db_status}")
    print(f"   GPU:      {gpu_status}")
    print()


# ==============================================================================
# Database Statistieken
# ==============================================================================
def show_database_stats(save_to_file=False, output_dir=None):
    """Toon database statistieken"""
    if not save_to_file:
        print("\n" + "="*60)
        print("DATABASE STATISTIEKEN")
        print("="*60 + "\n")
    
    try:
        from database.db import get_cursor
        
        stats = [
            ("kfl.mtf_signals_current_lead", "SELECT COUNT(*) FROM kfl.mtf_signals_current_lead"),
            ("kfl.mtf_signals_lead (historical)", "SELECT COUNT(*) FROM kfl.mtf_signals_lead"),
            ("qbn.barrier_outcomes (First-Touch)", "SELECT COUNT(*) FROM qbn.barrier_outcomes"),
            ("qbn.cpt_cache (BN Brain)", "SELECT COUNT(*) FROM qbn.cpt_cache"),
            ("qbn.output_entry (Predictions)", "SELECT COUNT(*) FROM qbn.output_entry"),
        ]
        
        report_lines = []
        with get_cursor() as cur:
            header = f"{'Tabel':<45} {'Rijen':>15}"
            if not save_to_file:
                print(header)
                print("-"*60)
            report_lines.append(header)
            report_lines.append("-" * 60)
            
            for table_name, query in stats:
                try:
                    cur.execute(query)
                    count = cur.fetchone()[0]
                    line = f"{table_name:<45} {count:>15,}"
                    if not save_to_file: print(line)
                    report_lines.append(line)
                except Exception as e:
                    line = f"{table_name:<45} {'FOUT: ' + str(e)[:20]:>15}"
                    if not save_to_file: print(line)
                    report_lines.append(line)
        
        # Laatste MTF signal timestamp
        report_lines.append("\n" + "-"*60)
        with get_cursor() as cur:
            cur.execute("SELECT MAX(time_1) FROM kfl.mtf_signals_current_lead")
            last_time = cur.fetchone()[0]
            if last_time:
                line = f"Laatste MTF signal: {last_time}"
                if not save_to_file: print(line)
                report_lines.append(line)

        # Outcome coverage (Barrier-based)
        with get_cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(first_significant_barrier) as with_hits,
                    COUNT(training_weight) as with_weights,
                    AVG(training_weight) as avg_uniqueness
                FROM qbn.barrier_outcomes
            """)
            row = cur.fetchone()
            if row and row[0] > 0:
                l1 = f"Barrier Hits coverage: {100*row[1]/row[0]:.1f}%"
                l2 = f"Avg Uniqueness (IDA): {row[3]:.3f} (N_eff={row[2]:,})"
                if not save_to_file:
                    print(l1)
                    print(l2)
                report_lines.append(l1)
                report_lines.append(l2)
        
        if save_to_file:
            target_dir = output_dir or (PROJECT_ROOT / '_validation')
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = target_dir / f"db_stats_{timestamp}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Database Statistics Report\n\n")
                f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
                f.write("```text\n")
                f.write("\n".join(report_lines))
                f.write("\n```\n")
            return filename
            
    except Exception as e:
        print(f"❌ Fout bij ophalen stats: {e}")
    
    if not save_to_file:
        input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# GPU Benchmark
# ==============================================================================
def run_gpu_benchmark(save_to_file=False, output_dir=None):
    """GPU performance benchmark"""
    if not save_to_file:
        print("\n" + "="*60)
        print("GPU PERFORMANCE BENCHMARK")
        print("="*60 + "\n")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            msg = "❌ Geen GPU beschikbaar"
            if not save_to_file:
                print(msg)
                input("\nDruk op Enter om terug te gaan...")
            return msg
        
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        
        report_lines = []
        l1 = f"GPU: {props.name}"
        l2 = f"Memory: {props.total_memory / 1024**3:.1f} GB"
        l3 = f"Compute capability: {props.major}.{props.minor}"
        l4 = f"CUDA cores: {props.multi_processor_count * 128}"
        
        for l in [l1, l2, l3, l4]:
            if not save_to_file: print(l)
            report_lines.append(l)
        
        if not save_to_file: print("\n🔄 Running benchmarks...")
        report_lines.append("\n🔄 Benchmarks:")
        
        # Matrix multiplication benchmark
        sizes = [1000, 2000, 4000, 8000]
        
        header = f"\n{'Matrix Size':<15} {'Time (ms)':>12} {'TFLOPS':>10}"
        if not save_to_file:
            print(header)
            print("-"*40)
        report_lines.append(header)
        report_lines.append("-" * 40)
        
        for size in sizes:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            torch.cuda.synchronize()
            _ = torch.mm(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            iterations = 10
            for _ in range(iterations):
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / iterations * 1000
            
            # Calculate TFLOPS
            flops = 2 * size**3 / (elapsed / 1000) / 1e12
            
            line = f"{size}x{size:<9} {elapsed:>10.2f}ms {flops:>9.2f}"
            if not save_to_file: print(line)
            report_lines.append(line)
        
        # Memory bandwidth test
        if not save_to_file: print("\n🔄 Memory bandwidth test...")
        size = 100_000_000
        data = torch.randn(size, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = data + 1
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        
        bandwidth = (size * 4 * 2) / elapsed / 1e9
        line = f"Bandwidth: {bandwidth:.1f} GB/s"
        if not save_to_file: print(line)
        report_lines.append("\n" + line)
        
        if save_to_file:
            target_dir = output_dir or (PROJECT_ROOT / '_validation')
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = target_dir / f"gpu_benchmark_{timestamp}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# GPU Performance Benchmark Report\n\n")
                f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
                f.write("```text\n")
                f.write("\n".join(report_lines))
                f.write("\n```\n")
            return filename
            
        print("\n✅ GPU benchmark voltooid!")
        
    except Exception as e:
        import traceback
        print(f"❌ Fout: {e}")
        if not save_to_file: traceback.print_exc()
    
    if not save_to_file:
        input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Archive Reports
# ==============================================================================
def run_archive_reports(silent=False, files_to_archive=None):
    """Archiveert validatierapporten naar een timestamped submap"""
    if not silent:
        print("\n" + "="*60)
        print("📦 ARCHIVE VALIDATION REPORTS")
        print("="*60 + "\n")
    
    import shutil
    
    validation_dir = PROJECT_ROOT / '_validation'
    if not validation_dir.exists():
        if not silent:
            print("📭 Geen validatiemap gevonden.")
            input("\nDruk op Enter om terug te gaan...")
        return
        
    archive_root = validation_dir / 'archive'
    archive_root.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = archive_root / timestamp
    
    if files_to_archive is None:
        reports = list(validation_dir.glob("*.json")) + list(validation_dir.glob("*.md"))
        reports = [f for f in reports if f.is_file()]
    else:
        reports = files_to_archive
    
    if not reports:
        if not silent:
            print("📭 Geen nieuwe rapporten om te archiveren.")
            input("\nDruk op Enter om terug te gaan...")
        return
        
    if not silent:
        print(f"📂 Archiveren van {len(reports)} bestanden naar {target_dir}...")
    
    target_dir.mkdir(exist_ok=True)
    
    count = 0
    for report in reports:
        try:
            shutil.move(str(report), str(target_dir / report.name))
            count += 1
        except Exception as e:
            if not silent:
                print(f"   ❌ Fout bij verplaatsen {report.name}: {e}")
            
    if not silent:
        print(f"\n✅ {count} bestanden gearchiveerd!")
        input("\nDruk op Enter om terug te gaan...")
    else:
        print(f"✅ {count} individuele rapporten gearchiveerd naar {target_dir}")
