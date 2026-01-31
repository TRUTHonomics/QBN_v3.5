#!/usr/bin/env python3
"""
Validation Menu voor QBN v3.1
=============================
Container: QBN_v3.1_Validation (ROLE=validation)
Doel: Model validatie en quality assurance
"""

import sys
import os
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from menus.shared import (
    clear_screen, print_header, show_database_stats, 
    run_gpu_benchmark, run_archive_reports
)

CURRENT_RUN_ID = None  # Set door run_full_validation_cycle (optioneel)


def _db_has_column(schema: str, table: str, column: str) -> bool:
    """Check via information_schema of een kolom bestaat (runtime-safe)."""
    try:
        from database.db import get_cursor
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                  AND column_name = %s
                LIMIT 1
                """,
                (schema, table, column),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def _resolve_latest_run_id(asset_id: int) -> str:
    """
    Probeer de meest recente run_id te bepalen voor dit asset.

    REASON: Validation moet dezelfde dataset evalueren als de training run.
    TODO-verify: als DB schema afwijkt, kan deze fallback een lege string returnen.
    """
    from database.db import get_cursor

    # 1) Prefer CPT cache (meest direct gekoppeld aan training output)
    if _db_has_column("qbn", "cpt_cache", "run_id"):
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT run_id
                FROM qbn.cpt_cache
                WHERE scope_key = %s
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                (f"asset_{asset_id}",),
            )
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])

    # 2) Fallback: barrier_outcomes run_id
    if _db_has_column("qbn", "barrier_outcomes", "run_id"):
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT run_id
                FROM qbn.barrier_outcomes
                WHERE asset_id = %s
                  AND run_id IS NOT NULL
                ORDER BY time_1 DESC
                LIMIT 1
                """,
                (asset_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])

    # 3) Last resort: thresholds run_id-less; return empty sentinel
    return ""


# ==============================================================================
# Reporting & Archiving Helpers
# ==============================================================================
def get_report_dir(asset_id: int, step_num: int, step_name: str) -> Path:
    """
    Maakt en retourneert het pad naar de rapportage map voor een specifieke stap.
    Structuur: _validation/asset_X/N_name/
    """
    aid_str = str(asset_id) if asset_id and int(asset_id) > 0 else "all"
    asset_dir = PROJECT_ROOT / '_validation' / f'asset_{aid_str}'
    step_dir = asset_dir / f'{step_num}_{step_name.lower().replace(" ", "_")}'
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def prepare_report_dir(report_dir: Path, pattern: str = "*.*"):
    """
    Archiveert bestaande bestanden in de map naar een .archive submap.
    """
    archive_dir = report_dir / '.archive'
    files = list(report_dir.glob(pattern))
    # Filter de .archive map zelf uit en directories
    files = [f for f in files if f.is_file()]
    
    if files:
        archive_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_archive_dir = archive_dir / timestamp
        run_archive_dir.mkdir(exist_ok=True)
        
        for f in files:
            try:
                shutil.move(str(f), str(run_archive_dir / f.name))
            except Exception as e:
                print(f"   ⚠️ Kon {f.name} niet archiveren: {e}")


def save_markdown_report(report_dir: Path, filename_prefix: str, title: str, content_lines: list, asset_id: int = None):
    """
    Slaat een lijst van regels op als Markdown bestand.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = report_dir / f"{filename_prefix}_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Gegenereerd:** {datetime.now().isoformat()}\n")
        if asset_id:
            f.write(f"**Asset ID:** {asset_id}\n")
        f.write("\n")
        f.write("\n".join(content_lines))
    
    return filename


def show_menu():
    """Toon validation menu - Herordend naar chronologische validatie flow"""
    print("--- Categorie 1: Data Fundament (Fase 1) ---")
    print("  1.  📊 Database Statistieken (IN: db / OUT: console)")
    print("  2.  📋 Barrier Outcome Status & Distribution (IN: qbn.barrier_outcomes / OUT: reports)")
    print("  3.  🛡️  Barrier Outcome Coverage (IN: qbn.barrier_outcomes / OUT: console)")
    print("  4.  ⚡ Barrier Performance Benchmarks (IN: kfl.klines_raw / OUT: reports)")
    print()
    print("--- Categorie 2: Signal & Weight Validation (Fase 2) ---")
    print("  5.  ✅ Signal Classification Check (IN: qbn.signal_classification / OUT: console)")
    print("  6.  🔍 Validate IDA Weights (Dry-Run) (IN: qbn.barrier_outcomes / OUT: reports)")
    print("  7.  🔗 Concordance Analysis (IN: kfl.mtf_signals_* / OUT: console)")
    print()
    print("--- Categorie 3: BN Brain Health (Fase 3) ---")
    print("  8.  💾 CPT Cache Status (IN: qbn.cpt_cache / OUT: console)")
    print("  9.  🛡️  CPT Health Report (Entropy/Coverage) (IN: qbn.cpt_cache / OUT: console)")
    print("  10. 🧪 CPT Stability Validation (IN: qbn.cpt_cache / OUT: console)")
    print("  11. 📊 Semantic Score Analysis (IN: qbn.cpt_cache / OUT: console)")
    print()
    print("--- Categorie 4: Diepe Diagnostiek (Fase 4) ---")
    print("  12. 🔬 Node-Level Diagnostics (IN: all / OUT: reports)")
    print()
    print("--- Categorie 5: Performance & Readiness (Fase 5) ---")
    print("  13. 📊 Prediction Accuracy Report (IN: qbn.output_entry, qbn.barrier_outcomes / OUT: console)")
    print("  14. 🎯 Position Prediction Accuracy (IN: qbn.output_position, qbn.barrier_outcomes / OUT: console)")
    print("  15. 🚶 Walk-Forward Validation (IN: qbn.signal_outcomes / OUT: console)")
    print("  16. 📈 Backtest Simulation (IN: kfl.klines_raw / OUT: reports)")
    print("  17. 🏁 Production Readiness Check (GO/NO-GO) (IN: all / OUT: reports)")
    print()
    print("--- Categorie 6: Strategie Optimalisatie ---")
    print("  18. 🔍 Grid Search Configurator (Parameter Optimization)")
    print()
    print("--- Categorie 7: Rapportage & Automatisering ---")
    print("  19. 📄 Generate Validation Report (MD) (IN: db / OUT: md)")
    print("  20. 📦 Archiveer Rapporten (IN: reports / OUT: zip)")
    print("  21. 🚀 VOLLEDIGE VALIDATIE CYCLE (Auto-Run)")
    print()
    print("  99. 🔄 Refresh status")
    print("  0.  Exit")
    print()
    return input("Keuze: ").strip()


# ==============================================================================
# System Status (Fase 1)
# ==============================================================================
def run_database_stats(asset_id=0):
    """Wrapper voor show_database_stats met rapportage"""
    report_dir = get_report_dir(asset_id, 1, "Database Statistieken")
    prepare_report_dir(report_dir)
    show_database_stats(save_to_file=True, output_dir=report_dir)


def run_barrier_validation(asset_id=None, auto=False):
    """Valideer barrier data integriteit en toon status & distributie"""
    print("\n" + "=" * 60)
    print("📋 BARRIER OUTCOME STATUS & DISTRIBUTION")
    print("=" * 60 + "\n")
    
    if asset_id is None:
        asset_id = input("Asset ID (leeg voor alle) []: ").strip()
    
    report_dir = get_report_dir(asset_id, 2, "Barrier Outcome Status")
    prepare_report_dir(report_dir)
    
    print(f"\n📊 Start diepe statistische analyse...")
    cmd_stats = [sys.executable, 'scripts/analyze_barrier_outcomes.py']
    if asset_id:
        cmd_stats.extend(['--asset-id', str(asset_id)])
    
    cmd_stats.extend(['--output-dir', str(report_dir)])
    subprocess.run(cmd_stats, cwd=PROJECT_ROOT)
        
    if not auto:
        input("\nDruk op Enter om terug te gaan...")


def run_barrier_coverage_status(asset_id=0, auto=False):
    """Toon barrier outcome coverage status"""
    print("\n" + "="*60)
    print("🛡️ BARRIER OUTCOME COVERAGE STATUS")
    print("="*60 + "\n")
    
    try:
        from database.db import get_cursor
        
        with get_cursor() as cur:
            params = []
            clauses = []
            if asset_id and int(asset_id) > 0:
                clauses.append("asset_id = %s")
                params.append(int(asset_id))

            # Optioneel filteren op run_id (full validation cycle)
            if CURRENT_RUN_ID and _db_has_column("qbn", "barrier_outcomes", "run_id"):
                clauses.append("run_id = %s")
                params.append(CURRENT_RUN_ID)

            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(
                f"""
                SELECT asset_id, COUNT(*) as total, MIN(time_1) as first_obs, MAX(time_1) as last_obs
                FROM qbn.barrier_outcomes
                {where_sql}
                GROUP BY asset_id ORDER BY asset_id
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        
        if not rows:
            print(f"❌ Geen barrier outcomes gevonden {'voor asset ' + str(asset_id) if asset_id else ''}")
            if not auto: input("\nDruk op Enter om terug te gaan...")
            return

        header = f"{'ID':<5} {'Total':>12} {'First':<20} {'Last':<20}"
        sep = "-" * 60
        print(f"📊 Barrier coverage voor {len(rows)} assets:\n")
        print(header)
        print(sep)
        
        report_lines = ["```text", header, sep]
        for aid, total, first, last in rows:
            line = f"{aid:<5} {total:>12,} {str(first)[:19]:<20} {str(last)[:19]:<20}"
            print(line)
            report_lines.append(line)
        report_lines.append("```")
        
        report_dir = get_report_dir(asset_id, 3, "Barrier Outcome Coverage")
        prepare_report_dir(report_dir)
        save_markdown_report(report_dir, "coverage_report", "Barrier Outcome Coverage Status", report_lines)
        
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    if not auto:
        input("\nDruk op Enter om terug te gaan...")


def run_barrier_benchmarks(auto=False):
    """Run performance benchmarks voor barrier operations"""
    print("\n" + "=" * 60)
    print("⚡ BARRIER PERFORMANCE BENCHMARKS")
    print("=" * 60 + "\n")
    
    report_dir = get_report_dir(0, 4, "Barrier Performance Benchmarks")
    prepare_report_dir(report_dir)
    
    print("🔄 Start benchmarks (CPU single, GPU batch, Inference)...")
    cmd = [sys.executable, 'scripts/benchmark_barriers.py', '--output-dir', str(report_dir)]
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    if not auto:
        input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Signal & Weight Validation (Fase 2)
# ==============================================================================
def run_signal_classification_check(asset_id=0, auto=False):
    """Signal Classification Check"""
    print("\n" + "="*60)
    print("✅ SIGNAL CLASSIFICATION CHECK")
    print("="*60 + "\n")

    try:
        from database.db import get_cursor
        with get_cursor() as cur:
            cur.execute("SELECT signal_name, semantic_class, polarity, indicator_base FROM qbn.signal_classification ORDER BY semantic_class, signal_name")
            rows = cur.fetchall()

        if not rows:
            print("⚠️ Geen signal classificaties gevonden")
            if not auto: input("\nDruk op Enter om terug te gaan...")
            return
            
        current_class = None
        class_counts = {'LEADING': 0, 'COINCIDENT': 0, 'CONFIRMING': 0}
        report_lines = []

        for name, sem_class, polarity, indicator in rows:
            if sem_class != current_class:
                if current_class: report_lines.append("```\n")
                current_class = sem_class
                report_lines.append(f"### {sem_class}\n```text")
                print(f"\n--- {sem_class} ---")

            class_counts[sem_class] = class_counts.get(sem_class, 0) + 1
            pol_icon = "📈" if polarity == 'BULLISH' else "📉" if polarity == 'BEARISH' else "➖"
            line = f"   {name:<35} {pol_icon} {polarity:<8} [{indicator}]"
            print(line)
            report_lines.append(line)

        report_lines.append("```")
        summary_lines = [f"   Leading signals:    {class_counts.get('LEADING', 0)}", f"   Coincident signals: {class_counts.get('COINCIDENT', 0)}", f"   Confirming signals: {class_counts.get('CONFIRMING', 0)}", f"   Total:              {len(rows)}"]
        report_lines.extend(["\n### Summary\n```text"] + summary_lines + ["```"])
        
        report_dir = get_report_dir(asset_id, 5, "Signal Classification")
        prepare_report_dir(report_dir)
        save_markdown_report(report_dir, "classification_report", "Signal Classification Check", report_lines)

    except Exception as e:
        print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_ida_validation(asset_id=None, auto=False):
    """Valideer IDA Training Weights met visualisaties (Dry-Run)"""
    print("\n" + "=" * 60)
    print("🔍 IDA WEIGHT VALIDATION (Dry-Run)")
    print("=" * 60 + "\n")
    
    if asset_id is None:
        asset_id = input("Asset ID (bijv. 1): ").strip()
    if not asset_id:
        print("❌ Asset ID is verplicht")
        if not auto: input("\nDruk op Enter om terug te gaan...")
        return
    
    report_dir = get_report_dir(asset_id, 6, "IDA Weight Validation")
    prepare_report_dir(report_dir)
    
    cmd = [sys.executable, 'scripts/validate_ida_weights.py', '--asset-id', str(asset_id), '--output-dir', str(report_dir)]
    print(f"\n🔄 Start IDA weight validatie voor asset {asset_id}...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_concordance_analysis(asset_id=None, auto=False):
    """Concordance matrix analyse"""
    print("\n" + "="*60)
    print("🔗 CONCORDANCE ANALYSE")
    print("="*60 + "\n")
    
    try:
        from inference.concordance_matrix import ConcordanceMatrix
        from config.bayesian_config import SignalState
        from database.db import get_cursor
        import pandas as pd
        
        matrix = ConcordanceMatrix()
        if asset_id is None: asset_id = input("Asset ID [1]: ").strip() or "1"
        
        with get_cursor() as cur:
            cur.execute("SELECT time_1 as time, rsi_oversold_d, rsi_oversold_240, rsi_oversold_60, rsi_oversold_1, concordance_score_d FROM kfl.mtf_signals_lead WHERE asset_id = %s ORDER BY time_1 DESC LIMIT 43200", (int(asset_id),))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
        if not rows: 
            print(f"❌ Geen data voor asset {asset_id}")
            if not auto: input("\nDruk op Enter om terug te gaan...")
            return
        
        df = pd.DataFrame(rows, columns=columns)
        scenario_counts = {}
        for _, row in df.iterrows():
            htf = SignalState(row['rsi_oversold_d']) if pd.notna(row['rsi_oversold_d']) else SignalState.NEUTRAL
            mtf = SignalState(row['rsi_oversold_240']) if pd.notna(row['rsi_oversold_240']) else SignalState.NEUTRAL
            ltf = SignalState(row['rsi_oversold_60']) if pd.notna(row['rsi_oversold_60']) else SignalState.NEUTRAL
            utf = SignalState(row['rsi_oversold_1']) if pd.notna(row['rsi_oversold_1']) else SignalState.NEUTRAL
            scenario = matrix.classify_scenario(htf, mtf, ltf, utf)
            scenario_counts[scenario.value] = scenario_counts.get(scenario.value, 0) + 1
        
        report_lines = ["### Scenario Analyse (RSI)", "```text", f"{'Scenario':<25} {'Count':>10} {'%':>10}", "-"*45]
        for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
            pct = count / len(df) * 100
            line = f"{scenario:<25} {count:>10} {pct:>9.1f}%"
            print(line); report_lines.append(line)
        report_lines.append("```")
        
        if not df['concordance_score_d'].dropna().empty:
            stats = [("Mean", df['concordance_score_d'].mean()), ("Median", df['concordance_score_d'].median()), ("Std", df['concordance_score_d'].std())]
            report_lines.append("\n### Stats\n```text")
            for s, v in stats: line = f"   {s:<8}: {v:.3f}"; print(line); report_lines.append(line)
            report_lines.append("```")
        
        report_dir = get_report_dir(asset_id, 7, "Concordance Analysis")
        prepare_report_dir(report_dir)
        save_markdown_report(report_dir, "concordance_report", "Concordance Matrix Analyse", report_lines, asset_id=asset_id)
        
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# BN Brain Health (Fase 3)
# ==============================================================================
def run_cpt_cache_status(asset_id=0, auto=False):
    """Toon CPT cache status"""
    print("\n" + "=" * 60)
    print("💾 CPT CACHE STATUS")
    print("=" * 60 + "\n")
    
    try:
        # REASON: In full validation cycle willen we dezelfde training-run evalueren.
        # EXPL: Als run_id beschikbaar is, filteren we direct op qbn.cpt_cache.run_id.
        if CURRENT_RUN_ID and _db_has_column("qbn", "cpt_cache", "run_id"):
            from database.db import get_cursor
            with get_cursor() as cur:
                where_clause = ""
                params = [CURRENT_RUN_ID]
                if asset_id and int(asset_id) > 0:
                    where_clause = "AND scope_key = %s"
                    params.append(f"asset_{int(asset_id)}")

                cur.execute(
                    f"""
                    SELECT
                        scope_key,
                        COUNT(*) as node_count,
                        MIN(generated_at) as oldest,
                        MAX(generated_at) as newest,
                        SUM(observations) as total_observations
                    FROM qbn.cpt_cache
                    WHERE run_id = %s
                      AND model_version >= '3.1'
                      {where_clause}
                    GROUP BY scope_key
                    ORDER BY scope_key
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()

            assets = []
            for row in rows:
                assets.append(
                    {
                        "asset_id": row[0],
                        "node_count": row[1],
                        "oldest": row[2].isoformat() if row[2] else None,
                        "newest": row[3].isoformat() if row[3] else None,
                        "total_observations": row[4],
                    }
                )
            status = {"total_assets": len(assets), "assets": assets}
        else:
            from inference.cpt_generator import ConditionalProbabilityTableGenerator
            generator = ConditionalProbabilityTableGenerator()
            status = generator.get_cpt_cache_status()
        report_lines = []
        if status.get('error'):
            print(f"❌ Fout: {status['error']}")
            report_lines.append(f"❌ Fout: {status['error']}")
        elif status['total_assets'] == 0:
            print("📭 Geen CPT's in cache")
            report_lines.append("📭 Geen CPT's in cache")
        else:
            header = f"   {'Asset':<10} {'Nodes':<8} {'Obs':<12} {'Gegenereerd'}"
            sep = "   " + "-" * 55
            report_lines.extend([f"📊 {status['total_assets']} assets in cache:\n", "```text", header, sep])
            for asset in status['assets']:
                generated = asset['newest'][:19] if asset['newest'] else 'onbekend'
                obs = asset['total_observations'] or 0
                line = f"   {asset['asset_id']:<10} {asset['node_count']:<8} {obs:<12} {generated}"
                print(line); report_lines.append(line)
            report_lines.append("```")
        
        report_dir = get_report_dir(asset_id, 8, "CPT Cache Status")
        prepare_report_dir(report_dir)
        save_markdown_report(report_dir, "cache_status", "CPT Cache Status Report", report_lines)
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_cpt_health_report(asset_id=0, auto=False):
    """CPT Health Report met kwaliteitsmetrics"""
    print("\n" + "="*60)
    print("🛡️ CPT HEALTH REPORT")
    print("="*60 + "\n")
    
    try:
        from database.db import get_cursor
        with get_cursor() as cur:
            # REASON: Gebruik DISTINCT ON om alleen de LAATSTE CPT per node te tonen
            where_clause = f"AND asset_id = {asset_id}" if asset_id and int(asset_id) > 0 else ""
            run_clause = ""
            if CURRENT_RUN_ID and _db_has_column("qbn", "cpt_cache", "run_id"):
                run_clause = f"AND run_id = '{CURRENT_RUN_ID}'"
            cur.execute(f"""
                SELECT DISTINCT ON (asset_id, node_name) 
                    asset_id, node_name, coverage, entropy, info_gain, 
                    stability_score, semantic_score, observations, generated_at 
                FROM qbn.cpt_cache 
                WHERE 1=1 {where_clause} {run_clause}
                ORDER BY asset_id, node_name, generated_at DESC
            """)
            rows = cur.fetchall()
            
        if not rows: 
            print(f"📭 Geen CPT's in cache {'voor asset ' + str(asset_id) if asset_id else ''}")
            if not auto: input("\nDruk op Enter om terug te gaan...")
            return
            
        total_nodes = len(rows)
        avg_coverage = sum(r[2] or 0 for r in rows) / total_nodes
        avg_stability = sum(r[5] or 0 for r in rows) / total_nodes
        summary_lines = ["📊 SUMMARY", f"   Total nodes:      {total_nodes}", f"   Avg coverage:     {avg_coverage:.1%}", f"   Avg stability:    {avg_stability:.2f}"]
        for sl in summary_lines: print(sl)
        
        header = f"{'Asset':<6} {'Node':<25} {'Cov':>5} {'Entr':>5} {'Gain':>5} {'Stab':>5} {'Sem':>5} {'Obs':>8}"
        report_lines = ["### Summary", "```text"] + summary_lines + ["```\n", "### Details", "```text", header, "-" * 75]
        for r in rows:
            f_cov = f"{r[2]:.0%}" if r[2] else "-"
            f_entr = f"{r[3]:.1f}" if r[3] else "-"
            f_gain = f"{r[4]:.2f}" if r[4] else "-"
            f_stab = f"{r[5]:.2f}" if r[5] else "-"
            f_sem = f"{r[6]:.2f}" if r[6] else "-"
            f_obs = f"{r[7]:,}" if r[7] else "0"
            status = "✅" if (not r[2] or r[2] >= 0.5) and (not r[5] or r[5] >= 0.7) else "⚠️"
            line = f"{r[0]:<6} {r[1]:<25} {f_cov:>5} {f_entr:>5} {f_gain:>5} {f_stab:>5} {f_sem:>5} {f_obs:>8} {status}"
            print(line); report_lines.append(line)
        report_lines.append("```")
        
        report_dir = get_report_dir(asset_id, 9, "CPT Health Report")
        prepare_report_dir(report_dir)
        save_markdown_report(report_dir, "health_report", "CPT Health Report", report_lines)
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_cpt_stability_validation(asset_id=None, auto=False):
    """CPT Stability Validation"""
    print("\n" + "="*60); print("🧪 CPT STABILITY VALIDATION"); print("="*60 + "\n")
    try:
        from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
        if asset_id is None:
            asset_id = input("Asset ID [1]: ").strip() or "1"
            lookback = input("Lookback dagen [3650]: ").strip() or "3650"
        else: lookback = "3650"
        
        report_dir = get_report_dir(asset_id, 10, "CPT Stability")
        prepare_report_dir(report_dir)
        
        print(f"\n🔄 Valideren van CPTs voor asset {asset_id} (lookback={lookback})...")
        generator = QBNv3CPTGenerator()
        generator.validate_existing_cpts(int(asset_id), lookback_days=int(lookback))
        print(f"\n✅ Stability validation voltooid!")
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_semantic_score_analysis(asset_id=0, auto=False):
    """Semantic Score Analysis"""
    print("\n" + "="*60); print("📊 SEMANTIC SCORE ANALYSIS"); print("="*60 + "\n")
    try:
        from database.db import get_cursor
        with get_cursor() as cur:
            # Filter op asset indien nodig
            # REASON: Gebruik DISTINCT ON om alleen de LAATSTE CPT per node te tonen
            where_clause = f"AND asset_id = {asset_id}" if asset_id and int(asset_id) > 0 else ""
            run_clause = ""
            if CURRENT_RUN_ID and _db_has_column("qbn", "cpt_cache", "run_id"):
                run_clause = f"AND run_id = '{CURRENT_RUN_ID}'"
            cur.execute(f"SELECT DISTINCT ON (asset_id, node_name) asset_id, node_name, semantic_score, coverage, stability_score FROM qbn.cpt_cache WHERE 1=1 {where_clause} {run_clause} AND semantic_score IS NOT NULL ORDER BY asset_id, node_name, generated_at DESC LIMIT 20")
            rows = cur.fetchall()
        report_lines = ["```text"]
        if not rows: print("⚠️ Geen semantic scores"); report_lines.append("⚠️ Geen semantic scores")
        else:
            header = f"{'Asset':<6} {'Node':<30} {'Semantic':>10} {'Coverage':>10} {'Stability':>10}"
            report_lines.extend(["🔴 Nodes met laagste semantic scores:\n", header, "-" * 70])
            for r in rows:
                line = f"{r[0]:<6} {r[1]:<30} {r[2]:>9.2f} {r[3]:>9.0%} {r[4]:>9.2f} " + ("🔴" if r[2] < 0.5 else "⚠️" if r[2] < 0.7 else "✅")
                print(line); report_lines.append(line)
        report_lines.append("```")
        report_dir = get_report_dir(asset_id, 11, "Semantic Score Analysis")
        prepare_report_dir(report_dir); save_markdown_report(report_dir, "semantic_analysis", "Semantic Score Analysis", report_lines)
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Diepe Diagnostiek (Fase 4)
# ==============================================================================
def run_node_level_diagnostics_logic(asset_id: int):
    """Interne logica voor node diagnostics zonder wachten op input."""
    from validation.node_diagnostics import NodeDiagnosticValidator
    from validation.node_diagnostic_report import generate_markdown_report
    
    report_dir = get_report_dir(asset_id, 12, "Node-Level Diagnostics")
    prepare_report_dir(report_dir)
    
    validator = NodeDiagnosticValidator(asset_id, run_id=CURRENT_RUN_ID or None)
    results = validator.run_full_diagnostic(3650)
    if results:
        report_path = generate_markdown_report(asset_id, results, output_dir=report_dir)
        print(f"   ✅ Node-Level Diagnostics voltooid: {report_path.name}")


def run_node_level_diagnostics(auto=False):
    """Node-Level Diagnostics - individuele node validatie"""
    print("\n" + "="*60); print("🔬 NODE-LEVEL DIAGNOSTICS"); print("="*60 + "\n")
    try:
        asset_id = int(input("Asset ID [1]: ").strip() or "1")
        days = int(input("Aantal dagen data [3650]: ").strip() or "3650")
        
        from validation.node_diagnostics import NodeDiagnosticValidator
        from validation.node_diagnostic_report import print_console_diagnostic, generate_markdown_report
        
        report_dir = get_report_dir(asset_id, 12, "Node-Level Diagnostics")
        prepare_report_dir(report_dir)
        
        print(f"\n🔄 Laden van data en uitvoeren van inference...")
        validator = NodeDiagnosticValidator(asset_id, run_id=CURRENT_RUN_ID or None)
        results = validator.run_full_diagnostic(days)
        
        if not results: print("\n❌ Geen resultaten")
        else:
            report_path = generate_markdown_report(asset_id, results, output_dir=report_dir)
            print_console_diagnostic(asset_id, results, report_path)
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Performance & Readiness (Fase 5)
# ==============================================================================
def run_prediction_accuracy_report(asset_id=None, auto=False):
    """Prediction Accuracy Report (v3.1 JSONB aware)"""
    print("\n" + "="*60); print("📊 PREDICTION ACCURACY REPORT"); print("="*60 + "\n")
    try:
        from database.db import get_cursor
        
        # REASON: In auto mode, detecteer automatisch welke mode data heeft
        if auto:
            days = "3650"
            with get_cursor() as cur:
                # Check welke outcome_mode het meeste data heeft
                cur.execute("""
                    SELECT outcome_mode, COUNT(*) as cnt 
                    FROM qbn.output_entry 
                    WHERE asset_id = %s OR %s = 0
                    GROUP BY outcome_mode 
                    ORDER BY cnt DESC 
                    LIMIT 1
                """, (asset_id or 0, asset_id or 0))
                row = cur.fetchone()
                if row and row[0]:
                    outcome_mode = row[0]
                    print(f"ℹ️ Auto-detected mode: {outcome_mode} ({row[1]} entries)")
                else:
                    outcome_mode = "barrier"  # fallback
        elif asset_id is None:
            asset_id = int(input("Asset ID (0 voor alle) [0]: ").strip() or "0")
            days = input("Aantal dagen terug [3650]: ").strip() or "3650"
            mode_choice = input("Outcome Mode (1: Barrier/Dual, 2: PiT, 3: Alle) [1]: ").strip() or "1"
            if mode_choice == "1":
                outcome_mode = "barrier"
            elif mode_choice == "2":
                outcome_mode = "point_in_time"
            else:
                outcome_mode = "all"
        else: 
            days = "3650"
            outcome_mode = "barrier"

        # REASON: Bepaal mode filter voor query
        if outcome_mode == "all":
            mode_filter = "1=1"  # Geen filter
            mode_display = "all modes"
        elif outcome_mode in ("barrier", "dual"):
            mode_filter = "(o.outcome_mode = 'barrier' OR o.outcome_mode = 'dual')"
            mode_display = "barrier/dual"
        else:
            mode_filter = "o.outcome_mode = 'point_in_time'"
            mode_display = "point_in_time"

        # REASON: Query die zowel JSONB barrier predictions als reguliere predictions ondersteunt
        # EXPL: Gebruikt COALESCE om beide formats te ondersteunen
        query = f"""
            SELECT 
                o.asset_id, 
                COUNT(*) as total,
                COALESCE(
                    -- Barrier mode: check JSONB predictions tegen barrier_outcomes
                    CASE WHEN SUM(CASE WHEN o.barrier_prediction_1h IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN
                        SUM(CASE WHEN (o.barrier_prediction_1h->>'expected_direction' = 'up' AND s.first_significant_barrier LIKE 'up%%') 
                                  OR (o.barrier_prediction_1h->>'expected_direction' = 'down' AND s.first_significant_barrier LIKE 'down%%') 
                                  OR (o.barrier_prediction_1h->>'expected_direction' = 'neutral' AND s.first_significant_barrier = 'none') THEN 1 ELSE 0 END)::float 
                            / NULLIF(COUNT(*), 0)
                    ELSE NULL END,
                    -- Fallback: directional predictions (prediction_1h column)
                    SUM(CASE WHEN (o.prediction_1h LIKE '%%Bullish' AND s.first_significant_barrier LIKE 'up%%')
                              OR (o.prediction_1h LIKE '%%Bearish' AND s.first_significant_barrier LIKE 'down%%')
                              OR (o.prediction_1h = 'Neutral' AND s.first_significant_barrier = 'none') THEN 1 ELSE 0 END)::float 
                        / NULLIF(COUNT(*), 0),
                    0.0
                ) as acc_1h,
                COALESCE(
                    CASE WHEN SUM(CASE WHEN o.barrier_prediction_4h IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN
                        SUM(CASE WHEN (o.barrier_prediction_4h->>'expected_direction' = 'up' AND s.first_significant_barrier LIKE 'up%%') 
                                  OR (o.barrier_prediction_4h->>'expected_direction' = 'down' AND s.first_significant_barrier LIKE 'down%%') 
                                  OR (o.barrier_prediction_4h->>'expected_direction' = 'neutral' AND s.first_significant_barrier = 'none') THEN 1 ELSE 0 END)::float 
                            / NULLIF(COUNT(*), 0)
                    ELSE NULL END,
                    SUM(CASE WHEN (o.prediction_4h LIKE '%%Bullish' AND s.first_significant_barrier LIKE 'up%%')
                              OR (o.prediction_4h LIKE '%%Bearish' AND s.first_significant_barrier LIKE 'down%%')
                              OR (o.prediction_4h = 'Neutral' AND s.first_significant_barrier = 'none') THEN 1 ELSE 0 END)::float 
                        / NULLIF(COUNT(*), 0),
                    0.0
                ) as acc_4h,
                COALESCE(
                    CASE WHEN SUM(CASE WHEN o.barrier_prediction_1d IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN
                        SUM(CASE WHEN (o.barrier_prediction_1d->>'expected_direction' = 'up' AND s.first_significant_barrier LIKE 'up%%') 
                                  OR (o.barrier_prediction_1d->>'expected_direction' = 'down' AND s.first_significant_barrier LIKE 'down%%') 
                                  OR (o.barrier_prediction_1d->>'expected_direction' = 'neutral' AND s.first_significant_barrier = 'none') THEN 1 ELSE 0 END)::float 
                            / NULLIF(COUNT(*), 0)
                    ELSE NULL END,
                    SUM(CASE WHEN (o.prediction_1d LIKE '%%Bullish' AND s.first_significant_barrier LIKE 'up%%')
                              OR (o.prediction_1d LIKE '%%Bearish' AND s.first_significant_barrier LIKE 'down%%')
                              OR (o.prediction_1d = 'Neutral' AND s.first_significant_barrier = 'none') THEN 1 ELSE 0 END)::float 
                        / NULLIF(COUNT(*), 0),
                    0.0
                ) as acc_1d
            FROM qbn.output_entry o 
            JOIN qbn.barrier_outcomes s ON o.asset_id = s.asset_id AND o.time = s.time_1
            WHERE o.time > NOW() - (%s * INTERVAL '1 day') 
              AND {mode_filter}
        """
        
        # REASON: Bind accuracy report aan dezelfde training-run indien mogelijk.
        if CURRENT_RUN_ID and _db_has_column("qbn", "barrier_outcomes", "run_id"):
            query += " AND s.run_id = %s"
        
        if asset_id and asset_id > 0: 
            query += f" AND o.asset_id = {asset_id}"
        query += " GROUP BY o.asset_id ORDER BY o.asset_id"
        
        params = [int(days)]
        if CURRENT_RUN_ID and _db_has_column("qbn", "barrier_outcomes", "run_id"):
            params.append(CURRENT_RUN_ID)
        with get_cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
        
        report_lines = [f"Mode: {mode_display}, Lookback: {days}d\n", "```text", f"{'Asset':<8} {'Total':>8} {'Acc 1h':>10} {'Acc 4h':>10} {'Acc 1d':>10}", "-" * 50]
        if not rows: 
            print("⚠️ Geen predictions gevonden")
            report_lines.append("⚠️ Geen predictions gevonden")
        else:
            for r in rows:
                acc_1h = r[2] if r[2] is not None else 0.0
                acc_4h = r[3] if r[3] is not None else 0.0
                acc_1d = r[4] if r[4] is not None else 0.0
                line = f"{r[0]:<8} {r[1]:>8} {acc_1h:>10.1%} {acc_4h:>10.1%} {acc_1d:>10.1%}"
                print(line)
                report_lines.append(line)
        report_lines.append("```")
        report_dir = get_report_dir(asset_id or 0, 13, "Prediction Accuracy")
        prepare_report_dir(report_dir)
        save_markdown_report(report_dir, "accuracy_report", "Prediction Accuracy", report_lines, asset_id=asset_id)
    except Exception as e: 
        print(f"❌ Fout: {e}")
        import traceback
        traceback.print_exc()
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_position_prediction_accuracy_report(asset_id=None, auto=False):
    """Position Prediction Accuracy Report (v3.1 Event-Driven)"""
    print("\n" + "="*60); print("🎯 POSITION PREDICTION ACCURACY REPORT (v3.1)"); print("="*60 + "\n")
    
    if asset_id is None:
        asset_id = int(input("Asset ID (0 voor alle) [0]: ").strip() or "0")
    
    if asset_id == 0:
        # Default to 1 if not provided in auto mode, or ask
        if auto:
            asset_id = 1
        else:
            print("⚠️  Specific Asset ID required for event replay.")
            aid_in = input("Asset ID [1]: ").strip() or "1"
            asset_id = int(aid_in)
        
    report_dir = get_report_dir(asset_id, 14, "Position Prediction Accuracy")
    prepare_report_dir(report_dir)
    
    cmd = [sys.executable, 'scripts/validate_position_management.py', '--asset-id', str(asset_id), '--output-dir', str(report_dir)]
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_walk_forward_validation(asset_id=None, auto=False):
    """Walk-Forward Validation"""
    print("\n" + "="*60); print("🚶 WALK-FORWARD VALIDATION"); print("="*60 + "\n")
    try:
        aid = asset_id or input("Asset ID [1]: ").strip() or "1"
        days = "3650" if asset_id else input("Dagen [3650]: ").strip() or "3650"
        
        report_dir = get_report_dir(aid, 15, "Walk-Forward Validation")
        prepare_report_dir(report_dir)
        
        cmd = [sys.executable, 'scripts/run_walk_forward.py', '--asset-id', str(aid), '--days', str(days), '--output-dir', str(report_dir)]
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_backtest_simulation(auto=False):
    """Backtest Simulation"""
    print("\n" + "="*60); print("📈 BACKTEST SIMULATION"); print("="*60 + "\n")
    try:
        aid = input("Asset ID [1]: ").strip() or "1"
        report_dir = get_report_dir(aid, 16, "Backtest Simulation")
        prepare_report_dir(report_dir)
        print("⚠️ Backtest script nog niet volledig structureel geïntegreerd.")
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


def run_production_readiness_check(asset_id=None, auto=False):
    """Production Readiness Check - GO/NO-GO voor inference"""
    print("\n" + "="*60); print("🏁 PRODUCTION READINESS CHECK"); print("="*60 + "\n")
    try:
        from validation.production_readiness import ProductionReadinessValidator
        from validation.readiness_report import ReadinessReportGenerator, print_console_results
        aid = int(asset_id) if asset_id else int(input("Asset ID [1]: ").strip() or "1")
        
        report_dir = get_report_dir(aid, 17, "Production Readiness")
        prepare_report_dir(report_dir)
        
        validator = ProductionReadinessValidator(aid, run_id=CURRENT_RUN_ID or None)
        verdict, results = validator.run_all_checks()
        print_console_results(aid, validator.asset_symbol, verdict, results)
        
        # ReadinessReportGenerator supports output_dir in __init__
        report_gen = ReadinessReportGenerator(output_dir=report_dir)
        report_path = report_gen.generate(aid, validator.asset_symbol, verdict, results)
        print(f"\nRapport opgeslagen: {report_path}")
    except Exception as e: print(f"❌ Fout: {e}")
    if not auto: input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Strategie Optimalisatie (Fase 6)
# ==============================================================================
def run_grid_search_configurator(auto=False):
    """Grid Search Configurator - Interactieve parameter optimalisatie"""
    print("\n" + "="*60)
    print("🔍 GRID SEARCH CONFIGURATOR")
    print("="*60 + "\n")
    
    try:
        # Roep de grid search configurator aan
        cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'grid_search_configurator.py')]
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except Exception as e:
        print(f"❌ Fout: {e}")
    
    if not auto:
        input("\nDruk op Enter om terug te gaan...")


# ==============================================================================
# Rapportage & Automatisering (Fase 7)
# ==============================================================================
def generate_validation_report_logic():
    """Interne logica voor globaal rapport overzicht."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_dir = PROJECT_ROOT / '_validation'
        validation_dir.mkdir(exist_ok=True)
        filename = validation_dir / f"global_validation_report_{timestamp}.md"
        
        print(f"🔄 Genereer globaal rapport naar {filename.name}...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# QBN v3 Global Validation Report\n\n")
            f.write(f"**Gegenereerd:** {datetime.now().isoformat()}\n\n")
            f.write("Dit rapport geeft een overzicht van de beschikbare validatiedata per asset.\n\n")
            f.write("## Beschikbare Asset Data\n\n")
            
            asset_dirs = sorted(list(validation_dir.glob("asset_*")))
            if not asset_dirs:
                f.write("Geen asset-specifieke validatiedata gevonden.\n")
            else:
                for adir in asset_dirs:
                    asset_name = adir.name.replace("asset_", "Asset ")
                    f.write(f"### {asset_name}\n")
                    steps = sorted(list(adir.glob("[0-9]*_*")))
                    for step in steps:
                        f.write(f"- {step.name}\n")
                    f.write("\n")
        
        print(f"   ✅ Globaal rapport gegenereerd: {filename.name}")
    except Exception as e:
        print(f"   ❌ Fout bij genereren globaal rapport: {e}")


def run_full_validation_cycle():
    """
    Voert alle validatietesten sequentieel uit voor een geselecteerd asset.
    
    REASON: Alle 17 menu-stappen worden doorlopen in chronologische volgorde.
    Zware/optionele stappen (benchmarks, backtest) kunnen worden overgeslagen.
    """
    print("\n" + "="*60); print("🚀 START VOLLEDIGE VALIDATIE CYCLE"); print("="*60 + "\n")
    aid_str = input("Asset ID voor volledige cycle [1]: ") or "1"
    try: aid = int(aid_str)
    except ValueError: print("❌ Ongeldige Asset ID"); return

    # REASON: Validation moet dezelfde training-run evalueren om false positives/negatives te voorkomen.
    # EXPL: Als user niets invult, proberen we automatisch de nieuwste run_id te pakken.
    global CURRENT_RUN_ID
    run_id_in = input("Run ID (leeg = auto-detect latest) []: ").strip()
    CURRENT_RUN_ID = run_id_in or _resolve_latest_run_id(aid)
    if CURRENT_RUN_ID:
        print(f"ℹ️ Using run_id for validation: {CURRENT_RUN_ID}")
    else:
        print("⚠️  Geen run_id kunnen detecteren. Validation draait zonder run binding (legacy gedrag).")
    
    # REASON: Optionele heavy steps kunnen overgeslagen worden
    skip_benchmarks = input("Skip Barrier Benchmarks? (y/N): ").strip().lower() == 'y'
    skip_backtest = True  # Backtest is nog niet volledig geïntegreerd

    # REASON: Volledige steps lijst conform menu-nummering (1-17)
    steps = [
        # Fase 1: Data Fundament
        ("1. Database Statistieken", run_database_stats, [aid], False),
        ("2. Barrier Outcome Status", run_barrier_validation, [aid], False),
        ("3. Barrier Coverage", run_barrier_coverage_status, [aid], False),
        ("4. Barrier Benchmarks", run_barrier_benchmarks, [], skip_benchmarks),
        
        # Fase 2: Signal & Weight Validation
        ("5. Signal Classification", run_signal_classification_check, [aid], False),
        ("6. IDA Weight Validation", run_ida_validation, [aid], False),
        ("7. Concordance Analysis", run_concordance_analysis, [aid], False),
        
        # Fase 3: BN Brain Health
        ("8. CPT Cache Status", run_cpt_cache_status, [aid], False),
        ("9. CPT Health Report", run_cpt_health_report, [aid], False),
        ("10. CPT Stability", run_cpt_stability_validation, [aid], False),
        ("11. Semantic Score", run_semantic_score_analysis, [aid], False),
        
        # Fase 4: Diepe Diagnostiek
        ("12. Node-Level Diagnostics", run_node_level_diagnostics_logic, [aid], False),
        
        # Fase 5: Performance & Readiness
        ("13. Prediction Accuracy", run_prediction_accuracy_report, [aid], False),
        ("14. Position Prediction Accuracy", run_position_prediction_accuracy_report, [aid], False),
        ("15. Walk-Forward Validation", run_walk_forward_validation, [aid], False),
        ("16. Backtest Simulation", run_backtest_simulation, [], skip_backtest),
        ("17. Production Readiness", run_production_readiness_check, [aid], False),
    ]

    issues = []
    skipped = []
    
    for i, (name, func, args, skip) in enumerate(steps, 1):
        if skip:
            print(f"\n--- ⏭️ Stap {i}/{len(steps)}: {name} [OVERGESLAGEN] ---")
            skipped.append(name)
            continue
            
        print(f"\n--- ⏳ Stap {i}/{len(steps)}: {name} ---")
        
        try:
            # REASON: Bepaal hoe de functie aangeroepen moet worden
            # Sommige functies accepteren geen auto parameter
            if func == run_node_level_diagnostics_logic:
                func(*args)
            elif func == run_database_stats:
                func(*args)
            elif func == run_barrier_benchmarks:
                func(auto=True)
            elif func == run_backtest_simulation:
                func(auto=True)
            else:
                func(*args, auto=True)
            
        except Exception as e:
            err_msg = f"❌ CRITICAL FOUT bij {name}: {e}"
            print(err_msg)
            issues.append(err_msg)
            
    # Scan voor issues in de gegenereerde rapporten van deze run
    try:
        aid_str = str(aid) if aid > 0 else "all"
        asset_dir = PROJECT_ROOT / '_validation' / f'asset_{aid_str}'
        if asset_dir.exists():
            for report in asset_dir.rglob("*.md"):
                # Alleen rapporten van de laatste 10 minuten (deze run)
                if (time.time() - report.stat().st_mtime) < 600:
                    with open(report, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '⚠️' in content or '❌' in content or 'FAILED' in content.upper():
                            # Haal de specifieke regel met het issue
                            for line in content.split('\n'):
                                if '⚠️' in line or '❌' in line or 'FAILED' in line.upper():
                                    issues.append(f"{report.parent.name}: {line.strip()}")
    except Exception as e:
        print(f"⚠️ Fout bij scannen rapporten voor samenvatting: {e}")

    print("\n" + "="*60)
    print("🏁 VALIDATIE CYCLE VOLTOOID")
    print("="*60)
    
    # REASON: Rapporteer uitgevoerde, overgeslagen en gefaalde stappen
    executed_count = len(steps) - len(skipped) - len([i for i in issues if 'CRITICAL' in i])
    print(f"\n📊 Samenvatting: {executed_count} uitgevoerd, {len(skipped)} overgeslagen")
    
    if skipped:
        print(f"\n⏭️  Overgeslagen Stappen ({len(skipped)}):")
        for step in skipped:
            print(f"  - {step}")
    
    if issues:
        print(f"\n⚠️  Gevonden Issues ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Geen warnings of errors gevonden in de logs.")
        
    print("="*60)
    input("\nDruk op Enter om terug naar het menu...")


def handle_choice(choice: str):
    """Dispatch voor validation container"""
    handlers = {
        '1': run_database_stats, 
        '2': run_barrier_validation, 
        '3': run_barrier_coverage_status,
        '4': run_barrier_benchmarks, 
        '5': run_signal_classification_check, 
        '6': run_ida_validation,
        '7': run_concordance_analysis, 
        '8': run_cpt_cache_status, 
        '9': run_cpt_health_report,
        '10': run_cpt_stability_validation, 
        '11': run_semantic_score_analysis,
        '12': run_node_level_diagnostics, 
        '13': run_prediction_accuracy_report,
        '14': run_position_prediction_accuracy_report, 
        '15': run_walk_forward_validation,
        '16': run_backtest_simulation, 
        '17': run_production_readiness_check,
        '18': run_grid_search_configurator,
        '19': generate_validation_report_logic, 
        '20': run_archive_reports, 
        '21': run_full_validation_cycle,
    }
    
    if choice in handlers:
        # Voor sommige handlers moeten we eerst een asset_id vragen als we in het menu zitten
        need_aid = ['1', '3', '5', '8', '9', '11']
        if choice in need_aid:
            aid_str = input("Asset ID (0 voor alle) [0]: ").strip() or "0"
            handlers[choice](asset_id=int(aid_str))
        else:
            handlers[choice]()
        return True
    elif choice == '99': return True
    elif choice == '0': return False
    else: print("\n⚠️  Ongeldige keuze"); time.sleep(1); return True


def run():
    """Main loop voor validation menu"""
    from menus.shared import wait_for_database
    if not wait_for_database(): print("⚠️  Doorgaan zonder database connectie...")

    # REASON: Startup check voor threshold config
    try:
        from config.threshold_loader import ThresholdLoader
        from core.config_warnings import warn_fallback_active
        
        # Check Asset 1 als baseline
        status = ThresholdLoader.check_database_availability(asset_id=1)
        if not status.get('available') or status.get('status') == 'incomplete':
            warn_fallback_active(
                component="ValidationContainer",
                config_name="global_threshold_config",
                fallback_values={'status': status.get('status', 'unknown')},
                reason="Geen of incomplete threshold configuratie in database",
                fix_command="Draai 'Threshold Optimalisatie' in de Training Container"
            )
    except Exception as e:
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
        clear_screen(); print_header('validation'); choice = show_menu()
        if not handle_choice(choice): print("\n👋 Tot ziens!\n"); break

if __name__ == '__main__':
    run()
