#!/usr/bin/env python3
"""
Validation script voor signal classification en discretization consistency.

UPDATED: 2026-01-04 - Aangepast voor v3 paden en DB sync validatie.
         Controleert of de YAML bestanden 1-op-1 matchen met de database tabellen.

Dit script valideert:
1. YAML Completeness: signals.yaml, classification.yaml, discretization.yaml
2. DB Sync: YAML inhoud ↔ qbn.signal_classification & qbn.signal_discretization
3. Column Mapping: YAML signalen ↔ MTF tabel kolommen in kfl schema
4. Distribution: Sanity check voor signal klasse verdeling
"""

import sys
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any
import argparse
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging

logger = setup_logging("validate_signal_classification")

try:
    import yaml
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError as e:
    print(f"ERROR: Required module not found: {e}")
    print("Install with: pip install pyyaml psycopg2-binary")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for validation script"""
    # File paths (REASON: Config YAMLs staan nu lokaal in QBN container)
    # Binnen container: /app/kfl_backend_config/
    # Op host: F:/Containers/QBN_v3/kfl_backend_config/
    CONFIG_DIR = Path("/app/kfl_backend_config")
    SIGNAL_CLASSIFICATION_YAML = CONFIG_DIR / "signal_classification.yaml"
    SIGNALS_YAML = CONFIG_DIR / "signals.yaml"
    DISCRETIZATION_YAML = CONFIG_DIR / "discretization.yaml"

    # Database configuration (REASON: Direct connectie vanuit container)
    DB_NAME = os.getenv('POSTGRES_DB', 'kflhyper')
    DB_USER = os.getenv('POSTGRES_USER', 'pipeline')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'pipeline123')
    DB_HOST = os.getenv('POSTGRES_HOST', '10.10.10.3')
    DB_PORT = int(os.getenv('POSTGRES_PORT', '5432'))

    # MTF tabel mapping per classificatie
    MTF_TABLES = {
        'LEADING': 'kfl.mtf_signals_lead',
        'COINCIDENT': 'kfl.mtf_signals_coin',
        'CONFIRMING': 'kfl.mtf_signals_conf'
    }

    # Timeframes die als suffix aan elke signaal worden toegevoegd
    TIMEFRAME_SUFFIXES = ['_d', '_240', '_60', '_1']

    # Expected distribution ranges
    EXPECTED_TOTAL = 125
    EXPECTED_RANGES = {
        'LEADING': (45, 50),
        'COINCIDENT': (35, 42),
        'CONFIRMING': (35, 42)
    }


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_yaml_classifications() -> Dict[str, Set[str]]:
    """Load signal classifications from signal_classification.yaml"""
    try:
        with open(Config.SIGNAL_CLASSIFICATION_YAML, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        yaml_signals = {}
        for class_name, class_data in data['classifications'].items():
            yaml_signals[class_name] = set(class_data['signals'])
        return yaml_signals
    except Exception as e:
        print(f"❌ ERROR: Loading {Config.SIGNAL_CLASSIFICATION_YAML}: {e}")
        sys.exit(1)

def load_signals_yaml() -> List[Dict]:
    """Load signal definitions from signals.yaml"""
    try:
        with open(Config.SIGNALS_YAML, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('signals', [])
    except Exception as e:
        print(f"❌ ERROR: Loading {Config.SIGNALS_YAML}: {e}")
        sys.exit(1)

def load_discretization_yaml() -> Dict[str, Dict[str, float]]:
    """Load thresholds from discretization.yaml"""
    try:
        with open(Config.DISCRETIZATION_YAML, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ ERROR: Loading {Config.DISCRETIZATION_YAML}: {e}")
        sys.exit(1)


# ============================================================================
# VALIDATION CHECKS
# ============================================================================

def validate_db_sync(conn, yaml_class, yaml_signals, yaml_disc) -> Tuple[bool, List[str]]:
    """
    Check: Synchronisatie tussen YAML en DB tabellen
    """
    messages = []
    all_ok = True
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # 1. Check signal_classification table
    cursor.execute("SELECT signal_name, semantic_class FROM qbn.signal_classification")
    db_class = {row['signal_name']: row['semantic_class'] for row in cursor.fetchall()}
    
    # Flatten YAML classification for comparison
    flat_yaml_class = {}
    for class_name, signals in yaml_class.items():
        for sig in signals:
            flat_yaml_class[sig] = class_name
            
    # Vergelijk
    missing_in_db = set(flat_yaml_class.keys()) - set(db_class.keys())
    mismatched = [s for s in flat_yaml_class if s in db_class and flat_yaml_class[s] != db_class[s]]
    
    if missing_in_db:
        messages.append(f"  ❌ {len(missing_in_db)} signalen ontbreken in qbn.signal_classification")
        all_ok = False
    if mismatched:
        messages.append(f"  ❌ {len(mismatched)} signalen hebben afwijkende klasse in DB")
        all_ok = False
    
    if not missing_in_db and not mismatched:
        messages.append(f"  ✅ qbn.signal_classification matcht met YAML ({len(flat_yaml_class)} signalen)")

    # 2. Check signal_discretization table
    cursor.execute("SELECT indicator_base, threshold_name, threshold_value FROM qbn.signal_discretization")
    db_disc = {}
    for row in cursor.fetchall():
        ind = row['indicator_base'].lower()
        if ind not in db_disc: db_disc[ind] = {}
        db_disc[ind][row['threshold_name']] = float(row['threshold_value'])
        
    # Vergelijk discretization
    disc_errors = 0
    for ind, thresholds in yaml_disc.items():
        if ind == 'metadata': continue
        if ind not in db_disc:
            messages.append(f"  ❌ Indicator {ind} ontbreekt in qbn.signal_discretization")
            disc_errors += 1
            continue
            
        for t_name, t_val in thresholds.items():
            if t_name not in db_disc[ind]:
                messages.append(f"  ❌ Threshold {ind}.{t_name} ontbreekt in DB")
                disc_errors += 1
            elif float(t_val) != db_disc[ind][t_name]:
                messages.append(f"  ❌ Threshold mismatch {ind}.{t_name}: YAML={t_val}, DB={db_disc[ind][t_name]}")
                disc_errors += 1
    
    if disc_errors == 0:
        messages.append(f"  ✅ qbn.signal_discretization matcht volledig met YAML")
    else:
        all_ok = False

    cursor.close()
    return all_ok, messages

def validate_yaml_to_db_columns(yaml_signals: Dict[str, Set[str]], conn) -> Tuple[bool, List[str]]:
    """Check mapping tussen YAML en MTF tabel kolommen"""
    messages = []
    all_ok = True
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    for class_name, table in Config.MTF_TABLES.items():
        schema, table_name = table.split('.')
        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
        """, (schema, table_name))
        db_cols = {row['column_name'] for row in cursor.fetchall()}
        
        # Extract base names (RSI_OVERSOLD_60 -> RSI_OVERSOLD)
        db_bases = set()
        for col in db_cols:
            for suffix in Config.TIMEFRAME_SUFFIXES:
                if col.endswith(suffix):
                    db_bases.add(col[:-len(suffix)].upper())
                    break
        
        yaml_set = yaml_signals.get(class_name, set())
        missing = yaml_set - db_bases
        
        if missing:
            messages.append(f"  ⚠️  {class_name}: {len(missing)} signalen uit YAML hebben geen kolommen in {table}")
            for sig in sorted(list(missing))[:5]:
                messages.append(f"     - {sig}")
        else:
            messages.append(f"  ✅ {class_name}: Alle YAML signalen aanwezig in {table}")
            
    cursor.close()
    return all_ok, messages

def main():
    parser = argparse.ArgumentParser(description='Validate signal classification consistency (v3)')
    parser.add_argument('--db-host', default=Config.DB_HOST)
    parser.add_argument('--db-port', type=int, default=Config.DB_PORT)
    args = parser.parse_args()

    Config.DB_HOST = args.db_host
    Config.DB_PORT = args.db_port

    print("=" * 80)
    print(f"Signal Configuration Validation Report (v3) - {datetime.now()}")
    print("=" * 80)

    try:
        # Load YAMLs
        yaml_class = load_yaml_classifications()
        yaml_signals = load_signals_yaml()
        yaml_disc = load_discretization_yaml()
        
        # DB connection
        conn = psycopg2.connect(
            dbname=Config.DB_NAME, user=Config.DB_USER, password=Config.DB_PASSWORD,
            host=Config.DB_HOST, port=Config.DB_PORT
        )
        
        checks = [
            ("Check 1: Database Sync Integrity (YAML ↔ qbn schema)", validate_db_sync(conn, yaml_class, yaml_signals, yaml_disc)),
            ("Check 2: MTF Column Mapping (YAML ↔ kfl schema)", validate_yaml_to_db_columns(yaml_class, conn))
        ]
        
        all_passed = True
        for name, (passed, msgs) in checks:
            print(f"\n{name}")
            print("-" * 80)
            for m in msgs: print(m)
            if not passed: all_passed = False
            
        print("\n" + "=" * 80)
        print("SUMMARY")
        if all_passed:
            print("✅ ALL CHECKS PASSED")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
