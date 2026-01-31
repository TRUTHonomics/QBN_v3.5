# =============================================================================
# Database Sync Script voor Signal Configuraties
# =============================================================================
# 
# Dit script synchroniseert de lokale YAML configuratiebestanden naar de
# PostgreSQL database (qbn schema). Dit zorgt voor een "Source of Truth"
# die voor alle containers (real-time, backfill, inference) gelijk is.
#
# Bronnen (lokaal in QBN_v3 container):
# - /app/kfl_backend_config/signals.yaml
# - /app/kfl_backend_config/signal_classification.yaml
# - /app/kfl_backend_config/discretization.yaml
#
# Doelen:
# - qbn.signal_classification
# - qbn.signal_discretization
#
# REASON: Dit script is gekopieerd vanuit KFL_backend_v3 en aangepast voor
# de lokale config locatie in de QBN_v3 container.
# =============================================================================

import os
import sys
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values

# REASON: Volg logregels voor Database Sync
def setup_logging():
    # Bepaal project root op basis van dit script
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "_log"
    archive_dir = log_dir / "archive"
    log_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    script_name = "db_sync"
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Archiveer oude logs
    for old_log in log_dir.glob(f"{script_name}_*.log"):
        try:
            shutil.move(str(old_log), str(archive_dir / old_log.name))
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True
    )
    l = logging.getLogger(__name__)
    l.info(f"🚀 New {script_name} run started. Logging to: {log_file}")
    return l

# Configuratie paden - aangepast voor QBN_v3 container
# REASON: Config YAMLs staan nu lokaal in /app/kfl_backend_config
CONFIG_DIR = Path('/app/kfl_backend_config')
SIGNALS_YAML = CONFIG_DIR / 'signals.yaml'
SIGNAL_CLASS_YAML = CONFIG_DIR / 'signal_classification.yaml'
DISCRETIZATION_YAML = CONFIG_DIR / 'discretization.yaml'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = setup_logging()


def get_db_connection():
    """Maak database connectie op basis van environment variabelen."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', '10.10.10.3'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'kflhyper'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline123')
    )


def sync_signal_classification():
    """Synchroniseer signals.yaml en signal_classification.yaml naar qbn.signal_classification."""
    logger.info("Synchroniseren van signal classificaties...")
    
    if not SIGNALS_YAML.exists():
        logger.error(f"signals.yaml niet gevonden: {SIGNALS_YAML}")
        return False
    if not SIGNAL_CLASS_YAML.exists():
        logger.error(f"signal_classification.yaml niet gevonden: {SIGNAL_CLASS_YAML}")
        return False

    with open(SIGNALS_YAML, 'r', encoding='utf-8') as f:
        signals_data = yaml.safe_load(f)
    
    with open(SIGNAL_CLASS_YAML, 'r', encoding='utf-8') as f:
        class_data = yaml.safe_load(f)

    # Build lookup van classifications
    class_lookup = {}
    for cat, data in class_data.get('classifications', {}).items():
        for sig_id in data.get('signals', []):
            class_lookup[sig_id] = cat

    # Bereid data voor UPSERT
    upsert_data = []
    polarity_map = {
        'long': 'bullish',
        'short': 'bearish',
        'neutral': 'neutral',
        'bullish': 'bullish',
        'bearish': 'bearish'
    }
    
    for sig in signals_data.get('signals', []):
        sig_id = sig.get('id')
        sem_class = class_lookup.get(sig_id, sig.get('classification', 'UNKNOWN')).upper()
        
        raw_polarity = sig.get('polarity', 'NEUTRAL').lower()
        mapped_polarity = polarity_map.get(raw_polarity, raw_polarity)
        
        upsert_data.append((
            sig_id,
            sem_class,
            sig.get('indicator', 'UNKNOWN'),
            str(sig.get('parameters', {})), # variant info
            mapped_polarity,
            sig.get('name', '')
        ))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Upsert naar database
            query = """
                INSERT INTO qbn.signal_classification 
                (signal_name, semantic_class, indicator_base, indicator_variant, polarity, description)
                VALUES %s
                ON CONFLICT (signal_name) DO UPDATE SET
                    semantic_class = EXCLUDED.semantic_class,
                    indicator_base = EXCLUDED.indicator_base,
                    indicator_variant = EXCLUDED.indicator_variant,
                    polarity = EXCLUDED.polarity,
                    description = EXCLUDED.description,
                    updated_at = NOW()
            """
            execute_values(cur, query, upsert_data)
            conn.commit()
            logger.info(f"✅ {len(upsert_data)} signalen gesynchroniseerd naar qbn.signal_classification")
            return True
    except Exception as e:
        logger.error(f"❌ Fout bij sync signal_classification: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def sync_discretization():
    """Synchroniseer discretization.yaml naar qbn.signal_discretization."""
    logger.info("Synchroniseren van discretisatie thresholds...")
    
    if not DISCRETIZATION_YAML.exists():
        logger.error(f"Discretization file niet gevonden: {DISCRETIZATION_YAML}")
        return False

    with open(DISCRETIZATION_YAML, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Bereid data voor UPSERT
    upsert_data = []
    for indicator, thresholds in data.items():
        if indicator == 'metadata':
            continue
        
        # REASON: Handle nested structures like 'horizons' from threshold analysis
        if indicator == 'horizons':
            for horizon, horizon_data in thresholds.items():
                if isinstance(horizon_data, dict):
                    for key, val in horizon_data.items():
                        if key in ['composite', 'alignment'] and isinstance(val, dict):
                            for t_name, t_val in val.items():
                                upsert_data.append((
                                    f"{key.upper()}_{horizon.upper()}",
                                    t_name,
                                    float(t_val),
                                    f"Threshold uit threshold_analysis voor {horizon}"
                                ))
            continue
            
        if not isinstance(thresholds, dict):
            continue
            
        for t_name, t_val in thresholds.items():
            try:
                upsert_data.append((
                    indicator.upper(),
                    t_name,
                    float(t_val),
                    f"Threshold geladen uit YAML voor {indicator}"
                ))
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-numeric threshold: {indicator}.{t_name} = {t_val}")
                continue

    if not upsert_data:
        logger.warning("Geen valid thresholds gevonden om te synchroniseren")
        return False

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO qbn.signal_discretization 
                (indicator_base, threshold_name, threshold_value, description)
                VALUES %s
                ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
                    threshold_value = EXCLUDED.threshold_value,
                    description = EXCLUDED.description,
                    updated_at = NOW()
            """
            execute_values(cur, query, upsert_data)
            conn.commit()
            logger.info(f"✅ {len(upsert_data)} thresholds gesynchroniseerd naar qbn.signal_discretization")
            return True
    except Exception as e:
        logger.error(f"❌ Fout bij sync signal_discretization: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def main():
    """Main entry point voor sync."""
    logger.info(f"Config directory: {CONFIG_DIR}")
    logger.info(f"  signals.yaml exists: {SIGNALS_YAML.exists()}")
    logger.info(f"  signal_classification.yaml exists: {SIGNAL_CLASS_YAML.exists()}")
    logger.info(f"  discretization.yaml exists: {DISCRETIZATION_YAML.exists()}")
    
    success = True
    if not sync_signal_classification():
        success = False
    if not sync_discretization():
        success = False
    
    if not success:
        sys.exit(1)
    
    logger.info("🚀 Database configuratie sync voltooid.")


if __name__ == "__main__":
    main()

