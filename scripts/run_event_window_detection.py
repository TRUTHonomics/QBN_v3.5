#!/usr/bin/env python3
"""
Wrapper script voor Event Window Detection in Dagster context.

REASON: EventWindowDetector vereist data loading + config + save-helpers.
Dit script repliceert de flow uit training_menu.py maar als standalone script.
"""
import argparse
import sys
import logging
from pathlib import Path

# Zorg dat /app in sys.path staat
sys.path.insert(0, "/app")

from inference.event_window_detector import (
    EventWindowDetector,
    EventWindowConfig,
    load_barrier_outcomes,
    save_event_labels_to_db,
    save_events_to_cache,
)
from config.threshold_loader import ThresholdLoader
from core.step_validation import validate_step_input, log_handshake_out, StepValidationError
from database.db import get_cursor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s, %(message)s")


def _get_optimal_leading_thresholds(asset_id: int) -> tuple[float, float]:
    """Haal optimale thresholds uit database (gebruikt door training menu)."""
    loader = ThresholdLoader(asset_id=asset_id, horizon="1h")
    strong = loader.composite_strong_threshold
    delta = 0.08  # Default delta (menu gebruikt hardcoded 0.08)
    return strong, delta


def main():
    parser = argparse.ArgumentParser(description="Event Window Detection voor Dagster")
    parser.add_argument("--asset-id", type=int, required=True, help="Asset ID")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for traceability")
    args = parser.parse_args()

    asset_id = args.asset_id
    run_id = args.run_id
    logger.info(f"Starting event detection for asset {asset_id} (run_id={run_id})")

    # Load barrier data
    barrier_data = load_barrier_outcomes(asset_id)
    if barrier_data.empty:
        logger.error(f"Geen barrier data voor asset {asset_id}")
        sys.exit(1)

    # Validation guards: check upstream barrier_outcomes + composite_threshold_config
    if run_id:
        try:
            with get_cursor() as cur:
                # Check barrier_outcomes met run_id
                validate_step_input(
                    conn=cur.connection,
                    step_name="event_window_detection_barriers",
                    upstream_table="qbn.barrier_outcomes",
                    asset_id=asset_id,
                    run_id=None,  # REASON: barrier_outcomes is global, heeft geen run_id kolom
                    min_rows=50,
                    log_run_id=run_id  # REASON: Log wel de echte run_id voor traceability
                )
                # Check composite_threshold_config
                validate_step_input(
                    conn=cur.connection,
                    step_name="event_window_detection_thresholds",
                    upstream_table="qbn.composite_threshold_config",
                    asset_id=asset_id,
                    run_id=run_id,
                    min_rows=1
                )
        except StepValidationError as e:
            logger.info(f"Upstream validation note: {e}")
        except Exception as e:
            logger.warning(f"Upstream validation failed (DB issue): {e}")

    # Haal optimale drempels uit DB
    opt_strong, opt_delta = _get_optimal_leading_thresholds(asset_id)
    logger.info(f"Using thresholds: strong={opt_strong:.2f}, delta={opt_delta:.2f}")

    # Config
    config = EventWindowConfig(
        absolute_threshold=opt_strong, delta_threshold=opt_delta, max_window_minutes=1440
    )

    # Detect
    detector = EventWindowDetector(config)
    events, labeled_data = detector.detect_events(barrier_data, asset_id)

    # Save
    save_event_labels_to_db(asset_id, labeled_data, run_id=run_id)
    save_events_to_cache(events, run_id=run_id)

    logger.info(f"✅ {len(events)} events gedetecteerd en opgeslagen (run_id={run_id})")


if __name__ == "__main__":
    main()
