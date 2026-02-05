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
    args = parser.parse_args()

    asset_id = args.asset_id
    logger.info(f"Starting event detection for asset {asset_id}")

    # Load barrier data
    barrier_data = load_barrier_outcomes(asset_id)
    if barrier_data.empty:
        logger.error(f"Geen barrier data voor asset {asset_id}")
        sys.exit(1)

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
    save_event_labels_to_db(asset_id, labeled_data)
    save_events_to_cache(events)

    logger.info(f"✅ {len(events)} events gedetecteerd en opgeslagen")


if __name__ == "__main__":
    main()
