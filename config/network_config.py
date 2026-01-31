"""
Configuration constants for QBN v3 Network and CPT Generation.

IMPORTANT: Composite/Alignment thresholds zijn verplaatst naar de database.
Gebruik ThresholdLoader in plaats van de DEPRECATED constanten hieronder.

MIGRATIE:
    # Oud (deprecated):
    from config.network_config import COMPOSITE_NEUTRAL_BAND, COMPOSITE_STRONG_THRESHOLD
    
    # Nieuw (aanbevolen):
    from config.threshold_loader import ThresholdLoader
    loader = ThresholdLoader(asset_id=1, horizon='1h')
    neutral_band = loader.composite_neutral_band
    strong_threshold = loader.composite_strong_threshold
"""

import warnings
import logging

logger = logging.getLogger(__name__)

# =========================================================================
# CPT Generation Parameters (niet verplaatst naar DB - blijven statisch)
# =========================================================================

MIN_OBS_PER_CELL = 30       # Minimum observations per CPT cell
MIN_TOTAL_OBS = 1000        # Minimum total observations per node for valid CPT
COVERAGE_THRESHOLD = 0.5    # Min 50% coverage for valid CPT (otherwise reduce states)

# CPT Freshness
FRESHNESS_THRESHOLD_HOURS = 24  # Refresh CPTs if older than 24h

# Model Version
# v3.1: Entry_Confidence verwijderd uit Prediction node parents
MODEL_VERSION = '3.1'

# =========================================================================
# DEPRECATED: Composite Aggregation Thresholds
# =========================================================================
# REASON: Deze waarden zijn verplaatst naar qbn.composite_threshold_config in de database.
# Ze worden per asset en horizon geoptimaliseerd via Stap 8 (Threshold Optimalisatie).
# De constanten hieronder dienen alleen als fallback voor backward compatibility.

# DEPRECATED - Gebruik ThresholdLoader.composite_neutral_band
COMPOSITE_NEUTRAL_BAND = 0.15

# DEPRECATED - Gebruik ThresholdLoader.composite_strong_threshold  
COMPOSITE_STRONG_THRESHOLD = 0.5

# DEPRECATED - Gebruik ThresholdLoader.alignment_high_threshold
ALIGNMENT_HIGH_THRESHOLD = 0.17

# DEPRECATED - Gebruik ThresholdLoader.alignment_low_threshold
ALIGNMENT_LOW_THRESHOLD = -0.2


def _emit_deprecation_warning(constant_name: str):
    """Emit deprecation warning bij gebruik van oude constanten."""
    warnings.warn(
        f"{constant_name} is deprecated. Gebruik ThresholdLoader uit config.threshold_loader "
        f"voor asset- en horizon-specifieke thresholds uit de database.",
        DeprecationWarning,
        stacklevel=3
    )


def get_composite_neutral_band() -> float:
    """
    DEPRECATED: Gebruik ThresholdLoader.composite_neutral_band.
    
    Returns fallback waarde voor backward compatibility.
    """
    _emit_deprecation_warning("COMPOSITE_NEUTRAL_BAND")
    return COMPOSITE_NEUTRAL_BAND


def get_composite_strong_threshold() -> float:
    """
    DEPRECATED: Gebruik ThresholdLoader.composite_strong_threshold.
    
    Returns fallback waarde voor backward compatibility.
    """
    _emit_deprecation_warning("COMPOSITE_STRONG_THRESHOLD")
    return COMPOSITE_STRONG_THRESHOLD
