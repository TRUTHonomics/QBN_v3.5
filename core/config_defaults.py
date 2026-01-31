"""
Centrale configuratie defaults voor QBN v3.1.
ALLE defaults moeten hier staan - NERGENS ANDERS hardcoded!
"""

# =============================================================================
# COMPOSITE THRESHOLDS
# Gebaseerd op typische score distributie: mean~0, std~0.08, range [-0.21, 0.19]
# =============================================================================
DEFAULT_COMPOSITE_NEUTRAL_BAND = 0.05
DEFAULT_COMPOSITE_STRONG_THRESHOLD = 0.15  # NIET 0.5!

# =============================================================================
# ALIGNMENT THRESHOLDS
# =============================================================================
DEFAULT_ALIGNMENT_HIGH_THRESHOLD = 0.10
DEFAULT_ALIGNMENT_LOW_THRESHOLD = -0.10

# =============================================================================
# EVENT WINDOW DETECTION
# =============================================================================
DEFAULT_EVENT_ABSOLUTE_THRESHOLD = 0.15
DEFAULT_EVENT_DELTA_THRESHOLD = 0.08
DEFAULT_EVENT_MAX_WINDOW_MINUTES = 1440

# =============================================================================
# BARRIER CONFIGURATION
# =============================================================================
DEFAULT_SIGNIFICANT_BARRIER_ATR = 0.75

# =============================================================================
# POSITION DELTA THRESHOLDS (v3.2 / v3.4)
# Voor Position_Confidence en Position_Prediction delta-based training
# =============================================================================
DEFAULT_DELTA_THRESHOLD_LEADING = 0.03  # Momentum (meest sensitief)
DEFAULT_DELTA_THRESHOLD_COINCIDENT = 0.08  # Volatility
DEFAULT_DELTA_THRESHOLD_CONFIRMING = 0.10  # Exit Timing (meest stabiel)

# Delta states: deteriorating | stable | improving
# Threshold bepaalt grens tussen states:
#   delta < -threshold -> deteriorating
#   delta > +threshold -> improving
#   anders -> stable