"""
Alignment Engine voor QBN v3.

Berekent de alignment tussen Coincident en Confirming signalen om
timing-zekerheid (Entry_Confidence) te kwantificeren.

Rationale: Voorstel 2 - Timing confidence uit Coincident + Confirming alignment.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

from core.config_defaults import DEFAULT_ALIGNMENT_HIGH_THRESHOLD, DEFAULT_ALIGNMENT_LOW_THRESHOLD
from core.config_warnings import warn_fallback_active

class DirectionGroup(Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"

class SignalStrength(Enum):
    STRONG = 2
    MODERATE = 1
    NONE = 0

class AlignmentCategory(Enum):
    FULL_ALIGNED = "FULL_ALIGNED"
    PARTIAL = "PARTIAL"
    NEUTRAL = "NEUTRAL"
    OPPOSING = "OPPOSING"

@dataclass
class AlignmentConfig:
    """Configuratie voor alignment logica."""
    # REASON: Gebruik centrale defaults van core/config_defaults.py
    high_threshold: float = DEFAULT_ALIGNMENT_HIGH_THRESHOLD      # Score >= dit = 'high'
    low_threshold: float = DEFAULT_ALIGNMENT_LOW_THRESHOLD       # Score < dit = 'low'
    coincident_weight: float = 0.5    # Gelijk gewicht (default)
    confirming_weight: float = 0.5
    require_strength_for_high: bool = True
    min_strength_for_high: float = 0.5
    treat_double_neutral_as: str = 'medium'

    @classmethod
    def default(cls) -> 'AlignmentConfig':
        return cls()

    @classmethod
    def from_db(cls, asset_id: int, horizon: str = '1h') -> 'AlignmentConfig':
        """Laad alignment thresholds uit de database."""
        try:
            from config.threshold_loader import ThresholdLoader
            loader = ThresholdLoader(asset_id=asset_id, horizon=horizon)
            
            if loader.is_from_database:
                return cls(
                    high_threshold=loader.alignment_high_threshold,
                    low_threshold=loader.alignment_low_threshold
                )
            else:
                warn_fallback_active(
                    component="AlignmentConfig",
                    config_name=f"asset_{asset_id}_{horizon}_alignment",
                    fallback_values={'high': DEFAULT_ALIGNMENT_HIGH_THRESHOLD, 'low': DEFAULT_ALIGNMENT_LOW_THRESHOLD},
                    reason="Geen alignment thresholds in DB gevonden",
                    fix_command="Draai 'Threshold Optimalisatie'"
                )
                return cls.default()
        except Exception as e:
            warn_fallback_active(
                component="AlignmentConfig",
                config_name=f"asset_{asset_id}_{horizon}_alignment",
                fallback_values={'high': DEFAULT_ALIGNMENT_HIGH_THRESHOLD, 'low': DEFAULT_ALIGNMENT_LOW_THRESHOLD},
                reason=f"Fout bij database lookup: {e}"
            )
            return cls.default()

    @classmethod
    def conservative(cls) -> 'AlignmentConfig':
        return cls(high_threshold=0.40, low_threshold=-0.10, min_strength_for_high=0.75)

    @classmethod
    def aggressive(cls) -> 'AlignmentConfig':
        return cls(high_threshold=0.10, low_threshold=-0.30, require_strength_for_high=False)

@dataclass
class AlignmentResult:
    """Volledig alignment analyse resultaat."""
    category: AlignmentCategory
    confidence: str  # 'low', 'medium', 'high'
    score: float  # -1.0 tot +1.0
    strength: float  # 0.0 tot 1.0
    direction_agreement: bool
    details: Dict[str, Any] = field(default_factory=dict)

# Mappings conform 2.2_alignment_logic_timing.md
STATE_TO_DIRECTION = {
    'strong_bullish': DirectionGroup.BULLISH,
    'bullish': DirectionGroup.BULLISH,
    'neutral': DirectionGroup.NEUTRAL,
    'bearish': DirectionGroup.BEARISH,
    'strong_bearish': DirectionGroup.BEARISH
}

STATE_TO_STRENGTH = {
    'strong_bullish': SignalStrength.STRONG,
    'bullish': SignalStrength.MODERATE,
    'neutral': SignalStrength.NONE,
    'bearish': SignalStrength.MODERATE,
    'strong_bearish': SignalStrength.STRONG
}

STATE_TO_NUMERIC = {
    'strong_bullish': 2.0,
    'bullish': 1.0,
    'neutral': 0.0,
    'bearish': -1.0,
    'strong_bearish': -2.0
}

class AlignmentEngine:
    """
    High-performance alignment engine voor QBN v3.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig.default()
        self._lookup_table = self._build_lookup_table()

    def _build_lookup_table(self) -> Dict[Tuple[str, str], AlignmentResult]:
        """Pre-compute alle 25 combinaties."""
        table = {}
        states = ['strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish']
        for coinc in states:
            for conf in states:
                table[(coinc, conf)] = self._compute_alignment_runtime(coinc, conf)
        return table

    def get_alignment(self, coincident: str, confirming: str) -> AlignmentResult:
        """O(1) alignment lookup."""
        key = (coincident, confirming)
        if key not in self._lookup_table:
            return self._compute_alignment_runtime(coincident, confirming)
        return self._lookup_table[key]

    def _compute_alignment_runtime(self, coincident: str, confirming: str) -> AlignmentResult:
        """Runtime berekening van alignment."""
        coinc_val = STATE_TO_NUMERIC.get(coincident, 0.0)
        conf_val = STATE_TO_NUMERIC.get(confirming, 0.0)
        
        coinc_dir = STATE_TO_DIRECTION.get(coincident, DirectionGroup.NEUTRAL)
        conf_dir = STATE_TO_DIRECTION.get(confirming, DirectionGroup.NEUTRAL)
        
        coinc_strength = abs(coinc_val)
        conf_strength = abs(conf_val)

        # 1. Score berekening (gewogen)
        # Normaliseer naar [-1, +1] per input
        coinc_norm = coinc_val / 2.0
        conf_norm = conf_val / 2.0
        
        # Product-gebaseerde alignment score (zoals in doc sectie 6.1)
        # we gebruiken hier de product-formule omdat die beter opposing vs neutral onderscheidt
        raw_alignment = (coinc_val * conf_val) / 4.0
        
        # Sterkte
        strength = (coinc_strength + conf_strength) / 4.0

        # 2. Categorie Bepaling
        category = self._derive_category(coinc_dir, conf_dir)

        # 3. Confidence Mapping met config thresholds
        if raw_alignment >= self.config.high_threshold:
            if self.config.require_strength_for_high and strength < self.config.min_strength_for_high:
                confidence = 'medium'
            else:
                confidence = 'high'
        elif raw_alignment < self.config.low_threshold:
            confidence = 'low'
        else:
            confidence = 'medium'

        # Special case: double neutral
        if coincident == 'neutral' and confirming == 'neutral':
            confidence = self.config.treat_double_neutral_as

        return AlignmentResult(
            category=category,
            confidence=confidence,
            score=raw_alignment,
            strength=strength,
            direction_agreement=(coinc_dir == conf_dir or DirectionGroup.NEUTRAL in (coinc_dir, conf_dir)),
            details={
                'coincident_value': coinc_val,
                'confirming_value': conf_val,
                'coincident_direction': coinc_dir.value,
                'confirming_direction': conf_dir.value
            }
        )

    def _derive_category(self, coinc_dir: DirectionGroup, conf_dir: DirectionGroup) -> AlignmentCategory:
        if coinc_dir == conf_dir and coinc_dir != DirectionGroup.NEUTRAL:
            return AlignmentCategory.FULL_ALIGNED
        if coinc_dir == DirectionGroup.NEUTRAL and conf_dir == DirectionGroup.NEUTRAL:
            return AlignmentCategory.NEUTRAL
        if coinc_dir != DirectionGroup.NEUTRAL and conf_dir != DirectionGroup.NEUTRAL:
            return AlignmentCategory.OPPOSING
        return AlignmentCategory.PARTIAL

