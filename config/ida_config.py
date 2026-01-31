"""
IDA Configuration - Information-Driven Attribution

Configuratie voor de López de Prado Uniqueness + Soft-Attribution Delta methodiek.
Dit berekent training_weight voor barrier outcomes om seriële correlatie te corrigeren.
"""

from dataclasses import dataclass, field
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class IDAConfig:
    """
    Configuratie voor Information-Driven Attribution.
    
    De IDA-methodiek verdeelt een totaalgewicht van 1.0 over alle signalen 
    die dezelfde barrier-hit claimen, waarbij signalen met hogere 
    "informatiewaarde" (delta + absolute score) meer gewicht krijgen.
    
    Formule per signaal i in cluster:
        effective_score_i = score_i (UP) of -score_i (DOWN)
        Delta_i = max(0, effective_score_i - effective_score_{i-1})
        A_i = (score_weight * |effective_score_i|) + (delta_weight * Delta_i) + epsilon
        Weight_i = A_i / sum(A_j)
    """
    
    # Weging parameters (moeten optellen tot 1.0)
    delta_weight: float = 0.8      # Gewicht voor score-verandering (boodschapper)
    score_weight: float = 0.2      # Gewicht voor absolute score (toestand)
    
    # Stabilisatie
    epsilon: float = 0.01          # Voorkomt zero-weights
    min_weight_floor: float = 0.05 # Minimum weight per signaal (safeguard)
    
    # N_eff monitoring
    neff_warning_ratio: float = 0.3  # Warn als N_eff < 30% van N_raw
    
    # Data leakage preventie
    allowed_tables: List[str] = field(default_factory=lambda: [
        'kfl.mtf_signals_lead',  # Signalen (alle kolommen behalve outcome-gerelateerd)
        'kfl.klines_raw',        # OHLCV data (WHERE time <= time_1)
        'kfl.indicators',        # Indicatoren (WHERE time <= time_1)
    ])
    
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        'barrier_outcomes',      # Outcome data
        'first_significant',     # Barrier hit info  
        'training_weight',       # Meta-info
        'signal_outcomes',       # Legacy outcomes
        'outcome',               # Algemeen outcome-gerelateerd
    ])
    
    def __post_init__(self):
        """Valideer configuratie na initialisatie."""
        self._validate()
    
    def _validate(self):
        """Valideer dat delta_weight + score_weight = 1.0"""
        total = self.delta_weight + self.score_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"delta_weight ({self.delta_weight}) + score_weight ({self.score_weight}) "
                f"= {total}, moet 1.0 zijn"
            )
        
        if self.epsilon <= 0:
            raise ValueError(f"epsilon moet positief zijn, kreeg: {self.epsilon}")
        
        if self.min_weight_floor < 0 or self.min_weight_floor > 0.5:
            raise ValueError(
                f"min_weight_floor moet tussen 0 en 0.5 liggen, kreeg: {self.min_weight_floor}"
            )
        
        if self.neff_warning_ratio <= 0 or self.neff_warning_ratio > 1.0:
            raise ValueError(
                f"neff_warning_ratio moet tussen 0 en 1 liggen, kreeg: {self.neff_warning_ratio}"
            )
    
    def validate_query_no_leakage(self, query: str) -> bool:
        """
        Pre-flight check voor data leakage.
        
        Args:
            query: SQL query string
            
        Returns:
            True als geen leakage gedetecteerd
            
        Raises:
            ValueError: Als een forbidden pattern gevonden wordt
        """
        query_lower = query.lower()
        for pattern in self.forbidden_patterns:
            if pattern in query_lower:
                raise ValueError(f"DATA LEAKAGE DETECTED: '{pattern}' in query")
        return True
    
    @classmethod
    def baseline(cls) -> 'IDAConfig':
        """Baseline configuratie (80/20 split)."""
        return cls(delta_weight=0.8, score_weight=0.2)
    
    @classmethod
    def balanced(cls) -> 'IDAConfig':
        """Gebalanceerde configuratie (50/50 split)."""
        return cls(delta_weight=0.5, score_weight=0.5)
    
    @classmethod
    def delta_only(cls) -> 'IDAConfig':
        """Puur delta-gedreven configuratie."""
        return cls(delta_weight=1.0, score_weight=0.0)
    
    @classmethod
    def aggressive(cls) -> 'IDAConfig':
        """Agressieve delta-bias configuratie (90/10 split)."""
        return cls(delta_weight=0.9, score_weight=0.1)
    
    @classmethod
    def get_ablation_configs(cls) -> dict:
        """
        Retourneer alle configuraties voor ablatiestudie.
        
        Returns:
            Dict met naam -> IDAConfig
        """
        return {
            'baseline': cls.baseline(),
            'balanced': cls.balanced(),
            'delta_only': cls.delta_only(),
            'aggressive': cls.aggressive(),
        }


# REASON: Constanten voor stationariteits-check
STATIONARITY_DEFAULTS = {
    'adf_significance': 0.05,      # p-waarde threshold voor ADF test
    'rolling_window': 1000,         # Window voor rolling mean
    'trend_threshold': 0.001,       # Max acceptabele trend in rolling mean
    'outlier_zscore': 3.0,          # Z-score threshold voor outliers
    'max_outlier_ratio': 0.02,      # Max 2% outliers toegestaan
}
