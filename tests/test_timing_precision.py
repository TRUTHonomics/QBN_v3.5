import pytest
import pandas as pd
import numpy as np
from inference.entry_confidence_generator import EntryConfidenceGenerator
from inference.timing_precision_analyzer import TimingPrecisionAnalyzer

class TestTimingPrecisionAnalyzer:
    """Test de metrics berekening van de TimingPrecisionAnalyzer."""

    def setup_method(self):
        self.analyzer = TimingPrecisionAnalyzer()

    def test_analyze_from_data(self):
        # Mock data: 100 rows, 55% win rate for high, 50% for medium, 45% for low
        n = 100
        data = pd.DataFrame({
            'entry_confidence': (['high'] * 40) + (['medium'] * 30) + (['low'] * 30),
            'outcome_1h': ([1]*22 + [-1]*18) + ([1]*15 + [-1]*15) + ([1]*12 + [-1]*18),
            'return_1h_pct': ([0.5]*22 + [-0.5]*18) + ([0.1]*15 + [-0.1]*15) + ([0.0]*12 + [-0.8]*18)
        })
        
        metrics = self.analyzer.analyze_from_data(data, horizon='1h')
        
        assert metrics.win_rate_high == 22/40
        assert metrics.win_rate_medium == 15/30
        assert metrics.win_rate_low == 12/30
        assert metrics.is_monotonic == True
        assert metrics.observations == 100

class TestRefinementLogic:
    """Test de outcome refinement logica in de generator."""

    def setup_method(self):
        self.generator = EntryConfidenceGenerator()

    def test_refinement_downgrade(self):
        """Test of 'high' wordt gedowngrade bij slechte win rate."""
        # 100 observations voor strong_bullish/strong_bullish (zou high moeten zijn)
        # Maar we geven ze een win rate van 20%
        n = 100
        data = pd.DataFrame({
            'coincident_composite': ['strong_bullish'] * n,
            'confirming_composite': ['strong_bullish'] * n,
            'outcome_1h': ([1]*20 + [-1]*80) # 20% win rate
        })
        
        # Phase 1: Base distributie
        base_cpt = self.generator.generate_cpt([])
        base_high_prob = base_cpt[('strong_bullish', 'strong_bullish')]['high']
        
        # Phase 2: Refined
        refined_cpt = self.generator._refine_with_outcomes(base_cpt, data, '1h')
        refined_high_prob = refined_cpt[('strong_bullish', 'strong_bullish')]['high']
        
        assert refined_high_prob < base_high_prob
        assert refined_cpt[('strong_bullish', 'strong_bullish')]['medium'] > base_cpt[('strong_bullish', 'strong_bullish')]['medium']

    def test_refinement_upgrade(self):
        """Test of 'medium' wordt geupgrade bij uitstekende win rate."""
        # neutral/neutral is normaal 'medium'
        # we geven ze 80% win rate
        n = 100
        data = pd.DataFrame({
            'coincident_composite': ['neutral'] * n,
            'confirming_composite': ['neutral'] * n,
            'outcome_1h': ([1]*80 + [-1]*20)
        })
        
        base_cpt = self.generator.generate_cpt([])
        base_med_prob = base_cpt[('neutral', 'neutral')]['medium']
        
        refined_cpt = self.generator._refine_with_outcomes(base_cpt, data, '1h')
        
        assert refined_cpt[('neutral', 'neutral')]['high'] > base_cpt[('neutral', 'neutral')]['high']
        assert refined_cpt[('neutral', 'neutral')]['medium'] < base_med_prob

    def test_refinement_no_data(self):
        """Test dat CPT ongewijzigd blijft als er geen data is."""
        base_cpt = self.generator.generate_cpt([])
        empty_data = pd.DataFrame(columns=['coincident_composite', 'confirming_composite', 'outcome_1h'])
        
        refined_cpt = self.generator._refine_with_outcomes(base_cpt, empty_data, '1h')
        
        for key in base_cpt:
            assert refined_cpt[key] == base_cpt[key]

