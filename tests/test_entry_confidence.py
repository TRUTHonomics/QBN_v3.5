import pytest
import numpy as np
from inference.entry_confidence_generator import EntryConfidenceGenerator
from inference.node_types import CompositeState

class TestEntryConfidenceDerivation:
    """Test de derive_confidence logica."""

    def setup_method(self):
        self.generator = EntryConfidenceGenerator()

    @pytest.mark.parametrize("coinc,conf,expected", [
        # HIGH: Aligned bullish
        ('strong_bullish', 'strong_bullish', 'high'),
        ('strong_bullish', 'bullish', 'high'),
        ('bullish', 'strong_bullish', 'high'),
        ('bullish', 'bullish', 'high'),
        # HIGH: Aligned bearish
        ('strong_bearish', 'strong_bearish', 'high'),
        ('strong_bearish', 'bearish', 'high'),
        ('bearish', 'strong_bearish', 'high'),
        ('bearish', 'bearish', 'high'),
        # LOW: Opposing
        ('strong_bullish', 'strong_bearish', 'low'),
        ('bullish', 'bearish', 'low'),
        ('bearish', 'bullish', 'low'),
        ('strong_bearish', 'strong_bullish', 'low'),
        # MEDIUM: Neutral combinations
        ('neutral', 'neutral', 'medium'),
        ('bullish', 'neutral', 'medium'),
        ('neutral', 'bearish', 'medium'),
        ('strong_bullish', 'neutral', 'medium'),
        ('neutral', 'strong_bearish', 'medium'),
    ])
    def test_derive_confidence(self, coinc, conf, expected):
        """Test all alignment scenarios."""
        result = self.generator.derive_confidence(coinc, conf)
        assert result == expected, f"Expected {expected} for ({coinc}, {conf}), got {result}"


class TestAlignmentScore:
    """Test numerieke alignment score berekening."""

    def setup_method(self):
        self.generator = EntryConfidenceGenerator()

    def test_max_alignment_bullish(self):
        score = self.generator.compute_alignment_score('strong_bullish', 'strong_bullish')
        assert score == 1.0

    def test_max_alignment_bearish(self):
        score = self.generator.compute_alignment_score('strong_bearish', 'strong_bearish')
        assert score == 1.0

    def test_max_opposing(self):
        score = self.generator.compute_alignment_score('strong_bullish', 'strong_bearish')
        assert score == -1.0

    def test_neutral(self):
        score = self.generator.compute_alignment_score('neutral', 'neutral')
        assert score == 0.0
        
    def test_partial_alignment(self):
        # 1.0 * 1.0 / 4.0 = 0.25
        score = self.generator.compute_alignment_score('bullish', 'bullish')
        assert score == 0.25


class TestCPTGeneration:
    """Test CPT generatie."""

    def setup_method(self):
        self.generator = EntryConfidenceGenerator(laplace_alpha=1.0)

    def test_deterministic_cpt_completeness(self):
        """Verify all 25 parent combinations are covered."""
        # Note: current generate_cpt implementation needs training_data as List[Dict]
        cpt = self.generator.generate_cpt([])
        assert len(cpt) == 25  # 5x5 parent combinations

    def test_cpt_probabilities_sum_to_one(self):
        """Verify probability distributions are valid."""
        cpt = self.generator.generate_cpt([])
        for key, distribution in cpt.items():
            total = sum(distribution.values())
            assert abs(total - 1.0) < 1e-6, f"Distribution for {key} sums to {total}"

