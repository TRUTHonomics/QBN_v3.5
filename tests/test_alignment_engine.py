import pytest
import numpy as np
from inference.alignment_engine import (
    AlignmentEngine, 
    AlignmentConfig, 
    AlignmentCategory, 
    DirectionGroup
)

class TestAlignmentCategories:
    """Test alle alignment categorie bepalingen."""

    def setup_method(self):
        self.engine = AlignmentEngine()

    # === FULL ALIGNED TESTS ===
    @pytest.mark.parametrize("coinc,conf", [
        ('strong_bullish', 'strong_bullish'),
        ('strong_bullish', 'bullish'),
        ('bullish', 'bullish'),
        ('bearish', 'bearish'),
        ('strong_bearish', 'bearish'),
        ('strong_bearish', 'strong_bearish'),
    ])
    def test_full_aligned(self, coinc, conf):
        result = self.engine.get_alignment(coinc, conf)
        assert result.category == AlignmentCategory.FULL_ALIGNED
        assert result.confidence == 'high'

    # === OPPOSING TESTS ===
    @pytest.mark.parametrize("coinc,conf", [
        ('strong_bullish', 'bearish'),
        ('strong_bullish', 'strong_bearish'),
        ('bullish', 'bearish'),
        ('bullish', 'strong_bearish'),
        ('bearish', 'bullish'),
        ('bearish', 'strong_bullish'),
        ('strong_bearish', 'bullish'),
        ('strong_bearish', 'strong_bullish'),
    ])
    def test_opposing(self, coinc, conf):
        result = self.engine.get_alignment(coinc, conf)
        assert result.category == AlignmentCategory.OPPOSING
        assert result.confidence == 'low'

    # === PARTIAL TESTS ===
    @pytest.mark.parametrize("coinc,conf", [
        ('strong_bullish', 'neutral'),
        ('bullish', 'neutral'),
        ('neutral', 'bullish'),
        ('neutral', 'strong_bullish'),
        ('neutral', 'bearish'),
        ('neutral', 'strong_bearish'),
        ('bearish', 'neutral'),
        ('strong_bearish', 'neutral'),
    ])
    def test_partial(self, coinc, conf):
        result = self.engine.get_alignment(coinc, conf)
        assert result.category == AlignmentCategory.PARTIAL
        assert result.confidence == 'medium'

    # === NEUTRAL TEST ===
    def test_double_neutral(self):
        result = self.engine.get_alignment('neutral', 'neutral')
        assert result.category == AlignmentCategory.NEUTRAL
        assert result.confidence == 'medium'

class TestAlignmentScores:
    """Test numerieke alignment score berekeningen."""

    def setup_method(self):
        self.engine = AlignmentEngine()

    def test_max_positive_alignment(self):
        result = self.engine.get_alignment('strong_bullish', 'strong_bullish')
        assert result.score == 1.0

    def test_max_negative_alignment(self):
        result = self.engine.get_alignment('strong_bullish', 'strong_bearish')
        assert result.score == -1.0

    def test_neutral_score(self):
        result = self.engine.get_alignment('neutral', 'neutral')
        assert result.score == 0.0

    def test_asymmetric_positive(self):
        # (2.0 * 1.0) / 4.0 = 0.5
        result = self.engine.get_alignment('strong_bullish', 'bullish')
        assert result.score == 0.5

    def test_score_symmetry(self):
        """Score moet symmetrisch zijn voor coinc/conf swap."""
        res1 = self.engine.get_alignment('strong_bullish', 'bullish')
        res2 = self.engine.get_alignment('bullish', 'strong_bullish')
        assert res1.score == res2.score

class TestAlignmentConfig:
    """Test configuratie varianten."""

    def test_conservative_config(self):
        config = AlignmentConfig.conservative()
        engine = AlignmentEngine(config)
        
        # In conservative is high_threshold 0.40 en min_strength 0.75
        # strong_bullish (2) + bullish (1) -> score 0.5, strength (2+1)/4 = 0.75
        # Dit zou 'high' moeten zijn
        res = engine.get_alignment('strong_bullish', 'bullish')
        assert res.confidence == 'high'
        
        # bullish (1) + bullish (1) -> score 0.25, strength 0.5
        # Dit zou 'medium' moeten zijn in conservative (threshold 0.40)
        res = engine.get_alignment('bullish', 'bullish')
        assert res.confidence == 'medium'

    def test_aggressive_config(self):
        config = AlignmentConfig.aggressive()
        engine = AlignmentEngine(config)
        
        # In aggressive is high_threshold 0.10
        # bullish (1) + bullish (1) -> score 0.25
        # Dit is 'high' in aggressive
        res = engine.get_alignment('bullish', 'bullish')
        assert res.confidence == 'high'

