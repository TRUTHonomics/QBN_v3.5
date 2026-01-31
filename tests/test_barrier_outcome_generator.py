import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from inference.barrier_config import BarrierConfig, BarrierOutcomeResult
from inference.barrier_outcome_generator import BarrierOutcomeGenerator


class TestBarrierConfig:
    """Tests voor BarrierConfig."""
    
    def test_default_config(self):
        """Test default configuratie."""
        config = BarrierConfig()
        assert config.up_barriers == [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        assert config.significant_threshold == 0.75
        assert config.max_observation_min == 2880
    
    def test_validation_empty_barriers(self):
        """Test dat lege barriers ValueError raised."""
        with pytest.raises(ValueError, match="mag niet leeg"):
            BarrierConfig(up_barriers=[])
    
    def test_validation_negative_barrier(self):
        """Test dat negatieve barriers ValueError raised."""
        with pytest.raises(ValueError, match="positief"):
            BarrierConfig(up_barriers=[-0.5, 0.75])
    
    def test_validation_invalid_threshold(self):
        """Test ongeldige significant_threshold."""
        with pytest.raises(ValueError):
            BarrierConfig(significant_threshold=0)


class TestBarrierOutcomeGenerator:
    """Tests voor BarrierOutcomeGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Maak generator met default config."""
        config = BarrierConfig()
        # Mock from_database om DB calls te vermijden bij init
        with patch.object(BarrierConfig, 'from_database', return_value=config):
            return BarrierOutcomeGenerator(config)
    
    @pytest.fixture
    def sample_klines(self):
        """Maak sample kline data."""
        base_price = 100.0
        atr = 2.0
        
        # Simuleer prijsbeweging: eerst licht down, dan up
        data = []
        for i in range(60):  # 60 minuten
            if i < 10:
                # Eerste 10 min: licht down
                high = base_price - (i * 0.05)
                low = high - 0.1
            elif i < 30:
                # 10-30 min: omhoog
                high = base_price + ((i - 10) * 0.1)
                low = high - 0.1
            else:
                # 30-60 min: stabiliseren
                high = base_price + 2.0
                low = base_price + 1.8
            
            data.append({
                'time': datetime(2026, 1, 10, 14, 0) + timedelta(minutes=i+1),
                'high': high,
                'low': low
            })
        
        return pd.DataFrame(data)
    
    def test_validate_inputs_valid(self, generator):
        """Test validatie met geldige inputs."""
        generator._validate_inputs(atr=2.0, ref_price=100.0)  # Should not raise
    
    def test_validate_inputs_invalid_atr(self, generator):
        """Test validatie met ongeldige ATR."""
        with pytest.raises(ValueError, match="ATR"):
            generator._validate_inputs(atr=0, ref_price=100.0)
    
    def test_validate_inputs_invalid_price(self, generator):
        """Test validatie met ongeldige prijs."""
        with pytest.raises(ValueError, match="price"):
            generator._validate_inputs(atr=2.0, ref_price=-100.0)
    
    def test_calculate_barriers_up(self, generator, sample_klines):
        """Test berekening van up barriers."""
        ref_price = 100.0
        atr = 2.0
        barriers = [0.25, 0.50, 0.75]
        
        result = generator._calculate_barriers(
            sample_klines, ref_price, atr, barriers, direction='up'
        )
        
        # +0.25 ATR = 100.5 -> zou rond minuut 15 moeten zijn
        assert result['025'] is not None
        assert result['025'] > 10
        
        # +0.75 ATR = 101.5 -> zou rond minuut 25 moeten zijn
        assert result['075'] is not None
    
    def test_calculate_barriers_down(self, generator, sample_klines):
        """Test berekening van down barriers."""
        ref_price = 100.0
        atr = 2.0
        barriers = [0.25, 0.50]
        
        result = generator._calculate_barriers(
            sample_klines, ref_price, atr, barriers, direction='down'
        )
        
        # -0.25 ATR = 99.5 -> zou in eerste 10 min moeten zijn
        assert result['025'] is not None
        assert result['025'] <= 10
    
    def test_calculate_extremes(self, generator, sample_klines):
        """Test berekening van extremen."""
        ref_price = 100.0
        atr = 2.0
        
        result = generator._calculate_extremes(sample_klines, ref_price, atr)
        
        assert result['max_up_atr'] > 0
        assert result['max_down_atr'] < 0
        assert result['time_to_max_up_min'] is not None
        assert result['time_to_max_down_min'] is not None
    
    def test_determine_first_significant_up_first(self, generator):
        """Test first significant wanneer up eerst is."""
        up_times = {'075': 20, '100': 40}
        down_times = {'075': 30, '100': None}
        
        barrier, time = generator.determine_first_significant(
            up_times, down_times, threshold=0.75
        )
        
        assert barrier == 'up_075'
        assert time == 20
    
    def test_determine_first_significant_down_first(self, generator):
        """Test first significant wanneer down eerst is."""
        up_times = {'075': 30}
        down_times = {'075': 15}
        
        barrier, time = generator.determine_first_significant(
            up_times, down_times, threshold=0.75
        )
        
        assert barrier == 'down_075'
        assert time == 15
    
    def test_determine_first_significant_none(self, generator):
        """Test first significant wanneer geen barrier geraakt."""
        up_times = {'075': None}
        down_times = {'075': None}
        
        barrier, time = generator.determine_first_significant(
            up_times, down_times, threshold=0.75
        )
        
        assert barrier == 'none'
        assert time is None


class TestBarrierOutcomeResult:
    """Tests voor BarrierOutcomeResult."""
    
    def test_to_dict(self):
        """Test conversie naar dictionary."""
        result = BarrierOutcomeResult(
            asset_id=1,
            time_1=datetime(2026, 1, 10, 14, 0),
            atr_at_signal=2.0,
            reference_price=100.0,
            max_observation_min=2880,
            time_to_up_barriers={'075': 20, '100': None},
            time_to_down_barriers={'075': 15},
            max_up_atr=0.85,
            max_down_atr=-0.42,
            first_significant_barrier='down_075',
            first_significant_time_min=15
        )
        
        d = result.to_dict()
        
        assert d['asset_id'] == 1
        assert d['time_to_up_075_atr'] == 20
        assert d['time_to_up_100_atr'] is None
        assert d['time_to_down_075_atr'] == 15
        assert d['first_significant_barrier'] == 'down_075'
