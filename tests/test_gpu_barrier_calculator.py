import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

from inference.barrier_config import BarrierConfig
from inference.barrier_outcome_generator import BarrierOutcomeGenerator
from inference.gpu_barrier_calculator import GPUBarrierCalculator


class TestGPUBarrierCalculator:
    """Tests voor GPUBarrierCalculator, vergelijkt met CPU implementation."""
    
    @pytest.fixture
    def config(self):
        return BarrierConfig(
            up_barriers=[0.50, 1.00],
            down_barriers=[0.50, 1.00],
            significant_threshold=0.50,
            max_observation_min=60
        )
    
    @pytest.fixture
    def cpu_gen(self, config):
        return BarrierOutcomeGenerator(config)
    
    @pytest.fixture
    def gpu_calc(self, config):
        return GPUBarrierCalculator(
            barriers=config.up_barriers,
            max_obs_min=config.max_observation_min
        )
    
    @pytest.fixture
    def sample_data(self, config):
        """Maak sample data voor 5 timestamps."""
        N = 5
        T = config.max_observation_min
        
        ref_prices = np.array([100.0, 200.0, 50.0, 150.0, 300.0], dtype=np.float32)
        atrs = np.array([2.0, 4.0, 1.0, 3.0, 6.0], dtype=np.float32)
        
        # (N, T, 2) array: [high, low]
        prices_batch = np.zeros((N, T, 2), dtype=np.float32)
        
        for i in range(N):
            base = ref_prices[i]
            atr = atrs[i]
            # Simuleer wat beweging
            for t in range(T):
                # Langzaam omhoog/omlaag afhankelijk van i
                if i % 2 == 0:
                    change = (t + 1) * (atr * 0.05)  # Omhoog
                else:
                    change = -(t + 1) * (atr * 0.05) # Omlaag
                
                prices_batch[i, t, 0] = base + change + (atr * 0.01) # high
                prices_batch[i, t, 1] = base + change - (atr * 0.01) # low
                
        return ref_prices, atrs, prices_batch

    def test_gpu_vs_cpu_numeric(self, cpu_gen, gpu_calc, sample_data):
        """Verifieer dat GPU resultaten numeriek identiek zijn aan CPU."""
        ref_prices, atrs, prices_batch = sample_data
        N = len(ref_prices)
        
        # GPU calculation
        gpu_results = gpu_calc.calculate_batch(prices_batch, ref_prices, atrs)
        
        # CPU calculation voor elke timestamp in de batch
        for i in range(N):
            # Converteer batch data voor deze timestamp naar DataFrame zoals CPU verwacht
            klines_data = []
            for t in range(prices_batch.shape[1]):
                klines_data.append({
                    'high': float(prices_batch[i, t, 0]),
                    'low': float(prices_batch[i, t, 1])
                })
            df_klines = pd.DataFrame(klines_data)
            
            # Bereken barriers via CPU private methods (overeenkomstig met GPU logic)
            cpu_up = cpu_gen._calculate_barriers(
                df_klines, ref_prices[i], atrs[i], cpu_gen.config.up_barriers, 'up'
            )
            cpu_down = cpu_gen._calculate_barriers(
                df_klines, ref_prices[i], atrs[i], cpu_gen.config.down_barriers, 'down'
            )
            cpu_ext = cpu_gen._calculate_extremes(df_klines, ref_prices[i], atrs[i])
            
            # Vergelijk UP barriers
            for level_val in cpu_gen.config.up_barriers:
                key = f"up_{int(level_val * 100):03d}"
                cpu_val = cpu_up.get(f"{int(level_val * 100):03d}")
                gpu_val = int(gpu_results[key][i])
                
                # GPU gebruikt -1 voor niet geraakt, CPU gebruikt None
                expected = cpu_val if cpu_val is not None else -1
                assert gpu_val == expected, f"Mismatch in {key} voor index {i}"
            
            # Vergelijk DOWN barriers
            for level_val in cpu_gen.config.down_barriers:
                key = f"down_{int(level_val * 100):03d}"
                cpu_val = cpu_down.get(f"{int(level_val * 100):03d}")
                gpu_val = int(gpu_results[key][i])
                
                expected = cpu_val if cpu_val is not None else -1
                assert gpu_val == expected, f"Mismatch in {key} voor index {i}"
            
            # Vergelijk extremen
            assert pytest.approx(gpu_results['max_up_atr'][i], rel=1e-5) == cpu_ext['max_up_atr']
            assert pytest.approx(gpu_results['max_down_atr'][i], rel=1e-5) == cpu_ext['max_down_atr']

    def test_determine_first_significant_batch(self, gpu_calc):
        """Test de batch logica voor first significant barrier."""
        # Mock results dict
        results = {
            'up_050': np.array([10, -1, 30, 15, 20], dtype=np.int16),
            'down_050': np.array([20, 10, 30, 15, -1], dtype=np.int16),
            'max_up_atr': np.array([0.6, 0.1, 0.5, 0.5, 0.5]) # Not used by this method but needed for len
        }
        
        names, times = gpu_calc.determine_first_significant_batch(results, threshold=0.50)
        
        # Index 0: up_050 (10) < down_050 (20) -> up_050, 10
        assert names[0] == 'up_050'
        assert times[0] == 10
        
        # Index 1: up_050 (-1) , down_050 (10) -> down_050, 10
        assert names[1] == 'down_050'
        assert times[1] == 10
        
        # Index 2: up_050 (30) == down_050 (30) -> up_050, 30 (bias to up)
        assert names[2] == 'up_050'
        assert times[2] == 30
        
        # Index 3: up_050 (15) == down_050 (15) -> up_050, 15
        assert names[3] == 'up_050'
        assert times[3] == 15
        
        # Index 4: up_050 (20), down_050 (-1) -> up_050, 20
        assert names[4] == 'up_050'
        assert times[4] == 20

    def test_empty_or_missing_data(self, gpu_calc):
        """Test gedrag bij lege of missende data."""
        N = 2
        T = 60
        prices = np.full((N, T, 2), 100.0, dtype=np.float32) # Allemaal flat
        ref_prices = np.array([100.0, 100.0], dtype=np.float32)
        atrs = np.array([2.0, 2.0], dtype=np.float32)
        
        results = gpu_calc.calculate_batch(prices, ref_prices, atrs)
        
        # Geen enkele barrier geraakt
        for key in ['up_050', 'up_100', 'down_050', 'down_100']:
            if key in results:
                assert np.all(results[key] == -1)
        
        assert np.all(results['max_up_atr'] == 0.0)
        assert np.all(results['max_down_atr'] == 0.0)
