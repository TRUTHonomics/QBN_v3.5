import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from inference.barrier_config import BarrierConfig, BarrierConfigLoader, validate_barrier_config

class TestBarrierConfigLoader:
    """Tests voor de BarrierConfigLoader en fallback chain."""
    
    @pytest.fixture
    def temp_yaml(self, tmp_path):
        """Maak een tijdelijk YAML bestand voor tests."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml_path = config_dir / "barrier_config.yaml"
        
        data = {
            'test_profile': {
                'up_barriers': [1.0, 2.0],
                'down_barriers': [1.0, 2.0],
                'significant_threshold': 1.0,
                'max_observation_min': 100
            },
            'asset_overrides': {
                999: {
                    'significant_threshold': 2.0
                }
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
            
        return yaml_path

    def test_load_from_yaml(self, temp_yaml):
        """Test of laden uit YAML correct werkt."""
        with patch.object(BarrierConfigLoader, 'DEFAULT_YAML_PATH', temp_yaml):
            config = BarrierConfigLoader._load_from_yaml('test_profile')
            
            assert config.config_name == 'test_profile'
            assert config.up_barriers == [1.0, 2.0]
            assert config.significant_threshold == 1.0
            assert config.max_observation_min == 100

    def test_apply_asset_overrides(self, temp_yaml):
        """Test of asset overrides correct worden toegepast."""
        config = BarrierConfig(
            config_name='test_profile',
            significant_threshold=1.0
        )
        
        with patch.object(BarrierConfigLoader, 'DEFAULT_YAML_PATH', temp_yaml):
            # Test override
            overridden = BarrierConfigLoader._apply_asset_overrides(config, 999)
            assert overridden.significant_threshold == 2.0
            
            # Geen override voor onbekend asset (gebruik nieuwe config om side-effects te vermijden)
            config2 = BarrierConfig(
                config_name='test_profile',
                significant_threshold=1.0
            )
            normal = BarrierConfigLoader._apply_asset_overrides(config2, 1)
            assert normal.significant_threshold == 1.0

    def test_fallback_chain_db_hit(self):
        """Test fallback chain: DB hit."""
        mock_config = BarrierConfig(config_name='db_config')
        
        with patch.object(BarrierConfig, 'from_database', return_value=mock_config):
            config = BarrierConfigLoader.load('db_config', prefer_database=True)
            assert config.config_name == 'db_config'
            BarrierConfig.from_database.assert_called_once_with('db_config')

    def test_fallback_chain_db_fail_yaml_hit(self, temp_yaml):
        """Test fallback chain: DB fail -> YAML hit."""
        with patch.object(BarrierConfig, 'from_database', side_effect=Exception("DB Error")):
            with patch.object(BarrierConfigLoader, 'DEFAULT_YAML_PATH', temp_yaml):
                config = BarrierConfigLoader.load('test_profile', prefer_database=True)
                assert config.config_name == 'test_profile'
                assert config.significant_threshold == 1.0

    def test_fallback_chain_all_fail_defaults(self, tmp_path):
        """Test fallback chain: Alles faalt -> Defaults."""
        non_existent = tmp_path / "nothing.yaml"
        with patch.object(BarrierConfig, 'from_database', side_effect=Exception("DB Error")):
            with patch.object(BarrierConfigLoader, 'DEFAULT_YAML_PATH', non_existent):
                config = BarrierConfigLoader.load('any', prefer_database=True)
                assert config.config_name == 'any'
                assert config.significant_threshold == 0.75 # Default value

    def test_validate_barrier_config(self):
        """Test de validatie waarschuwingen."""
        # 1. Geen waarschuwingen
        config_ok = BarrierConfig(
            up_barriers=[0.5, 1.0],
            significant_threshold=0.5
        )
        assert len(validate_barrier_config(config_ok)) == 0
        
        # 2. Hoge barriers
        config_high = BarrierConfig(up_barriers=[4.0])
        assert any("extreem hoog" in w for w in validate_barrier_config(config_high))
        
        # 3. Threshold niet in barriers
        config_mismatch = BarrierConfig(
            up_barriers=[0.5, 1.0],
            significant_threshold=0.75
        )
        assert any("niet in up_barriers" in w for w in validate_barrier_config(config_mismatch))

    @patch('database.db.get_cursor')
    def test_save_to_database(self, mock_get_cursor):
        """Test of save_to_database de juiste SQL aanroept."""
        mock_cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cur
        
        config = BarrierConfig(
            config_name='test_save',
            up_barriers=[1.0],
            down_barriers=[1.0],
            significant_threshold=1.0,
            max_observation_min=2880
        )
        
        config.save_to_database(notes="Test note")
        
        # Check of execute is aangeroepen met de juiste query
        args, _ = mock_cur.execute.call_args
        sql = args[0]
        params = args[1]
        
        assert "INSERT INTO qbn.barrier_config" in sql
        assert params[0] == 'test_save'
        assert params[1] == [1.0]
        assert params[5] == "Test note"
