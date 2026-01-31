import pytest
from unittest.mock import MagicMock, patch
from inference.inference_pool import InferencePool
from inference.entry_model_inference import EntryModelInference

@pytest.fixture
def mock_loader():
    with patch('inference.inference_pool.InferenceLoader') as mock:
        loader_instance = mock.return_value
        # Mock de load_inference_engine methode
        loader_instance.load_inference_engine.side_effect = lambda aid: MagicMock(spec=EntryModelInference)
        yield loader_instance

def test_pool_preloading(mock_loader):
    """Test of de pool correct assets preloaded."""
    tracked_assets = [1, 2, 3]
    pool = InferencePool(tracked_assets)
    
    pool.preload_all(max_workers=2)
    
    assert len(pool.engines) == 3
    assert set(pool.engines.keys()) == {1, 2, 3}
    assert mock_loader.load_inference_engine.call_count == 3

def test_pool_lazy_loading(mock_loader):
    """Test of de pool assets lazy loadt indien niet gepreloade."""
    pool = InferencePool([]) # Geen tracked assets
    
    # Eerste call naar asset 10
    engine = pool.get_engine(10)
    
    assert 10 in pool.engines
    assert mock_loader.load_inference_engine.called
    assert mock_loader.load_inference_engine.call_args[0][0] == 10

def test_pool_refresh(mock_loader):
    """Test het herladen van een engine."""
    pool = InferencePool([1])
    pool.preload_all()
    
    original_engine = pool.get_engine(1)
    mock_loader.load_inference_engine.reset_mock()
    
    pool.refresh_engine(1)
    
    assert mock_loader.load_inference_engine.called
    new_engine = pool.get_engine(1)
    assert new_engine != original_engine # In werkelijkheid mock objecten, maar test de flow

