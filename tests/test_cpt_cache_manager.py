import pytest
from inference.cpt_cache_manager import CPTCacheManager
from datetime import datetime, timezone, timedelta

@pytest.fixture
def cache_manager():
    return CPTCacheManager()

def test_cache_stats_initial(cache_manager):
    stats = cache_manager.get_cache_stats()
    assert stats['cache_hits'] == 0
    assert stats['cache_misses'] == 0

def test_save_and_get_cpt(cache_manager):
    # Merk op: dit vereist een werkende database connectie
    # Voor unit tests zonder DB zouden we kunnen mocken, 
    # maar in dit project testen we vaak tegen de (test) DB.
    asset_id = 9889 # TEST ASSET
    node_name = "test_node"
    cpt_data = {"test": 1.0}
    
    cache_manager.save_cpt(asset_id, node_name, cpt_data)
    loaded = cache_manager.get_cpt(asset_id, node_name)
    
    assert loaded == cpt_data
    assert cache_manager.cache_hits == 1

