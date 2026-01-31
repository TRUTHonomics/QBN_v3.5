"""
Config Completeness Validator voor QBN v3.1.
Controleert of alle verwachte configuratie-entries in de database aanwezig zijn.
"""

import logging
from typing import Dict, List, Tuple
from database.db import get_cursor
from core.config_warnings import warn_fallback_active

logger = logging.getLogger(__name__)

def validate_threshold_config_completeness(asset_id: int) -> Dict:
    """
    Controleer of alle verwachte config entries aanwezig zijn voor een asset.
    Returns dict met status per horizon en config_type.
    """
    horizons = ['1h', '4h', '1d']
    config_types = [
        'leading_composite', 'coincident_composite', 'confirming_composite',
        'leading_alignment', 'coincident_alignment', 'confirming_alignment'
    ]
    
    results = {}
    missing_any = False
    
    try:
        with get_cursor() as cur:
            for horizon in horizons:
                results[horizon] = {}
                for c_type in config_types:
                    cur.execute("""
                        SELECT param_name FROM qbn.composite_threshold_config
                        WHERE asset_id = %s AND horizon = %s AND config_type = %s
                    """, (asset_id, horizon, c_type))
                    
                    params = [row[0] for row in cur.fetchall()]
                    
                    # Verwachte parameters per type
                    if 'composite' in c_type:
                        expected = ['neutral_band', 'strong_threshold']
                    else:
                        expected = ['high_threshold', 'low_threshold']
                        
                    missing = [p for p in expected if p not in params]
                    
                    if missing:
                        results[horizon][c_type] = {
                            'status': 'MISSING',
                            'params': missing
                        }
                        missing_any = True
                        warn_fallback_active(
                            component="ConfigValidator",
                            config_name=f"asset_{asset_id}_{horizon}_{c_type}",
                            fallback_values={'missing': missing},
                            reason="Incomplete parameters in database",
                            fix_command="Draai 'Threshold Optimalisatie' (Optie 7)"
                        )
                    else:
                        results[horizon][c_type] = {'status': 'OK'}
                        
        return {
            'asset_id': asset_id,
            'complete': not missing_any,
            'details': results
        }
        
    except Exception as e:
        logger.error(f"Fout bij config validatie: {e}")
        return {'error': str(e), 'complete': False}

if __name__ == "__main__":
    # Test voor Asset 1
    res = validate_threshold_config_completeness(1)
    print(f"Validatie resultaat: {'✅ COMPLEET' if res['complete'] else '❌ INCOMPLEET'}")
    if not res['complete']:
        print("Details:")
        for h, types in res.get('details', {}).items():
            for t, info in types.items():
                if info['status'] == 'MISSING':
                    print(f"  - {h} {t}: missing {info['params']}")
