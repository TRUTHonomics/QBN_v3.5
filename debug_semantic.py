
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.cpt_cache_manager import CPTCacheManager
from inference.validation.cpt_validator import CPTValidator
import json

def debug_semantic_score(asset_id=1, node_name="Trade_Hypothesis"):
    cache = CPTCacheManager()
    validator = CPTValidator()
    
    print(f"\n--- Debugging Semantic Score for {node_name} (Asset {asset_id}) ---")
    
    # 1. Haal CPT op
    cpt = cache.get_cpt(asset_id, node_name, max_age_hours=99999)
    if not cpt:
        print(f"❌ Geen CPT gevonden voor {node_name}")
        return

    print(f"✅ CPT geladen. Node: {cpt.get('node')}")
    print(f"States: {cpt.get('states')}")
    
    # 2. Handmatige check van de validator logic
    cond_probs = cpt.get('conditional_probabilities', {})
    print(f"Aantal parent combinaties: {len(cond_probs)}")
    
    bullish_states = [
        'strong_bullish', 'bullish', 'slight_bullish', 'excellent', 'good',
        'up_strong', 'up_weak', 'up',
        'strong_long', 'weak_long', 'long'
    ]
    bearish_states = [
        'strong_bearish', 'bearish', 'slight_bearish', 'poor',
        'down_strong', 'down_weak', 'down',
        'strong_short', 'weak_short', 'short'
    ]
    
    total_checks = 0
    correct_shifts = 0
    
    for combo_key, probs in list(cond_probs.items())[:10]: # Check eerste 10
        combo_lower = combo_key.lower()
        is_bullish_parent = any(s in combo_lower for s in bullish_states)
        is_bearish_parent = any(s in combo_lower for s in bearish_states)
        
        print(f"\nCombo: '{combo_key}'")
        print(f"  - Bullish Parent? {is_bullish_parent}")
        print(f"  - Bearish Parent? {is_bearish_parent}")
        
        if is_bullish_parent or is_bearish_parent:
            total_checks += 1
            bull_mass = sum(probs.get(s, 0) for s in probs if s.lower() in bullish_states)
            bear_mass = sum(probs.get(s, 0) for s in probs if s.lower() in bearish_states)
            print(f"  - Bull mass: {bull_mass:.4f}, Bear mass: {bear_mass:.4f}")
            
            if is_bullish_parent and bull_mass > bear_mass:
                correct_shifts += 1
                print("  ✅ Correct (Bullish)")
            elif is_bearish_parent and bear_mass > bull_mass:
                correct_shifts += 1
                print("  ✅ Correct (Bearish)")
            else:
                print("  ❌ Incorrect shift")

    if total_checks > 0:
        final_score = correct_shifts / total_checks
        print(f"\nFinal Calculated Score (Sample): {final_score:.2f}")
    else:
        print("\n❌ Geen directionele checks kunnen uitvoeren (geen bullish/bearish parents gevonden in sample)")

if __name__ == "__main__":
    debug_semantic_score(1, "Trade_Hypothesis")
    debug_semantic_score(1, "Prediction_1h")
