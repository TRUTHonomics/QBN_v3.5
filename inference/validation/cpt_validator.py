"""
CPT Validator - Uitgebreide kwaliteitscontrole voor Conditional Probability Tables.

Berekent:
- Shannon Entropy (per rij en gemiddelde)
- Information Gain (KL Divergence t.o.v. prior)
- Stability Score (KL Divergence tussen twee periodes)
- Semantic Sanity Score (correlatie tussen signalen en outcomes)
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CPTValidator:
    """
    Validator voor CPT kwaliteit en statistische validiteit.
    """

    @staticmethod
    def calculate_entropy(probabilities: Dict[str, float]) -> float:
        """
        Bereken Shannon Entropy voor een probability distribution.
        H(X) = -sum(p * log2(p))
        """
        probs = np.array(list(probabilities.values()))
        # Filter 0 probs om log2 fouten te voorkomen
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def calculate_kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Bereken Kullback-Leibler Divergence tussen verdeling P en Q.
        D_KL(P || Q) = sum(P(i) * log2(P(i) / Q(i)))
        """
        states = set(p.keys()).union(set(q.keys()))
        kl = 0.0
        # Kleine epsilon voor stabiliteit
        epsilon = 1e-10
        
        for state in states:
            prob_p = p.get(state, 0) + epsilon
            prob_q = q.get(state, 0) + epsilon
            kl += prob_p * np.log2(prob_p / prob_q)
            
        return float(kl)

    def validate_cpt_quality(self, cpt_data: Dict[str, Any], prior_cpt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Voer een volledige kwaliteitsanalyse uit op een CPT.
        """
        metrics = {
            'entropy': 0.0,
            'info_gain': 0.0,
            'semantic_score': 0.0,
            'status': 'unknown'
        }
        
        node_type = cpt_data.get('type', 'conditional')
        states = cpt_data.get('states', [])
        
        if node_type == 'prior':
            metrics['entropy'] = self.calculate_entropy(cpt_data.get('probabilities', {}))
            metrics['info_gain'] = 0.0  # Prior heeft geen gain t.o.v. zichzelf
            metrics['semantic_score'] = 1.0 # REASON: Priors hebben geen semantische trading regels
        else:
            # Voor conditional: gemiddelde entropy over alle ouder-combinaties
            cond_probs = cpt_data.get('conditional_probabilities', {})
            if not cond_probs:
                return metrics
                
            entropies = [self.calculate_entropy(p) for p in cond_probs.values()]
            metrics['entropy'] = float(np.mean(entropies))
            
            # Information Gain t.o.v. prior (indien beschikbaar)
            if prior_cpt:
                prior_probs = prior_cpt.get('probabilities', {})
                kl_divs = [self.calculate_kl_divergence(p, prior_probs) for p in cond_probs.values()]
                metrics['info_gain'] = float(np.mean(kl_divs))

            # Semantic Sanity Check (indicatie)
            metrics['semantic_score'] = self._check_semantic_sanity(cpt_data)

        return metrics

    def _check_semantic_sanity(self, cpt_data: Dict[str, Any]) -> float:
        """
        Controleert of de CPT 'logisch' is voor trading.
        Bijv: Als de parent 'Bullish' is, moet de kans op een 'Bullish' outcome hoger zijn.
        Score tussen 0 en 1.
        """
        node_name = cpt_data.get('node', '').lower()
        cond_probs = cpt_data.get('conditional_probabilities', {})
        
        # REASON: Root nodes (HTF_Regime) en Composite nodes hebben geen parent-directionaliteit.
        # EXPL: Ze worden geconditioneerd op regime-states (trending/ranging), niet bullish/bearish.
        #       Semantic check is niet van toepassing, dus default 1.0.
        root_nodes = ['htf_regime', 'regime']
        composite_nodes = ['leading_composite', 'coincident_composite', 'confirming_composite']
        
        if any(root in node_name for root in root_nodes):
            return 1.0
        if any(comp in node_name for comp in composite_nodes):
            return 1.0  # Composites aggregeren signalen, geen directe semantic check
        
        # v3.4: Specifieke checks voor position subprediction nodes
        if 'momentum_prediction' in node_name:
            return self._check_momentum_semantic(cpt_data)
        if 'volatility_regime' in node_name:
            return self._check_volatility_semantic(cpt_data)
        if 'exit_timing' in node_name:
            return self._check_exit_timing_semantic(cpt_data)
            
        if not cond_probs:
            return 1.0 # Kan niet checken
            
        # Alleen relevant voor nodes met duidelijke Bullish/Bearish states
        # UPDATE v3.1: Support voor up/down en long/short terminologie
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
        
        # Helper voor case-insensitive match
        def is_state_in_list(state_name: str, target_list: List[str]) -> bool:
            return state_name.lower() in target_list

        total_checks = 0
        correct_shifts = 0
        
        for combo_key, probs in cond_probs.items():
            combo_lower = combo_key.lower()
            
            # Kijk of de combo_key 'bullish' of 'bearish' termen bevat
            is_bullish_parent = any(s in combo_lower for s in bullish_states)
            is_bearish_parent = any(s in combo_lower for s in bearish_states)
            
            # Voorkom conflicten (als parent zowel bullish als bearish termen bevat, skip)
            if is_bullish_parent and is_bearish_parent:
                continue

            if is_bullish_parent or is_bearish_parent:
                bull_mass = sum(probs.get(s, 0) for s in probs if is_state_in_list(s, bullish_states))
                bear_mass = sum(probs.get(s, 0) for s in probs if is_state_in_list(s, bearish_states))
                
                # REASON: Gebruik een tolerantie-marge om te voorkomen dat ruis (bijv. 0.2% vs 0.3%) 
                # de score ruïneert. Alleen significante directionele afwijkingen tellen mee.
                diff = abs(bull_mass - bear_mass)
                tolerance = 0.01 # 1% marge
                
                if diff < tolerance:
                    # Te weinig verschil om een semantisch oordeel te vellen (ruis)
                    continue
                
                total_checks += 1
                if is_bullish_parent and bull_mass > bear_mass:
                    correct_shifts += 1
                elif is_bearish_parent and bear_mass > bull_mass:
                    correct_shifts += 1
                    
        return correct_shifts / total_checks if total_checks > 0 else 1.0

    def calculate_stability(self, cpt_a: Dict[str, Any], cpt_b: Dict[str, Any]) -> float:
        """
        Berekent stabiliteit tussen twee CPT's (bv. verschillende periodes).
        Score: 1.0 - Jensen-Shannon Distance (bounded [0,1]).
        """
        # REASON: Gebruik Jensen-Shannon Distance i.p.v. KL Divergence voor stabiliteit.
        # EXPL: JS is symmetrisch, begrensd tussen 0 en 1, en gaat beter om met missing states.
        
        has_cond_a = 'conditional_probabilities' in cpt_a and cpt_a['conditional_probabilities']
        has_cond_b = 'conditional_probabilities' in cpt_b and cpt_b['conditional_probabilities']
        
        has_prior_a = 'probabilities' in cpt_a and cpt_a['probabilities']
        has_prior_b = 'probabilities' in cpt_b and cpt_b['probabilities']

        def jensen_shannon_distance(p_dist: Dict, q_dist: Dict) -> float:
            # Zorg dat alle keys strings zijn voor vergelijking
            p_keys = {str(k): v for k, v in p_dist.items()}
            q_keys = {str(k): v for k, v in q_dist.items()}
            
            states = sorted(list(set(p_keys.keys()).union(set(q_keys.keys()))))
            if not states:
                return 1.0
                
            p = np.array([p_keys.get(s, 1e-10) for s in states])
            q = np.array([q_keys.get(s, 1e-10) for s in states])
            
            p = p / p.sum()
            q = q / q.sum()
            
            m = 0.5 * (p + q)
            
            def kl_div(a, b):
                mask = (a > 0) & (b > 0)
                return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

            js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
            return float(np.sqrt(max(0, js_div)))

        if has_cond_a and has_cond_b:
            # REASON: Maak combo-keys vergelijking robuust voor JSON-stringified vs Python-tuple keys.
            # EXPL: We converteren alle keys naar strings voor de vergelijking.
            probs_a = {str(k): v for k, v in cpt_a['conditional_probabilities'].items()}
            probs_b = {str(k): v for k, v in cpt_b['conditional_probabilities'].items()}
            
            common_combos = set(probs_a.keys()).intersection(set(probs_b.keys()))
            if not common_combos:
                logger.warning(f"No common parent combos for stability check of node {cpt_a.get('node')}")
                return 0.0
                
            distances = []
            for combo in common_combos:
                distances.append(jensen_shannon_distance(probs_a[combo], probs_b[combo]))
                
            avg_distance = np.mean(distances)
            return float(max(0.0, 1.0 - avg_distance))
            
        elif has_prior_a and has_prior_b:
            dist = jensen_shannon_distance(cpt_a['probabilities'], cpt_b['probabilities'])
            stability = float(max(0.0, 1.0 - dist))
            logger.debug(f"Stability for prior node {cpt_a.get('node')}: dist={dist:.4f}, stability={stability:.4f}")
            return stability
            
        logger.warning(f"Stability check failed: missing probability data for node {cpt_a.get('node')}")
        return 0.0

    # =========================================================================
    # v3.3 POSITION SUBPREDICTION SEMANTIC CHECKS
    # =========================================================================
    
    def _check_momentum_semantic(self, cpt_data: Dict[str, Any]) -> float:
        """
        Semantic check voor Momentum_Prediction.
        
        Verwachting: improving delta_leading → hogere kans op bullish
        """
        cond_probs = cpt_data.get('conditional_probabilities', {})
        if not cond_probs:
            return 1.0
        
        checks = 0
        correct = 0
        
        for combo_key, probs in cond_probs.items():
            parts = combo_key.split('|')
            if len(parts) < 1:
                continue
            
            delta_state = parts[0].lower()
            
            # Check: improving delta → meer bullish
            if delta_state == 'improving':
                if probs.get('bullish', 0) > probs.get('bearish', 0):
                    correct += 1
                checks += 1
            # Check: deteriorating delta → meer bearish
            elif delta_state == 'deteriorating':
                if probs.get('bearish', 0) > probs.get('bullish', 0):
                    correct += 1
                checks += 1
        
        return correct / checks if checks > 0 else 1.0
    
    def _check_volatility_semantic(self, cpt_data: Dict[str, Any]) -> float:
        """
        Semantic check voor Volatility_Regime.
        
        Verwachting: normale spreiding over states (geen extreme bias)
        """
        cond_probs = cpt_data.get('conditional_probabilities', {})
        if not cond_probs:
            return 1.0
        
        # Check dat geen enkele state > 90% in alle combinaties
        high_bias_count = 0
        total = 0
        
        for probs in cond_probs.values():
            max_prob = max(probs.values()) if probs else 0
            if max_prob > 0.9:
                high_bias_count += 1
            total += 1
        
        # Als minder dan 20% van combinaties extreme bias heeft, is het OK
        if total == 0:
            return 1.0
        return 1.0 - (high_bias_count / total) if high_bias_count / total < 0.2 else 0.8
    
    def _check_exit_timing_semantic(self, cpt_data: Dict[str, Any]) -> float:
        """
        Semantic check voor Exit_Timing.
        
        Verwachting:
        - deteriorating + late time → meer exit_now
        - improving + early time → meer extend
        """
        cond_probs = cpt_data.get('conditional_probabilities', {})
        if not cond_probs:
            return 1.0
        
        checks = 0
        correct = 0
        
        for combo_key, probs in cond_probs.items():
            parts = combo_key.split('|')
            if len(parts) < 2:
                continue
            
            delta_state = parts[0].lower()
            time_bucket = parts[1].lower() if len(parts) > 1 else ''
            
            # Late time (12-24h) + deteriorating → exit_now
            if '12-24h' in time_bucket and delta_state == 'deteriorating':
                if probs.get('exit_now', 0) > probs.get('extend', 0):
                    correct += 1
                checks += 1
            
            # Early time (0-1h) + improving → extend
            elif '0-1h' in time_bucket and delta_state == 'improving':
                if probs.get('extend', 0) > probs.get('exit_now', 0):
                    correct += 1
                checks += 1
        
        return correct / checks if checks > 0 else 1.0
