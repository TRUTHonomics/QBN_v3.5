"""
QBN v3.1 CPT Generator - Trade-Aligned Training (Dual-Prediction Architecture).

Genereert CPT's voor de QBN_v3.1 network structuur:
- HTF_Regime (root) → 5/11 states
- Composite nodes (Leading/Coincident/Confirming) → 5 states
- Trade_Hypothesis (v3 entry decision) → 5 states
- Position_Confidence (v3.1: Coincident + Confirming) → 3 states
- Position_Prediction (v3.1: actieve positie uitkomst) → 3 states
- Prediction nodes (1h/4h/1d) → 5 barrier states

v3.1 WIJZIGINGEN:
- Entry_Confidence is VOLLEDIG VERWIJDERD (niet backwards compat)
- Prediction nodes hebben 2 parents: HTF_Regime + Trade_Hypothesis
- Position_Confidence heeft nu Coincident + Confirming als parents
- Position_Prediction toegevoegd voor actieve posities

Training op echte koersuitkomsten (barrier-based).
GPU-only mode voor maximale performance.
"""

import logging
import os
import shutil
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import Counter
import json

from inference.cpt_generator import ConditionalProbabilityTableGenerator, OUTCOME_STATE_LIST
from inference.network_structure import QBNv3NetworkStructure
from inference.signal_aggregator import SignalAggregator
from inference.node_types import SemanticClass, CompositeState, OutcomeState, RegimeState, BarrierOutcomeState
from inference.regime_detector import HTFRegimeDetector
from inference.gpu.gpu_cpt_generator import GPUCPTGenerator
from core.step_validation import validate_step_input, log_handshake_out, StepValidationError
from inference.validation.cpt_validator import CPTValidator
from database.db import get_cursor
from config.network_config import MIN_OBS_PER_CELL, MIN_TOTAL_OBS, COVERAGE_THRESHOLD
from config.threshold_loader import ThresholdLoader

# v3 Generators for labeling
from .trade_hypothesis_generator import TradeHypothesisGenerator
from .position_confidence_generator import PositionConfidenceGenerator
from .position_prediction_generator import PositionPredictionGenerator
from .event_window_detector import EventWindowDetector, EventWindowConfig

# v3.3 Triple Composite generators
from .momentum_prediction_generator import MomentumPredictionGenerator
from .volatility_regime_generator import VolatilityRegimeGenerator
from .exit_timing_generator import ExitTimingGenerator

logger = logging.getLogger(__name__)

@dataclass
class CPTValidationResult:
    """Resultaat van CPT validatie."""
    is_valid: bool
    coverage: float
    sparse_cells: int
    total_cells: int
    recommendation: str
    observations: int
    entropy: float = 0.0
    info_gain: float = 0.0
    stability_score: float = 1.0
    semantic_score: float = 1.0

class QBNv3CPTGenerator:
    """
    CPT Generator voor QBN v3 met outcome-based training en GPU acceleratie.
    """
    
    NEUTRAL_DOWNSAMPLE_WEIGHT = 1.0
    
    def __init__(
        self,
        laplace_alpha: float = 1.0,
        min_obs_per_cell: int = MIN_OBS_PER_CELL,
        min_total_obs: int = MIN_TOTAL_OBS,
        coverage_threshold: float = COVERAGE_THRESHOLD,
        run_id: str = None,
        neutral_downsample: float = 1.0
    ):
        self.laplace_alpha = laplace_alpha
        self.min_obs_per_cell = min_obs_per_cell
        self.min_total_obs = min_total_obs
        self.coverage_threshold = coverage_threshold
        self.run_id = run_id
        self.neutral_downsample = neutral_downsample
        
        self._setup_file_logging()
        self.base_generator = ConditionalProbabilityTableGenerator(laplace_alpha)
        self.gpu_generator = GPUCPTGenerator(laplace_alpha=laplace_alpha)
        self.validator = CPTValidator()
        self.network = QBNv3NetworkStructure()
        self.signal_aggregator = None
        self.regime_detector = HTFRegimeDetector()
        
        # REASON: Cache voor ThresholdLoaders per horizon
        self._threshold_loaders: Dict[str, ThresholdLoader] = {}
        self._current_asset_id: Optional[int] = None
        
        # v3 Sub-generators
        self.hypothesis_gen = TradeHypothesisGenerator(laplace_alpha)
        # v3.2: Position generators met delta mode en weighting
        self.position_gen = PositionConfidenceGenerator(
            laplace_alpha=laplace_alpha,
            use_delta_mode=True  # v3.2 delta-based training
        )
        self.position_prediction_gen = PositionPredictionGenerator(
            laplace_alpha=laplace_alpha,
            use_weighted_mode=True  # v3.2 uniqueness weighting
        )
        self.event_detector = EventWindowDetector(EventWindowConfig())
        
        # REASON: Cache voor Combination Alpha resultaten (Phase 2.5)
        self._combination_alpha_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"QBN v3 CPT Generator initialized (α={laplace_alpha}, min_obs={min_obs_per_cell})")

    def _load_combination_alpha_results(self, asset_id: int):
        """Laad de Combination Alpha resultaten uit de database (Phase 2.5)."""
        if self._current_asset_id == asset_id and self._combination_alpha_results:
            return

        logger.info(f"🔍 Loading Combination Alpha results for asset {asset_id}...")
        query = """
            SELECT horizon, combination_key, target_type, odds_ratio, or_ci_lower, classification, p_value_corrected
            FROM qbn.combination_alpha
            WHERE asset_id = %s
            ORDER BY analyzed_at DESC
        """
        
        results = {}
        with get_cursor() as cur:
            cur.execute(query, (asset_id,))
            rows = cur.fetchall()
            
            for row in rows:
                horizon, combo_key, target_type, odds_ratio, ci_lower, classification, p_corr = row
                # We slaan alleen de meest recente op per (horizon, combo, target)
                key = f"{horizon}|{combo_key}|{target_type}"
                if key not in results:
                    results[key] = {
                        'odds_ratio': float(odds_ratio),
                        'ci_lower': float(ci_lower),
                        'classification': classification,
                        'p_value_corrected': float(p_corr) if p_corr is not None else 1.0
                    }
        
        self._combination_alpha_results = results
        
        # REASON: Detail-logging voor geladen Alpha resultaten
        n_total = len(results)
        n_golden = sum(1 for r in results.values() if r['classification'] == 'golden_rule')
        n_promising = sum(1 for r in results.values() if r['classification'] == 'promising')
        n_noise = sum(1 for r in results.values() if r['classification'] == 'noise')
        
        logger.info(f"✅ Loaded {n_total} Combination Alpha records for asset {asset_id}:")
        logger.info(f"   - Golden Rules: {n_golden}")
        logger.info(f"   - Promising:    {n_promising}")
        logger.info(f"   - Noise:         {n_noise}")

    def _apply_combination_alpha_refinement(self, cpt_data: Dict, horizon: str, asset_id: int) -> Dict:
        """
        Verfijn de CPT distributies op basis van Combination Alpha resultaten (Phase 2.5).

        v3.1: Updated voor 2-tuple CPT keys (regime|hypothesis).

        Strategie:
        1. Als een combinatie 'noise' is (niet significant), trek de distributie naar uniform.
        2. Als een combinatie 'golden_rule' is, behoud of versterk de distributie.
        """
        if not self._combination_alpha_results:
            self._load_combination_alpha_results(asset_id)

        if not self._combination_alpha_results:
            return cpt_data

        cond_probs = cpt_data.get('conditional_probabilities', {})
        if not cond_probs:
            return cpt_data

        # REASON: Mapping van Decision states terug naar Composite states voor lookup
        hyp_to_comp = {v: k for k, v in self.hypothesis_gen.LEADING_TO_HYPOTHESIS.items()}

        refined_probs = {}
        n_refined = 0
        n_noise = 0

        states = cpt_data.get('states', [])
        uniform_prob = 1.0 / len(states) if states else 1.0/7.0

        for parent_combo, distribution in cond_probs.items():
            # v3.1: parent_combo is nu "regime|hypothesis" (2-tuple)
            parts = parent_combo.split('|')
            if len(parts) != 2:
                # Skip onverwachte key formats
                refined_probs[parent_combo] = distribution
                continue

            regime, hypothesis = parts

            # Map hypothesis naar leading state
            leading_state = hyp_to_comp.get(hypothesis, 'neutral')

            # v3.1: Simplified - check combination alpha direct op leading state
            # (zonder confidence dimensie)
            is_noise = True
            found_any_alpha = False

            # Check alle mogelijke coincident/confirming combinaties voor deze leading state
            comp_states = [s.value for s in CompositeState]
            for coinc in comp_states:
                for conf in comp_states:
                    combo_key = f"{leading_state}|{coinc}|{conf}"

                    bull_key = f"{horizon}|{combo_key}|bullish"
                    bear_key = f"{horizon}|{combo_key}|bearish"

                    bull_res = self._combination_alpha_results.get(bull_key)
                    bear_res = self._combination_alpha_results.get(bear_key)

                    if bull_res or bear_res:
                        found_any_alpha = True
                        # Als er MAAR EEN combinatie is die NIET noise is, dan behouden we de data
                        is_bull_noise = bull_res['classification'] == 'noise' if bull_res else True
                        is_bear_noise = bear_res['classification'] == 'noise' if bear_res else True

                        if not is_bull_noise or not is_bear_noise:
                            is_noise = False
                            break
                if not is_noise:
                    break

            # Als we geen alpha data hebben voor deze scenario's, gaan we uit van 'niet noise' (behoud data)
            if not found_any_alpha:
                is_noise = False

            if is_noise:
                # REASON: Noise filtering - trek naar uniform om valse zekerheid te voorkomen
                refined_probs[parent_combo] = {s: uniform_prob for s in states}
                n_noise += 1
            else:
                # Behoud originele distributie
                refined_probs[parent_combo] = distribution

            n_refined += 1
            
        if n_noise > 0:
            logger.info(f"✨ Combination Alpha refinement: {n_noise}/{n_refined} rows neutralized as 'noise' for {cpt_data['node']}")
            cpt_data['conditional_probabilities'] = refined_probs
            if 'metadata' not in cpt_data: cpt_data['metadata'] = {}
            cpt_data['metadata']['combination_alpha_refined'] = True
            cpt_data['metadata']['noise_rows_count'] = n_noise
            
        return cpt_data

    def _get_threshold_loader(self, asset_id: int, horizon: str) -> ThresholdLoader:
        """
        Haal of maak ThresholdLoader voor specifiek asset en horizon.
        
        REASON: Caching voorkomt herhaalde database reads voor dezelfde asset/horizon.
        """
        if self._current_asset_id != asset_id:
            # Reset cache bij asset switch
            self._threshold_loaders = {}
            self._current_asset_id = asset_id
        
        if horizon not in self._threshold_loaders:
            self._threshold_loaders[horizon] = ThresholdLoader(asset_id, horizon)
            
        return self._threshold_loaders[horizon]

    def load_signal_classification(self, asset_id: Optional[int] = None, horizon: Optional[str] = None, filter_suffix: Optional[str] = None):
        """
        Laad signal classification en weights uit database.
        
        REASON: In v3.1 moeten ALLE geclassificeerde signalen geladen worden,
        ook als ze (nog) geen gewichten hebben (zoals Coincident/Confirming).
        """
        # STAP 1: Haal ALLE basis-classificaties op
        class_query = """
        SELECT signal_name, semantic_class, polarity
        FROM qbn.signal_classification
        """
        
        # STAP 2: Haal weights op indien asset_id en horizon meegegeven
        weights_data = {} # (full_name, horizon) -> weight
        if asset_id and horizon:
            weights_query = """
            SELECT DISTINCT ON (signal_name, horizon)
                signal_name, horizon, weight
            FROM qbn.signal_weights
            WHERE (asset_id = %s OR asset_id = 0)
            ORDER BY signal_name, horizon, asset_id DESC
            """
            with get_cursor() as cur:
                cur.execute(weights_query, (asset_id,))
                for full_name, h, weight in cur.fetchall():
                    weights_data[(full_name.lower(), h)] = float(weight or 1.0)

        with get_cursor() as cur:
            cur.execute(class_query)
            base_signals = cur.fetchall()
        
        suffix = filter_suffix if filter_suffix else '60'
        classification = {}
        
        for base_name, sem_class, polarity in base_signals:
            # Genereer volledige naam (lowercase + suffix)
            full_name = f"{base_name.lower()}_{suffix}"
            
            # Koppel gewicht (gebruik 1.0 als default)
            # We zoeken specifiek voor de horizon die we nu trainen
            weight = weights_data.get((full_name, horizon), 1.0) if horizon else 1.0
            
            classification[full_name] = {
                'semantic_class': sem_class,
                'polarity': polarity,
                'weight': weight
            }
        
        # REASON: Laad ThresholdLoader voor DB-driven thresholds
        threshold_loader = None
        if asset_id and horizon:
            logger.debug(f"🔍 Loading thresholds for asset {asset_id}, horizon {horizon}...")
            threshold_loader = self._get_threshold_loader(asset_id, horizon)
        
        self.signal_aggregator = SignalAggregator(classification, threshold_loader=threshold_loader)
        logger.info(f"✅ Loaded {len(classification)} signal classifications and weights (suffix={suffix})")

    # ========================================================================
    # v3 SPECIFIC CPT GENERATION
    # ========================================================================

    def generate_trade_hypothesis_cpt(self, asset_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """Genereer Trade_Hypothesis CPT (ALLEEN Leading)."""
        node_name = "Trade_Hypothesis"
        parent_nodes = ['leading_composite']
        
        # Voor training leiden we de label direct af van Leading_Composite
        # EXPL: In v3 is de hypothese direct gekoppeld aan de leading indicators.
        cpt_data = self.gpu_generator._generate_conditional_cpt_gpu(
            data, node_name, parent_nodes, db_columns=['trade_hypothesis'],
            aggregation_method='direct', 
            states=self.hypothesis_gen.HYPOTHESIS_STATES
        )
        
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        cpt_data['observations'] = len(data)
        # REASON: Geen outcome_mode meer voor structurele nodes
        return cpt_data

    def generate_position_confidence_cpt(self, asset_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genereer Position_Confidence CPT (Data-Driven v3.2).
        
        v3.2 NIEUW:
        - Delta-based parents (verandering sinds entry)
        - Uniqueness weighting (López de Prado)
        - Laadt delta thresholds uit database
        """
        node_name = "Position_Confidence"
        logger.info(f"🎯 Generating data-driven {node_name} CPT v3.2 for asset {asset_id}")
        
        # v3.2: Laad delta thresholds uit database
        try:
            from config.position_delta_loader import PositionDeltaThresholdLoader
            delta_loader = PositionDeltaThresholdLoader(asset_id)
            
            # Update position generator met geladen thresholds
            self.position_gen.delta_threshold_coinc = delta_loader.cumulative_coincident_threshold
            self.position_gen.delta_threshold_conf = delta_loader.cumulative_confirming_threshold
            
            logger.info(f"   Delta thresholds loaded: coinc={self.position_gen.delta_threshold_coinc:.3f}, "
                       f"conf={self.position_gen.delta_threshold_conf:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load delta thresholds: {e}. Using defaults.")
        
        # v3.3: Bereken composite scores als ze ontbreken (nodig voor event detection en delta berekening)
        if 'leading_score' not in data.columns or 'coincident_score' not in data.columns or 'confirming_score' not in data.columns:
            logger.info("   Calculating composite scores for event detection and delta computation...")
            data = self._add_composite_scores(data, asset_id)
        
        # 1. Detecteer events in de data (v3.2: inclusief delta scores en uniqueness weights)
        events, labeled_data = self.event_detector.detect_events(data, asset_id)
        
        if not events:
            logger.warning(f"⚠️ No events detected for asset {asset_id}, skipping data-driven {node_name}")
            return self._create_uniform_composite_cpt(node_name)
            
        # 2. Filter voor rijen binnen events
        in_event_data = labeled_data[labeled_data['event_id'].notna()].copy()
        
        # v3.2: Log delta score statistieken (defensief tegen None/NaN/object dtype)
        if 'delta_cum_coincident' in in_event_data.columns:
            # REASON: Kolom kan object dtype zijn als alle waarden None zijn - forceer numeric
            numeric_deltas = pd.to_numeric(in_event_data['delta_cum_coincident'], errors='coerce')
            valid_deltas = numeric_deltas.dropna()
            if len(valid_deltas) > 0:
                logger.info(f"   Delta coinc stats: mean={valid_deltas.mean():.4f}, std={valid_deltas.std():.4f}")
            else:
                logger.warning("   ⚠️ No valid delta_cum_coincident values - composite scores missing in input data")
        
        if 'uniqueness_weight' in in_event_data.columns:
            # REASON: Kolom kan object dtype zijn als alle waarden None zijn - forceer numeric
            numeric_weights = pd.to_numeric(in_event_data['uniqueness_weight'], errors='coerce')
            valid_weights = numeric_weights.dropna()
            if len(valid_weights) > 0:
                total_weight = valid_weights.sum()
                logger.info(f"   Uniqueness weighting: {len(in_event_data)} rows -> {total_weight:.1f} effective obs")
            else:
                logger.warning("   ⚠️ No valid uniqueness_weight values")
        
        # 3. Genereer CPT (v3.2: delta-based met weighting)
        cpt_results = self.position_gen.generate_cpt(in_event_data)
        
        # 4. Save to DB
        self.position_gen.save_cpt(asset_id, self.run_id)
        
        # Formatteer voor return (gelijk aan database format)
        formatted_probs = {
            "|".join(k): v for k, v in cpt_results.items()
        }
        
        # v3.2: Bepaal parents op basis van delta mode
        if self.position_gen.use_delta_mode:
            parents = ['Delta_Coincident', 'Delta_Confirming', 'Time_Since_Entry']
            version = '3.2'
        else:
            parents = ['Coincident_Composite', 'Confirming_Composite', 'Time_Since_Entry']
            version = '3.1'
        
        cpt_data = {
            'node': node_name,
            'parents': parents,
            'states': self.position_gen.CONFIDENCE_STATES,
            'conditional_probabilities': formatted_probs,
            'type': 'conditional',
            'metadata': {
                'version': version,
                'n_events': len(events),
                'observations': len(in_event_data),
                'delta_mode': self.position_gen.use_delta_mode,
                'delta_threshold_coinc': self.position_gen.delta_threshold_coinc,
                'delta_threshold_conf': self.position_gen.delta_threshold_conf
            },
            'outcome_mode': 'barrier',
            'observations': len(in_event_data)  # REASON: Top-level observations voor CPTCacheManager
        }
        
        # REASON: Valideer CPT kwaliteit (coverage, entropy) zoals bij globale nodes
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        
        return cpt_data

    # =========================================================================
    # v3.3 TRIPLE COMPOSITE CPT GENERATORS
    # =========================================================================

    def generate_momentum_prediction_cpt(self, asset_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genereer Momentum_Prediction CPT (v3.3).
        
        Leading-based voorspelling van prijsrichting.
        Parents: Delta_Leading, Time_Since_Entry
        States: bearish, neutral, bullish
        """
        node_name = "Momentum_Prediction"
        logger.info(f"🎯 Generating {node_name} CPT v3.3 for asset {asset_id}")
        
        # Detecteer events voor training data
        events, labeled_data = self.event_detector.detect_events(data, asset_id)
        
        if not events:
            logger.warning(f"⚠️ No events for {node_name}, returning uniform")
            return self._create_uniform_position_subpred_cpt(node_name, 
                ['bearish', 'neutral', 'bullish'],
                ['Delta_Leading', 'Time_Since_Entry'])
        
        # Filter event data
        in_event_data = labeled_data[labeled_data['event_id'].notna()].copy()
        
        # Generate CPT
        generator = MomentumPredictionGenerator()
        cpt_results = generator.generate_cpt(in_event_data)
        
        # Format voor return
        formatted_probs = {
            "|".join(k): v for k, v in cpt_results.items()
        }
        
        cpt_data = {
            'node': node_name,
            'parents': ['Delta_Leading', 'Time_Since_Entry'],
            'states': ['bearish', 'neutral', 'bullish'],
            'conditional_probabilities': formatted_probs,
            'type': 'conditional',
            'metadata': {'version': '3.3', 'n_events': len(events)},
            'observations': len(in_event_data)
        }
        
        # REASON: Valideer CPT kwaliteit (coverage, entropy) zoals bij globale nodes
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        
        return cpt_data

    def generate_volatility_regime_cpt(self, asset_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genereer Volatility_Regime CPT (v3.3).
        
        Coincident-based volatiliteit voorspelling.
        Parents: Delta_Coincident, Time_Since_Entry
        States: low_vol, normal, high_vol
        """
        node_name = "Volatility_Regime"
        logger.info(f"🎯 Generating {node_name} CPT v3.3 for asset {asset_id}")
        
        events, labeled_data = self.event_detector.detect_events(data, asset_id)
        
        if not events:
            logger.warning(f"⚠️ No events for {node_name}, returning uniform")
            return self._create_uniform_position_subpred_cpt(node_name,
                ['low_vol', 'normal', 'high_vol'],
                ['Delta_Coincident', 'Time_Since_Entry'])
        
        in_event_data = labeled_data[labeled_data['event_id'].notna()].copy()
        
        generator = VolatilityRegimeGenerator()
        cpt_results = generator.generate_cpt(in_event_data)
        
        formatted_probs = {
            "|".join(k): v for k, v in cpt_results.items()
        }
        
        cpt_data = {
            'node': node_name,
            'parents': ['Delta_Coincident', 'Time_Since_Entry'],
            'states': ['low_vol', 'normal', 'high_vol'],
            'conditional_probabilities': formatted_probs,
            'type': 'conditional',
            'metadata': {'version': '3.3', 'n_events': len(events)},
            'observations': len(in_event_data)
        }
        
        # REASON: Valideer CPT kwaliteit (coverage, entropy) zoals bij globale nodes
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        
        return cpt_data

    def generate_exit_timing_cpt(self, asset_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genereer Exit_Timing CPT (v3.3).
        
        Confirming-based exit timing voorspelling.
        Parents: Delta_Confirming, Time_Since_Entry, Current_PnL_ATR
        States: exit_now, hold, extend
        """
        node_name = "Exit_Timing"
        logger.info(f"🎯 Generating {node_name} CPT v3.3 for asset {asset_id}")
        
        events, labeled_data = self.event_detector.detect_events(data, asset_id)
        
        if not events:
            logger.warning(f"⚠️ No events for {node_name}, returning uniform")
            return self._create_uniform_position_subpred_cpt(node_name,
                ['exit_now', 'hold', 'extend'],
                ['Delta_Confirming', 'Time_Since_Entry', 'Current_PnL_ATR'])
        
        in_event_data = labeled_data[labeled_data['event_id'].notna()].copy()
        
        generator = ExitTimingGenerator()
        cpt_results = generator.generate_cpt(in_event_data)
        
        formatted_probs = {
            "|".join(k): v for k, v in cpt_results.items()
        }
        
        cpt_data = {
            'node': node_name,
            'parents': ['Delta_Confirming', 'Time_Since_Entry', 'Current_PnL_ATR'],
            'states': ['exit_now', 'hold', 'extend'],
            'conditional_probabilities': formatted_probs,
            'type': 'conditional',
            'metadata': {'version': '3.3', 'n_events': len(events)},
            'observations': len(in_event_data)
        }
        
        # REASON: Valideer CPT kwaliteit (coverage, entropy) zoals bij globale nodes
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        
        return cpt_data


    def _create_uniform_position_subpred_cpt(
        self, 
        node_name: str, 
        states: List[str],
        parents: List[str]
    ) -> Dict[str, Any]:
        """Helper: maak uniform CPT voor position subprediction nodes."""
        uniform_prob = 1.0 / len(states)
        probs = {state: uniform_prob for state in states}
        
        return {
            'node': node_name,
            'parents': parents,
            'states': states,
            'conditional_probabilities': {'uniform': probs},
            'type': 'conditional',
            'metadata': {'version': '3.3', 'uniform': True},
            'observations': 0
        }

    def _generate_position_prediction_cpt(self, asset_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genereer Position_Prediction CPT (v3.1).
        
        Deze node voorspelt de uitkomst (target_hit, stoploss_hit, timeout) voor
        actieve posities op basis van:
        - Position_Confidence
        - Time_Since_Entry
        - Current_PnL_ATR
        """
        node_name = "Position_Prediction"
        logger.info(f"🎯 Generating {node_name} CPT for asset {asset_id}")

        # 1. Detecteer events (training data moet rijen binnen event windows hebben)
        events, labeled_data = self.event_detector.detect_events(data, asset_id)
        
        if not events:
            logger.warning(f"⚠️ No events detected for asset {asset_id}, skipping {node_name}")
            return self._create_uniform_composite_cpt(node_name)

        # 2. Filter voor rijen binnen events
        in_event_data = labeled_data[labeled_data['event_id'].notna()].copy()
        
        # 3. Voeg Current_PnL_ATR toe (moet in 'data' zitten of berekend worden)
        if 'current_pnl_atr' not in in_event_data.columns:
            # REASON: Position_Prediction en Exit_Timing gebruiken PnL-states in ATR units.
            # Zonder current_pnl_atr valt alles terug naar 'breakeven' en verliest de CPT discriminatie.
            if (
                'return_since_entry' in in_event_data.columns
                and 'entry_atr' in in_event_data.columns
                and 'entry_close' in in_event_data.columns
            ):
                entry_atr = pd.to_numeric(in_event_data['entry_atr'], errors='coerce')
                entry_close = pd.to_numeric(in_event_data['entry_close'], errors='coerce')
                ret = pd.to_numeric(in_event_data['return_since_entry'], errors='coerce')

                # return_since_entry = (current_close - entry_close) / entry_close (direction-aware)
                # current_pnl_atr = (current_close - entry_close) / entry_atr
                # => current_pnl_atr = ret * entry_close / entry_atr
                pnl_atr = (ret * entry_close) / entry_atr.replace(0, np.nan)
                in_event_data['current_pnl_atr'] = pnl_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                avg_abs = float(np.nanmean(np.abs(in_event_data['current_pnl_atr'].to_numpy())))
                logger.info(f"Computed current_pnl_atr from return_since_entry (avg_abs={avg_abs:.3f})")
            elif (
                'return_since_entry' in in_event_data.columns
                and 'entry_atr' in in_event_data.columns
                and 'close' in in_event_data.columns
            ):
                # Fallback: derive entry_close uit current close en return (entry_close = close / (1 + ret))
                # REASON: sommige datasets missen entry_close kolom, maar hebben wel close + return_since_entry.
                entry_atr = pd.to_numeric(in_event_data['entry_atr'], errors='coerce')
                close = pd.to_numeric(in_event_data['close'], errors='coerce')
                ret = pd.to_numeric(in_event_data['return_since_entry'], errors='coerce')

                denom = (1.0 + ret).replace(0, np.nan)
                entry_close_est = close / denom
                pnl_atr = (ret * entry_close_est) / entry_atr.replace(0, np.nan)
                in_event_data['current_pnl_atr'] = pnl_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                avg_abs = float(np.nanmean(np.abs(in_event_data['current_pnl_atr'].to_numpy())))
                logger.info(f"Computed current_pnl_atr via derived entry_close (avg_abs={avg_abs:.3f})")
            else:
                logger.warning("current_pnl_atr not found and cannot be computed, defaulting to 0.0")
                in_event_data['current_pnl_atr'] = 0.0

        # 4. Genereer CPT
        cpt_results = self.position_prediction_gen.generate_cpt(in_event_data)
        
        # 5. Save to DB
        self.position_prediction_gen.save_cpt(asset_id, self.run_id)
        
        # Formatteer voor return
        formatted_probs = {
            "|".join(k): v for k, v in cpt_results.items()
        }
        
        cpt_data = {
            'node': node_name,
            'parents': ['Position_Confidence', 'Time_Since_Entry', 'Current_PnL_ATR'],
            'states': self.position_prediction_gen.PREDICTION_STATES,
            'conditional_probabilities': formatted_probs,
            'type': 'conditional',
            'metadata': {
                'version': '3.1',
                'n_events': len(events),
                'observations': len(in_event_data)
            },
            'outcome_mode': 'position'
        }
        
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        cpt_data['observations'] = len(in_event_data)
        
        return cpt_data

    def _fetch_barrier_outcomes(self, asset_id: int, window_min: int = 240, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Haal barrier outcomes op met state mapping.
        
        Args:
            asset_id: Asset identifier
            window_min: Window in minuten (bijv. 240 voor 4h)
            lookback_days: Aantal dagen terug (None = alles)
        """
        # REASON: Als lookback_days None is, halen we 10 jaar aan data op (effectief alles)
        # EXPL: Defaulting naar 30 dagen zorgde voor te kleine datasets in full training runs.
        lookback = lookback_days if lookback_days is not None else 3650

        # REASON: training_weight bevat IDA gewichten (López de Prado Soft-Attribution Delta)
        # EXPL: Dit voorkomt dat langdurige trends de CPT domineren door signalen met
        #       hogere informatiewaarde (delta) meer gewicht te geven.
        query = """
            SELECT 
                bo.time_1,
                bo.first_significant_barrier,
                bo.first_significant_time_min,
                bo.training_weight,
                CASE
                    WHEN bo.first_significant_barrier IS NULL 
                         OR bo.first_significant_barrier = 'none'
                         OR bo.first_significant_time_min > %(window)s
                    THEN 'neutral'
                    WHEN bo.first_significant_barrier LIKE 'up_1%%' 
                         AND bo.first_significant_time_min <= %(window)s
                    THEN 'up_strong'
                    WHEN bo.first_significant_barrier LIKE 'up_%%' 
                         AND bo.first_significant_time_min <= %(window)s
                    THEN 'up_weak'
                    WHEN bo.first_significant_barrier LIKE 'down_1%%' 
                         AND bo.first_significant_time_min <= %(window)s
                    THEN 'down_strong'
                    WHEN bo.first_significant_barrier LIKE 'down_%%' 
                         AND bo.first_significant_time_min <= %(window)s
                    THEN 'down_weak'
                    ELSE 'neutral'
                END as barrier_state
            FROM qbn.barrier_outcomes bo
            WHERE bo.asset_id = %(asset_id)s
              AND bo.time_1 >= NOW() - INTERVAL '1 day' * %(lookback)s
            ORDER BY bo.time_1
        """
        
        with get_cursor() as cur:
            cur.execute(query, {
                'asset_id': asset_id, 
                'window': window_min,
                'lookback': lookback
            })
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=columns)
        logger.info(f"Fetched {len(df)} barrier outcomes for asset {asset_id} (window={window_min}m)")
        return df

    def generate_prediction_cpt(
        self,
        asset_id: int,
        horizon: str,
        data: pd.DataFrame,
        lookback_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Genereer CPT voor prediction node op basis van HTF + Hypothesis + Confidence."""
        node_name = f"Prediction_{horizon}"
        
        # Mapping van horizon naar window in minuten
        window_map = {'1h': 60, '4h': 240, '1d': 1440}
        window_min = window_map.get(horizon, 240)
        
        # REASON: Altijd barrier modus gebruiken
        outcome_mode = 'barrier'
        logger.info(f"Generating {node_name} CPT for asset {asset_id} (mode={outcome_mode})")
        
        # Haal barrier outcomes op voor de specifieke window van deze horizon
        # REASON: Default naar 10 jaar als geen lookback is opgegeven voor volledige training
        outcomes = self._fetch_barrier_outcomes(asset_id, window_min, lookback_days=lookback_days)
        if outcomes.empty:
            logger.warning(f"No barrier outcomes found for {node_name}")
            return self._create_uniform_prediction_cpt(node_name)
        
        # Verwijder een eventueel bestaande barrier_state kolom om duplicates te voorkomen
        if 'barrier_state' in data.columns:
            data = data.drop(columns=['barrier_state'])
            
        # REASON: Merge met training_weight voor IDA weighting (López de Prado)
        # EXPL: training_weight corrigeert voor seriële correlatie in barrier outcomes
        merge_cols = ['time_1', 'barrier_state']
        if 'training_weight' in outcomes.columns:
            merge_cols.append('training_weight')
        merged_data = pd.merge(data, outcomes[merge_cols], on='time_1', how='inner')
        outcome_col = 'barrier_state'
        states = BarrierOutcomeState.state_names()

        if len(merged_data) < self.min_total_obs:
            logger.warning(f"Insufficient data for {node_name}: {len(merged_data)} < {self.min_total_obs}")
            return self._create_uniform_prediction_cpt(node_name)
        
        # REASON: IDA row_weights voor seriële correlatie correctie
        row_weights = None
        if 'training_weight' in merged_data.columns:
            row_weights = merged_data['training_weight'].fillna(1.0).values
            eff_obs = row_weights.sum()
            logger.info(f"   IDA weighting: {len(merged_data)} rows -> {eff_obs:.1f} effective obs")
        else:
            logger.warning(f"   ⚠️ No training_weight found - using uniform weights. Run compute_barrier_weights.py first.")
        
        # v3.1 Parents: HTF_Regime + Trade_Hypothesis (Entry_Confidence verwijderd)
        parent_nodes = ['htf_regime', 'trade_hypothesis']
        
        # REASON: Neutral downsampling om class imbalance te corrigeren
        state_weights = {'neutral': self.neutral_downsample, 'Neutral': self.neutral_downsample}
        
        cpt_data = self.gpu_generator._generate_conditional_cpt_gpu(
            merged_data, node_name, parent_nodes, db_columns=[outcome_col], aggregation_method='direct', 
            states=states,
            state_weights=state_weights,
            row_weights=row_weights
        )
        
        # Metadata toevoegen over de outcome mode
        cpt_data['outcome_mode'] = outcome_mode
        cpt_data['window_min'] = window_min
        
        # REASON: Apply Combination Alpha refinement (Phase 2.5)
        cpt_data = self._apply_combination_alpha_refinement(cpt_data, horizon, asset_id)
        
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        cpt_data['observations'] = len(merged_data)
        
        return cpt_data

    def generate_composite_cpt(
        self,
        asset_id: int,
        semantic_class: SemanticClass,
        data: pd.DataFrame,
        horizon: str = '1h'
    ) -> Dict[str, Any]:
        """Genereer CPT voor composite node (Leading/Coincident/Confirming)."""
        node_name = f"{semantic_class.value.capitalize()}_Composite"
        
        logger.info(f"Generating {node_name} CPT for asset {asset_id} (horizon={horizon})")
        
        self.load_signal_classification(asset_id=asset_id, horizon=horizon, filter_suffix='60')
            
        available_signal_cols = []
        weights_list = []
        for sig_full_name, info in self.signal_aggregator.signal_classification.items():
            if info['semantic_class'] == semantic_class.value:
                if sig_full_name in data.columns:
                    available_signal_cols.append(sig_full_name)
                    weights_list.append(info['weight'])
        
        if not available_signal_cols:
            logger.warning(f"No signal columns found for {node_name}")
            return self._create_uniform_composite_cpt(node_name)

        weights_array = np.array(weights_list)
        parent_nodes = ['htf_regime']
        state_weights = {'neutral': self.neutral_downsample}
        
        # REASON: Haal thresholds uit ThresholdLoader voor de juiste aggregatie
        t_loader = self._get_threshold_loader(asset_id, horizon)
        params = t_loader.get_composite_params(semantic_class.value)
        thresholds_dict = {
            'neutral_band': params['neutral_band'],
            'strong_threshold': params['strong_threshold']
        }
        
        cpt_data = self.gpu_generator._generate_conditional_cpt_gpu(
            data, node_name, parent_nodes, db_columns=available_signal_cols, 
            aggregation_method='weighted_majority', 
            states=[s.value for s in CompositeState],
            weights=weights_array,
            thresholds=thresholds_dict,
            state_weights=state_weights
        )
        
        validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
        cpt_data['validation'] = validation.__dict__
        cpt_data['observations'] = len(data)
        
        from config.bayesian_config import QBNv2Config
        config = QBNv2Config()
        cpt_data['state_reduction_level'] = "REDUCED" if config.use_reduced_regimes else "FULL"
        
        return cpt_data

    def generate_htf_regime_cpt(
        self,
        asset_id: int,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Genereer prior CPT voor HTF_Regime root node."""
        node_name = "HTF_Regime"
        logger.info(f"Generating {node_name} prior CPT for asset {asset_id}")
        
        states = RegimeState.all_states()
        from config.bayesian_config import QBNv2Config
        config = QBNv2Config()
        reduction_level = "REDUCED" if config.use_reduced_regimes else "FULL"
        
        state_to_idx = {state: i for i, state in enumerate(states)}
        data['regime_idx'] = data['htf_regime'].map(state_to_idx)
        
        indices = np.arange(len(states))
        state_weights = {s: self.neutral_downsample for s in states if 'ranging' in s or 'neutral' in s}
        
        freq_dict_idx = self.gpu_generator._count_frequencies_gpu(data['regime_idx'].values, indices)
        
        weighted_counts = {}
        total_weighted = 0.0
        for idx in indices:
            state_val = states[idx]
            count = freq_dict_idx.get(idx, 0)
            weight = state_weights.get(state_val, 1.0)
            weighted_count = count * weight
            weighted_counts[state_val] = weighted_count
            total_weighted += weighted_count

        probabilities = {}
        for state in states:
            count = weighted_counts.get(state, 0.0)
            prob = (count + self.laplace_alpha) / (total_weighted + self.laplace_alpha * len(states))
            probabilities[state] = prob
            
        cpt_data = {
            'node': node_name,
            'parents': [],
            'states': states,
            'probabilities': probabilities,
            'type': 'prior',
            'observations': len(data),
            'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'state_reduction_level': reduction_level
        }
        
        validation = self._validate_cpt_quality(cpt_data)
        cpt_data['validation'] = validation.__dict__
        
        return cpt_data

    def generate_all_cpts(
        self,
        asset_id: int,
        lookback_days: Optional[int] = None,
        save_to_db: bool = True,
        validate_quality: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Volledige pipeline voor een enkel asset."""
        # REASON: Altijd barrier modus gebruiken
        logger.info(f"🚀 Starting full QBN v3 CPT generation for asset {asset_id} (mode=barrier)")

        self._snapshot_db_config(asset_id)
        data = self._prepare_merged_dataset([asset_id], lookback_days)

        if data.empty:
            logger.error(f"No data found for asset {asset_id}")
            return {}

        # Validation guards: check upstream barrier_outcomes + event_windows
        if hasattr(self, 'run_id') and self.run_id:
            try:
                from database.db import get_cursor
                with get_cursor() as cur:
                    validate_step_input(
                        conn=cur.connection,
                        step_name="cpt_generation_barriers",
                        upstream_table="qbn.barrier_outcomes",
                        asset_id=asset_id,
                        run_id=None,  # REASON: barrier_outcomes is global, heeft geen run_id kolom
                        min_rows=100,
                        extra_where="training_weight IS NOT NULL",
                        log_run_id=self.run_id  # REASON: Log wel de echte run_id voor traceability
                    )
                    validate_step_input(
                        conn=cur.connection,
                        step_name="cpt_generation_events",
                        upstream_table="qbn.event_windows",
                        asset_id=asset_id,
                        run_id=self.run_id,
                        min_rows=10
                    )
            except StepValidationError as e:
                logger.info(f"Upstream validation note: {e}")
            except Exception as e:
                logger.warning(f"Upstream validation failed (DB issue): {e}")

        cpts = {}

        # 1. HTF Regime (Root)
        cpts['HTF_Regime'] = self.generate_htf_regime_cpt(asset_id, data=data)

        # 2. Composite Nodes
        default_horizon = '1h'
        for sem_class in SemanticClass:
            node_name = f"{sem_class.value.capitalize()}_Composite"
            cpts[node_name] = self.generate_composite_cpt(asset_id, sem_class, data=data, horizon=default_horizon)

        # 3. v3.1 Intermediate Nodes (Entry_Confidence verwijderd)
        cpts['Trade_Hypothesis'] = self.generate_trade_hypothesis_cpt(asset_id, data=data)
        
        # v3.2 Legacy: Position_Confidence (behouden voor backward compatibility)
        cpts['Position_Confidence'] = self.generate_position_confidence_cpt(asset_id, data=data)
        
        # 4. v3.4 Direct Sub-Predictions Position Nodes
        logger.info(f"🔧 Generating v3.4 Direct Sub-Predictions CPTs for asset {asset_id}")
        cpts['Momentum_Prediction'] = self.generate_momentum_prediction_cpt(asset_id, data=data)
        cpts['Volatility_Regime'] = self.generate_volatility_regime_cpt(asset_id, data=data)
        cpts['Exit_Timing'] = self.generate_exit_timing_cpt(asset_id, data=data)
        
        # v3.4: Risk_Adjusted_Confidence NIET meer genereren (deprecated)
        # v3.3 LEGACY: Uncomment de volgende regel voor backward compatibility
        # cpts['Risk_Adjusted_Confidence'] = self.generate_risk_adjusted_confidence_cpt()

        # 5. v3.4 Position Prediction (direct met MP/VR/ET als parents)
        cpts['Position_Prediction'] = self._generate_position_prediction_cpt(asset_id, data=data)

        # 5. Prediction Nodes
        for horizon in ['1h', '4h', '1d']:
            node_name = f"Prediction_{horizon}"
            cpts[node_name] = self.generate_prediction_cpt(
                asset_id, horizon, data=data, 
                lookback_days=lookback_days
            )

        if validate_quality and lookback_days and lookback_days > 14:
            logger.info(f"🧪 Running quality validation (stability check) for asset {asset_id}")
            self._run_stability_validation(asset_id, cpts, lookback_days)

        if save_to_db:
            from inference.cpt_cache_manager import CPTCacheManager
            cache = CPTCacheManager()
            scope_key = f'asset_{asset_id}'
            for node_name, cpt_data in cpts.items():
                # REASON: CPTCacheManager bepaalt nu zelf of mode relevant is
                cache.save_cpt(
                    asset_id, node_name, cpt_data,
                    scope_type='single',
                    scope_key=scope_key,
                    source_assets=[asset_id],
                    run_id=self.run_id
                )

        logger.info(f"✅ Generated {len(cpts)} CPTs for asset {asset_id}")
        
        # HANDSHAKE_OUT logging (alleen als daadwerkelijk naar DB geschreven)
        if save_to_db:
            log_handshake_out(
                step="qbn_v3_cpt_generator",
                target="qbn.cpt_cache",
                run_id=self.run_id or "N/A",
                rows=len(cpts),
                operation="INSERT"
            )
        
        return cpts

    def generate_composite_cpts(
        self,
        scope_key: str,
        asset_ids: List[int],
        lookback_days: Optional[int] = None,
        save_to_db: bool = True,
        validate_quality: bool = True,
        reference_asset_id: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Genereer CPTs voor meerdere assets gecombineerd.

        Args:
            scope_key: Unieke identifier voor deze scope (bijv. 'top_10', 'all_assets')
            asset_ids: Lijst van asset IDs om te combineren
            lookback_days: Training window in dagen
            save_to_db: Opslaan in database
            validate_quality: Kwaliteitsvalidatie uitvoeren
            reference_asset_id: Asset ID voor thresholds (eerste asset als None)

        Returns:
            Dict met node_name -> cpt_data mapping
        """
        # REASON: Altijd barrier modus gebruiken
        outcome_mode = 'barrier'
        scope_type = 'global' if scope_key == 'all_assets' else 'composite'
        logger.info(f"🚀 Starting composite CPT generation: {scope_key} ({len(asset_ids)} assets, mode={outcome_mode})")
        logger.info(f"   Assets: {asset_ids[:10]}{'...' if len(asset_ids) > 10 else ''}")

        # Reference asset voor thresholds en classificatie
        ref_asset = reference_asset_id or asset_ids[0]
        self._snapshot_db_config(ref_asset)

        # Fetch en merge data voor alle assets
        data = self._prepare_merged_dataset(asset_ids, lookback_days, reference_asset_id=ref_asset)

        if data.empty:
            logger.error(f"No data found for composite scope {scope_key}")
            return {}

        logger.info(f"📊 Composite dataset: {len(data)} total rows from {data['asset_id'].nunique()} assets")

        cpts = {}

        # 1. HTF Regime (Root) - over alle assets
        cpts['HTF_Regime'] = self._generate_composite_htf_regime_cpt(data, asset_ids)

        # 2. Composite Nodes - over alle assets
        default_horizon = '1h'
        for sem_class in SemanticClass:
            node_name = f"{sem_class.value.capitalize()}_Composite"
            cpts[node_name] = self._generate_composite_semantic_cpt(
                data, sem_class, ref_asset, horizon=default_horizon
            )

        # 3. v3.1 Intermediate Nodes (Entry_Confidence verwijderd)
        cpts['Trade_Hypothesis'] = self.generate_trade_hypothesis_cpt(ref_asset, data=data)
        cpts['Position_Confidence'] = self.generate_position_confidence_cpt(ref_asset, data=data)

        # 4. Prediction Nodes - over alle assets
        for horizon in ['1h', '4h', '1d']:
            node_name = f"Prediction_{horizon}"
            cpts[node_name] = self.generate_prediction_cpt(
                ref_asset, horizon, data=data, 
                lookback_days=lookback_days
            )

        # Voeg metadata toe aan alle CPTs
        for node_name, cpt_data in cpts.items():
            cpt_data['source_assets'] = asset_ids
            cpt_data['scope_key'] = scope_key
            cpt_data['scope_type'] = scope_type

        if save_to_db:
            from inference.cpt_cache_manager import CPTCacheManager
            cache = CPTCacheManager()
            for node_name, cpt_data in cpts.items():
                # REASON: CPTCacheManager bepaalt nu zelf of mode relevant is
                cache.save_composite_cpt(
                    scope_key=scope_key,
                    node_name=node_name,
                    cpt_data=cpt_data,
                    source_assets=asset_ids,
                    scope_type=scope_type,
                    run_id=self.run_id
                )

        logger.info(f"✅ Generated {len(cpts)} composite CPTs for scope '{scope_key}'")
        return cpts

    def _generate_composite_htf_regime_cpt(
        self,
        data: pd.DataFrame,
        source_assets: List[int]
    ) -> Dict[str, Any]:
        """Genereer HTF Regime CPT over meerdere assets."""
        node_name = "HTF_Regime"
        logger.info(f"Generating composite {node_name} prior CPT")

        states = RegimeState.all_states()
        from config.bayesian_config import QBNv2Config
        config = QBNv2Config()
        reduction_level = "REDUCED" if config.use_reduced_regimes else "FULL"

        state_to_idx = {state: i for i, state in enumerate(states)}
        data['regime_idx'] = data['htf_regime'].map(state_to_idx)

        indices = np.arange(len(states))
        state_weights = {s: self.NEUTRAL_DOWNSAMPLE_WEIGHT for s in states if 'ranging' in s or 'neutral' in s}

        freq_dict_idx = self.gpu_generator._count_frequencies_gpu(data['regime_idx'].values, indices)

        weighted_counts = {}
        total_weighted = 0.0
        for idx in indices:
            state_val = states[idx]
            count = freq_dict_idx.get(idx, 0)
            weight = state_weights.get(state_val, 1.0)
            weighted_count = count * weight
            weighted_counts[state_val] = weighted_count
            total_weighted += weighted_count

        probabilities = {}
        for state in states:
            count = weighted_counts.get(state, 0.0)
            prob = (count + self.laplace_alpha) / (total_weighted + self.laplace_alpha * len(states))
            probabilities[state] = prob

        cpt_data = {
            'node': node_name,
            'parents': [],
            'states': states,
            'probabilities': probabilities,
            'type': 'prior',
            'observations': len(data),
            'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'state_reduction_level': reduction_level,
            'source_assets': source_assets
        }

        validation = self._validate_cpt_quality(cpt_data)
        cpt_data['validation'] = validation.__dict__

        return cpt_data

    def _generate_composite_semantic_cpt(
        self,
        data: pd.DataFrame,
        semantic_class: SemanticClass,
        reference_asset_id: int,
        horizon: str = '1h'
    ) -> Dict[str, Any]:
        """Genereer semantic composite CPT over meerdere assets."""
        node_name = f"{semantic_class.value.capitalize()}_Composite"

        logger.info(f"Generating composite {node_name} CPT")

        self.load_signal_classification(asset_id=reference_asset_id, horizon=horizon, filter_suffix='60')

        available_signal_cols = []
        weights_list = []
        for sig_full_name, info in self.signal_aggregator.signal_classification.items():
            if info['semantic_class'] == semantic_class.value:
                if sig_full_name in data.columns:
                    available_signal_cols.append(sig_full_name)
                    weights_list.append(info['weight'])

        if not available_signal_cols:
            logger.warning(f"No signal columns found for composite {node_name}")
            return self._create_uniform_composite_cpt(node_name)

        weights_array = np.array(weights_list)
        parent_nodes = ['htf_regime']
        state_weights = {'neutral': self.NEUTRAL_DOWNSAMPLE_WEIGHT}

        threshold_loader = self._get_threshold_loader(reference_asset_id, horizon)
        params = threshold_loader.get_composite_params(semantic_class.value)
        thresholds_dict = {
            'neutral_band': params['neutral_band'],
            'strong_threshold': params['strong_threshold']
        }

        cpt_data = self.gpu_generator._generate_conditional_cpt_gpu(
            data, node_name, parent_nodes, db_columns=available_signal_cols,
            aggregation_method='weighted_majority',
            states=[s.value for s in CompositeState],
            weights=weights_array,
            thresholds=thresholds_dict,
            state_weights=state_weights
        )

        validation = self._validate_cpt_quality(cpt_data, asset_id=reference_asset_id)
        cpt_data['validation'] = validation.__dict__
        cpt_data['observations'] = len(data)

        from config.bayesian_config import QBNv2Config
        config = QBNv2Config()
        cpt_data['state_reduction_level'] = "REDUCED" if config.use_reduced_regimes else "FULL"

        return cpt_data

    def validate_existing_cpts(
        self,
        asset_id: int,
        lookback_days: int = 30
    ):
        """Valideert bestaande CPT's in de database zonder ze opnieuw te genereren."""
        logger.info(f"🧪 Starting validation of existing CPTs for asset {asset_id} (lookback={lookback_days} days)")
        
        self._snapshot_db_config(asset_id)
        
        from inference.cpt_cache_manager import CPTCacheManager
        cache = CPTCacheManager()
        
        # 1. Haal alle node namen op voor dit asset
        with get_cursor() as cur:
            logger.debug(f"🔍 Fetching existing CPT nodes from qbn.cpt_cache for asset {asset_id}...")
            cur.execute("SELECT node_name FROM qbn.cpt_cache WHERE asset_id = %s", (asset_id,))
            node_names = [row[0] for row in cur.fetchall()]
            
        if not node_names:
            logger.error(f"No existing CPTs found in cache for asset {asset_id}")
            return
            
        logger.info(f"✅ Found {len(node_names)} CPT nodes to validate")
            
        # 2. Dataset voorbereiden (nodig voor quality checks)
        data = self._prepare_merged_dataset(asset_id, lookback_days)
        if data.empty:
            logger.error(f"No data found for validation (asset {asset_id})")
            return

        cpts_to_validate = {}
        for node_name in node_names:
            # Sla de cache leeftijd over om altijd de laatste te pakken
            cpt_data = cache.get_cpt(asset_id, node_name, max_age_hours=999999) 
            if cpt_data:
                cpts_to_validate[node_name] = cpt_data

        if not cpts_to_validate:
            logger.error(f"Could not load CPT data for asset {asset_id}")
            return

        # 3. Kwaliteit en Stabiliteit checks
        for node_name, cpt_data in cpts_to_validate.items():
            validation = self._validate_cpt_quality(cpt_data, asset_id=asset_id)
            cpt_data['validation'] = validation.__dict__
            # REASON: Gebruik de bestaande observations uit de cache i.p.v. de validatie-set grootte
            # EXPL: Voorkomt dat een stabiliteitscheck met 30d lookback de 50k training observations overschrijft.
            cpt_data['observations'] = cpt_data.get('observations', len(data))

        if lookback_days > 14:
            logger.info(f"🧪 Running stability validation for asset {asset_id}")
            self._run_stability_validation(asset_id, cpts_to_validate, lookback_days)

        # 4. Resultaten opslaan
        for node_name, cpt_data in cpts_to_validate.items():
            # Gebruik bestaande run_id of laat save_cpt een nieuwe genereren
            run_id = cpt_data.get('run_id')
            # REASON: Sla alleen de validatie-metrics op, overschrijf de tabel niet volledig
            cache.save_cpt(asset_id, node_name, cpt_data, run_id=run_id)
            
        logger.info(f"✅ Validation of {len(cpts_to_validate)} existing CPTs completed for asset {asset_id}")

    # ========================================================================
    # DATA PREPARATION & VECTORIZED LABELING
    # ========================================================================
    
    def preprocess_dataset(self, data: pd.DataFrame, asset_id: int) -> pd.DataFrame:
        """
        Public methode om alle v3 labels toe te passen op een bestaande dataset.
        Handig voor backtesting waar data buiten de generator wordt opgehaald.
        """
        df = data.copy()
        logger.info(f"🏷️ Preprocessing dataset for asset {asset_id} ({len(df)} rows)...")
        
        # 1. HTF Regime
        df['htf_regime'] = self._vectorized_classify_regime(df)
        
        # 2. Composite states berekenen
        self.load_signal_classification(asset_id=asset_id, horizon='1h', filter_suffix='60')
        for sem_class in SemanticClass:
            class_signals = [(sig, info['weight']) for sig, info in self.signal_aggregator.signal_classification.items()
                            if info['semantic_class'] == sem_class.value]
            cols = [s for s, w in class_signals if s in df.columns]
            weights = np.array([w for s, w in class_signals if s in df.columns])

            comp_col = f"{sem_class.value.lower()}_composite"
            if cols:
                # REASON: Haal thresholds uit ThresholdLoader voor de juiste aggregatie
                t_loader = self._get_threshold_loader(asset_id, '1h')
                params = t_loader.get_composite_params(sem_class.value)
                thresholds_dict = {
                    'neutral_band': params['neutral_band'],
                    'strong_threshold': params['strong_threshold']
                }
                df[comp_col] = self.gpu_generator._aggregate_signals_gpu(
                    df, cols, weights=weights, thresholds=thresholds_dict
                )
            else:
                df[comp_col] = 'neutral'

        # 3. v3.1 Node labels afleiden
        # Trade_Hypothesis (Leading)
        df['trade_hypothesis'] = df['leading_composite'].apply(self.hypothesis_gen.derive_hypothesis)

        # Position_Confidence (v3.1: Coincident + Confirming)
        # Note: derive_confidence neemt nu beide composite states
        df['position_confidence'] = df.apply(
            lambda r: self.position_gen.get_confidence(
                r['coincident_composite'],
                r['confirming_composite'],
                time_since_entry_min=0
            )[0],
            axis=1
        )

        # 4. Barrier state afleiden indien nodig (voor backtesting)
        # REASON: Backtest data loader haalt raw barrier columns op, maar we hebben 'barrier_state' nodig voor refinement.
        if 'first_significant_barrier' in df.columns and 'barrier_state' not in df.columns:
            logger.info("Computing barrier_state from raw barrier columns...")
            horizon_min = 60 # Default 1h voor entry refinement
            
            def derive_barrier_state(row):
                barrier = row.get('first_significant_barrier')
                time_min = row.get('first_significant_time_min')
                
                if pd.isna(barrier) or barrier == 'none' or (time_min is not None and time_min > horizon_min):
                    return 'neutral'
                
                if barrier.startswith('up_1'): return 'up_strong'
                if barrier.startswith('up_'): return 'up_weak'
                if barrier.startswith('down_1'): return 'down_strong'
                if barrier.startswith('down_'): return 'down_weak'
                return 'neutral'
            
            df['barrier_state'] = df.apply(derive_barrier_state, axis=1)

        return df

    def _add_composite_scores(self, data: pd.DataFrame, asset_id: int) -> pd.DataFrame:
        """
        Bereken en voeg composite scores toe aan de dataset.
        
        REASON: EventWindowDetector.detect_events() heeft leading_score, coincident_score 
        en confirming_score nodig voor event detection en delta berekening.
        
        v3.3 FIX: leading_score wordt nu ook berekend (was eerder ontbrekend, waardoor
        event detection faalde en v3.3 nodes altijd neutral waren).
        
        v3.5 FIX: Gebruikt nu Lopez de Prado formule: sum(signal * weight * polarity) / sum(|weight|)
        consistent met ThresholdOptimizer. Voorheen: sum(signal * polarity) / N.
        
        Args:
            data: DataFrame met individuele signaal kolommen (bijv. rsi_60, macd_60)
            asset_id: Asset ID voor logging
            
        Returns:
            DataFrame met toegevoegde leading_score, coincident_score en confirming_score kolommen
        """
        logger.info(f"   🔧 Adding composite scores for asset {asset_id}...")
        
        # Laad signal classificatie - INCLUSIEF LEADING
        polarity_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
        
        with get_cursor() as cur:
            cur.execute("""
                SELECT signal_name, semantic_class, COALESCE(polarity, 'neutral') as polarity
                FROM qbn.signal_classification
                WHERE semantic_class IN ('LEADING', 'COINCIDENT', 'CONFIRMING')
            """)
            rows = cur.fetchall()
        
        config = {'LEADING': [], 'COINCIDENT': [], 'CONFIRMING': []}
        for signal_name, semantic_class, polarity in rows:
            config[semantic_class].append({
                'signal_name': signal_name.lower(),
                'col_name': f"{signal_name.lower()}_60",
                'polarity': polarity_map.get(polarity.lower(), 0)
            })
        
        # Laad weights uit qbn.signal_weights (fallback naar 1.0)
        # REASON: HYPOTHESIS layer (LEADING) gebruikt altijd 1h weights, CONFIDENCE layer horizon-specifiek
        weights_data = {}  # (signal_name_60, horizon) -> weight
        with get_cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (signal_name, horizon)
                    signal_name, horizon, weight
                FROM qbn.signal_weights
                WHERE (asset_id = %s OR asset_id = 0)
                ORDER BY signal_name, horizon, asset_id DESC
            """, (asset_id,))
            for signal_name, horizon, weight in cur.fetchall():
                weights_data[(signal_name.lower(), horizon)] = float(weight or 1.0)
        
        logger.info(f"      Found {len(config['LEADING'])} LEADING, {len(config['COINCIDENT'])} COINCIDENT, {len(config['CONFIRMING'])} CONFIRMING signals")
        logger.info(f"      Loaded {len(weights_data)} signal weights from qbn.signal_weights")
        
        # Bereken leading_score met Lopez de Prado formule
        if config['LEADING']:
            lead_scores = np.zeros(len(data))
            total_weight = 0.0
            valid_signals = 0
            for sig in config['LEADING']:
                if sig['col_name'] in data.columns:
                    signal_values = pd.to_numeric(data[sig['col_name']], errors='coerce').fillna(0).values
                    # LEADING is HYPOTHESIS layer -> gebruik 1h weights
                    weight = weights_data.get((sig['col_name'], '1h'), 1.0)
                    lead_scores += signal_values * sig['polarity'] * weight
                    total_weight += abs(weight)
                    valid_signals += 1
            
            if total_weight > 0:
                data['leading_score'] = lead_scores / total_weight
                logger.info(f"      leading_score: {valid_signals} signals, total_weight={total_weight:.1f}, range [{data['leading_score'].min():.3f}, {data['leading_score'].max():.3f}]")
            else:
                data['leading_score'] = 0.0
                logger.warning("      ⚠️ No LEADING signal columns found in data")
        else:
            data['leading_score'] = 0.0
        
        # Bereken coincident_score met Lopez de Prado formule
        if config['COINCIDENT']:
            coin_scores = np.zeros(len(data))
            total_weight = 0.0
            valid_signals = 0
            for sig in config['COINCIDENT']:
                if sig['col_name'] in data.columns:
                    signal_values = pd.to_numeric(data[sig['col_name']], errors='coerce').fillna(0).values
                    # COINCIDENT is CONFIDENCE layer -> gebruik 1h weights als default
                    weight = weights_data.get((sig['col_name'], '1h'), 1.0)
                    coin_scores += signal_values * sig['polarity'] * weight
                    total_weight += abs(weight)
                    valid_signals += 1
            
            if total_weight > 0:
                data['coincident_score'] = coin_scores / total_weight
                logger.info(f"      coincident_score: {valid_signals} signals, total_weight={total_weight:.1f}, range [{data['coincident_score'].min():.3f}, {data['coincident_score'].max():.3f}]")
            else:
                data['coincident_score'] = 0.0
                logger.warning("      ⚠️ No COINCIDENT signal columns found in data")
        else:
            data['coincident_score'] = 0.0
        
        # Bereken confirming_score met Lopez de Prado formule
        if config['CONFIRMING']:
            conf_scores = np.zeros(len(data))
            total_weight = 0.0
            valid_signals = 0
            for sig in config['CONFIRMING']:
                if sig['col_name'] in data.columns:
                    signal_values = pd.to_numeric(data[sig['col_name']], errors='coerce').fillna(0).values
                    # CONFIRMING is CONFIDENCE layer -> gebruik 1h weights als default
                    weight = weights_data.get((sig['col_name'], '1h'), 1.0)
                    conf_scores += signal_values * sig['polarity'] * weight
                    total_weight += abs(weight)
                    valid_signals += 1
            
            if total_weight > 0:
                data['confirming_score'] = conf_scores / total_weight
                logger.info(f"      confirming_score: {valid_signals} signals, total_weight={total_weight:.1f}, range [{data['confirming_score'].min():.3f}, {data['confirming_score'].max():.3f}]")
            else:
                data['confirming_score'] = 0.0
                logger.warning("      ⚠️ No CONFIRMING signal columns found in data")
        else:
            data['confirming_score'] = 0.0
        
        # v3.4: Valideer dat close en atr beschikbaar zijn voor position training
        required_position_cols = ['close', 'atr']
        missing = [c for c in required_position_cols if c not in data.columns]
        if missing:
            logger.warning(f"      ⚠️ Missing columns for position training: {missing}")
        else:
            logger.info(f"      Position training columns present: close range [{data['close'].min():.2f}, {data['close'].max():.2f}], "
                        f"atr range [{data['atr'].min():.4f}, {data['atr'].max():.4f}]")
        
        return data

    def _prepare_merged_dataset(
        self,
        asset_ids: List[int],
        lookback_days: Optional[int],
        reference_asset_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Bereidt de volledige dataset voor met v3 labels.

        Args:
            asset_ids: Lijst van asset IDs om data voor te fetchen
            lookback_days: Aantal dagen terug voor data window
            reference_asset_id: Asset ID voor thresholds (eerste asset als None)
        """
        if isinstance(asset_ids, int):
            asset_ids = [asset_ids]

        logger.info(f"Preparing merged dataset for {len(asset_ids)} asset(s): {asset_ids[:5]}{'...' if len(asset_ids) > 5 else ''}")

        dfs = self._fetch_modular_data(asset_ids, lookback_days)
        base_df = dfs.get('base')
        if base_df is None or base_df.empty:
            return pd.DataFrame()

        base_df = base_df.sort_values(['asset_id', 'time_1'])

        def clean_merge(left, right, name):
            if right is None or right.empty: return left
            right = right.sort_values(['asset_id', 'time_1'])
            dup_cols = [c for c in right.columns if c in left.columns and c not in ['time_1', 'asset_id']]
            if dup_cols: right = right.drop(columns=dup_cols)
            # Merge per asset_id
            return pd.merge(left, right, on=['asset_id', 'time_1'], how='left')

        merged = base_df
        for key in ['lead', 'coin', 'conf', 'outcomes']:
            merged = clean_merge(merged, dfs.get(key), key)

        gc.collect()

        # Labels toevoegen via centrale preprocess methode
        # REASON: Voorkom logica-drift tussen training en backtesting
        merged = self.preprocess_dataset(merged, asset_ids[0] if isinstance(asset_ids, list) else asset_ids)

        # 0. Barrier Outcomes toevoegen voor training
        if len(asset_ids) == 1:
            asset_id = asset_ids[0]
            # REASON: Gebruik de doorgegeven lookback_days (of 3650 voor 'all') i.p.v. default 30
            outcomes = self._fetch_barrier_outcomes(asset_id, window_min=240, lookback_days=lookback_days)
            if not outcomes.empty:
                merged = pd.merge(merged, outcomes[['time_1', 'barrier_state']], on='time_1', how='left')
                logger.info(f"Added barrier_state to merged dataset for asset {asset_id}")

        logger.info(f"Merged dataset ready: {len(merged)} rows from {len(asset_ids)} asset(s)")
        return merged

    def _fetch_modular_data(
        self,
        asset_ids: List[int],
        lookback_days: Optional[int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Haalt tabellen op met 60m boundary filtering voor meerdere assets.

        Args:
            asset_ids: Lijst van asset IDs (of enkele int voor backwards compatibility)
            lookback_days: Aantal dagen terug voor data window
        """
        # Backwards compatibility: single int -> list
        if isinstance(asset_ids, int):
            asset_ids = [asset_ids]

        time_filter = ""
        # Params: asset_ids array + optioneel lookback_days
        params = [asset_ids]
        if lookback_days:
            time_filter = "AND time_1 >= NOW() - INTERVAL '%s days'"
            params.append(lookback_days)

        with get_cursor() as cur:
            cur.execute("SELECT signal_name FROM qbn.signal_classification")
            base_signals = [row[0].lower() for row in cur.fetchall()]

        signal_cols = [f"{sig}_60" for sig in base_signals]
        signal_cols.extend(['adx_signal_d', 'adx_signal_240'])
        signal_cols = list(set(signal_cols))

        # v3.4 FIX: close en atr_14 toegevoegd aan base voor position training
        # REASON: EventWindowDetector heeft close (voor return_since_entry) en atr (voor atr_ratio) nodig
        tables = {
            'base': ('kfl.indicators', ['time', 'asset_id', 'close', 'atr_14']),
            'lead': ('kfl.mtf_signals_lead', ['time_1', 'asset_id']),
            'coin': ('kfl.mtf_signals_coin', ['time_1', 'asset_id']),
            'conf': ('kfl.mtf_signals_conf', ['time_1', 'asset_id']),
            'outcomes': ('qbn.barrier_outcomes', ['time_1', 'asset_id', 'first_significant_barrier', 'first_significant_time_min'])
        }
        
        # REASON: qbn.signal_outcomes (point-in-time) is volledig verwijderd
        dfs = {}
        def _fetch_task(key: str, table_info: tuple):
            table_name, required_cols = table_info
            try:
                with get_cursor() as cur:
                    # REASON: Log start of table fetch
                    logger.debug(f"🔍 Fetching data from {table_name} for role '{key}' ({len(asset_ids)} assets)...")

                    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema || '.' || table_name = '{table_name}'")
                    existing_cols = [row[0] for row in cur.fetchall()]
                    valid_signal_cols = [c for c in signal_cols if c in existing_cols]
                    final_cols = list(set(required_cols + valid_signal_cols))

                    time_col = "time" if key == 'base' else "time_1"
                    col_query = ", ".join([f't."{c}" as time_1' if c == time_col else f't."{c}"' for c in final_cols])
                    t_filter = time_filter.replace('time_1', f't."{time_col}"')

                    # REASON: Voeg interval_min filter toe aan de brontabel 't' om duplicaten (1m, 240m, D) te voorkomen
                    interval_filter = "AND t.interval_min = '60'" if key == 'base' else ""

                    # REASON: Gebruik ANY(%s) voor multi-asset queries
                    query = f"""
                        SELECT {col_query}
                        FROM {table_name} t
                        JOIN kfl.indicators i ON t.asset_id = i.asset_id AND t.{time_col} = i.time
                        WHERE t.asset_id = ANY(%s) AND i.interval_min = '60' {interval_filter} {t_filter}
                        ORDER BY t.asset_id, t.{time_col} ASC
                    """
                    cur.execute(query, params)
                    cols = [desc[0] for desc in cur.description]
                    data_rows = cur.fetchall()
                    df = pd.DataFrame(data_rows, columns=cols)

                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if not numeric_cols.empty: df[numeric_cols] = df[numeric_cols].astype(np.float32)

                    # REASON: Log success and data summary
                    logger.info(f"✅ Success: Loaded {len(df)} rows from {table_name} ({len(cols)} columns)")
                    return key, df
            except Exception as e:
                logger.error(f"❌ Error fetching {table_name}: {str(e)}")
                return key, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_fetch_task, k, v) for k, v in tables.items()]
            for future in futures:
                key, df = future.result()
                dfs[key] = df

        # v3.4 FIX: Rename atr_14 naar atr voor EventWindowDetector compatibility
        # REASON: Database heeft atr_14, maar EventWindowDetector verwacht 'atr'
        if 'base' in dfs and 'atr_14' in dfs['base'].columns:
            dfs['base'] = dfs['base'].rename(columns={'atr_14': 'atr'})
            logger.info("   Renamed atr_14 -> atr for position training")

        logger.info(f"📊 Completed data fetching for {len(asset_ids)} asset(s). Total tables loaded: {len(dfs)}")
        return dfs

    def _vectorized_classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized regime detectie met 11-to-5 state reduction support."""
        if 'adx_signal_d' not in df.columns or 'adx_signal_240' not in df.columns:
            return pd.Series(['macro_ranging'] * len(df), index=df.index)

        adx_d = df['adx_signal_d'].fillna(0).astype(int)
        adx_240 = df['adx_signal_240'].fillna(0).astype(int)
        results = np.full(len(df), RegimeState.MACRO_RANGING.value, dtype=object)
        
        bull_d = adx_d > 0
        results[bull_d & (adx_240 > 0) & ((adx_d == 2) | (adx_240 == 2))] = RegimeState.SYNC_STRONG_BULLISH.value
        results[bull_d & (adx_240 > 0) & (adx_d != 2) & (adx_240 != 2)] = RegimeState.SYNC_BULLISH.value
        results[bull_d & (adx_240 < 0)] = RegimeState.BULLISH_RETRACING.value
        results[bull_d & (adx_240 == 0)] = RegimeState.BULLISH_CONSOLIDATING.value
        
        bear_d = adx_d < 0
        results[bear_d & (adx_240 < 0) & ((adx_d == -2) | (adx_240 == -2))] = RegimeState.SYNC_STRONG_BEARISH.value
        results[bear_d & (adx_240 < 0) & (adx_d != -2) & (adx_240 != -2)] = RegimeState.SYNC_BEARISH.value
        results[bear_d & (adx_240 > 0)] = RegimeState.BEARISH_RETRACING.value
        results[bear_d & (adx_240 == 0)] = RegimeState.BEARISH_CONSOLIDATING.value
        
        range_d = adx_d == 0
        results[range_d & (adx_240 > 0)] = RegimeState.BULLISH_EMERGING.value
        results[range_d & (adx_240 < 0)] = RegimeState.BEARISH_EMERGING.value
        
        from config.bayesian_config import QBNv2Config
        from .state_reduction import reduce_regime_array
        if QBNv2Config().use_reduced_regimes:
            results = reduce_regime_array(results)
        return pd.Series(results, index=df.index)

    def _setup_file_logging(self, asset_id: Optional[int] = None):
        # KFL logregels: _log/, {YYMMDD-HH-MM-ss}_{scriptname}.log, archive *_{scriptname}.log, format timestamp, level, message
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        log_dir = project_root / "_log"
        archive_dir = log_dir / "archive"
        log_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        script_name = "qbn_v3_cpt_generator"
        ts = datetime.now().strftime("%y%m%d-%H-%M-%S")
        log_file = log_dir / f"{ts}_{script_name}.log"

        for old_log in log_dir.glob(f"*_{script_name}.log"):
            try:
                if old_log.is_file():
                    shutil.move(str(old_log), str(archive_dir / old_log.name))
            except Exception:
                pass

        file_fmt = "%(asctime)s, %(levelname)s, %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(file_fmt, datefmt=date_fmt))

        root_logger = logging.getLogger()
        is_backtest = any("rolling_backtest" in getattr(h, "baseFilename", "") for h in root_logger.handlers)
        is_strategy_finder = any("strategy_finder" in getattr(h, "baseFilename", "") for h in root_logger.handlers)

        if not is_backtest and not is_strategy_finder:
            for h in root_logger.handlers[:]:
                if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith(".log"):
                    root_logger.removeHandler(h)
            root_logger.addHandler(file_handler)
        else:
            logger.addHandler(file_handler)
            logger.propagate = False

        logger.info("Started. Logging to: %s", log_file)

    def _snapshot_db_config(self, asset_id: Optional[int] = None):
        """Snapshot database config naar gestructureerde output directory."""
        from core.output_manager import ValidationOutputManager
        
        output_mgr = ValidationOutputManager()
        output_dir = output_mgr.create_output_dir(
            script_name="cpt_generation",
            asset_id=asset_id or 0,
            run_id=self.run_id
        )
        
        snapshot_file = output_dir / "config_snapshot_v3.json"
        
        try:
            with get_cursor() as cur:
                logger.debug("📸 Taking snapshot of qbn.signal_classification...")
                cur.execute("SELECT * FROM qbn.signal_classification")
                classification = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
                
                logger.debug("📸 Taking snapshot of qbn.signal_discretization...")
                cur.execute("SELECT * FROM qbn.signal_discretization")
                discretization = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
                
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump({"timestamp": datetime.now().isoformat(), "asset_id": asset_id, "signal_classification": classification, "signal_discretization": discretization}, f, indent=4, default=str)
            logger.info(f"✅ Config snapshot saved to {snapshot_file}")
        except Exception as e: logger.error(f"❌ Config snapshot failed: {e}")

    def _validate_cpt_quality(self, cpt_data: Dict, prior_cpt: Optional[Dict] = None, asset_id: Optional[int] = None) -> CPTValidationResult:
        cond_probs = cpt_data.get('conditional_probabilities', {})
        prior_probs = cpt_data.get('probabilities', {})
        total_cells = sum(len(v) for v in cond_probs.values()) if cond_probs else len(prior_probs)
        filled_cells = sum(1 for v in (cond_probs.values() if cond_probs else [prior_probs]) for p in (v.values() if isinstance(v, dict) else [v]) if p > 0)
        coverage = filled_cells / total_cells if total_cells > 0 else 0
        
        if prior_cpt is None and asset_id:
            prior_cpt = self.base_generator.load_cpt_from_database(asset_id, cpt_data.get('node'))
        
        metrics = self.validator.validate_cpt_quality(cpt_data, prior_cpt or {'probabilities': {s: 1/len(cpt_data.get('states', [1])) for s in cpt_data.get('states', [])}})
        return CPTValidationResult(
            is_valid=(coverage >= self.coverage_threshold), coverage=float(coverage), sparse_cells=0, total_cells=int(total_cells),
            recommendation='ready' if coverage > 0.5 else 'more_data', observations=int(cpt_data.get('observations', 0)),
            entropy=float(metrics.get('entropy', 0.0)), info_gain=float(metrics.get('info_gain', 0.0)), semantic_score=float(metrics.get('semantic_score', 1.0))
        )

    def _run_stability_validation(self, asset_id: int, current_cpts: Dict, lookback_days: int):
        half = lookback_days // 2
        try:
            old_gen = QBNv3CPTGenerator(self.laplace_alpha)
            old_cpts = old_gen.generate_all_cpts(asset_id, lookback_days=half, save_to_db=False, validate_quality=False)
            for name, cpt in current_cpts.items():
                if name in old_cpts:
                    stability = self.validator.calculate_stability(cpt, old_cpts[name])
                    if 'validation' not in cpt: cpt['validation'] = {}
                    cpt['validation']['stability_score'] = stability
        except Exception as e: logger.error(f"Stability check failed: {e}")

    def validate_existing_cpts(self, asset_id: int, lookback_days: int = 3650) -> Dict[str, Dict]:
        """
        Publieke methode voor CPT stability validatie vanuit validation menu.
        
        REASON: Laadt bestaande CPTs uit cache en vergelijkt ze met een vers
        gegenereerde set op basis van halve lookback periode om drift te detecteren.
        
        Args:
            asset_id: Asset ID om te valideren
            lookback_days: Lookback periode voor vergelijking
            
        Returns:
            Dict met node_name -> validation_results mapping
        """
        from inference.cpt_cache_manager import CPTCacheManager
        
        logger.info(f"🔍 Starting CPT stability validation for asset {asset_id} (lookback={lookback_days}d)")
        
        # Laad huidige CPTs uit cache
        cache = CPTCacheManager()
        current_cpts, scope = cache.get_cpts_for_asset_cascade(asset_id)
        
        if not current_cpts:
            logger.warning(f"❌ Geen CPTs gevonden in cache voor asset {asset_id}")
            print(f"❌ Geen CPTs gevonden in cache voor asset {asset_id}")
            return {}
        
        logger.info(f"📊 Gevonden: {len(current_cpts)} CPTs in scope '{scope}'")
        print(f"\n📊 Gevonden: {len(current_cpts)} CPTs in scope '{scope}'")
        
        # Genereer vergelijkings-CPTs met halve lookback
        half_lookback = lookback_days // 2
        logger.info(f"🔄 Genereren vergelijkings-CPTs met {half_lookback}d lookback...")
        print(f"🔄 Genereren vergelijkings-CPTs met {half_lookback}d lookback...")
        
        results = {}
        try:
            comparison_gen = QBNv3CPTGenerator(self.laplace_alpha)
            comparison_cpts = comparison_gen.generate_all_cpts(
                asset_id, 
                lookback_days=half_lookback, 
                save_to_db=False, 
                validate_quality=False
            )
            
            print(f"\n{'Node':<30} {'Stability':>10} {'Status':>10}")
            print("-" * 55)
            
            for node_name, current_cpt in current_cpts.items():
                if node_name in comparison_cpts:
                    stability = self.validator.calculate_stability(current_cpt, comparison_cpts[node_name])
                    
                    # Bepaal status
                    if stability >= 0.8:
                        status = "✅ STABLE"
                    elif stability >= 0.5:
                        status = "⚠️ DRIFT"
                    else:
                        status = "❌ UNSTABLE"
                    
                    results[node_name] = {
                        'stability_score': stability,
                        'status': status,
                        'current_obs': current_cpt.get('observations', 0),
                        'comparison_obs': comparison_cpts[node_name].get('observations', 0)
                    }
                    
                    print(f"{node_name:<30} {stability:>10.3f} {status:>10}")
                else:
                    results[node_name] = {
                        'stability_score': None,
                        'status': "⚠️ NO COMPARISON",
                        'current_obs': current_cpt.get('observations', 0),
                        'comparison_obs': 0
                    }
                    print(f"{node_name:<30} {'N/A':>10} {'NO COMP':>10}")
            
            # Summary
            stable_count = sum(1 for r in results.values() if r.get('stability_score', 0) and r['stability_score'] >= 0.8)
            drift_count = sum(1 for r in results.values() if r.get('stability_score') and 0.5 <= r['stability_score'] < 0.8)
            unstable_count = sum(1 for r in results.values() if r.get('stability_score') and r['stability_score'] < 0.5)
            
            print(f"\n📈 Summary: {stable_count} stable, {drift_count} drift, {unstable_count} unstable")
            logger.info(f"✅ Stability validation complete: {stable_count} stable, {drift_count} drift, {unstable_count} unstable")
            
        except Exception as e:
            logger.error(f"❌ Stability validation failed: {e}")
            print(f"❌ Stability validation failed: {e}")
        
        return results

    # Uniform fallbacks
    def _create_uniform_metadata(self): return {"coverage": 0.0, "entropy": 0.0, "info_gain": 0.0, "stability_score": 0.0, "semantic_score": 0.0}
    def _create_uniform_prediction_cpt(self, node_name: str):
        states = BarrierOutcomeState.state_names()
        return {
            'node': node_name, 
            'states': states, 
            'conditional_probabilities': {}, 
            'probabilities': {s: 1/len(states) for s in states}, 
            'type': 'conditional', 
            'observations': 0, 
            'outcome_mode': 'barrier',
            'validation': self._create_uniform_metadata()
        }
    def _create_uniform_composite_cpt(self, node_name: str):
        states = [s.value for s in CompositeState]
        return {
            'node': node_name, 
            'states': states, 
            'conditional_probabilities': {}, 
            'probabilities': {s: 1/len(states) for s in states}, 
            'type': 'conditional', 
            'observations': 0, 
            'validation': self._create_uniform_metadata()
        }
    def _create_uniform_regime_cpt(self):
        states = RegimeState.all_states()
        return {'node': 'HTF_Regime', 'states': states, 'probabilities': {s: 1/len(states) for s in states}, 'type': 'prior', 'observations': 0, 'validation': self._create_uniform_metadata()}

