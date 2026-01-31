import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# pgmpy imports voor inference
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from .node_types import (
    NodeType, SemanticClass, OutcomeState, 
    RegimeState, CompositeState, NodeDefinition
)
from .network_structure import QBNv3NetworkStructure
from .signal_aggregator import SignalAggregator
from .regime_detector import HTFRegimeDetector
from .state_reduction import StateReductionLevel, get_state_mapping, recommend_reduction_level
from .cpt_generator import ConditionalProbabilityTableGenerator, OUTCOME_STATE_LIST, SIGNAL_STATES
from config.bayesian_config import QBNv2Config, SignalState

logger = logging.getLogger(__name__)

class QBNv3Network:
    """
    Hoofdklasse voor het QBN v3 Bayesian Network.
    Integreert alle modules voor hiërarchische inference.
    """
    
    def __init__(self, config: QBNv2Config, signal_classification: Dict[str, Dict]):
        self.config = config
        self.structure = QBNv3NetworkStructure()
        self.aggregator = SignalAggregator(signal_classification)
        self.regime_detector = HTFRegimeDetector(
            adx_trending_threshold=config.adx_trending_threshold,
            adx_strong_threshold=config.adx_strong_threshold
        )
        self.cpt_generator = ConditionalProbabilityTableGenerator(
            laplace_alpha=config.laplace_smoothing_alpha
        )
        
        self._pgmpy_model: Optional[DiscreteBayesianNetwork] = None
        self._inference_engine: Optional[VariableElimination] = None
        self._is_fitted = False
        
        logger.info("QBNv3Network geïnitialiseerd")

    def fit(self, asset_id: int, lookback_days: Optional[int] = None):
        """
        Train het netwerk door CPT's te genereren uit de database.
        """
        logger.info(f"Fitten van netwerk voor asset {asset_id}")
        
        # 1. Initialiseer pgmpy model op basis van de network_structure
        edges = []
        for node_name, node_def in self.structure.nodes.items():
            for child in node_def.children:
                edges.append((node_name, child))
        
        self._pgmpy_model = DiscreteBayesianNetwork(edges)
        
        # 2. Genereer en voeg CPD's toe voor elke node
        for node_name, node_def in self.structure.nodes.items():
            try:
                # Bepaal target kolommen of db kolommen op basis van node type
                target_col = None
                db_cols = None
                num_states = node_def.num_states
                
                if node_def.node_type == NodeType.TARGET:
                    # Prediction_1h -> outcome_1h
                    horizon = node_name.split('_')[1]
                    target_col = f"outcome_{horizon}"
                
                # Genereer CPT data
                cpt_data = self.cpt_generator.generate_cpt_for_asset(
                    asset_id=asset_id,
                    node_name=node_name,
                    parent_nodes=node_def.parents,
                    lookback_days=lookback_days,
                    target_column=target_col,
                    num_states=num_states
                )
                
                # Convert naar TabularCPD
                cpd = self._convert_to_pgmpy_cpd(node_name, node_def, cpt_data)
                self._pgmpy_model.add_cpds(cpd)
                
            except Exception as e:
                logger.error(f"Fout bij genereren CPT voor {node_name}: {e}")
                # Fallback naar uniforme CPD
                cpd = self._create_uniform_cpd(node_name, node_def)
                self._pgmpy_model.add_cpds(cpd)

        # 3. Valideer model
        if not self._pgmpy_model.check_model():
            raise ValueError("pgmpy model validatie mislukt")
            
        self._inference_engine = VariableElimination(self._pgmpy_model)
        self._is_fitted = True
        logger.info(f"Netwerk succesvol gefit voor asset {asset_id}")

    def get_inference_engine(self, asset_id: int) -> 'TradeAlignedInference':
        """
        Exporteert de getrainde CPT's naar de snelle v3 Inference Engine.
        REASON: pgmpy is te traag en niet thread-safe voor live inference.
        """
        from .inference_loader import InferenceLoader
        loader = InferenceLoader()
        return loader.load_inference_engine(asset_id)

    def infer_predictions(self, active_signals: Dict[str, int], indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        DEPRECATED: Gebruik TradeAlignedInference voor live voorspellingen.
        Deze methode blijft behouden voor pgmpy-gebaseerde validatie checks.
        """
        if not self._is_fitted:
            raise RuntimeError("Netwerk moet eerst gefit worden via fit()")

        # 1. Detecteer HTF Regime
        regime = self.regime_detector.detect_regime(
            adx_d=indicators.get('adx_d'),
            adx_240=indicators.get('adx_240'),
            di_plus_d=indicators.get('di_plus_d'),
            di_minus_d=indicators.get('di_minus_d'),
            macd_histogram_d=indicators.get('macd_h_d')
        )

        # 2. Aggregeer Composites
        composites = self.aggregator.aggregate_all_classes(active_signals)

        # 3. Bereid evidence voor
        # REASON: v3 evidence mapping (voor pgmpy validatie)
        # EXPL: We berekenen de deterministische v3 nodes om de pgmpy query te versnellen.
        from .trade_hypothesis_generator import TradeHypothesisGenerator
        from .entry_confidence_generator import EntryConfidenceGenerator
        
        hyp_gen = TradeHypothesisGenerator()
        conf_gen = EntryConfidenceGenerator()
        
        hypothesis = hyp_gen.derive_hypothesis(composites[SemanticClass.LEADING].value)
        confidence = conf_gen.derive_confidence(
            composites[SemanticClass.COINCIDENT].value,
            composites[SemanticClass.CONFIRMING].value
        )
        
        evidence = {
            "HTF_Regime": regime.value,
            "Leading_Composite": composites[SemanticClass.LEADING].value,
            "Coincident_Composite": composites[SemanticClass.COINCIDENT].value,
            "Confirming_Composite": composites[SemanticClass.CONFIRMING].value,
            "Trade_Hypothesis": hypothesis,
            "Entry_Confidence": confidence
        }

        # 4. Run Inference
        results = {}
        prediction_nodes = [n for n in self.structure.nodes if n.startswith("Prediction_")]
        
        for node in prediction_nodes:
            query = self._inference_engine.query(variables=[node], evidence=evidence, show_progress=False)
            
            # Map probabilities naar states
            probs = query.values.flatten()
            dist = {state: float(prob) for state, prob in zip(OUTCOME_STATE_LIST, probs)}
            
            best_state = max(dist.items(), key=lambda x: x[1])
            expected_atr = self.cpt_generator.calculate_expected_atr_move(dist)
            
            results[node] = {
                "state": best_state[0],
                "probability": best_state[1],
                "expected_atr_move": expected_atr,
                "distribution": dist
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": regime.value,
            "evidence": evidence,
            "predictions": results
        }

    def _convert_to_pgmpy_cpd(self, node_name: str, node_def: NodeDefinition, cpt_data: Dict[str, Any]) -> TabularCPD:
        """Helper om CPT data om te zetten naar pgmpy TabularCPD."""
        states = node_def.states
        
        if cpt_data['type'] in ['prior', 'uniform_prior']:
            probs = cpt_data.get('probabilities', {s: 1.0/len(states) for s in states})
            values = [[probs.get(s, 1.0/len(states))] for s in states]
            
            return TabularCPD(
                variable=node_name,
                variable_card=len(states),
                values=values,
                state_names={node_name: states}
            )
        else:
            # Conditional CPT
            cond_probs = cpt_data['conditional_probabilities']
            parent_names = node_def.parents
            parent_cards = [self.structure.nodes[p].num_states for p in parent_names]
            
            # pgmpy verwacht een matrix waar kolommen de parent combinaties zijn
            values = []
            for state in states:
                row = []
                for combo in cond_probs.keys():
                    prob = cond_probs[combo].get(state, 1.0/len(states))
                    row.append(prob)
                values.append(row)
            
            state_names = {node_name: states}
            for p in parent_names:
                state_names[p] = self.structure.nodes[p].states
                
            return TabularCPD(
                variable=node_name,
                variable_card=len(states),
                values=values,
                evidence=parent_names,
                evidence_card=parent_cards,
                state_names=state_names
            )

    def _create_uniform_cpd(self, node_name: str, node_def: NodeDefinition) -> TabularCPD:
        """Fallback voor ontbrekende CPT data."""
        states = node_def.states
        prob = 1.0 / len(states)
        
        if node_def.is_root:
            values = [[prob] for _ in states]
            return TabularCPD(variable=node_name, variable_card=len(states), values=values, state_names={node_name: states})
        else:
            parent_names = node_def.parents
            parent_cards = [self.structure.nodes[p].num_states for p in parent_names]
            total_combos = np.prod(parent_cards)
            values = [[prob] * total_combos for _ in states]
            
            state_names = {node_name: states}
            for p in parent_names:
                state_names[p] = self.structure.nodes[p].states
                
            return TabularCPD(
                variable=node_name,
                variable_card=len(states),
                values=values,
                evidence=parent_names,
                evidence_card=parent_cards,
                state_names=state_names
            )

