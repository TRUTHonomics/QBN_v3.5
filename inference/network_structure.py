from typing import Dict, List, Set, Tuple
import networkx as nx
from .node_types import (
    NodeDefinition, NodeType, OutcomeState, 
    RegimeState, CompositeState
)

class QBNv3NetworkStructure:
    """
    Definieert de DAG structuur voor QBN v3.4 (Direct Sub-Predictions Architecture).

    Structuur:
        HTF_Regime (root)
            ↓
        Leading_Composite, Coincident_Composite, Confirming_Composite
            ↓                        ↓                     ↓
        Trade_Hypothesis    Momentum_Prediction   Volatility_Regime   Exit_Timing
            (Entry)              └──────────────────┼──────────────────┘
            ↓                                       ↓
        Prediction_1h, Prediction_4h, Prediction_1d  Position_Prediction
                      (Entry targets)              (Position target)

    v3.4 CHANGES:
    - Risk_Adjusted_Confidence ensemble VERWIJDERD
    - Position_Prediction krijgt DIRECT input van MP/VR/ET (27 combinaties)
    - Position_Confidence legacy node VERWIJDERD
    - Alle sub-predictions gaan direct naar TSEM voor actionable decisions
    """
    
    def __init__(self):
        self.nodes: Dict[str, NodeDefinition] = {}
        self.graph = nx.DiGraph()
        self._build_structure()
    
    def _build_structure(self):
        """Bouw de complete network structuur voor v3."""
        
        # =========================================================
        # ROOT NODE: HTF Regime
        # =========================================================
        self._add_node(NodeDefinition(
            name="HTF_Regime",
            node_type=NodeType.ROOT,
            states=RegimeState.all_states(),
            parents=[],
            children=["Leading_Composite", "Coincident_Composite", "Confirming_Composite"],
            description="Higher Timeframe regime bepaald door D/240 timeframe signalen"
        ))
        
        # =========================================================
        # COMPOSITE NODES: Semantische aggregatie
        # =========================================================
        composite_states = [s.value for s in CompositeState]
        
        self._add_node(NodeDefinition(
            name="Leading_Composite",
            node_type=NodeType.COMPOSITE,
            states=composite_states,
            parents=["HTF_Regime"],
            children=["Trade_Hypothesis", "Momentum_Prediction"],  # v3.4: naar Momentum_Prediction
            description="Aggregatie van LEADING signalen"
        ))
        
        self._add_node(NodeDefinition(
            name="Coincident_Composite",
            node_type=NodeType.COMPOSITE,
            states=composite_states,
            parents=["HTF_Regime"],
            children=["Volatility_Regime"],  # v3.4: Position_Confidence verwijderd
            description="Aggregatie van COINCIDENT signalen"
        ))
        
        self._add_node(NodeDefinition(
            name="Confirming_Composite",
            node_type=NodeType.COMPOSITE,
            states=composite_states,
            parents=["HTF_Regime"],
            children=["Exit_Timing"],  # v3.4: Position_Confidence verwijderd
            description="Aggregatie van CONFIRMING signalen"
        ))
        
        # =========================================================
        # v3.1 INTERMEDIATE LAYERS (Dual-Prediction Architecture)
        # =========================================================

        # 1. Trade_Hypothesis (ALLEEN Leading)
        self._add_node(NodeDefinition(
            name="Trade_Hypothesis",
            node_type=NodeType.ENTRY,
            states=["no_setup", "weak_long", "strong_long", "weak_short", "strong_short"],
            parents=["Leading_Composite"],
            children=["Prediction_1h", "Prediction_4h", "Prediction_1d"],
            description="Trade hypothese gebaseerd op LEADING signalen alleen"
        ))

        # =========================================================
        # v3.4 POSITION SUBPREDICTION NODES (Direct Sub-Predictions)
        # =========================================================
        
        # 2a. Momentum_Prediction (Leading-based)
        self._add_node(NodeDefinition(
            name="Momentum_Prediction",
            node_type=NodeType.POSITION_SUBPREDICTION,
            states=["bearish", "neutral", "bullish"],
            parents=["Leading_Composite"],
            children=["Position_Prediction"],  # v3.4: Direct naar Position_Prediction
            description="Leading-based prijsrichting voorspelling (v3.4)"
        ))
        
        # 2b. Volatility_Regime (Coincident-based)
        self._add_node(NodeDefinition(
            name="Volatility_Regime",
            node_type=NodeType.POSITION_SUBPREDICTION,
            states=["low_vol", "normal", "high_vol"],
            parents=["Coincident_Composite"],
            children=["Position_Prediction"],  # v3.4: Direct naar Position_Prediction
            description="Coincident-based volatiliteit voorspelling (v3.4)"
        ))
        
        # 2c. Exit_Timing (Confirming-based)
        self._add_node(NodeDefinition(
            name="Exit_Timing",
            node_type=NodeType.POSITION_SUBPREDICTION,
            states=["exit_now", "hold", "extend"],
            parents=["Confirming_Composite"],
            children=["Position_Prediction"],  # v3.4: Direct naar Position_Prediction
            description="Confirming-based exit timing voorspelling (v3.4)"
        ))

        # v3.4: Position_Prediction met DIRECTE sub-prediction parents (27 combinaties)
        # REASON: Risk_Adjusted_Confidence ensemble verwijderd - zie 260124_v34_direct_subpredictions.md
        self._add_node(NodeDefinition(
            name="Position_Prediction",
            node_type=NodeType.TARGET,
            states=["target_hit", "stoploss_hit", "timeout"],
            parents=["Momentum_Prediction", "Volatility_Regime", "Exit_Timing"],  # v3.4: 3 directe parents
            children=[],
            description="Voorspelt uitkomst voor actieve posities. "
                        "v3.4: Direct gekoppeld aan MP/VR/ET (27 combinaties)."
        ))
        
        # =========================================================
        # TARGET NODES: Multi-horizon Predictions
        # =========================================================
        outcome_states = OutcomeState.state_names()
        
        for horizon in ["1h", "4h", "1d"]:
            self._add_node(NodeDefinition(
                name=f"Prediction_{horizon}",
                node_type=NodeType.TARGET,
                states=outcome_states,
                parents=["HTF_Regime", "Trade_Hypothesis"],
                children=[],
                description=f"Prediction voor {horizon} horizon (barrier-based: up_strong/up_weak/neutral/down_weak/down_strong)"
            ))
    
    def _add_node(self, node: NodeDefinition):
        """Voeg node toe aan structuur."""
        self.nodes[node.name] = node
        self.graph.add_node(node.name, 
            type=node.node_type.value,
            states=node.states,
            num_states=node.num_states
        )
        
        # Add edges
        for parent in node.parents:
            self.graph.add_edge(parent, node.name)
    
    def get_node(self, name: str) -> NodeDefinition:
        """Haal node definitie op."""
        return self.nodes[name]
    
    def get_parents(self, node_name: str) -> List[str]:
        """Haal parent nodes op."""
        return list(self.graph.predecessors(node_name))
    
    def get_children(self, node_name: str) -> List[str]:
        """Haal child nodes op."""
        return list(self.graph.successors(node_name))
    
    def topological_order(self) -> List[str]:
        """Return nodes in topologische volgorde (voor inference)."""
        return list(nx.topological_sort(self.graph))
    
    def validate_dag(self) -> bool:
        """Valideer dat graph een geldige DAG is."""
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_cpt_dimensions(self, node_name: str) -> Tuple[int, ...]:
        """
        Bereken CPT dimensies voor een node.
        
        Returns:
            Tuple van (parent1_states, parent2_states, ..., node_states)
        """
        node = self.nodes[node_name]
        parent_dims = [self.nodes[p].num_states for p in node.parents]
        return tuple(parent_dims + [node.num_states])
    
    def get_total_cpt_size(self, node_name: str) -> int:
        """Bereken totale CPT grootte voor een node."""
        dims = self.get_cpt_dimensions(node_name)
        size = 1
        for d in dims:
            size *= d
        return size

