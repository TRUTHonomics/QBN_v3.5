#!/usr/bin/env python3
"""
ConditionalProbabilityTableGenerator - CPT generatie uit historische data

ARCHITECTUUR NOOT:
- CPT generator leest uit kfl.mtf_signals_lead (hypertable)
- Signal columns: {indicator}_signal_{interval} (bv. rsi_signal_d, macd_signal_240)
- Outcome columns: outcome_1h, outcome_4h, outcome_1d (7 ATR-relatieve bins)
- Laplace smoothing voor robuuste probabiliteitsschattingen

OUTCOME STATES (-3 tot +3):
  -3: Strong_Bearish  (return < -1.25 * ATR)
  -2: Bearish         (-1.25*ATR <= return < -0.75*ATR)
  -1: Slight_Bearish  (-0.75*ATR <= return < -0.25*ATR)
   0: Neutral         (-0.25*ATR <= return < +0.25*ATR)
  +1: Slight_Bullish  (+0.25*ATR <= return < +0.75*ATR)
  +2: Bullish         (+0.75*ATR <= return < +1.25*ATR)
  +3: Strong_Bullish  (return >= +1.25*ATR)
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone, timedelta
import json
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Core imports
from config.bayesian_config import SignalState
from database.db import get_cursor

logger = logging.getLogger(__name__)

# Outcome state definitions (7 ATR-relative bins)
OUTCOME_STATES = {
    -3: 'Strong_Bearish',
    -2: 'Bearish',
    -1: 'Slight_Bearish',
    0: 'Neutral',
    1: 'Slight_Bullish',
    2: 'Bullish',
    3: 'Strong_Bullish'
}

OUTCOME_STATE_LIST = ['Strong_Bearish', 'Bearish', 'Slight_Bearish', 'Neutral', 
                      'Slight_Bullish', 'Bullish', 'Strong_Bullish']

# ATR midpoints for expected value calculation
# REASON: Uitgelijnd met de nieuwe lineaire drempels (0.25, 0.75, 1.25)
OUTCOME_ATR_MIDPOINTS = {
    -3: -1.75,  # Midden van < -1.25 (schatting)
    -2: -1.0,   # Midden van -1.25 tot -0.75
    -1: -0.5,   # Midden van -0.75 tot -0.25
    0: 0.0,     # Midden van -0.25 tot 0.25
    1: 0.5,     # Midden van 0.25 tot 0.75
    2: 1.0,     # Midden van 0.75 tot 1.25
    3: 1.75     # Midden van > 1.25 (schatting)
}

# Signal states (3 states for intermediate nodes)
SIGNAL_STATES = ['Bullish', 'Bearish', 'Neutral']


class ConditionalProbabilityTableGenerator:
    """
    Genereert Conditional Probability Tables uit historische data.
    
    Gebruikt kfl.mtf_signals_lead hypertable voor frequency counting
    met Laplace smoothing en batch SQL optimalisaties.
    
    KFL Column Format:
    - rsi_signal_d, macd_signal_d, bb_signal_d, keltner_signal_d, atr_signal_d (Daily)
    - rsi_signal_240, macd_signal_240, bb_signal_240, keltner_signal_240, atr_signal_240 (4H)
    - rsi_signal_60, macd_signal_60, bb_signal_60, keltner_signal_60, atr_signal_60 (1H)
    - rsi_signal_1, macd_signal_1, bb_signal_1, keltner_signal_1, atr_signal_1 (1m)
    """
    
    def __init__(self, laplace_alpha: float = 1.0):
        """
        Initialize CPT Generator
        
        Args:
            laplace_alpha: Laplace smoothing parameter (α = 1 default)
        """
        self.laplace_alpha = laplace_alpha
        self.min_observations = 100
        self._cache = {}
        
        logger.info(f"CPT Generator initialized with α={laplace_alpha}")
    
    # ========================================================================
    # DATABASE PERSISTENCE METHODS
    # ========================================================================
    
    def save_cpt_to_database(self, asset_id: int, node_name: str, cpt_data: Dict[str, Any], 
                            lookback_days: Optional[int] = None,
                            model_version: str = '1.0',
                            cpt_scope: str = 'ASSET') -> bool:
        """
        UPSERT CPT naar qbn.cpt_cache.
        
        Args:
            asset_id: Asset ID
            node_name: Node naam (bijv. 'structural_trend')
            cpt_data: CPT dictionary met probabilities
            lookback_days: Dagen lookback (None = alle data)
            model_version: QBN model versie (default '1.0')
            cpt_scope: Bereik van de training data (default 'ASSET')
            
        Returns:
            True als succesvol opgeslagen
        """
        try:
            observations = cpt_data.get('observations') or cpt_data.get('total_observations', 0)
            version_hash = cpt_data.get('version_hash', self._generate_version_hash(cpt_data))
            
            # Extract validation metadata if present
            validation = cpt_data.get('validation', {})
            state_reduction = cpt_data.get('state_reduction_level', 'FULL')
            coverage = validation.get('coverage')
            sparse_cells = validation.get('sparse_cells')
            
            query = """
            INSERT INTO qbn.cpt_cache (
                asset_id, node_name, cpt_data, lookback_days, observations, 
                generated_at, version_hash, model_version, cpt_scope, 
                state_reduction, coverage, sparse_cells, last_used
            )
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (asset_id, node_name) DO UPDATE SET
                cpt_data = EXCLUDED.cpt_data,
                lookback_days = EXCLUDED.lookback_days,
                observations = EXCLUDED.observations,
                generated_at = NOW(),
                version_hash = EXCLUDED.version_hash,
                model_version = EXCLUDED.model_version,
                cpt_scope = EXCLUDED.cpt_scope,
                state_reduction = EXCLUDED.state_reduction,
                coverage = EXCLUDED.coverage,
                sparse_cells = EXCLUDED.sparse_cells,
                last_used = NOW()
            """
            
            with get_cursor(commit=True) as cur:
                cur.execute(query, (
                    asset_id, 
                    node_name, 
                    json.dumps(cpt_data), 
                    lookback_days,
                    observations,
                    version_hash,
                    model_version,
                    cpt_scope,
                    state_reduction,
                    coverage,
                    sparse_cells
                ))
            
            logger.debug(f"Saved CPT for asset {asset_id}, node {node_name} to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CPT to database: {e}")
            return False
    
    def load_cpt_from_database(self, asset_id: int, node_name: str) -> Optional[Dict[str, Any]]:
        """
        Laad CPT uit qbn.cpt_cache.
        
        Args:
            asset_id: Asset ID
            node_name: Node naam
            
        Returns:
            CPT dictionary of None als niet gevonden
        """
        try:
            query = """
            SELECT cpt_data, lookback_days, observations, generated_at, version_hash
            FROM qbn.cpt_cache
            WHERE asset_id = %s AND node_name = %s
            """
            
            with get_cursor() as cur:
                cur.execute(query, (asset_id, node_name))
                row = cur.fetchone()
            
            if row:
                cpt_data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                cpt_data['_db_lookback_days'] = row[1]
                cpt_data['_db_observations'] = row[2]
                cpt_data['_db_generated_at'] = row[3].isoformat() if row[3] else None
                cpt_data['_db_version_hash'] = row[4]
                logger.debug(f"Loaded CPT for asset {asset_id}, node {node_name} from database")
                return cpt_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load CPT from database: {e}")
            return None
    
    def load_all_cpts_for_asset(self, asset_id: int) -> Dict[str, Dict[str, Any]]:
        """
        Laad alle CPT's voor een asset in 1 query.
        
        Args:
            asset_id: Asset ID
            
        Returns:
            Dict met node_name -> CPT data
        """
        try:
            query = """
            SELECT node_name, cpt_data, lookback_days, observations, generated_at, version_hash
            FROM qbn.cpt_cache
            WHERE asset_id = %s
            """
            
            with get_cursor() as cur:
                cur.execute(query, (asset_id,))
                rows = cur.fetchall()
            
            result = {}
            for row in rows:
                node_name = row[0]
                cpt_data = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                cpt_data['_db_lookback_days'] = row[2]
                cpt_data['_db_observations'] = row[3]
                cpt_data['_db_generated_at'] = row[4].isoformat() if row[4] else None
                cpt_data['_db_version_hash'] = row[5]
                result[node_name] = cpt_data
            
            logger.info(f"Loaded {len(result)} CPTs for asset {asset_id} from database")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load CPTs from database: {e}")
            return {}
    
    def is_cpt_fresh(self, asset_id: int, max_age_hours: int = 24) -> bool:
        """
        Check of CPT's voor asset nog vers zijn.
        
        Args:
            asset_id: Asset ID
            max_age_hours: Maximale leeftijd in uren (default 24)
            
        Returns:
            True als CPT's bestaan en jonger zijn dan max_age_hours
        """
        try:
            query = """
            SELECT MIN(generated_at), COUNT(*)
            FROM qbn.cpt_cache
            WHERE asset_id = %s
            """
            
            with get_cursor() as cur:
                cur.execute(query, (asset_id,))
                row = cur.fetchone()
            
            if not row or row[1] == 0:
                return False
            
            oldest_generated = row[0]
            if oldest_generated is None:
                return False
            
            age = datetime.now(timezone.utc) - oldest_generated
            is_fresh = age.total_seconds() < (max_age_hours * 3600)
            
            logger.debug(f"CPT freshness for asset {asset_id}: {is_fresh} (age: {age})")
            return is_fresh
            
        except Exception as e:
            logger.error(f"Failed to check CPT freshness: {e}")
            return False
    
    def get_cpt_cache_status(self) -> Dict[str, Any]:
        """
        Haal status van CPT cache op voor alle assets.
        
        Returns:
            Dict met cache statistieken
        """
        try:
            query = """
            SELECT 
                asset_id,
                COUNT(*) as node_count,
                MIN(generated_at) as oldest,
                MAX(generated_at) as newest,
                SUM(observations) as total_observations
            FROM qbn.cpt_cache
            GROUP BY asset_id
            ORDER BY asset_id
            """
            
            with get_cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
            
            assets = []
            for row in rows:
                assets.append({
                    'asset_id': row[0],
                    'node_count': row[1],
                    'oldest': row[2].isoformat() if row[2] else None,
                    'newest': row[3].isoformat() if row[3] else None,
                    'total_observations': row[4]
                })
            
            return {
                'total_assets': len(assets),
                'assets': assets
            }
            
        except Exception as e:
            logger.error(f"Failed to get CPT cache status: {e}")
            return {'total_assets': 0, 'assets': [], 'error': str(e)}
    
    def generate_cpt_for_asset(self, 
                              asset_id: int, 
                              node_name: str,
                              parent_nodes: List[str],
                              lookback_days: Optional[int] = None,
                              db_columns: Optional[List[str]] = None,
                              aggregation_method: str = 'majority',
                              target_column: Optional[str] = None,
                              num_states: int = 3) -> Dict[str, Any]:
        """
        Genereer CPT voor specifieke asset en node.
        
        Args:
            asset_id: Asset ID uit symbols table
            node_name: Target node name
            parent_nodes: List van parent node names
            lookback_days: Aantal dagen historische data (None = alle data)
            db_columns: Database kolommen om te gebruiken voor signal aggregation
            aggregation_method: Methode voor signal aggregation ('majority', 'weighted', 'average')
            target_column: Outcome kolom voor prediction nodes (bijv. 'outcome_1h')
            num_states: Aantal states (3 voor signals, 7 voor outcomes)
            
        Returns:
            Dict met CPT data en metadata
        """
        logger.info(f"Generating CPT for asset {asset_id}, node {node_name} (lookback: {lookback_days or 'all'}, target: {target_column or 'signals'})")
        
        cache_key = self._generate_cache_key(asset_id, node_name, parent_nodes, lookback_days, db_columns, aggregation_method)
        
        if cache_key in self._cache:
            logger.debug(f"Using cached CPT for {cache_key}")
            return self._cache[cache_key]
        
        data = self._fetch_historical_data(asset_id, lookback_days)
        
        if data.empty:
            logger.warning(f"No historical data found for asset {asset_id}")
            return self._create_uniform_cpt(node_name, parent_nodes, num_states)
        
        # REASON: Filter op records met ingevulde outcomes voor target nodes
        if target_column and target_column in data.columns:
            valid_data = data[data[target_column].notna()]
            if len(valid_data) == 0:
                logger.warning(f"No valid outcome data for {node_name}, using uniform CPT")
                return self._create_uniform_cpt(node_name, parent_nodes, num_states)
            data = valid_data
            logger.debug(f"Filtered to {len(data)} records with valid {target_column}")
        
        cpt_data = self._generate_cpt_from_data(
            data, node_name, parent_nodes, db_columns, aggregation_method,
            target_column=target_column, num_states=num_states
        )
        cpt_data['lookback_days'] = lookback_days  # REASON: Track lookback voor database opslag
        
        self._cache[cache_key] = cpt_data
        
        return cpt_data
    
    def generate_cpt_batch(self, 
                          asset_ids: List[int],
                          node_name: str, 
                          parent_nodes: List[str],
                          lookback_days: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        Batch generatie van CPTs voor meerdere assets.
        Gebruikt TimescaleDB parallel workers voor performance.
        
        Args:
            asset_ids: Lijst van asset IDs
            node_name: Node naam
            parent_nodes: Parent node namen
            lookback_days: Dagen lookback (None = alle data)
        """
        logger.info(f"Batch generating CPTs for {len(asset_ids)} assets (lookback: {lookback_days or 'all'})")
        
        results = {}
        
        for asset_id in asset_ids:
            try:
                cpt = self.generate_cpt_for_asset(asset_id, node_name, parent_nodes, lookback_days)
                results[asset_id] = cpt
            except Exception as e:
                logger.error(f"Failed to generate CPT for asset {asset_id}: {e}")
                results[asset_id] = self._create_uniform_cpt(node_name, parent_nodes)
        
        return results
    
    def _fetch_historical_data(self, asset_id: int, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical signal data uit kfl.mtf_signals_lead/coin/conf hypertables.
        
        Args:
            asset_id: Asset ID
            lookback_days: Aantal dagen (None = alle beschikbare data)
        
        REASON: Updated query to JOIN lead, coin, and conf tables due to schema split.
        """
        # REASON: lookback_days=None betekent alle beschikbare data gebruiken
        if lookback_days is not None:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=lookback_days)
            time_filter = "AND mtf.time_1 >= %s AND mtf.time_1 <= %s"
            params = (asset_id, start_time, end_time)
        else:
            time_filter = ""
            params = (asset_id,)
        
        # REASON: JOIN lead, coin and conf tables to get all signal components
        # REASON: Use kfl.mtf_signals_lead as primary table, JOIN others on asset_id and time_1
        query = f"""
        SELECT
            mtf.time_1 as time,
            mtf.asset_id,
            -- LEADING signals (lead table)
            mtf.rsi_signal_d, mtf.rsi_signal_240, mtf.rsi_signal_60, mtf.rsi_signal_1,
            mtf.stoch_signal_d, mtf.stoch_signal_240, mtf.stoch_signal_60, mtf.stoch_signal_1,

            -- COIN signals (coin table)
            coin.macd_signal_d, coin.macd_signal_240, coin.macd_signal_60, coin.macd_signal_1,
            coin.bb_signal_d, coin.bb_signal_240, coin.bb_signal_60, coin.bb_signal_1,
            coin.keltner_signal_d, coin.keltner_signal_240, coin.keltner_signal_60, coin.keltner_signal_1,
            coin.atr_signal_d, coin.atr_signal_240, coin.atr_signal_60, coin.atr_signal_1,
            coin.cmf_signal_d, coin.cmf_signal_240, coin.cmf_signal_60, coin.cmf_signal_1,

            -- CONFIRMATION signals (conf table)
            conf.adx_signal_d, conf.adx_signal_240, conf.adx_signal_60, conf.adx_signal_1,

            -- Concordance metrics
            mtf.concordance_score,

            -- Outcome targets from separate table
            o.outcome_1h, o.outcome_4h, o.outcome_1d,
            o.return_1h_pct, o.return_4h_pct, o.return_1d_pct,
            COALESCE(o.atr_at_signal, mtf.atr_at_signal) as atr_at_signal
        FROM kfl.mtf_signals_lead mtf
        LEFT JOIN kfl.mtf_signals_coin coin 
            ON coin.asset_id = mtf.asset_id AND coin.time_1 = mtf.time_1
        LEFT JOIN kfl.mtf_signals_conf conf
            ON conf.asset_id = mtf.asset_id AND conf.time_1 = mtf.time_1
        LEFT JOIN qbn.signal_outcomes o
            ON o.asset_id = mtf.asset_id
            AND o.time_1 = mtf.time_1
        WHERE mtf.asset_id = %s
          {time_filter}
        ORDER BY mtf.time_1 ASC
        """

        
        try:
            with get_cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                columns = [desc[0] for desc in cur.description]
                
                df = pd.DataFrame(rows, columns=columns)
                
                logger.debug(f"Fetched {len(df)} historical records for asset {asset_id} (lookback: {lookback_days or 'all'})")
                return df
                
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def _generate_cpt_from_data(self, 
                               data: pd.DataFrame, 
                               node_name: str, 
                               parent_nodes: List[str],
                               db_columns: Optional[List[str]] = None,
                               aggregation_method: str = 'majority',
                               target_column: Optional[str] = None,
                               num_states: int = 3) -> Dict[str, Any]:
        """
        Genereer CPT uit DataFrame met frequency counting en Laplace smoothing.
        
        Args:
            data: DataFrame met historische data
            node_name: Naam van de node
            parent_nodes: Lijst van parent node namen
            db_columns: Database kolommen voor signal aggregatie
            aggregation_method: Aggregatie methode ('majority', 'weighted', 'average')
            target_column: Outcome kolom voor prediction nodes (bijv. 'outcome_1h')
            num_states: Aantal states (3 voor signals, 7 voor outcomes)
        """
        # REASON: Gebruik 7 states voor outcome nodes, 3 voor signal nodes
        if target_column or num_states == 7:
            states = OUTCOME_STATE_LIST  # 7 states
        else:
            states = SIGNAL_STATES  # 3 states
        
        if not parent_nodes:
            # Root node - calculate prior probabilities
            if target_column and target_column in data.columns:
                # REASON: Voor outcome nodes, gebruik echte koersuitkomsten
                valid_data = data[data[target_column].notna()]
                if len(valid_data) == 0:
                    logger.warning(f"No valid outcome data for {node_name}")
                    return self._create_uniform_cpt(node_name, parent_nodes, num_states)
                
                # Map integer outcomes to state names
                outcome_states = valid_data[target_column].apply(
                    lambda x: OUTCOME_STATES.get(int(x), 'Neutral') if pd.notna(x) else 'Neutral'
                )
                counts = outcome_states.value_counts()
                total = len(valid_data)
            elif db_columns and aggregation_method == 'majority':
                aggregated_signals = self._aggregate_signals(data, db_columns, aggregation_method)
                counts = aggregated_signals.value_counts()
                total = len(data)
            else:
                return self._create_uniform_cpt(node_name, parent_nodes, num_states)
            
            # Apply Laplace smoothing
            probabilities = {}
            for state in states:
                count = counts.get(state, 0)
                prob = (count + self.laplace_alpha) / (total + self.laplace_alpha * len(states))
                probabilities[state] = prob
            
            cpt_data = {
                'node': node_name,
                'parents': [],
                'states': states,
                'probabilities': probabilities,
                'type': 'prior',
                'observations': total,
                'db_columns_used': db_columns,
                'target_column': target_column,
                'aggregation_method': aggregation_method,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        else:
            # Child node - calculate conditional probabilities
            parent_combinations = {}
            
            for _, row in data.iterrows():
                parent_state = tuple(row[parent] for parent in parent_nodes if parent in row)
                
                if len(parent_state) != len(parent_nodes):
                    continue
                
                # REASON: Voor target nodes, gebruik echte outcome als child state
                if target_column and target_column in row and pd.notna(row[target_column]):
                    child_state = OUTCOME_STATES.get(int(row[target_column]), 'Neutral')
                elif db_columns and aggregation_method == 'majority':
                    child_state = self._aggregate_row_signals(row, db_columns, aggregation_method)
                else:
                    child_state = parent_state[0] if parent_state else 'Neutral'
                
                if parent_state not in parent_combinations:
                    parent_combinations[parent_state] = Counter()
                
                parent_combinations[parent_state][child_state] += 1
            
            # Calculate conditional probabilities with Laplace smoothing
            conditional_probs = {}
            
            for parent_combo, child_counts in parent_combinations.items():
                total_for_combo = sum(child_counts.values())
                
                combo_probs = {}
                for state in states:
                    count = child_counts.get(state, 0)
                    prob = (count + self.laplace_alpha) / (total_for_combo + self.laplace_alpha * len(states))
                    combo_probs[state] = prob
                
                conditional_probs[parent_combo] = combo_probs
            
            cpt_data = {
                'node': node_name,
                'parents': parent_nodes,
                'states': states,
                'conditional_probabilities': conditional_probs,
                'type': 'conditional',
                'parent_combinations': len(parent_combinations),
                'total_observations': len(data),
                'db_columns_used': db_columns,
                'target_column': target_column,
                'aggregation_method': aggregation_method,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
        
        cpt_data['version_hash'] = self._generate_version_hash(cpt_data)
        
        return cpt_data
    
    def _aggregate_signals(self, data: pd.DataFrame, db_columns: List[str], method: str) -> pd.Series:
        """
        Aggregate signals uit meerdere database kolommen.
        
        Args:
            data: DataFrame met signal data
            db_columns: Kolommen om te aggregaten
            method: Aggregation methode ('majority', 'weighted', 'average')
            
        Returns:
            Series met geaggregeerde signal states
        """
        if method == 'majority':
            aggregated = []
            for _, row in data.iterrows():
                signals = [row[col] for col in db_columns if col in row and pd.notna(row[col])]
                if signals:
                    signal_counts = Counter(signals)
                    most_common = signal_counts.most_common(1)[0][0]
                    # Map integer to state string
                    most_common = SignalState.to_string(most_common) if isinstance(most_common, int) else most_common
                    aggregated.append(most_common)
                else:
                    aggregated.append('Neutral')
            
            return pd.Series(aggregated, index=data.index)
        
        elif method == 'weighted':
            return self._aggregate_signals(data, db_columns, 'majority')
        
        elif method == 'average':
            return self._aggregate_signals(data, db_columns, 'majority')
        
        else:
            return self._aggregate_signals(data, db_columns, 'majority')
    
    def _aggregate_row_signals(self, row: pd.Series, db_columns: List[str], method: str) -> str:
        """
        Aggregate signals uit één rij voor meerdere kolommen.
        """
        signals = [row[col] for col in db_columns if col in row and pd.notna(row[col])]
        if signals:
            signal_counts = Counter(signals)
            most_common = signal_counts.most_common(1)[0][0]
            # Map integer to state string
            return SignalState.to_string(most_common) if isinstance(most_common, int) else most_common
        return 'Neutral'
    
    def _create_uniform_cpt(self, node_name: str, parent_nodes: List[str], 
                            num_states: int = 3) -> Dict[str, Any]:
        """
        Creëer uniforme CPT als fallback.
        
        Args:
            node_name: Node naam
            parent_nodes: Parent node namen
            num_states: Aantal states (3 voor signals, 7 voor outcomes)
        """
        if num_states == 7:
            states = OUTCOME_STATE_LIST
        else:
            states = SIGNAL_STATES
            
        uniform_prob = 1.0 / len(states)
        
        if not parent_nodes:
            probabilities = {state: uniform_prob for state in states}
            
            cpt_data = {
                'node': node_name,
                'parents': [],
                'states': states,
                'probabilities': probabilities,
                'type': 'uniform_prior',
                'observations': 0,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            cpt_data = {
                'node': node_name,
                'parents': parent_nodes,
                'states': states,
                'uniform_probability': uniform_prob,
                'type': 'uniform_conditional',
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
        
        cpt_data['version_hash'] = self._generate_version_hash(cpt_data)
        return cpt_data
    
    def _generate_cache_key(self, asset_id: int, node_name: str, parent_nodes: List[str], lookback_days: Optional[int], db_columns: Optional[List[str]], aggregation_method: str) -> str:
        """Generate cache key voor CPT"""
        lookback_str = str(lookback_days) if lookback_days is not None else "all"
        key_data = f"{asset_id}_{node_name}_{'_'.join(sorted(parent_nodes))}_{lookback_str}"
        if db_columns:
            key_data += f"_{'_'.join(sorted(db_columns))}"
        key_data += f"_{aggregation_method}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_version_hash(self, cpt_data: Dict[str, Any]) -> str:
        """Genereer version hash voor CPT data."""
        hash_data = {
            'node': cpt_data['node'],
            'parents': cpt_data.get('parents', []),
            'type': cpt_data['type'],
            'generated_at': cpt_data['generated_at']
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def validate_cpt(self, cpt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valideer CPT data op consistency."""
        validation = {
            'is_valid': True,
            'issues': []
        }
        
        try:
            if cpt_data['type'] == 'prior':
                total_prob = sum(cpt_data['probabilities'].values())
                if abs(total_prob - 1.0) > 1e-6:
                    validation['issues'].append(f"Prior probabilities sum to {total_prob}, not 1.0")
                    validation['is_valid'] = False
            
            elif cpt_data['type'] == 'conditional':
                for parent_combo, probs in cpt_data['conditional_probabilities'].items():
                    total_prob = sum(probs.values())
                    if abs(total_prob - 1.0) > 1e-6:
                        validation['issues'].append(f"Conditional prob for {parent_combo} sums to {total_prob}")
                        validation['is_valid'] = False
            
        except Exception as e:
            validation['issues'].append(f"Validation error: {e}")
            validation['is_valid'] = False
        
        return validation
    
    def clear_cache(self):
        """Clear CPT cache"""
        self._cache.clear()
        logger.info("CPT cache cleared")
    
    # ========================================================================
    # OUTCOME UTILITY METHODS
    # ========================================================================
    
    @staticmethod
    def outcome_id_to_name(state_id: int) -> str:
        """Convert outcome state ID naar naam."""
        return OUTCOME_STATES.get(state_id, 'Neutral')
    
    @staticmethod
    def outcome_name_to_id(state_name: str) -> int:
        """Convert outcome state naam naar ID."""
        for state_id, name in OUTCOME_STATES.items():
            if name == state_name:
                return state_id
        return 0
    
    @staticmethod
    def calculate_expected_atr_move(probabilities: Dict[str, float]) -> float:
        """
        Bereken verwachte ATR beweging uit probability distribution.
        
        Args:
            probabilities: Dict van state_name -> probability
            
        Returns:
            Expected ATR move (gewogen gemiddelde)
        """
        expected = 0.0
        for state_name, prob in probabilities.items():
            state_id = ConditionalProbabilityTableGenerator.outcome_name_to_id(state_name)
            midpoint = OUTCOME_ATR_MIDPOINTS.get(state_id, 0.0)
            expected += prob * midpoint
        return expected
    
    @staticmethod
    def get_outcome_state_list() -> List[str]:
        """Return list van outcome state namen."""
        return OUTCOME_STATE_LIST.copy()
    
    @staticmethod
    def get_signal_state_list() -> List[str]:
        """Return list van signal state namen."""
        return SIGNAL_STATES.copy()


def create_cpt_generator(laplace_alpha: float = 1.0) -> ConditionalProbabilityTableGenerator:
    """Factory function voor CPT Generator"""
    return ConditionalProbabilityTableGenerator(laplace_alpha)

