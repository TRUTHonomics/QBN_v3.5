"""
Inference Loader voor QBN v3.4.
Laadt CPTs, signal classificaties en huidige evidence uit de database.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import re

from database.db import get_cursor
from config.network_config import MODEL_VERSION
from config.threshold_loader import ThresholdLoader
from .trade_aligned_inference import TradeAlignedInference, SignalEvidence
from .cpt_generator import SIGNAL_STATES, OUTCOME_STATE_LIST

logger = logging.getLogger(__name__)

class InferenceLoader:
    """Laadt inference componenten en evidence uit de database."""
    
    def __init__(self):
        self.model_version = MODEL_VERSION
        self._classification_cache = {}  # Cache per (asset_id, filter_suffix)

    def load_inference_engine(self, asset_id: int, horizon: str = '1h', run_id: Optional[str] = None) -> TradeAlignedInference:
        """
        Laad een complete inference engine voor een specifiek asset.
        """
        cpts = self.load_all_cpts_for_asset(asset_id, run_id=run_id)
        
        # v3.1: Load Position_Prediction CPT apart
        position_pred_cpt = None
        v33_cpts = {}
        
        if self.model_version >= '3.1':
            from .cpt_cache_manager import CPTCacheManager
            cache = CPTCacheManager()
            
            position_pred_cpt = cache.get_cpt(
                scope_key=f"asset_{asset_id}",
                node_name='Position_Prediction',
                model_version=self.model_version,
                run_id=run_id
            )
            
            # v3.4: Load Sub-Prediction CPTs
            for node_name in ['Momentum_Prediction', 'Volatility_Regime', 'Exit_Timing']:
                cpt = cache.get_cpt(
                    scope_key=f"asset_{asset_id}",
                    node_name=node_name,
                    model_version=self.model_version,
                    run_id=run_id
                )
                if cpt:
                    v33_cpts[node_name] = cpt
                    logger.debug(f"Loaded v3.4 CPT: {node_name} for asset {asset_id}")
            
            if v33_cpts:
                logger.info(f"Loaded {len(v33_cpts)} position CPTs for asset {asset_id}")
            
        signal_classification = self.load_signal_classification(asset_id, use_cache=True, filter_suffix='60')
        threshold_loader = ThresholdLoader(asset_id, horizon)
        
        return TradeAlignedInference(
            cpts, 
            signal_classification, 
            threshold_loader=threshold_loader,
            position_prediction_cpt=position_pred_cpt,
            v33_cpts=v33_cpts
        )

    def load_all_cpts_for_asset(self, asset_id: int, run_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Haal alle CPTs voor een asset op met cascade lookup.

        Lookup volgorde:
        1. Asset-specifieke CPT (scope_key = 'asset_{id}')
        2. Preferred scope uit qbn.asset_cpt_mapping
        3. Fallback scope (default: 'global')
        4. Uniform defaults als geen CPT gevonden
        """
        from .cpt_cache_manager import CPTCacheManager

        cache = CPTCacheManager()
        cpts, source_scope = cache.get_cpts_for_asset_cascade(asset_id, run_id=run_id)

        if cpts:
            logger.info(f"Loaded {len(cpts)} CPTs for asset {asset_id} from scope '{source_scope}'")
            return cpts

        # Fallback naar oude directe lookup (backwards compatibility)
        query = """
        SELECT node_name, cpt_data
        FROM qbn.cpt_cache
        WHERE asset_id = %s AND model_version = %s
        """
        params = [asset_id, self.model_version]
        if run_id:
            query += " AND run_id = %s"
            params.append(run_id)

        cpts = {}
        with get_cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

            for node_name, cpt_data in rows:
                if isinstance(cpt_data, str):
                    cpts[node_name] = json.loads(cpt_data)
                else:
                    cpts[node_name] = cpt_data

        if not cpts:
            logger.warning(f"Geen CPTs gevonden voor asset {asset_id}, model {self.model_version} - gebruik uniforme defaults")
            cpts = self._create_default_cpts()

        return cpts

    def load_signal_classification(self, asset_id: int, use_cache: bool = True, filter_suffix: Optional[str] = None) -> Dict[str, Dict]:
        """
        Laad signal classificatie inclusief gewichten (alpha-weights).
        
        REASON: In v3.1 moeten ALLE geclassificeerde signalen geladen worden,
        ook als ze (nog) geen gewichten hebben (zoals Coincident/Confirming).
        
        Naming conventions in database:
        - signal_classification: UPPERCASE, geen suffix (bijv. 'RSI_OVERSOLD')
        - signal_weights: lowercase, met suffix (bijv. 'rsi_oversold_60')
        - kfl.mtf_signals_*: lowercase, met suffix (bijv. 'rsi_oversold_60')
        
        Args:
            asset_id: De asset ID
            use_cache: Of de cache gebruikt moet worden
            filter_suffix: Optioneel suffix om op te filteren (bijv. '60'). 
                          Indien meegegeven, worden alleen signalen geladen die eindigen op deze suffix.
        """
        cache_key = (asset_id, filter_suffix)
        if use_cache and cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        # STAP 1: Haal ALLE classificaties op (dit is de bron van waarheid)
        class_query = """
        SELECT signal_name, semantic_class, polarity
        FROM qbn.signal_classification
        """
        
        # STAP 2: Haal weights op voor dit asset
        weights_query = """
        SELECT DISTINCT ON (signal_name, horizon)
            signal_name, 
            horizon, 
            weight
        FROM qbn.signal_weights
        WHERE (asset_id = %s OR asset_id = 0)
        ORDER BY signal_name, horizon, asset_id DESC, last_trained_at DESC
        """
        
        classification = {}
        
        with get_cursor() as cur:
            # 1. Laad alle basis-classificaties
            cur.execute(class_query)
            base_signals = cur.fetchall()
            
            # 2. Laad alle beschikbare gewichten
            cur.execute(weights_query, (asset_id,))
            weights_data = {} # (full_name, horizon) -> weight
            for full_name, horizon, weight in cur.fetchall():
                weights_data[(full_name.lower(), horizon)] = float(weight or 1.0)

            # 3. Combineer: Voor elk basis-signaal maken we de volledige naam met de gewenste suffix
            # en koppelen we de gewichten als die er zijn.
            suffix = filter_suffix if filter_suffix else '60'
            
            for base_name, sem_class, polarity in base_signals:
                # Genereer de volledige naam (lowercase + suffix)
                # bijv. RSI_OVERSOLD -> rsi_oversold_60
                full_name = f"{base_name.lower()}_{suffix}"
                
                classification[full_name] = {
                    'semantic_class': sem_class,
                    'polarity': polarity,
                    'weights': {
                        '1h': weights_data.get((full_name, '1h'), 1.0),
                        '4h': weights_data.get((full_name, '4h'), 1.0),
                        '1d': weights_data.get((full_name, '1d'), 1.0)
                    }
                }
        
        self._classification_cache[cache_key] = classification
        logger.info(f"Loaded {len(classification)} signal classifications (suffix={suffix}) for asset {asset_id}")
        return classification

    def load_current_evidence(self, asset_id: int) -> SignalEvidence:
        """
        Haal de meest recente signalen en indicators op voor live inference.
        """
        # 1. Fetch live signals van MTF current tables
        # We joinen lead, coin en conf op asset_id
        signal_query = """
        SELECT 
            lead.time_1,
            lead.*, 
            coin.*, 
            conf.*
        FROM kfl.mtf_signals_current_lead lead
        LEFT JOIN kfl.mtf_signals_current_coin coin ON coin.asset_id = lead.asset_id
        LEFT JOIN kfl.mtf_signals_current_conf conf ON conf.asset_id = lead.asset_id
        WHERE lead.asset_id = %s
        """
        
        # 2. Fetch raw regime indicators van indicators_unified_cache
        # We hebben Daily (1440) en 4H (240) nodig
        regime_query = """
        SELECT 
            interval_min, adx_14, dm_plus_14, dm_minus_14, macd_12_26_9_histogram
        FROM kfl.indicators_unified_cache
        WHERE asset_id = %s AND interval_min IN ('1440', '240')
        ORDER BY time DESC
        LIMIT 2
        """
        
        evidence = None
        with get_cursor() as cur:
            # Signals
            cur.execute(signal_query, (asset_id,))
            sig_row = cur.fetchone()
            if not sig_row:
                raise ValueError(f"Geen live signals gevonden voor asset {asset_id}")
                
            cols = [desc[0] for desc in cur.description]
            sig_data = dict(zip(cols, sig_row))
            
            evidence = SignalEvidence(
                asset_id=asset_id,
                timestamp=sig_data['time_1']
            )
            
            # Sorteer signalen in de juiste semantische klasse
            classification = self.load_signal_classification(asset_id, use_cache=True)
            
            for full_sig_name, info in classification.items():
                sem_class = info['semantic_class']
                
                # REASON: classification bevat nu de volledige kolomnamen (met suffix)
                # EXPL: We checken direct of deze naam in de opgehaalde sig_data zit.
                if full_sig_name in sig_data and sig_data[full_sig_name] is not None:
                    val = int(sig_data[full_sig_name])
                    if sem_class == 'LEADING':
                        evidence.leading_signals[full_sig_name] = val
                    elif sem_class == 'COINCIDENT':
                        evidence.coincident_signals[full_sig_name] = val
                    elif sem_class == 'CONFIRMING':
                        evidence.confirming_signals[full_sig_name] = val
            
            # REASON: De _compute_regime methode haalt nu adx_signal_d/240 
            # direct uit de evidence signal dicts die we hierboven gevuld hebben.
            # Er is geen aparte indicators_unified_cache query meer nodig voor Regime.
            
        return evidence

    def load_rolling_evidence(self, asset_id: int) -> SignalEvidence:
        """
        Haal de meest recente rolling signals op voor live inference.
        Leest uit kfl.rolling_signals_current.
        """
        query = """
        SELECT * FROM kfl.rolling_signals_current
        WHERE asset_id = %s
        """
        
        evidence = None
        with get_cursor() as cur:
            cur.execute(query, (asset_id,))
            row = cur.fetchone()
            if not row:
                logger.warning(f"Geen rolling signals gevonden voor asset {asset_id}")
                return self.load_current_evidence(asset_id)  # Fallback naar gewone live data
                
            cols = [desc[0] for desc in cur.description]
            data = dict(zip(cols, row))
            
            evidence = SignalEvidence(
                asset_id=asset_id,
                timestamp=data['time_1'],
                rolling_60m_completeness=data.get('rolling_60m_completeness', 1.0)
            )
            
            # Sorteer signalen in de juiste semantische klasse
            classification = self.load_signal_classification(asset_id, use_cache=True)
            
            for full_sig_name, info in classification.items():
                sem_class = info['semantic_class']
                
                # Strip suffix voor lookup in rolling data (omdat rolling tabel geen suffixes heeft per timeframe)
                # MAAR: de rolling tabel heeft nu kolommen ZONDER suffix zoals rsi_oversold
                # De classification heeft namen MET suffix zoals rsi_oversold_60
                # We moeten de match maken.
                
                base_sig_name = re.sub(r'(_d|_240|_60|_1)$', '', full_sig_name.lower())
                
                if base_sig_name in data and data[base_sig_name] is not None:
                    val = int(data[base_sig_name])
                    if sem_class == 'LEADING':
                        evidence.leading_signals[full_sig_name] = val
                    elif sem_class == 'COINCIDENT':
                        evidence.coincident_signals[full_sig_name] = val
                    elif sem_class == 'CONFIRMING':
                        evidence.confirming_signals[full_sig_name] = val
                        
        return evidence

    def _create_default_cpts(self) -> Dict[str, Dict[str, Any]]:
        """Uniforme fallbacks voor ontbrekende CPTs."""
        return {
            'HTF_Regime': {'probabilities': {'bullish_trend': 0.33, 'ranging': 0.34, 'bearish_trend': 0.33}},
            'Entry_Timing': {'probabilities': {'poor': 0.25, 'neutral': 0.25, 'good': 0.25, 'excellent': 0.25}},
            'Prediction_1h': {'conditional_probabilities': {}}, # Wordt uniform in engine
            'Prediction_4h': {'conditional_probabilities': {}},
            'Prediction_1d': {'conditional_probabilities': {}}
        }

