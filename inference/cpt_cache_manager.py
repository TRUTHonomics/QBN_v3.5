"""
CPT Cache Manager for QBN v3.
Manages CPT persistence, freshness tracking, and retrieval.

Supports:
- Single-asset CPTs (scope_type='single')
- Composite CPTs (scope_type='composite', e.g. 'top_10')
- Global CPTs (scope_type='global', all assets)
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
import json
import numpy as np

from database.db import get_cursor
from config.network_config import FRESHNESS_THRESHOLD_HOURS, MODEL_VERSION

logger = logging.getLogger(__name__)

class CPTCacheManager:
    """
    Manages CPT caching with freshness tracking and multi-asset support.

    Cache strategy:
    - Store CPT's in qbn.cpt_cache table
    - Track created_at and last_used timestamps
    - Detect stale CPT's (>24h old)
    - Support composite CPTs via scope_key and source_assets
    """

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self._asset_mapping_cache: Dict[int, Dict[str, str]] = {}

    def get_cpt(
        self,
        asset_id: Optional[int] = None,
        node_name: str = '',
        max_age_hours: int = FRESHNESS_THRESHOLD_HOURS,
        outcome_mode: str = 'barrier',
        scope_key: Optional[str] = None,
        model_version: str = MODEL_VERSION,
        run_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Haal CPT uit cache indien fresh genoeg.
        """
        # REASON: Voor structurele nodes gebruiken we 'not_applicable'
        if node_name not in ['Entry_Confidence', 'Prediction_1h', 'Prediction_4h', 'Prediction_1d', 'Position_Prediction']:
            search_mode = 'not_applicable'
        else:
            search_mode = outcome_mode

        # Bepaal zoekcriteria: scope_key heeft voorrang, anders asset_id
        if scope_key:
            where_clause = "WHERE scope_key = %s"
            params = [scope_key]
        else:
            where_clause = "WHERE asset_id = %s"
            params = [asset_id]

        query = f"""
        SELECT cpt_data, generated_at, last_used, run_id
        FROM qbn.cpt_cache
        {where_clause}
          AND node_name = %s
          AND model_version = %s
          AND outcome_mode = %s
        """

        # REASON: Validation kan een specifieke training-run willen evalueren.
        # EXPL: Indien run_id is gezet, filteren we hard op die run.
        if run_id:
            query += "\n  AND run_id = %s"

        query += """
        ORDER BY generated_at DESC
        LIMIT 1
        """
        params.extend([node_name, model_version, search_mode])
        if run_id:
            params.append(run_id)

        with get_cursor() as cur:
            cur.execute(query, tuple(params))
            row = cur.fetchone()

        if not row:
            self.cache_misses += 1
            return None

        cpt_data, generated_at, last_used, run_id = row
        
        # Ensure generated_at is timezone-aware
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)

        # Check freshness
        age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600

        if age_hours > max_age_hours:
            logger.info(f"Stale CPT for {node_name} ({scope_key or f'asset {asset_id}'}): {age_hours:.1f}h old (Run: {run_id})")
            self.cache_misses += 1
            return None

        # Update metadata in cpt_data for traceability
        if isinstance(cpt_data, dict):
            cpt_data['run_id'] = run_id
            cpt_data['generated_at'] = generated_at.isoformat()

        # Update last_used timestamp
        self._update_last_used(asset_id or 0, node_name, run_id=run_id)

        self.cache_hits += 1
        logger.debug(f"Cache HIT: {node_name} ({scope_key or f'asset {asset_id}'}), run={run_id}, age={age_hours:.1f}h")

        return cpt_data

    def save_cpt(
        self,
        asset_id: int,
        node_name: str,
        cpt_data: Dict[str, Any],
        cpt_scope: str = 'ASSET',
        scope_type: str = 'single',
        scope_key: Optional[str] = None,
        source_assets: Optional[List[int]] = None,
        outcome_mode: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """
        Sla CPT op in cache.

        Args:
            asset_id: Asset ID (0 voor composite/global CPTs)
            node_name: Naam van de CPT node
            cpt_data: CPT data dictionary
            cpt_scope: Legacy scope parameter (backwards compatibility)
            scope_type: Type CPT: 'single', 'composite', 'global'
            scope_key: Unieke identifier (bijv. 'asset_1', 'top_10', 'all_assets')
            source_assets: Lijst van asset IDs gebruikt voor training
            outcome_mode: Mode van outcomes (bijv. 'barrier'). 
                         Indien None, wordt gecontroleerd of de node deze nodig heeft.
            run_id: Run identifier for traceability
        """
        # REASON: Bepaal of deze node uitkomst-afhankelijk is
        outcome_dependent_nodes = ['Entry_Confidence', 'Prediction_1h', 'Prediction_4h', 'Prediction_1d', 'Position_Prediction']
        if node_name not in outcome_dependent_nodes:
            save_mode = 'not_applicable'
        else:
            save_mode = outcome_mode or cpt_data.get('outcome_mode', 'barrier')

        # Auto-generate run_id if missing to avoid NOT NULL constraint violation
        if run_id is None:
            run_id = cpt_data.get('run_id') or datetime.now().strftime("%Y%m%d-%H%M%S")

        # Auto-generate scope_key voor single-asset CPTs
        if scope_key is None:
            scope_key = f'asset_{asset_id}' if asset_id > 0 else 'unknown'
        if source_assets is None:
            source_assets = [asset_id] if asset_id > 0 else []

        # Extract validation metadata from cpt_data if present
        validation = cpt_data.get('validation', {})
        state_reduction = cpt_data.get('state_reduction_level', 'FULL')
        coverage = validation.get('coverage')
        sparse_cells = validation.get('sparse_cells')
        observations = cpt_data.get('observations', 0)

        # Validatie metrieken (cast naar native float voor psycopg2 compatibiliteit)
        entropy = float(validation.get('entropy')) if validation.get('entropy') is not None else None
        info_gain = float(validation.get('info_gain')) if validation.get('info_gain') is not None else None
        stability_score = float(validation.get('stability_score')) if validation.get('stability_score') is not None else None
        semantic_score = float(validation.get('semantic_score')) if validation.get('semantic_score') is not None else None

        query = """
        INSERT INTO qbn.cpt_cache (
            asset_id, node_name, model_version, cpt_scope, cpt_data,
            generated_at, last_used, state_reduction, coverage, sparse_cells, observations,
            entropy, info_gain, stability_score, semantic_score,
            scope_type, scope_key, source_assets, outcome_mode, run_id
        )
        VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scope_key, node_name, model_version, outcome_mode, run_id)
        DO UPDATE SET
            asset_id = EXCLUDED.asset_id,
            cpt_data = EXCLUDED.cpt_data,
            cpt_scope = EXCLUDED.cpt_scope,
            generated_at = NOW(),
            last_used = NOW(),
            state_reduction = EXCLUDED.state_reduction,
            coverage = EXCLUDED.coverage,
            sparse_cells = EXCLUDED.sparse_cells,
            observations = EXCLUDED.observations,
            entropy = EXCLUDED.entropy,
            info_gain = EXCLUDED.info_gain,
            stability_score = EXCLUDED.stability_score,
            semantic_score = EXCLUDED.semantic_score,
            scope_type = EXCLUDED.scope_type,
            source_assets = EXCLUDED.source_assets
        """

        with get_cursor(commit=True) as cur:
            cur.execute(query, (
                asset_id, node_name, MODEL_VERSION, cpt_scope,
                json.dumps(cpt_data, default=self._json_serial),
                state_reduction, coverage, sparse_cells, observations,
                entropy, info_gain, stability_score, semantic_score,
                scope_type, scope_key, source_assets, save_mode, run_id
            ))
            
            # Retentie: bewaar 3 meest recente runs (alleen voor asset-specifieke CPTs)
            if run_id and asset_id > 0:
                from core.run_retention import retain_recent_runs_auto
                retain_recent_runs_auto(cur.connection, "qbn.cpt_cache", asset_id)

        # REASON: Vermeld mode alleen indien relevant (conform gebruikerswens)
        mode_str = f", mode={save_mode}" if save_mode != 'not_applicable' else ""
        logger.info(f"Saved CPT to cache: {node_name} (scope_key={scope_key}, type={scope_type}{mode_str})")

    def save_composite_cpt(
        self,
        scope_key: str,
        node_name: str,
        cpt_data: Dict[str, Any],
        source_assets: List[int],
        scope_type: str = 'composite',
        outcome_mode: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """
        Sla composite CPT op met asset_id=0.

        Args:
            scope_key: Unieke identifier (bijv. 'top_10', 'all_assets')
            node_name: Naam van de CPT node
            cpt_data: CPT data dictionary
            source_assets: Lijst van asset IDs gebruikt voor training
            scope_type: 'composite' of 'global'
            outcome_mode: Mode van outcomes
            run_id: Run identifier for traceability
        """
        self.save_cpt(
            asset_id=0,
            node_name=node_name,
            cpt_data=cpt_data,
            cpt_scope=scope_type.upper(),
            scope_type=scope_type,
            scope_key=scope_key,
            source_assets=source_assets,
            outcome_mode=outcome_mode,
            run_id=run_id
        )

    @staticmethod
    def _json_serial(obj):
        """JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if hasattr(obj, 'item'):  # NumPy scalars
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")

    def _update_last_used(self, asset_id: int, node_name: str, run_id: Optional[str] = None):
        """Update last_used timestamp."""
        query = """
        UPDATE qbn.cpt_cache
        SET last_used = NOW()
        WHERE asset_id = %s AND node_name = %s AND model_version = %s
        """
        params = [asset_id, node_name, MODEL_VERSION]
        
        if run_id:
            query += " AND run_id = %s"
            params.append(run_id)

        with get_cursor(commit=True) as cur:
            cur.execute(query, tuple(params))

    def get_stale_cpts(self, age_threshold_hours: int = FRESHNESS_THRESHOLD_HOURS) -> List[Tuple[int, str]]:
        """
        Vind alle stale CPT's (>threshold hours old).
        """
        query = """
        SELECT asset_id, node_name
        FROM qbn.cpt_cache
        WHERE model_version = %s
          AND generated_at < NOW() - INTERVAL '%s hours'
        ORDER BY generated_at ASC
        """

        with get_cursor() as cur:
            cur.execute(query, (MODEL_VERSION, age_threshold_hours))
            rows = cur.fetchall()

        return [(row[0], row[1]) for row in rows]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate
        }

    # =========================================================================
    # MULTI-ASSET / COMPOSITE CPT SUPPORT
    # =========================================================================

    def get_cpts_by_scope(
        self,
        scope_key: str,
        max_age_hours: int = FRESHNESS_THRESHOLD_HOURS,
        run_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Haal alle CPTs op voor een specifieke scope.
        REASON: Pakt automatisch de meest recente run_id voor deze scope.

        Args:
            scope_key: Scope identifier (bijv. 'asset_1', 'top_10', 'all_assets')
            max_age_hours: Maximum leeftijd in uren

        Returns:
            Dict met node_name -> cpt_data mapping
        """
        if run_id:
            # REASON: Forceer scope load voor een specifieke training-run (validation alignment).
            query = """
            SELECT node_name, cpt_data, generated_at, outcome_mode, run_id
            FROM qbn.cpt_cache
            WHERE scope_key = %s
              AND model_version = %s
              AND run_id = %s
              AND generated_at > NOW() - INTERVAL '%s hours'
              AND (outcome_mode = 'barrier' OR outcome_mode = 'not_applicable')
            ORDER BY node_name
            """
            params = [scope_key, MODEL_VERSION, run_id, max_age_hours]
        else:
            # REASON: Haal zowel 'barrier' als 'not_applicable' op voor de LAATSTE run
            query = """
            WITH latest_run AS (
                SELECT run_id
                FROM qbn.cpt_cache
                WHERE scope_key = %s
                  AND model_version = %s
                  AND generated_at > NOW() - INTERVAL '%s hours'
                ORDER BY generated_at DESC
                LIMIT 1
            )
            SELECT node_name, cpt_data, generated_at, outcome_mode, run_id
            FROM qbn.cpt_cache
            WHERE scope_key = %s
              AND model_version = %s
              AND run_id = (SELECT run_id FROM latest_run)
              AND (outcome_mode = 'barrier' OR outcome_mode = 'not_applicable')
            ORDER BY node_name
            """
            params = [scope_key, MODEL_VERSION, max_age_hours, scope_key, MODEL_VERSION]

        cpts = {}
        with get_cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

            if not rows:
                self.cache_misses += 1
                logger.debug(f"Cache MISS: no CPTs for scope {scope_key}")
                return {}

            run_id = rows[0][4]
            for node_name, cpt_data, generated_at, mode, r_id in rows:
                if isinstance(cpt_data, str):
                    data = json.loads(cpt_data)
                else:
                    data = cpt_data
                
                # Zorg dat metadata in de data zit voor de engine
                if isinstance(data, dict):
                    if mode != 'not_applicable':
                        data['outcome_mode'] = mode
                    data['run_id'] = r_id
                
                cpts[node_name] = data

        if cpts:
            self.cache_hits += len(cpts)
            logger.info(f"Loaded {len(cpts)} CPTs for scope '{scope_key}' (Run: {run_id})")
        
        return cpts

    def get_asset_mapping(self, asset_id: int) -> Optional[Dict[str, str]]:
        """
        Haal de CPT scope mapping op voor een asset.

        Returns:
            Dict met 'preferred_scope' en 'fallback_scope', of None indien niet geconfigureerd.
        """
        if asset_id in self._asset_mapping_cache:
            return self._asset_mapping_cache[asset_id]

        query = """
        SELECT preferred_scope, fallback_scope
        FROM qbn.asset_cpt_mapping
        WHERE asset_id = %s
        """

        with get_cursor() as cur:
            cur.execute(query, (asset_id,))
            row = cur.fetchone()

        if row:
            mapping = {
                'preferred_scope': row[0],
                'fallback_scope': row[1] or 'global'
            }
            self._asset_mapping_cache[asset_id] = mapping
            return mapping

        return None

    def get_cpts_for_asset_cascade(
        self,
        asset_id: int,
        max_age_hours: int = FRESHNESS_THRESHOLD_HOURS,
        run_id: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        Cascade lookup voor CPTs van een asset.

        Volgorde:
        1. Asset-specifieke CPT (scope_key = 'asset_{id}')
        2. Preferred scope uit qbn.asset_cpt_mapping
        3. Fallback scope (default: 'global')

        Args:
            asset_id: Asset ID
            max_age_hours: Maximum leeftijd in uren

        Returns:
            Tuple van (cpts_dict, source_scope_key)
        """
        # Stap 1: Direct asset lookup
        scope_key = f'asset_{asset_id}'
        cpts = self.get_cpts_by_scope(scope_key, max_age_hours, run_id=run_id)
        if cpts:
            return cpts, scope_key

        # Stap 2: Check mapping
        mapping = self.get_asset_mapping(asset_id)
        if mapping and mapping['preferred_scope']:
            cpts = self.get_cpts_by_scope(mapping['preferred_scope'], max_age_hours, run_id=run_id)
            if cpts:
                return cpts, mapping['preferred_scope']

        # Stap 3: Fallback naar global (indien aanwezig)
        fallback = mapping['fallback_scope'] if mapping else 'global'
        cpts = self.get_cpts_by_scope(fallback, max_age_hours, run_id=run_id)
        if cpts:
            return cpts, fallback

        # Stap 4: Geen CPT gevonden
        logger.warning(f"Geen CPT gevonden voor asset {asset_id} in cascade lookup")
        return {}, 'none'

    def set_asset_mapping(
        self,
        asset_id: int,
        preferred_scope: Optional[str] = None,
        fallback_scope: str = 'global'
    ):
        """
        Configureer de CPT scope mapping voor een asset.

        Args:
            asset_id: Asset ID
            preferred_scope: Gewenste scope (None = gebruik asset-specifiek indien beschikbaar)
            fallback_scope: Fallback scope (default: 'global')
        """
        query = """
        INSERT INTO qbn.asset_cpt_mapping (asset_id, preferred_scope, fallback_scope, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (asset_id)
        DO UPDATE SET
            preferred_scope = EXCLUDED.preferred_scope,
            fallback_scope = EXCLUDED.fallback_scope,
            updated_at = NOW()
        """

        with get_cursor(commit=True) as cur:
            cur.execute(query, (asset_id, preferred_scope, fallback_scope))

        # Invalidate cache
        if asset_id in self._asset_mapping_cache:
            del self._asset_mapping_cache[asset_id]

        logger.info(f"Set CPT mapping for asset {asset_id}: preferred={preferred_scope}, fallback={fallback_scope}")

    def get_scope_info(self, scope_key: str) -> Optional[Dict[str, Any]]:
        """
        Haal metadata op voor een scope.

        Returns:
            Dict met scope_type, source_assets, generated_at, node_count
        """
        query = """
        SELECT
            scope_type,
            source_assets,
            MAX(generated_at) as latest_generated,
            COUNT(*) as node_count,
            SUM(observations) as total_observations
        FROM qbn.cpt_cache
        WHERE scope_key = %s AND model_version = %s
        GROUP BY scope_type, source_assets
        """

        with get_cursor() as cur:
            cur.execute(query, (scope_key, MODEL_VERSION))
            row = cur.fetchone()

        if row:
            return {
                'scope_key': scope_key,
                'scope_type': row[0],
                'source_assets': row[1] or [],
                'generated_at': row[2],
                'node_count': row[3],
                'total_observations': row[4]
            }

        return None

    def list_available_scopes(self) -> List[Dict[str, Any]]:
        """
        Lijst alle beschikbare CPT scopes.

        Returns:
            Lijst van scope info dictionaries
        """
        query = """
        SELECT
            scope_key,
            scope_type,
            source_assets,
            MAX(generated_at) as latest_generated,
            COUNT(*) as node_count
        FROM qbn.cpt_cache
        WHERE model_version = %s AND scope_key IS NOT NULL
        GROUP BY scope_key, scope_type, source_assets
        ORDER BY scope_type, scope_key
        """

        scopes = []
        with get_cursor() as cur:
            cur.execute(query, (MODEL_VERSION,))
            for row in cur.fetchall():
                scopes.append({
                    'scope_key': row[0],
                    'scope_type': row[1],
                    'source_assets': row[2] or [],
                    'generated_at': row[3],
                    'node_count': row[4]
                })

        return scopes

