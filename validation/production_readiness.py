"""
Production Readiness Validator for QBN v3.

Performs comprehensive validation of all components required for production inference:
- Data Layer: Outcome coverage, signal data availability
- Configuration Layer: Threshold config, signal weights, signal classification
- CPT Quality: Availability, key coverage, entropy, staleness
- Inference Simulation: Key matching, latency

Returns GO/NO-GO verdict based on pass/fail thresholds.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import time

from database.db import get_cursor
from config.network_config import MODEL_VERSION, FRESHNESS_THRESHOLD_HOURS

logger = logging.getLogger(__name__)

# Thresholds for pass/warn/fail
THRESHOLDS = {
    'outcome_coverage': {'pass': 0.80, 'warn': 0.50},
    'signal_weights_coverage': {'pass': 0.80, 'warn': 0.50},
    'threshold_config_entries': {'pass': 9, 'warn': 3},
    'cpt_node_coverage': {'pass': 0.80, 'warn': 0.60},
    'cpt_key_coverage': {'pass': 0.70, 'warn': 0.40},
    'cpt_entropy_low': 0.3,
    'cpt_entropy_high': 3.0,
    'cpt_staleness_hours': {'pass': 24, 'warn': 72},
    'key_matching_hit_rate': {'pass': 0.90, 'warn': 0.70},
    'inference_latency_ms': {'pass': 50, 'warn': 200},
}


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    value: Any
    message: str
    threshold: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ProductionReadinessValidator:
    """
    Validates all components required for production inference.

    Usage:
        validator = ProductionReadinessValidator(asset_id=1)
        verdict, results = validator.run_all_checks()
        # verdict is 'GO' or 'NO-GO'
    """

    def __init__(self, asset_id: int, run_id: Optional[str] = None):
        self.asset_id = asset_id
        self.run_id = run_id
        self.results: List[CheckResult] = []
        self._asset_symbol: Optional[str] = None

    def _run_id_clause(self, table: str, alias: str = "") -> Tuple[str, List[Any]]:
        """
        Bouw optionele run_id filter clause voor qbn-tabellen.

        REASON: Validation moet dezelfde training-run evalueren (traceerbaarheid).
        TODO-verify: Als een tabel geen run_id kolom heeft, valt dit terug naar no-op.
        """
        if not self.run_id:
            return "", []
        try:
            with get_cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema='qbn'
                      AND table_name=%s
                      AND column_name='run_id'
                    LIMIT 1
                    """,
                    (table,),
                )
                has = cur.fetchone() is not None
        except Exception:
            has = False
        if not has:
            return "", []
        prefix = f"{alias}." if alias else ""
        return f" AND {prefix}run_id = %s", [self.run_id]

    def _check_run_id_exists_in_table(self, table: str) -> bool:
        """
        Defensieve check: bestaat de opgegeven run_id in de gegeven tabel?

        REASON: Als een run_id niet in een tabel voorkomt, zullen queries 0 rijen
        opleveren, wat misleidende N/A of FAIL resultaten geeft. Dit helpt bij
        troubleshooting van configuratie mismatches.
        """
        if not self.run_id:
            return True  # Geen run_id filter -> altijd OK
        try:
            with get_cursor() as cur:
                # Check of kolom run_id bestaat
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema='qbn'
                      AND table_name=%s
                      AND column_name='run_id'
                    LIMIT 1
                    """,
                    (table,),
                )
                if not cur.fetchone():
                    return True  # Geen run_id kolom -> filter niet actief
                # Check of run_id bestaat in tabel
                cur.execute(
                    f"SELECT 1 FROM qbn.{table} WHERE run_id = %s LIMIT 1",
                    (self.run_id,),
                )
                exists = cur.fetchone() is not None
                if not exists:
                    logger.warning(
                        f"⚠️ run_id '{self.run_id}' niet gevonden in qbn.{table}. "
                        f"Dit kan leiden tot 0-rij resultaten (N/A/FAIL). "
                        f"Controleer of training consistent dezelfde run_id heeft."
                    )
                return exists
        except Exception as e:
            logger.debug(f"_check_run_id_exists_in_table({table}): {e}")
            return True  # Bij fout niet blokkeren

    @property
    def asset_symbol(self) -> str:
        """Get asset symbol for display."""
        if self._asset_symbol is None:
            with get_cursor() as cur:
                # REASON: Kolom heet kraken_symbol, niet symbol
                cur.execute(
                    "SELECT kraken_symbol FROM symbols.symbols WHERE id = %s",
                    (self.asset_id,)
                )
                row = cur.fetchone()
                self._asset_symbol = row[0] if row else f"Asset_{self.asset_id}"
        return self._asset_symbol

    # =========================================================================
    # DATA LAYER CHECKS
    # =========================================================================

    def check_outcome_coverage(self) -> List[CheckResult]:
        """
        Validate outcome coverage for all horizons (1h, 4h, 1d).

        REASON: Coverage moet alleen rijen meetellen die OUD GENOEG zijn
        om een outcome te kunnen hebben. Anders krijg je misleidende
        percentages (bijv. 25% voor 4h = 1/4 omdat alle 1h rijen meetellen).

        Pass: >80% coverage
        Warn: 50-80%
        Fail: <50%
        """
        results = []
        # REASON: horizon_hours bepaalt hoe oud een rij moet zijn voor geldige outcome
        horizon_config = [
            ('1h', 1),
            ('4h', 4),
            ('1d', 24),
        ]

        # Defensive check: run_id moet bestaan in barrier_outcomes
        self._check_run_id_exists_in_table("barrier_outcomes")

        try:
            for horizon, hours in horizon_config:
                with get_cursor() as cur:
                    # REASON: Tel alleen rijen die oud genoeg zijn (time_1 + horizon < NOW)
                    # Voor barrier_outcomes beschouwen we een rij als 'voorzien van outcome' 
                    # als first_significant_barrier is ingevuld (ook als het 'none' is).
                    run_clause, run_params = self._run_id_clause("barrier_outcomes")
                    cur.execute(f"""
                        SELECT
                            COUNT(*) as eligible,
                            COUNT(first_significant_barrier) as has_outcome
                        FROM qbn.barrier_outcomes
                        WHERE asset_id = %s
                          AND time_1 <= NOW() - INTERVAL '{hours} hours'
                          {run_clause}
                    """, tuple([self.asset_id] + run_params))
                    row = cur.fetchone()

                eligible, has_outcome = row if row else (0, 0)

                if eligible == 0:
                    results.append(CheckResult(
                        name=f'Outcome Coverage {horizon}',
                        status='WARN',
                        value='N/A',
                        message=f'No rows old enough for {horizon} outcome',
                        threshold='>80%'
                    ))
                    continue

                coverage = has_outcome / eligible

                if coverage >= THRESHOLDS['outcome_coverage']['pass']:
                    status = 'PASS'
                elif coverage >= THRESHOLDS['outcome_coverage']['warn']:
                    status = 'WARN'
                else:
                    status = 'FAIL'

                results.append(CheckResult(
                    name=f'Outcome Coverage {horizon}',
                    status=status,
                    value=f'{coverage:.0%}',
                    message=f'{has_outcome:,} of {eligible:,} eligible rows',
                    threshold='>80%'
                ))

        except Exception as e:
            for horizon, _ in horizon_config:
                results.append(CheckResult(
                    name=f'Outcome Coverage {horizon}',
                    status='FAIL',
                    value='ERROR',
                    message=str(e),
                    threshold='>80%'
                ))

        return results

    def check_signal_data_availability(self) -> CheckResult:
        """
        Check if MTF signal data is available for the asset.
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM kfl.mtf_signals_lead
                    WHERE asset_id = %s
                """, (self.asset_id,))
                count = cur.fetchone()[0]

            if count > 0:
                return CheckResult(
                    name='Signal Data Availability',
                    status='PASS',
                    value=f'{count:,}',
                    message=f'{count:,} signal rows available',
                    threshold='>0'
                )
            else:
                return CheckResult(
                    name='Signal Data Availability',
                    status='FAIL',
                    value=0,
                    message='No MTF signal data found',
                    threshold='>0'
                )

        except Exception as e:
            return CheckResult(
                name='Signal Data Availability',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>0'
            )

    # =========================================================================
    # CONFIGURATION LAYER CHECKS
    # =========================================================================

    def check_threshold_config(self) -> CheckResult:
        """
        Validate qbn.composite_threshold_config for all horizons.

        Expected: 3 horizons (1h, 4h, 1d) with entries.
        Pass: 9+ entries (3 horizons x multiple params)
        Warn: 3-8 entries
        Fail: <3 entries
        """
        # Defensive check: run_id moet bestaan in composite_threshold_config
        self._check_run_id_exists_in_table("composite_threshold_config")

        try:
            with get_cursor() as cur:
                run_clause, run_params = self._run_id_clause("composite_threshold_config")
                cur.execute("""
                    SELECT horizon, COUNT(*)
                    FROM qbn.composite_threshold_config
                    WHERE asset_id = %s
                    """ + run_clause + """
                    GROUP BY horizon
                    ORDER BY horizon
                """, tuple([self.asset_id] + run_params))
                rows = cur.fetchall()

            total_entries = sum(r[1] for r in rows)
            horizons_found = [r[0] for r in rows]

            if total_entries >= THRESHOLDS['threshold_config_entries']['pass']:
                status = 'PASS'
            elif total_entries >= THRESHOLDS['threshold_config_entries']['warn']:
                status = 'WARN'
            else:
                status = 'FAIL'

            return CheckResult(
                name='Threshold Config',
                status=status,
                value=f'{total_entries} entries',
                message=f'Horizons: {", ".join(horizons_found) if horizons_found else "none"}',
                threshold='>=9 entries',
                details={'horizons': horizons_found, 'total': total_entries}
            )

        except Exception as e:
            return CheckResult(
                name='Threshold Config',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>=9 entries'
            )

    def check_signal_weights(self) -> CheckResult:
        """
        Validate signal weights coverage and layering.

        Compares signals in signal_classification with signal_weights.
        Also checks if both HYPOTHESIS and CONFIDENCE layers are present.
        Pass: >=80% coverage AND both layers present
        Warn: 50-80% OR one layer missing
        Fail: <50%
        """
        try:
            with get_cursor() as cur:
                # REASON: In v3.1 we primarily care about HYPOTHESIS (Leading) weights.
                # So we count total signals in LEADING class as the baseline.
                cur.execute("SELECT COUNT(*) FROM qbn.signal_classification WHERE semantic_class = 'LEADING'")
                total_required_signals = cur.fetchone()[0]

                # Count signals with weights AND check layers
                # REASON: In v3.1 weights are lowercase with suffix (rsi_oversold_60)
                # while classification is UPPERCASE without suffix (RSI_OVERSOLD).
                # We use regex to match them.
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT sw.signal_name) as weighted_count,
                        COUNT(DISTINCT sw.layer) as layer_count,
                        ARRAY_AGG(DISTINCT sw.layer) as layers
                    FROM qbn.signal_weights sw
                    JOIN qbn.signal_classification sc 
                      ON UPPER(REGEXP_REPLACE(sw.signal_name, '(_d|_240|_60|_1)$', '')) = sc.signal_name
                    WHERE (sw.asset_id = %s OR sw.asset_id = 0)
                      AND sc.semantic_class = 'LEADING'
                """, (self.asset_id,))
                row = cur.fetchone()
                
                signals_with_weights = row[0] or 0
                layers_found = row[2] or []

            if total_required_signals == 0:
                return CheckResult(
                    name='Signal Weights',
                    status='WARN',
                    value='0/0',
                    message='No LEADING signals in classification table',
                    threshold='>=80% + HYP'
                )

            coverage = signals_with_weights / total_required_signals if total_required_signals > 0 else 0
            
            # REASON: In v3.1 only HYPOTHESIS (Leading) weights are required.
            has_required_layers = 'HYPOTHESIS' in layers_found

            if coverage >= THRESHOLDS['signal_weights_coverage']['pass'] and has_required_layers:
                status = 'PASS'
            elif coverage >= THRESHOLDS['signal_weights_coverage']['warn']:
                status = 'WARN'
            else:
                status = 'FAIL'

            layer_msg = f"Layers: {', '.join(layers_found)}" if layers_found else "No layers"
            return CheckResult(
                name='Signal Weights',
                status=status,
                value=f'{signals_with_weights}/{total_required_signals} ({coverage:.0%})',
                message=f'{signals_with_weights} LEADING signals weighted. {layer_msg}',
                threshold='>=80% + HYP'
            )

        except Exception as e:
            return CheckResult(
                name='Signal Weights',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>=80%'
            )

    def check_signal_classification(self) -> CheckResult:
        """
        Check that signal_classification table has entries.
        """
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT semantic_class, COUNT(*)
                    FROM qbn.signal_classification
                    GROUP BY semantic_class
                    ORDER BY semantic_class
                """)
                rows = cur.fetchall()

            total = sum(r[1] for r in rows)

            if total == 0:
                return CheckResult(
                    name='Signal Classification',
                    status='FAIL',
                    value=0,
                    message='No signal classifications found',
                    threshold='>0'
                )

            breakdown = {r[0]: r[1] for r in rows}

            return CheckResult(
                name='Signal Classification',
                status='PASS',
                value=f'{total} entries',
                message=f'L:{breakdown.get("LEADING", 0)} C:{breakdown.get("COINCIDENT", 0)} F:{breakdown.get("CONFIRMING", 0)}',
                threshold='>0',
                details=breakdown
            )

        except Exception as e:
            return CheckResult(
                name='Signal Classification',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>0'
            )

    # =========================================================================
    # CPT QUALITY CHECKS
    # =========================================================================

    def check_cpt_availability(self) -> CheckResult:
        """
        Check if CPTs are available via cascade lookup.
        Uses the CPTCacheManager for proper scope resolution.
        """
        # Defensive check: run_id moet bestaan in split-tabellen
        # REASON: v3.4 gebruikt split-tabellen; check primary entry table
        self._check_run_id_exists_in_table("cpt_cache_entry")

        try:
            from inference.cpt_cache_manager import CPTCacheManager

            cache = CPTCacheManager()
            cpts, source_scope = cache.get_cpts_for_asset_cascade(
                self.asset_id,
                max_age_hours=FRESHNESS_THRESHOLD_HOURS * 3,  # Allow older CPTs for check
                run_id=self.run_id
            )

            if not cpts:
                return CheckResult(
                    name='CPT Availability',
                    status='FAIL',
                    value=0,
                    message='No CPTs available via cascade lookup',
                    threshold='>0 nodes'
                )

            # Expected nodes v3.4: 1 structural + 7 entry + 4 position = 12 nodes
            expected_nodes = 12

            if len(cpts) >= expected_nodes * THRESHOLDS['cpt_node_coverage']['pass']:
                status = 'PASS'
            elif len(cpts) >= expected_nodes * THRESHOLDS['cpt_node_coverage']['warn']:
                status = 'WARN'
            else:
                status = 'WARN'  # Some CPTs found, not critical

            return CheckResult(
                name='CPT Availability',
                status=status,
                value=f'{len(cpts)} nodes',
                message=f'via scope "{source_scope}"',
                threshold=f'>={int(expected_nodes * 0.8)} nodes',
                details={'scope': source_scope, 'nodes': list(cpts.keys())}
            )

        except Exception as e:
            return CheckResult(
                name='CPT Availability',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>0 nodes'
            )

    def check_cpt_key_coverage(self) -> CheckResult:
        """
        Test how many % of theoretical parent combinations have CPT data.

        For Prediction nodes: regime x hypothesis x confidence = 5 x 5 x 3 = 75 combinations
        Pass: >70%
        Warn: 40-70%
        Fail: <40%
        """
        try:
            with get_cursor() as cur:
                # Get a sample CPT to analyze key coverage
                # REASON: v3.4 - query cpt_cache_entry voor Prediction nodes
                run_clause, run_params = self._run_id_clause("cpt_cache_entry")
                cur.execute("""
                    SELECT node_name, cpt_data, coverage
                    FROM qbn.cpt_cache_entry
                    WHERE (scope_key = %s OR scope_key = 'global')
                      AND (node_name LIKE 'Prediction%%' OR node_name = 'Position_Prediction')
                """ + run_clause + """
                    ORDER BY generated_at DESC
                    LIMIT 1
                """, tuple([f'asset_{self.asset_id}'] + run_params))
                row = cur.fetchone()

            if not row:
                # Fall back to any CPT for this asset
                with get_cursor() as cur:
                    run_clause, run_params = self._run_id_clause("cpt_cache_entry")
                    cur.execute("""
                        SELECT AVG(coverage) FROM qbn.cpt_cache_entry
                        WHERE (asset_id = %s OR scope_key = %s)
                          AND coverage IS NOT NULL
                    """ + run_clause + """
                    """, tuple([self.asset_id, f'asset_{self.asset_id}'] + run_params))
                    avg_cov = cur.fetchone()[0]

                if avg_cov is None:
                    return CheckResult(
                        name='CPT Key Coverage',
                        status='WARN',
                        value='N/A',
                        message='No coverage data in CPT cache',
                        threshold='>70%'
                    )

                coverage = avg_cov
            else:
                node_name, cpt_data, coverage = row
                if coverage is None:
                    # Calculate from cpt_data if not stored
                    if isinstance(cpt_data, dict):
                        cond_probs = cpt_data.get('conditional_probabilities', {})
                        coverage = len(cond_probs) / 75 if cond_probs else 0
                    else:
                        coverage = 0

            if coverage >= THRESHOLDS['cpt_key_coverage']['pass']:
                status = 'PASS'
            elif coverage >= THRESHOLDS['cpt_key_coverage']['warn']:
                status = 'WARN'
            else:
                status = 'FAIL'

            return CheckResult(
                name='CPT Key Coverage',
                status=status,
                value=f'{coverage:.0%}',
                message='of theoretical parent combinations',
                threshold='>70%'
            )

        except Exception as e:
            return CheckResult(
                name='CPT Key Coverage',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>70%'
            )

    def check_cpt_entropy(self) -> CheckResult:
        """
        Validate average entropy of CPTs.

        Pass: 0.5-2.5 (informative but not random)
        Warn: 0.3-0.5 or 2.5-3.0
        Fail: <0.3 (overfit) or >3.0 (random)
        """
        try:
            with get_cursor() as cur:
                # REASON: v3.4 - query over alle drie split-tabellen met UNION
                if self.run_id:
                    run_clause = " AND run_id = %s"
                    run_params = [self.run_id]
                else:
                    run_clause = ""
                    run_params = []
                
                cur.execute("""
                    SELECT AVG(entropy), MIN(entropy), MAX(entropy), COUNT(*)
                    FROM (
                        SELECT entropy FROM qbn.cpt_cache_structural 
                        WHERE (asset_id = %s OR scope_key = %s) AND entropy IS NOT NULL""" + run_clause + """
                        UNION ALL
                        SELECT entropy FROM qbn.cpt_cache_entry 
                        WHERE (asset_id = %s OR scope_key = %s) AND entropy IS NOT NULL""" + run_clause + """
                        UNION ALL
                        SELECT entropy FROM qbn.cpt_cache_position 
                        WHERE (asset_id = %s OR scope_key = %s) AND entropy IS NOT NULL""" + run_clause + """
                    ) AS all_cpts
                """, tuple([
                    self.asset_id, f'asset_{self.asset_id}'] + run_params + [
                    self.asset_id, f'asset_{self.asset_id}'] + run_params + [
                    self.asset_id, f'asset_{self.asset_id}'] + run_params
                ))
                row = cur.fetchone()

            if not row or row[3] == 0:
                return CheckResult(
                    name='CPT Entropy',
                    status='WARN',
                    value='N/A',
                    message='No entropy data in CPT cache',
                    threshold='0.5-2.5'
                )

            avg_entropy, min_entropy, max_entropy, count = row

            if avg_entropy is None:
                return CheckResult(
                    name='CPT Entropy',
                    status='WARN',
                    value='N/A',
                    message='Entropy not calculated',
                    threshold='0.5-2.5'
                )

            low = THRESHOLDS['cpt_entropy_low']
            high = THRESHOLDS['cpt_entropy_high']

            if 0.5 <= avg_entropy <= 2.5:
                status = 'PASS'
            elif low <= avg_entropy < 0.5 or 2.5 < avg_entropy <= high:
                status = 'WARN'
            else:
                status = 'FAIL'

            return CheckResult(
                name='Average Entropy',
                status=status,
                value=f'{avg_entropy:.2f}',
                message=f'range: {min_entropy:.2f}-{max_entropy:.2f} across {count} nodes',
                threshold='0.5-2.5'
            )

        except Exception as e:
            return CheckResult(
                name='Average Entropy',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='0.5-2.5'
            )

    def check_cpt_staleness(self) -> CheckResult:
        """
        Check if CPTs are fresh enough.

        Pass: <24h old
        Warn: 24-72h old
        Fail: >72h old
        """
        try:
            with get_cursor() as cur:
                # REASON: v3.4 - query over alle drie split-tabellen
                # run_id filtering alleen als run_id beschikbaar is
                if self.run_id:
                    run_clause = " AND run_id = %s"
                    run_params = [self.run_id]
                else:
                    run_clause = ""
                    run_params = []
                
                cur.execute("""
                    SELECT GREATEST(
                        (SELECT MAX(generated_at) FROM qbn.cpt_cache_structural 
                         WHERE (asset_id = %s OR scope_key = %s)""" + run_clause + """),
                        (SELECT MAX(generated_at) FROM qbn.cpt_cache_entry 
                         WHERE (asset_id = %s OR scope_key = %s)""" + run_clause + """),
                        (SELECT MAX(generated_at) FROM qbn.cpt_cache_position 
                         WHERE (asset_id = %s OR scope_key = %s)""" + run_clause + """)
                    ) as max_generated
                """, tuple([
                    self.asset_id, f'asset_{self.asset_id}'] + run_params + [
                    self.asset_id, f'asset_{self.asset_id}'] + run_params + [
                    self.asset_id, f'asset_{self.asset_id}'] + run_params
                ))
                row = cur.fetchone()

            if not row or row[0] is None:
                return CheckResult(
                    name='CPT Staleness',
                    status='FAIL',
                    value='N/A',
                    message='No CPTs found',
                    threshold='<24h'
                )

            generated_at = row[0]
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - generated_at
            age_hours = age.total_seconds() / 3600

            if age_hours < THRESHOLDS['cpt_staleness_hours']['pass']:
                status = 'PASS'
                age_str = f'{age_hours:.0f}h ago'
            elif age_hours < THRESHOLDS['cpt_staleness_hours']['warn']:
                status = 'WARN'
                age_str = f'{age_hours:.0f}h ago'
            else:
                status = 'FAIL'
                age_str = f'{age.days}d {age.seconds // 3600}h ago'

            return CheckResult(
                name='CPT Staleness',
                status=status,
                value=age_str,
                message=f'Last generated: {generated_at.strftime("%Y-%m-%d %H:%M")}',
                threshold='<24h'
            )

        except Exception as e:
            return CheckResult(
                name='CPT Staleness',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='<24h'
            )

    # =========================================================================
    # INFERENCE SIMULATION
    # =========================================================================

    def check_key_matching(self) -> CheckResult:
        """
        Test CPT key coverage: hoeveel unieke parent-combinaties zijn aanwezig?

        REASON: v3.4 Prediction nodes hebben 2 parents (HTF_Regime|Trade_Hypothesis),
        Position_Prediction heeft 3 parents (Momentum|Volatility|Exit_Timing).
        We accepteren 2+ segmenten en rapporteren de gevonden variatie.

        Pass: >=5 unieke keys met variatie in meerdere dimensies
        Warn: 3-4 keys of beperkte variatie
        Fail: <3 keys
        """
        try:
            from inference.cpt_cache_manager import CPTCacheManager

            cache = CPTCacheManager()
            cpts, source_scope = cache.get_cpts_for_asset_cascade(self.asset_id, run_id=self.run_id)

            if not cpts:
                return CheckResult(
                    name='Key Matching Test',
                    status='FAIL',
                    value='0 keys',
                    message='No CPTs available for testing',
                    threshold='>=5 keys'
                )

            # Find Prediction_1h node (primary inference target)
            pred_node = cpts.get('Prediction_1h')
            if not pred_node:
                # Fallback to any Prediction node
                for node_name, cpt_data in cpts.items():
                    if ('Prediction' in node_name or node_name == 'Position_Prediction') and 'conditional_probabilities' in cpt_data:
                        pred_node = cpt_data
                        break

            if not pred_node or 'conditional_probabilities' not in pred_node:
                return CheckResult(
                    name='Key Matching Test',
                    status='WARN',
                    value='N/A',
                    message='No Prediction node with conditional_probabilities found',
                    threshold='>=5 keys'
                )

            cond_probs = pred_node.get('conditional_probabilities', {})
            available_keys = list(cond_probs.keys())
            num_keys = len(available_keys)

            # REASON: v3.4 key structuur is regime|hypothesis (2 segmenten)
            # Position_Prediction gebruikt momentum|volatility|exit_timing (3 segmenten)
            regimes = set()
            hypotheses = set()
            third_dim = set()  # Optioneel 3e segment voor Position nodes

            for key in available_keys:
                parts = key.split('|')
                if len(parts) >= 2:
                    regimes.add(parts[0])
                    hypotheses.add(parts[1])
                if len(parts) >= 3:
                    third_dim.add(parts[2])

            # REASON: Diversity score op basis van 2 primaire dimensies
            # Plus bonus voor 3e dimensie indien aanwezig
            diversity_score = len(regimes) + len(hypotheses) + len(third_dim)

            if num_keys >= 5 and diversity_score >= 4:
                status = 'PASS'
            elif num_keys >= 3:
                status = 'WARN'
            else:
                status = 'FAIL'

            # REASON: Duidelijke message die aangeeft hoeveel variatie per dimensie
            if third_dim:
                msg = f'Dim1:{len(regimes)} Dim2:{len(hypotheses)} Dim3:{len(third_dim)}'
            else:
                msg = f'Regimes:{len(regimes)} Hypotheses:{len(hypotheses)}'

            return CheckResult(
                name='Key Matching Test',
                status=status,
                value=f'{num_keys} keys',
                message=msg,
                threshold='>=5 keys',
                details={
                    'dimension_1': list(regimes),
                    'dimension_2': list(hypotheses),
                    'dimension_3': list(third_dim) if third_dim else [],
                    'sample_keys': available_keys[:5]
                }
            )

        except Exception as e:
            return CheckResult(
                name='Key Matching Test',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='>=5 keys'
            )

    def check_inference_latency(self) -> CheckResult:
        """
        Measure inference latency with sample runs.

        Pass: <50ms avg
        Warn: 50-200ms
        Fail: >200ms
        """
        try:
            from inference.cpt_cache_manager import CPTCacheManager

            cache = CPTCacheManager()

            # Measure time for cascade lookup
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                cpts, scope = cache.get_cpts_for_asset_cascade(self.asset_id, run_id=self.run_id)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)

            avg_latency = sum(latencies) / len(latencies)

            if avg_latency < THRESHOLDS['inference_latency_ms']['pass']:
                status = 'PASS'
            elif avg_latency < THRESHOLDS['inference_latency_ms']['warn']:
                status = 'WARN'
            else:
                status = 'FAIL'

            return CheckResult(
                name='Inference Latency',
                status=status,
                value=f'{avg_latency:.0f}ms avg',
                message=f'CPT lookup latency over 10 runs',
                threshold='<50ms'
            )

        except Exception as e:
            return CheckResult(
                name='Inference Latency',
                status='FAIL',
                value='ERROR',
                message=str(e),
                threshold='<50ms'
            )

    def check_entry_position_correlation(self) -> CheckResult:
        """
        Verify that entry quality correlates with position success (backtest data).

        Pass: backtest data present and win rate >= 40%
        Warn: no backtest data or win rate 30-40%
        Fail: win rate < 30% with sufficient trades
        """
        try:
            from validation.entry_position_correlation import analyze_entry_position_correlation
            result = analyze_entry_position_correlation(self.asset_id)
            by_dir = result.get("by_direction", {})
            if not by_dir:
                return CheckResult(
                    name="Entry-Position Correlation",
                    status="WARN",
                    value="N/A",
                    message="No backtest trades. Run Walk-Forward Backtest (stap 13) first.",
                )
            total_wins = sum(s["wins"] for s in by_dir.values())
            total_losses = sum(s["losses"] for s in by_dir.values())
            total = total_wins + total_losses
            win_rate = (total_wins / total * 100) if total else 0
            if total < 5:
                return CheckResult(
                    name="Entry-Position Correlation",
                    status="WARN",
                    value=f"{win_rate:.1f}% win rate ({total} trades)",
                    message="Few trades. Run longer backtest for reliable correlation.",
                )
            if win_rate >= 40:
                status = "PASS"
            elif win_rate >= 30:
                status = "WARN"
            else:
                status = "FAIL"
            return CheckResult(
                name="Entry-Position Correlation",
                status=status,
                value=f"{win_rate:.1f}% win rate",
                message=f"Backtest: {total_wins} wins, {total_losses} losses",
                threshold=">=40%",
            )
        except Exception as e:
            return CheckResult(
                name="Entry-Position Correlation",
                status="WARN",
                value="ERROR",
                message=str(e),
            )

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run_all_checks(self) -> Tuple[str, List[CheckResult]]:
        """
        Run all production readiness checks.

        Returns:
            Tuple of (verdict, results) where verdict is 'GO' or 'NO-GO'
        """
        self.results = []

        # Data Layer Checks
        self.results.extend(self.check_outcome_coverage())
        self.results.append(self.check_signal_data_availability())

        # Configuration Layer Checks
        self.results.append(self.check_threshold_config())
        self.results.append(self.check_signal_weights())
        self.results.append(self.check_signal_classification())

        # CPT Quality Checks
        self.results.append(self.check_cpt_availability())
        self.results.append(self.check_cpt_key_coverage())
        self.results.append(self.check_cpt_entropy())
        self.results.append(self.check_cpt_staleness())

        # Inference Simulation
        self.results.append(self.check_key_matching())
        self.results.append(self.check_inference_latency())

        # Entry-Position Correlation (backtest data)
        self.results.append(self.check_entry_position_correlation())

        # Determine verdict
        fail_count = sum(1 for r in self.results if r.status == 'FAIL')
        verdict = 'NO-GO' if fail_count > 0 else 'GO'

        return verdict, self.results

    def get_summary(self) -> Dict[str, int]:
        """Get summary counts of check results."""
        return {
            'pass': sum(1 for r in self.results if r.status == 'PASS'),
            'warn': sum(1 for r in self.results if r.status == 'WARN'),
            'fail': sum(1 for r in self.results if r.status == 'FAIL'),
            'total': len(self.results)
        }
