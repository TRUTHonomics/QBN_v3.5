"""
Event Window Detector voor QBN v3.1

Detecteert en labelt "Event Windows" - periodes tussen een Leading spike
(trade trigger) en het bereiken van een barrier outcome.

Deze module is fundamenteel voor:
- Position_Confidence data-driven training (Plan 03)
- Position_Prediction CPT generatie (Plan 04)

Event Detection Algoritme:
1. Detecteer spike wanneer |leading_score| > threshold OF delta > threshold
2. Markeer start van event window
3. Vind einde: wanneer barrier is geraakt OF timeout (24h max)
4. Label alle rijen in het window met event_id, time_since_entry

USAGE:
    from inference.event_window_detector import EventWindowDetector, EventWindowConfig

    config = EventWindowConfig(absolute_threshold=0.5, delta_threshold=0.3)
    detector = EventWindowDetector(config)
    events, labeled_data = detector.detect_events(barrier_data, asset_id)
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.config_defaults import (
    DEFAULT_EVENT_ABSOLUTE_THRESHOLD,
    DEFAULT_EVENT_DELTA_THRESHOLD,
    DEFAULT_EVENT_MAX_WINDOW_MINUTES,
    DEFAULT_SIGNIFICANT_BARRIER_ATR
)
from core.config_warnings import warn_fallback_active
from config.ida_config import IDAConfig
from database.db import get_cursor

logger = logging.getLogger(__name__)


@dataclass
class EventWindowConfig:
    """Configuratie voor Event Window Detection."""

    # Spike detectie thresholds
    # REASON: Gebruik centrale defaults van core/config_defaults.py
    absolute_threshold: float = DEFAULT_EVENT_ABSOLUTE_THRESHOLD      # |leading_score| > X triggers spike
    delta_threshold: float = DEFAULT_EVENT_DELTA_THRESHOLD         # score change > X triggers spike

    # Window parameters
    max_window_minutes: int = DEFAULT_EVENT_MAX_WINDOW_MINUTES       # 24h max window
    min_window_minutes: int = 60         # 1h minimum voor valide event
    cooldown_minutes: int = 60           # Minimum tijd tussen events

    # Barrier configuratie
    significant_barrier_atr: float = DEFAULT_SIGNIFICANT_BARRIER_ATR  # Minimum barrier voor "significant" outcome

    # Direction determination
    direction_threshold: float = 0.0     # Score > 0 = long, < 0 = short


@dataclass
class EventWindow:
    """Representatie van een gedetecteerd event window."""

    event_id: str
    asset_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    trigger_score: float
    trigger_delta: float
    direction: str  # 'long' | 'short'
    outcome: str    # 'up_strong' | 'up_weak' | 'down_strong' | 'down_weak' | 'timeout'
    duration_minutes: int
    n_rows: int
    
    # v3.2: Entry scores voor delta berekening
    entry_coincident_score: float = 0.0
    entry_confirming_score: float = 0.0
    
    # v3.3: Entry leading score voor Momentum_Prediction
    entry_leading_score: float = 0.0
    entry_atr: float = 0.0  # ATR bij entry voor Volatility_Regime
    entry_close: float = 0.0  # Close prijs bij entry voor Exit_Timing

    def to_dict(self) -> Dict[str, Any]:
        """Convert naar dictionary voor database opslag."""
        return {
            'event_id': self.event_id,
            'asset_id': self.asset_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'trigger_score': self.trigger_score,
            'trigger_delta': self.trigger_delta,
            'direction': self.direction,
            'outcome': self.outcome,
            'duration_minutes': self.duration_minutes,
            'n_rows': self.n_rows,
            'entry_coincident_score': self.entry_coincident_score,
            'entry_confirming_score': self.entry_confirming_score,
            'entry_leading_score': self.entry_leading_score,
            'entry_atr': self.entry_atr,
            'entry_close': self.entry_close
        }


@dataclass
class EventDetectionStats:
    """Statistieken van event detectie."""

    total_events: int
    avg_duration_min: float
    median_duration_min: float
    events_with_timeout: int
    outcome_distribution: Dict[str, int]
    direction_distribution: Dict[str, int]
    avg_rows_per_event: float


class EventWindowDetector:
    """
    Detecteert event windows in barrier outcome data.

    Een event window is de periode vanaf het moment dat een Leading signal
    een significante waarde bereikt (spike) tot het moment dat een barrier
    outcome wordt bereikt of de maximum window tijd verstrijkt.
    """

    def __init__(self, config: Optional[EventWindowConfig] = None, asset_id: Optional[int] = None):
        """
        Initialiseer detector met configuratie.

        Args:
            config: EventWindowConfig, of None voor defaults
            asset_id: Optioneel asset ID voor dynamic threshold lookup
        """
        self.config = config or EventWindowConfig()
        
        # REASON: Dynamic threshold lookup if asset_id is provided
        if asset_id is not None:
            try:
                from config.threshold_loader import ThresholdLoader
                loader = ThresholdLoader(asset_id=asset_id, horizon='1h')
                
                if loader.is_from_database:
                    # Update config with DB values if available
                    # Leading composite strong_threshold corresponds to absolute_threshold
                    self.config.absolute_threshold = loader.composite_strong_threshold
                    logger.info(f"EventWindowDetector: Loaded dynamic absolute_threshold={self.config.absolute_threshold} for asset {asset_id}")
                else:
                    warn_fallback_active(
                        component="EventWindowDetector",
                        config_name=f"asset_{asset_id}_thresholds",
                        fallback_values={'abs': self.config.absolute_threshold, 'delta': self.config.delta_threshold},
                        reason="Geen thresholds in DB gevonden voor dit asset",
                        fix_command="Draai 'Threshold Optimalisatie'"
                    )
            except Exception as e:
                warn_fallback_active(
                    component="EventWindowDetector",
                    config_name=f"asset_{asset_id}_thresholds",
                    fallback_values={'abs': self.config.absolute_threshold, 'delta': self.config.delta_threshold},
                    reason=f"Fout bij dynamic lookup: {e}"
                )

        logger.info(
            f"EventWindowDetector initialized "
            f"(abs_threshold={self.config.absolute_threshold}, "
            f"delta_threshold={self.config.delta_threshold}, "
            f"max_window={self.config.max_window_minutes}m)"
        )

    def detect_events(
        self,
        data: pd.DataFrame,
        asset_id: int
    ) -> Tuple[List[EventWindow], pd.DataFrame]:
        """
        Detecteer event windows in de data.

        Args:
            data: DataFrame met kolommen:
                - time_1: timestamp
                - leading_composite of leading_score: composite score
                - first_significant_barrier: barrier outcome
                - first_significant_time_min: tijd tot barrier
                - coincident_score (optioneel): voor delta berekening
                - confirming_score (optioneel): voor delta berekening
            asset_id: Asset identifier

        Returns:
            Tuple van (list van EventWindow, DataFrame met event labels)
            
        v3.2 NIEUW: labeled_data bevat nu ook:
            - entry_coincident_score: coincident score bij entry
            - entry_confirming_score: confirming score bij entry
            - delta_cum_coincident: cumulatieve delta (direction-aware)
            - delta_cum_confirming: cumulatieve delta (direction-aware)
            - uniqueness_weight: 1/N per event (López de Prado)
        """
        logger.info(f"🎯 Detecting event windows for asset {asset_id} ({len(data)} rows)")

        # Valideer input
        required_cols = ['time_1']
        score_col = self._find_score_column(data)

        if score_col is None:
            logger.error("No score column found (leading_composite or leading_score)")
            return [], data

        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return [], data

        # Sort by time
        data = data.sort_values('time_1').reset_index(drop=True)

        # Calculate delta scores (handle None/NaN gracefully)
        # REASON: Fix TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
        # Zorg dat de score kolom numeric is en None/NaN als 0.0 of ffill wordt behandeld
        data[score_col] = pd.to_numeric(data[score_col], errors='coerce').fillna(0.0)
        data['_score_delta'] = data[score_col].diff().abs().fillna(0.0)
        
        # v3.2/v3.3: Zorg dat composite scores numeric zijn
        for col in ['coincident_score', 'confirming_score', 'leading_score', 'atr', 'close']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)

        # Detect events
        events = []
        labeled_data = data.copy()
        labeled_data['event_id'] = None
        labeled_data['time_since_entry_min'] = None
        
        # v3.2: Initialiseer delta score kolommen
        labeled_data['entry_coincident_score'] = None
        labeled_data['entry_confirming_score'] = None
        labeled_data['delta_cum_coincident'] = None
        labeled_data['delta_cum_confirming'] = None
        
        # v3.3: Initialiseer leading delta en retrospectieve kolommen
        labeled_data['entry_leading_score'] = None
        labeled_data['delta_cum_leading'] = None
        labeled_data['entry_atr'] = None
        labeled_data['atr_ratio'] = None
        labeled_data['entry_close'] = None
        labeled_data['return_since_entry'] = None

        i = 0
        last_event_end = None

        while i < len(data):
            row = data.iloc[i]

            # Check cooldown
            if last_event_end is not None:
                time_since_last = (row['time_1'] - last_event_end).total_seconds() / 60
                if time_since_last < self.config.cooldown_minutes:
                    i += 1
                    continue

            # Check for spike
            score = row[score_col]
            delta = row['_score_delta']

            if self._is_spike(score, delta):
                # Found spike - start event window
                event, end_idx = self._extract_event_window(
                    data, i, asset_id, score_col
                )

                if event is not None:
                    events.append(event)

                    # Label rows in window
                    start_time = data.iloc[i]['time_1']
                    
                    # v3.2: Haal entry scores op bij start van event
                    entry_coinc = data.iloc[i].get('coincident_score', 0.0) or 0.0
                    entry_conf = data.iloc[i].get('confirming_score', 0.0) or 0.0
                    
                    # v3.3: Haal ook leading, atr en close op
                    entry_lead = data.iloc[i].get('leading_score', 0.0) or 0.0
                    entry_atr = data.iloc[i].get('atr', 0.0) or 0.0
                    entry_close = data.iloc[i].get('close', 0.0) or 0.0
                    
                    # Update event met entry scores
                    event.entry_coincident_score = float(entry_coinc)
                    event.entry_confirming_score = float(entry_conf)
                    event.entry_leading_score = float(entry_lead)
                    event.entry_atr = float(entry_atr)
                    event.entry_close = float(entry_close)
                    
                    for j in range(i, end_idx + 1):
                        labeled_data.at[j, 'event_id'] = event.event_id
                        row_time = data.iloc[j]['time_1']
                        time_since_entry = int((row_time - start_time).total_seconds() / 60)
                        labeled_data.at[j, 'time_since_entry_min'] = time_since_entry
                        labeled_data.at[j, 'event_outcome'] = event.outcome
                        labeled_data.at[j, 'event_direction'] = event.direction
                        
                        # v3.2: Bereken delta scores (direction-aware)
                        labeled_data.at[j, 'entry_coincident_score'] = entry_coinc
                        labeled_data.at[j, 'entry_confirming_score'] = entry_conf
                        
                        current_coinc = data.iloc[j].get('coincident_score', 0.0) or 0.0
                        current_conf = data.iloc[j].get('confirming_score', 0.0) or 0.0
                        
                        delta_coinc = float(current_coinc) - float(entry_coinc)
                        delta_conf = float(current_conf) - float(entry_conf)
                        
                        # Direction-aware transformatie: voor short positions, inverteer delta
                        # Positieve delta = gunstig voor de positie
                        if event.direction == 'short':
                            delta_coinc *= -1
                            delta_conf *= -1
                        
                        labeled_data.at[j, 'delta_cum_coincident'] = delta_coinc
                        labeled_data.at[j, 'delta_cum_confirming'] = delta_conf
                        
                        # v3.3: Bereken delta_leading, atr_ratio, return_since_entry
                        labeled_data.at[j, 'entry_leading_score'] = entry_lead
                        current_lead = data.iloc[j].get('leading_score', 0.0) or 0.0
                        delta_lead = float(current_lead) - float(entry_lead)
                        if event.direction == 'short':
                            delta_lead *= -1
                        labeled_data.at[j, 'delta_cum_leading'] = delta_lead
                        
                        # ATR ratio voor Volatility_Regime
                        labeled_data.at[j, 'entry_atr'] = entry_atr
                        current_atr = data.iloc[j].get('atr', 0.0) or 0.0
                        atr_ratio = float(current_atr) / float(entry_atr) if entry_atr > 0 else 1.0
                        labeled_data.at[j, 'atr_ratio'] = atr_ratio
                        
                        # Return since entry voor Momentum_Prediction training
                        labeled_data.at[j, 'entry_close'] = entry_close
                        current_close = data.iloc[j].get('close', 0.0) or 0.0
                        if entry_close > 0:
                            ret = (float(current_close) - float(entry_close)) / float(entry_close)
                            # Direction-aware: voor short, inverteer return
                            if event.direction == 'short':
                                ret *= -1
                            labeled_data.at[j, 'return_since_entry'] = ret
                        else:
                            labeled_data.at[j, 'return_since_entry'] = 0.0

                    last_event_end = data.iloc[end_idx]['time_1']
                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1

        # Remove helper column
        labeled_data = labeled_data.drop(columns=['_score_delta'], errors='ignore')
        
        # v3.2: Bereken uniqueness_weight via IDA (López de Prado Soft-Attribution Delta)
        # REASON: IDA geeft meer gewicht aan signalen met hogere informatiewaarde (delta)
        #         in plaats van gelijke 1/N verdeling per event
        labeled_data['uniqueness_weight'] = self._compute_ida_weights_for_events(labeled_data)
        
        # Zet weight op 0 voor rijen zonder event
        labeled_data.loc[labeled_data['event_id'].isna(), 'uniqueness_weight'] = 0.0
        
        # Log IDA weighting effectiviteit
        in_event_mask = labeled_data['event_id'].notna()
        if in_event_mask.any():
            raw_rows = in_event_mask.sum()
            eff_obs = labeled_data.loc[in_event_mask, 'uniqueness_weight'].sum()
            logger.info(f"   IDA weighting: {raw_rows} in-event rows -> {eff_obs:.1f} effective obs")

        logger.info(f"✅ Detected {len(events)} event windows for asset {asset_id}")
        return events, labeled_data
    
    def _compute_ida_weights_for_events(self, labeled_data: pd.DataFrame) -> pd.Series:
        """
        Bereken IDA weights (López de Prado Soft-Attribution Delta) voor event windows.
        
        In tegenstelling tot simple 1/N weighting, geeft IDA meer gewicht aan rijen
        met hogere informatiewaarde (delta scores) binnen een event.
        
        Formula per rij i in event:
            score_i = |delta_cum_coincident_i|
            instant_delta_i = max(0, delta_cum_coincident_i - delta_cum_coincident_{i-1})
            A_i = (0.2 * score_i) + (0.8 * instant_delta_i) + epsilon
            weight_i = A_i / sum(A_j in event)
        
        Returns:
            Series met uniqueness_weight per rij
        """
        config = IDAConfig.baseline()
        weights = pd.Series(0.0, index=labeled_data.index)
        
        # Groepeer per event_id
        for event_id, event_df in labeled_data.groupby('event_id'):
            if pd.isna(event_id):
                continue
            
            # Sorteer op tijd binnen event
            event_df = event_df.sort_values('time_1')
            idx = event_df.index
            
            # Score basis: delta_cum_coincident (direction-aware)
            if 'delta_cum_coincident' in event_df.columns:
                score = event_df['delta_cum_coincident'].abs().fillna(0)
                instant_delta = event_df['delta_cum_coincident'].diff().clip(lower=0).fillna(0)
            else:
                # Fallback naar simple 1/N als geen delta scores beschikbaar
                fallback_val = 1.0 / len(event_df)
                for i in idx:
                    weights.at[i] = fallback_val
                continue
            
            # IDA Attribution formula
            attrib = (
                config.score_weight * score +
                config.delta_weight * instant_delta +
                config.epsilon
            )
            
            # Normaliseer binnen event
            total = attrib.sum()
            if total > 0:
                event_weights = attrib / total
            else:
                # Fallback naar simple 1/N
                event_weights = pd.Series(1.0 / len(event_df), index=idx)
            
            # REASON: Expliciete loop met .at[] omzeilt pandas dtype inference problemen
            # die optreden bij Series.update() of .loc[] met arrays
            for i, val in zip(event_weights.index, event_weights.values):
                weights.at[i] = float(val)
        
        return weights

    def _find_score_column(self, data: pd.DataFrame) -> Optional[str]:
        """Vind de score kolom in de data."""
        candidates = ['leading_composite', 'leading_score', 'leading']
        for col in candidates:
            if col in data.columns:
                return col
        return None

    def _is_spike(self, score: float, delta: float) -> bool:
        """
        Bepaal of huidige observatie een spike is.

        Args:
            score: Huidige leading score
            delta: Absolute change sinds vorige observatie

        Returns:
            True als spike gedetecteerd
        """
        # Absolute threshold check
        if abs(score) >= self.config.absolute_threshold:
            return True

        # Delta threshold check
        if delta >= self.config.delta_threshold:
            return True

        return False

    def _extract_event_window(
        self,
        data: pd.DataFrame,
        start_idx: int,
        asset_id: int,
        score_col: str
    ) -> Tuple[Optional[EventWindow], int]:
        """
        Extraheer een event window vanaf start_idx.

        Args:
            data: Volledige dataset
            start_idx: Start index van de spike
            asset_id: Asset identifier
            score_col: Naam van score kolom

        Returns:
            Tuple van (EventWindow of None, end index)
        """
        start_row = data.iloc[start_idx]
        start_time = start_row['time_1']
        trigger_score = start_row[score_col]
        trigger_delta = start_row.get('_score_delta', 0)

        # Bepaal direction
        direction = 'long' if trigger_score > self.config.direction_threshold else 'short'

        # Zoek einde van event window
        end_idx = start_idx
        outcome = 'timeout'
        end_time = start_time

        for j in range(start_idx, len(data)):
            row = data.iloc[j]
            current_time = row['time_1']
            elapsed_min = int((current_time - start_time).total_seconds() / 60)

            # Check max window
            if elapsed_min > self.config.max_window_minutes:
                end_idx = j - 1 if j > start_idx else start_idx
                end_time = data.iloc[end_idx]['time_1']
                outcome = 'timeout'
                break

            # Check barrier hit
            if 'first_significant_barrier' in row and pd.notna(row['first_significant_barrier']):
                barrier = row['first_significant_barrier']
                barrier_time = row.get('first_significant_time_min', elapsed_min)

                if barrier != 'none' and barrier_time is not None:
                    # Barrier is geraakt
                    outcome = self._map_barrier_to_outcome(barrier)
                    end_idx = j
                    end_time = current_time
                    break

            end_idx = j
            end_time = current_time

        # Calculate duration
        duration_min = int((end_time - start_time).total_seconds() / 60)

        # Check minimum duration
        if duration_min < self.config.min_window_minutes:
            return None, end_idx

        # Count rows in window
        n_rows = end_idx - start_idx + 1

        # Generate event ID
        event_id = f"E{asset_id}_{start_time.strftime('%Y%m%d%H%M')}_{uuid.uuid4().hex[:6]}"

        event = EventWindow(
            event_id=event_id,
            asset_id=asset_id,
            start_time=start_time,
            end_time=end_time,
            trigger_score=float(trigger_score),
            trigger_delta=float(trigger_delta),
            direction=direction,
            outcome=outcome,
            duration_minutes=duration_min,
            n_rows=n_rows
        )

        return event, end_idx

    def _map_barrier_to_outcome(self, barrier: str) -> str:
        """
        Map barrier string naar outcome state.

        Args:
            barrier: e.g., 'up_050', 'down_100', 'none'

        Returns:
            Outcome state: 'up_strong', 'up_weak', 'down_strong', 'down_weak', 'timeout'
        """
        if barrier is None or barrier == 'none':
            return 'timeout'

        barrier_lower = barrier.lower()

        # Check direction
        is_up = barrier_lower.startswith('up')
        is_down = barrier_lower.startswith('down')

        # Check strength (based on ATR level in barrier name)
        # up_025, up_050, up_075 = weak; up_100, up_125, up_150 = strong
        is_strong = any(x in barrier_lower for x in ['100', '125', '150', '1_'])

        if is_up:
            return 'up_strong' if is_strong else 'up_weak'
        elif is_down:
            return 'down_strong' if is_strong else 'down_weak'
        else:
            return 'timeout'

    def validate_events(self, events: List[EventWindow]) -> EventDetectionStats:
        """
        Valideer en genereer statistieken voor gedetecteerde events.

        Args:
            events: Lijst van EventWindow objecten

        Returns:
            EventDetectionStats met samenvattende statistieken
        """
        if not events:
            return EventDetectionStats(
                total_events=0,
                avg_duration_min=0,
                median_duration_min=0,
                events_with_timeout=0,
                outcome_distribution={},
                direction_distribution={},
                avg_rows_per_event=0
            )

        durations = [e.duration_minutes for e in events]
        outcomes = [e.outcome for e in events]
        directions = [e.direction for e in events]
        rows = [e.n_rows for e in events]

        outcome_dist = {}
        for o in outcomes:
            outcome_dist[o] = outcome_dist.get(o, 0) + 1

        direction_dist = {}
        for d in directions:
            direction_dist[d] = direction_dist.get(d, 0) + 1

        return EventDetectionStats(
            total_events=len(events),
            avg_duration_min=float(np.mean(durations)),
            median_duration_min=float(np.median(durations)),
            events_with_timeout=outcome_dist.get('timeout', 0),
            outcome_distribution=outcome_dist,
            direction_distribution=direction_dist,
            avg_rows_per_event=float(np.mean(rows))
        )


def load_barrier_outcomes(asset_id: int, lookback_days: int = 365) -> pd.DataFrame:
    """
    Laad barrier outcomes met leading composite scores.

    Args:
        asset_id: Asset identifier
        lookback_days: Aantal dagen terug

    Returns:
        DataFrame met barrier outcomes en leading scores
    """
    # REASON: Gebruik leading_score direct uit barrier_outcomes (gevuld door materialize_leading_scores.py)
    # EXPL: kfl.mtf_signals_lead heeft geen weighted_score kolom, deze wordt gematerialiseerd in QBN schema.
    query = """
        SELECT
            time_1,
            first_significant_barrier,
            first_significant_time_min,
            COALESCE(leading_score, 0) as leading_composite
        FROM qbn.barrier_outcomes
        WHERE asset_id = %(asset_id)s
          AND time_1 >= NOW() - INTERVAL '%(lookback)s days'
        ORDER BY time_1
    """

    with get_cursor() as cur:
        cur.execute(query, {'asset_id': asset_id, 'lookback': lookback_days})
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    logger.info(f"Loaded {len(df)} barrier outcomes for asset {asset_id}")
    return df


def save_event_labels_to_db(asset_id: int, labeled_data: pd.DataFrame, run_id: Optional[str] = None):
    """
    Sla event labels op in de database.

    Updates qbn.barrier_outcomes met event_id kolom.

    Args:
        asset_id: Asset identifier
        labeled_data: DataFrame met event_id kolom
        run_id: Optional run identifier
    """
    # Filter rows with event labels
    labeled_rows = labeled_data[labeled_data['event_id'].notna()]

    if labeled_rows.empty:
        logger.warning("No labeled rows to save")
        return

    logger.info(f"Saving {len(labeled_rows)} event labels for asset {asset_id}")

    with get_cursor(commit=True) as cur:
        for _, row in labeled_rows.iterrows():
            cur.execute("""
                UPDATE qbn.barrier_outcomes
                SET event_id = %s
                WHERE asset_id = %s AND time_1 = %s
            """, (row['event_id'], asset_id, row['time_1']))

    logger.info(f"✅ Event labels saved to qbn.barrier_outcomes")


def save_events_to_cache(events: List[EventWindow], run_id: Optional[str] = None):
    """
    Sla events op in de event_windows cache tabel.

    Args:
        events: Lijst van EventWindow objecten
        run_id: Optional run identifier
    """
    if not events:
        return

    logger.info(f"Saving {len(events)} events to qbn.event_windows")

    with get_cursor(commit=True) as cur:
        for event in events:
            cur.execute("""
                INSERT INTO qbn.event_windows (
                    event_id, asset_id, start_time, end_time,
                    trigger_score, direction, outcome,
                    duration_minutes, n_rows, run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET
                    end_time = EXCLUDED.end_time,
                    outcome = EXCLUDED.outcome,
                    duration_minutes = EXCLUDED.duration_minutes,
                    n_rows = EXCLUDED.n_rows
            """, (
                event.event_id, event.asset_id, event.start_time, event.end_time,
                event.trigger_score, event.direction, event.outcome,
                event.duration_minutes, event.n_rows, run_id
            ))

    logger.info(f"✅ Events saved to qbn.event_windows cache")
    
    # HANDSHAKE_OUT logging
    from core.step_validation import log_handshake_out
    log_handshake_out(
        step="run_event_window_detection",
        target="qbn.event_windows",
        run_id=run_id or "N/A",
        rows=len(events),
        operation="INSERT"
    )
