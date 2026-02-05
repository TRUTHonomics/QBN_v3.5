"""
Walk-Forward Validator voor QBN v2.

Deze module simuleert real-time inference op historische data door gebruik te maken
van een schuivend trainingsvenster (Moving Window).
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid
import pandas as pd
import numpy as np
from decimal import Decimal

from inference.trade_aligned_inference import TradeAlignedInference, SignalEvidence, DualInferenceResult
from inference.gpu.gpu_inference_engine import GPUInferenceEngine
from inference.inference_loader import InferenceLoader
from inference.node_types import OutcomeState, SemanticClass, BarrierOutcomeState
from inference.target_generator import ATR_THRESHOLDS
from database.db import get_cursor
from validation.backtest_config import BacktestConfig
from validation.trade_simulator import TradeSimulator, Trade

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Engine voor walk-forward validatie van het QBN v2 model.
    """

    def __init__(
        self,
        asset_id: int,
        output_dir: Optional[Path] = None,
        backtest_config: Optional[BacktestConfig] = None,
        backtest_id: Optional[str] = None,
        run_id: Optional[str] = None,
        persist_predictions: bool = True,
    ):
        self.asset_id = asset_id
        self.output_dir = output_dir or Path("_validation")
        # REASON: Gebruik de door TSEM geregistreerde backtest_id. Als hij ontbreekt
        # genereren we hier één keer zodat zowel trades als walk-forward predictions
        # consistent dezelfde sessie-id gebruiken.
        self.backtest_id = backtest_id or f"bt_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.loader = InferenceLoader()

        # Walk-forward predictions persistence (retrospectief, niet-live)
        self.persist_predictions = persist_predictions
        self.run_id = run_id or self._resolve_latest_complete_run_id()
        if self.persist_predictions and not self._db_table_exists("qbn", "walkforward_predictions"):
            logger.warning("⚠️ qbn.walkforward_predictions bestaat niet. Predictions worden niet gepersisteerd.")
            self.persist_predictions = False
        if self.persist_predictions and not self.run_id:
            logger.warning("⚠️ Geen complete run_id kunnen resolven. Predictions worden niet gepersisteerd.")
            self.persist_predictions = False
        
        # REASON: Gebruik InferenceLoader.load_inference_engine om CPTs, 
        # classificatie EN ThresholdLoader in één keer correct te laden.
        self.engine_cpu = self.loader.load_inference_engine(asset_id, horizon='1h', run_id=self.run_id or None)
        
        # Initialiseer de GPU engine voor batch verwerking
        # REASON: Gebruik dezelfde CPTs, classificatie EN threshold_loader als de CPU engine
        self.engine_gpu = GPUInferenceEngine(
            cpts=self.engine_cpu.cpts,
            signal_classification=self.engine_cpu.signal_classification,
            threshold_loader=self.engine_cpu._threshold_loader
        )
        
        # Bewaar referenties voor rapportage en data fetching
        self.cpts = self.engine_cpu.cpts
        self.classification = self.engine_cpu.signal_classification
        
        self.steps_results = []
        self.all_predictions = [] # Voor geaggregeerde metrieken
        self.params = {} # Voor rapportage
        
        # Backtest mode
        self.backtest_config = backtest_config
        self.trade_simulator = TradeSimulator(backtest_config) if backtest_config else None

    def _db_table_exists(self, schema: str, table: str) -> bool:
        """Runtime-safe check of een tabel bestaat."""
        try:
            with get_cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s
                      AND table_name = %s
                    LIMIT 1
                    """,
                    (schema, table),
                )
                return cur.fetchone() is not None
        except Exception:
            return False

    def _resolve_latest_complete_run_id(self) -> str:
        """
        Selecteer de meest recente COMPLETE training run voor dit asset.

        Definitie (conform validation pipeline):
        - >=10 CPT nodes in qbn.cpt_cache voor scope_key=asset_{id}
        - >=1 threshold entry in qbn.composite_threshold_config
        """
        try:
            with get_cursor() as cur:
                cur.execute(
                    """
                    WITH valid_cpt_runs AS (
                        SELECT
                            c.run_id,
                            COUNT(DISTINCT c.node_name) AS cpt_nodes,
                            MAX(c.generated_at) AS last_cpt_generated_at
                        FROM qbn.cpt_cache c
                        WHERE c.scope_key = %s
                          AND c.run_id IS NOT NULL
                        GROUP BY c.run_id
                        HAVING COUNT(DISTINCT c.node_name) >= 10
                    ),
                    valid_threshold_runs AS (
                        SELECT
                            t.run_id,
                            COUNT(*) AS threshold_entries,
                            MAX(t.updated_at) AS last_threshold_updated_at
                        FROM qbn.composite_threshold_config t
                        WHERE t.asset_id = %s
                          AND t.run_id IS NOT NULL
                        GROUP BY t.run_id
                        HAVING COUNT(*) >= 1
                    )
                    SELECT v.run_id
                    FROM valid_cpt_runs v
                    JOIN valid_threshold_runs t ON v.run_id = t.run_id
                    ORDER BY GREATEST(v.last_cpt_generated_at, t.last_threshold_updated_at) DESC
                    LIMIT 1
                    """,
                    (f"asset_{self.asset_id}", self.asset_id),
                )
                row = cur.fetchone()
                return str(row[0]) if row and row[0] else ""
        except Exception:
            return ""

    def _to_str(self, v) -> Optional[str]:
        """Robuust stringify voor enums/np scalars."""
        if v is None:
            return None
        if hasattr(v, "value"):
            return str(v.value)
        return str(self._normalize_value(v))

    def _barrier_pred_json(self, predicted_state: Optional[str], dist: Optional[dict]) -> Optional[dict]:
        """
        Maak een minimale barrier_prediction JSONB compatible structuur.
        Verwacht: expected_direction in {up,down,neutral}.
        """
        if not predicted_state:
            return None
        st = str(predicted_state).lower()
        if st.startswith("up"):
            direction = "up"
        elif st.startswith("down"):
            direction = "down"
        else:
            direction = "neutral"
        out = {"expected_state": predicted_state, "expected_direction": direction}
        if dist is not None:
            out["distribution"] = dist
        return out

    def _save_window_predictions_to_db(
        self,
        step_predictions: List[dict],
        train_window_start: datetime,
        train_window_end: datetime,
    ) -> None:
        """Persist predictions van 1 window naar qbn.walkforward_predictions."""
        if not self.persist_predictions:
            return
        if not step_predictions:
            return

        rows = []
        for p in step_predictions:
            t = p.get("time")
            pred_1h = self._to_str(p.get("prediction_1h"))
            pred_4h = self._to_str(p.get("prediction_4h"))
            pred_1d = self._to_str(p.get("prediction_1d"))

            dist_1h = self._normalize_value(p.get("dist_1h"))
            dist_4h = self._normalize_value(p.get("dist_4h"))
            dist_1d = self._normalize_value(p.get("dist_1d"))

            barrier_1h = self._barrier_pred_json(pred_1h, dist_1h if isinstance(dist_1h, dict) else None)
            barrier_4h = self._barrier_pred_json(pred_4h, dist_4h if isinstance(dist_4h, dict) else None)
            barrier_1d = self._barrier_pred_json(pred_1d, dist_1d if isinstance(dist_1d, dict) else None)

            rows.append(
                (
                    t,
                    int(self.asset_id),
                    str(self.run_id),
                    self._to_str(p.get("trade_hypothesis")),
                    None,  # entry_confidence (niet beschikbaar in huidige batch output)
                    self._to_str(p.get("leading_composite")),
                    self._to_str(p.get("coincident_composite")),
                    self._to_str(p.get("confirming_composite")),
                    pred_1h,
                    pred_4h,
                    pred_1d,
                    json.dumps(dist_1h, default=str) if dist_1h is not None else None,
                    json.dumps(dist_4h, default=str) if dist_4h is not None else None,
                    json.dumps(dist_1d, default=str) if dist_1d is not None else None,
                    json.dumps(barrier_1h, default=str) if barrier_1h is not None else None,
                    json.dumps(barrier_4h, default=str) if barrier_4h is not None else None,
                    json.dumps(barrier_1d, default=str) if barrier_1d is not None else None,
                    train_window_start,
                    train_window_end,
                    str(self.backtest_id),
                )
            )

        with get_cursor() as cur:
            cur.executemany(
                """
                INSERT INTO qbn.walkforward_predictions (
                    time,
                    asset_id,
                    run_id,
                    trade_hypothesis,
                    entry_confidence,
                    leading_composite,
                    coincident_composite,
                    confirming_composite,
                    prediction_1h,
                    prediction_4h,
                    prediction_1d,
                    distribution_1h,
                    distribution_4h,
                    distribution_1d,
                    barrier_prediction_1h,
                    barrier_prediction_4h,
                    barrier_prediction_1d,
                    train_window_start,
                    train_window_end,
                    backtest_run_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s
                )
                """,
                rows,
            )

    def _get_earliest_data_timestamp(self) -> Optional[datetime]:
        """Haal het vroegste timestamp op voor dit asset uit de database."""
        query = "SELECT MIN(time) FROM kfl.indicators WHERE asset_id = %s"
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        return None

    def run_validation(self, start_date: datetime, end_date: datetime, train_window_days: int = 90, test_step_days: int = 7):
        """
        Voer de walk-forward validatie uit over een tijdsperiode.
        """
        # REASON: Zorg voor timezone-aware vergelijkingen
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # REASON: Stel vast wanneer de data voor dit asset echt begint
        # EXPL: Voorkomt queries op periodes zonder data.
        earliest_db_time = self._get_earliest_data_timestamp()
        
        if earliest_db_time:
            if earliest_db_time.tzinfo is None:
                earliest_db_time = earliest_db_time.replace(tzinfo=timezone.utc)
                
            if start_date < earliest_db_time:
                logger.info(f"📍 Start datum {start_date} aangepast naar vroegste beschikbare data: {earliest_db_time}")
                start_date = earliest_db_time
        else:
            logger.error(f"❌ Geen data gevonden voor asset {self.asset_id} in kfl.indicators")
            return

        self.params = {
            'asset_id': self.asset_id,
            'start_date': start_date,
            'end_date': end_date,
            'train_window_days': train_window_days,
            'test_step_days': test_step_days,
            'atr_thresholds': ATR_THRESHOLDS,
            'signal_weights': {sig: info['weight'] for sig, info in self.classification.items() if 'weight' in info}
        }
        
        logger.info(f"🚀 Start Vectorized Walk-Forward Validatie voor Asset {self.asset_id}")
        logger.info(f"📅 Periode: {start_date.date()} tot {end_date.date()}")
        
        # 1. Haal ALLE data in één keer op voor de gehele periode
        all_data = self._fetch_historical_data(start_date, end_date)
        if all_data.empty:
            logger.error("❌ Geen data gevonden voor de opgegeven periode.")
            return

        # 2. Voer Batch Inference uit op de GPU (voor de gehele dataset)
        # REASON: Dit vervangt de trage rij-voor-rij loop op de CPU.
        logger.info(f"🧠 Start batch inference op {len(all_data)} rijen...")
        batch_results = self.engine_gpu.infer_batch(all_data)
        
        # 3. Walk-Forward Loop (nu alleen nog maar resultaten aggregeren)
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + timedelta(days=train_window_days)
            test_end = train_end + timedelta(days=test_step_days)
            
            if test_end > end_date:
                break
                
            # Filter de resultaten voor dit window
            mask = (all_data['time_1'] >= train_end) & (all_data['time_1'] < test_end)
            window_df = all_data[mask]
            
            if window_df.empty:
                current_train_start += timedelta(days=test_step_days)
                continue
            
            # Slices van batch_results voor dit window
            indices = window_df.index
            
            step_predictions = []
            for idx in indices:
                # Map batch results terug naar de entry structuur voor rapportage
                # REASON: Behoud compatibiliteit met bestaande rapportage functies.
                pred_entry = {
                    'time': all_data.loc[idx, 'time_1'],
                    'prediction_1h': batch_results['predictions']['1h']['states'][idx],
                    'actual_1h': all_data.loc[idx, 'outcome_1h'],
                    'dist_1h': batch_results['predictions']['1h']['distributions'][idx],
                    'prediction_4h': batch_results['predictions']['4h']['states'][idx],
                    'actual_4h': all_data.loc[idx, 'outcome_4h'],
                    'dist_4h': batch_results['predictions']['4h']['distributions'][idx],
                    'prediction_1d': batch_results['predictions']['1d']['states'][idx],
                    'actual_1d': all_data.loc[idx, 'outcome_1d'],
                    'dist_1d': batch_results['predictions']['1d']['distributions'][idx],
                    'regime': batch_results['regime'][idx],
                    'trade_hypothesis': batch_results['trade_hypothesis'][idx],
                    'leading_composite': batch_results['leading_composite'][idx],
                    'coincident_composite': batch_results['coincident_composite'][idx],
                    'confirming_composite': batch_results['confirming_composite'][idx]
                }
                step_predictions.append(pred_entry)
                self.all_predictions.append(pred_entry)
                
                # Backtest mode: simuleer trades
                if self.trade_simulator:
                    self._process_backtest_tick(idx, all_data, batch_results)

            # Persist window predictions (retrospectief) indien geconfigureerd
            self._save_window_predictions_to_db(
                step_predictions,
                train_window_start=current_train_start,
                train_window_end=train_end,
            )
            
            # 3. Bereken stap-metrieken
            step_metrics = self._calculate_metrics(step_predictions)
            self.steps_results.append({
                'window': f"{train_end.date()} to {test_end.date()}",
                'metrics': step_metrics,
                'count': len(step_predictions)
            })
            
            # Verschuif venster
            current_train_start += timedelta(days=test_step_days)

        # Sla eindrapport op
        if self.trade_simulator:
            self._save_backtest_report()
        else:
            self._save_report()

    def _convert_to_evidence(self, row: pd.Series) -> SignalEvidence:
        """Map database rij naar SignalEvidence object."""
        evidence = SignalEvidence(
            asset_id=self.asset_id,
            timestamp=row['time_1']
        )
        
        # REASON: classification bevat nu volledige namen MET suffix (rsi_oversold_d)
        # EXPL: We gebruiken deze direct als kolomnaam, zonder extra suffix toe te voegen.
        for full_sig_name, info in self.classification.items():
            sem_class = info['semantic_class']
            
            # Check of deze kolom in de data zit
            if full_sig_name in row and pd.notna(row[full_sig_name]):
                val = int(row[full_sig_name])
                
                if sem_class == SemanticClass.LEADING.value:
                    evidence.leading_signals[full_sig_name] = val
                elif sem_class == SemanticClass.COINCIDENT.value:
                    evidence.coincident_signals[full_sig_name] = val
                elif sem_class == SemanticClass.CONFIRMING.value:
                    evidence.confirming_signals[full_sig_name] = val
                    
        return evidence

    def _fetch_table_data(self, query: str, params: tuple) -> pd.DataFrame:
        """Helper om data op te halen in een aparte thread."""
        with get_cursor() as cur:
            cur.execute(query, params)
            if not cur.description:
                return pd.DataFrame()
            return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])

    def _fetch_historical_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Haal signalen en outcomes parallel op.
        
        REASON: Parallelle queries en efficiënte concat/join voor maximale snelheid.
        """
        logger.info(f"📥 Ophalen data voor asset {self.asset_id} ({start.date()} tot {end.date()}) parallel...")
        
        from concurrent.futures import ThreadPoolExecutor
        
        queries = []
        
        # 1. Klines (for close price)
        q_klines = """
            SELECT time as time_1, close
            FROM kfl.klines_raw
            WHERE asset_id = %s AND interval_min = '60' AND time BETWEEN %s AND %s
            ORDER BY time ASC
        """
        queries.append(('klines', q_klines, (self.asset_id, start, end)))
        
        # 2. Indicators (for ATR)
        q_ind = """
            SELECT time as time_1, atr_14
            FROM kfl.indicators 
            WHERE asset_id = %s AND interval_min = '60' AND time BETWEEN %s AND %s
            ORDER BY time ASC
        """
        queries.append(('indicators', q_ind, (self.asset_id, start, end)))
        
        # 3. Signals
        for table in ['kfl.mtf_signals_lead', 'kfl.mtf_signals_coin', 'kfl.mtf_signals_conf']:
            q_sig = f"SELECT * FROM {table} WHERE asset_id = %s AND time_1 BETWEEN %s AND %s"
            queries.append((table, q_sig, (self.asset_id, start, end)))
            
        # 4. Outcomes
        q_out = """
            SELECT time_1, first_significant_barrier, first_significant_time_min
            FROM qbn.barrier_outcomes 
            WHERE asset_id = %s AND time_1 BETWEEN %s AND %s
        """
        queries.append(('outcomes', q_out, (self.asset_id, start, end)))
        
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_name = {executor.submit(self._fetch_table_data, q, p): name for name, q, p in queries}
            for future in future_to_name:
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"❌ Fout bij ophalen {name}: {e}")
                    return pd.DataFrame()

        # Merge Logic: Start met klines, merge indicators en signals
        df = results['klines']
        if df.empty:
            return df
            
        # Merge indicators (ATR)
        ind_df = results.get('indicators')
        if ind_df is not None and not ind_df.empty:
            ind_df.set_index('time_1', inplace=True)
            df.set_index('time_1', inplace=True)
            df = df.join(ind_df, how='left')
            df.reset_index(inplace=True)
        
        # Rename columns to match expected format
        # REASON: Backtest draait op 60min interval (hardcoded in queries)
        df.rename(columns={'atr_14': 'atr', 'close': 'close_60'}, inplace=True)
        
        # Set time_1 as index for signal merging
        df.set_index('time_1', inplace=True)
        
        # Merge signals
        for table in ['kfl.mtf_signals_lead', 'kfl.mtf_signals_coin', 'kfl.mtf_signals_conf']:
            sig_df = results.get(table)
            if sig_df is not None and not sig_df.empty:
                sig_df.set_index('time_1', inplace=True)
                # Drop asset_id indien aanwezig
                if 'asset_id' in sig_df.columns:
                    sig_df.drop(columns=['asset_id'], inplace=True)
                
                # Alleen kolommen die nog niet bestaan (voorkom _x _y suffixes)
                cols_to_use = sig_df.columns.difference(df.columns)
                df = df.join(sig_df[cols_to_use], how='left')
                
        # Merge outcomes
        out_df = results.get('outcomes')
        if out_df is not None and not out_df.empty:
            out_df.set_index('time_1', inplace=True)
            df = df.join(out_df, how='left')
            
        # Reset index om time_1 weer als kolom te hebben
        df.reset_index(inplace=True)

        logger.info(f"📊 Dataset samengesteld: {len(df)} rijen, {len(df.columns)} kolommen")

        # REASON: On-the-fly mapping van barriers naar discrete states voor alle 3 horizons
        if not df.empty:
            # We maken 3 kolommen aan met default 'neutral'
            for horizon in ['1h', '4h', '1d']:
                df[f'outcome_{horizon}'] = 'neutral'

            if 'first_significant_barrier' in df.columns:
                # Vectorized mapping using apply (sneller dan iterrows)
                # Helper functie voor apply
                from inference.node_types import BarrierOutcomeState
                def map_barrier(row, window_min):
                    return BarrierOutcomeState.from_barrier(
                        row['first_significant_barrier'], 
                        row['first_significant_time_min'], 
                        window_min
                    ).value

                # Apply voor elke horizon
                for horizon, window_min in [('1h', 60), ('4h', 240), ('1d', 1440)]:
                    df[f'outcome_{horizon}'] = df.apply(lambda row: map_barrier(row, window_min), axis=1)
                
        return df

    def _calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Bereken validatie metrieken."""
        if not predictions:
            return {}
            
        metrics = {}
        for h in ["1h", "4h", "1d"]:
            preds = [p[f'prediction_{h}'] for p in predictions if p[f'actual_{h}'] is not None]
            acts = [p[f'actual_{h}'] for p in predictions if p[f'actual_{h}'] is not None]
            dists = [p[f'dist_{h}'] for p in predictions if p[f'actual_{h}'] is not None]
            
            if not preds:
                continue
                
            # 1. Accuracy (Meest waarschijnlijke state)
            correct = sum(1 for p, a in zip(preds, acts) if p == a)
            metrics[f'accuracy_{h}'] = correct / len(preds)
            
            # 2. Directional Accuracy (Bullish/Bearish/Neutral)
            def get_dir(s):
                if s is None: return 0
                s_lower = s.lower()
                if 'up' in s_lower or 'bullish' in s_lower: return 1
                if 'down' in s_lower or 'bearish' in s_lower: return -1
                return 0
                
            dir_correct = sum(1 for p, a in zip(preds, acts) if get_dir(p) == get_dir(a))
            metrics[f'dir_accuracy_{h}'] = dir_correct / len(preds)
            
            # 3. Brier Score (Probabilistisch)
            brier_sum = 0.0
            for dist, actual in zip(dists, acts):
                # We maken een one-hot vector voor de actual state
                for state, prob in dist.items():
                    target = 1.0 if state == actual else 0.0
                    brier_sum += (prob - target) ** 2
            
            metrics[f'brier_score_{h}'] = brier_sum / len(preds)
            
        return metrics

    def summary_report(self) -> str:
        """Genereer een tekstuele samenvatting van de resultaten."""
        if not self.all_predictions:
            return "Geen resultaten om te rapporteren."
            
        final_metrics = self._calculate_metrics(self.all_predictions)
        
        report = []
        report.append("="*60)
        report.append(f"📊 WALK-FORWARD VALIDATION SUMMARY - ASSET {self.asset_id}")
        report.append("="*60)
        report.append(f"Totaal aantal inferences: {len(self.all_predictions)}")
        report.append(f"Aantal windows:          {len(self.steps_results)}")
        report.append("-" * 40)
        
        for h in ["1h", "4h", "1d"]:
            acc = final_metrics.get(f'accuracy_{h}', 0)
            dir_acc = final_metrics.get(f'dir_accuracy_{h}', 0)
            brier = final_metrics.get(f'brier_score_{h}', 0)
            
            report.append(f"Horizon {h}:")
            report.append(f"  Accuracy:      {acc:.2%}")
            report.append(f"  Dir Accuracy:  {dir_acc:.2%}")
            report.append(f"  Brier Score:   {brier:.4f}")
            
            # Prediction Distribution
            preds = [p[f'prediction_{h}'] for p in self.all_predictions]
            dist = pd.Series(preds).value_counts(normalize=True).sort_index()
            
            report.append("  Distributie:")
            for state, pct in dist.items():
                bar = "█" * int(pct * 20)
                report.append(f"    {state:<15}: {pct:>6.1%} {bar}")
            report.append("")
            
        # v3 Intermediate Nodes
        report.append("v3 INTERMEDIATE NODES DISTRIBUTIONS:")
        for node in ['trade_hypothesis', 'entry_confidence', 'position_confidence']:
            states = [p[node] for p in self.all_predictions if node in p]
            if states:
                dist = pd.Series(states).value_counts(normalize=True).sort_index()
                report.append(f"  {node.replace('_', ' ').title()}:")
                for state, pct in dist.items():
                    bar = "▒" * int(pct * 20)
                report.append(f"    {state:<15}: {pct:>6.1%} {bar}")
            report.append("")
            
        report.append("="*60)
        return "\n".join(report)

    def _save_report(self):
        """Sla resultaten op naar disk (JSON en Markdown)."""
        log_dir = self.output_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = log_dir / f"wf_report_asset_{self.asset_id}_{timestamp}.json"
        md_filename = log_dir / f"wf_report_asset_{self.asset_id}_{timestamp}.md"
        
        summary_metrics = self._calculate_metrics(self.all_predictions)
        
        # 1. Save JSON
        data = {
            'asset_id': self.asset_id,
            'timestamp': timestamp,
            'params': self.params,
            'summary': summary_metrics,
            'steps': self.steps_results
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=str)
            
        # 2. Save Markdown
        report_text = self.summary_report()
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Walk-Forward Validation Report\n\n")
            f.write(f"**Asset ID:** {self.asset_id}\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
            
            f.write("## Parameters\n")
            f.write(f"- **Start Date:** {self.params.get('start_date')}\n")
            f.write(f"- **End Date:** {self.params.get('end_date')}\n")
            f.write(f"- **Train Window:** {self.params.get('train_window_days')} days\n")
            f.write(f"- **Test Step:** {self.params.get('test_step_days')} days\n")
            f.write(f"- **ATR Thresholds:** {self.params.get('atr_thresholds')}\n\n")
            
            f.write("### Signal Weights (1h/4h/1d)\n")
            f.write("| Signal | 1h | 4h | 1d |\n")
            f.write("|--------|----|----|----|\n")
            for sig, weights in self.params.get('signal_weights', {}).items():
                f.write(f"| {sig} | {weights.get('1h', 1.0)} | {weights.get('4h', 1.0)} | {weights.get('1d', 1.0)} |\n")
            f.write("\n")
            
            f.write("## Summary Metrics\n")
            f.write("```text\n")
            f.write(report_text)
            f.write("\n```\n\n")
            
            f.write("## Step Details\n")
            f.write("| Window | Acc (1h) | Dir Acc (1h) | Brier (1h) | Inferences |\n")
            f.write("|--------|----------|--------------|------------|------------|\n")
            for step in self.steps_results:
                m = step['metrics']
                f.write(f"| {step['window']} | {m.get('accuracy_1h', 0):.2%} | {m.get('dir_accuracy_1h', 0):.2%} | {m.get('brier_score_1h', 0):.4f} | {step['count']} |\n")
        
        logger.info(f"💾 Rapporten opgeslagen naar {json_filename} en {md_filename}")

    def _process_backtest_tick(self, idx: int, all_data: pd.DataFrame, batch_results: Dict):
        """
        Process een single tick in backtest mode.
        
        Args:
            idx: Index in all_data DataFrame
            all_data: Volledige dataset
            batch_results: GPU batch inference results
        """
        row = all_data.loc[idx]
        current_time = row['time_1']
        current_price = row['close_60']
        atr = row.get('atr_14', row.get('atr', 20.0))  # Fallback voor oude data
        
        # Get current raw composite scores for this tick
        current_scores = {
            'leading': float(batch_results['raw_composite_scores']['leading'][idx]),
            'coincident': float(batch_results['raw_composite_scores']['coincident'][idx]),
            'confirming': float(batch_results['raw_composite_scores']['confirming'][idx]),
        }
        
        # --- Position-side inference voor open trades (v3.4) ---
        # REASON: Bereken echte sub-predictions op basis van delta sinds entry
        momentum_prediction = "neutral"
        volatility_regime = "normal"
        exit_timing = "hold"
        position_confidence = "neutral"
        
        if self.trade_simulator.open_trades:
            trade = self.trade_simulator.open_trades[0]  # We hebben max 1 open trade
            
            # Bereken position-side inference als we entry scores hebben
            if trade.entry_composite_scores:
                # Time since entry
                time_since_entry = (current_time - trade.entry_timestamp).total_seconds() / 60.0
                
                # PnL in ATR units
                if trade.direction == 'long':
                    pnl_price = current_price - trade.entry_price
                else:
                    pnl_price = trade.entry_price - current_price
                pnl_atr = pnl_price / trade.atr_at_entry if trade.atr_at_entry > 0 else 0.0
                
                # Call GPU engine for position-side inference
                pos_result = self.engine_gpu.infer_position(
                    current_scores=current_scores,
                    entry_scores=trade.entry_composite_scores,
                    time_since_entry_min=time_since_entry,
                    current_pnl_atr=pnl_atr
                )
                
                momentum_prediction = pos_result['momentum_prediction']
                volatility_regime = pos_result['volatility_regime']
                exit_timing = pos_result['exit_timing']
                position_confidence = pos_result['position_confidence']
        
        # Construct DualInferenceResult from batch results
        inference_result = DualInferenceResult(
            asset_id=self.asset_id,
            timestamp=current_time,
            regime=batch_results['regime'][idx],
            trade_hypothesis=batch_results['trade_hypothesis'][idx],
            leading_composite=batch_results['leading_composite'][idx],
            coincident_composite=batch_results['coincident_composite'][idx],
            confirming_composite=batch_results['confirming_composite'][idx],
            # Entry predictions from GPU batch results
            entry_predictions={
                '1h': batch_results['predictions']['1h']['states'][idx],
                '4h': batch_results['predictions']['4h']['states'][idx],
                '1d': batch_results['predictions']['1d']['states'][idx],
            },
            entry_distributions={
                '1h': batch_results['predictions']['1h']['distributions'][idx],
                '4h': batch_results['predictions']['4h']['distributions'][idx],
                '1d': batch_results['predictions']['1d']['distributions'][idx],
            },
            # v3.4: Real sub-predictions from position-side inference
            momentum_prediction=momentum_prediction,
            volatility_regime=volatility_regime,
            exit_timing=exit_timing,
            position_confidence=position_confidence,
        )
        
        # Check voor entry (alleen als geen open trades)
        if not self.trade_simulator.open_trades:
            if self.trade_simulator.should_enter_trade(inference_result):
                # v3.4: Pass current composite scores for delta calculation
                self.trade_simulator.open_trade(
                    inference_result, 
                    current_price, 
                    atr, 
                    entry_composite_scores=current_scores
                )
        
        # Update open trades
        if self.trade_simulator.open_trades:
            # Fetch 1m OHLC voor deze candle (60m period)
            ohlc_1m = self.trade_simulator.get_ohlc_1m(
                current_time - timedelta(hours=1), 
                current_time
            )
            self.trade_simulator.update_open_trades(current_time, ohlc_1m, inference_result)
    
    def _save_backtest_report(self):
        """Sla backtest resultaten op naar database en disk."""
        import uuid
        from datetime import datetime
        
        if not self.trade_simulator:
            return
        
        # REASON: Gebruik de door TSEM geregistreerde backtest_id, genereer alleen als fallback
        if self.backtest_id:
            backtest_id = self.backtest_id
        else:
            # Fallback voor standalone validatie runs (niet via TSEM)
            backtest_id = f"bt_{self.asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Bereken metrics
        metrics = self.trade_simulator.get_metrics()
        
        # REASON: Converteer alle metrics/materialen naar standaard Python types
        metrics = self._normalize_value(metrics)

        # 1. UPDATE bestaande backtest_runs record met resultaten
        # REASON: TSEM heeft al een record geïnsert met status='pending', we updaten met resultaten
        with get_cursor() as cur:
            cur.execute("""
                UPDATE qbn.backtest_runs SET
                    final_capital_usd = %s,
                    total_pnl_usd = %s,
                    total_pnl_pct = %s,
                    total_trades = %s,
                    winning_trades = %s,
                    losing_trades = %s,
                    breakeven_trades = %s,
                    win_rate_pct = %s,
                    sharpe_ratio = %s,
                    sortino_ratio = %s,
                    max_drawdown_pct = %s,
                    max_drawdown_usd = %s,
                    profit_factor = %s,
                    avg_win_usd = %s,
                    avg_loss_usd = %s,
                    avg_trade_duration_hours = %s,
                    status = 'completed',
                    completed_at = NOW()
                WHERE backtest_id = %s
            """, (
                metrics.get('final_capital_usd'),
                metrics.get('total_pnl_usd'),
                metrics.get('total_pnl_pct'),
                metrics.get('total_trades'),
                metrics.get('winning_trades'),
                metrics.get('losing_trades'),
                metrics.get('breakeven_trades'),
                metrics.get('win_rate_pct'),
                metrics.get('sharpe_ratio'),
                metrics.get('sortino_ratio'),
                metrics.get('max_drawdown_pct'),
                metrics.get('max_drawdown_usd'),
                metrics.get('profit_factor'),
                metrics.get('avg_win_usd'),
                metrics.get('avg_loss_usd'),
                metrics.get('avg_trade_duration_hours'),
                backtest_id
            ))
        
        # 2. Schrijf trades naar qbn.backtest_trades
        for trade in self.trade_simulator.closed_trades:
            with get_cursor() as cur:
                cur.execute("""
                    INSERT INTO qbn.backtest_trades (
                        backtest_id, asset_id, signal_timestamp, direction,
                        entry_timestamp, entry_price, position_size_usd, position_size_units,
                        entry_fees_usd, entry_slippage_pct,
                        htf_regime, trade_hypothesis, momentum_prediction, volatility_regime,
                        exit_timing, position_confidence, leading_composite, coincident_composite,
                        confirming_composite,
                        planned_stop_loss, planned_take_profit, atr_at_entry,
                        exit_timestamp, exit_price, exit_reason, exit_fees_usd,
                        gross_pnl_usd, net_pnl_usd, pnl_pct,
                        mae_pct, mfe_pct, holding_duration_hours,
                        trailing_stop_activated, trailing_stop_highest_price, trailing_stop_lowest_price
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    backtest_id, self.asset_id, trade.signal_timestamp, trade.direction,
                    trade.entry_timestamp, self._normalize_value(trade.entry_price), self._normalize_value(trade.position_size_usd),
                    self._normalize_value(trade.position_size_units), self._normalize_value(trade.entry_fees_usd), self._normalize_value(trade.entry_slippage_pct),
                    trade.htf_regime, trade.trade_hypothesis, trade.momentum_prediction,
                    trade.volatility_regime, trade.exit_timing, trade.position_confidence,
                    trade.leading_composite, trade.coincident_composite, trade.confirming_composite,
                    self._normalize_value(trade.planned_stop_loss), self._normalize_value(trade.planned_take_profit), self._normalize_value(trade.atr_at_entry),
                    trade.exit_timestamp, self._normalize_value(trade.exit_price), trade.exit_reason, self._normalize_value(trade.exit_fees_usd),
                    self._normalize_value(trade.gross_pnl_usd), self._normalize_value(trade.net_pnl_usd), self._normalize_value(trade.pnl_pct),
                    self._normalize_value(trade.mae_pct), self._normalize_value(trade.mfe_pct), self._normalize_value(trade.holding_duration_hours),
                    trade.trailing_stop_activated, self._normalize_value(trade.trailing_stop_highest_price),
                    self._normalize_value(trade.trailing_stop_lowest_price)
                ))
        
        # 3. Schrijf markdown rapport
        log_dir = self.output_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = log_dir / f"backtest_report_asset_{self.asset_id}_{timestamp}.md"
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Backtest Report - {backtest_id}\n\n")
            f.write(f"**Asset ID:** {self.asset_id}\n")
            f.write(f"**Period:** {self.backtest_config.start_date.date()} to {self.backtest_config.end_date.date()}\n")
            f.write(f"**Initial Capital:** ${self.backtest_config.initial_capital_usd:,.2f}\n")
            f.write(f"**Leverage:** {self.backtest_config.leverage}x\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Final Capital | ${metrics.get('final_capital_usd', 0):,.2f} |\n")
            f.write(f"| Total PnL | ${metrics.get('total_pnl_usd', 0):,.2f} ({metrics.get('total_pnl_pct', 0):.2f}%) |\n")
            f.write(f"| Total Trades | {metrics.get('total_trades', 0)} |\n")
            f.write(f"| Win Rate | {metrics.get('win_rate_pct', 0):.2f}% |\n")
            f.write(f"| Profit Factor | {metrics.get('profit_factor', 0):.2f} |\n")
            f.write(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |\n")
            f.write(f"| Max Drawdown | {metrics.get('max_drawdown_pct', 0):.2f}% (${metrics.get('max_drawdown_usd', 0):,.2f}) |\n")
            f.write(f"| Avg Win | ${metrics.get('avg_win_usd', 0):.2f} |\n")
            f.write(f"| Avg Loss | ${metrics.get('avg_loss_usd', 0):.2f} |\n")
            f.write(f"| Avg Duration | {metrics.get('avg_trade_duration_hours', 0):.1f}h |\n\n")
            
            f.write("## Trade History\n\n")
            f.write("| Entry | Exit | Direction | Entry Price | Exit Price | PnL | PnL % | Duration | Exit Reason |\n")
            f.write("|-------|------|-----------|-------------|------------|-----|-------|----------|-------------|\n")
            for trade in self.trade_simulator.closed_trades:
                f.write(f"| {trade.entry_timestamp.strftime('%Y-%m-%d %H:%M')} | "
                       f"{trade.exit_timestamp.strftime('%Y-%m-%d %H:%M')} | "
                       f"{trade.direction.upper()} | ${trade.entry_price:.2f} | ${trade.exit_price:.2f} | "
                       f"${trade.net_pnl_usd:.2f} | {trade.pnl_pct:.2f}% | "
                       f"{trade.holding_duration_hours:.1f}h | {trade.exit_reason} |\n")
            
            f.write(f"\n## Configuration\n\n")
            f.write(f"```json\n{json.dumps(self.backtest_config.to_dict(), indent=2, default=str)}\n```\n")
        
        logger.info(f"✅ Backtest rapport opgeslagen: {md_filename}")
        logger.info(f"✅ Backtest ID: {backtest_id}")
        logger.info(f"📊 Total PnL: ${metrics.get('total_pnl_usd', 0):,.2f} ({metrics.get('total_pnl_pct', 0):.2f}%)")
        logger.info(f"📈 Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")

    def _normalize_value(self, value):
        """Normalize numpy/decimal scalars to plain Python."""
        if isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, np.generic):
            return float(value)
        if isinstance(value, Decimal):
            return float(value)
        return value
