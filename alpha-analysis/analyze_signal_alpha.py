import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yaml
from sklearn.metrics import mutual_info_score
import io

# Voeg project root toe aan path voor database modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db import get_cursor
from core.logging_utils import setup_logging

# REASON: Data-driven signaal analyse voor gewichtsbepaling.
# Gebruikt Mutual Information (MI) en Hit Rate om voorspellingskracht te meten.
# Inclusief Odd/Even folding voor stabiliteitscontrole en OOS validatie op 2025 data.

ASSET_ID = 1  # Bitcoin PoC
HORIZONS = ['1h', '4h', '1d']
WEIGHTS_OUTPUT_YAML = os.path.join(os.path.dirname(__file__), 'signal_weights.yaml')

class SignalAlphaAnalyzer:
    def __init__(self, asset_id: int = ASSET_ID, layer: str = 'HYPOTHESIS', run_id: str = None):
        self.asset_id = asset_id
        self.layer = layer.upper() # HYPOTHESIS of CONFIDENCE
        self.run_id = run_id
        # REASON: Haal catalogus direct uit de database in plaats van uit een CSV
        self.catalog = self._load_catalog_from_db()
        self.df = None
        self.results = []
        # REASON: Metadata voor reproduceerbaarheid
        self.metadata = {
            'total_rows': 0,
            'train_start': None,
            'train_end': None,
            'oos_start': None,
            'oos_end': None
        }
        setup_logging("analyze_signal_alpha")
        logging.info(f"SignalAlphaAnalyzer initialized for asset {asset_id} | Layer: {self.layer}")

    def _load_catalog_from_db(self) -> pd.DataFrame:
        """
        Bouwt de signal catalogus dynamisch op vanuit qbn.signal_classification.
        REASON: Centralisatie van metadata in de database.
        
        FILTER: In HYPOTHESIS mode laden we alleen LEADING signals.
               In CONFIDENCE mode laden we COINCIDENT en CONFIRMING signals.
        """
        logging.info(f"Loading signal catalog from qbn.signal_classification for layer {self.layer}...")
        
        if self.layer == 'HYPOTHESIS':
            filter_clause = "semantic_class = 'LEADING'"
        else:
            filter_clause = "semantic_class IN ('COINCIDENT', 'CONFIRMING')"
            
        query = f"""
            SELECT signal_name, semantic_class, polarity 
            FROM qbn.signal_classification
            WHERE {filter_clause}
        """
        
        with get_cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            
        data = []
        for row in rows:
            # REASON: row is een tuple, gebruik indices in plaats van kolomnamen
            sig_name = row[0].lower()
            sem_class = row[1]
            pol_str = row[2]
            
            # Mapping naar fysieke tabel naam
            table_map = {
                'LEADING': 'mtf_signals_lead',
                'COINCIDENT': 'mtf_signals_coin',
                'CONFIRMING': 'mtf_signals_conf'
            }
            table_name = table_map.get(sem_class, 'unknown')
            
            # Mapping van polarity naar integer
            pol_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
            polarity = pol_map.get(pol_str, 0)
            
            # Voeg _60 suffix toe voor de Tactical Layer
            column_name = f"{sig_name}_60"
            
            data.append({
                'column_name': column_name,
                'table_name': table_name,
                'semantic_class': sem_class,
                'polarity': polarity
            })
            
        df = pd.DataFrame(data)
        logging.info(f"Successfully loaded {len(df)} signals from database for Tactical Layer (60m).")
        return df

    def fetch_data(self):
        """
        Haalt data op uit qbn.barrier_outcomes en berekent uniqueness gewichten (1/N).
        REASON: Migratie naar First-Touch Barriers en voorkomen van Double Counting.
        """
        import time
        start_fetch = time.time()
        logging.info(f"Starting modular data fetch with uniqueness weighting for asset {self.asset_id}...")

        # 1. Fetch Outcomes (Fundament) - Gefilterd op 60m boundaries
        logging.info("Fetching outcomes from qbn.barrier_outcomes (60m boundary)...")
        outcome_query = f"""
            SELECT o.time_1, o.first_significant_barrier, o.first_significant_time_min, o.asset_id
            FROM qbn.barrier_outcomes o
            JOIN kfl.indicators i ON o.asset_id = i.asset_id AND o.time_1 = i.time
            WHERE o.asset_id = {self.asset_id} AND i.interval_min = '60'
        """
        with get_cursor() as cur:
            cur.execute(outcome_query)
            df_final = pd.DataFrame(cur.fetchall(), columns=['time_1', 'first_significant_barrier', 'first_significant_time_min', 'asset_id'])
        
        if df_final.empty:
            logging.error("No outcomes found for 60m boundaries in qbn.barrier_outcomes!")
            return False

        # 1b. Bepaal horizon-specifieke outcomes en Uniqueness
        df_final['time_1'] = pd.to_datetime(df_final['time_1'])
        df_final['hit_timestamp'] = df_final['time_1'] + pd.to_timedelta(df_final['first_significant_time_min'], unit='m')
        
        # Uniqueness (1/N)
        mask_hit = (df_final['first_significant_barrier'] != 'none') & (df_final['first_significant_barrier'].notna())
        df_final['uniqueness_weight'] = 1.0
        
        if mask_hit.any():
            # Clusteren op asset, barrier type en de exacte seconde dat de hit plaatsvond
            counts = df_final[mask_hit].groupby(['asset_id', 'first_significant_barrier', 'hit_timestamp']).size().reset_index(name='cluster_n')
            df_final = df_final.merge(counts, on=['asset_id', 'first_significant_barrier', 'hit_timestamp'], how='left')
            df_final.loc[mask_hit, 'uniqueness_weight'] = 1.0 / df_final.loc[mask_hit, 'cluster_n']
            df_final.drop(columns=['cluster_n'], inplace=True)
            
            n_clusters = len(counts)
            avg_uniqueness = df_final.loc[mask_hit, 'uniqueness_weight'].mean()
            logging.info(f"Clustered {mask_hit.sum()} hits into {n_clusters} unique events. Avg uniqueness: {avg_uniqueness:.3f}")

        # Map naar binaire outcomes voor MI/HitRate analyse per horizon
        for horizon in HORIZONS:
            h_min = 60 if horizon == '1h' else (240 if horizon == '4h' else 1440)
            col = f'outcome_{horizon}'
            df_final[col] = 0
            
            is_up = df_final['first_significant_barrier'].str.startswith('up', na=False)
            is_down = df_final['first_significant_barrier'].str.startswith('down', na=False)
            is_within = df_final['first_significant_time_min'] <= h_min
            
            df_final.loc[is_up & is_within, col] = 1
            df_final.loc[is_down & is_within, col] = -1

        logging.info(f"Fetched {len(df_final)} unique 60m outcomes mapped to {HORIZONS} horizons.")

        # 2. Fetch Signals per Category (Modular joins in Python)
        categories = {
            'lead': 'mtf_signals_lead',
            'coin': 'mtf_signals_coin',
            'conf': 'mtf_signals_conf'
        }

        for cat_key, table_name in categories.items():
            # REASON: Gebruik alleen uursignalen (_60) voor de Tactical Layer om vertroebeling te voorkomen
            sig_cols = [c for c in self.catalog[self.catalog['table_name'] == table_name]['column_name'].tolist() if c.endswith('_60')]
            if not sig_cols:
                continue
            
            logging.info(f"Fetching {len(sig_cols)} {cat_key} signals (60m only) from kfl.{table_name}...")
            
            # REASON: Join met indicators om alleen de uursluitingen op te halen
            sig_query = f"""
                SELECT s.time_1, {', '.join(['s.' + c for c in sig_cols])} 
                FROM kfl.{table_name} s
                JOIN kfl.indicators i ON s.asset_id = i.asset_id AND s.time_1 = i.time
                WHERE s.asset_id = {self.asset_id} AND i.interval_min = '60'
            """
            
            with get_cursor() as cur:
                cur.execute(sig_query)
                df_cat = pd.DataFrame(cur.fetchall(), columns=['time_1'] + sig_cols)
            
            logging.info(f"Merging {len(df_cat)} {cat_key} signals...")
            df_final = pd.merge(df_final, df_cat, on='time_1', how='left')
            del df_cat # Vrijmaken geheugen

        self.df = df_final
        self.df['time_1'] = pd.to_datetime(self.df['time_1'])
        self.metadata['total_rows'] = len(self.df)
        
        # Check op ontbrekende data
        missing_counts = self.df[self.catalog['column_name']].isna().sum().sum()
        if missing_counts > 0:
            logging.warning(f"Detected {missing_counts} missing signal values (NaN) after merge.")

        end_fetch = time.time()
        logging.info(f"Data fetching and merging complete. Total rows: {len(self.df)}. Time: {end_fetch - start_fetch:.2f}s")
        return True

    def apply_folding(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Verdeelt data in Set A (Odd), Set B (Even) en OOS (2025)."""
        df = self.df.copy()
        
        # Out-of-sample: 2025
        oos_mask = df['time_1'].dt.year >= 2025
        df_oos = df[oos_mask].copy()
        df_train = df[~oos_mask].copy()
        
        # REASON: Leg periodes vast voor reproduceerbaarheid
        if not df_train.empty:
            self.metadata['train_start'] = df_train['time_1'].min()
            self.metadata['train_end'] = df_train['time_1'].max()
        if not df_oos.empty:
            self.metadata['oos_start'] = df_oos['time_1'].min()
            self.metadata['oos_end'] = df_oos['time_1'].max()

        # Odd/Even Folding (maand nummer)
        df_train['month'] = df_train['time_1'].dt.month
        odd_mask = df_train['month'] % 2 != 0
        df_odd = df_train[odd_mask].copy()
        df_even = df_train[~odd_mask].copy()
        
        # Boundary Protection: verwijder laatste 24u van elke maand om leakage te voorkomen
        def protect_boundaries(data):
            if data.empty: return data
            # Bepaal de laatste timestamp per maand via transform (vectorized)
            # REASON: tz_localize(None) voorkomt UserWarning bij conversie naar PeriodIndex
            max_times = data.groupby(data['time_1'].dt.tz_localize(None).dt.to_period('M'))['time_1'].transform('max')
            # Behoud alleen data die meer dan 24u vóór het einde van die maand ligt
            mask = data['time_1'] <= (max_times - timedelta(hours=24))
            return data[mask].copy()

        df_odd = protect_boundaries(df_odd)
        df_even = protect_boundaries(df_even)
        
        logging.info(f"Split: Odd={len(df_odd)}, Even={len(df_even)}, OOS={len(df_oos)}")
        return df_odd, df_even, df_oos

    def calculate_mi(self, x, y, w=None):
        """
        Bereken Mutual Information score (gewogen).
        REASON: mutual_info_score van sklearn ondersteunt geen gewichten, 
        dus we bouwen zelf de gewogen contingency table.
        """
        mask = (y != -99) & ~np.isnan(y) & ~np.isnan(x)
        if w is not None: mask &= ~np.isnan(w)
        if mask.sum() < 100: return 0.0
        
        x_m, y_m = x[mask], y[mask]
        w_m = w[mask] if w is not None else np.ones(len(x_m))

        # Bouw gewogen contingency table
        labels_x, x_idx = np.unique(x_m, return_inverse=True)
        labels_y, y_idx = np.unique(y_m, return_inverse=True)
        
        contingency = np.zeros((len(labels_x), len(labels_y)))
        # Vectorized weighted add to contingency table
        np.add.at(contingency, (x_idx, y_idx), w_m)
        
        return float(mutual_info_score(None, None, contingency=contingency))

    def calculate_hit_rate(self, signal_values, outcomes, polarity, weights=None):
        """
        Bereken Hit Rate gebaseerd op signaal polariteit (gewogen).
        REASON: Voorkomt dat over-geclusterde trends de alpha score domineren.
        """
        if polarity == 0: return 0.5
        
        mask = (outcomes != -99) & ~np.isnan(outcomes) & (signal_values != 0) & ~np.isnan(signal_values)
        if weights is not None: mask &= ~np.isnan(weights)
        if mask.sum() < 50: return 0.5
        
        predicted_dir = np.sign(signal_values[mask]) * polarity
        actual_dir = np.sign(outcomes[mask])
        w = weights[mask] if weights is not None else np.ones(len(actual_dir))
        
        hits = (predicted_dir == actual_dir).astype(float)
        weighted_hits = np.sum(hits * w)
        total_weight = np.sum(w)
        
        if total_weight == 0: return 0.5
        return float(weighted_hits / total_weight)

    def run_analysis(self):
        """Voert de volledige alfa-analyse uit per signaal en horizon."""
        if self.df is None or self.df.empty:
            logging.error("No data available for analysis. Skipping...")
            return

        # REASON: In HYPOTHESIS mode kijken we alleen naar de 1h horizon (initiële trigger).
        # In CONFIDENCE mode kijken we naar alle horizons voor momentum-bevestiging.
        current_horizons = ['1h'] if self.layer == 'HYPOTHESIS' else HORIZONS
        logging.info(f"Starting uniqueness-weighted signal analysis loop for horizons: {current_horizons}...")
        
        df_odd, df_even, df_oos = self.apply_folding()
        
        all_mi_scores = [] # Voor normalisatie
        total_signals = len(self.catalog)
        
        for i, (_, sig_info) in enumerate(self.catalog.iterrows()):
            sig_col = sig_info['column_name']
            polarity = sig_info['polarity']
            semantic_class = sig_info.get('semantic_class', 'UNKNOWN')
            
            # Skip als signaal niet aanwezig is in de dataset (bijv. missing in kfl tables)
            if sig_col not in self.df.columns:
                continue
            
            if i % 50 == 0:
                logging.info(f"Progress: {i}/{total_signals} signals analyzed...")
            
            for horizon in current_horizons:
                outcome_col = f"outcome_{horizon}"
                
                # Bereken metrieken op Sets A en B met uniqueness_weight
                mi_odd = self.calculate_mi(df_odd[sig_col].values, df_odd[outcome_col].values, df_odd['uniqueness_weight'].values)
                mi_even = self.calculate_mi(df_even[sig_col].values, df_even[outcome_col].values, df_even['uniqueness_weight'].values)
                
                hr_odd = self.calculate_hit_rate(df_odd[sig_col].values, df_odd[outcome_col].values, polarity, df_odd['uniqueness_weight'].values)
                hr_even = self.calculate_hit_rate(df_even[sig_col].values, df_even[outcome_col].values, polarity, df_even['uniqueness_weight'].values)
                
                # Stabiliteit (Symmetrie)
                stability = 1.0 - abs(mi_odd - mi_even) / max(mi_odd, mi_even, 1e-9)
                stability = max(0.1, stability)
                
                # OOS Performance
                mi_oos = self.calculate_mi(df_oos[sig_col].values, df_oos[outcome_col].values, df_oos['uniqueness_weight'].values)
                hr_oos = self.calculate_hit_rate(df_oos[sig_col].values, df_oos[outcome_col].values, polarity, df_oos['uniqueness_weight'].values)
                
                # Gemiddelde MI voor basis gewicht
                mi_avg = (mi_odd + mi_even) / 2.0
                all_mi_scores.append(mi_avg)
                
                self.results.append({
                    'signal_name': sig_col,
                    'horizon': horizon,
                    'semantic_class': semantic_class,
                    'mi_avg': mi_avg,
                    'mi_odd': mi_odd,
                    'mi_even': mi_even,
                    'hr_avg': (hr_odd + hr_even) / 2.0,
                    'stability': stability,
                    'oos_mi': mi_oos,
                    'oos_hr': hr_oos
                })
        
        # Finale gewichtsberekening via Within-Class Normalisatie
        # REASON: Bereken mediaan MI per semantische klasse om te voorkomen dat 
        # sterkere klassen (Coin) de zwakkere klassen (Conf) wegdrukken.
        medians_per_class = {}
        # Gebruik de semantische klassen die in deze run aanwezig zijn
        present_classes = set(r['semantic_class'] for r in self.results)
        
        for sem_class in present_classes:
            class_mi_scores = [r['mi_avg'] for r in self.results if r['semantic_class'] == sem_class]
            if class_mi_scores:
                medians_per_class[sem_class] = np.median(class_mi_scores)
            else:
                medians_per_class[sem_class] = 1e-6

        for res in self.results:
            sem_class = res['semantic_class']
            mi_median = medians_per_class.get(sem_class, 1e-6)
            if mi_median == 0: mi_median = 1e-6
            
            # Gewicht = (MI / Median MI van eigen klasse) * Stability
            if res['mi_avg'] == 0:
                raw_weight = 0.0
            else:
                raw_weight = (res['mi_avg'] / mi_median) * res['stability']
            
            # REASON: Rare signals (MI=0) krijgen gewicht 0.0 om het BN te ontlasten.
            res['weight'] = float(np.clip(raw_weight, 0.0, 2.5))
        
        logging.info(f"Within-class normalization complete for classes: {list(medians_per_class.keys())}")
        logging.info(f"Analysis complete for {len(self.catalog)} signals across {len(HORIZONS)} horizons.")

    def save_results(self):
        """Slaat de alpha-scores op in qbn.signal_weights en een YAML backup."""
        if not self.results:
            logging.warning("No results to save.")
            return

        logging.info(f"Saving {len(self.results)} alpha results to qbn.signal_weights (Layer: {self.layer})...")
        
        insert_query = """
        INSERT INTO qbn.signal_weights (
            asset_id, signal_name, horizon, semantic_class, weight, mutual_information, 
            hit_rate, stability_score, oos_performance, 
            train_start, train_end, oos_start, oos_end, total_rows,
            last_trained_at, layer, run_id
        ) VALUES %s
        ON CONFLICT (asset_id, signal_name, horizon, layer) DO UPDATE SET
            semantic_class = EXCLUDED.semantic_class,
            weight = EXCLUDED.weight,
            mutual_information = EXCLUDED.mutual_information,
            hit_rate = EXCLUDED.hit_rate,
            stability_score = EXCLUDED.stability_score,
            oos_performance = EXCLUDED.oos_performance,
            train_start = EXCLUDED.train_start,
            train_end = EXCLUDED.train_end,
            oos_start = EXCLUDED.oos_start,
            oos_end = EXCLUDED.oos_end,
            total_rows = EXCLUDED.total_rows,
            last_trained_at = NOW(),
            run_id = EXCLUDED.run_id;
        """
        
        from psycopg2.extras import execute_values
        
        # REASON: Converteer numpy types naar Python native types voor psycopg2 compatibiliteit
        data_to_insert = [
            (
                self.asset_id,
                r['signal_name'], 
                r['horizon'], 
                r['semantic_class'],
                float(r['weight']), 
                float(r['mi_avg']),
                float(r['hr_avg']), 
                float(r['stability']), 
                float(r['oos_mi']), 
                self.metadata['train_start'],
                self.metadata['train_end'],
                self.metadata['oos_start'],
                self.metadata['oos_end'],
                int(self.metadata['total_rows']),
                datetime.now(),
                self.layer,
                self.run_id
            )
            for r in self.results
        ]
        
        with get_cursor(commit=True) as cur:
            execute_values(cur, insert_query, data_to_insert)
            
        # YAML Backup
        weights_dict = {}
        for r in self.results:
            if r['signal_name'] not in weights_dict:
                weights_dict[r['signal_name']] = {}
            weights_dict[r['signal_name']][r['horizon']] = r['weight']
            
        with open(WEIGHTS_OUTPUT_YAML, 'w') as f:
            yaml.dump(weights_dict, f)
            
        logging.info(f"Results saved to database and {WEIGHTS_OUTPUT_YAML}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Signal Alpha Analysis for QBN v3')
    parser.add_argument('--asset', type=int, default=ASSET_ID, help='Asset ID for analysis (default: 1)')
    parser.add_argument('--layer', type=str, default='HYPOTHESIS', choices=['HYPOTHESIS', 'CONFIDENCE'], 
                        help='Target layer: HYPOTHESIS (Leading) or CONFIDENCE (Coin/Conf)')
    parser.add_argument('--run-id', type=str, help='Run identifier for traceability')
    args = parser.parse_args()

    analyzer = SignalAlphaAnalyzer(asset_id=args.asset, layer=args.layer, run_id=args.run_id)
    if analyzer.fetch_data():
        analyzer.run_analysis()
        analyzer.save_results()
    else:
        logging.error(f"Analysis aborted for asset {args.asset} due to missing data.")
