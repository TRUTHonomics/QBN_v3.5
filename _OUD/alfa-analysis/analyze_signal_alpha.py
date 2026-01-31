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

# REASON: Data-driven signaal analyse voor gewichtsbepaling.
# Gebruikt Mutual Information (MI) en Hit Rate om voorspellingskracht te meten.
# Inclusief Odd/Even folding voor stabiliteitscontrole en OOS validatie op 2025 data.

# Logger setup
def setup_logging(asset_id: int):
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_log')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'alpha_analysis_asset_{asset_id}_{timestamp}.log')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return log_file

ASSET_ID = 1  # Bitcoin PoC
HORIZONS = ['1h', '4h', '1d']
CATALOG_PATH = os.path.join(os.path.dirname(__file__), 'signal_catalog.csv')
WEIGHTS_OUTPUT_YAML = os.path.join(os.path.dirname(__file__), 'signal_weights.yaml')

class SignalAlphaAnalyzer:
    def __init__(self, asset_id: int = ASSET_ID):
        self.asset_id = asset_id
        self.catalog = pd.read_csv(CATALOG_PATH)
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
        setup_logging(asset_id)
        logging.info(f"SignalAlphaAnalyzer initialized for asset {asset_id}")

    def fetch_data(self):
        """
        Haalt data op in batches per tabel en voegt ze samen in de container.
        REASON: Offloading DB van complexe joins op hypertables.
        """
        import time
        start_fetch = time.time()
        logging.info(f"Starting modular data fetch for asset {self.asset_id}...")

        # 1. Fetch Outcomes (Fundament)
        logging.info("Fetching outcomes from qbn.signal_outcomes...")
        outcome_query = f"SELECT time_1, outcome_1h, outcome_4h, outcome_1d FROM qbn.signal_outcomes WHERE asset_id = {self.asset_id}"
        with get_cursor() as cur:
            cur.execute(outcome_query)
            df_final = pd.DataFrame(cur.fetchall(), columns=['time_1', 'outcome_1h', 'outcome_4h', 'outcome_1d'])
        
        logging.info(f"Fetched {len(df_final)} outcomes.")
        if df_final.empty:
            logging.error("No outcomes found! Run outcome backfill first.")
            return

        # 2. Fetch Signals per Category (Modular joins in Python)
        categories = {
            'lead': 'mtf_signals_lead',
            'coin': 'mtf_signals_coin',
            'conf': 'mtf_signals_conf'
        }

        for cat_key, table_name in categories.items():
            sig_cols = self.catalog[self.catalog['table_name'] == table_name]['column_name'].tolist()
            if not sig_cols:
                continue
            
            logging.info(f"Fetching {len(sig_cols)} {cat_key} signals from kfl.{table_name}...")
            # We doen dit in chunks om de database connectie niet te lang open te houden
            # maar voor één asset is een enkele fetch meestal veilig mits geen joins.
            sig_query = f"SELECT time_1, {', '.join(sig_cols)} FROM kfl.{table_name} WHERE asset_id = {self.asset_id}"
            
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

    def calculate_mi(self, signal_values, outcomes):
        """Bereken Mutual Information score."""
        # REASON: Filter zowel NaN in signalen als in outcomes om sklearn ValueError te voorkomen.
        mask = (outcomes != -99) & ~np.isnan(outcomes) & ~np.isnan(signal_values)
        if mask.sum() < 100: return 0.0
        return float(mutual_info_score(signal_values[mask], outcomes[mask]))

    def calculate_hit_rate(self, signal_values, outcomes, polarity):
        """Bereken Hit Rate gebaseerd op signaal polariteit."""
        if polarity == 0: return 0.5 # Neutrale signalen hebben geen hit rate in die zin
        
        # REASON: Filter zowel NaN in signalen als in outcomes.
        mask = (outcomes != -99) & ~np.isnan(outcomes) & (signal_values != 0) & ~np.isnan(signal_values)
        if mask.sum() < 50: return 0.5
        
        # Voorspelde richting: signaal_waarde * polariteit (1 of -1)
        predicted_dir = np.sign(signal_values[mask]) * polarity
        actual_dir = np.sign(outcomes[mask])
        
        # Hit rate is % waar richtingen matchen (let op: outcome 0 wordt als mismatch gezien bij signaal != 0)
        return float((predicted_dir == actual_dir).mean())

    def run_analysis(self):
        """Voert de volledige alfa-analyse uit per signaal en horizon."""
        logging.info("Starting signal analysis loop...")
        df_odd, df_even, df_oos = self.apply_folding()
        
        all_mi_scores = [] # Voor normalisatie
        total_signals = len(self.catalog)
        
        for i, (_, sig_info) in enumerate(self.catalog.iterrows()):
            sig_col = sig_info['column_name']
            polarity = sig_info['polarity']
            semantic_class = sig_info.get('semantic_class', 'UNKNOWN')
            
            if i % 50 == 0:
                logging.info(f"Progress: {i}/{total_signals} signals analyzed...")
            
            for horizon in HORIZONS:
                outcome_col = f"outcome_{horizon}"
                
                # Bereken metrieken op Sets A en B
                mi_odd = self.calculate_mi(df_odd[sig_col].values, df_odd[outcome_col].values)
                mi_even = self.calculate_mi(df_even[sig_col].values, df_even[outcome_col].values)
                
                hr_odd = self.calculate_hit_rate(df_odd[sig_col].values, df_odd[outcome_col].values, polarity)
                hr_even = self.calculate_hit_rate(df_even[sig_col].values, df_even[outcome_col].values, polarity)
                
                # Stabiliteit (Symmetrie)
                stability = 1.0 - abs(mi_odd - mi_even) / max(mi_odd, mi_even, 1e-9)
                stability = max(0.1, stability)
                
                # OOS Performance
                mi_oos = self.calculate_mi(df_oos[sig_col].values, df_oos[outcome_col].values)
                hr_oos = self.calculate_hit_rate(df_oos[sig_col].values, df_oos[outcome_col].values, polarity)
                
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
        
        # Finale gewichtsberekening via normalisatie
        mi_median = np.median([r['mi_avg'] for r in self.results])
        if mi_median == 0: mi_median = 1e-6
        
        for res in self.results:
            # Gewicht = (MI / Median MI) * Stability
            # REASON: MI meet afhankelijkheid, Stability straft wisselvalligheid af.
            if res['mi_avg'] == 0:
                raw_weight = 0.0
            else:
                raw_weight = (res['mi_avg'] / mi_median) * res['stability']
            
            # REASON: Rare signals (MI=0) krijgen gewicht 0.0 om het BN te ontlasten.
            res['weight'] = float(np.clip(raw_weight, 0.0, 2.5))
            
        logging.info(f"Analysis complete for {len(self.catalog)} signals across {len(HORIZONS)} horizons.")

    def save_results(self):
        """Slaat resultaten op in Database en YAML."""
        logging.info("Saving results to database...")
        
        # Voorbereiden voor bulk insert
        insert_query = """
        INSERT INTO qbn.signal_weights (
            signal_name, horizon, semantic_class, weight, mutual_information, 
            hit_rate, stability_score, oos_performance, 
            train_start, train_end, oos_start, oos_end, total_rows,
            last_trained_at
        ) VALUES %s
        ON CONFLICT (signal_name, horizon) DO UPDATE SET
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
            last_trained_at = NOW();
        """
        
        from psycopg2.extras import execute_values
        
        # REASON: Converteer numpy types naar Python native types voor psycopg2 compatibiliteit
        data_to_insert = [
            (
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
                datetime.now()
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
    parser = argparse.ArgumentParser(description='Signal Alpha Analysis for QBN v2')
    parser.add_argument('--asset', type=int, default=ASSET_ID, help='Asset ID for analysis (default: 1)')
    args = parser.parse_args()

    analyzer = SignalAlphaAnalyzer(asset_id=args.asset)
    analyzer.fetch_data()
    analyzer.run_analysis()
    analyzer.save_results()
