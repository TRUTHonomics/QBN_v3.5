import os
import sys
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Voeg project root toe aan path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import get_cursor

def extract_signals():
    """
    Extraheert alle signaalkolommen uit de MTF tabellen en bepaalt de polarity.
    Output: signal_catalog.csv
    """
    tables = ['mtf_signals_lead', 'mtf_signals_coin', 'mtf_signals_conf']
    ignore_cols = [
        'asset_id', 'time_d', 'time_close_d', 'time_240', 'time_close_240',
        'time_60', 'time_close_60', 'time_1', 'time_close_1', 'created_at',
        'source_script', 'concordance_sum', 'concordance_count', 'concordance_score',
        'concordance_sum_d', 'concordance_sum_240', 'concordance_sum_60', 'concordance_sum_1',
        'concordance_count_d', 'concordance_count_240', 'concordance_count_60', 'concordance_count_1',
        'concordance_score_d', 'concordance_score_240', 'concordance_score_60', 'concordance_score_1'
    ]

    all_signals = []

    with get_cursor() as cur:
        for table in tables:
            print(f"Processing table: {table}")
            cur.execute(f"SELECT * FROM kfl.{table} LIMIT 0")
            cols = [desc[0] for desc in cur.description]
            
            semantic_class = table.split('_')[-1].upper() # LEAD -> LEADING, COIN -> COINCIDENT, CONF -> CONFIRMING
            if semantic_class == 'LEAD': semantic_class = 'LEADING'
            elif semantic_class == 'COIN': semantic_class = 'COINCIDENT'
            elif semantic_class == 'CONF': semantic_class = 'CONFIRMING'

            for col in cols:
                if col in ignore_cols:
                    continue
                
                # Bepaal polarity op basis van naam
                polarity = 1 # Default bullish/positive correlation
                name_lower = col.lower()
                
                if any(x in name_lower for x in ['bear', 'short', 'overbought', 'resistance', 'down']):
                    polarity = -1
                elif any(x in name_lower for x in ['bull', 'long', 'oversold', 'support', 'up']):
                    polarity = 1
                
                # Speciale gevallen
                if 'divergence' in name_lower:
                    if 'bear' in name_lower: polarity = -1
                    else: polarity = 1
                
                all_signals.append({
                    'signal_name': col,
                    'table_name': f"kfl.{table}",
                    'semantic_class': semantic_class,
                    'polarity': polarity
                })

    df = pd.DataFrame(all_signals)
    output_path = os.path.join(os.path.dirname(__file__), 'signal_catalog.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Signal catalog opgeslagen naar {output_path} ({len(df)} signalen)")

if __name__ == "__main__":
    extract_signals()

