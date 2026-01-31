#!/usr/bin/env python3
"""Check actual column names in qbn.ml_multi_timeframe_signals table."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor

def check_table_columns():
    """Query the actual column names from information_schema."""
    query = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'qbn'
      AND table_name = 'ml_multi_timeframe_signals'
    ORDER BY ordinal_position;
    """

    try:
        with get_cursor() as cur:
            cur.execute(query)
            columns = cur.fetchall()

            print("=== qbn.ml_multi_timeframe_signals columns ===")
            for col_name, data_type, nullable in columns:
                null_info = "NOT NULL" if nullable == "NO" else "NULL"
                print(f"{col_name}: {data_type} {null_info}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_table_columns()








