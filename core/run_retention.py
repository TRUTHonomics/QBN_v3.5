"""
Run Retention: bewaar alleen de N meest recente runs per asset.

REASON: Voorkomt onbeperkte data-accumulatie in run-scoped tabellen.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def retain_recent_runs(
    conn,
    table: str,
    asset_id: int,
    n: int = 3,
    timestamp_col: str = "updated_at"
) -> int:
    """
    Verwijder alle runs behalve de N meest recente voor een asset.
    
    Args:
        conn: psycopg2 connection object
        table: Volledig qualified table name (bijv. 'qbn.signal_weights')
        asset_id: Asset ID om te filteren
        n: Aantal runs om te behouden (default 3)
        timestamp_col: Kolom naam voor timestamp (default 'updated_at')
        
    Returns:
        Aantal verwijderde rows
        
    REASON: Query gebruikt CTE om eerst de N meest recente run_id's te identificeren,
    vervolgens ALLEEN die runs te behouden. De WHERE run_id IS NOT NULL guard
    voorkomt dat legacy data zonder run_id wordt verwijderd.
    """
    sql = f"""
        WITH recent_runs AS (
            SELECT DISTINCT run_id, MAX({timestamp_col}) as last_ts
            FROM {table}
            WHERE asset_id = %s AND run_id IS NOT NULL
            GROUP BY run_id
            ORDER BY last_ts DESC
            LIMIT %s
        )
        DELETE FROM {table}
        WHERE asset_id = %s
          AND run_id IS NOT NULL
          AND run_id NOT IN (SELECT run_id FROM recent_runs)
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (asset_id, n, asset_id))
        deleted = cur.rowcount
        conn.commit()
    
    if deleted > 0:
        logger.info(f"🗑️  Retained {n} most recent runs in {table} for asset {asset_id}, deleted {deleted} rows from older runs")
    
    return deleted


# Timestamp kolommen per tabel
TIMESTAMP_COLUMNS = {
    "qbn.composite_threshold_config": "updated_at",
    "qbn.signal_weights": "last_trained_at",
    "qbn.combination_alpha": "analyzed_at",
    "qbn.event_windows": "created_at",
    "qbn.position_delta_threshold_config": "updated_at",
    "qbn.cpt_cache": "generated_at",
}


def retain_recent_runs_auto(
    conn,
    table: str,
    asset_id: int,
    n: int = 3
) -> int:
    """
    Convenience functie die automatisch de juiste timestamp kolom kiest.
    
    REASON: Scripts hoeven niet handmatig de timestamp kolom op te zoeken.
    """
    timestamp_col = TIMESTAMP_COLUMNS.get(table, "updated_at")
    return retain_recent_runs(conn, table, asset_id, n, timestamp_col)
