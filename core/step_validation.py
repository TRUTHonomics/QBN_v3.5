"""
Pre-validatie guards voor pipeline stappen.

REASON: Verifieert dat upstream data aanwezig is voordat een stap start.
Voorkomt silent failures waarbij een stap draait met oude/verkeerde data.
"""
from __future__ import annotations

import logging
from typing import Optional

import psycopg2
from psycopg2.extensions import connection as Connection

logger = logging.getLogger(__name__)


class StepValidationError(Exception):
    """Upstream data ontbreekt of komt van verkeerde run."""
    pass


def validate_step_input(
    conn: Connection,
    step_name: str,
    upstream_table: str,
    asset_id: int,
    run_id: Optional[str] = None,
    min_rows: int = 1,
    extra_where: str = "",
    log_run_id: Optional[str] = None,
) -> int:
    """
    Verifieert dat upstream tabel data bevat voor dit asset/run_id.
    
    Args:
        conn: PostgreSQL connection
        step_name: Naam van de huidige pipeline stap
        upstream_table: Schema-qualified tabel naam (bijv. 'qbn.barrier_outcomes')
        asset_id: Asset ID
        run_id: Optionele run_id filter (voor WHERE clause)
        min_rows: Minimaal verwacht aantal rijen
        extra_where: Extra WHERE conditions (bijv. 'leading_score IS NOT NULL')
        log_run_id: Override voor logging (als run_id None is maar we willen wel echte run_id loggen)
    
    Returns:
        Aantal gevonden rijen
    
    Raises:
        StepValidationError: Als niet genoeg data aanwezig is
    """
    where_parts = [f"asset_id = {asset_id}"]
    
    if run_id:
        where_parts.append(f"run_id = '{run_id}'")
    
    if extra_where:
        where_parts.append(f"({extra_where})")
    
    where_clause = " AND ".join(where_parts)
    sql = f"SELECT COUNT(*) FROM {upstream_table} WHERE {where_clause}"
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            count = cur.fetchone()[0]
    except psycopg2.Error as e:
        raise StepValidationError(
            f"Failed to validate {upstream_table} for step {step_name}: {e}"
        ) from e
    
    # Log handshake (gebruik log_run_id als override, anders run_id, anders "N/A")
    log_handshake_in(
        step=step_name,
        source=upstream_table,
        run_id=log_run_id or run_id or "N/A",
        rows=count,
        filter_clause=extra_where or "none",
    )
    
    if count < min_rows:
        raise StepValidationError(
            f"Step {step_name}: insufficient data in {upstream_table}. "
            f"Expected >= {min_rows}, found {count} (asset_id={asset_id}, run_id={run_id})"
        )
    
    return count


def log_handshake_in(
    step: str,
    source: str,
    run_id: str,
    rows: int,
    filter_clause: str = "",
) -> None:
    """
    Logt HANDSHAKE_IN: welke data is gelezen door deze stap.
    
    Format: HANDSHAKE_IN | step=X | run_id=Y | source=Z | rows=N | filter=...
    """
    msg = f"HANDSHAKE_IN | step={step} | run_id={run_id} | source={source} | rows={rows}"
    if filter_clause:
        msg += f" | filter={filter_clause}"
    
    logger.info(msg)
    print(msg, flush=True)  # Ook naar stdout voor Dagster parsing


def log_handshake_out(
    step: str,
    target: str,
    run_id: str,
    rows: int,
    operation: str = "INSERT/UPDATE",
) -> None:
    """
    Logt HANDSHAKE_OUT: welke data is geschreven door deze stap.
    
    Format: HANDSHAKE_OUT | step=X | run_id=Y | target=Z | rows=N | operation=...
    """
    msg = f"HANDSHAKE_OUT | step={step} | run_id={run_id} | target={target} | rows={rows} | operation={operation}"
    
    logger.info(msg)
    print(msg, flush=True)  # Ook naar stdout voor Dagster parsing
