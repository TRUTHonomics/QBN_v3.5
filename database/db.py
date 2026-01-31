"""Database helper voor QBN (QuantBayes Nexus).

Bevat één globale connection-pool, helpers voor batch-insert met
psycopg2.extras.execute_values en garandeert dat alle tijdstempels
timezone-aware (UTC) zijn.

ARCHITECTUUR NOOT:
- QBN leest uit KFL tabellen (kfl.mtf_signals_current_lead, kfl.mtf_signals_lead)
- QBN schrijft naar qbn.bayesian_predictions
- Signal processing en MTF building gebeurt in KFL_backend_v3

Gebruik:

    from database.db import insert_many, get_cursor

    # Lees geconsolideerde MTF signals voor inference
    with get_cursor() as cur:
        cur.execute('''
            SELECT 
                l.*, c.*, f.*
            FROM kfl.mtf_signals_current_lead l
            LEFT JOIN kfl.mtf_signals_current_coin c ON c.asset_id = l.asset_id AND c.time_1 = l.time_1
            LEFT JOIN kfl.mtf_signals_current_conf f ON f.asset_id = l.asset_id AND f.time_1 = l.time_1
            WHERE l.asset_id = %s
        ''', (asset_id,))
        row = cur.fetchone()
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from typing import Iterable, List, Sequence, Tuple, Any, Optional
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

# ------------------------------------------------------------
#           environment (.env) loader + var aliases
# ------------------------------------------------------------

# Probeer .env of .env.local automatisch te laden (niet-fataal)
for _fname in (".env.local", ".env"):
    if Path(_fname).is_file():
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(_fname, override=False)
        except ImportError:
            # python-dotenv niet geïnstalleerd; sla silently over
            pass

# Map DB_* naar de juiste POSTGRES_* variabele.
_alias_map = {
    "DB_HOST": "POSTGRES_HOST",
    "DB_PORT": "POSTGRES_PORT",
    "DB_NAME": "POSTGRES_DB",
    "DB_USER": "POSTGRES_USER",
    "DB_PASS": "POSTGRES_PASSWORD",
}

def _getenv(key: str, default: str) -> str:
    alias = _alias_map.get(key)
    # REASON: Prioriteit aan POSTGRES_* variabelen voor Docker consistentie
    return os.getenv(alias) or os.getenv(key) or default

# BELANGRIJK: Database draait op 10.10.10.3 (niet localhost!)
_DB_HOST: str = _getenv("DB_HOST", "10.10.10.3")
_DB_PORT: int = int(_getenv("DB_PORT", "5432"))
_DB_NAME: str = _getenv("DB_NAME", "kflhyper")
_DB_USER: str = _getenv("DB_USER", "qbn")
_DB_PASS: str = _getenv("DB_PASS", "1234")

_MIN_CONN: int = int(os.getenv("DB_POOL_MIN", "4"))
_MAX_CONN: int = int(os.getenv("DB_POOL_MAX", "48"))

# ------------------------------------------------------------
#                     interne pool-singleton
# ------------------------------------------------------------
_pool: Optional[ThreadedConnectionPool] = None
_pool_pid: Optional[int] = None

def _init_pool() -> ThreadedConnectionPool:
    """Initialiseer de globale connection-pool met process-safety."""
    global _pool, _pool_pid

    # REASON: Detecteer process-switches (multiprocessing) om shared connections te voorkomen.
    # EXPL: psycopg2 verbindingen kunnen niet veilig gedeeld worden tussen processen.
    current_pid = os.getpid()
    if _pool is not None and _pool_pid != current_pid:
        _pool = None # Forceer re-initialisatie in kind-proces

    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=_MIN_CONN,
            maxconn=_MAX_CONN,
            host=_DB_HOST,
            port=_DB_PORT,
            dbname=_DB_NAME,
            user=_DB_USER,
            password=_DB_PASS,
        )
        _pool_pid = current_pid
    return _pool

# ------------------------------------------------------------
#                   connection helper
# ------------------------------------------------------------
@contextmanager
def get_cursor(commit: bool = False):
    """Contextmanager die een cursor uit de pool levert met agressieve herstel-logica."""
    pool = _init_pool()
    conn = pool.getconn()
    
    # REASON: Controleer of de verbinding echt nog werkt
    is_dead = False
    try:
        if conn.closed != 0:
            is_dead = True
        else:
            # Test de verbinding met een minimale query
            with conn.cursor() as test_cur:
                test_cur.execute("SELECT 1")
    except Exception:
        is_dead = True

    if is_dead:
        try:
            pool.putconn(conn, close=True)
        except:
            pass
        conn = pool.getconn()

    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        if conn.closed == 0:
            pool.putconn(conn)
        else:
            try:
                pool.putconn(conn, close=True)
            except:
                pass

# ------------------------------------------------------------
#              hulpmethoden voor batch-insert
# ------------------------------------------------------------

def _ensure_utc(value: Any) -> Any:
    """Zorg dat datetime-objecten timezone-aware UTC zijn."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return value

def _prepare_rows(rows: Iterable[Sequence[Any]]) -> List[Tuple[Any, ...]]:
    """Converteer row-iterable naar lijst en zorg dat alle
    datetime-kolommen UTC zijn."""
    prepared: List[Tuple[Any, ...]] = []
    for row in rows:
        prepared.append(tuple(_ensure_utc(v) for v in row))
    return prepared

def insert_many(
    *,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    page_size: int = 1000,
):
    """Batch-insert via execute_values.

    Parameters
    ----------
    table : str
        Volledig gekwalificeerde tabelnaam (schema.tabel).
    columns : Sequence[str]
        Kolomnamen in de volgorde van *rows*.
    rows : Iterable[Sequence[Any]]
        De te inserten records.
    page_size : int, default 1000
        Aantal records per execute_values-batch.
    """
    prepared_rows = _prepare_rows(rows)
    if not prepared_rows:
        return  # niets te doen

    cols_formatted = ", ".join(columns)
    sql = f"INSERT INTO {table} ({cols_formatted}) VALUES %s ON CONFLICT DO NOTHING"

    with get_cursor(commit=True) as cur:
        execute_values(cur, sql, prepared_rows, page_size=page_size)

# ------------------------------------------------------------
#                   util om pool te sluiten
# ------------------------------------------------------------

def close_pool():
    """Sluit alle connecties in de pool."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


