from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection


@dataclass(frozen=True)
class PostgresResource:
    """
    Minimal Postgres resource voor Dagster assets (Fase 1).

    REASON: We houden dit bewust simpel (geen SQLAlchemy) om alleen connectie
    + een query te testen.
    """

    host: str
    port: int
    dbname: str
    user: str
    password: str
    connect_timeout_seconds: int = 5
    application_name: Optional[str] = "dagster_qbn"

    @contextmanager
    def get_conn(self) -> Iterator[PgConnection]:
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            connect_timeout=self.connect_timeout_seconds,
            application_name=self.application_name,
        )
        try:
            yield conn
        finally:
            conn.close()

