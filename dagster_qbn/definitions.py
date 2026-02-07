from __future__ import annotations

import os

from dagster import Definitions

from dagster_qbn.assets import (
    backtest,
    barrier_outcomes,
    barrier_outcomes_leading,
    barrier_outcomes_weights,
    barrier_status,
    composite_threshold_config,
    cpt_cache,
    cpt_health,
    event_windows,
    ida_weights,
    kfl_klines_raw,
    kfl_indicators_and_signals,
    kfl_mtf_signals_lead,
    kfl_mtf_signals_coin,
    kfl_mtf_signals_conf,
    node_diagnostics,
    position_delta_threshold_config,
    prediction_accuracy,
    production_readiness,
    qbn_db_health_check,
    signal_classification,
    combination_alpha,
    signal_weights,
    training_analysis,
)
from dagster_qbn.jobs import full_pipeline_job, full_training_job, full_validation_job
from dagster_qbn.resources.postgres import PostgresResource
from dagster_qbn.resources.training_run_config import TrainingRunConfig


# REASON: Houd Fase 1 simpel: env vars uit `.env.local` (docker-compose env_file).
# `.env.local` in QBN_v4 gebruikt DB_* variabelen.
host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST") or "10.10.10.3"
port = int(os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT") or "5432")
dbname = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB") or "kflhyper"
user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER") or "qbn"
password = os.getenv("DB_PASS") or os.getenv("POSTGRES_PASSWORD")
if not password:
    raise RuntimeError("Missing required env var: DB_PASS (or POSTGRES_PASSWORD)")

postgres = PostgresResource(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password,
)


# REASON: training_run_config configure_at_launch → asset_id kiesbaar in Launchpad onder Resources
training_run_config = TrainingRunConfig.configure_at_launch()

defs = Definitions(
    assets=[
        # Fase 1: Health check
        qbn_db_health_check,
        # KFL Source Assets (cross-location referenties naar kfl_v6)
        kfl_klines_raw,
        kfl_indicators_and_signals,
        kfl_mtf_signals_lead,
        kfl_mtf_signals_coin,
        kfl_mtf_signals_conf,
        # QBN Table Assets (training pipeline)
        composite_threshold_config,
        barrier_outcomes,
        barrier_outcomes_leading,
        barrier_outcomes_weights,
        signal_weights,
        combination_alpha,
        event_windows,
        position_delta_threshold_config,
        cpt_cache,
        training_analysis,
        # Validation Assets
        barrier_status,
        signal_classification,
        ida_weights,
        cpt_health,
        node_diagnostics,
        backtest,
        prediction_accuracy,
        production_readiness,
    ],
    jobs=[
        full_training_job,
        full_validation_job,
        full_pipeline_job,
    ],
    resources={
        "postgres": postgres,
        "training_run_config": training_run_config,
    },
)

