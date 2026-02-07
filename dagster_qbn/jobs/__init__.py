"""
Jobs voor QBN v4 training pipeline.
"""
from __future__ import annotations

from dagster import AssetSelection, define_asset_job


# REASON: Job voor volledige training run (alle QBN table assets)
full_training_job = define_asset_job(
    name="full_training_run",
    description="Voert volledige QBN training pipeline uit (threshold → barrier → ... → CPT → analysis)",
    selection=AssetSelection.keys(
        ["qbn", "composite_threshold_config"],
        ["qbn", "barrier_outcomes"],
        ["qbn", "barrier_outcomes_leading"],
        ["qbn", "barrier_outcomes_weights"],
        ["qbn", "signal_weights"],
        ["qbn", "combination_alpha"],
        ["qbn", "event_windows"],
        ["qbn", "position_delta_threshold_config"],
        ["qbn", "cpt_cache"],
        ["qbn", "training_analysis"],  # Post-processing analyzer
    ),
)


# REASON: Job voor validation run (alle validation assets)
full_validation_job = define_asset_job(
    name="full_validation_run",
    description="Volledige QBN validation cycle (na training) - eindigt met GO/NO-GO verdict",
    selection=AssetSelection.keys(
        ["validation", "barrier_status"],
        ["validation", "signal_classification"],
        ["validation", "ida_weights"],
        ["validation", "cpt_health"],
        ["validation", "node_diagnostics"],
        ["validation", "backtest"],
        ["validation", "prediction_accuracy"],
        ["validation", "production_readiness"],
    ),
)


# REASON: Gecombineerde job voor training + validation in één keer
full_pipeline_job = define_asset_job(
    name="full_pipeline_run",
    description="Training + Validation in één run (QBN table assets → analysis → validation assets → GO/NO-GO)",
    selection=AssetSelection.keys(
        # Training assets
        ["qbn", "composite_threshold_config"],
        ["qbn", "barrier_outcomes"],
        ["qbn", "barrier_outcomes_leading"],
        ["qbn", "barrier_outcomes_weights"],
        ["qbn", "signal_weights"],
        ["qbn", "combination_alpha"],
        ["qbn", "event_windows"],
        ["qbn", "position_delta_threshold_config"],
        ["qbn", "cpt_cache"],
        ["qbn", "training_analysis"],  # Post-processing analyzer
        # Validation assets
        ["validation", "barrier_status"],
        ["validation", "signal_classification"],
        ["validation", "ida_weights"],
        ["validation", "cpt_health"],
        ["validation", "node_diagnostics"],
        ["validation", "backtest"],
        ["validation", "prediction_accuracy"],
        ["validation", "production_readiness"],
    ),
)
