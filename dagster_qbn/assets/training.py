"""
Training assets voor QBN v4 pipeline (Fase 2).

REASON: Wraps bestaande training scripts via subprocess om GPU-code niet te hoeven refactoren.
Elk asset runt in de QBN_v4_Training container via docker exec.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any

from dagster import AssetKey, MetadataValue, asset


# REASON: Helper om scripts in training-container uit te voeren (GPU-toegang)
def _run_in_training_container(
    script_path: str, context, asset_id: int, asset_flag: str = "--asset-id"
) -> dict[str, Any]:
    """
    Voert script uit in QBN_v4_Training container via docker exec.

    Args:
        script_path: Relatief pad naar script (/app/...)
        context: Dagster context
        asset_id: Asset ID om door te geven
        asset_flag: Flag voor asset ID (meestal --asset-id, alpha_analysis gebruikt --asset)

    Returns:
        Dict met stdout, stderr, returncode, asset_id en elapsed_time_sec.
    """
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        f"/app/{script_path}",
        asset_flag,
        str(asset_id),
    ]

    context.log.info(f"Executing: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    # Log output (scripts often use stderr for normal logging; only warn on failure)
    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        if result.returncode != 0:
            context.log.warning(f"STDERR:\n{result.stderr}")
        else:
            context.log.info(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Script failed with exit code {result.returncode}")

    return {
        "asset_id": asset_id,
        "returncode": result.returncode,
        "stdout_lines": len(result.stdout.splitlines()),
        "stderr_lines": len(result.stderr.splitlines()),
        "elapsed_time_sec": elapsed,
    }


# ============================================================================
# TRAINING ASSETS (in dependency order)
# ============================================================================


@asset(
    key=AssetKey(["training", "threshold_optimization"]),
    description="Stap 1: Optimaliseer thresholds voor Leading Composite signalen (MI/CART/Logistic)",
    metadata={
        "reads_from": ["qbn.barrier_outcomes", "qbn.signal_classification"],
        "writes_to": ["qbn.composite_threshold_config"],
        "gpu_required": False,
        "script": "scripts/run_threshold_analysis.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "training",
        "category": "threshold_optimization",
        "data_volume": "medium",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def threshold_optimization(context) -> dict:
    """
    Script: scripts/run_threshold_analysis.py
    Output: qbn.composite_threshold_config
    GPU: Nee
    Asset ID: uit Resources → training_run_config (Launchpad)
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container("scripts/run_threshold_analysis.py", context, asset_id)

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.composite_threshold_config"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "barrier_backfill"]),
    deps=[AssetKey(["training", "threshold_optimization"])],
    description="Stap 3: Bereken barrier outcomes (bullish/bearish hits) met GPU-versnelling",
    metadata={
        "reads_from": ["kfl.ohlcv", "kfl.mtf_signals_lead"],
        "writes_to": ["qbn.barrier_outcomes"],
        "gpu_required": True,
        "script": "scripts/barrier_backfill.py",
    },
    tags={
        "schema": "mixed",
        "gpu": "true",
        "stage": "training",
        "category": "barrier_computation",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def barrier_backfill(context) -> dict:
    """
    Script: scripts/barrier_backfill.py
    Output: qbn.barrier_outcomes
    GPU: Ja (CuPy)
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container("scripts/barrier_backfill.py", context, asset_id)

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.barrier_outcomes"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "materialize_leading_scores"]),
    deps=[AssetKey(["training", "barrier_backfill"])],
    description="Stap 3.5: Materialize leading_score in barrier_outcomes",
    metadata={
        "reads_from": ["kfl.mtf_signals_lead", "qbn.signal_classification"],
        "writes_to": ["qbn.barrier_outcomes"],
        "gpu_required": False,
        "script": "scripts/materialize_leading_scores.py",
        "operation": "UPDATE leading_score column",
    },
    tags={
        "schema": "mixed",
        "gpu": "false",
        "stage": "training",
        "category": "signal_aggregation",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def materialize_leading_scores(context) -> dict:
    """
    Script: scripts/materialize_leading_scores.py
    Output: leading_score kolom in qbn.barrier_outcomes
    GPU: Nee
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container("scripts/materialize_leading_scores.py", context, asset_id)

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.barrier_outcomes"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "compute_ida_weights"]),
    deps=[AssetKey(["training", "materialize_leading_scores"])],
    description="Stap 4: Compute López de Prado IDA training weights",
    metadata={
        "reads_from": ["qbn.barrier_outcomes"],
        "writes_to": ["qbn.barrier_outcomes"],
        "gpu_required": False,
        "script": "scripts/compute_barrier_weights.py",
        "operation": "UPDATE training_weight column",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "training",
        "category": "weight_computation",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def compute_ida_weights(context) -> dict:
    """
    Script: scripts/compute_barrier_weights.py
    Output: training_weight kolom in qbn.barrier_outcomes
    GPU: Nee
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container("scripts/compute_barrier_weights.py", context, asset_id)

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.barrier_outcomes"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "alpha_analysis"]),
    deps=[AssetKey(["training", "barrier_backfill"])],
    description="Stap 6: Entry Hypothesis Alpha Analysis (voorspellende waarde Leading signalen)",
    metadata={
        "reads_from": ["qbn.barrier_outcomes", "qbn.signal_classification"],
        "writes_to": ["qbn.signal_alpha_results"],
        "gpu_required": False,
        "script": "alpha-analysis/analyze_signal_alpha.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "analysis",
        "category": "alpha_analysis",
        "data_volume": "medium",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def alpha_analysis(context) -> dict:
    """
    Script: alpha-analysis/analyze_signal_alpha.py
    Output: Alpha metrics in database
    GPU: Nee
    """
    asset_id = context.resources.training_run_config.asset_id
    # REASON: Dit script gebruikt --asset in plaats van --asset-id
    result = _run_in_training_container("alpha-analysis/analyze_signal_alpha.py", context, asset_id, asset_flag="--asset")

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.signal_alpha_results"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "combination_analysis"]),
    deps=[AssetKey(["training", "barrier_backfill"])],
    description="Stap 8: Analyse signaalcombinaties (OR-logica) en classificeer Golden Rule/Promising/Noise",
    metadata={
        "reads_from": ["qbn.barrier_outcomes", "qbn.signal_classification"],
        "writes_to": ["qbn.combination_alpha"],
        "gpu_required": True,
        "script": "scripts/run_combination_analysis.py",
        "output_artifacts": ["JSON reports"],
    },
    tags={
        "schema": "qbn",
        "gpu": "true",
        "stage": "analysis",
        "category": "combination_analysis",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def combination_analysis(context) -> dict:
    """
    Script: scripts/run_combination_analysis.py
    Output: Combination results in database + JSON reports
    GPU: Ja
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container("scripts/run_combination_analysis.py", context, asset_id)

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.combination_alpha"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "event_window_detection"]),
    deps=[AssetKey(["training", "barrier_backfill"]), AssetKey(["training", "threshold_optimization"])],
    description="Stap 9: Detecteer event windows (hoge Leading composite spikes)",
    metadata={
        "reads_from": ["qbn.barrier_outcomes", "qbn.composite_threshold_config"],
        "writes_to": ["qbn.barrier_outcomes", "qbn.event_windows"],
        "gpu_required": False,
        "script": "scripts/run_event_window_detection.py",
        "operation": "UPDATE event_id column + INSERT events",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "training",
        "category": "event_detection",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def event_window_detection(context) -> dict:
    """
    Script: scripts/run_event_window_detection.py (wrapper)
    Output: event_id labels in qbn.barrier_outcomes + qbn.event_windows
    GPU: Nee
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container("scripts/run_event_window_detection.py", context, asset_id)

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.barrier_outcomes", "qbn.event_windows"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "position_delta_thresholds"]),
    deps=[AssetKey(["training", "event_window_detection"])],
    description="Stap 9.5: Optimaliseer delta thresholds voor Position Management",
    metadata={
        "reads_from": ["qbn.barrier_outcomes"],
        "writes_to": ["qbn.position_delta_threshold_config"],
        "gpu_required": False,
        "script": "scripts/run_position_delta_threshold_analysis.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "training",
        "category": "threshold_optimization",
        "data_volume": "medium",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def position_delta_thresholds(context) -> dict:
    """
    Script: scripts/run_position_delta_threshold_analysis.py
    Output: Thresholds in qbn.position_delta_threshold_config
    GPU: Nee
    """
    asset_id = context.resources.training_run_config.asset_id
    result = _run_in_training_container(
        "scripts/run_position_delta_threshold_analysis.py", context, asset_id
    )

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(result["stdout_lines"]),
            "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
            "tables_written": MetadataValue.json(["qbn.position_delta_threshold_config"]),
        }
    )

    return result


@asset(
    key=AssetKey(["training", "cpt_generation"]),
    deps=[
        AssetKey(["training", "compute_ida_weights"]),
        AssetKey(["training", "event_window_detection"]),
        AssetKey(["training", "position_delta_thresholds"]),
    ],
    description="Stap 10: Genereer CPTs voor alle Bayesian Network nodes",
    metadata={
        "reads_from": ["qbn.barrier_outcomes", "qbn.event_windows", "qbn.signal_classification"],
        "writes_to": ["qbn.cpt_cache"],
        "gpu_required": False,
        "script": "inference/qbn_v3_cpt_generator.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "training",
        "category": "cpt_generation",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def cpt_generation(context) -> dict:
    """
    Script: inference/qbn_v3_cpt_generator.py (via helper)
    Output: CPTs in qbn.cpt_cache
    GPU: Nee
    """
    asset_id = context.resources.training_run_config.asset_id
    
    # REASON: CPT generator wordt via python -c aangeroepen
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        "-c",
        f"from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator; "
        f"gen = QBNv3CPTGenerator(); "
        f"gen.generate_all_cpts(asset_id={asset_id})",
    ]

    context.log.info(f"Executing: {' '.join(cmd[:4])} [python -c ...]")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        if result.returncode != 0:
            context.log.warning(f"STDERR:\n{result.stderr}")
        else:
            context.log.info(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"CPT generation failed with exit code {result.returncode}")

    context.add_output_metadata(
        {
            "asset_id": MetadataValue.int(asset_id),
            "stdout_lines": MetadataValue.int(len(result.stdout.splitlines())),
            "execution_time_sec": MetadataValue.float(elapsed),
            "tables_written": MetadataValue.json(["qbn.cpt_cache"]),
        }
    )

    return {
        "asset_id": asset_id,
        "returncode": result.returncode,
        "stdout_lines": len(result.stdout.splitlines()),
        "elapsed_time_sec": elapsed,
    }
