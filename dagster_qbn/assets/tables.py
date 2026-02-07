"""
QBN Table Assets - Postgres tabellen als Dagster assets.

REASON: In de Runs-graph zie je nu tabellennamen i.p.v. script-namen.
Elk asset materialiseert een Postgres-tabel door het bijbehorende script uit te voeren.
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dagster import AssetKey, MetadataValue, SourceAsset, asset


# ============================================================================
# SOURCE ASSETS - KFL Input Tables
# ============================================================================
# REASON: Deze SourceAssets verwijzen naar de KFL code location (kfl_v6).
# Dagster linkt assets met dezelfde AssetKey automatisch cross-location.

kfl_klines_raw = SourceAsset(
    key=AssetKey(["kfl", "klines_raw"]),
    description="Ruwe OHLCV kline data geproduceerd door real-time WebSocket pipeline (VM104)",
)

kfl_indicators_and_signals = SourceAsset(
    key=AssetKey(["kfl", "indicators_and_signals"]),
    description="GPU-accelerated indicators + signals (lead/coin/conf) vanuit KFL_backend_GPU_v6",
)

kfl_mtf_signals_lead = SourceAsset(
    key=AssetKey(["kfl", "mtf_signals_lead"]),
    description="KFL Multi-Timeframe LEADING signals (60m Tactical Layer)",
)

kfl_mtf_signals_coin = SourceAsset(
    key=AssetKey(["kfl", "mtf_signals_coin"]),
    description="KFL Multi-Timeframe COINCIDENT signals",
)

kfl_mtf_signals_conf = SourceAsset(
    key=AssetKey(["kfl", "mtf_signals_conf"]),
    description="KFL Multi-Timeframe CONFIRMING signals",
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

_RUN_ID_FILE = "/tmp/qbn_current_run_id"


def _resolve_run_id(cfg) -> str:
    """
    Geeft cfg.run_id terug, of genereert eenmalig een nieuw run_id.
    
    REASON: Container-file approach zorgt ervoor dat alle assets binnen
    één pipeline run hetzelfde run_id gebruiken zonder de Dagster graph te wijzigen.
    """
    if cfg.run_id:
        return cfg.run_id
    
    # Check of een eerder asset in deze run al een run_id heeft geschreven
    try:
        result = subprocess.run(
            ["docker", "exec", "QBN_v4_Training", "cat", _RUN_ID_FILE],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    
    # Genereer nieuw run_id en schrijf naar container-file
    new_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    subprocess.run(
        ["docker", "exec", "QBN_v4_Training", "bash", "-c", f"echo '{new_id}' > {_RUN_ID_FILE}"],
        check=True,
        timeout=5
    )
    return new_id


def _run_training_script(
    script_path: str,
    context,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """
    Voert een training script uit in de QBN_v4_Training container.

    Args:
        script_path: Relatief pad naar script (/app/...)
        context: Dagster context
        extra_args: Extra command-line argumenten

    Returns:
        Dict met stdout, stderr, returncode en elapsed_time_sec.
    """
    cfg = context.resources.training_run_config
    run_id = _resolve_run_id(cfg)
    
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        f"/app/{script_path}",
    ]

    # Voeg extra args toe
    args = extra_args or []
    
    # Auto-inject --run-id als het niet al aanwezig is
    if "--run-id" not in args:
        args.extend(["--run-id", run_id])
    
    cmd.extend(args)

    context.log.info(f"Executing: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    # Log output
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
        "returncode": result.returncode,
        "stdout_lines": len(result.stdout.splitlines()),
        "stderr_lines": len(result.stderr.splitlines()),
        "elapsed_time_sec": elapsed,
        "run_id": run_id,
    }


# ============================================================================
# QBN TABLE ASSETS (in data flow order)
# ============================================================================


@asset(
    key=AssetKey(["qbn", "composite_threshold_config"]),
    description="Threshold configuratie voor Leading Composite signalen (MI/CART/Logistic)",
    metadata={
        "table_name": "qbn.composite_threshold_config",
        "reads_from": ["qbn.barrier_outcomes", "qbn.signal_classification"],
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
def composite_threshold_config(context) -> dict:
    """
    Optimaliseert thresholds voor Leading Composite signalen.
    
    Script: scripts/run_threshold_analysis.py
    Output: qbn.composite_threshold_config
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
        "--methods", cfg.methods,
        "--horizons", cfg.horizons,
        "--targets", cfg.targets,
        "--lookback-days", str(cfg.lookback_days),
        "--apply-results",  # Direct opslaan in DB
    ]

    if not cfg.enforce_diversity:
        args.append("--no-diversity-check")

    # REASON: --run-id wordt automatisch toegevoegd door _run_training_script

    result = _run_training_script("scripts/run_threshold_analysis.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
        "methods": MetadataValue.text(cfg.methods),
        "horizons": MetadataValue.text(cfg.horizons),
    })

    return result


@asset(
    key=AssetKey(["qbn", "barrier_outcomes"]),
    deps=[
        AssetKey(["kfl", "klines_raw"]),
        AssetKey(["kfl", "mtf_signals_lead"]),
        AssetKey(["qbn", "composite_threshold_config"]),
    ],
    description="Barrier outcomes (bullish/bearish hits) - GPU accelerated",
    metadata={
        "table_name": "qbn.barrier_outcomes",
        "reads_from": ["kfl.klines_raw", "kfl.mtf_signals_lead"],
        "gpu_required": True,
        "script": "scripts/barrier_backfill.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "true",
        "stage": "training",
        "category": "barrier_computation",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def barrier_outcomes(context) -> dict:
    """
    Berekent barrier outcomes met GPU-versnelling.
    
    Script: scripts/barrier_backfill.py
    Output: qbn.barrier_outcomes
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
        "--batch-size", str(cfg.batch_size),
    ]

    if cfg.incremental:
        args.append("--incremental")
    if cfg.overwrite:
        args.append("--overwrite")
    # REASON: --run-id wordt automatisch toegevoegd door _run_training_script

    result = _run_training_script("scripts/barrier_backfill.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
        "batch_size": MetadataValue.int(cfg.batch_size),
        "mode": MetadataValue.text("incremental" if cfg.incremental else "full"),
    })

    return result


@asset(
    key=AssetKey(["qbn", "barrier_outcomes_leading"]),
    deps=[AssetKey(["qbn", "barrier_outcomes"]), AssetKey(["kfl", "mtf_signals_lead"])],
    description="Materialize leading_score kolom in barrier_outcomes",
    metadata={
        "table_name": "qbn.barrier_outcomes",
        "column": "leading_score",
        "reads_from": ["kfl.mtf_signals_lead", "qbn.signal_classification"],
        "gpu_required": False,
        "script": "scripts/materialize_leading_scores.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "training",
        "category": "signal_aggregation",
        "data_volume": "large",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def barrier_outcomes_leading(context) -> dict:
    """
    Materialiseert leading_score in qbn.barrier_outcomes.
    
    Script: scripts/materialize_leading_scores.py
    Output: UPDATE qbn.barrier_outcomes SET leading_score = ...
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
        "--batch-size", str(cfg.batch_size),
    ]

    if cfg.overwrite:
        args.append("--overwrite")
    # REASON: --run-id wordt automatisch toegevoegd door _run_training_script

    result = _run_training_script("scripts/materialize_leading_scores.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
        "operation": MetadataValue.text("UPDATE leading_score column"),
    })

    return result


@asset(
    key=AssetKey(["qbn", "barrier_outcomes_weights"]),
    deps=[AssetKey(["qbn", "barrier_outcomes_leading"])],
    description="Compute López de Prado IDA training weights",
    metadata={
        "table_name": "qbn.barrier_outcomes",
        "column": "training_weight",
        "reads_from": ["qbn.barrier_outcomes"],
        "gpu_required": False,
        "script": "scripts/compute_barrier_weights.py",
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
def barrier_outcomes_weights(context) -> dict:
    """
    Berekent training weights (IDA) in qbn.barrier_outcomes.
    
    Script: scripts/compute_barrier_weights.py
    Output: UPDATE qbn.barrier_outcomes SET training_weight = ...
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
        "--config", cfg.ida_config,
    ]

    # REASON: --run-id wordt automatisch toegevoegd door _run_training_script

    result = _run_training_script("scripts/compute_barrier_weights.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
        "ida_config": MetadataValue.text(cfg.ida_config),
    })

    return result


@asset(
    key=AssetKey(["qbn", "signal_weights"]),
    deps=[AssetKey(["qbn", "barrier_outcomes"])],
    description="Entry Hypothesis Alpha Analysis - voorspellende waarde Leading signalen",
    metadata={
        "table_name": "qbn.signal_weights",
        "reads_from": ["qbn.barrier_outcomes", "qbn.signal_classification"],
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
def signal_weights(context) -> dict:
    """
    Berekent alpha scores (voorspellende waarde) voor signalen.
    
    Script: alpha-analysis/analyze_signal_alpha.py
    Output: qbn.signal_weights
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset", str(cfg.asset_id),  # Dit script gebruikt --asset i.p.v. --asset-id
        "--layer", cfg.alpha_layer,
    ]

    # REASON: --run-id wordt automatisch toegevoegd door _run_training_script

    result = _run_training_script("alpha-analysis/analyze_signal_alpha.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
        "layer": MetadataValue.text(cfg.alpha_layer),
    })

    return result


@asset(
    key=AssetKey(["qbn", "combination_alpha"]),
    deps=[AssetKey(["qbn", "barrier_outcomes"])],
    description="Signaalcombinatie analyse (OR-logica) - Golden Rule/Promising/Noise",
    metadata={
        "table_name": "qbn.combination_alpha",
        "reads_from": ["qbn.barrier_outcomes", "qbn.signal_classification"],
        "gpu_required": True,
        "script": "scripts/run_combination_analysis.py",
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
def combination_alpha(context) -> dict:
    """
    Analyseert signaalcombinaties met bootstrap statistiek.

    Script: scripts/run_combination_analysis.py
    Output: qbn.combination_alpha + JSON reports
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
        "--lookback-days", str(cfg.lookback_days),
        "--n-bootstrap", str(cfg.n_bootstrap),
        "--save-db",
    ]

    if not cfg.use_gpu:
        args.append("--no-gpu")
    # REASON: --run-id wordt automatisch toegevoegd door _run_training_script

    result = _run_training_script("scripts/run_combination_analysis.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
        "n_bootstrap": MetadataValue.int(cfg.n_bootstrap),
    })

    return result


@asset(
    key=AssetKey(["qbn", "event_windows"]),
    deps=[
        AssetKey(["qbn", "barrier_outcomes"]),
        AssetKey(["qbn", "composite_threshold_config"]),
    ],
    description="Event windows - periodes tussen Leading spike en barrier outcome",
    metadata={
        "table_name": "qbn.event_windows",
        "reads_from": ["qbn.barrier_outcomes", "qbn.composite_threshold_config"],
        "also_writes_to": ["qbn.barrier_outcomes.event_id"],
        "gpu_required": False,
        "script": "scripts/run_event_window_detection.py",
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
def event_windows(context) -> dict:
    """
    Detecteert event windows (Leading spikes tot barrier hit).
    
    Script: scripts/run_event_window_detection.py
    Output: qbn.event_windows + UPDATE qbn.barrier_outcomes SET event_id = ...
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
    ]

    result = _run_training_script("scripts/run_event_window_detection.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
    })

    return result


@asset(
    key=AssetKey(["qbn", "position_delta_threshold_config"]),
    deps=[AssetKey(["qbn", "event_windows"])],
    description="Delta thresholds voor Position Management (Coincident/Confirming)",
    metadata={
        "table_name": "qbn.position_delta_threshold_config",
        "reads_from": ["qbn.barrier_outcomes"],
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
def position_delta_threshold_config(context) -> dict:
    """
    Optimaliseert delta thresholds voor Position Management.
    
    Script: scripts/run_position_delta_threshold_analysis.py
    Output: qbn.position_delta_threshold_config
    """
    cfg = context.resources.training_run_config

    args = [
        "--asset-id", str(cfg.asset_id),
        "--lookback", str(cfg.lookback_days),
    ]

    if not cfg.enforce_diversity:
        args.append("--no-diversity")

    result = _run_training_script("scripts/run_position_delta_threshold_analysis.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
    })

    return result


@asset(
    key=AssetKey(["qbn", "cpt_cache"]),
    deps=[
        AssetKey(["qbn", "barrier_outcomes_weights"]),
        AssetKey(["qbn", "event_windows"]),
        AssetKey(["qbn", "position_delta_threshold_config"]),
    ],
    description="Conditional Probability Tables voor Bayesian Network nodes",
    metadata={
        "table_name": "qbn.cpt_cache",
        "reads_from": ["qbn.barrier_outcomes", "qbn.event_windows", "qbn.signal_classification"],
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
def cpt_cache(context) -> dict:
    """
    Genereert CPTs voor alle Bayesian Network nodes.
    
    Script: inference/qbn_v3_cpt_generator.py
    Output: qbn.cpt_cache
    """
    cfg = context.resources.training_run_config
    run_id = _resolve_run_id(cfg)

    # REASON: CPT generator wordt via python -c aangeroepen
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        "-c",
        f"from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator; "
        f"gen = QBNv3CPTGenerator(run_id='{run_id}'); "
        f"gen.generate_all_cpts(asset_id={cfg.asset_id})",
    ]

    context.log.info(f"Executing: {' '.join(cmd[:4])} [python -c ...] (run_id={run_id})")

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

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "run_id": MetadataValue.text(run_id),
        "execution_time_sec": MetadataValue.float(elapsed),
        "stdout_lines": MetadataValue.int(len(result.stdout.splitlines())),
    })

    return {
        "returncode": result.returncode,
        "stdout_lines": len(result.stdout.splitlines()),
        "elapsed_time_sec": elapsed,
        "run_id": run_id,
    }


@asset(
    key=AssetKey(["qbn", "training_analysis"]),
    deps=[AssetKey(["qbn", "cpt_cache"])],
    description="Post-processing: Analyse training run resultaten en genereer rapporten",
    metadata={
        "table_name": "N/A (generates reports only)",
        "reads_from": ["logs", "database"],
        "gpu_required": False,
        "script": "analysis/pipeline_run_analyzer.py",
    },
    tags={
        "schema": "qbn",
        "gpu": "false",
        "stage": "post_processing",
        "category": "analysis",
        "data_volume": "low",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def training_analysis(context) -> dict:
    """
    Analyseert training run resultaten en genereert rapporten.

    Script: analysis/pipeline_run_analyzer.py
    Output: _validation/{timestamp}-pipeline_analysis-asset_{id}-{run_id}/
    """
    cfg = context.resources.training_run_config
    run_id = _resolve_run_id(cfg)

    context.log.info(f"Starting pipeline analysis for run_id={run_id}, asset={cfg.asset_id}")

    # Optioneel: Dagster terminal log voor handshake-detectie (bij runs via Dagster)
    dagster_log = os.environ.get("DAGSTER_LOG_PATH")
    if not dagster_log:
        try:
            tmp = Path("/tmp")
            if tmp.exists():
                for log_path in sorted(tmp.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(1000)
                        if run_id in content and "dagster" in content.lower():
                            dagster_log = str(log_path)
                            break
                    except Exception:
                        continue
        except Exception as e:
            context.log.warning(f"Could not scan for Dagster terminal log: {e}")

    cmd = [
        "docker", "exec", "QBN_v4_Training",
        "python", "analysis/pipeline_run_analyzer.py",
        "--asset-id", str(cfg.asset_id),
        "--run-id", run_id,
    ]
    if dagster_log:
        cmd.extend(["--dagster-log", dagster_log])
        context.log.info(f"Using Dagster log: {dagster_log}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    
    context.log.info("Pipeline analysis complete")
    context.log.info(result.stdout)
    
    # Parse output directory from stdout
    import re
    output_dir_match = re.search(r'Output: (.+)', result.stdout)
    output_dir = output_dir_match.group(1) if output_dir_match else "unknown"
    
    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "run_id": MetadataValue.text(run_id),
        "output_directory": MetadataValue.text(output_dir),
        "analysis_time": MetadataValue.text(datetime.now().isoformat()),
    })
    
    return {"run_id": run_id, "output_directory": output_dir}

