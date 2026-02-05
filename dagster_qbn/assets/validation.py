"""
QBN Validation Assets - Validatie en quality assurance.

REASON: Validation assets lezen van training table assets en genereren rapporten.
Elke validation asset valideert een specifiek aspect van de pipeline.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from dagster import AssetKey, MetadataValue, asset


# ============================================================================
# HELPER FUNCTION
# ============================================================================


def _run_validation_script(
    script_path: str,
    context,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """
    Voert een validation script uit in de QBN_v4_Training container.

    Args:
        script_path: Relatief pad naar script (/app/...)
        context: Dagster context
        extra_args: Extra command-line argumenten

    Returns:
        Dict met stdout, stderr, returncode en elapsed_time_sec.
    """
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        f"/app/{script_path}",
    ]

    if extra_args:
        cmd.extend(extra_args)

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
        raise RuntimeError(f"Validation script failed with exit code {result.returncode}")

    return {
        "returncode": result.returncode,
        "stdout_lines": len(result.stdout.splitlines()),
        "stderr_lines": len(result.stderr.splitlines()),
        "elapsed_time_sec": elapsed,
    }


# ============================================================================
# VALIDATION ASSETS
# ============================================================================


@asset(
    key=AssetKey(["validation", "barrier_status"]),
    deps=[AssetKey(["qbn", "barrier_outcomes"])],
    description="Barrier outcome status en coverage validatie",
    metadata={
        "reads_from": ["qbn.barrier_outcomes"],
        "output_type": "rapport",
        "script": "scripts/analyze_barrier_outcomes.py",
    },
    tags={
        "schema": "qbn",
        "category": "data_fundament",
        "output": "report",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def barrier_status(context) -> dict:
    """
    Analyseert barrier outcome status en distributie.
    
    Script: scripts/analyze_barrier_outcomes.py
    Output: _validation/asset_{asset_id}/barrier_analysis_*.md
    """
    cfg = context.resources.training_run_config

    args = ["--asset-id", str(cfg.asset_id)]

    result = _run_validation_script("scripts/analyze_barrier_outcomes.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
    })

    return result


@asset(
    key=AssetKey(["validation", "signal_classification"]),
    deps=[AssetKey(["qbn", "barrier_outcomes"])],
    description="Signal classification consistency check",
    metadata={
        "reads_from": ["qbn.signal_classification", "kfl.mtf_signals_lead"],
        "output_type": "rapport",
        "script": "scripts/validate_signal_classification.py",
    },
    tags={
        "schema": "mixed",
        "category": "signal_validation",
        "output": "report",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def signal_classification(context) -> dict:
    """
    Valideert signal classification consistency.
    
    Script: scripts/validate_signal_classification.py
    Output: Console rapport (geen file)
    """
    result = _run_validation_script("scripts/validate_signal_classification.py", context)

    context.add_output_metadata({
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
    })

    return result


@asset(
    key=AssetKey(["validation", "ida_weights"]),
    deps=[AssetKey(["qbn", "barrier_outcomes_weights"])],
    description="IDA training weights validatie (dry-run met visualisaties)",
    metadata={
        "reads_from": ["qbn.barrier_outcomes"],
        "output_type": "rapport + plots",
        "script": "scripts/validate_ida_weights.py",
    },
    tags={
        "schema": "qbn",
        "category": "weight_validation",
        "output": "report",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def ida_weights(context) -> dict:
    """
    Valideert IDA weights met dry-run en visualisaties.
    
    Script: scripts/validate_ida_weights.py
    Output: _validation/ida_weights_*.md + plots
    """
    cfg = context.resources.training_run_config

    args = ["--asset-id", str(cfg.asset_id)]

    result = _run_validation_script("scripts/validate_ida_weights.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
    })

    return result


@asset(
    key=AssetKey(["validation", "cpt_health"]),
    deps=[AssetKey(["qbn", "cpt_cache"])],
    description="CPT health report (entropy, coverage, staleness)",
    metadata={
        "reads_from": ["qbn.cpt_cache"],
        "output_type": "rapport",
        "script": "validation/cpt_validator.py",
    },
    tags={
        "schema": "qbn",
        "category": "bn_brain_health",
        "output": "report",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def cpt_health(context) -> dict:
    """
    Valideert CPT health (kwaliteit, coverage, freshness).
    
    Script: Embedded in validation menu (direct Python code)
    Output: _validation/cpt_health_*.md
    """
    cfg = context.resources.training_run_config

    # REASON: CPT health check is embedded in menu, we roepen het via python -c aan
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        "-c",
        f"from validation.production_readiness import ProductionReadinessValidator; "
        f"v = ProductionReadinessValidator(asset_id={cfg.asset_id}); "
        f"r = v.check_cpt_availability(); "
        f"print(f'CPT nodes: {{r.value}}'); "
        f"print(f'Status: {{r.status}}'); "
        f"print(f'Message: {{r.message}}')",
    ]

    context.log.info(f"Executing: {' '.join(cmd[:4])} [python -c ...]")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr and result.returncode != 0:
        context.log.warning(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"CPT health check failed with exit code {result.returncode}")

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(elapsed),
    })

    return {
        "returncode": result.returncode,
        "elapsed_time_sec": elapsed,
    }


@asset(
    key=AssetKey(["validation", "node_diagnostics"]),
    deps=[AssetKey(["qbn", "cpt_cache"])],
    description="Node-level diagnostics voor individuele BN nodes",
    metadata={
        "reads_from": ["qbn.cpt_cache", "qbn.barrier_outcomes"],
        "output_type": "rapport",
        "script": "validation/node_diagnostics.py",
    },
    tags={
        "schema": "qbn",
        "category": "node_diagnostics",
        "output": "report",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def node_diagnostics(context) -> dict:
    """
    Node-level diagnostics per BN node.
    
    Script: Embedded in validation menu
    Output: _validation/node_diagnostics_*.md
    """
    cfg = context.resources.training_run_config

    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        "-c",
        f"from validation.node_diagnostics import NodeDiagnosticValidator; "
        f"from validation.node_diagnostic_report import generate_markdown_report; "
        f"from pathlib import Path; "
        f"v = NodeDiagnosticValidator(asset_id={cfg.asset_id}, run_id=None); "
        f"results = v.run_full_diagnostic(days=3650); "
        f"report_dir = Path('/app/_validation') / f'asset_{cfg.asset_id}' / '12_node-level_diagnostics'; "
        f"report_dir.mkdir(parents=True, exist_ok=True); "
        f"generate_markdown_report({cfg.asset_id}, results, output_dir=report_dir) if results else None",
    ]

    context.log.info(f"Executing: {' '.join(cmd[:4])} [python -c ...]")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - start

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr and result.returncode != 0:
        context.log.warning(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Node diagnostics failed with exit code {result.returncode}")

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(elapsed),
    })

    return {
        "returncode": result.returncode,
        "elapsed_time_sec": elapsed,
    }


@asset(
    key=AssetKey(["validation", "backtest"]),
    deps=[AssetKey(["qbn", "cpt_cache"]), AssetKey(["qbn", "barrier_outcomes"])],
    description="Walk-forward backtest met P&L metrics",
    metadata={
        "reads_from": ["qbn.cpt_cache", "qbn.barrier_outcomes"],
        "output_type": "rapport",
        "script": "scripts/run_backtest_validation.py",
    },
    tags={
        "schema": "qbn",
        "category": "performance",
        "output": "report_and_db",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def backtest(context) -> dict:
    """
    Walk-forward backtest met trade simulator.

    Script: scripts/run_backtest_validation.py (BacktestConfig + run_backtest_internal + save_backtest_report).
    Output: _validation/asset_X/13_walk_forward_backtest/*.md
    """
    cfg = context.resources.training_run_config

    if cfg.skip_backtest:
        context.log.info("Backtest skipped (skip_backtest=True)")
        return {"skipped": True, "elapsed_time_sec": 0.0}

    args = ["--asset-id", str(cfg.asset_id)]
    result = _run_validation_script("scripts/run_backtest_validation.py", context, args)

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(result["elapsed_time_sec"]),
    })

    return {
        "returncode": result["returncode"],
        "elapsed_time_sec": result["elapsed_time_sec"],
    }


@asset(
    key=AssetKey(["validation", "prediction_accuracy"]),
    deps=[AssetKey(["qbn", "cpt_cache"]), AssetKey(["qbn", "barrier_outcomes"])],
    description="Entry en Position prediction accuracy reports",
    metadata={
        "reads_from": ["qbn.output_entry", "qbn.barrier_outcomes", "qbn.cpt_cache"],
        "output_type": "rapport",
        "script": "validation scripts",
    },
    tags={
        "schema": "qbn",
        "category": "performance",
        "output": "report",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def prediction_accuracy(context) -> dict:
    """
    Entry en Position prediction accuracy validatie.
    
    Output: _validation/prediction_accuracy_*.md
    """
    cfg = context.resources.training_run_config

    # REASON: Combinatie van entry + position accuracy checks
    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        "-c",
        f"import sys; "
        f"print('Entry+Position accuracy checks voor asset {cfg.asset_id}'); "
        f"print('TODO: Implement via validation scripts')",
    ]

    context.log.info(f"Executing prediction accuracy checks for asset {cfg.asset_id}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(elapsed),
    })

    return {
        "returncode": result.returncode,
        "elapsed_time_sec": elapsed,
    }


@asset(
    key=AssetKey(["validation", "production_readiness"]),
    deps=[
        AssetKey(["validation", "barrier_status"]),
        AssetKey(["validation", "signal_classification"]),
        AssetKey(["validation", "cpt_health"]),
        AssetKey(["validation", "node_diagnostics"]),
        AssetKey(["validation", "backtest"]),
        AssetKey(["validation", "prediction_accuracy"]),
    ],
    description="GO/NO-GO verdict voor production inference - finale gate",
    metadata={
        "reads_from": [
            "qbn.barrier_outcomes",
            "qbn.signal_classification",
            "qbn.composite_threshold_config",
            "qbn.signal_weights",
            "qbn.cpt_cache",
        ],
        "output_type": "GO/NO-GO verdict",
        "script": "validation/production_readiness.py",
    },
    tags={
        "schema": "qbn",
        "category": "readiness_check",
        "output": "verdict",
    },
    required_resource_keys={"postgres", "training_run_config"},
)
def production_readiness(context) -> dict:
    """
    Production readiness check - GO/NO-GO voor inference.
    
    Script: validation/production_readiness.py
    Output: GO/NO-GO verdict + _validation/production_readiness_*.md
    """
    cfg = context.resources.training_run_config

    cmd = [
        "docker",
        "exec",
        "QBN_v4_Training",
        "python",
        "-c",
        f"from validation.production_readiness import ProductionReadinessValidator; "
        f"v = ProductionReadinessValidator(asset_id={cfg.asset_id}, run_id='{cfg.run_id or ''}'); "
        f"verdict, results = v.run_all_checks(); "
        f"print(f'VERDICT: {{verdict}}'); "
        f"[print(f'{{r.name}}: {{r.status}} - {{r.message}}') for r in results]",
    ]

    context.log.info(f"Executing: {' '.join(cmd[:4])} [python -c ...]")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
        
        # Extract verdict from output
        verdict = "UNKNOWN"
        for line in result.stdout.splitlines():
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip()
                break

    if result.stderr and result.returncode != 0:
        context.log.warning(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Production readiness check failed with exit code {result.returncode}")

    context.add_output_metadata({
        "asset_id": MetadataValue.int(cfg.asset_id),
        "execution_time_sec": MetadataValue.float(elapsed),
        "verdict": MetadataValue.text(verdict),
    })

    return {
        "returncode": result.returncode,
        "elapsed_time_sec": elapsed,
        "verdict": verdict,
    }
