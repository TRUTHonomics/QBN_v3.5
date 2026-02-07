"""
Run-config resource: volledige pipeline configuratie instelbaar bij launch.

REASON: Gebruiker kiest in de Launchpad onder Resources → training_run_config alle parameters.
Alle training assets lezen hun configuratie van deze resource.
"""
from __future__ import annotations

from typing import Optional

from dagster import ConfigurableResource


class TrainingRunConfig(ConfigurableResource):
    """
    Volledige configuratie voor een training run. Instelbaar in de Launchpad onder Resources.

    Gebruik: Bij Materialize / full_training_run configureer je hier alle pipeline parameters.
    """

    # === Basis (alle assets) ===
    asset_id: int = 1
    """Asset ID voor training (bijv. 1=Bitcoin, 9889=test). Gebruikt door: ALLE assets."""

    run_id: Optional[str] = None
    """Optionele identifier voor traceability. Gebruikt door: ALLE assets."""

    # === Data Window ===
    lookback_days: int = 365
    """Aantal dagen training data. Gebruikt door: composite_threshold_config, combination_alpha, position_delta_threshold_config."""

    # === Barrier Backfill ===
    batch_size: int = 100000
    """Batch size voor GPU/database processing. Gebruikt door: barrier_outcomes, barrier_outcomes_leading."""

    incremental: bool = False
    """Alleen nieuwe data verwerken (skip bestaande). Gebruikt door: barrier_outcomes."""

    overwrite: bool = False
    """Bestaande outcomes overschrijven. Gebruikt door: barrier_outcomes, barrier_outcomes_leading."""

    # === Threshold Optimization ===
    methods: str = "mi,cart,logreg"
    """Threshold methods (comma-separated: mi, cart, logreg). Gebruikt door: composite_threshold_config."""

    horizons: str = "1h,4h,1d"
    """Horizon windows (comma-separated). Gebruikt door: composite_threshold_config."""

    targets: str = "leading"
    """Target signals voor threshold analyse. Gebruikt door: composite_threshold_config."""

    enforce_diversity: bool = True
    """Diversity constraints afdwingen (min 3 actieve states). Gebruikt door: composite_threshold_config, position_delta_threshold_config."""

    # === Alpha Analysis ===
    alpha_layer: str = "HYPOTHESIS"
    """Layer voor alpha analyse (HYPOTHESIS=Leading, CONFIDENCE=Coin/Conf). Gebruikt door: signal_weights."""

    # === IDA Weights ===
    ida_config: str = "baseline"
    """IDA weight strategie (baseline, balanced, delta_only, aggressive). Gebruikt door: barrier_outcomes_weights."""

    # === Combination Analysis ===
    n_bootstrap: int = 1000
    """Bootstrap iterations voor statistische significantie. Gebruikt door: combination_alpha."""

    use_gpu: bool = True
    """GPU gebruiken voor combination analysis. Gebruikt door: combination_alpha."""

    # === Validation ===
    skip_benchmarks: bool = False
    """Skip GPU/CPU performance benchmarks (sneller maar minder info). Gebruikt door: validation/barrier_benchmarks."""

    skip_backtest: bool = False
    """Skip walk-forward backtest (kost meeste tijd). Gebruikt door: validation/backtest."""

    validation_lookback_days: int = 30
    """Lookback dagen voor CPT stability validation. Gebruikt door: validation/cpt_stability."""
