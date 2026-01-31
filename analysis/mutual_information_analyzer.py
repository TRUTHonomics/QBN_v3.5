# analysis/mutual_information_analyzer.py
"""
Mutual Information Grid Search voor Threshold Optimalisatie.

Zoekt de optimale combinatie van COMPOSITE_NEUTRAL_BAND en COMPOSITE_STRONG_THRESHOLD
die de mutual information tussen composite states en price outcomes maximaliseert.

REFACTORED: 
- Minimum neutral_band verhoogd van 0.001 naar 0.05 om extreme bias te voorkomen
- State diversity constraint toegevoegd: minimaal 3 actieve states vereist
- Neutral state moet tussen 10-60% liggen voor gezonde distributie
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.metrics import mutual_info_score

from .threshold_optimizer import ThresholdOptimizer, ThresholdAnalysisResult

logger = logging.getLogger(__name__)

# Check matplotlib availability (headless mode)
try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not available - plotting disabled")


# REASON: Diversity constraints om extreme bias en "stuck states" te voorkomen
MIN_ACTIVE_STATES = 3          # Minimaal 3 states met >5% representatie
MIN_NEUTRAL_PERCENT = 0.10     # Neutral state minimaal 10%
MAX_NEUTRAL_PERCENT = 0.60     # Neutral state maximaal 60%
MIN_STATE_THRESHOLD = 0.05     # Threshold voor "actieve" state (5%)
MIN_BULLISH_SIDE_PERCENT = 0.05  # (bullish + strong_bullish) minimaal 5%
MIN_BEARISH_SIDE_PERCENT = 0.05  # (bearish + strong_bearish) minimaal 5%


class MutualInformationAnalyzer(ThresholdOptimizer):
    """
    Grid Search over threshold parameters om MI te maximaliseren.
    
    REFACTORED: Nu met diversity constraints om extreme bias te voorkomen.
    
    Genereert:
    - 2D Heatmap van MI scores voor elke (neutral_band, strong_threshold) combinatie
    - Optimale waarden gemarkeerd
    - Aparte analyse per horizon (1h, 4h, 1d)
    - Diversity score naast MI score
    """
    
    # Grid search parameters
    # REASON: Minimum verhoogd van 0.001 naar 0.05 om extreme bias te voorkomen.
    # De oude waarde van 0.001 leidde tot 99%+ bullish bias in Leading_Composite.
    NEUTRAL_BAND_VALUES = [
        0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25
    ]
    STRONG_THRESHOLD_VALUES = [
        0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60
    ]
    
    # Diversity constraint parameters
    MIN_ACTIVE_STATES = MIN_ACTIVE_STATES
    MIN_NEUTRAL_PERCENT = MIN_NEUTRAL_PERCENT
    MAX_NEUTRAL_PERCENT = MAX_NEUTRAL_PERCENT
    
    def __init__(
        self,
        asset_id: int,
        lookback_days: int = 180,
        train_ratio: float = 0.8,
        min_samples: int = 5000,
        output_dir: Optional[Path] = None,
        enforce_diversity: bool = True
    ):
        super().__init__(asset_id, lookback_days, train_ratio, min_samples)
        self.output_dir = output_dir or Path("_validation/threshold_analysis")
        self.enforce_diversity = enforce_diversity
    
    def _evaluate_threshold_combination(
        self,
        scores: pd.Series,
        outcomes: pd.Series,
        bullish_neutral_band: float,
        bullish_strong_thresh: float,
        bearish_neutral_band: float,
        bearish_strong_thresh: float
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Evalueer een threshold combinatie met MI en diversity constraints.
        
        Args:
            scores: Series met composite scores
            outcomes: Series met outcome directions
            bullish_neutral_band: Threshold voor bullish (positieve) kant
            bullish_strong_thresh: Threshold voor strong_bullish
            bearish_neutral_band: Threshold voor bearish (negatieve) kant (positieve magnitude)
            bearish_strong_thresh: Threshold voor strong_bearish (positieve magnitude)
            
        Returns:
            Tuple van (effective_score, raw_mi, state_distribution)
            
            effective_score: MI score, of -1.0 indien diversity constraints falen
        """
        # Discretiseer naar states (asymmetrisch)
        states = scores.apply(
            lambda x: self.discretize_composite_asymmetric(
                x,
                bullish_neutral_band=bullish_neutral_band,
                bullish_strong_threshold=bullish_strong_thresh,
                bearish_neutral_band=bearish_neutral_band,
                bearish_strong_threshold=bearish_strong_thresh
            )
        )
        
        # Bereken state distribution
        distribution = states.value_counts(normalize=True)
        dist_dict = distribution.to_dict()
        
        # Bereken raw MI
        raw_mi = mutual_info_score(states, outcomes)
        
        if not self.enforce_diversity:
            return raw_mi, raw_mi, dist_dict
        
        # DIVERSITY CONSTRAINT 1: Minimaal N actieve states (>5% representatie)
        active_states = (distribution > MIN_STATE_THRESHOLD).sum()
        if active_states < self.MIN_ACTIVE_STATES:
            logger.debug(f"Rejected: only {active_states} active states at "
                        f"neutral={neutral_band}, strong={strong_thresh}")
            return -1.0, raw_mi, dist_dict
        
        # DIVERSITY CONSTRAINT 2: Neutral state tussen 10-60%
        neutral_pct = dist_dict.get('neutral', 0)
        if neutral_pct < self.MIN_NEUTRAL_PERCENT:
            logger.debug(f"Rejected: neutral={neutral_pct:.1%} < {self.MIN_NEUTRAL_PERCENT:.0%} at "
                        f"neutral_band={neutral_band}, strong={strong_thresh}")
            return -1.0, raw_mi, dist_dict
        
        if neutral_pct > self.MAX_NEUTRAL_PERCENT:
            logger.debug(f"Rejected: neutral={neutral_pct:.1%} > {self.MAX_NEUTRAL_PERCENT:.0%} at "
                        f"neutral_band={neutral_band}, strong={strong_thresh}")
            return -1.0, raw_mi, dist_dict

        # DIVERSITY CONSTRAINT 3: beide kanten moeten voorkomen (anders krijg je stuck bias)
        bullish_pct = dist_dict.get('bullish', 0) + dist_dict.get('strong_bullish', 0)
        bearish_pct = dist_dict.get('bearish', 0) + dist_dict.get('strong_bearish', 0)
        if bullish_pct < MIN_BULLISH_SIDE_PERCENT or bearish_pct < MIN_BEARISH_SIDE_PERCENT:
            logger.debug(
                f"Rejected: side imbalance bull={bullish_pct:.1%}, bear={bearish_pct:.1%} "
                f"at bull(nb={bullish_neutral_band}, st={bullish_strong_thresh}), "
                f"bear(nb={bearish_neutral_band}, st={bearish_strong_thresh})"
            )
            return -1.0, raw_mi, dist_dict
        
        # Alle constraints OK - gebruik raw MI als score
        return raw_mi, raw_mi, dist_dict
        
    def analyze(self, horizon: str, target: str = 'leading', df: Optional[pd.DataFrame] = None) -> ThresholdAnalysisResult:
        """
        Voer MI Grid Search uit voor een specifieke horizon en target composite.
        
        REFACTORED: Nu met diversity constraints om extreme bias te voorkomen.
        
        Args:
            horizon: De te analyseren horizon ('1h', '4h', '1d')
            target: De target composite ('leading', 'coincident', 'confirming')
            df: Optioneel reeds geladen DataFrame
            
        Returns:
            ThresholdAnalysisResult met optimale threshold waarden
        """
        logger.info(f"Starting MI Grid Search for horizon {horizon}, target {target}")
        logger.info(f"Diversity constraints: enforce={self.enforce_diversity}, "
                   f"min_active_states={self.MIN_ACTIVE_STATES}, "
                   f"neutral_range=[{self.MIN_NEUTRAL_PERCENT:.0%}, {self.MAX_NEUTRAL_PERCENT:.0%}]")
        
        # Load data
        df = self.load_data(horizon, df=df)
        train_df, test_df = self.train_test_split(df)
        
        # Determine score column
        score_col = f"{target}_score"
        if score_col not in train_df.columns:
            raise ValueError(f"Score column {score_col} not found in data")
        
        # Log score distribution voor debugging
        score_stats = train_df[score_col].describe()
        logger.info(f"Score distribution: mean={score_stats['mean']:.4f}, "
                   f"std={score_stats['std']:.4f}, min={score_stats['min']:.4f}, max={score_stats['max']:.4f}")
            
        # 4D grid search (asymmetrisch) met diversity constraints
        best_score = -1.0
        best_mi = -1.0
        best_bull_nb = self.NEUTRAL_BAND_VALUES[len(self.NEUTRAL_BAND_VALUES) // 2]
        best_bull_st = self.STRONG_THRESHOLD_VALUES[len(self.STRONG_THRESHOLD_VALUES) // 2]
        best_bear_nb = best_bull_nb
        best_bear_st = best_bull_st
        best_distribution: Dict[str, float] = {}
        valid_combinations = 0
        total_combinations = 0

        for bull_st in self.STRONG_THRESHOLD_VALUES:
            for bull_nb in self.NEUTRAL_BAND_VALUES:
                if bull_nb >= bull_st:
                    continue
                for bear_st in self.STRONG_THRESHOLD_VALUES:
                    for bear_nb in self.NEUTRAL_BAND_VALUES:
                        if bear_nb >= bear_st:
                            continue
                        total_combinations += 1

                        effective_score, raw_mi, dist = self._evaluate_threshold_combination(
                            train_df[score_col],
                            train_df['outcome_direction'],
                            bullish_neutral_band=bull_nb,
                            bullish_strong_thresh=bull_st,
                            bearish_neutral_band=bear_nb,
                            bearish_strong_thresh=bear_st
                        )

                        if effective_score >= 0:
                            valid_combinations += 1

                        if effective_score > best_score:
                            best_score = effective_score
                            best_mi = raw_mi
                            best_bull_nb = bull_nb
                            best_bull_st = bull_st
                            best_bear_nb = bear_nb
                            best_bear_st = bear_st
                            best_distribution = dist

        if valid_combinations == 0:
            logger.warning(
                f"No valid asymmetric threshold combinations found for {target}! "
                f"Relaxing constraints and using best raw MI."
            )
            # Fallback: kies op raw MI zonder constraints (recompute cheaply)
            best_raw = -1.0
            for bull_st in self.STRONG_THRESHOLD_VALUES:
                for bull_nb in self.NEUTRAL_BAND_VALUES:
                    if bull_nb >= bull_st:
                        continue
                    for bear_st in self.STRONG_THRESHOLD_VALUES:
                        for bear_nb in self.NEUTRAL_BAND_VALUES:
                            if bear_nb >= bear_st:
                                continue
                            _, raw_mi, _ = self._evaluate_threshold_combination(
                                train_df[score_col],
                                train_df['outcome_direction'],
                                bullish_neutral_band=bull_nb,
                                bullish_strong_thresh=bull_st,
                                bearish_neutral_band=bear_nb,
                                bearish_strong_thresh=bear_st
                            )
                            if raw_mi > best_raw:
                                best_raw = raw_mi
                                best_bull_nb, best_bull_st = bull_nb, bull_st
                                best_bear_nb, best_bear_st = bear_nb, bear_st
            best_mi = best_raw
            best_score = best_raw

            states = train_df[score_col].apply(
                lambda x: self.discretize_composite_asymmetric(
                    x,
                    bullish_neutral_band=best_bull_nb,
                    bullish_strong_threshold=best_bull_st,
                    bearish_neutral_band=best_bear_nb,
                    bearish_strong_threshold=best_bear_st
                )
            )
            best_distribution = states.value_counts(normalize=True).to_dict()

        logger.info(
            f"Asymmetric grid search complete. Valid={valid_combinations}/{total_combinations} "
            f"(target={target}, horizon={horizon})"
        )
        logger.info(
            f"Best MI={best_mi:.4f} at "
            f"bull(nb={best_bull_nb}, st={best_bull_st}), "
            f"bear(nb={best_bear_nb}, st={best_bear_st})"
        )
        logger.info(
            f"State distribution: {', '.join(f'{k}={v:.1%}' for k, v in sorted(best_distribution.items()))}"
        )

        # Valideer op test set
        test_target_states = test_df[score_col].apply(
            lambda x: self.discretize_composite_asymmetric(
                x,
                bullish_neutral_band=best_bull_nb,
                bullish_strong_threshold=best_bull_st,
                bearish_neutral_band=best_bear_nb,
                bearish_strong_threshold=best_bear_st
            )
        )
        test_mi = mutual_info_score(test_target_states, test_df['outcome_direction'])
        test_distribution = test_target_states.value_counts(normalize=True).to_dict()
        logger.info(f"Test set MI: {test_mi:.4f}")
        
        # State distribution met optimale thresholds (op volledige data)
        final_states = df[score_col].apply(
            lambda x: self.discretize_composite_asymmetric(
                x,
                bullish_neutral_band=best_bull_nb,
                bullish_strong_threshold=best_bull_st,
                bearish_neutral_band=best_bear_nb,
                bearish_strong_threshold=best_bear_st
            )
        )
        state_distribution = final_states.value_counts(normalize=True).to_dict()
        
        # Bereken diversity score
        active_states = sum(1 for v in state_distribution.values() if v > MIN_STATE_THRESHOLD)
        diversity_score = active_states / 5.0  # Normalized (5 mogelijke states)

        # Legacy compat velden (symmetrische envelope)
        legacy_nb = max(best_bull_nb, best_bear_nb)
        legacy_st = max(best_bull_st, best_bear_st)

        return ThresholdAnalysisResult(
            method='mi_grid',
            horizon=horizon,
            target=target,
            optimal_neutral_band=legacy_nb,
            optimal_strong_threshold=legacy_st,
            optimal_bullish_neutral_band=best_bull_nb,
            optimal_bearish_neutral_band=best_bear_nb,
            optimal_bullish_strong_threshold=best_bull_st,
            optimal_bearish_strong_threshold=best_bear_st,
            score=best_mi,
            score_name='MI',
            metadata={
                'train_mi': best_mi,
                'test_mi': test_mi,
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'grid_mode': 'asymmetric_4d',
                'state_distribution': state_distribution,
                'test_distribution': test_distribution,
                'diversity_score': diversity_score,
                'active_states': active_states,
                'valid_combinations': valid_combinations,
                'total_combinations': total_combinations,
                'diversity_enforced': self.enforce_diversity,
                'neutral_band_values': self.NEUTRAL_BAND_VALUES,
                'strong_threshold_values': self.STRONG_THRESHOLD_VALUES,
                'bullish_neutral_band': best_bull_nb,
                'bearish_neutral_band': best_bear_nb,
                'bullish_strong_threshold': best_bull_st,
                'bearish_strong_threshold': best_bear_st,
            }
        )
    
    def analyze_per_composite(self, horizon: str) -> Dict[str, ThresholdAnalysisResult]:
        """
        Voer MI Grid Search uit voor elke composite apart.
        
        REFACTORED: Gebruikt nu de volledige analyze() methode met diversity constraints.
        
        Returns:
            Dict mapping composite name naar ThresholdAnalysisResult
        """
        logger.info(f"Analyzing per-composite MI for horizon {horizon} with diversity constraints")
        
        # Load data once
        df = self.load_data(horizon)
        
        results = {}
        composites = ['leading', 'coincident', 'confirming']
        
        for composite in composites:
            logger.info(f"Analyzing {composite} composite...")
            result = self.analyze(horizon, target=composite, df=df)
            results[composite] = result
            
            # Log summary
            dist = result.metadata.get('state_distribution', {})
            logger.info(f"  {composite}: MI={result.score:.4f}, "
                       f"neutral_band={result.optimal_neutral_band}, "
                       f"strong_threshold={result.optimal_strong_threshold}, "
                       f"diversity={result.metadata.get('diversity_score', 0):.2f}")
            logger.info(f"    Distribution: {', '.join(f'{k}={v:.1%}' for k, v in sorted(dist.items()))}")
        
        return results
    
    def _generate_heatmap(
        self,
        results_matrix: np.ndarray,
        horizon: str,
        target: str,
        best_neutral: float,
        best_strong: float,
        diversity_matrix: Optional[np.ndarray] = None
    ):
        """
        Genereer en sla MI heatmap op.
        
        Args:
            results_matrix: 2D array met MI scores
            horizon: Horizon naam
            target: Target composite naam
            best_neutral: Optimale neutral_band waarde
            best_strong: Optimale strong_threshold waarde
            diversity_matrix: Optioneel, 2D array met 1=valid, 0=invalid combinaties
        """
        # Maak output directory
        horizon_dir = self.output_dir / horizon / "mi_heatmaps"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Heatmap
        sns.heatmap(
            results_matrix,
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            xticklabels=[f'{v:.2f}' for v in self.NEUTRAL_BAND_VALUES],
            yticklabels=[f'{v:.2f}' for v in self.STRONG_THRESHOLD_VALUES],
            ax=ax
        )
        
        # Markeer invalid combinaties (diversity constraint failures)
        if diversity_matrix is not None:
            for i in range(diversity_matrix.shape[0]):
                for j in range(diversity_matrix.shape[1]):
                    if diversity_matrix[i, j] == 0:
                        # Markeer invalid met X
                        ax.scatter(
                            j + 0.5, i + 0.5,
                            marker='x', s=100, c='gray', alpha=0.5
                        )
        
        # Markeer optimaal punt
        best_j = self.NEUTRAL_BAND_VALUES.index(best_neutral)
        best_i = self.STRONG_THRESHOLD_VALUES.index(best_strong)
        ax.scatter(
            best_j + 0.5, best_i + 0.5,
            marker='*', s=500, c='blue', edgecolors='white', linewidths=2
        )
        
        # Bereken validity info
        valid_count = int(diversity_matrix.sum()) if diversity_matrix is not None else "N/A"
        total_count = results_matrix.size
        
        ax.set_xlabel('COMPOSITE_NEUTRAL_BAND')
        ax.set_ylabel('COMPOSITE_STRONG_THRESHOLD')
        ax.set_title(
            f'Mutual Information Grid Search - {horizon} ({target})\n'
            f'Optimal: neutral_band={best_neutral}, strong_threshold={best_strong}, '
            f'MI={results_matrix[best_i, best_j]:.4f}\n'
            f'Valid combinations: {valid_count}/{total_count} (X = diversity constraint failed)'
        )
        
        # Opslaan
        output_path = horizon_dir / f'mi_heatmap_{horizon}_{target}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved heatmap to {output_path}")
    
    def generate_comparison_plot(self, results: Dict[str, ThresholdAnalysisResult]):
        """
        Genereer vergelijkingsplot tussen horizons.
        """
        if not HAS_PLOTTING:
            logger.warning("Plotting not available")
            return
        
        comparison_dir = self.output_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (horizon, result) in enumerate(results.items()):
            ax = axes[idx]
            matrix = np.array(result.metadata.get('results_matrix', []))
            
            if matrix.size == 0:
                continue
            
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.4f',
                cmap='YlOrRd',
                xticklabels=[f'{v:.2f}' for v in self.NEUTRAL_BAND_VALUES],
                yticklabels=[f'{v:.2f}' for v in self.STRONG_THRESHOLD_VALUES],
                ax=ax,
                cbar=idx == 2  # Alleen rechter plot heeft colorbar
            )
            
            # Markeer optimaal punt
            best_neutral = result.optimal_neutral_band
            best_strong = result.optimal_strong_threshold
            
            if best_neutral in self.NEUTRAL_BAND_VALUES and best_strong in self.STRONG_THRESHOLD_VALUES:
                best_j = self.NEUTRAL_BAND_VALUES.index(best_neutral)
                best_i = self.STRONG_THRESHOLD_VALUES.index(best_strong)
                ax.scatter(
                    best_j + 0.5, best_i + 0.5,
                    marker='*', s=300, c='blue', edgecolors='white', linewidths=2
                )
            
            ax.set_xlabel('NEUTRAL_BAND')
            ax.set_ylabel('STRONG_THRESHOLD' if idx == 0 else '')
            ax.set_title(f'{horizon} (MI={result.score:.4f})')
        
        fig.suptitle('MI Grid Search Comparison Across Horizons', fontsize=14)
        plt.tight_layout()
        
        output_path = comparison_dir / 'horizon_comparison_mi.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved comparison plot to {output_path}")

