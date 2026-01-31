# analysis/decision_tree_analyzer.py
"""
Decision Tree (CART) Analyse voor Threshold Optimalisatie.

Laat het CART algoritme automatisch optimale split-points vinden
op basis van de raw normalized signal scores.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from core.config_defaults import DEFAULT_COMPOSITE_NEUTRAL_BAND, DEFAULT_COMPOSITE_STRONG_THRESHOLD
from .threshold_optimizer import ThresholdOptimizer, ThresholdAnalysisResult

logger = logging.getLogger(__name__)

# Check matplotlib availability (headless mode)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib not available - plotting disabled")


class DecisionTreeAnalyzer(ThresholdOptimizer):
    """
    Decision Tree (CART) Auto-Discovery voor threshold optimalisatie.
    
    Genereert:
    - Tree visualisatie met split points
    - Feature importance bar chart
    - Optimale split points per feature (signal type)
    """
    
    def __init__(
        self,
        asset_id: int,
        lookback_days: int = 180,
        train_ratio: float = 0.8,
        min_samples: int = 5000,
        output_dir: Optional[Path] = None,
        max_depth: int = 4,
        min_samples_leaf: int = 100
    ):
        super().__init__(asset_id, lookback_days, train_ratio, min_samples)
        self.output_dir = output_dir or Path("_validation/threshold_analysis")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
    def analyze(self, horizon: str, target: str = 'leading', df: Optional[pd.DataFrame] = None) -> ThresholdAnalysisResult:
        """
        Voer CART analyse uit voor een specifieke horizon en target.
        
        Args:
            horizon: De te analyseren horizon ('1h', '4h', '1d')
            target: De target composite ('leading', 'coincident', 'confirming' or 'all')
            df: Optioneel reeds geladen DataFrame
            
        Returns:
            ThresholdAnalysisResult met gevonden split points
        """
        logger.info(f"Starting Decision Tree CART analysis for horizon {horizon}, target {target}")
        
        # Load data
        df = self.load_data(horizon, df=df)
        train_df, test_df = self.train_test_split(df)
        
        # REASON: Pas min_samples_leaf aan op basis van de dataset grootte per horizon.
        horizon_min_samples = self.min_samples_leaf
        if horizon == '1d':
            horizon_min_samples = max(10, int(len(train_df) * 0.05))
        elif horizon == '4h':
            horizon_min_samples = max(20, int(len(train_df) * 0.02))
            
        # Prepare features (composite scores)
        feature_cols = ['leading_score', 'coincident_score', 'confirming_score']
        X_train = train_df[feature_cols].values
        y_train = train_df['outcome_direction'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['outcome_direction'].values
        
        # Train decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=horizon_min_samples,
            random_state=42
        )
        tree.fit(X_train, y_train)
        
        # Evalueer
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(tree, X_train, y_train, cv=5)
        
        logger.info(f"Tree trained. Train accuracy: {train_score:.3f}, "
                   f"Test accuracy: {test_score:.3f}, CV mean: {cv_scores.mean():.3f}")
        
        # Extract split points
        split_points = self._extract_split_points(tree, feature_cols)
        
        # Feature importance
        importance = dict(zip(feature_cols, tree.feature_importances_))
        
        # REASON: Als target 'all' is, berekenen we thresholds voor alle kolommen en slaan ze op in metadata.
        all_thresholds = {}
        if target == 'all':
            for col in feature_cols:
                short_name = col.replace('_score', '')
                nb, st, bull_nb, bull_st, bear_nb, bear_st = self._derive_thresholds(split_points, col)
                all_thresholds[short_name] = {
                    'neutral_band': nb,
                    'strong_threshold': st,
                    'bullish_neutral_band': bull_nb,
                    'bullish_strong_threshold': bull_st,
                    'bearish_neutral_band': bear_nb,
                    'bearish_strong_threshold': bear_st
                }
            
            # Default voor de main fields (leading)
            optimal_neutral, optimal_strong = all_thresholds['leading']['neutral_band'], all_thresholds['leading']['strong_threshold']
            bull_nb = all_thresholds['leading']['bullish_neutral_band']
            bull_st = all_thresholds['leading']['bullish_strong_threshold']
            bear_nb = all_thresholds['leading']['bearish_neutral_band']
            bear_st = all_thresholds['leading']['bearish_strong_threshold']
        else:
            target_score_col = f"{target}_score"
            optimal_neutral, optimal_strong, bull_nb, bull_st, bear_nb, bear_st = self._derive_thresholds(split_points, target_score_col)
        
        # Generate visualizations
        if HAS_PLOTTING:
            self._generate_tree_plot(tree, feature_cols, horizon, target)
            self._generate_importance_plot(importance, horizon, target)
        
        return ThresholdAnalysisResult(
            method='cart',
            horizon=horizon,
            target=target,
            optimal_neutral_band=optimal_neutral,
            optimal_strong_threshold=optimal_strong,
            optimal_bullish_neutral_band=bull_nb,
            optimal_bullish_strong_threshold=bull_st,
            optimal_bearish_neutral_band=bear_nb,
            optimal_bearish_strong_threshold=bear_st,
            score=cv_scores.mean(),
            score_name='CV_Accuracy',
            feature_importance=importance,
            metadata={
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'split_points': split_points,
                'tree_depth': tree.get_depth(),
                'n_leaves': tree.get_n_leaves(),
                'max_depth_param': self.max_depth,
                'min_samples_leaf_param': self.min_samples_leaf,
                'all_thresholds': all_thresholds if target == 'all' else None
            }
        )
    
    def _extract_split_points(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        Extraheer alle split points uit de decision tree.
        
        Returns:
            Dict mapping feature name naar lijst van split thresholds
        """
        tree_ = tree.tree_
        split_points = {name: [] for name in feature_names}
        
        def recurse(node_id: int):
            # Als het geen leaf node is
            if tree_.feature[node_id] != -2:  # -2 = leaf
                feature_idx = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                
                feature_name = feature_names[feature_idx]
                split_points[feature_name].append(threshold)
                
                # Recursief door left en right child
                recurse(tree_.children_left[node_id])
                recurse(tree_.children_right[node_id])
        
        recurse(0)  # Start bij root
        
        # Sorteer en deduplicate
        for name in split_points:
            split_points[name] = sorted(set(split_points[name]))
        
        logger.debug(f"Extracted split points: {split_points}")
        return split_points
    
    def _derive_thresholds(
        self,
        split_points: Dict[str, List[float]],
        target_score_col: str = 'leading_score'
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Leid threshold waarden af uit gevonden split points.
        
        Strategie:
        - bullish/bearish drempels apart (asymmetrisch)
        - legacy neutral_band/strong_threshold = max(bullish,bearish) (compat)
        """
        # Focus op target score splits
        target_splits = split_points.get(target_score_col, [])
        
        if not target_splits:
            logger.warning(f"No split points found for {target_score_col}, using defaults")
            nb = DEFAULT_COMPOSITE_NEUTRAL_BAND
            st = DEFAULT_COMPOSITE_STRONG_THRESHOLD
            return nb, st, nb, st, nb, st
        
        pos = sorted([s for s in target_splits if s > 0])
        neg = sorted([abs(s) for s in target_splits if s < 0])
        
        # Bullish kant
        bull_nb_candidates = [s for s in pos if 0.03 < s < 0.30]
        bull_nb = bull_nb_candidates[0] if bull_nb_candidates else DEFAULT_COMPOSITE_NEUTRAL_BAND
        bull_st_candidates = [s for s in pos if s > 0.25]
        bull_st = max(bull_st_candidates) if bull_st_candidates else DEFAULT_COMPOSITE_STRONG_THRESHOLD
        
        # Bearish kant (magnitude)
        bear_nb_candidates = [s for s in neg if 0.03 < s < 0.30]
        bear_nb = bear_nb_candidates[0] if bear_nb_candidates else DEFAULT_COMPOSITE_NEUTRAL_BAND
        bear_st_candidates = [s for s in neg if s > 0.25]
        bear_st = max(bear_st_candidates) if bear_st_candidates else DEFAULT_COMPOSITE_STRONG_THRESHOLD
        
        # Legacy envelope (compat)
        neutral_band = max(bull_nb, bear_nb)
        strong_threshold = max(bull_st, bear_st)
        
        logger.info(
            f"Derived thresholds for {target_score_col}: "
            f"bull(nb={bull_nb:.3f}, st={bull_st:.3f}), "
            f"bear(nb={bear_nb:.3f}, st={bear_st:.3f}), "
            f"legacy(nb={neutral_band:.3f}, st={strong_threshold:.3f})"
        )
        
        return (
            round(neutral_band, 2),
            round(strong_threshold, 2),
            round(bull_nb, 2),
            round(bull_st, 2),
            round(bear_nb, 2),
            round(bear_st, 2)
        )
    
    def _generate_tree_plot(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str],
        horizon: str,
        target: str
    ):
        """Genereer decision tree visualisatie."""
        horizon_dir = self.output_dir / horizon / "decision_tree"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        # Tree plot
        fig, ax = plt.subplots(figsize=(20, 12))
        
        class_names = ['Bearish', 'Neutral', 'Bullish']
        
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        
        ax.set_title(f'Decision Tree - {horizon} ({target})\nDepth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}')
        
        output_path = horizon_dir / f'tree_{horizon}_{target}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved tree plot to {output_path}")
    
    def _generate_importance_plot(
        self,
        importance: Dict[str, float],
        horizon: str,
        target: str
    ):
        """Genereer feature importance bar chart."""
        horizon_dir = self.output_dir / horizon / "decision_tree"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(importance.keys())
        values = list(importance.values())
        
        # Sorteer op importance
        sorted_idx = np.argsort(values)[::-1]
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Leading, Coincident, Confirming
        
        bars = ax.barh(features, values, color=[colors[i % 3] for i in range(len(features))])
        
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title(f'Feature Importance - {horizon} ({target})')
        ax.set_xlim(0, max(values) * 1.1)
        
        # Voeg waarde labels toe
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        
        output_path = horizon_dir / f'feature_importance_{horizon}_{target}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved importance plot to {output_path}")
    
    def analyze_individual_signals(self, horizon: str) -> Dict[str, Dict]:
        """
        Analyseer individuele signalen (niet alleen composites).
        
        Returns:
            Dict met per-signaal importance en split points
        """
        logger.info(f"Analyzing individual signals for horizon {horizon}")
        
        df = self.load_data(horizon)
        train_df, test_df = self.train_test_split(df)
        
        # Alle individuele signal kolommen
        signal_cols = self.get_signal_columns()
        all_signals = []
        for cols in signal_cols.values():
            all_signals.extend(cols)
        
        # Filter voor beschikbare kolommen
        available_signals = [c for c in all_signals if c in train_df.columns]
        
        if not available_signals:
            logger.warning("No individual signal columns found")
            return {}
        
        X_train = train_df[available_signals].values
        y_train = train_df['outcome_direction'].values
        
        # Train tree op individuele signalen
        tree = DecisionTreeClassifier(
            max_depth=6,  # Dieper voor meer detail
            min_samples_leaf=50,
            random_state=42
        )
        tree.fit(X_train, y_train)
        
        # Extract results
        importance = dict(zip(available_signals, tree.feature_importances_))
        split_points = self._extract_split_points(tree, available_signals)
        
        # Top 10 belangrijkste signalen
        top_signals = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'feature_importance': importance,
            'split_points': split_points,
            'top_signals': dict(top_signals),
            'train_accuracy': tree.score(X_train, y_train)
        }

