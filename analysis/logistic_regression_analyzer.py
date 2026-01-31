# analysis/logistic_regression_analyzer.py
"""
Logistic Regression Weight Optimalisatie voor Threshold Analyse.

Optimaliseert signaal-gewichten voor maximale classificatie-performance
en genereert ROC curves en coefficient plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score

from .threshold_optimizer import ThresholdOptimizer, ThresholdAnalysisResult

logger = logging.getLogger(__name__)

# Check matplotlib availability (headless mode)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib not available - plotting disabled")


class LogisticRegressionAnalyzer(ThresholdOptimizer):
    """
    Logistic Regression voor optimale signaal-gewichten.
    
    Genereert:
    - Coefficient plot (horizontale bars)
    - ROC curve met AUC score
    - Optimale gewichten per signaal type
    """
    
    def __init__(
        self,
        asset_id: int,
        lookback_days: int = 180,
        train_ratio: float = 0.8,
        min_samples: int = 5000,
        output_dir: Optional[Path] = None,
        regularization: float = 1.0
    ):
        super().__init__(asset_id, lookback_days, train_ratio, min_samples)
        self.output_dir = output_dir or Path("_validation/threshold_analysis")
        self.regularization = regularization  # C parameter (inverse of regularization strength)
        
    def analyze(self, horizon: str, target: str = 'leading', df: Optional[pd.DataFrame] = None) -> ThresholdAnalysisResult:
        """
        Voer Logistic Regression analyse uit voor een specifieke horizon en target.
        
        Returns:
            ThresholdAnalysisResult met geoptimaliseerde signaal gewichten
        """
        logger.info(f"Starting Logistic Regression analysis for horizon {horizon}, target {target}")
        
        # Load data
        df = self.load_data(horizon, df=df)
        train_df, test_df = self.train_test_split(df)
        
        # Prepare features (composite scores)
        feature_cols = ['leading_score', 'coincident_score', 'confirming_score']
        X_train = train_df[feature_cols].values
        y_train = (train_df['outcome'] > 0).astype(int).values  # Binary: win/loss
        X_test = test_df[feature_cols].values
        y_test = (test_df['outcome'] > 0).astype(int).values
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        model = LogisticRegression(
            C=self.regularization,
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # Evalueer op test set
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model trained. CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f}), "
                   f"Test AUC: {test_auc:.3f}")
        
        # Coefficients (gewichten)
        coefficients = dict(zip(feature_cols, model.coef_[0]))
        
        # Denormalize coefficients voor interpretatie
        denorm_coefficients = {}
        for i, col in enumerate(feature_cols):
            denorm_coefficients[col] = model.coef_[0][i] / scaler.scale_[i]
        
        # Generate visualizations
        if HAS_PLOTTING:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            self._generate_roc_curve(fpr, tpr, test_auc, horizon, target)
            self._generate_coefficient_plot(coefficients, horizon, target)
        
        return ThresholdAnalysisResult(
            method='logreg',
            horizon=horizon,
            target=target,
            score=cv_scores.mean(),
            score_name='AUC',
            signal_weights=denorm_coefficients,
            metadata={
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'test_auc': test_auc,
                'intercept': model.intercept_[0],
                'coefficients_normalized': coefficients,
                'coefficients_denormalized': denorm_coefficients,
                'regularization_C': self.regularization,
                'scaler_means': scaler.mean_.tolist(),
                'scaler_scales': scaler.scale_.tolist()
            }
        )
    
    def analyze_individual_signals(self, horizon: str) -> ThresholdAnalysisResult:
        """
        Analyseer individuele signalen ipv composites.
        
        Returns:
            ThresholdAnalysisResult met per-signaal gewichten
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
            raise ValueError("No individual signal columns found")
        
        X_train = train_df[available_signals].values
        y_train = (train_df['outcome'] > 0).astype(int).values
        X_test = test_df[available_signals].values
        y_test = (test_df['outcome'] > 0).astype(int).values
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train with L1 regularization (sparse coefficients)
        model = LogisticRegression(
            penalty='elasticnet',
            l1_ratio=1.0,
            solver='saga',
            C=self.regularization,
            random_state=42,
            max_iter=2000
        )
        model.fit(X_train_scaled, y_train)
        
        # Evalueer
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Coefficients
        coefficients = dict(zip(available_signals, model.coef_[0]))
        
        # Denormalize
        denorm_coefficients = {}
        for i, col in enumerate(available_signals):
            denorm_coefficients[col] = model.coef_[0][i] / scaler.scale_[i]
        
        # Non-zero coefficients (L1 selectie)
        selected_signals = {k: v for k, v in denorm_coefficients.items() if abs(v) > 0.001}
        
        logger.info(f"L1 selected {len(selected_signals)}/{len(available_signals)} signals")
        
        # Generate coefficient plot
        if HAS_PLOTTING:
            self._generate_individual_coefficient_plot(denorm_coefficients, horizon)
        
        return ThresholdAnalysisResult(
            method='logreg_individual',
            horizon=horizon,
            score=test_auc,
            score_name='AUC',
            signal_weights=selected_signals,
            metadata={
                'test_auc': test_auc,
                'all_coefficients': denorm_coefficients,
                'selected_count': len(selected_signals),
                'total_signals': len(available_signals)
            }
        )
    
    def _generate_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        horizon: str,
        target: str
    ):
        """Genereer ROC curve plot."""
        horizon_dir = self.output_dir / horizon / "logistic_regression"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # ROC curve
        ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        
        # Diagonal reference
        ax.plot([0, 1], [0, 1], color='#95a5a6', linestyle='--', lw=1, label='Random (AUC = 0.50)')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {horizon} ({target})\nLogistic Regression on Composite Scores')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # AUC fill
        ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
        
        output_path = horizon_dir / f'roc_curve_{horizon}_{target}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved ROC curve to {output_path}")
    
    def _generate_coefficient_plot(
        self,
        coefficients: Dict[str, float],
        horizon: str,
        target: str
    ):
        """Genereer coefficient bar chart voor composites."""
        horizon_dir = self.output_dir / horizon / "logistic_regression"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(coefficients.keys())
        values = list(coefficients.values())
        
        # Kleuren gebaseerd op teken
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        bars = ax.barh(features, values, color=colors)
        
        # Zero line
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        ax.set_xlabel('Coefficient (normalized)')
        ax.set_title(f'Logistic Regression Coefficients - {horizon} ({target})')
        
        # Waarde labels
        for bar, val in zip(bars, values):
            x_pos = val + 0.02 if val > 0 else val - 0.02
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha=ha, fontsize=10)
        
        output_path = horizon_dir / f'coefficients_{horizon}_{target}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved coefficient plot to {output_path}")
    
    def _generate_individual_coefficient_plot(
        self,
        coefficients: Dict[str, float],
        horizon: str
    ):
        """Genereer coefficient bar chart voor individuele signalen."""
        horizon_dir = self.output_dir / horizon / "logistic_regression"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sorteer op absolute waarde
        sorted_coef = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [x[0] for x in sorted_coef]
        values = [x[1] for x in sorted_coef]
        
        # Kleuren gebaseerd op teken
        colors = ['#2ecc71' if v > 0 else '#e74c3c' if v < 0 else '#95a5a6' for v in values]
        
        bars = ax.barh(range(len(features)), values, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        
        # Zero line
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        ax.set_xlabel('Coefficient (denormalized)')
        ax.set_title(f'Individual Signal Coefficients (L1 Regularized) - {horizon}')
        
        # Highlight non-zero signals
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.001:
                x_pos = val + 0.01 if val > 0 else val - 0.01
                ha = 'left' if val > 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', ha=ha, fontsize=8)
        
        output_path = horizon_dir / f'individual_coefficients_{horizon}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved individual coefficient plot to {output_path}")
    
    def compute_optimal_weights(self, horizon: str) -> Dict[str, float]:
        """
        Bereken optimale gewichten die direct in signal_aggregator kunnen worden gebruikt.
        
        Returns:
            Dict mapping signal_name naar optimaal gewicht
        """
        result = self.analyze_individual_signals(horizon)
        
        # Scale naar bruikbare gewichten (0.5 - 2.5 range zoals in huidige systeem)
        raw_weights = result.signal_weights
        
        if not raw_weights:
            return {}
        
        # Normaliseer: map max naar 2.5, min naar 0.5, zero naar 1.0
        max_abs = max(abs(v) for v in raw_weights.values())
        
        optimal_weights = {}
        for signal, coef in raw_weights.items():
            if max_abs > 0:
                # Scale naar [-1.5, +1.5] en shift naar [0.5, 2.5]
                scaled = (coef / max_abs) * 1.0 + 1.0  # Maps to [0, 2]
                scaled = max(0.5, min(2.5, scaled))  # Clamp
            else:
                scaled = 1.0
            optimal_weights[signal] = round(scaled, 2)
        
        return optimal_weights

