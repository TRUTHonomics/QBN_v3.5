"""
Combination Alpha Visualizations - Forest Plot + Sensitivity/Specificity Scatter.

ARCHITECTUUR NOOT:
- Forest Plot: Toont OR met 95% CI per combinatie
- Sens/Spec Scatter: Trade-off tussen sensitivity en specificity
- Alle plots worden opgeslagen naar _validation/combination_alpha/

Gebruik:
    from validation.combination_visualizations import CombinationVisualizer
    
    viz = CombinationVisualizer(output_dir)
    viz.create_forest_plot(results, horizon='1h')
    viz.create_sens_spec_scatter(results)
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Import result type
try:
    from analysis.combination_alpha_analyzer import CombinationResult, AnalysisResult
except ImportError:
    CombinationResult = Any
    AnalysisResult = Any


class CombinationVisualizer:
    """
    Creates visualizations for combination alpha analysis results.
    """
    
    # Color scheme
    COLORS = {
        'golden_rule': '#2ecc71',    # Green
        'promising': '#f39c12',       # Orange
        'noise': '#95a5a6',           # Gray
        'ci_bar': '#3498db',          # Blue
        'reference': '#e74c3c',       # Red (OR=1 line)
        'bootstrap': '#9b59b6',       # Purple
    }
    
    # Style settings
    FIGURE_DPI = 150
    FONT_SIZE = 10
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize Combination Visualizer.
        
        Args:
            output_dir: Directory for saving plots (default: _validation/combination_alpha)
        """
        if output_dir is None:
            output_dir = Path('_validation/combination_alpha')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = self.FONT_SIZE
        
        logger.info(f"CombinationVisualizer initialized, output: {self.output_dir}")
    
    def create_forest_plot(
        self,
        results: List[CombinationResult],
        horizon: str,
        title: Optional[str] = None,
        max_combinations: int = 50,
        show_bootstrap: bool = True
    ) -> Path:
        """
        Create Forest Plot showing OR with 95% CI per combination.
        
        Args:
            results: List of CombinationResult
            horizon: '1h', '4h', or '1d'
            title: Plot title (auto-generated if None)
            max_combinations: Maximum combinations to show
            show_bootstrap: Show bootstrap CI if available
            
        Returns:
            Path to saved figure
        """
        # Filter by horizon
        horizon_results = [r for r in results if r.horizon == horizon]
        
        # Sort by OR (descending) and limit
        horizon_results = sorted(
            horizon_results,
            key=lambda x: x.odds_ratio,
            reverse=True
        )[:max_combinations]
        
        if not horizon_results:
            logger.warning(f"No results for horizon {horizon}")
            return None
        
        n_combinations = len(horizon_results)
        
        # Create figure
        fig_height = max(6, n_combinations * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_height), dpi=self.FIGURE_DPI)
        
        # Y positions
        y_positions = np.arange(n_combinations)
        
        # Plot each combination
        for i, result in enumerate(horizon_results):
            y = n_combinations - 1 - i  # Reverse order (highest OR at top)
            
            # Get color based on classification
            color = self.COLORS.get(result.classification, self.COLORS['noise'])
            
            # Plot point estimate
            ax.plot(
                result.odds_ratio, y,
                'o', color=color, markersize=8, zorder=3
            )
            
            # Plot parametric CI
            ax.hlines(
                y, result.or_ci_lower, result.or_ci_upper,
                colors=self.COLORS['ci_bar'], linewidth=2, alpha=0.7, zorder=2
            )
            
            # Plot bootstrap CI if available
            if show_bootstrap and result.bootstrap_ci_lower is not None:
                ax.hlines(
                    y - 0.15, result.bootstrap_ci_lower, result.bootstrap_ci_upper,
                    colors=self.COLORS['bootstrap'], linewidth=1.5, 
                    linestyles='dashed', alpha=0.7, zorder=2
                )
            
            # Add label
            label = self._format_combination_label(result)
            ax.text(
                0.02, y, label,
                transform=ax.get_yaxis_transform(),
                fontsize=8, va='center', ha='left'
            )
        
        # Reference line at OR = 1
        ax.axvline(
            x=1.0, color=self.COLORS['reference'],
            linestyle='--', linewidth=1.5, alpha=0.7, label='OR = 1.0 (no effect)'
        )
        
        # Reference line at OR = 2 (clinical relevance)
        ax.axvline(
            x=2.0, color=self.COLORS['golden_rule'],
            linestyle=':', linewidth=1.5, alpha=0.5, label='OR = 2.0 (clinical relevance)'
        )
        
        # Formatting
        ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
        ax.set_ylabel('Combination', fontsize=12)
        
        if title is None:
            title = f'Forest Plot - Combination Alpha Analysis ({horizon})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Y-axis ticks
        ax.set_yticks([])
        
        # X-axis log scale for better visualization
        ax.set_xscale('log')
        ax.set_xlim(left=0.1)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.COLORS['golden_rule'], label='Golden Rule'),
            mpatches.Patch(color=self.COLORS['promising'], label='Promising'),
            mpatches.Patch(color=self.COLORS['noise'], label='Noise'),
            plt.Line2D([0], [0], color=self.COLORS['ci_bar'], linewidth=2, label='Parametric 95% CI'),
        ]
        if show_bootstrap:
            legend_elements.append(
                plt.Line2D([0], [0], color=self.COLORS['bootstrap'], 
                          linewidth=1.5, linestyle='dashed', label='Bootstrap 95% CI')
            )
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f'forest_plot_{horizon}.png'
        plt.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forest plot saved to {output_path}")
        return output_path
    
    def create_sens_spec_scatter(
        self,
        results: List[CombinationResult],
        horizon: Optional[str] = None,
        title: Optional[str] = None
    ) -> Path:
        """
        Create Sensitivity vs Specificity scatter plot.
        
        Shows trade-off between sensitivity and specificity.
        Diagonal line represents "no discrimination" (random classifier).
        
        Args:
            results: List of CombinationResult
            horizon: Filter by horizon (None = all)
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        # Filter by horizon if specified
        if horizon:
            results = [r for r in results if r.horizon == horizon]
        
        if not results:
            logger.warning("No results for sens/spec scatter")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.FIGURE_DPI)
        
        # Group by classification
        for classification in ['golden_rule', 'promising', 'noise']:
            class_results = [r for r in results if r.classification == classification]
            if not class_results:
                continue
            
            sens = [r.sensitivity for r in class_results]
            spec = [r.specificity for r in class_results]
            
            ax.scatter(
                spec, sens,
                c=self.COLORS[classification],
                s=100, alpha=0.7,
                label=f'{classification.replace("_", " ").title()} (n={len(class_results)})',
                edgecolors='white', linewidth=0.5
            )
        
        # Diagonal line (random classifier)
        ax.plot(
            [0, 1], [0, 1],
            'k--', alpha=0.3, label='Random classifier'
        )
        
        # Perfect classifier point
        ax.plot(
            1, 1, 'r*', markersize=15, label='Perfect classifier'
        )
        
        # Formatting
        ax.set_xlabel('Specificity (True Negative Rate)', fontsize=12)
        ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
        
        if title is None:
            horizon_str = f' - {horizon}' if horizon else ' - All Horizons'
            title = f'Sensitivity vs Specificity{horizon_str}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        suffix = f'_{horizon}' if horizon else '_all'
        output_path = self.output_dir / f'sens_spec_scatter{suffix}.png'
        plt.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sens/Spec scatter saved to {output_path}")
        return output_path
    
    def create_or_heatmap(
        self,
        results: List[CombinationResult],
        horizon: str,
        title: Optional[str] = None
    ) -> Path:
        """
        Create heatmap of Odds Ratios by combination components.
        
        Args:
            results: List of CombinationResult
            horizon: Filter by horizon
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        # Filter by horizon
        horizon_results = [r for r in results if r.horizon == horizon]
        
        if not horizon_results:
            logger.warning(f"No results for heatmap {horizon}")
            return None
        
        # Build matrix for leading x coincident (averaged over confirming)
        states = ['strong_bearish', 'bearish', 'neutral', 'bullish', 'strong_bullish']
        matrix = np.zeros((5, 5))
        counts = np.zeros((5, 5))
        
        for result in horizon_results:
            parts = result.combination_key.split('|')
            l_idx = states.index(parts[0])
            c_idx = states.index(parts[1])
            
            matrix[l_idx, c_idx] += result.odds_ratio
            counts[l_idx, c_idx] += 1
        
        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.where(counts > 0, matrix / counts, np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.FIGURE_DPI)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=3.0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Odds Ratio', fontsize=11)
        
        # Labels
        state_labels = ['Strong↓', 'Bearish', 'Neutral', 'Bullish', 'Strong↑']
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(state_labels)
        ax.set_yticklabels(state_labels)
        
        ax.set_xlabel('Coincident State', fontsize=12)
        ax.set_ylabel('Leading State', fontsize=12)
        
        if title is None:
            title = f'Odds Ratio Heatmap (Averaged over Confirming) - {horizon}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                if not np.isnan(matrix[i, j]):
                    text_color = 'white' if matrix[i, j] > 2.0 or matrix[i, j] < 0.7 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=9, color=text_color)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'or_heatmap_{horizon}.png'
        plt.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"OR heatmap saved to {output_path}")
        return output_path
    
    def create_summary_dashboard(
        self,
        analysis_result: AnalysisResult
    ) -> Path:
        """
        Create summary dashboard with multiple subplots.
        
        Args:
            analysis_result: Full AnalysisResult object
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.FIGURE_DPI)
        
        # Layout: 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        all_results = analysis_result.all_results()
        
        # 1. Classification pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_classification_pie(ax1, analysis_result)
        
        # 2-4. Forest plots per horizon (simplified)
        for i, (horizon, results) in enumerate([
            ('1h', analysis_result.results_1h),
            ('4h', analysis_result.results_4h),
            ('1d', analysis_result.results_1d)
        ]):
            ax = fig.add_subplot(gs[0, 1] if i == 0 else gs[0, 2] if i == 1 else gs[1, 0])
            self._plot_mini_forest(ax, results, horizon)
        
        # 5. OR distribution histogram
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_or_distribution(ax5, all_results)
        
        # 6. Stats summary table
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_stats_table(ax6, analysis_result)
        
        # Main title
        fig.suptitle(
            f'Combination Alpha Analysis - Asset {analysis_result.asset_id}\n'
            f'{analysis_result.target_type.title()} Target | '
            f'{analysis_result.lookback_days} Days | '
            f'{analysis_result.timestamp.strftime("%Y-%m-%d %H:%M")}',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        output_path = self.output_dir / f'dashboard_asset_{analysis_result.asset_id}.png'
        plt.savefig(output_path, dpi=self.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dashboard saved to {output_path}")
        return output_path
    
    def _plot_classification_pie(self, ax, result: AnalysisResult):
        """Plot classification distribution pie chart."""
        sizes = [result.n_golden_rules, result.n_promising, result.n_noise]
        labels = ['Golden Rules', 'Promising', 'Noise']
        colors = [self.COLORS['golden_rule'], self.COLORS['promising'], self.COLORS['noise']]
        
        # Filter out zeros
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        
        ax.set_title('Classification Distribution', fontsize=11, fontweight='bold')
    
    def _plot_mini_forest(self, ax, results: List[CombinationResult], horizon: str):
        """Plot simplified forest plot for a horizon."""
        # Sort by OR, take top 10
        sorted_results = sorted(results, key=lambda x: x.odds_ratio, reverse=True)[:10]
        
        if not sorted_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{horizon}', fontsize=11, fontweight='bold')
            return
        
        n = len(sorted_results)
        y_pos = np.arange(n)
        
        for i, r in enumerate(sorted_results):
            y = n - 1 - i
            color = self.COLORS.get(r.classification, self.COLORS['noise'])
            ax.plot(r.odds_ratio, y, 'o', color=color, markersize=6)
            ax.hlines(y, r.or_ci_lower, r.or_ci_upper, colors=color, linewidth=1, alpha=0.7)
        
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlim(left=0.1)
        ax.set_yticks([])
        ax.set_xlabel('OR', fontsize=9)
        ax.set_title(f'Top 10 - {horizon}', fontsize=11, fontweight='bold')
    
    def _plot_or_distribution(self, ax, results: List[CombinationResult]):
        """Plot OR distribution histogram."""
        ors = [r.odds_ratio for r in results]
        
        ax.hist(ors, bins=30, color=self.COLORS['ci_bar'], alpha=0.7, edgecolor='white')
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='OR=1')
        ax.axvline(x=2, color='green', linestyle=':', linewidth=1.5, label='OR=2')
        
        ax.set_xlabel('Odds Ratio', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('OR Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    def _plot_stats_table(self, ax, result: AnalysisResult):
        """Plot statistics summary table."""
        ax.axis('off')
        
        all_results = result.all_results()
        
        # Calculate stats
        ors = [r.odds_ratio for r in all_results]
        
        data = [
            ['Total Combinations', str(result.n_total_combinations)],
            ['Golden Rules', str(result.n_golden_rules)],
            ['Promising', str(result.n_promising)],
            ['Noise', str(result.n_noise)],
            ['', ''],
            ['Mean OR', f'{np.mean(ors):.2f}'],
            ['Median OR', f'{np.median(ors):.2f}'],
            ['Max OR', f'{np.max(ors):.2f}'],
            ['', ''],
            ['Analysis Time', f'{result.total_time_seconds:.1f}s'],
            ['Bootstrap N', str(result.n_bootstrap)],
        ]
        
        table = ax.table(
            cellText=data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax.set_title('Summary Statistics', fontsize=11, fontweight='bold', y=0.9)
    
    def _format_combination_label(self, result: CombinationResult) -> str:
        """Format combination key for display."""
        parts = result.combination_key.split('|')
        short_names = {
            'strong_bearish': 'S↓',
            'bearish': '↓',
            'neutral': '—',
            'bullish': '↑',
            'strong_bullish': 'S↑'
        }
        short = [short_names.get(p, p) for p in parts]
        return f"[{'/'.join(short)}] n={result.n_with_combination}"


def create_visualizer(output_dir: Optional[Path] = None) -> CombinationVisualizer:
    """Factory function voor CombinationVisualizer."""
    return CombinationVisualizer(output_dir)

