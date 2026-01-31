# validation/threshold_validation_report.py
"""
Markdown Rapport Generator voor Threshold Optimalisatie.

Genereert een gecombineerd rapport met alle analyse resultaten,
grafieken, en aanbevelingen.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ThresholdValidationReport:
    """Generator voor Threshold Optimalisatie rapporten."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, results: Dict[str, Any]) -> Path:
        """
        Genereer Markdown rapport.
        
        Args:
            results: Dict met analyse resultaten per horizon en methode
            
        Returns:
            Path naar gegenereerd rapport
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        asset_id = results.get('asset_id', 'unknown')
        report_path = self.output_dir / f'threshold_report_{asset_id}_{timestamp}.md'
        
        lines = []
        
        # Header
        lines.extend(self._generate_header(results))
        
        # Executive Summary
        lines.extend(self._generate_executive_summary(results))
        
        # Per-horizon details
        for horizon in ['1h', '4h', '1d']:
            if horizon in results.get('horizons', {}):
                lines.extend(self._generate_horizon_section(horizon, results['horizons'][horizon]))
        
        # Recommendations
        lines.extend(self._generate_recommendations(results))
        
        # Methodology
        lines.extend(self._generate_methodology())
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated report: {report_path}")
        return report_path
    
    def _generate_header(self, results: Dict) -> list:
        """Genereer rapport header."""
        return [
            "# Threshold Optimalisatie Rapport",
            "",
            f"**Asset ID:** {results.get('asset_id', 'N/A')}",
            f"**Lookback:** {results.get('lookback_days', 'N/A')} dagen",
            f"**Analyse Timestamp:** {results.get('analysis_timestamp', 'N/A')}",
            "",
            "---",
            ""
        ]
    
    def _generate_executive_summary(self, results: Dict) -> list:
        """Genereer executive summary tabel."""
        lines = [
            "## Executive Summary",
            "",
            "### Optimale Thresholds per Horizon",
            ""
        ]
        
        summary = results.get('summary', {})
        
        for horizon in ['1h', '4h', '1d']:
            if horizon not in summary:
                continue
            
            lines.append(f"#### {horizon} Horizon")
            lines.append("")
            lines.append("| Methode | NEUTRAL_BAND | STRONG_THRESHOLD | Score |")
            lines.append("|---------|--------------|------------------|-------|")
            
            horizon_data = summary[horizon]
            for method, data in horizon_data.items():
                neutral = data.get('neutral_band', 'N/A')
                strong = data.get('strong_threshold', 'N/A')
                score = data.get('score', 0)
                score_name = data.get('score_name', '')
                
                if neutral != 'N/A':
                    lines.append(f"| {method.upper()} | {neutral} | {strong} | {score_name}={score:.4f} |")
                else:
                    lines.append(f"| {method.upper()} | N/A | N/A | {score_name}={score:.4f} |")
            
            lines.extend(["", ""])
        
        return lines
    
    def _generate_horizon_section(self, horizon: str, horizon_results: Dict) -> list:
        """Genereer gedetailleerde sectie per horizon."""
        lines = [
            f"## {horizon} Horizon - Gedetailleerde Analyse",
            "",
            "---",
            ""
        ]
        
        # MI Grid Search
        if 'mi_grid' in horizon_results:
            lines.extend(self._format_mi_results(horizon, horizon_results['mi_grid']))
        
        # CART
        if 'cart' in horizon_results:
            lines.extend(self._format_cart_results(horizon, horizon_results['cart']))
        
        # LogReg
        if 'logreg' in horizon_results:
            lines.extend(self._format_logreg_results(horizon, horizon_results['logreg']))
        
        return lines
    
    def _format_mi_results(self, horizon: str, data: Dict) -> list:
        """Format MI Grid Search resultaten."""
        if 'error' in data:
            return [f"### MI Grid Search - {horizon}", "", f"**Error:** {data['error']}", ""]
        
        lines = [
            f"### MI Grid Search - {horizon}",
            "",
            f"**Optimale Parameters:**",
            f"- NEUTRAL_BAND: {data.get('optimal_neutral_band', 'N/A')}",
            f"- STRONG_THRESHOLD: {data.get('optimal_strong_threshold', 'N/A')}",
            f"- Mutual Information: {data.get('score', 0):.4f}",
            "",
        ]
        
        metadata = data.get('metadata', {})
        if metadata:
            lines.extend([
                "**Statistieken:**",
                f"- Train MI: {metadata.get('train_mi', 0):.4f}",
                f"- Test MI: {metadata.get('test_mi', 0):.4f}",
                f"- Train samples: {metadata.get('train_samples', 0):,}",
                f"- Test samples: {metadata.get('test_samples', 0):,}",
                ""
            ])
        
        # State distribution
        state_dist = metadata.get('state_distribution', {})
        if state_dist:
            lines.extend([
                "**State Distribution (met optimale thresholds):**",
                ""
            ])
            for state, pct in sorted(state_dist.items()):
                lines.append(f"- {state}: {pct*100:.1f}%")
            lines.append("")
        
        # Heatmap reference
        lines.extend([
            "**Visualisatie:**",
            "",
            f"![MI Heatmap]({horizon}/mi_heatmaps/mi_heatmap_{horizon}.png)",
            ""
        ])
        
        return lines
    
    def _format_cart_results(self, horizon: str, data: Dict) -> list:
        """Format CART resultaten."""
        if 'error' in data:
            return [f"### Decision Tree (CART) - {horizon}", "", f"**Error:** {data['error']}", ""]
        
        lines = [
            f"### Decision Tree (CART) - {horizon}",
            "",
            f"**Afgeleide Parameters:**",
            f"- NEUTRAL_BAND: {data.get('optimal_neutral_band', 'N/A')}",
            f"- STRONG_THRESHOLD: {data.get('optimal_strong_threshold', 'N/A')}",
            f"- CV Accuracy: {data.get('score', 0):.4f}",
            "",
        ]
        
        metadata = data.get('metadata', {})
        if metadata:
            lines.extend([
                "**Model Statistieken:**",
                f"- Train accuracy: {metadata.get('train_accuracy', 0):.4f}",
                f"- Test accuracy: {metadata.get('test_accuracy', 0):.4f}",
                f"- CV mean: {metadata.get('cv_mean', 0):.4f} (+/- {metadata.get('cv_std', 0):.4f})",
                f"- Tree depth: {metadata.get('tree_depth', 'N/A')}",
                f"- Number of leaves: {metadata.get('n_leaves', 'N/A')}",
                ""
            ])
        
        # Feature importance
        importance = data.get('feature_importance', {})
        if importance:
            lines.extend([
                "**Feature Importance (Gini):**",
                ""
            ])
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {feature}: {imp:.4f}")
            lines.append("")
        
        # Split points
        split_points = metadata.get('split_points', {})
        if split_points:
            lines.extend([
                "**Gevonden Split Points:**",
                ""
            ])
            for feature, splits in split_points.items():
                if splits:
                    splits_str = ', '.join([f'{s:.3f}' for s in splits])
                    lines.append(f"- {feature}: [{splits_str}]")
            lines.append("")
        
        # Visualisaties
        lines.extend([
            "**Visualisaties:**",
            "",
            f"![Decision Tree]({horizon}/decision_tree/tree_{horizon}.png)",
            "",
            f"![Feature Importance]({horizon}/decision_tree/feature_importance_{horizon}.png)",
            ""
        ])
        
        return lines
    
    def _format_logreg_results(self, horizon: str, data: Dict) -> list:
        """Format Logistic Regression resultaten."""
        if 'error' in data:
            return [f"### Logistic Regression - {horizon}", "", f"**Error:** {data['error']}", ""]
        
        lines = [
            f"### Logistic Regression - {horizon}",
            "",
            f"**Performance:**",
            f"- AUC: {data.get('score', 0):.4f}",
            "",
        ]
        
        metadata = data.get('metadata', {})
        if metadata:
            lines.extend([
                "**Statistieken:**",
                f"- CV AUC mean: {metadata.get('cv_auc_mean', 0):.4f} (+/- {metadata.get('cv_auc_std', 0):.4f})",
                f"- Test AUC: {metadata.get('test_auc', 0):.4f}",
                f"- Regularization C: {metadata.get('regularization_C', 'N/A')}",
                ""
            ])
        
        # Coefficients
        weights = data.get('signal_weights', {})
        if weights:
            lines.extend([
                "**Geoptimaliseerde Gewichten (denormalized):**",
                ""
            ])
            for signal, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"- {signal}: {weight:.4f}")
            lines.append("")
        
        # Visualisaties
        lines.extend([
            "**Visualisaties:**",
            "",
            f"![ROC Curve]({horizon}/logistic_regression/roc_curve_{horizon}.png)",
            "",
            f"![Coefficients]({horizon}/logistic_regression/coefficients_{horizon}.png)",
            ""
        ])
        
        return lines
    
    def _generate_recommendations(self, results: Dict) -> list:
        """Genereer aanbevelingen op basis van resultaten."""
        lines = [
            "## Aanbevelingen",
            "",
            "---",
            ""
        ]
        
        summary = results.get('summary', {})
        
        # Analyseer consistentie tussen methoden
        for horizon in ['1h', '4h', '1d']:
            if horizon not in summary:
                continue
            
            horizon_data = summary[horizon]
            
            # Verzamel neutral_band waarden
            neutral_values = []
            strong_values = []
            
            for method, data in horizon_data.items():
                if data.get('neutral_band'):
                    neutral_values.append(data['neutral_band'])
                if data.get('strong_threshold'):
                    strong_values.append(data['strong_threshold'])
            
            if neutral_values:
                avg_neutral = sum(neutral_values) / len(neutral_values)
                avg_strong = sum(strong_values) / len(strong_values) if strong_values else 0.50
                
                lines.extend([
                    f"### {horizon} Horizon",
                    "",
                    f"**Aanbevolen configuratie:**",
                    f"- COMPOSITE_NEUTRAL_BAND: {avg_neutral:.2f}",
                    f"- COMPOSITE_STRONG_THRESHOLD: {avg_strong:.2f}",
                    ""
                ])
                
                # Check consistentie
                if neutral_values:
                    spread = max(neutral_values) - min(neutral_values)
                    if spread > 0.05:
                        lines.append(f"⚠️ Let op: Methoden geven verschillende neutral_band waarden "
                                   f"(spread: {spread:.2f}). Overweeg verdere analyse.")
                    else:
                        lines.append(f"✓ Methoden zijn consistent (spread: {spread:.2f})")
                
                lines.append("")
        
        # Algemene aanbevelingen
        lines.extend([
            "### Algemene Opmerkingen",
            "",
            "1. **MI Grid Search** geeft de meest directe maat voor informatiewinst",
            "2. **CART** is het meest interpreteerbaar en toont interactie-effecten",
            "3. **LogReg** is het beste voor het optimaliseren van signaal-gewichten",
            "",
            "**Volgende stappen:**",
            "- Pas de aanbevolen thresholds toe met `--apply-results`",
            "- Voer een walk-forward validatie uit om de impact te meten",
            "- Monitor de state distribution na aanpassing",
            ""
        ])
        
        return lines
    
    def _generate_methodology(self) -> list:
        """Genereer methodologie sectie."""
        return [
            "## Methodologie",
            "",
            "---",
            "",
            "### Data Selectie",
            "",
            "- Alleen 60-minuut boundary data (geen 1-minuut sampling)",
            "- Temporele train/test split (80/20) om lookahead bias te voorkomen",
            "- Minimum 5000 samples vereist voor statistische significantie",
            "",
            "### Mutual Information Grid Search",
            "",
            "Grid search over alle combinaties van NEUTRAL_BAND en STRONG_THRESHOLD.",
            "Meet de mutual information tussen gediscretiseerde composite states en outcome direction.",
            "",
            "### Decision Tree (CART)",
            "",
            "Laat het CART algoritme automatisch optimale split-points vinden.",
            "Beperkt tot max_depth=4 en min_samples_leaf=100 om overfitting te voorkomen.",
            "",
            "### Logistic Regression",
            "",
            "L2-geregulariseerde logistische regressie voor binaire win/loss classificatie.",
            "Coefficienten worden gedenormaliseerd voor directe interpretatie.",
            "",
            "---",
            "",
            "*Rapport gegenereerd door QBN v3 Threshold Optimizer*",
            ""
        ]

