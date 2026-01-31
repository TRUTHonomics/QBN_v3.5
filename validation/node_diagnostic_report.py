"""
Node Diagnostic Report Generator voor QBN v3.

Genereert console output en Markdown rapporten van node diagnostics.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from validation.node_diagnostics import NodeDiagnosticResult


def print_console_diagnostic(
    asset_id: int,
    results: Dict[str, NodeDiagnosticResult],
    report_path: Optional[Path] = None
):
    """
    Print diagnostic resultaten naar console met visuele formatting.
    """
    print()
    print("=" * 70)
    print(f"NODE-LEVEL DIAGNOSTIC - Asset {asset_id}")
    print("=" * 70)
    print()
    
    # Summary counts
    pass_count = sum(1 for r in results.values() if r.status == 'PASS')
    warn_count = sum(1 for r in results.values() if r.status == 'WARN')
    fail_count = sum(1 for r in results.values() if r.status == 'FAIL')
    
    print(f"Summary: {pass_count} PASS | {warn_count} WARN | {fail_count} FAIL")
    print()
    
    # Group by layer
    layers = [
        ("STRUCTURAL LAYER", ['htf_regime']),
        ("TACTICAL LAYER", ['leading_composite', 'coincident_composite', 'confirming_composite']),
        ("ENTRY LAYER", ['trade_hypothesis']),
        ("TIMING LAYER", ['entry_confidence']),
        ("MANAGEMENT LAYER", ['position_confidence']),
        ("PREDICTION LAYER", ['prediction_1h', 'prediction_4h', 'prediction_1d']),
    ]
    
    for layer_name, node_keys in layers:
        layer_results = [results.get(k) for k in node_keys if k in results]
        if not layer_results:
            continue
        
        print(f"[{layer_name}]")
        
        for r in layer_results:
            if r is None:
                continue
            
            status_icon = _status_icon(r.status)
            print(f"  {r.node_name}")
            print(f"    Status: {status_icon} {r.status}")
            
            # State distribution
            if r.state_distribution:
                print(f"    State Distribution:")
                for state, pct in sorted(r.state_distribution.items(), key=lambda x: -x[1]):
                    bar = _make_bar(pct, 20)
                    stuck_mark = " ⚠️ STUCK" if r.stuck_state == state else ""
                    print(f"      {state:<20} {pct:>6.1%} {bar}{stuck_mark}")
            
            # MI scores
            if any([r.mi_1h, r.mi_4h, r.mi_1d]):
                mi_parts = []
                if r.mi_1h is not None:
                    mi_parts.append(f"1h:{r.mi_1h:.3f}")
                if r.mi_4h is not None:
                    mi_parts.append(f"4h:{r.mi_4h:.3f}")
                if r.mi_1d is not None:
                    mi_parts.append(f"1d:{r.mi_1d:.3f}")
                print(f"    Mutual Information: {' | '.join(mi_parts)}")
            
            # Directional alignment
            if r.directional_alignment:
                print(f"    Directional Alignment:")
                for key, val in r.directional_alignment.items():
                    if isinstance(val, float):
                        icon = "✅" if val > 0.55 else "⚠️" if val > 0.45 else "❌"
                        print(f"      {key}: {val:.0%} {icon}")
            
            # Issues
            if r.issues:
                print(f"    Issues:")
                for issue in r.issues:
                    print(f"      ⚠️ {issue}")
            
            print()
    
    # Overall verdict
    print("=" * 70)
    if fail_count > 0:
        print(f"VERDICT: ❌ {fail_count} node(s) failing - investigate issues above")
    elif warn_count > 0:
        print(f"VERDICT: ⚠️ {warn_count} warning(s) - review before production")
    else:
        print("VERDICT: ✅ All nodes passing")
    print("=" * 70)
    
    if report_path:
        print(f"\nRapport opgeslagen: {report_path}")


def generate_markdown_report(
    asset_id: int,
    results: Dict[str, NodeDiagnosticResult],
    output_dir: Optional[Path] = None
) -> Path:
    """
    Genereer Markdown rapport van diagnostic resultaten.
    
    Returns:
        Path naar gegenereerd rapport
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / '_validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"node_diagnostic_asset_{asset_id}_{timestamp}.md"
    
    lines = []
    
    # Header
    lines.append(f"# Node-Level Diagnostic Report - Asset {asset_id}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    pass_count = sum(1 for r in results.values() if r.status == 'PASS')
    warn_count = sum(1 for r in results.values() if r.status == 'WARN')
    fail_count = sum(1 for r in results.values() if r.status == 'FAIL')
    
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ✅ PASS | {pass_count} |")
    lines.append(f"| ⚠️ WARN | {warn_count} |")
    lines.append(f"| ❌ FAIL | {fail_count} |")
    lines.append("")
    
    # Verdict
    if fail_count > 0:
        lines.append(f"**Verdict:** ❌ **{fail_count} node(s) failing** - Critical issues found")
    elif warn_count > 0:
        lines.append(f"**Verdict:** ⚠️ **{warn_count} warning(s)** - Review recommended")
    else:
        lines.append("**Verdict:** ✅ **All nodes passing**")
    lines.append("")
    
    # Results table
    lines.append("## Results Overview")
    lines.append("")
    lines.append("| Node | Status | MI (best) | Key Issue |")
    lines.append("|------|--------|-----------|-----------|")
    
    for key, r in results.items():
        status_md = _status_icon(r.status) + " " + r.status
        
        # Best MI
        mi_vals = [v for v in [r.mi_1h, r.mi_4h, r.mi_1d] if v is not None]
        best_mi = f"{max(mi_vals):.3f}" if mi_vals else "N/A"
        
        # First issue
        first_issue = r.issues[0] if r.issues else "-"
        if len(first_issue) > 40:
            first_issue = first_issue[:37] + "..."
        
        lines.append(f"| {r.node_name} | {status_md} | {best_mi} | {first_issue} |")
    
    lines.append("")
    
    # Detailed sections per node
    lines.append("## Detailed Analysis")
    lines.append("")
    
    layers = [
        ("Structural Layer", ['htf_regime']),
        ("Tactical Layer", ['leading_composite', 'coincident_composite', 'confirming_composite']),
        ("Entry Layer", ['trade_hypothesis']),
        ("Timing Layer", ['entry_confidence']),
        ("Management Layer", ['position_confidence']),
        ("Prediction Layer", ['prediction_1h', 'prediction_4h', 'prediction_1d']),
    ]
    
    for layer_name, node_keys in layers:
        layer_results = [(k, results.get(k)) for k in node_keys if k in results]
        if not layer_results:
            continue
        
        lines.append(f"### {layer_name}")
        lines.append("")
        
        for key, r in layer_results:
            if r is None:
                continue
            
            lines.append(f"#### {r.node_name}")
            lines.append("")
            lines.append(f"**Status:** {_status_icon(r.status)} {r.status}")
            lines.append(f"**Samples:** {r.sample_count}")
            lines.append("")
            
            # State distribution
            if r.state_distribution:
                lines.append("**State Distribution:**")
                lines.append("")
                lines.append("| State | Percentage |")
                lines.append("|-------|------------|")
                for state, pct in sorted(r.state_distribution.items(), key=lambda x: -x[1]):
                    stuck = " ⚠️" if r.stuck_state == state else ""
                    lines.append(f"| {state} | {pct:.1%}{stuck} |")
                lines.append("")
            
            # MI
            if any([r.mi_1h, r.mi_4h, r.mi_1d]):
                lines.append("**Mutual Information:**")
                lines.append("")
                if r.mi_1h is not None:
                    lines.append(f"- 1h: {r.mi_1h:.4f}")
                if r.mi_4h is not None:
                    lines.append(f"- 4h: {r.mi_4h:.4f}")
                if r.mi_1d is not None:
                    lines.append(f"- 1d: {r.mi_1d:.4f}")
                lines.append("")
            
            # Issues
            if r.issues:
                lines.append("**Issues:**")
                lines.append("")
                for issue in r.issues:
                    lines.append(f"- {issue}")
                lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    
    failing_nodes = [r for r in results.values() if r.status == 'FAIL']
    if failing_nodes:
        lines.append("### Critical Issues to Fix")
        lines.append("")
        for r in failing_nodes:
            lines.append(f"**{r.node_name}:**")
            for issue in r.issues:
                lines.append(f"- {issue}")
            lines.append(f"- **Fix:** {_get_fix_recommendation(r.node_name, r.issues)}")
            lines.append("")
    
    warning_nodes = [r for r in results.values() if r.status == 'WARN']
    if warning_nodes:
        lines.append("### Warnings to Review")
        lines.append("")
        for r in warning_nodes:
            lines.append(f"- **{r.node_name}:** {r.issues[0] if r.issues else 'Check state distribution'}")
    
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by QBN v3 Node Diagnostic Validator*")
    
    # Write file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    return filename


def _status_icon(status: str) -> str:
    """Get emoji for status."""
    return {
        'PASS': '✅',
        'WARN': '⚠️',
        'FAIL': '❌',
        'PENDING': '⏳'
    }.get(status, '❓')


def _make_bar(pct: float, width: int = 20) -> str:
    """Create ASCII bar for percentage."""
    filled = int(pct * width)
    return '█' * filled + '░' * (width - filled)


def _get_fix_recommendation(node_name: str, issues: List[str]) -> str:
    """Get fix recommendation based on node and issues."""
    recommendations = {
        'HTF_Regime': 'Check ADX signal data availability. Verify regime detector thresholds.',
        'Leading_Composite': 'Review leading signal weights. Check if signals are activating correctly.',
        'Coincident_Composite': 'Review coincident signal weights. May need to lower activation thresholds.',
        'Confirming_Composite': 'Review confirming signal weights. Check trend indicators.',
        'Trade_Hypothesis': 'Leading composite may not be providing enough variation. Check TradeHypothesisGenerator thresholds.',
        'Entry_Confidence': 'Check alignment logic between coincident and confirming composites.',
        'Position_Confidence': 'Review confirming composite output.',
        'Prediction_1h': 'CPT may have insufficient coverage. Retrain with more data or adjust state reduction.',
        'Prediction_4h': 'CPT may have insufficient coverage. Retrain with more data.',
        'Prediction_1d': 'CPT may have insufficient coverage. Need more historical data for daily predictions.',
    }
    
    # Check for specific issues
    issues_str = ' '.join(issues).lower()
    
    if 'stuck' in issues_str or 'no_setup' in issues_str:
        if 'hypothesis' in node_name.lower():
            return 'TradeHypothesisGenerator is not producing trade signals. Check leading composite thresholds.'
        return 'Node is stuck in one state. Check input data and thresholds.'
    
    if 'low mi' in issues_str:
        return 'Node has no predictive power. May need better feature engineering or more data.'
    
    if 'calibration' in issues_str:
        return 'Predicted probabilities do not match observed frequencies. CPT needs retraining.'
    
    return recommendations.get(node_name, 'Review node configuration and input data.')
