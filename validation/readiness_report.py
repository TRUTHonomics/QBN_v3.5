"""
Readiness Report Generator for QBN v3.

Generates Markdown reports from ProductionReadinessValidator results.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from validation.production_readiness import CheckResult


class ReadinessReportGenerator:
    """
    Generates Markdown report from validation results.

    Usage:
        generator = ReadinessReportGenerator()
        report_path = generator.generate(asset_id, verdict, results)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / '_validation'
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        asset_id: int,
        asset_symbol: str,
        verdict: str,
        results: List[CheckResult]
    ) -> Path:
        """
        Generate Markdown report.

        Args:
            asset_id: Asset ID
            asset_symbol: Asset symbol (e.g., 'BTCUSDT')
            verdict: 'GO' or 'NO-GO'
            results: List of CheckResult objects

        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"readiness_report_asset_{asset_id}_{timestamp}.md"

        content = self._build_report(asset_id, asset_symbol, verdict, results)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        return filename

    def _build_report(
        self,
        asset_id: int,
        asset_symbol: str,
        verdict: str,
        results: List[CheckResult]
    ) -> str:
        """Build the Markdown report content."""
        lines = []

        # Header
        lines.append(f"# Production Readiness Report - Asset {asset_id} ({asset_symbol})")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        verdict_emoji = "✅" if verdict == "GO" else "❌"
        verdict_text = "System ready for production inference" if verdict == "GO" else "System NOT ready - issues must be resolved"
        lines.append(f"**Verdict: {verdict_emoji} {verdict}** - {verdict_text}")
        lines.append("")

        # Summary counts
        summary = self._get_summary(results)
        lines.append(f"| Checks Passed | Warnings | Failed |")
        lines.append(f"|:-------------:|:--------:|:------:|")
        lines.append(f"| {summary['pass']} | {summary['warn']} | {summary['fail']} |")
        lines.append("")

        # Results Table
        lines.append("## Check Results")
        lines.append("")
        lines.append("| Check | Status | Value | Threshold |")
        lines.append("|-------|--------|-------|-----------|")

        for r in results:
            status_icon = self._status_icon(r.status)
            lines.append(f"| {r.name} | {status_icon} {r.status} | {r.value} | {r.threshold or '-'} |")

        lines.append("")

        # Failed Checks Details
        failed = [r for r in results if r.status == 'FAIL']
        if failed:
            lines.append("## Failed Checks - Action Required")
            lines.append("")

            for r in failed:
                lines.append(f"### {r.name}")
                lines.append("")
                lines.append(f"- **Status:** {r.status}")
                lines.append(f"- **Value:** {r.value}")
                lines.append(f"- **Message:** {r.message}")
                lines.append(f"- **Threshold:** {r.threshold}")
                lines.append("")
                lines.append(f"**Resolution:** {self._get_resolution(r.name)}")
                lines.append("")

        # Warnings
        warnings = [r for r in results if r.status == 'WARN']
        if warnings:
            lines.append("## Warnings")
            lines.append("")
            lines.append("These checks passed but may need attention:")
            lines.append("")

            for r in warnings:
                lines.append(f"- **{r.name}**: {r.value} ({r.message})")

            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if verdict == "GO":
            lines.append("System is ready for production. Consider:")
            lines.append("")
            lines.append("1. Monitor CPT freshness - regenerate if older than 24h")
            lines.append("2. Review any warnings above")
            lines.append("3. Run this check regularly before inference cycles")
        else:
            lines.append("Before deploying to production:")
            lines.append("")
            for i, r in enumerate(failed, 1):
                lines.append(f"{i}. Fix: {r.name} - {r.message}")

        lines.append("")
        lines.append("---")
        lines.append("*Report generated by QBN v3 Production Readiness Validator*")

        return "\n".join(lines)

    def _get_summary(self, results: List[CheckResult]) -> Dict[str, int]:
        """Get summary counts."""
        return {
            'pass': sum(1 for r in results if r.status == 'PASS'),
            'warn': sum(1 for r in results if r.status == 'WARN'),
            'fail': sum(1 for r in results if r.status == 'FAIL'),
        }

    def _status_icon(self, status: str) -> str:
        """Get emoji for status."""
        return {
            'PASS': '✅',
            'WARN': '⚠️',
            'FAIL': '❌'
        }.get(status, '❓')

    def _get_resolution(self, check_name: str) -> str:
        """Get resolution advice for a failed check."""
        resolutions = {
            'Outcome Coverage 1h': 'Run target generation: Training Menu > Option 1 (Generate Training Targets)',
            'Outcome Coverage 4h': 'Run target generation: Training Menu > Option 1 (Generate Training Targets)',
            'Outcome Coverage 1d': 'Run target generation: Training Menu > Option 1 (Generate Training Targets)',
            'Signal Data Availability': 'Ensure MTF signal pipeline is running and populating kfl.mtf_signals_lead',
            'Threshold Config': 'Run threshold optimization: Training Menu > Option 8 (Composite Threshold Optimalisatie) or populate qbn.composite_threshold_config manually',
            'Signal Weights': 'Run signal weight calculation or populate qbn.signal_weights for this asset',
            'Signal Classification': 'Populate qbn.signal_classification with signal definitions',
            'CPT Availability': 'Generate CPTs: Training Menu > Option 3 (Generate CPTs)',
            'CPT Key Coverage': 'Regenerate CPTs with more training data or relax state reduction',
            'Average Entropy': 'Review CPT quality - entropy outside normal range indicates issues with training data or state reduction',
            'CPT Staleness': 'Regenerate CPTs: Training Menu > Option 3 (Generate CPTs)',
            'Key Matching Test': 'CPT parent combinations do not match inference engine expectations - check state naming consistency',
            'Inference Latency': 'Check database connection and consider caching optimizations',
        }
        return resolutions.get(check_name, 'Review the check details and address the underlying issue')


def print_console_results(
    asset_id: int,
    asset_symbol: str,
    verdict: str,
    results: List[CheckResult],
    report_path: Optional[Path] = None
):
    """
    Print formatted results to console.

    Args:
        asset_id: Asset ID
        asset_symbol: Asset symbol
        verdict: 'GO' or 'NO-GO'
        results: List of CheckResult objects
        report_path: Path to saved report (optional)
    """
    print()
    print("=" * 60)
    print(f"QBN v3 PRODUCTION READINESS CHECK - Asset {asset_id} ({asset_symbol})")
    print("=" * 60)
    print()

    # Group results by category
    categories = [
        ('DATA LAYER', ['Outcome Coverage', 'Signal Data']),
        ('CONFIGURATION', ['Threshold Config', 'Signal Weights', 'Signal Classification']),
        ('CPT QUALITY', ['CPT Availability', 'CPT Key Coverage', 'Entropy', 'Staleness']),
        ('INFERENCE SIMULATION', ['Key Matching', 'Inference Latency']),
    ]

    def find_results(keywords):
        """Find results matching any keyword."""
        matched = []
        for r in results:
            if any(kw.lower() in r.name.lower() for kw in keywords):
                matched.append(r)
        return matched

    for cat_name, keywords in categories:
        cat_results = find_results(keywords)
        if cat_results:
            print(f"[{cat_name}]")
            for r in cat_results:
                status_str = r.status
                # Pad name and value for alignment
                name = r.name
                value = str(r.value)
                dots = '.' * (35 - len(name))
                print(f"  {name} {dots} {value:<15} {status_str}")
            print()

    # Verdict
    print("=" * 60)
    if verdict == "GO":
        print(f"VERDICT: GO - System ready for production inference")
    else:
        fail_count = sum(1 for r in results if r.status == 'FAIL')
        print(f"VERDICT: NO-GO - {fail_count} check(s) failed")
    print("=" * 60)

    if report_path:
        print(f"\nRapport opgeslagen: {report_path}")
