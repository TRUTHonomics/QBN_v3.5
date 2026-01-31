#!/usr/bin/env python3
"""
CLI Entry Point voor Combination Alpha Analysis.

ARCHITECTUUR NOOT:
- Draait op GPU Node (10.10.10.2)
- Leest data van Database VM (10.10.10.3) zonder zware JOINs
- Alle compute op GPU

Gebruik:
    python scripts/run_combination_analysis.py --asset-id 1 --target bullish
    python scripts/run_combination_analysis.py --asset-id 1 --all-targets
    python scripts/run_combination_analysis.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.gpu_config import GPUConfig
from analysis.combination_alpha_analyzer import CombinationAlphaAnalyzer, AnalysisResult
from validation.combination_visualizations import CombinationVisualizer
from validation.combination_report import CombinationReportGenerator
from core.logging_utils import setup_logging

# Use Path for project root
PROJECT_ROOT = Path(__file__).parent.parent
logger = setup_logging("run_combination_analysis")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Combination Alpha Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--asset-id', '-a',
        type=int,
        required=True,
        help='Asset ID to analyze'
    )
    
    # Target type
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        '--target', '-t',
        type=str,
        choices=['bullish', 'bearish', 'significant'],
        default='bullish',
        help='Target type for analysis'
    )
    target_group.add_argument(
        '--all-targets',
        action='store_true',
        help='Run analysis for all target types'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--lookback-days', '-d',
        type=int,
        default=None,
        help='Number of days for historical data (default: all data)'
    )
    parser.add_argument(
        '--min-samples', '-m',
        type=int,
        default=30,
        help='Minimum samples per combination'
    )
    parser.add_argument(
        '--n-bootstrap', '-b',
        type=int,
        default=10000,
        help='Number of bootstrap iterations'
    )
    parser.add_argument(
        '--no-bootstrap',
        action='store_true',
        help='Skip bootstrap CI calculation (faster)'
    )
    
    # Correction method
    parser.add_argument(
        '--correction', '-c',
        type=str,
        choices=['fdr_bh', 'fdr_by', 'bonferroni', 'holm'],
        default='fdr_bh',
        help='Multiple testing correction method'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='_validation/combination_alpha',
        help='Output directory for reports and visualizations'
    )
    parser.add_argument(
        '--save-db',
        action='store_true',
        help='Save results to database'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    # GPU options
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU)'
    )
    parser.add_argument(
        '--gpu-device',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (errors only)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        help='Run identifier for traceability'
    )
    
    return parser.parse_args()


def run_analysis(
    asset_id: int,
    target_type: str,
    args: argparse.Namespace
) -> Optional[AnalysisResult]:
    """
    Run analysis for a single target type.
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting analysis: asset={asset_id}, target={target_type}")
    logger.info(f"=" * 60)
    
    # Configure GPU
    gpu_config = GPUConfig(
        use_gpu=not args.no_gpu,
        device_id=args.gpu_device
    )
    
    # Create analyzer
    analyzer = CombinationAlphaAnalyzer(
        gpu_config=gpu_config,
        n_bootstrap=args.n_bootstrap,
        correction_method=args.correction,
        alpha=args.alpha,
        run_id=args.run_id
    )
    
    # Run analysis
    try:
        result = analyzer.run_full_analysis(
            asset_id=asset_id,
            target_type=target_type,
            lookback_days=args.lookback_days,
            min_samples=args.min_samples,
            run_bootstrap=not args.no_bootstrap
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    
    # Print summary
    print_summary(result)
    
    # Save outputs
    output_dir = Path(args.output_dir) / f"asset_{asset_id}" / target_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to database
    if args.save_db:
        try:
            n_saved = analyzer.save_to_database(result)
            logger.info(f"Saved {n_saved} results to database")
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
    
    # Save to JSON
    if args.save_json:
        json_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.save_to_json(result, json_path)
    
    # Generate visualizations
    if not args.no_visualizations:
        generate_visualizations(result, output_dir)
    
    # Generate report
    if not args.no_report:
        generate_report(result, output_dir)
    
    return result


def print_summary(result: AnalysisResult):
    """Print analysis summary to console."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    lookback_str = f"{result.lookback_days} days" if result.lookback_days else "all data"
    print(f"\nAsset ID:        {result.asset_id}")
    print(f"Target Type:     {result.target_type}")
    print(f"Lookback:        {lookback_str}")
    print(f"Analysis Time:   {result.total_time_seconds:.1f}s")
    
    print(f"\nTotal Combinations Analyzed: {result.n_total_combinations}")
    print("-" * 40)
    
    # Classification breakdown
    print(f"  Golden Rules:  {result.n_golden_rules:4d} ({100*result.n_golden_rules/max(1,result.n_total_combinations):.1f}%)")
    print(f"  Promising:     {result.n_promising:4d} ({100*result.n_promising/max(1,result.n_total_combinations):.1f}%)")
    print(f"  Noise:         {result.n_noise:4d} ({100*result.n_noise/max(1,result.n_total_combinations):.1f}%)")
    
    # Per horizon breakdown
    print("\nPer Horizon:")
    for horizon, results in [
        ('1h', result.results_1h),
        ('4h', result.results_4h),
        ('1d', result.results_1d)
    ]:
        golden = sum(1 for r in results if r.classification == 'golden_rule')
        promising = sum(1 for r in results if r.classification == 'promising')
        print(f"  {horizon}: {len(results):3d} total, {golden:2d} golden, {promising:2d} promising")
    
    # Top Golden Rules
    all_results = result.all_results()
    golden_rules = [r for r in all_results if r.classification == 'golden_rule']
    golden_rules = sorted(golden_rules, key=lambda x: x.odds_ratio, reverse=True)
    
    if golden_rules:
        print("\nTop 5 Golden Rules:")
        print("-" * 40)
        for r in golden_rules[:5]:
            print(f"  {r.horizon} | {r.combination_key}")
            print(f"       OR={r.odds_ratio:.2f} [{r.or_ci_lower:.2f}-{r.or_ci_upper:.2f}], n={r.n_with_combination}")
    
    print("\n" + "=" * 60)


def generate_visualizations(result: AnalysisResult, output_dir: Path):
    """Generate all visualizations."""
    logger.info("Generating visualizations...")
    
    visualizer = CombinationVisualizer(output_dir)
    all_results = result.all_results()
    
    # Forest plots per horizon
    for horizon in ['1h', '4h', '1d']:
        try:
            visualizer.create_forest_plot(all_results, horizon)
        except Exception as e:
            logger.warning(f"Failed to create forest plot for {horizon}: {e}")
    
    # Sens/Spec scatter
    try:
        visualizer.create_sens_spec_scatter(all_results)
    except Exception as e:
        logger.warning(f"Failed to create sens/spec scatter: {e}")
    
    # OR heatmaps
    for horizon in ['1h', '4h', '1d']:
        try:
            visualizer.create_or_heatmap(all_results, horizon)
        except Exception as e:
            logger.warning(f"Failed to create OR heatmap for {horizon}: {e}")
    
    # Summary dashboard
    try:
        visualizer.create_summary_dashboard(result)
    except Exception as e:
        logger.warning(f"Failed to create dashboard: {e}")
    
    logger.info(f"Visualizations saved to {output_dir}")


def generate_report(result: AnalysisResult, output_dir: Path):
    """Generate markdown report."""
    logger.info("Generating report...")
    
    try:
        generator = CombinationReportGenerator(output_dir)
        report_path = generator.generate_full_report(result)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.warning(f"Failed to generate report: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine targets to analyze
    if args.all_targets:
        targets = ['bullish', 'bearish', 'significant']
    else:
        targets = [args.target]
    
    # Run analysis for each target
    results = []
    for target in targets:
        try:
            result = run_analysis(args.asset_id, target, args)
            results.append(result)
        except Exception as e:
            logger.error(f"Analysis failed for target {target}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary across all targets
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("MULTI-TARGET SUMMARY")
        print("=" * 60)
        for result in results:
            print(f"\n{result.target_type.upper()}:")
            print(f"  Golden Rules: {result.n_golden_rules}")
            print(f"  Promising:    {result.n_promising}")
    
    logger.info("Analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

