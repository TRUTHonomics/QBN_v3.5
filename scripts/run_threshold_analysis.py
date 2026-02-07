#!/usr/bin/env python3
# scripts/run_threshold_analysis.py
"""
CLI Entry Point voor Composite Threshold Optimalisatie Analyse.

Voert MI Grid Search, Decision Tree CART, en Logistic Regression analyses uit
om optimale thresholds te bepalen voor de QBN v3 Composite en Alignment modules.

Usage:
    docker exec -it QBN_v3.1_Validation python scripts/run_threshold_analysis.py \
        --asset-id 1 \
        --methods mi,cart,logreg \
        --horizons all \
        --lookback-days 180 \
        --output-dir _validation/threshold_analysis

    # Met YAML opslag en database sync:
    docker exec -it QBN_v3.1_Validation python scripts/run_threshold_analysis.py \
        --asset-id 1 \
        --apply-results
"""

import argparse
import logging
import sys
import json
import shutil
from pathlib import Path
import concurrent.futures

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from core.logging_utils import setup_logging
from core.step_validation import log_handshake_out
from core.output_manager import ValidationOutputManager
from core.run_retention import retain_recent_runs_auto
from database.db import get_cursor

logger = setup_logging("threshold_analysis")


def archive_old_results(output_dir: Path):
    """
    Archiveert oude resultaten en submappen naar de .archive map.
    De archiefmap krijgt de naam van de laatste timestamp van de gearchiveerde bestanden.
    """
    archive_root = output_dir / ".archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    
    # Lijst alle items in de output directory (behalve .archive)
    items = [item for item in output_dir.iterdir() if item.name != ".archive"]
    
    if not items:
        return
        
    # Zoek de laatste mtime (modification time)
    latest_mtime = 0
    for item in items:
        mtime = item.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            
    # Formatteer timestamp voor de nieuwe map
    timestamp_str = datetime.fromtimestamp(latest_mtime).strftime("%Y%m%d_%H%M%S")
    target_dir = archive_root / timestamp_str
    
    # Als de map al bestaat (bijv. snelle herstart), voeg een suffix toe
    counter = 1
    original_target = target_dir
    while target_dir.exists():
        target_dir = Path(f"{original_target}_{counter}")
        counter += 1
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Archiveren van {len(items)} oude items naar {target_dir}...")
    
    for item in items:
        try:
            shutil.move(str(item), str(target_dir / item.name))
        except Exception as e:
            logger.warning(f"Kon {item.name} niet archiveren: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Threshold Optimalisatie Analyse voor QBN v3'
    )
    
    parser.add_argument(
        '--asset-id', '-a',
        type=int,
        required=True,
        help='Asset ID om te analyseren (bijv. 1 voor BTC)'
    )
    
    parser.add_argument(
        '--methods', '-m',
        type=str,
        default='mi,cart,logreg',
        help='Comma-separated lijst van methoden: mi, cart, logreg (default: all)'
    )
    
    parser.add_argument(
        '--horizons', '-h_',
        type=str,
        default='all',
        help='Comma-separated lijst van horizons: 1h, 4h, 1d, or "all" (default: all)'
    )
    
    parser.add_argument(
        '--lookback-days', '-l',
        type=int,
        default=365,
        help='Aantal dagen historie (default: 365)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='_validation/threshold_analysis',
        help='Output directory voor rapporten en grafieken'
    )
    
    parser.add_argument(
        '--save-yaml',
        type=str,
        nargs='?',
        const='config/optimal_thresholds.yaml',
        help='Sla optimale resultaten op in YAML bestand (optioneel pad opgeven)'
    )
    
    parser.add_argument(
        '--apply-results',
        action='store_true',
        help='Sla op naar YAML en synchroniseer naar database'
    )
    
    parser.add_argument(
        '--just-apply',
        type=str,
        default=None,
        help='Sla resultaten uit een JSON bestand op naar YAML en database (geen herberekening)'
    )
    
    parser.add_argument(
        '--sync-only',
        type=str,
        default=None,
        help='Synchroniseer bestaande YAML naar database (pad naar YAML bestand)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Toon wijzigingen zonder uit te voeren'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=None,
        help='Minimum aantal samples vereist (default: horizon-specifieke defaults uit ThresholdOptimizer)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=3,
        help='Aantal parallelle workers voor analyse-stappen (default: 3)'
    )
    
    parser.add_argument(
        '--per-node',
        action='store_true',
        help='Genereer per-node analyse output (aparte JSON per composite node)'
    )
    
    parser.add_argument(
        '--targets', '-t',
        type=str,
        default='leading',
        help='Comma-separated lijst van semantic classes om te optimaliseren: leading, coincident, confirming (default: leading)'
    )
    
    parser.add_argument(
        '--no-diversity-check',
        action='store_true',
        help='Schakel diversity constraints uit (niet aanbevolen)'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        help='Run identifier for traceability'
    )
    
    return parser.parse_args()


def _run_single_method(
    method: str,
    horizon: str,
    target: str,
    df: Any,
    asset_id: int,
    lookback_days: int,
    output_dir: Path,
    min_samples: Optional[int],
    enforce_diversity: bool = True
) -> Tuple[str, Dict]:
    """Helper om een enkele analyse methode uit te voeren."""
    from analysis.mutual_information_analyzer import MutualInformationAnalyzer
    from analysis.decision_tree_analyzer import DecisionTreeAnalyzer
    from analysis.logistic_regression_analyzer import LogisticRegressionAnalyzer
    
    # REASON: Gebruik 'all' suffix voor multivariate methoden om redundantie te voorkomen.
    method_key = f"{method}_{target}"
    
    try:
        if method == 'mi':
            logger.info(f"\n[{horizon}] [{target}] Starting Mutual Information Grid Search...")
            logger.info(f"  Diversity constraints: {'ENABLED' if enforce_diversity else 'DISABLED'}")
            analyzer = MutualInformationAnalyzer(
                asset_id=asset_id,
                lookback_days=lookback_days,
                output_dir=output_dir,
                min_samples=min_samples,
                enforce_diversity=enforce_diversity
            )
            result = analyzer.analyze(horizon, target=target, df=df)
            
            # Log diversity metrics
            meta = result.metadata
            logger.info(f"  ✓ [{horizon}] [{target}] MI Grid complete")
            logger.info(f"    MI={result.score:.4f}, diversity={meta.get('diversity_score', 0):.2f}, "
                       f"valid_combos={meta.get('valid_combinations', 'N/A')}")
            
            dist = meta.get('state_distribution', {})
            if dist:
                logger.info(f"    States: {', '.join(f'{k}={v:.1%}' for k, v in sorted(dist.items()))}")
            
            return method_key, result.to_dict()
            
        elif method == 'cart':
            logger.info(f"\n[{horizon}] [{target}] Starting Decision Tree CART Analysis...")
            analyzer = DecisionTreeAnalyzer(
                asset_id=asset_id,
                lookback_days=lookback_days,
                output_dir=output_dir,
                min_samples=min_samples
            )
            result = analyzer.analyze(horizon, target=target, df=df)
            logger.info(f"  ✓ [{horizon}] [{target}] CART complete")
            return method_key, result.to_dict()
            
        elif method == 'logreg':
            logger.info(f"\n[{horizon}] [{target}] Starting Logistic Regression Analysis...")
            analyzer = LogisticRegressionAnalyzer(
                asset_id=asset_id,
                lookback_days=lookback_days,
                output_dir=output_dir,
                min_samples=min_samples
            )
            result = analyzer.analyze(horizon, target=target, df=df)
            logger.info(f"  ✓ [{horizon}] [{target}] LogReg complete")
            return method_key, result.to_dict()
            
    except Exception as e:
        logger.error(f"  ✗ [{horizon}] [{target}] {method.upper()} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"{method_key}_failed", {'error': str(e)}
    
    return "unknown", {}


def run_analysis(
    asset_id: int,
    methods: List[str],
    horizons: List[str],
    targets: List[str],
    lookback_days: int,
    output_dir: Path,
    min_samples: Optional[int],
    max_workers: int = 3,
    enforce_diversity: bool = True
) -> Dict:
    """
    Voer de geselecteerde analyses uit (parallel per horizon en target).
    
    Args:
        asset_id: Asset ID om te analyseren
        methods: Lijst van methoden ('mi', 'cart', 'logreg')
        horizons: Lijst van horizons ('1h', '4h', '1d')
        targets: Lijst van semantic classes ('leading', 'coincident', 'confirming')
        lookback_days: Aantal dagen historie
        output_dir: Output directory
        min_samples: Minimum samples vereist
        max_workers: Aantal parallelle workers
        enforce_diversity: Of diversity constraints toegepast worden (default: True)
    """
    from analysis.mutual_information_analyzer import MutualInformationAnalyzer
    
    results = {
        'asset_id': asset_id,
        'lookback_days': lookback_days,
        'analysis_timestamp': datetime.now().isoformat(),
        'diversity_enforced': enforce_diversity,
        'horizons': {},
        'summary': {}
    }
    
    # REASON: De targets worden nu dynamisch doorgegeven (v3.1)
    # targets = ['leading', 'coincident', 'confirming']
    
    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing horizon: {horizon}")
        logger.info(f"{'='*60}")
        
        # 1. Laad data eenmalig per horizon
        logger.info(f"Loading data for asset {asset_id}, horizon {horizon}...")
        loader = MutualInformationAnalyzer(
            asset_id=asset_id,
            lookback_days=lookback_days,
            min_samples=min_samples,
            enforce_diversity=enforce_diversity
        )
        shared_df = loader.load_data(horizon)
        
        # Log signal statistics
        stats = loader.get_signal_statistics()
        logger.info(f"Signal statistics: {stats['total_signals']} signals, "
                   f"weight coverage: {stats['weight_coverage']:.1f}%")
        for cls, count in stats['signals_per_class'].items():
            logger.info(f"  {cls}: {count} signals")
        
        # 2. Voer methoden parallel uit voor alle targets
        horizon_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Maak futures voor alle geselecteerde methoden en targets
            futures = []
            for method in methods:
                if method == 'mi':
                    # MI is univariaat, moet per target
                    for target in targets:
                        futures.append(
                            executor.submit(
                                _run_single_method,
                                method, horizon, target, shared_df,
                                asset_id, lookback_days, output_dir, min_samples,
                                enforce_diversity
                            )
                        )
                else:
                    # CART en LogReg zijn multivariaat, draai één keer voor 'all'
                    futures.append(
                        executor.submit(
                            _run_single_method,
                            method, horizon, 'all', shared_df,
                            asset_id, lookback_days, output_dir, min_samples,
                            enforce_diversity
                        )
                    )
            
            # Verzamel resultaten zodra ze klaar zijn
            for future in concurrent.futures.as_completed(futures):
                method_target_key, result_dict = future.result()
                horizon_results[method_target_key] = result_dict
        
        results['horizons'][horizon] = horizon_results
    
    # Generate summary
    results['summary'] = _generate_summary(results['horizons'])
    
    return results


def _generate_summary(horizon_results: Dict) -> Dict:
    """Genereer samenvatting van alle resultaten."""
    summary = {}
    
    for horizon, method_keys in horizon_results.items():
        summary[horizon] = {}
        
        for key, result in method_keys.items():
            if 'error' in result:
                continue
            
            # REASON: Toon multivariate resultaten overzichtelijk
            if 'all_thresholds' in result.get('metadata', {}) and result['metadata']['all_thresholds']:
                all_t = result['metadata']['all_thresholds']
                for target, vals in all_t.items():
                    summary[horizon][f"{key}_{target}"] = {
                        'neutral_band': vals['neutral_band'],
                        'strong_threshold': vals['strong_threshold'],
                        'score': result.get('score'),
                        'score_name': result.get('score_name')
                    }
            else:
                summary[horizon][key] = {
                    'neutral_band': result.get('optimal_neutral_band'),
                    'strong_threshold': result.get('optimal_strong_threshold'),
                    'score': result.get('score'),
                    'score_name': result.get('score_name')
                }
    
    return summary


def generate_report(results: Dict, output_dir: Path) -> Path:
    """Genereer Markdown rapport."""
    from validation.threshold_validation_report import ThresholdValidationReport
    
    reporter = ThresholdValidationReport(output_dir)
    report_path = reporter.generate(results)
    return report_path


def save_to_yaml(results: Dict, output_path: Optional[Path] = None, run_id: Optional[str] = None) -> Path:
    """Sla optimale resultaten op in YAML via ConfigPersister."""
    from analysis.config_persister import ConfigPersister
    persister = ConfigPersister(run_id=run_id)
    return persister.save_to_yaml(results, output_path)


def save_per_node_results(results: Dict, output_dir: Path, asset_id: int) -> List[Path]:
    """
    Sla per-node analyse resultaten op in aparte JSON bestanden.
    
    Dit maakt het makkelijker om specifieke nodes te analyseren en vergelijken.
    
    Output structuur:
        _validation/threshold_analysis/per_node/
            leading_composite_asset_1.json
            coincident_composite_asset_1.json
            confirming_composite_asset_1.json
    """
    per_node_dir = output_dir / "per_node"
    per_node_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    nodes = ['leading', 'coincident', 'confirming']
    
    for node in nodes:
        node_results = {
            'asset_id': asset_id,
            'node_name': f'{node}_composite',
            'analysis_timestamp': results.get('analysis_timestamp'),
            'diversity_enforced': results.get('diversity_enforced', True),
            'horizons': {}
        }
        
        # Verzamel resultaten voor deze node over alle horizons
        for horizon, horizon_data in results.get('horizons', {}).items():
            node_key = f'mi_{node}'
            if node_key in horizon_data:
                node_results['horizons'][horizon] = horizon_data[node_key]
        
        # Sla op
        output_path = per_node_dir / f'{node}_composite_asset_{asset_id}.json'
        with open(output_path, 'w') as f:
            json.dump(node_results, f, indent=2, default=str)
        
        saved_files.append(output_path)
        logger.info(f"Saved per-node results: {output_path}")
    
    # Genereer ook een overzichtsbestand met diversity metrics
    diversity_summary = {
        'asset_id': asset_id,
        'analysis_timestamp': results.get('analysis_timestamp'),
        'nodes': {}
    }
    
    for node in nodes:
        node_data = {'horizons': {}}
        for horizon, horizon_data in results.get('horizons', {}).items():
            node_key = f'mi_{node}'
            if node_key in horizon_data:
                data = horizon_data[node_key]
                meta = data.get('metadata', {})
                node_data['horizons'][horizon] = {
                    'neutral_band': data.get('optimal_neutral_band'),
                    'strong_threshold': data.get('optimal_strong_threshold'),
                    'mi_score': data.get('score'),
                    'diversity_score': meta.get('diversity_score'),
                    'active_states': meta.get('active_states'),
                    'valid_combinations': meta.get('valid_combinations'),
                    'state_distribution': meta.get('state_distribution', {})
                }
        diversity_summary['nodes'][f'{node}_composite'] = node_data
    
    summary_path = per_node_dir / f'diversity_summary_asset_{asset_id}.json'
    with open(summary_path, 'w') as f:
        json.dump(diversity_summary, f, indent=2, default=str)
    saved_files.append(summary_path)
    logger.info(f"Saved diversity summary: {summary_path}")
    
    return saved_files


def sync_to_database(yaml_path: Path, dry_run: bool = False, run_id: Optional[str] = None, asset_id: Optional[int] = None):
    """Synchroniseer YAML naar database via ConfigPersister."""
    from analysis.config_persister import ConfigPersister
    persister = ConfigPersister(run_id=run_id)
    persister.sync_from_yaml(yaml_path, dry_run=dry_run)
    
    # HANDSHAKE_OUT logging (alleen als niet dry_run - dry_run schrijft niet naar DB)
    if not dry_run:
        # Haal aantal rows op dat gesynct is (composite_threshold_config)
        with get_cursor() as cur:
            if run_id:
                cur.execute("SELECT COUNT(*) FROM qbn.composite_threshold_config WHERE run_id = %s", (run_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM qbn.composite_threshold_config")
            rows_synced = cur.fetchone()[0]
        
        log_handshake_out(
            step="run_threshold_analysis",
            target="qbn.composite_threshold_config",
            run_id=run_id or "N/A",
            rows=rows_synced,
            operation="INSERT/UPDATE"
        )
        
        # Retentie: bewaar 3 meest recente runs
        if run_id and asset_id:
            with get_cursor() as cur:
                retain_recent_runs_auto(cur.connection, "qbn.composite_threshold_config", asset_id)


def main():
    args = parse_args()
    
    # REASON: Genereer een default run_id als deze ontbreekt (v3.1)
    # De database tabel heeft run_id in de PRIMARY KEY en als NOT NULL.
    if not args.run_id:
        args.run_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Geen run_id opgegeven, gebruik default: {args.run_id}")

    # Parse methods
    methods = [m.strip().lower() for m in args.methods.split(',')]
    valid_methods = ['mi', 'cart', 'logreg']
    methods = [m for m in methods if m in valid_methods]
    
    if not methods:
        logger.error(f"No valid methods specified. Choose from: {valid_methods}")
        sys.exit(1)
    
    # Parse horizons
    if args.horizons.lower() == 'all':
        horizons = ['1h', '4h', '1d']
    else:
        horizons = [h.strip().lower() for h in args.horizons.split(',')]
    
    valid_horizons = ['1h', '4h', '1d']
    horizons = [h for h in horizons if h in valid_horizons]
    
    if not horizons:
        logger.error(f"No valid horizons specified. Choose from: {valid_horizons}")
        sys.exit(1)
    
    # Parse targets (v3.1)
    targets = [t.strip().lower() for t in args.targets.split(',')]
    valid_targets = ['leading', 'coincident', 'confirming']
    targets = [t for t in targets if t in valid_targets]
    
    if not targets:
        logger.error(f"No valid targets specified. Choose from: {valid_targets}")
        sys.exit(1)
    
    # REASON: Informeer gebruiker over v3.1 beleid m.b.t. Management Layer (v3.1)
    if any(t in ['coincident', 'confirming'] for t in targets):
        logger.warning("=" * 70)
        logger.warning("⚠️ ATTENTIE: COINCIDENT/CONFIRMING OPTIMALISATIE")
        logger.warning("In v3.1 zijn deze composites verplaatst naar de Management Layer.")
        logger.warning("Globale optimalisatie (zoals nu) is minder effectief dan event-driven.")
        logger.warning("De resultaten zijn bruikbaar als baseline, maar v3.2 zal deze")
        logger.warning("optimaliseren binnen de Event Windows.")
        logger.warning("=" * 70)

    # REASON: Gebruik ValidationOutputManager voor gestructureerde output met run_id traceerbaarheid
    if args.output_dir != '_validation/threshold_analysis':
        # Respecteer expliciet --output-dir argument (backward compatibility)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Gebruik nieuwe structuur met automatische archivering
        output_mgr = ValidationOutputManager()
        output_dir = output_mgr.create_output_dir(
            script_name="threshold_analysis",
            asset_id=args.asset_id,
            run_id=args.run_id
        )
    
    # Diversity enforcement
    enforce_diversity = not args.no_diversity_check
    
    logger.info("=" * 70)
    logger.info("COMPOSITE THRESHOLD OPTIMALISATIE")
    logger.info("=" * 70)
    logger.info(f"Asset ID: {args.asset_id}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Targets: {targets}")
    logger.info(f"Lookback: {args.lookback_days} days")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Per-node output: {'ENABLED' if args.per_node else 'DISABLED'}")
    logger.info(f"Diversity constraints: {'ENABLED' if enforce_diversity else 'DISABLED'}")
    logger.info("=" * 70)
    
    # Handle sync-only mode
    if args.sync_only:
        yaml_path = Path(args.sync_only)
        if not yaml_path.exists():
            logger.error(f"YAML file not found: {yaml_path}")
            sys.exit(1)
        logger.info(f"Syncing {yaml_path} to database...")
        sync_to_database(yaml_path, dry_run=args.dry_run, run_id=args.run_id, asset_id=args.asset_id)
        return

    # Handle just-apply mode
    if args.just_apply:
        json_path = Path(args.just_apply)
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            sys.exit(1)
        
        logger.info(f"Loading results from {json_path}...")
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        logger.info("Persisting results to YAML and database...")
        target_path = Path(args.save_yaml) if args.save_yaml else None
        yaml_path = save_to_yaml(results, target_path, run_id=args.run_id)
        sync_to_database(yaml_path, dry_run=args.dry_run, run_id=args.run_id, asset_id=args.asset_id)
        logger.info("✅ Persistence complete")
        return
    
    # Run analysis (OutputManager doet automatisch archivering)
    max_workers = args.workers
    results = run_analysis(
        asset_id=args.asset_id,
        methods=methods,
        horizons=horizons,
        targets=targets,
        lookback_days=args.lookback_days,
        output_dir=output_dir,
        min_samples=args.min_samples,
        max_workers=max_workers,
        enforce_diversity=enforce_diversity
    )
    
    # Save JSON results
    json_path = output_dir / f'threshold_analysis_{args.asset_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    latest_json_path = output_dir / f'latest_results_{args.asset_id}.json'
    
    results_json = json.dumps(results, indent=2, default=str)
    with open(json_path, 'w') as f:
        f.write(results_json)
    with open(latest_json_path, 'w') as f:
        f.write(results_json)
        
    logger.info(f"\nSaved JSON results to {json_path}")
    logger.info(f"Updated latest results at {latest_json_path}")
    
    # Save per-node results if requested
    if args.per_node:
        logger.info("\nGenerating per-node output...")
        per_node_files = save_per_node_results(results, output_dir, args.asset_id)
        logger.info(f"Saved {len(per_node_files)} per-node files")
    
    # Generate report
    try:
        report_path = generate_report(results, output_dir)
        logger.info(f"Generated report: {report_path}")
    except Exception as e:
        logger.warning(f"Could not generate report: {e}")
    
    # Save to YAML if requested
    yaml_path = None
    if args.save_yaml or args.apply_results:
        target_path = Path(args.save_yaml) if args.save_yaml else None
        yaml_path = save_to_yaml(results, target_path, run_id=args.run_id)
    
    # Sync to database if requested
    if args.apply_results:
        if args.dry_run:
            logger.info("\n[DRY RUN] Would sync to database...")
        else:
            if yaml_path:
                sync_to_database(yaml_path, dry_run=False, run_id=args.run_id, asset_id=args.asset_id)
            else:
                logger.error("Cannot sync to database: no YAML path available")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SAMENVATTING")
    logger.info("=" * 70)
    
    for horizon, summary in results['summary'].items():
        logger.info(f"\n{horizon} Horizon:")
        for method, data in summary.items():
            if data.get('neutral_band'):
                logger.info(f"  {method}: neutral_band={data['neutral_band']}, "
                           f"strong_threshold={data['strong_threshold']}, "
                           f"{data['score_name']}={data['score']:.4f}")
            elif data.get('score'):
                logger.info(f"  {method}: {data['score_name']}={data['score']:.4f}")
    
    # Print diversity summary for MI results
    if 'mi' in methods:
        logger.info("\n" + "-" * 70)
        logger.info("DIVERSITY METRICS (MI Grid Search)")
        logger.info("-" * 70)
        
        for horizon, horizon_data in results.get('horizons', {}).items():
            logger.info(f"\n{horizon} Horizon:")
            for target in targets:
                key = f'mi_{target}'
                if key in horizon_data:
                    meta = horizon_data[key].get('metadata', {})
                    dist = meta.get('state_distribution', {})
                    diversity = meta.get('diversity_score', 0)
                    valid = meta.get('valid_combinations', 'N/A')
                    active = meta.get('active_states', 'N/A')
                    
                    logger.info(f"  {target}:")
                    logger.info(f"    diversity_score={diversity:.2f}, active_states={active}, valid_combos={valid}")
                    if dist:
                        logger.info(f"    states: {', '.join(f'{k}={v:.1%}' for k, v in sorted(dist.items()))}")
    
    # REASON: Geef advies over vervolgstappen als we alleen leading hebben gedaan (v3.1)
    if targets == ['leading']:
        logger.info("\n" + "*" * 70)
        logger.info("💡 ADVIES: LEADING COMPOSITE OPTIMALISATIE VOLTOOID")
        logger.info("Gebruik deze drempels om Event Windows te detecteren (Menu Optie 6).")
        logger.info("Optimalisatie van Coincident/Confirming volgt in v3.2 via events.")
        logger.info("*" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSE VOLTOOID")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

