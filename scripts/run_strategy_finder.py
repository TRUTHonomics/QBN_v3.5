#!/usr/bin/env python3
"""
QBN Strategy Finder (Grid Search) - v3.4.

Optimaliseert strategie parameters door meerdere combinaties te testen op dezelfde voorspellingen.

v3.4 Architecture-Aligned:
- Entry beslissing ALLEEN op Trade_Hypothesis (strength filter, beide long en short)
- 4 Management signalen voor exit/sizing (Exit_Timing, Momentum, Volatility, Position_Prediction)
- Position-side nodes worden NIET gebruikt voor entry (architectuur aligned)

Entry Filters:
- entry_strength_threshold: 'weak' of 'strong' (trade beide long en short)
- regime_filter: Toegestane HTF_Regime states

QBN Management Signals:
- use_qbn_exit_timing: Exit bij Exit_Timing = 'exit_now'
- exit_on_momentum_reversal: Exit als momentum tegen positie draait
- volatility_position_sizing: Pas positiegrootte aan op volatility regime
- use_position_prediction_exit: Exit bij Position_Prediction = stoploss_hit/timeout

Parameter Grid Format:
{
    "entry_strength_threshold": ["weak", "strong"],
    "stop_loss_atr_mult": [1.0, 1.5, 2.0],
    "take_profit_atr_mult": [1.5, 2.0, 2.5],
    "use_qbn_exit_timing": [true, false],
    "leverage": [1.0, 10.0],
    ...
}
"""

import argparse
import logging
import sys
import os
import json
import itertools
import functools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging
from database.db import get_cursor
from simulation.data_loader import BacktestDataLoader
from validation.trade_simulator import TradeSimulator
from validation.backtest_config import BacktestConfig
from inference.trade_aligned_inference import DualInferenceResult
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
from inference.gpu.gpu_inference_engine import GPUInferenceEngine

logger = setup_logging("strategy_finder")

# Helper functie voor parallelisatie
# REASON: Moet op module-niveau staan voor pickling door ProcessPoolExecutor
def run_single_simulation(params, steps_data):
    """
    Run simulation met TradeSimulator (v3.4 logic).
    
    Args:
        params: Dictionary met BacktestConfig parameters
        steps_data: List van dicts met test_df en predictions per step
        
    Returns:
        Dict met metrics en equity curve
    """
    # Suppress TradeSimulator INFO logs to prevent spamming stdout/logs during grid search
    logging.getLogger("validation.trade_simulator").setLevel(logging.WARNING)

    # Build BacktestConfig from params
    # REASON: Extract BacktestConfig fields from params, use defaults voor ontbrekende
    # v3.4: Nieuwe parameter namen aligned met architectuur (entry_strength_threshold, etc.)
    config = BacktestConfig(
        asset_id=params.get('asset_id', 1),
        start_date=params.get('start_date', pd.Timestamp('2020-01-01', tz='UTC')),
        end_date=params.get('end_date', pd.Timestamp('2025-01-01', tz='UTC')),
        train_window_days=params.get('train_window_days', 365),
        initial_capital_usd=params.get('initial_capital_usd', 10000.0),
        leverage=params.get('leverage', 1.0),
        position_size_pct=params.get('position_size_pct', 100.0),
        slippage_pct=params.get('slippage_pct', 0.05),
        stop_loss_atr_mult=params.get('stop_loss_atr_mult', 1.5),
        take_profit_atr_mult=params.get('take_profit_atr_mult', 2.0),
        use_atr_based_exits=params.get('use_atr_based_exits', True),
        trailing_stop_enabled=params.get('trailing_stop_enabled', False),
        trailing_stop_pct=params.get('trailing_stop_pct', 50.0),
        trailing_activation_pct=params.get('trailing_activation_pct', 0.5),
        max_holding_time_hours=params.get('max_holding_time_hours', None),
        # v3.4: Entry filter - alleen strength threshold (beide long en short)
        entry_strength_threshold=params.get('entry_strength_threshold', 'weak'),
        regime_filter=params.get('regime_filter', []),
        # v3.4: Management signals (4 position-side nodes)
        use_qbn_exit_timing=params.get('use_qbn_exit_timing', True),
        exit_on_momentum_reversal=params.get('exit_on_momentum_reversal', False),
        volatility_position_sizing=params.get('volatility_position_sizing', False),
        use_position_prediction_exit=params.get('use_position_prediction_exit', False),
        maker_fee_pct=params.get('maker_fee_pct', 0.02),
        taker_fee_pct=params.get('taker_fee_pct', 0.05),
    )
    
    simulator = TradeSimulator(config)
    equity_curve = []
    
    for step in steps_data:
        test_df = step['test_df']
        preds = step['predictions']
        
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            current_time = row['time_1']
            current_price = row.get('close', row.get('close_60', 0))
            atr = row.get('atr_14', row.get('atr', 20.0))
            
            # Build DualInferenceResult from predictions
            inference_result = DualInferenceResult(
                asset_id=config.asset_id,
                timestamp=current_time,
                regime=preds['regime'][idx],
                trade_hypothesis=preds['trade_hypothesis'][idx],
                leading_composite=preds.get('leading_composite', ['neutral'] * len(test_df))[idx],
                coincident_composite=preds.get('coincident_composite', ['neutral'] * len(test_df))[idx],
                confirming_composite=preds.get('confirming_composite', ['neutral'] * len(test_df))[idx],
                entry_predictions={
                    '1h': preds['predictions']['1h']['states'][idx],
                    '4h': preds['predictions']['4h']['states'][idx],
                    '1d': preds['predictions']['1d']['states'][idx],
                },
                entry_distributions={
                    '1h': preds['predictions']['1h']['distributions'][idx],
                    '4h': preds['predictions']['4h']['distributions'][idx],
                    '1d': preds['predictions']['1d']['distributions'][idx],
                },
                # v3.4 fields - 4 management signals
                momentum_prediction=preds.get('momentum_prediction', ['neutral'] * len(test_df))[idx],
                volatility_regime=preds.get('volatility_regime', ['normal'] * len(test_df))[idx],
                exit_timing=preds.get('exit_timing', ['hold'] * len(test_df))[idx],
                position_prediction=preds.get('position_prediction', ['hold'] * len(test_df))[idx],
                # Legacy (deprecated in v3.4, kept for compatibility)
                position_confidence=preds.get('position_confidence', ['neutral'] * len(test_df))[idx],
            )
            
            # Check voor entry (alleen als geen open trades)
            if not simulator.open_trades:
                # v3.4: should_enter_trade returns (bool, direction)
                should_enter, direction = simulator.should_enter_trade(inference_result)
                if should_enter and direction:
                    # Extract raw composite scores if available
                    current_scores = None
                    if 'raw_composite_scores' in preds:
                        current_scores = {
                            'leading': float(preds['raw_composite_scores']['leading'][idx]),
                            'coincident': float(preds['raw_composite_scores']['coincident'][idx]),
                            'confirming': float(preds['raw_composite_scores']['confirming'][idx]),
                        }
                    # v3.4: pass direction explicitly to open_trade
                    simulator.open_trade(inference_result, current_price, atr, direction, current_scores)
            
            # Update open trades
            # REASON: Gebruik 60m OHLC als synthetic 1m data (snelheidsoptimalisatie)
            if simulator.open_trades:
                synthetic_ohlc = pd.DataFrame([{
                    'time': current_time,
                    'open': row.get('open', current_price),
                    'high': row.get('high', current_price),
                    'low': row.get('low', current_price),
                    'close': current_price
                }])
                simulator.update_open_trades(current_time, synthetic_ohlc, inference_result)
            
            # Track equity
            current_equity = simulator.current_capital
            if simulator.open_trades:
                # Mark-to-market van open position
                trade = simulator.open_trades[0]
                if trade.direction == 'long':
                    unrealized_pnl = (current_price - trade.entry_price) / trade.entry_price * trade.position_size_usd
                else:
                    unrealized_pnl = (trade.entry_price - current_price) / trade.entry_price * trade.position_size_usd
                current_equity = simulator.current_capital + unrealized_pnl
            
            equity_curve.append({'time': current_time, 'equity': current_equity})
    
    # Force-close any remaining open trades
    if simulator.open_trades and len(test_df) > 0:
        last_row = test_df.iloc[-1]
        last_price = last_row.get('close', last_row.get('close_60', 0))
        synthetic_close = pd.DataFrame([{
            'time': last_row['time_1'],
            'open': last_price,
            'high': last_price,
            'low': last_price,
            'close': last_price
        }])
        simulator._close_trade(
            simulator.open_trades[0],
            last_row['time_1'],
            last_price,
            'end_of_test'
        )
    
    # Get metrics
    metrics = simulator.get_metrics()
    metrics['params'] = params
    metrics['equity_curve'] = equity_curve
    
    return metrics

def run_strategy_finder(
    asset_id: int,
    start_date: str,
    end_date: str,
    train_window_days: int = 365,
    step_days: int = 7,
    param_grid: dict = None,
    neutral_downsample: float = 1.0,
    base_config_id: int = None
):
    # 1. Setup
    s_date = pd.Timestamp(start_date).tz_localize('UTC')
    e_date = pd.Timestamp(end_date).tz_localize('UTC')
    data_start_needed = s_date - timedelta(days=train_window_days)
    
    logger.info(f"🔍 Starting Strategy Finder for Asset {asset_id}")
    logger.info(f"📅 Period: {s_date.date()} to {e_date.date()} ({train_window_days}d training)")
    sys.stdout.flush()
    
    # Generate parameter combinations
    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"🎲 Testing {len(combinations)} parameter combinations")
    
    # v3.4: Limiet op aantal combinaties om runaway jobs te voorkomen
    MAX_COMBINATIONS = 3_000_000
    if len(combinations) > MAX_COMBINATIONS:
        logger.error(f"❌ Grid te groot: {len(combinations):,} > {MAX_COMBINATIONS:,}. Reduceer parameters.")
        print(f"❌ FOUT: {len(combinations):,} combinaties overschrijdt limiet van {MAX_COMBINATIONS:,}.", flush=True)
        print(f"💡 Tip: Schakel parameters uit of reduceer ranges in de Grid Search Configurator.", flush=True)
        return
    
    # 2. Components
    loader = BacktestDataLoader(asset_id)
    
    # 3. FETCH ONCE
    logger.info("📥 Fetching full dataset...")
    full_df = loader.fetch_data(data_start_needed, e_date)
    if full_df.empty:
        logger.error("❌ No data found.")
        return

    if full_df['time_1'].dt.tz is None:
        full_df['time_1'] = full_df['time_1'].dt.tz_localize('UTC')
    full_df.set_index('time_1', inplace=True, drop=False)
    full_df.sort_index(inplace=True)
    
    # 4. PREPROCESS ONCE
    logger.info("⚙️ Preprocessing...")
    generator = QBNv3CPTGenerator(neutral_downsample=neutral_downsample)
    full_df = generator.preprocess_dataset(full_df, asset_id)
    
    # 5. Main Loop: Generate Predictions for each step
    current_train_end = s_date
    steps = []
    
    # Bereken totaal aantal steps voor voortgangsindicatie
    total_steps = 0
    temp_end = s_date
    while temp_end < e_date:
        temp_end = min(temp_end + timedelta(days=step_days), e_date)
        total_steps += 1
    
    logger.info(f"🧠 Generating predictions for {total_steps} rolling window steps...")
    print(f"🧠 Training BN for {total_steps} rolling window steps...", flush=True)
    
    step_num = 0
    while current_train_end < e_date:
        step_num += 1
        test_end = min(current_train_end + timedelta(days=step_days), e_date)
        train_start = current_train_end - timedelta(days=train_window_days)
        
        print(f"  Step {step_num}/{total_steps}: Training on {train_start.date()} to {current_train_end.date()}...", flush=True)
        
        train_df = full_df.loc[train_start : current_train_end - timedelta(seconds=1)].copy().reset_index(drop=True)
        test_df = full_df.loc[current_train_end : test_end - timedelta(seconds=1)].copy().reset_index(drop=True)
        
        if len(train_df) < 1000 or len(test_df) == 0:
            current_train_end = test_end
            continue

        # Train & Infer
        step_generator = QBNv3CPTGenerator(neutral_downsample=neutral_downsample)
        step_generator.load_signal_classification(asset_id=asset_id, horizon='1h')
        
        # REASON: v3.4 node structure - Entry_Confidence node bestaat niet meer
        cpts = {
            'HTF_Regime': step_generator.generate_htf_regime_cpt(asset_id, data=train_df),
            'Trade_Hypothesis': step_generator.generate_trade_hypothesis_cpt(asset_id, data=train_df),
            'Prediction_1h': step_generator.generate_prediction_cpt(asset_id, '1h', data=train_df),
            'Prediction_4h': step_generator.generate_prediction_cpt(asset_id, '4h', data=train_df),
            'Prediction_1d': step_generator.generate_prediction_cpt(asset_id, '1d', data=train_df)
        }
        
        from inference.node_types import SemanticClass
        for sem_class in SemanticClass:
            cpts[f"{sem_class.value.capitalize()}_Composite"] = step_generator.generate_composite_cpt(asset_id, sem_class, data=train_df, horizon='1h')
        
        # v3.4: Add position-side nodes
        # REASON: Strategy Finder simuleert geen echte in-trade deltas, maar we voegen dummy nodes toe
        # zodat de engine niet crasht. De simulator gebruikt deze niet actief in 60m mode.
        cpts['Momentum_Prediction'] = step_generator.generate_momentum_prediction_cpt(asset_id, data=train_df)
        cpts['Volatility_Regime'] = step_generator.generate_volatility_regime_cpt(asset_id, data=train_df)
        cpts['Exit_Timing'] = step_generator.generate_exit_timing_cpt(asset_id, data=train_df)
        cpts['Position_Confidence'] = step_generator.generate_position_confidence_cpt(asset_id, data=train_df)
        cpts['Position_Prediction'] = step_generator._generate_position_prediction_cpt(asset_id, data=train_df)

        # REASON: Geef ThresholdLoader mee aan GPU engine om fallback warnings te voorkomen
        # De generator heeft deze al geladen voor asset/horizon
        threshold_loader = step_generator._get_threshold_loader(asset_id, '1h')

        engine = GPUInferenceEngine(
            cpts=cpts, 
            signal_classification=step_generator.signal_aggregator.signal_classification,
            threshold_loader=threshold_loader
        )
        predictions = engine.infer_batch(test_df)
        
        # Store step data for multiple strategy tests
        steps.append({
            'test_df': test_df,
            'predictions': predictions
        })
        
        # Cleanup
        import gc
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        except: pass
        del engine, step_generator, train_df
        gc.collect()
        
        current_train_end = test_end

    # 6. Grid Search: Test all combinations on the stored predictions
    num_workers = max(1, os.cpu_count() - 2)
    logger.info(f"📊 Running parallel simulations for {len(combinations)} combinations using {num_workers} workers...")
    print(f"📊 Simulating {len(combinations)} parameter combinations...", flush=True)
    
    # Inject common params in alle combinations (asset_id, dates, etc.)
    for combo in combinations:
        combo['asset_id'] = asset_id
        combo['start_date'] = s_date
        combo['end_date'] = e_date
        combo['train_window_days'] = train_window_days
    
    # Run simulations in parallel with progress bar
    # REASON: Gebruik ProcessPoolExecutor om CPU cores te benutten. 
    # We houden 2 cores vrij voor systeemstabiliteit (16c/32t node).
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # We moeten steps_data meegeven aan elke process. Voor grote datasets kan dit geheugen kosten, 
        # maar met 1 jaar 1h data is dit prima.
        sim_func = functools.partial(run_single_simulation, steps_data=steps)
        
        # REASON: tqdm toont voortgangsbalk in terminal. file=sys.stdout zorgt voor live output via SSH.
        all_results = list(tqdm(
            executor.map(sim_func, combinations),
            total=len(combinations),
            desc="Simulating",
            unit="combo",
            file=sys.stdout,
            ncols=80
        ))

    print(f"✅ All {len(combinations)} simulations complete.", flush=True)
    logger.info(f"   ✅ All {len(combinations)} simulations complete.")

    # 7. Sorting & Reporting
    # Sorteer op Return, dan op Profit Factor
    # REASON: "Strategy Finder" moet primair de meest winstgevende configuratie vinden.
    sorted_results = sorted(all_results, key=lambda x: (x.get('total_pnl_pct', 0), x.get('profit_factor', 0)), reverse=True)
    
    logger.info("\n" + "🏆 TOP 5 STRATEGIES " + "="*30)
    for i, r in enumerate(sorted_results[:5]):
        p = r['params']
        logger.info(f"#{i+1}: Return: {r.get('total_pnl_pct', 0):.2f}% | Trades: {r.get('total_trades', 0)} | PF: {r.get('profit_factor', 0):.2f} | WR: {r.get('win_rate_pct', 0):.1f}%")
        logger.info(f"    Params: {p}")
        
    # Save results
    output_path = Path("_validation/grid_search")
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # REASON: Haal equity curves eruit voor de JSON om bestandsgrootte te beperken
    # En converteer Timestamps naar strings voor JSON serialisatie
    json_results = []
    for r in sorted_results:
        r_copy = r.copy()
        
        # Converteer params Timestamps naar strings
        if 'params' in r_copy:
            new_params = r_copy['params'].copy()
            for k, v in new_params.items():
                if hasattr(v, 'isoformat'):
                    new_params[k] = v.isoformat()
                elif isinstance(v, pd.Timestamp):
                    new_params[k] = str(v)
            r_copy['params'] = new_params

        if 'equity_curve' in r_copy:
            if r in sorted_results[:5]:
                # Converteer timestamps in de curve voor de Top 5
                r_copy['equity_curve'] = [
                    {**e, 'time': e['time'].isoformat() if hasattr(e['time'], 'isoformat') else str(e['time'])}
                    for e in r_copy['equity_curve']
                ]
            else:
                # Verwijder curve voor de rest om JSON klein te houden
                r_copy.pop('equity_curve', None)
        # Converteer numpy types naar Python natives
        for key in ['sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 'max_drawdown_usd', 
                   'avg_win_usd', 'avg_loss_usd', 'avg_trade_duration_hours']:
            if key in r_copy and r_copy[key] is not None:
                try:
                    r_copy[key] = float(r_copy[key])
                except (TypeError, ValueError):
                    pass
        json_results.append(r_copy)

    json_file = output_path / f"grid_search_{asset_id}_{ts}.json"
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    # --- Visualisatie Fase ---
    viz_dir = Path("_validation/strategy_results") / f"run_{ts}_asset_{asset_id}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Metadata opslaan voor herleidbaarheid
        metadata = {
            'asset_id': asset_id,
            'start_date': start_date if isinstance(start_date, str) else str(start_date),
            'end_date': end_date if isinstance(end_date, str) else str(end_date),
            'train_window': train_window_days,
            'step_days': step_days,
            'neutral_downsample': neutral_downsample,
            'timestamp': ts,
            'param_grid': param_grid,
            'best_return': sorted_results[0].get('total_pnl_pct', 0) if sorted_results else 0
        }
        with open(viz_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info("🎨 Generating Visualizations...")
        
        # 1. Top 5 Equity Curves
        plot_equity_curves(sorted_results[:5], viz_dir, asset_id)
        
        # 2. TP vs SL Bubble Chart (only if those params exist)
        if any('stop_loss_atr_mult' in r.get('params', {}) for r in all_results):
            plot_tp_sl_bubble(all_results, viz_dir, asset_id)
        
        # 3. Parameter Heatmap (adapt based on available params)
        plot_parameter_heatmap(all_results, viz_dir, asset_id, start_date, end_date)

        logger.info(f"📊 Visualizations saved to {viz_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error generating visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # REASON: Altijd het resultatenpad printen voor het menu script, ook bij (gedeeltelijke) fouten
    print(f"RESULT_DIR:{viz_dir}")
    sys.stdout.flush()

    # 8. Database Logging
    # REASON: Sla de resultaten op in de database voor herleidbaarheid en latere analyse.
    best_params = sorted_results[0]['params']
    try:
        run_id = f"run_{ts}_asset_{asset_id}"
        with get_cursor(commit=True) as cur:
            cur.execute("""
                INSERT INTO tsem.grid_search_history (
                    run_id, asset_id, start_date, end_date, 
                    param_grid, best_params, best_return_pct, 
                    best_pf, total_trades, win_rate, results_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    best_params = EXCLUDED.best_params,
                    best_return_pct = EXCLUDED.best_return_pct,
                    best_pf = EXCLUDED.best_pf;
            """, (
                run_id,
                asset_id,
                start_date,
                end_date,
                json.dumps(param_grid),
                json.dumps(best_params),
                sorted_results[0].get('total_pnl_pct', 0),
                sorted_results[0].get('profit_factor', 0),
                sorted_results[0].get('total_trades', 0),
                sorted_results[0].get('win_rate_pct', 0) / 100.0,  # Convert % to ratio
                str(viz_dir)
            ))
        logger.info(f"💾 Grid search results logged to database (tsem.grid_search_history).")
    except Exception as e:
        logger.error(f"❌ Error logging to database: {e}")

    # 9. TSEM Strategy Export
    # REASON: Automatische conversie van Grid Search winnaars naar productie-ready strategieën.
    try:
        logger.info(f"📤 Exporting winning parameters to tsem.strategy_configs...")
        with get_cursor(commit=True) as cur:
            base_params = {}
            if base_config_id:
                # 1. Haal base config op
                cur.execute("SELECT params FROM tsem.strategy_configs WHERE config_id = %s", (base_config_id,))
                row = cur.fetchone()
                if row:
                    base_params = row[0]
                    logger.info(f"   Using base configuration ID {base_config_id}")
                else:
                    logger.error(f"   ❌ Base configuration ID {base_config_id} not found. Using empty base.")
            else:
                # REASON: Als geen base_id is meegegeven, probeer de laatste actieve config te pakken als template
                cur.execute("SELECT params FROM tsem.strategy_configs WHERE is_active = true ORDER BY updated_at DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    base_params = row[0]
                    logger.info("   No base_config_id provided. Using last active config as template.")
                else:
                    logger.info("   No base config or active config found. Creating standalone config.")

            # 2. Merge winning parameters met base params
            # We behouden exchange, symbol, etc. van de base config, maar overschrijven trading logic params
            merged_params = base_params.copy()
            merged_params.update(best_params)
            
            # Voeg herleidbaarheid toe
            merged_params['_grid_search_run'] = run_id
            merged_params['_asset_id'] = asset_id
            
            # 3. Opslaan als nieuwe entry
            new_config_name = f"GS_Winner_{ts}_A{asset_id}"
            cur.execute("""
                INSERT INTO tsem.strategy_configs (config_name, params, is_active)
                VALUES (%s, %s, false)
                RETURNING config_id
            """, (new_config_name, json.dumps(merged_params)))
            new_id = cur.fetchone()[0]
            logger.info(f"   ✅ Strategy exported as ID {new_id} ('{new_config_name}')")
            
    except Exception as e:
        logger.error(f"❌ Error exporting strategy: {e}")

    logger.info(f"\n✅ Grid search complete. Results saved to {output_path}")
    sys.stdout.flush()

def plot_equity_curves(top_results, viz_dir, asset_id):
    """
    Plot de equity curves van de Top 5 strategieën.
    REASON: Groepeer identieke curves om overlap te voorkomen en toon alle relevante params.
    """
    plt.figure(figsize=(14, 8))
    
    # Dictionary om groepen bij te houden: hash(equity_waarden) -> { 'times': [], 'equity': [], 'labels': [] }
    curve_groups = {}
    
    for i, res in enumerate(top_results):
        curve = res.get('equity_curve', [])
        if not curve: continue
        
        # Maak een hashable tuple van de equity waarden om identieke verloopjes te vinden
        # We ronden af op 2 decimalen om float noise te negeren
        equity_vals = tuple([round(e['equity'], 2) for e in curve])
        
        p = res['params']
        # Compact label formaat - adaptief op basis van beschikbare params (v3.4)
        label_parts = []
        if 'stop_loss_atr_mult' in p:
            label_parts.append(f"SL:{p['stop_loss_atr_mult']:.1f}")
        if 'take_profit_atr_mult' in p:
            label_parts.append(f"TP:{p['take_profit_atr_mult']:.1f}")
        if 'entry_strength_threshold' in p:
            label_parts.append(f"STR:{p['entry_strength_threshold']}")
        if 'use_qbn_exit_timing' in p:
            label_parts.append(f"ET:{p['use_qbn_exit_timing']}")
        label = "|".join(label_parts) if label_parts else f"Config {i+1}"
        
        if equity_vals not in curve_groups:
            curve_groups[equity_vals] = {
                'times': [pd.to_datetime(e['time']) for e in curve],
                'equity': [e['equity'] for e in curve],
                'labels': [label],
                'return_pct': res.get('total_pnl_pct', 0)
            }
        else:
            curve_groups[equity_vals]['labels'].append(label)

    # Plot elke unieke curve
    for equity_vals, group in curve_groups.items():
        # Combineer labels als er meerdere sets dezelfde curve geven
        if len(group['labels']) > 3:
            final_label = f"{group['return_pct']:.1f}%: " + ", ".join(group['labels'][:2]) + f" (+{len(group['labels'])-2} others)"
        else:
            final_label = f"{group['return_pct']:.1f}%: " + " | ".join(group['labels'])
            
        plt.plot(group['times'], group['equity'], label=final_label, linewidth=2, alpha=0.8)

    plt.title(f"Unique Equity Curves - Top Performing Strategy Groups (Asset {asset_id})", fontsize=14)
    plt.xlabel("Tijd", fontsize=10)
    plt.ylabel("Account Balance ($)", fontsize=10)
    plt.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1)) # Legenda buiten de plot
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / "equity_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Equity Curves plot saved (grouped identical runs).")

def plot_tp_sl_bubble(all_results, viz_dir, asset_id):
    """Bubble Chart: SL vs TP, Grootte = Trades, Kleur = Return."""
    df = pd.DataFrame([{**r['params'], 
                        'return_pct': r.get('total_pnl_pct', 0), 
                        'total_trades': r.get('total_trades', 0)} for r in all_results])
    
    if 'stop_loss_atr_mult' not in df.columns or 'take_profit_atr_mult' not in df.columns:
        logger.info("⚠️ Skipping TP/SL bubble chart (params not in grid)")
        return

    plt.figure(figsize=(12, 9))
    
    # REASON: Bubble chart is veel informatiever dan een heatmap voor TP/SL.
    # Grootte toont de statistische betrouwbaarheid (aantal trades).
    scatter = plt.scatter(
        df['stop_loss_atr_mult'], df['take_profit_atr_mult'], 
        s=df['total_trades'] * 5, # Size multiplier
        c=df['return_pct'], 
        cmap='RdYlGn', 
        alpha=0.6, 
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label='Return %')
    plt.title(f"Strategy Performance: TP vs SL ATR Multiplier\nSize = Total Trades | Asset {asset_id}", fontsize=14)
    plt.xlabel("SL ATR Multiplier", fontsize=12)
    plt.ylabel("TP ATR Multiplier", fontsize=12)
    
    # Legenda voor de grootte
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5)
    # Correcte labels voor trades
    trade_labels = [f"{int(float(l.split('{')[-1].split('}')[0]) / 5)}" for l in labels]
    plt.legend(handles, trade_labels, loc="upper right", title="Trades", fontsize=8)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(viz_dir / "tp_sl_bubble.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ TP/SL Bubble Chart saved.")

def plot_parameter_heatmap(all_results, viz_dir, asset_id, start_date, end_date):
    """Generic Parameter Heatmap (adapts to available params)."""
    df = pd.DataFrame([{**r['params'], 'return_pct': r.get('total_pnl_pct', 0)} for r in all_results])
    
    # Probeer verschillende parameter combinaties (v3.4 aligned)
    param_pairs = [
        ('stop_loss_atr_mult', 'take_profit_atr_mult', 'SL vs TP ATR Multiplier'),
        ('entry_strength_threshold', 'use_qbn_exit_timing', 'Entry Strength vs Exit Timing'),
        ('leverage', 'volatility_position_sizing', 'Leverage vs Volatility Sizing'),
    ]
    
    for param1, param2, title_suffix in param_pairs:
        if param1 in df.columns and param2 in df.columns:
            # Filter rows waar beide params niet None zijn
            df_filtered = df.dropna(subset=[param1, param2])
            if df_filtered.empty:
                continue
            
            try:
                pivot = df_filtered.pivot_table(index=param1, columns=param2, values='return_pct', aggfunc='max')
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0, annot_kws={"size": 8})
                plt.title(f"Heatmap: MAX Return % ({title_suffix})\nAsset {asset_id} | {start_date} to {end_date}", fontsize=12)
                filename = f"heatmap_{param1}_vs_{param2}.png"
                plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"✅ Parameter heatmap saved: {filename}")
                return  # Toon alleen de eerste match
            except Exception as e:
                logger.warning(f"Could not create heatmap for {param1} vs {param2}: {e}")
                continue
    
    logger.info("⚠️ No suitable parameter pairs found for heatmap")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-id', type=int, default=1)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--window', type=int, default=365)
    parser.add_argument('--step', type=int, default=7)
    parser.add_argument('--neutral-downsample', type=float, default=1.0)
    parser.add_argument('--grid', type=str, required=True, help='JSON string with parameter grid')
    parser.add_argument('--base-config-id', type=int, help='Base configuration ID for export')
    
    args = parser.parse_args()
    grid = json.loads(args.grid)
    
    run_strategy_finder(args.asset_id, args.start, args.end, args.window, args.step, grid, args.neutral_downsample, args.base_config_id)
