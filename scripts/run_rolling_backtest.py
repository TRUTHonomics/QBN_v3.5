#!/usr/bin/env python3
"""
QBN Rolling Window Backtester.

Voert een wetenschappelijk correcte walk-forward validatie uit:
1. Train model op data t/m T.
2. Simuleer trading op T+1 (zonder voorkennis).
3. Update model met data van T+1.
4. Herhaal.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging
from simulation.data_loader import BacktestDataLoader
from simulation.execution_simulator import ExecutionSimulator
from strategies.standard_strategies import ProbabilityThresholdStrategy, RelativeEdgeStrategy
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
from inference.gpu.gpu_inference_engine import GPUInferenceEngine
from inference.inference_loader import InferenceLoader

logger = setup_logging("rolling_backtest")

def run_backtest(
    asset_id: int,
    start_date: str,
    end_date: str,
    train_window_days: int = 365,
    step_days: int = 7,
    strategy_params: dict = None,
    neutral_downsample: float = 1.0
):
    # 1. Setup
    s_date = pd.Timestamp(start_date).tz_localize('UTC')
    e_date = pd.Timestamp(end_date).tz_localize('UTC')
    
    # REASON: Calculate absolute date range for single data fetch
    # EXPL: We need data from (start_date - train_window) to end_date
    data_start_needed = s_date - timedelta(days=train_window_days)
    data_end_needed = e_date
    
    logger.info(f"🚀 Starting Rolling Backtest for Asset {asset_id}")
    logger.info(f"📅 Trading Period: {s_date.date()} to {e_date.date()}")
    logger.info(f"📚 Training Window: {train_window_days} days")
    logger.info(f"🔄 Step Size: {step_days} days")
    logger.info(f"⚖️ Neutral Downsample: {neutral_downsample}")
    logger.info(f"📥 Full Data Range: {data_start_needed.date()} to {data_end_needed.date()}")
    
    # 2. Components
    loader = BacktestDataLoader(asset_id)
    simulator = ExecutionSimulator(initial_capital=10000.0)
    
    strat_params = strategy_params or {}
    
    # REASON: Kies strategie op basis van parameters
    # EXPL: Als 'edge_factor' aanwezig is, gebruik de RelativeEdgeStrategy
    if 'edge_factor' in strat_params:
        logger.info(f"⚖️ Using RelativeEdgeStrategy (Edge Factor: {strat_params['edge_factor']})")
        strategy = RelativeEdgeStrategy(
            edge_factor=strat_params.get('edge_factor', 1.5),
            min_total_signal=strat_params.get('min_total_signal', 0.02),
            horizon=strat_params.get('horizon', '1h'),
            atr_tp=strat_params.get('atr_tp', 2.0),
            atr_sl=strat_params.get('atr_sl', 1.5)
        )
    else:
        logger.info(f"🎯 Using ProbabilityThresholdStrategy (Min Prob: {strat_params.get('min_prob', 0.60)})")
    strategy = ProbabilityThresholdStrategy(
        min_prob=strat_params.get('min_prob', 0.60),
        horizon=strat_params.get('horizon', '1h'),
        atr_tp=strat_params.get('atr_tp', 2.0),
        atr_sl=strat_params.get('atr_sl', 1.5)
    )
    
    # 3. FETCH ONCE: Haal alle benodigde data in één keer op
    logger.info("📥 Fetching full dataset (single DB call)...")
    full_df = loader.fetch_data(data_start_needed, data_end_needed)
    
    if full_df.empty or len(full_df) < 1000:
        logger.error("❌ Insufficient data for backtest. Aborting.")
        return
        
    # REASON: Fix timezone mismatch (TZ-aware vs Naive)
    if 'time_1' in full_df.columns:
        # Zorg dat de kolom datetime objects bevat
        full_df['time_1'] = pd.to_datetime(full_df['time_1'])
        
        # Als time_1 naive is, maak UTC. Als al aware, convert naar UTC voor zekerheid.
        if full_df['time_1'].dt.tz is None:
            logger.info("ℹ️ Converting naive timestamps to UTC...")
            full_df['time_1'] = full_df['time_1'].dt.tz_localize('UTC')
        else:
            full_df['time_1'] = full_df['time_1'].dt.tz_convert('UTC')
            
    logger.info(f"✅ Full dataset loaded: {len(full_df)} rows")
    logger.info(f"   Date Range: {full_df['time_1'].min()} to {full_df['time_1'].max()}")
    
    # REASON: Gebruik de index voor sneller en robuuster slicen
    full_df.set_index('time_1', inplace=True, drop=False)
    full_df.sort_index(inplace=True)
    
    # 4. PREPROCESS ONCE: Bereken HTF_Regime en composites voor volledige dataset
    # REASON: Preprocessing op GPU is duur, doen we maar 1x
    logger.info("⚙️ Preprocessing full dataset (HTF_Regime, Composites)...")
    
    # We hebben een generator nodig voor preprocessing, maar gebruiken die niet voor CPTs hier
    preprocessing_generator = QBNv3CPTGenerator(neutral_downsample=neutral_downsample)
    full_df = preprocessing_generator.preprocess_dataset(full_df, asset_id)
    
    # Cleanup preprocessing generator (we maken per step een nieuwe voor CPTs)
    del preprocessing_generator
    
    logger.info(f"✅ Preprocessing complete. Columns: {len(full_df.columns)}")
    
    # Ensure time_1 column is available for slicing
    if 'time_1' not in full_df.columns:
        logger.error("❌ time_1 column missing after preprocessing!")
        return
    
    # 5. Main Loop (SLICE from memory, no more DB calls)
    current_train_end = s_date
    step_count = 0
    total_steps = int(np.ceil((e_date - s_date).days / step_days))
    
    while current_train_end < e_date:
        test_end = min(current_train_end + timedelta(days=step_days), e_date)
        train_start = current_train_end - timedelta(days=train_window_days)
        step_count += 1
        
        logger.info(f"\n🔄 Step {step_count}/{total_steps}: Testing {current_train_end.date()} to {test_end.date()} (Training on {train_start.date()}-{current_train_end.date()})")
        
        # A. SLICE TRAIN DATA (in-memory, instant)
        # REASON: Gebruik .loc voor robuuste tijdsgebaseerde slicing op de index
        train_df = full_df.loc[train_start : current_train_end - timedelta(seconds=1)].copy()
        # REASON: Reset index om ambiguïteit te voorkomen bij pd.merge in CPT generator
        train_df = train_df.reset_index(drop=True)
        
        if train_df.empty or len(train_df) < 1000:
            logger.warning(f"⚠️ Insufficient training data ({len(train_df)} rows) for {train_start} to {current_train_end}. Skipping step.")
            current_train_end = test_end
            continue
            
        logger.info(f"   📊 Train slice: {len(train_df)} rows")
        
        # B. TRAIN (Genereer CPTs op basis van historie)
        # REASON: Nieuwe generator per step om state vervuiling te voorkomen
        generator = QBNv3CPTGenerator(neutral_downsample=neutral_downsample)
            
        # Genereer CPTs in memory
        cpts = {}
        cpts['HTF_Regime'] = generator.generate_htf_regime_cpt(asset_id, data=train_df)
        
        # Load classification once for this step
        generator.load_signal_classification(asset_id=asset_id, horizon='1h', filter_suffix='60')
        
        # Composites
        from inference.node_types import SemanticClass
        for sem_class in SemanticClass:
            node_name = f"{sem_class.value.capitalize()}_Composite"
            cpts[node_name] = generator.generate_composite_cpt(asset_id, sem_class, data=train_df, horizon='1h')
            
        # Intermediate
        cpts['Trade_Hypothesis'] = generator.generate_trade_hypothesis_cpt(asset_id, data=train_df)
        cpts['Entry_Confidence'] = generator.generate_entry_confidence_cpt(asset_id, data=train_df) # Gebruikt deprecated logic voor compatibiliteit
        
        # Predictions
        for h in ['1h', '4h', '1d']:
            node_name = f"Prediction_{h}"
            cpts[node_name] = generator.generate_prediction_cpt(asset_id, h, data=train_df) # lookback ignored if data provided
            
        # C. INFERENCE (Voorspel op test window)
        # Laad nieuwe engine met getrainde CPTs
        # We hebben ook de signal classification nodig. Die zit in generator.signal_aggregator.
        if not generator.signal_aggregator:
             generator.load_signal_classification(asset_id=asset_id, horizon='1h')
             
        engine = GPUInferenceEngine(
            cpts=cpts,
            signal_classification=generator.signal_aggregator.signal_classification
        )
        
        # SLICE TEST DATA (in-memory, instant)
        # REASON: Gebruik .loc voor robuuste tijdsgebaseerde slicing op de index
        test_df = full_df.loc[current_train_end : test_end - timedelta(seconds=1)].copy()
        # REASON: Reset index om ambiguïteit te voorkomen bij inference
        test_df = test_df.reset_index(drop=True)
        
        if test_df.empty:
            logger.warning(f"⚠️ No test data found in slice {current_train_end} to {test_end}, skipping.")
            current_train_end = test_end
            continue
            
        logger.info(f"   📊 Test slice: {len(test_df)} rows")
            
        # Batch inference
        # GPU engine verwacht specifieke kolomnamen. Onze loader levert die (met _60 suffix).
        predictions = engine.infer_batch(test_df)
        
        # D. SIMULATE (Trade executie)
        # Loop door de candles
        # REASON: enumerate() voor 0-based index in predictions array
        # EXPL: test_df behoudt originele index uit full_df, predictions is 0-geïndexeerd
        last_row = None
        for i, (_, row) in enumerate(test_df.iterrows()):
            last_row = row
            # Update simulator met High/Low van deze candle (voor TP/SL checks van VORIGE signalen)
            simulator.update_price(row)
            
            # Bouw prediction object voor strategy (i = 0-based index in predictions)
            current_preds = {
                'regime': predictions['regime'][i],
                'trade_hypothesis': predictions['trade_hypothesis'][i],
                'prediction_1h': predictions['predictions']['1h']['distributions'][i],
                'prediction_4h': predictions['predictions']['4h']['distributions'][i],
                'prediction_1d': predictions['predictions']['1d']['distributions'][i]
            }
            
            # Vraag strategy om signaal
            market_data = row.to_dict()
            signal = strategy.on_data(
                timestamp=row['time_1'],
                predictions=current_preds,
                market_data=market_data,
                current_position=simulator.active_trade.direction if simulator.active_trade else None
            )
            
            # DEBUG: Log logic to understand 0 trades
            dist = current_preds['prediction_1h']
            p_up = sum(prob for state, prob in dist.items() if 'up' in state.lower() or 'bullish' in state.lower())
            p_down = sum(prob for state, prob in dist.items() if 'down' in state.lower() or 'bearish' in state.lower())
            
            # Log full distribution periodically or on significant signals
            if p_up > 0.40 or p_down > 0.40 or i % 1000 == 0:
                 logger.info(f"DEBUG [{row['time_1']}] P(Up)={p_up:.2f}, P(Down)={p_down:.2f}, Dist={dist}")
            
            # Verwerk signaal (opent mogelijk trade op CLOSE van deze candle)
            simulator.process_signal(signal, row)

        # REASON: Consistente metrics aan het einde van elke test-slice (geen open positie laten hangen)
        if last_row is not None:
            simulator.close_active_trade(last_row, reason="End of Test Slice")
            
        # E. CLEANUP (GPU Memory & Objects)
        # REASON: Voorkom container crash door OOM (Out of Memory) bij rolling training
        # EXPL: train_df en test_df zijn slices (.copy()), kunnen veilig verwijderd worden
        import gc
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except ImportError:
            pass
            
        del engine
        del generator
        del train_df
        del test_df
        del predictions
        gc.collect()
            
        # Move window
        current_train_end = test_end
        
    # 6. Reporting
    results = simulator.get_results()
    
    logger.info("\n" + "="*50)
    logger.info("🏁 BACKTEST COMPLETE")
    logger.info("="*50)
    logger.info(f"Return:        {results.get('return_pct', 0.0):.2f}%")
    logger.info(f"Trades:        {results.get('total_trades', 0)}")
    logger.info(f"Win Rate:      {results.get('win_rate', 0.0):.2%}")
    logger.info(f"Profit Factor: {results.get('profit_factor', 0.0):.2f}")
    logger.info(f"Max Drawdown:  {results.get('max_drawdown', 0.0):.2%}")
    logger.info(f"Final Capital: {results.get('final_capital', 10000.0):.2f}")
    logger.info("="*50)
    
    if results.get('total_trades', 0) == 0:
        logger.warning("⚠️ Geen trades uitgevoerd! Check min_prob threshold of signaal kwaliteit.")
    
    # Save detailed report
    output_dir = Path("_validation/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV van trades
    trades_df = pd.DataFrame([t.__dict__ for t in simulator.trades])
    if not trades_df.empty:
        trades_df.to_csv(output_dir / f"trades_{ts}.csv", index=False)
        
    # JSON results
    with open(output_dir / f"results_{ts}.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-id', type=int, default=1)
    parser.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--window', type=int, default=365)
    parser.add_argument('--step', type=int, default=7)
    parser.add_argument('--neutral-downsample', type=float, default=1.0, help='Weight for neutral observations (default 1.0)')
    parser.add_argument('--params', type=str, default='{}', help='JSON string with strategy params')
    
    args = parser.parse_args()
    
    import json
    strat_params = json.loads(args.params)
    
    run_backtest(
        args.asset_id, 
        args.start, 
        args.end, 
        args.window, 
        args.step,
        strategy_params=strat_params,
        neutral_downsample=args.neutral_downsample
    )
