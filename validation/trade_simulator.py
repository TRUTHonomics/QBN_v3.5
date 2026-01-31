"""
Trade Simulator voor QBN Backtest Engine.

Simuleert de volledige trade lifecycle op basis van QBN signalen en configureerbare
exit parameters. Gebruikt 1m OHLC data voor nauwkeurige intrabar stop/target simulatie.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from database.db import get_cursor

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Representatie van een simulated trade."""
    # Entry
    signal_timestamp: datetime
    entry_timestamp: datetime
    entry_price: float
    direction: str  # 'long' of 'short'
    position_size_usd: float
    position_size_units: float
    entry_fees_usd: float
    entry_slippage_pct: float
    
    # Planned Exits
    planned_stop_loss: float
    planned_take_profit: float
    atr_at_entry: float
    
    # QBN Context
    htf_regime: str = ""
    trade_hypothesis: str = ""
    momentum_prediction: str = ""
    volatility_regime: str = ""
    exit_timing: str = ""
    position_confidence: str = ""
    leading_composite: str = ""
    coincident_composite: str = ""
    confirming_composite: str = ""
    
    # v3.4: Raw composite scores at entry (for delta calculation)
    entry_composite_scores: Optional[Dict[str, float]] = None
    
    # Exit (filled during simulation)
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_fees_usd: float = 0.0
    
    # Trailing Stop State
    trailing_stop_activated: bool = False
    trailing_stop_highest_price: Optional[float] = None
    trailing_stop_lowest_price: Optional[float] = None
    
    # Performance Metrics (computed at exit)
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    mae_pct: float = 0.0  # Maximum Adverse Excursion
    mfe_pct: float = 0.0  # Maximum Favorable Excursion
    holding_duration_hours: float = 0.0
    
    # Tracking voor MAE/MFE
    _worst_price: Optional[float] = None
    _best_price: Optional[float] = None
    
    def update_extremes(self, price: float):
        """Update MAE en MFE tracking."""
        if self._worst_price is None:
            self._worst_price = price
            self._best_price = price
        
        if self.direction == 'long':
            self._worst_price = min(self._worst_price, price)
            self._best_price = max(self._best_price, price)
        else:  # short
            self._worst_price = max(self._worst_price, price)
            self._best_price = min(self._best_price, price)
    
    def compute_mae_mfe(self):
        """Bereken MAE en MFE percentages."""
        if self._worst_price is None or self._best_price is None:
            return
        
        if self.direction == 'long':
            self.mae_pct = ((self._worst_price - self.entry_price) / self.entry_price) * 100
            self.mfe_pct = ((self._best_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            self.mae_pct = ((self.entry_price - self._worst_price) / self.entry_price) * 100
            self.mfe_pct = ((self.entry_price - self._best_price) / self.entry_price) * 100
    
    def is_open(self) -> bool:
        """Check of trade nog open is."""
        return self.exit_timestamp is None


class TradeSimulator:
    """
    Simuleert trade lifecycle op basis van QBN signalen en configureerbare parameters.
    
    Flow:
    1. Ontvang QBN inference result (entry signaal)
    2. Evalueer entry criteria (filters)
    3. Open trade met configured parameters
    4. Simuleer intrabar price action (via 1m OHLC)
    5. Check exits: SL, TP, trailing stop, timeout, QBN exit signaal
    6. Sluit trade en bereken PnL
    """
    
    def __init__(self, config: 'BacktestConfig'):
        """
        Initialiseer Trade Simulator.
        
        Args:
            config: BacktestConfig object met alle parameters
        """
        from validation.backtest_config import BacktestConfig
        self.config = config
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.current_capital = config.initial_capital_usd
        self.peak_capital = config.initial_capital_usd
        
        # Cache voor 1m OHLC data (voor intrabar simulation)
        self._ohlc_cache: Dict[datetime, pd.DataFrame] = {}
        
        logger.info(f"Trade Simulator initialized: "
                   f"capital=${config.initial_capital_usd:,.2f}, "
                   f"leverage={config.leverage}x, "
                   f"SL={config.stop_loss_atr_mult}x ATR, "
                   f"TP={config.take_profit_atr_mult}x ATR")
    
    def should_enter_trade(self, inference_result: 'DualInferenceResult') -> Tuple[bool, Optional[str]]:
        """
        Entry beslissing gebaseerd ALLEEN op Trade_Hypothesis (Entry-side node).
        Position-side nodes worden NIET gebruikt voor entry (alleen voor management).
        
        Args:
            inference_result: QBN inference output
            
        Returns:
            Tuple[should_enter, direction] waar direction 'long' of 'short' is
        """
        hyp = inference_result.trade_hypothesis.lower()
        
        # Trade_Hypothesis states: no_setup, weak_long, strong_long, weak_short, strong_short
        if hyp == 'no_setup':
            return False, None
        
        # Parse hypothesis: "{strength}_{direction}"
        parts = hyp.split('_')
        if len(parts) != 2:
            return False, None
        
        strength, direction = parts[0], parts[1]
        
        # Validate
        if strength not in ['weak', 'strong']:
            return False, None
        if direction not in ['long', 'short']:
            return False, None
        
        # Check strength threshold (alleen 2 levels: weak < strong)
        min_strength = self.config.entry_strength_threshold.lower()
        if min_strength == 'strong' and strength == 'weak':
            return False, None  # Skip weak signals als threshold strong is
        
        # Optional: Regime filter (structural layer - valid voor entry)
        if self.config.regime_filter:
            regime = inference_result.regime.lower()
            allowed_regimes = [r.lower() for r in self.config.regime_filter]
            if regime not in allowed_regimes:
                return False, None
        
        return True, direction
    
    def open_trade(
        self, 
        inference_result: 'DualInferenceResult',
        current_price: float,
        atr: float,
        direction: str,
        entry_composite_scores: Optional[Dict[str, float]] = None
    ) -> Optional[Trade]:
        """
        Open een nieuwe trade op basis van QBN signaal.
        
        Args:
            inference_result: QBN inference output
            current_price: Huidige prijs
            atr: ATR waarde voor SL/TP berekening
            direction: 'long' of 'short' (afkomstig van should_enter_trade)
            entry_composite_scores: Optional composite scores voor delta tracking
        """
        # Bereken position size
        if self.config.position_size_usd:
            size_usd = self.config.position_size_usd
        else:
            # Percentage van current capital
            size_usd = self.current_capital * (self.config.position_size_pct / 100.0)
        
        # v3.4: Volatility-based position sizing
        if self.config.volatility_position_sizing:
            vol_regime = inference_result.volatility_regime.lower()
            if vol_regime == 'low_vol':
                size_usd *= 1.2  # Grotere positie in lage volatility
            elif vol_regime == 'high_vol':
                size_usd *= 0.5  # Kleinere positie in hoge volatility
            # 'normal' = 1.0x (geen aanpassing)
        
        # Apply leverage
        leveraged_size_usd = size_usd * self.config.leverage
        position_size_units = leveraged_size_usd / current_price
        
        # Entry slippage
        slippage = self.config.slippage_pct / 100.0
        entry_price = current_price * (1 + slippage) if direction == 'long' else current_price * (1 - slippage)
        
        # Entry fees (taker for market orders)
        entry_fees = leveraged_size_usd * (self.config.taker_fee_pct / 100.0)
        
        # Calculate stop loss and take profit
        if self.config.use_atr_based_exits:
            if direction == 'long':
                sl = entry_price - (atr * self.config.stop_loss_atr_mult)
                tp = entry_price + (atr * self.config.take_profit_atr_mult)
            else:  # short
                sl = entry_price + (atr * self.config.stop_loss_atr_mult)
                tp = entry_price - (atr * self.config.take_profit_atr_mult)
        else:
            # Fixed percentage
            sl_pct = self.config.stop_loss_pct or 2.0
            tp_pct = self.config.take_profit_pct or 3.0
            
            if direction == 'long':
                sl = entry_price * (1 - sl_pct / 100.0)
                tp = entry_price * (1 + tp_pct / 100.0)
            else:  # short
                sl = entry_price * (1 + sl_pct / 100.0)
                tp = entry_price * (1 - tp_pct / 100.0)
        
        trade = Trade(
            signal_timestamp=inference_result.timestamp,
            entry_timestamp=inference_result.timestamp,
            entry_price=entry_price,
            direction=direction,
            position_size_usd=leveraged_size_usd,
            position_size_units=position_size_units,
            entry_fees_usd=entry_fees,
            entry_slippage_pct=self.config.slippage_pct,
            planned_stop_loss=sl,
            planned_take_profit=tp,
            atr_at_entry=atr,
            # QBN Context
            htf_regime=inference_result.regime,
            trade_hypothesis=inference_result.trade_hypothesis,
            momentum_prediction=inference_result.momentum_prediction,
            volatility_regime=inference_result.volatility_regime,
            exit_timing=inference_result.exit_timing,
            position_confidence=inference_result.position_confidence,
            leading_composite=inference_result.leading_composite,
            coincident_composite=inference_result.coincident_composite,
            confirming_composite=inference_result.confirming_composite,
            # v3.4: Store raw scores for delta calculation
            entry_composite_scores=entry_composite_scores,
        )
        
        self.open_trades.append(trade)
        logger.debug(f"Opened {direction} trade @ {entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
        
        return trade
    
    def update_open_trades(
        self, 
        current_time: datetime,
        ohlc_1m: pd.DataFrame,
        latest_inference: Optional['DualInferenceResult'] = None
    ):
        """
        Update alle open trades met huidige price action.
        
        Simuleert intrabar (1m) price movements om exacte SL/TP hits te detecteren.
        
        Args:
            current_time: Huidige timestamp (60m candle close)
            ohlc_1m: DataFrame met 1m OHLC data voor het afgelopen uur
            latest_inference: Optioneel: laatste QBN inference voor exit signalen
        """
        for trade in self.open_trades[:]:  # Copy list om tijdens iteratie te kunnen verwijderen
            # Check timeout
            if self.config.max_holding_time_hours:
                hours_held = (current_time - trade.entry_timestamp).total_seconds() / 3600
                if hours_held >= self.config.max_holding_time_hours:
                    self._close_trade(trade, current_time, ohlc_1m.iloc[-1]['close'], 'timeout')
                    continue
            
            # Check QBN exit signals
            if self.config.use_qbn_exit_timing and latest_inference:
                if latest_inference.exit_timing.lower() == 'exit_now':
                    self._close_trade(trade, current_time, ohlc_1m.iloc[-1]['close'], 'qbn_exit_signal')
                    continue
            
            if self.config.exit_on_momentum_reversal and latest_inference:
                current_mp = latest_inference.momentum_prediction.lower()
                
                # Long met bearish momentum = exit (momentum draait tegen positie)
                if trade.direction == 'long' and current_mp == 'bearish':
                    self._close_trade(trade, current_time, ohlc_1m.iloc[-1]['close'], 'momentum_reversal')
                    continue
                # Short met bullish momentum = exit
                if trade.direction == 'short' and current_mp == 'bullish':
                    self._close_trade(trade, current_time, ohlc_1m.iloc[-1]['close'], 'momentum_reversal')
                    continue
            
            # v3.4: Position Prediction exit
            if self.config.use_position_prediction_exit and latest_inference:
                pp = latest_inference.position_prediction.lower() if hasattr(latest_inference, 'position_prediction') else ''
                if pp in ['stoploss_hit', 'timeout']:
                    self._close_trade(trade, current_time, ohlc_1m.iloc[-1]['close'], f'position_prediction_{pp}')
                    continue
            
            # Simulate intrabar SL/TP hits
            exit_reason = self._check_intrabar_exits(trade, ohlc_1m)
            if exit_reason:
                # Vind de exacte exit price en tijd
                exit_price, exit_time = self._find_exit_price(trade, ohlc_1m, exit_reason)
                self._close_trade(trade, exit_time, exit_price, exit_reason)
    
    def _check_intrabar_exits(self, trade: Trade, ohlc_1m: pd.DataFrame) -> Optional[str]:
        """
        Check of SL of TP wordt geraakt binnen de 1m candles.
        
        Returns:
            'stop_loss', 'take_profit', 'trailing_stop', of None
        """
        if ohlc_1m.empty:
            return None
        
        for idx, row in ohlc_1m.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            
            # Update MAE/MFE
            trade.update_extremes(high)
            trade.update_extremes(low)
            
            # Update trailing stop
            if self.config.trailing_stop_enabled:
                self._update_trailing_stop(trade, high, low)
            
            # Check exits (order matters!)
            if trade.direction == 'long':
                # Check SL first (prioriteit)
                if low <= trade.planned_stop_loss:
                    return 'stop_loss'
                # Check trailing stop
                if self.config.trailing_stop_enabled and trade.trailing_stop_activated:
                    if low <= trade.trailing_stop_lowest_price:
                        return 'trailing_stop'
                # Check TP
                if high >= trade.planned_take_profit:
                    return 'take_profit'
            else:  # short
                # Check SL first
                if high >= trade.planned_stop_loss:
                    return 'stop_loss'
                # Check trailing stop
                if self.config.trailing_stop_enabled and trade.trailing_stop_activated:
                    if high >= trade.trailing_stop_highest_price:
                        return 'trailing_stop'
                # Check TP
                if low <= trade.planned_take_profit:
                    return 'take_profit'
        
        return None
    
    def _update_trailing_stop(self, trade: Trade, high: float, low: float):
        """Update trailing stop level op basis van peak profit."""
        if trade.direction == 'long':
            current_profit_pct = ((high - trade.entry_price) / trade.entry_price) * 100
            
            # Activeer trailing als profit > activation threshold
            if not trade.trailing_stop_activated:
                if current_profit_pct >= self.config.trailing_activation_pct:
                    trade.trailing_stop_activated = True
                    trade.trailing_stop_highest_price = high
                    logger.debug(f"Trailing stop activated @ {high:.2f}")
            else:
                # Update trailing high
                if high > trade.trailing_stop_highest_price:
                    trade.trailing_stop_highest_price = high
                
                # Calculate trailing stop level
                trail_pct = self.config.trailing_stop_pct / 100.0
                profit = trade.trailing_stop_highest_price - trade.entry_price
                trade.trailing_stop_lowest_price = trade.entry_price + (profit * (1 - trail_pct))
        
        else:  # short
            current_profit_pct = ((trade.entry_price - low) / trade.entry_price) * 100
            
            if not trade.trailing_stop_activated:
                if current_profit_pct >= self.config.trailing_activation_pct:
                    trade.trailing_stop_activated = True
                    trade.trailing_stop_lowest_price = low
                    logger.debug(f"Trailing stop activated @ {low:.2f}")
            else:
                # Update trailing low
                if low < trade.trailing_stop_lowest_price:
                    trade.trailing_stop_lowest_price = low
                
                # Calculate trailing stop level
                trail_pct = self.config.trailing_stop_pct / 100.0
                profit = trade.entry_price - trade.trailing_stop_lowest_price
                trade.trailing_stop_highest_price = trade.entry_price - (profit * (1 - trail_pct))
    
    def _find_exit_price(
        self, 
        trade: Trade, 
        ohlc_1m: pd.DataFrame, 
        exit_reason: str
    ) -> Tuple[float, datetime]:
        """
        Vind de exacte exit price en tijd binnen de 1m candles.
        
        Returns:
            (exit_price, exit_timestamp)
        """
        for idx, row in ohlc_1m.iterrows():
            if exit_reason == 'stop_loss':
                if trade.direction == 'long' and row['low'] <= trade.planned_stop_loss:
                    return (trade.planned_stop_loss, row['time'])
                elif trade.direction == 'short' and row['high'] >= trade.planned_stop_loss:
                    return (trade.planned_stop_loss, row['time'])
            
            elif exit_reason == 'take_profit':
                if trade.direction == 'long' and row['high'] >= trade.planned_take_profit:
                    return (trade.planned_take_profit, row['time'])
                elif trade.direction == 'short' and row['low'] <= trade.planned_take_profit:
                    return (trade.planned_take_profit, row['time'])
            
            elif exit_reason == 'trailing_stop':
                if trade.direction == 'long' and row['low'] <= trade.trailing_stop_lowest_price:
                    return (trade.trailing_stop_lowest_price, row['time'])
                elif trade.direction == 'short' and row['high'] >= trade.trailing_stop_highest_price:
                    return (trade.trailing_stop_highest_price, row['time'])
        
        # Fallback: gebruik laatste candle close
        return (ohlc_1m.iloc[-1]['close'], ohlc_1m.iloc[-1]['time'])
    
    def _close_trade(
        self, 
        trade: Trade, 
        exit_time: datetime, 
        exit_price: float, 
        exit_reason: str
    ):
        """
        Sluit een trade en bereken PnL.
        
        Args:
            trade: Trade object om te sluiten
            exit_time: Timestamp van exit
            exit_price: Exit prijs
            exit_reason: Reden voor exit
        """
        trade.exit_timestamp = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Exit fees
        trade.exit_fees_usd = trade.position_size_usd * (self.config.taker_fee_pct / 100.0)
        
        # Bereken PnL
        if trade.direction == 'long':
            price_change = exit_price - trade.entry_price
        else:  # short
            price_change = trade.entry_price - exit_price
        
        trade.gross_pnl_usd = (price_change / trade.entry_price) * trade.position_size_usd
        trade.net_pnl_usd = trade.gross_pnl_usd - trade.entry_fees_usd - trade.exit_fees_usd
        trade.pnl_pct = (trade.net_pnl_usd / (trade.position_size_usd / self.config.leverage)) * 100
        
        # MAE/MFE
        trade.compute_mae_mfe()
        
        # Holding duration
        duration_seconds = (exit_time - trade.entry_timestamp).total_seconds()
        trade.holding_duration_hours = duration_seconds / 3600.0
        
        # Update capital
        self.current_capital += trade.net_pnl_usd
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Move trade from open to closed
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        logger.debug(f"Closed {trade.direction} trade: {exit_reason}, "
                    f"PnL=${trade.net_pnl_usd:.2f} ({trade.pnl_pct:.2f}%), "
                    f"held {trade.holding_duration_hours:.1f}h")
    
    def get_ohlc_1m(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Haal 1m OHLC data op voor intrabar simulatie.
        
        Args:
            start_time: Start timestamp (60m candle open)
            end_time: End timestamp (60m candle close)
            
        Returns:
            DataFrame met 1m OHLC data
        """
        cache_key = (start_time, end_time)
        if cache_key in self._ohlc_cache:
            return self._ohlc_cache[cache_key]
        
        query = """
            SELECT time, open, high, low, close
            FROM kfl.indicators
            WHERE asset_id = %s 
              AND interval_min = '1' 
              AND time >= %s 
              AND time < %s
            ORDER BY time ASC
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.config.asset_id, start_time, end_time))
            rows = cur.fetchall()
            if not rows:
                # Fallback: gebruik 60m data als 1m niet beschikbaar
                logger.warning(f"No 1m data available for {start_time}, using 60m approximation")
                return pd.DataFrame([{
                    'time': end_time,
                    'open': 0,
                    'high': 0,
                    'low': 0,
                    'close': 0
                }])
            
            df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close'])
            self._ohlc_cache[cache_key] = df
            return df
    
    def get_metrics(self) -> Dict:
        """Bereken backtest performance metrics."""
        if not self.closed_trades:
            return {}
        
        # Basic stats
        total_trades = len(self.closed_trades)
        winners = [t for t in self.closed_trades if t.net_pnl_usd > 0.01]
        losers = [t for t in self.closed_trades if t.net_pnl_usd < -0.01]
        breakeven = [t for t in self.closed_trades if abs(t.net_pnl_usd) <= 0.01]
        
        total_pnl = sum(t.net_pnl_usd for t in self.closed_trades)
        total_pnl_pct = ((self.current_capital - self.config.initial_capital_usd) / self.config.initial_capital_usd) * 100
        
        # Win rate
        win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.net_pnl_usd for t in winners) if winners else 0
        gross_loss = abs(sum(t.net_pnl_usd for t in losers)) if losers else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Averages
        avg_win = (gross_profit / len(winners)) if winners else 0
        avg_loss = (gross_loss / len(losers)) if losers else 0
        avg_duration = np.mean([t.holding_duration_hours for t in self.closed_trades])
        
        # Drawdown
        equity_curve = []
        running_capital = self.config.initial_capital_usd
        for t in self.closed_trades:
            running_capital += t.net_pnl_usd
            equity_curve.append(running_capital)
        
        peak = self.config.initial_capital_usd
        max_dd = 0
        max_dd_usd = 0
        for capital in equity_curve:
            peak = max(peak, capital)
            dd = ((peak - capital) / peak) * 100
            dd_usd = peak - capital
            max_dd = max(max_dd, dd)
            max_dd_usd = max(max_dd_usd, dd_usd)
        
        # Sharpe/Sortino (simplified - daily returns assumed)
        returns = [t.pnl_pct for t in self.closed_trades]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # Sortino: alleen downside volatility
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            else:
                sortino = sharpe
        else:
            sharpe = 0
            sortino = 0
        
        return {
            'final_capital_usd': self.current_capital,
            'total_pnl_usd': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'breakeven_trades': len(breakeven),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'avg_trade_duration_hours': avg_duration,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd,
            'max_drawdown_usd': max_dd_usd,
        }
