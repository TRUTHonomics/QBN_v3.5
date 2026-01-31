import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
from datetime import timedelta
import logging

from strategies.base_strategy import TradeSignal

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: Literal['LONG', 'SHORT']
    entry_price: float
    exit_price: float
    quantity: float
    tp_price: float
    sl_price: float
    status: Literal['OPEN', 'CLOSED']
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    fee: float = 0.0

    # REASON: Analyse data
    p_up: float = 0.0
    p_down: float = 0.0
    regime: str = ""
    hypothesis: str = ""
    entry_equity: float = 0.0
    exit_equity: float = 0.0

class ExecutionSimulator:
    """
    Simuleert trade executie met realistische TP/SL hits en fees.
    """
    
    def __init__(self, initial_capital: float = 10000.0, fee_pct: float = 0.06, slippage_pct: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fee_pct = fee_pct / 100.0
        self.slippage_pct = slippage_pct / 100.0
        self.trades: List[Trade] = []
        self.active_trade: Optional[Trade] = None
        self.equity_curve = []
        
    def process_signal(self, signal: TradeSignal, row: pd.Series):
        """
        Verwerk een nieuw signaal op een bepaalde candle (row).
        
        Args:
            signal: TradeSignal object van de strategy
            row: Pandas Series met OHLC + ATR data van de huidige candle (sluiting)
        """
        current_time = row['time_1']
        close_price = row['close']
        atr = row.get('atr_14', 0.0)
        
        # 1. Beheer actieve trade
        if self.active_trade:
            # Check exit signalen vanuit strategie (bijv. reversal)
            if signal.action == 'EXIT' or \
               (self.active_trade.direction == 'LONG' and signal.action == 'SHORT') or \
               (self.active_trade.direction == 'SHORT' and signal.action == 'LONG'):
                
                self._close_trade(self.active_trade, close_price, current_time, signal.reason)
                self.active_trade = None
                
                # Als het een reversal was, kunnen we direct openen in de nieuwe richting
                if signal.action in ['LONG', 'SHORT'] and signal.action != 'EXIT':
                    pass # Doorgaan naar open logica
                else:
                    return # Klaar voor deze candle
                    
        # 2. Open nieuwe trade (als geen actieve trade)
        if not self.active_trade and signal.action in ['LONG', 'SHORT']:
            direction = signal.action
            
            # Prijsbepaling met slippage
            slip_mult = (1 + self.slippage_pct) if direction == 'LONG' else (1 - self.slippage_pct)
            entry_price = close_price * slip_mult
            
            # TP/SL berekening
            tp_price = signal.tp_price
            sl_price = signal.sl_price
            
            if not tp_price and signal.tp_atr_mult and atr > 0:
                dist = signal.tp_atr_mult * atr
                tp_price = entry_price + dist if direction == 'LONG' else entry_price - dist
                
            if not sl_price and signal.sl_atr_mult and atr > 0:
                dist = signal.sl_atr_mult * atr
                sl_price = entry_price - dist if direction == 'LONG' else entry_price + dist
            
            # Fallback als ATR 0 is of geen params
            if not tp_price: tp_price = entry_price * (1.02 if direction == 'LONG' else 0.98)
            if not sl_price: sl_price = entry_price * (0.99 if direction == 'LONG' else 1.01)
            
            # Position sizing (Simple: Fixed 100% equity, all-in)
            # REASON: Hou cash/positie consistent. Bij entry gaat cash naar 0, bij exit komt volledige positie-waarde terug.
            entry_notional = self.current_capital
            entry_fee = entry_notional * self.fee_pct
            capital_after_entry_fee = entry_notional - entry_fee
            if capital_after_entry_fee <= 0:
                return

            quantity = capital_after_entry_fee / entry_price

            # All-in: cash wordt 0 tijdens de trade
            self.current_capital = 0.0
            
            self.active_trade = Trade(
                entry_time=current_time,
                exit_time=None,
                direction=direction,
                entry_price=entry_price,
                exit_price=0.0,
                quantity=quantity,
                tp_price=tp_price,
                sl_price=sl_price,
                status='OPEN',
                fee=entry_fee,
                exit_reason="",
                p_up=signal.meta.get('p_up', 0.0) if signal.meta else 0.0,
                p_down=signal.meta.get('p_down', 0.0) if signal.meta else 0.0,
                regime=signal.meta.get('regime', "") if signal.meta else "",
                hypothesis=signal.meta.get('hypothesis', "") if signal.meta else "",
                entry_equity=entry_notional
            )
            # logger.info(f"OPEN {direction} at {entry_price:.2f} (TP: {tp_price:.2f}, SL: {sl_price:.2f})")

    def update_price(self, row: pd.Series):
        """
        Check of TP/SL geraakt zijn in deze candle.
        Deze methode moet aangeroepen worden VOOR process_signal voor de volgende candle.
        
        We nemen aan dat we checken op de High/Low van de candle die VOLGT op de entry.
        """
        if not self.active_trade:
            # Update equity curve (flat)
            self.equity_curve.append({'time': row['time_1'], 'equity': self.current_capital})
            return

        trade = self.active_trade
        current_time = row['time_1']
        
        # Check SL first (Conservative assumption: SL hit before TP in same candle)
        sl_hit = False
        tp_hit = False
        
        if trade.direction == 'LONG':
            if row['low'] <= trade.sl_price:
                sl_hit = True
                exit_price = trade.sl_price # We gaan uit van fill op SL prijs (optimistisch, in echt slippage)
            elif row['high'] >= trade.tp_price:
                tp_hit = True
                exit_price = trade.tp_price
        else: # SHORT
            if row['high'] >= trade.sl_price:
                sl_hit = True
                exit_price = trade.sl_price
            elif row['low'] <= trade.tp_price:
                tp_hit = True
                exit_price = trade.tp_price
                
        if sl_hit:
            # Slippage op stop loss
            slip_mult = (1 - self.slippage_pct) if trade.direction == 'LONG' else (1 + self.slippage_pct)
            exit_price = exit_price * slip_mult
            self._close_trade(trade, exit_price, current_time, "Stop Loss")
            self.active_trade = None
        elif tp_hit:
            self._close_trade(trade, exit_price, current_time, "Take Profit")
            self.active_trade = None
            
        # Update equity curve
        # REASON: Equity is totale accountwaarde, niet alleen unrealized PnL.
        # In dit model is cash tijdens een trade 0 (all-in). Daarom mark-to-marketten we:
        # invested(notional) + unrealized_pnl - geschatte exit fee.
        curr_price = row['close']
        if trade.direction == 'LONG':
            unrealized_pnl = (curr_price - trade.entry_price) * trade.quantity
        else:
            unrealized_pnl = (trade.entry_price - curr_price) * trade.quantity

        # Entry fee zit al in trade.fee zolang de trade open is (exit fee komt pas bij close)
        invested = (trade.entry_price * trade.quantity) + trade.fee  # ≈ oorspronkelijke notional (voor all-in sizing)
        est_exit_fee = (trade.quantity * curr_price) * self.fee_pct
        equity = invested + unrealized_pnl - est_exit_fee
        self.equity_curve.append({'time': current_time, 'equity': equity})

    def close_active_trade(self, row: pd.Series, reason: str = "End of Test"):
        """
        Force-close any open trade at the provided row close.
        REASON: Strategy Finder / backtests moeten altijd eindigen zonder open positie,
        anders worden return_pct (met MTM) en profit_factor (alleen gesloten trades) inconsistent.
        """
        if not self.active_trade:
            return
        self._close_trade(self.active_trade, float(row['close']), row['time_1'], reason)
        self.active_trade = None

    def _close_trade(self, trade: Trade, price: float, time: pd.Timestamp, reason: str):
        trade.exit_price = price
        trade.exit_time = time
        trade.status = 'CLOSED'
        trade.exit_reason = reason
        
        # Calculate PnL
        if trade.direction == 'LONG':
            gross_pnl = (price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - price) * trade.quantity
            
        # Exit fee
        exit_fee = (trade.quantity * price) * self.fee_pct
        trade.fee += exit_fee
        
        # All-in exit: cash = positie waarde - exit fee
        self.current_capital = (trade.quantity * price) - exit_fee
        
        # Consistent PnL calculation based on equity change
        trade.exit_equity = self.current_capital
        trade.pnl = trade.exit_equity - trade.entry_equity

        # PnL% t.o.v. entry equity
        trade.pnl_pct = (trade.pnl / trade.entry_equity) * 100 if trade.entry_equity > 0 else 0.0
        
        self.trades.append(trade)
        # logger.info(f"CLOSED {trade.direction} at {price:.2f} ({reason}). PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")

    def get_results(self) -> Dict:
        """Genereer statistieken."""
        if not self.trades and not self.active_trade:
            return {'total_trades': 0, 'return_pct': 0.0, 'initial_capital': self.initial_capital, 'final_capital': self.current_capital}
            
        total_pnl = sum(t.pnl for t in self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        
        # Max Drawdown
        equity_series = pd.Series([e['equity'] for e in self.equity_curve]) if self.equity_curve else pd.Series([self.initial_capital])
        running_max = equity_series.cummax()
        # REASON: Vermijd deling door 0 (kan gebeuren als equity ooit 0 wordt)
        running_max_safe = running_max.replace(0, np.nan)
        drawdown = (equity_series - running_max_safe) / running_max_safe
        max_dd = float(drawdown.min()) if not drawdown.isna().all() else 0.0

        # REASON: Als er nog een open trade is, neem de laatste mark-to-market equity als final_capital
        final_capital = self.current_capital
        if self.active_trade and self.equity_curve:
            final_capital = float(self.equity_curve[-1]['equity'])
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'return_pct': ((final_capital - self.initial_capital) / self.initial_capital) * 100,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd
        }
