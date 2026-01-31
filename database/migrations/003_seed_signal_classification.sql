-- ============================================================================
-- Migration: 003_seed_signal_classification.sql
-- Purpose: Seed signal_classification table with all 125 signals
-- Date: 2025-12-10
-- Priority: HIGH
-- Description: Populate semantic classifications for QBN_v2 Bayesian Network
-- ============================================================================

-- Idempotent design: Clear existing data then insert
BEGIN;

TRUNCATE TABLE qbn.signal_classification;

-- ============================================================================
-- LEADING SIGNALS (48 total)
-- Early momentum signals that detect potential movements before they fully develop
-- ============================================================================

INSERT INTO qbn.signal_classification (signal_name, semantic_class, indicator_base, indicator_variant, polarity, description) VALUES

-- RSI Extremes and Divergences (6)
('RSI_OVERSOLD', 'LEADING', 'RSI', '14', 'bullish', 'RSI onder 30 - oversold conditie'),
('RSI_OVERBOUGHT', 'LEADING', 'RSI', '14', 'bearish', 'RSI boven 70 - overbought conditie'),
('RSI_EXTREME_OVERSOLD', 'LEADING', 'RSI', '14', 'bullish', 'RSI onder 20 - extreme oversold'),
('RSI_EXTREME_OVERBOUGHT', 'LEADING', 'RSI', '14', 'bearish', 'RSI boven 80 - extreme overbought'),
('RSI_DIVERGENCE_BULLISH', 'LEADING', 'RSI', '14', 'bullish', 'Bullish divergence met prijs'),
('RSI_DIVERGENCE_BEARISH', 'LEADING', 'RSI', '14', 'bearish', 'Bearish divergence met prijs'),

-- Fast MACD Variants (4) - NIEUW PvA 1.2
('MACD_6_13_4_BULLISH_CROSS', 'LEADING', 'MACD', '6_13_4', 'bullish', 'Fast MACD bullish cross'),
('MACD_6_13_4_BEARISH_CROSS', 'LEADING', 'MACD', '6_13_4', 'bearish', 'Fast MACD bearish cross'),
('MACD_8_24_9_BULLISH_CROSS', 'LEADING', 'MACD', '8_24_9', 'bullish', 'Medium-fast MACD bullish cross'),
('MACD_8_24_9_BEARISH_CROSS', 'LEADING', 'MACD', '8_24_9', 'bearish', 'Medium-fast MACD bearish cross'),

-- MACD Divergences (2)
('MACD_DIVERGENCE_BULLISH', 'LEADING', 'MACD', '12_26_9', 'bullish', 'MACD bullish divergence - oversold reversal'),
('MACD_DIVERGENCE_BEARISH', 'LEADING', 'MACD', '12_26_9', 'bearish', 'MACD bearish divergence - overbought reversal'),

-- Fast Ichimoku Variants (4) - NIEUW PvA 1.2
('ICHI_6_20_52_TK_CROSS_BULL', 'LEADING', 'ICHIMOKU', '6_20_52', 'bullish', 'Intraday TK cross bullish'),
('ICHI_6_20_52_TK_CROSS_BEAR', 'LEADING', 'ICHIMOKU', '6_20_52', 'bearish', 'Intraday TK cross bearish'),
('ICHI_6_20_52_KUMO_BREAKOUT_LONG', 'LEADING', 'ICHIMOKU', '6_20_52', 'bullish', 'Intraday kumo breakout long'),
('ICHI_6_20_52_KUMO_BREAKOUT_SHORT', 'LEADING', 'ICHIMOKU', '6_20_52', 'bearish', 'Intraday kumo breakout short'),

-- Ichimoku Kumo Twist (2)
('ICHIMOKU_KUMO_TWIST_BULL', 'LEADING', 'ICHIMOKU', '9_26_52', 'bullish', 'Kumo twist naar bullish'),
('ICHIMOKU_KUMO_TWIST_BEAR', 'LEADING', 'ICHIMOKU', '9_26_52', 'bearish', 'Kumo twist naar bearish'),

-- Stochastic Extremes and Divergences (6)
('STOCH_OVERSOLD', 'LEADING', 'STOCH', '14_3_3', 'bullish', 'Stochastic onder 20'),
('STOCH_OVERBOUGHT', 'LEADING', 'STOCH', '14_3_3', 'bearish', 'Stochastic boven 80'),
('STOCH_DIVERGENCE_BULL', 'LEADING', 'STOCH', '14_3_3', 'bullish', 'Bullish stochastic divergence'),
('STOCH_DIVERGENCE_BEAR', 'LEADING', 'STOCH', '14_3_3', 'bearish', 'Bearish stochastic divergence'),
('STOCH_HIDDEN_DIVERGENCE_BULL', 'LEADING', 'STOCH', '14_3_3', 'bullish', 'Hidden bullish divergence'),
('STOCH_HIDDEN_DIVERGENCE_BEAR', 'LEADING', 'STOCH', '14_3_3', 'bearish', 'Hidden bearish divergence'),

-- Keltner Channel Mean Reversion and Dynamic Levels (7)
('KC_MEAN_REVERSION_LONG', 'LEADING', 'KC', '20', 'bullish', 'Keltner mean reversion long'),
('KC_MEAN_REVERSION_SHORT', 'LEADING', 'KC', '20', 'bearish', 'Keltner mean reversion short'),
('KC_SQUEEZE', 'LEADING', 'KC', '20', 'neutral', 'Keltner squeeze - breakout verwacht'),
('KC_DYNAMIC_SUPPORT', 'LEADING', 'KC', '20', 'bullish', 'Prijs vindt support op KC lower band'),
('KC_DYNAMIC_RESISTANCE', 'LEADING', 'KC', '20', 'bearish', 'Prijs vindt resistance op KC upper band'),
('KC_PULLBACK_LONG', 'LEADING', 'KC', '20', 'bullish', 'Pullback naar KC mid band in uptrend'),
('KC_PULLBACK_SHORT', 'LEADING', 'KC', '20', 'bearish', 'Pullback naar KC mid band in downtrend'),

-- Bollinger Bands Mean Reversion (2)
('BB_SQUEEZE', 'LEADING', 'BB', '20', 'neutral', 'Bollinger squeeze - breakout verwacht'),
('BB_MEAN_REVERSION_LONG', 'LEADING', 'BB', '20', 'bullish', 'BB mean reversion long'),

-- OBV Divergences (2)
('OBV_BULLISH_DIVERGENCE', 'LEADING', 'OBV', NULL, 'bullish', 'OBV bullish divergence'),
('OBV_BEARISH_DIVERGENCE', 'LEADING', 'OBV', NULL, 'bearish', 'OBV bearish divergence'),

-- CMF Divergences (2)
('CMF_DIVERGENCE_BULLISH', 'LEADING', 'CMF', '20', 'bullish', 'CMF bullish divergence'),
('CMF_DIVERGENCE_BEARISH', 'LEADING', 'CMF', '20', 'bearish', 'CMF bearish divergence'),

-- Awesome Oscillator Patterns (4)
('AO_TWIN_PEAKS_BULLISH', 'LEADING', 'AO', '5_34', 'bullish', 'AO twin peaks bullish'),
('AO_TWIN_PEAKS_BEARISH', 'LEADING', 'AO', '5_34', 'bearish', 'AO twin peaks bearish'),
('AO_SAUCER_BULLISH', 'LEADING', 'AO', '5_34', 'bullish', 'AO saucer pattern bullish'),
('AO_SAUCER_BEARISH', 'LEADING', 'AO', '5_34', 'bearish', 'AO saucer pattern bearish'),

-- ADX Trend Exhaustion (2)
('ADX_TREND_EXHAUSTION', 'LEADING', 'ADX', '14', 'neutral', 'ADX daalt vanaf hoge waarde'),
('ADX_PEAK_REVERSAL', 'LEADING', 'ADX', '14', 'neutral', 'ADX piek reversal'),

-- Composite Mean Reversion (2)
('MEAN_REVERSION_SETUP_LONG', 'LEADING', 'COMPOSITE', NULL, 'bullish', 'Mean reversion setup met oversold condities'),
('MEAN_REVERSION_SETUP_SHORT', 'LEADING', 'COMPOSITE', NULL, 'bearish', 'Mean reversion setup met overbought condities'),

-- VPVR Support/Resistance (3)
('VPVR_HVN_RESISTANCE', 'LEADING', 'VPVR', '20', 'bearish', 'Prijs test HVN resistance'),
('VPVR_HVN_SUPPORT', 'LEADING', 'VPVR', '20', 'bullish', 'Prijs test HVN support'),
('VPVR_POC_TOUCH', 'LEADING', 'VPVR', '20', 'neutral', 'Prijs raakt Point of Control');

-- ============================================================================
-- COINCIDENT SIGNALS (39 total)
-- Real-time signals that confirm current price action and show momentum
-- ============================================================================

INSERT INTO qbn.signal_classification (signal_name, semantic_class, indicator_base, indicator_variant, polarity, description) VALUES

-- Standard MACD (12_26_9) (4)
('MACD_BULLISH_CROSS', 'COINCIDENT', 'MACD', '12_26_9', 'bullish', 'Standard MACD bullish cross'),
('MACD_BEARISH_CROSS', 'COINCIDENT', 'MACD', '12_26_9', 'bearish', 'Standard MACD bearish cross'),
('MACD_HISTOGRAM_POSITIVE', 'COINCIDENT', 'MACD', '12_26_9', 'bullish', 'MACD histogram positief'),
('MACD_HISTOGRAM_NEGATIVE', 'COINCIDENT', 'MACD', '12_26_9', 'bearish', 'MACD histogram negatief'),

-- Standard Ichimoku (9_26_52) (5)
('ICHIMOKU_TENKAN_KIJUN_CROSS_BULL', 'COINCIDENT', 'ICHIMOKU', '9_26_52', 'bullish', 'TK cross bullish'),
('ICHIMOKU_TENKAN_KIJUN_CROSS_BEAR', 'COINCIDENT', 'ICHIMOKU', '9_26_52', 'bearish', 'TK cross bearish'),
('ICHIMOKU_PRICE_ABOVE_KUMO', 'COINCIDENT', 'ICHIMOKU', '9_26_52', 'bullish', 'Prijs boven kumo'),
('ICHIMOKU_PRICE_BELOW_KUMO', 'COINCIDENT', 'ICHIMOKU', '9_26_52', 'bearish', 'Prijs onder kumo'),
('ICHIMOKU_PRICE_IN_KUMO', 'COINCIDENT', 'ICHIMOKU', '9_26_52', 'neutral', 'Prijs in kumo'),

-- Medium Ichimoku (7_22_44) (2) - NIEUW PvA 1.2
('ICHI_7_22_44_TK_CROSS_BULL', 'COINCIDENT', 'ICHIMOKU', '7_22_44', 'bullish', 'Crypto TK cross bullish'),
('ICHI_7_22_44_TK_CROSS_BEAR', 'COINCIDENT', 'ICHIMOKU', '7_22_44', 'bearish', 'Crypto TK cross bearish'),

-- Bollinger Bands Breakouts (2)
('BB_BREAKOUT_LONG', 'COINCIDENT', 'BB', '20', 'bullish', 'BB breakout naar boven'),
('BB_BREAKOUT_SHORT', 'COINCIDENT', 'BB', '20', 'bearish', 'BB breakout naar beneden'),

-- CMF Flow (4)
('CMF_BULLISH_BIAS', 'COINCIDENT', 'CMF', '20', 'bullish', 'CMF positief - koopdruk'),
('CMF_BEARISH_BIAS', 'COINCIDENT', 'CMF', '20', 'bearish', 'CMF negatief - verkoopdruk'),
('CMF_STRONG_BUYING', 'COINCIDENT', 'CMF', '20', 'bullish', 'CMF sterk positief'),
('CMF_STRONG_SELLING', 'COINCIDENT', 'CMF', '20', 'bearish', 'CMF sterk negatief'),

-- RSI Center Cross (2)
('RSI_CENTER_CROSS_BULL', 'COINCIDENT', 'RSI', '14', 'bullish', 'RSI kruist boven 50'),
('RSI_CENTER_CROSS_BEAR', 'COINCIDENT', 'RSI', '14', 'bearish', 'RSI kruist onder 50'),

-- Stochastic Cross (1)
('STOCH_BULLISH_CROSS', 'COINCIDENT', 'STOCH', '14_3_3', 'bullish', 'Stoch K kruist boven D'),

-- ATR Volatility (2)
('ATR_HIGH_VOLATILITY', 'COINCIDENT', 'ATR', '14', 'neutral', 'Hoge volatiliteit'),
('ATR_LOW_VOLATILITY', 'COINCIDENT', 'ATR', '14', 'neutral', 'Lage volatiliteit'),

-- Awesome Oscillator Zero Cross (2)
('AO_BULLISH_ZERO_CROSS', 'COINCIDENT', 'AO', '5_34', 'bullish', 'AO kruist boven nul'),
('AO_BEARISH_ZERO_CROSS', 'COINCIDENT', 'AO', '5_34', 'bearish', 'AO kruist onder nul'),

-- Directional Indicators Cross (2)
('DI_BULLISH_CROSS', 'COINCIDENT', 'DI', '14', 'bullish', '+DI kruist boven -DI'),
('DI_BEARISH_CROSS', 'COINCIDENT', 'DI', '14', 'bearish', '-DI kruist boven +DI'),

-- Supertrend Flips (2)
('SUPER_TREND_FLIP_BULL', 'COINCIDENT', 'SUPERTREND', '10_3', 'bullish', 'Supertrend flipt bullish'),
('SUPER_TREND_FLIP_BEAR', 'COINCIDENT', 'SUPERTREND', '10_3', 'bearish', 'Supertrend flipt bearish'),

-- Volatility Breakouts (2)
('VOLATILITY_BREAKOUT_LONG', 'COINCIDENT', 'COMPOSITE', NULL, 'bullish', 'Volatility breakout long'),
('VOLATILITY_BREAKOUT_SHORT', 'COINCIDENT', 'COMPOSITE', NULL, 'bearish', 'Volatility breakout short'),

-- Momentum Confluence (2)
('MOMENTUM_BULLISH_CONFLUENCE', 'COINCIDENT', 'COMPOSITE', NULL, 'bullish', 'Momentum indicators bullish confluence'),
('MOMENTUM_BEARISH_CONFLUENCE', 'COINCIDENT', 'COMPOSITE', NULL, 'bearish', 'Momentum indicators bearish confluence'),

-- Volume Confluence (2)
('VOLUME_BULLISH_CONFLUENCE', 'COINCIDENT', 'COMPOSITE', NULL, 'bullish', 'Volume indicators bullish confluence'),
('VOLUME_BEARISH_CONFLUENCE', 'COINCIDENT', 'COMPOSITE', NULL, 'bearish', 'Volume indicators bearish confluence'),

-- Risk Management (2)
('RISK_HIGH_VOLATILITY', 'COINCIDENT', 'COMPOSITE', NULL, 'neutral', 'Extreem hoge volatiliteit'),
('RISK_LOW_LIQUIDITY', 'COINCIDENT', 'COMPOSITE', NULL, 'neutral', 'Lage liquiditeit met hoge volatiliteit'),

-- Regime Volatile (1)
('REGIME_VOLATILE', 'COINCIDENT', 'COMPOSITE', NULL, 'neutral', 'Volatile maar niet trending regime'),

-- VPVR Breakouts (2)
('VPVR_LVN_BREAKOUT_UP', 'COINCIDENT', 'VPVR', '20', 'bullish', 'Prijs breekt door LVN naar boven'),
('VPVR_LVN_BREAKOUT_DOWN', 'COINCIDENT', 'VPVR', '20', 'bearish', 'Prijs breekt door LVN naar beneden');

-- ============================================================================
-- CONFIRMING SIGNALS (38 total)
-- Late confirmation signals for trend validation and strong confirmation
-- ============================================================================

INSERT INTO qbn.signal_classification (signal_name, semantic_class, indicator_base, indicator_variant, polarity, description) VALUES

-- ADX Trend Strength (4)
('ADX_TREND_CONFIRM', 'CONFIRMING', 'ADX', '14', 'neutral', 'ADX > 25 - trend bevestigd'),
('ADX_STRONG_TREND', 'CONFIRMING', 'ADX', '14', 'neutral', 'ADX > 30 - sterke trend'),
('ADX_NON_TRENDING_REGIME', 'CONFIRMING', 'ADX', '14', 'neutral', 'ADX < 20 - ranging'),
('ADX_WEAK_TREND', 'CONFIRMING', 'ADX', '14', 'neutral', 'ADX < 15 - zeer zwak'),

-- Slow MACD Variants (4) - NIEUW PvA 1.2
('MACD_5_35_5_BULLISH_CROSS', 'CONFIRMING', 'MACD', '5_35_5', 'bullish', 'Ultra-slow MACD bullish'),
('MACD_5_35_5_BEARISH_CROSS', 'CONFIRMING', 'MACD', '5_35_5', 'bearish', 'Ultra-slow MACD bearish'),
('MACD_20_50_15_BULLISH_CROSS', 'CONFIRMING', 'MACD', '20_50_15', 'bullish', 'Swing MACD bullish'),
('MACD_20_50_15_BEARISH_CROSS', 'CONFIRMING', 'MACD', '20_50_15', 'bearish', 'Swing MACD bearish'),

-- MACD Zero Line (2)
('MACD_ZERO_LINE_CROSS_BULL', 'CONFIRMING', 'MACD', '12_26_9', 'bullish', 'MACD kruist boven nullijn'),
('MACD_ZERO_LINE_CROSS_BEAR', 'CONFIRMING', 'MACD', '12_26_9', 'bearish', 'MACD kruist onder nullijn'),

-- Slow Ichimoku (10_30_60) (4) - NIEUW PvA 1.2
('ICHI_10_30_60_TK_CROSS_BULL', 'CONFIRMING', 'ICHIMOKU', '10_30_60', 'bullish', 'Swing TK cross bullish'),
('ICHI_10_30_60_TK_CROSS_BEAR', 'CONFIRMING', 'ICHIMOKU', '10_30_60', 'bearish', 'Swing TK cross bearish'),
('ICHI_10_30_60_KUMO_BREAKOUT_LONG', 'CONFIRMING', 'ICHIMOKU', '10_30_60', 'bullish', 'Swing kumo breakout long'),
('ICHI_10_30_60_KUMO_BREAKOUT_SHORT', 'CONFIRMING', 'ICHIMOKU', '10_30_60', 'bearish', 'Swing kumo breakout short'),

-- Standard Ichimoku Kumo Breakouts (2)
('ICHIMOKU_KUMO_BREAKOUT_LONG', 'CONFIRMING', 'ICHIMOKU', '9_26_52', 'bullish', 'Kumo breakout long'),
('ICHIMOKU_KUMO_BREAKOUT_SHORT', 'CONFIRMING', 'ICHIMOKU', '9_26_52', 'bearish', 'Kumo breakout short'),

-- Keltner Channel Trend Breakouts (2)
('KC_TREND_BREAKOUT_LONG', 'CONFIRMING', 'KC', '20', 'bullish', 'KC trend breakout long'),
('KC_TREND_BREAKOUT_SHORT', 'CONFIRMING', 'KC', '20', 'bearish', 'KC trend breakout short'),

-- OBV Trend Confirmation (4)
('OBV_TREND_CONFIRM_BULL', 'CONFIRMING', 'OBV', NULL, 'bullish', 'OBV bevestigt bullish'),
('OBV_TREND_CONFIRM_BEAR', 'CONFIRMING', 'OBV', NULL, 'bearish', 'OBV bevestigt bearish'),
('OBV_TREND_STRENGTH_BULL', 'CONFIRMING', 'OBV', NULL, 'bullish', 'OBV sterke bullish trend'),
('OBV_TREND_STRENGTH_BEAR', 'CONFIRMING', 'OBV', NULL, 'bearish', 'OBV sterke bearish trend'),

-- Directional Indicators Strong (2)
('DI_STRONG_BULLISH', 'CONFIRMING', 'DI', '14', 'bullish', '+DI > 25 en > -DI'),
('DI_STRONG_BEARISH', 'CONFIRMING', 'DI', '14', 'bearish', '-DI > 25 en > +DI'),

-- Supertrend Trend (2)
('SUPER_TREND_BULLISH', 'CONFIRMING', 'SUPERTREND', '10_3', 'bullish', 'Prijs boven supertrend'),
('SUPER_TREND_BEARISH', 'CONFIRMING', 'SUPERTREND', '10_3', 'bearish', 'Prijs onder supertrend'),

-- RSI Center Sustained (2)
('RSI_CENTER_BULLISH', 'CONFIRMING', 'RSI', '14', 'bullish', 'RSI sustained > 50'),
('RSI_CENTER_BEARISH', 'CONFIRMING', 'RSI', '14', 'bearish', 'RSI sustained < 50'),

-- Composite Confluence Signals (4)
('BULLISH_CONFLUENCE_STRONG', 'CONFIRMING', 'COMPOSITE', NULL, 'bullish', 'Sterke bullish confluence'),
('BEARISH_CONFLUENCE_STRONG', 'CONFIRMING', 'COMPOSITE', NULL, 'bearish', 'Sterke bearish confluence'),
('TREND_FOLLOWING_BREAKOUT_LONG', 'CONFIRMING', 'COMPOSITE', NULL, 'bullish', 'Trend following breakout long'),
('TREND_FOLLOWING_BREAKOUT_SHORT', 'CONFIRMING', 'COMPOSITE', NULL, 'bearish', 'Trend following breakout short'),

-- Multi-Timeframe Alignment (2)
('MTF_BULLISH_ALIGNMENT', 'CONFIRMING', 'COMPOSITE', NULL, 'bullish', 'Bullish alignment over meerdere timeframes'),
('MTF_BEARISH_ALIGNMENT', 'CONFIRMING', 'COMPOSITE', NULL, 'bearish', 'Bearish alignment over meerdere timeframes'),

-- Market Regime (3)
('REGIME_TRENDING_BULLISH', 'CONFIRMING', 'COMPOSITE', NULL, 'bullish', 'Bullish trending regime'),
('REGIME_TRENDING_BEARISH', 'CONFIRMING', 'COMPOSITE', NULL, 'bearish', 'Bearish trending regime'),
('REGIME_RANGING', 'CONFIRMING', 'COMPOSITE', NULL, 'neutral', 'Ranging regime'),

-- VPVR Value Area (1)
('VPVR_VALUE_AREA_INSIDE', 'CONFIRMING', 'VPVR', '20', 'neutral', 'Prijs binnen Value Area');

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Count by semantic class
SELECT semantic_class, COUNT(*) as count
FROM qbn.signal_classification
GROUP BY semantic_class
ORDER BY semantic_class;

-- Expected output:
--  semantic_class | count
-- ----------------+-------
--  CONFIRMING     | 38
--  COINCIDENT     | 39
--  LEADING        | 48
-- Total: 125

-- Verify no duplicates
SELECT signal_name, COUNT(*)
FROM qbn.signal_classification
GROUP BY signal_name
HAVING COUNT(*) > 1;

-- Expected: 0 rows (no duplicates)

-- Count by indicator base
SELECT indicator_base, COUNT(*) as count
FROM qbn.signal_classification
GROUP BY indicator_base
ORDER BY count DESC, indicator_base;

-- Sample query to verify new variants
SELECT signal_name, semantic_class, indicator_variant
FROM qbn.signal_classification
WHERE indicator_base IN ('MACD', 'ICHIMOKU')
  AND indicator_variant NOT IN ('12_26_9', '9_26_52')
ORDER BY indicator_base, indicator_variant;

-- Expected: 18 rows (8 MACD + 10 Ichimoku variants)
