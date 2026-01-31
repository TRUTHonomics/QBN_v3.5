-- ============================================================================
-- Migratie: indicators Archive Uitbreiding
-- Datum: 2025-12-13
-- Doel: Update archive trigger om alle nieuwe indicator kolommen mee te nemen
-- ============================================================================
--
-- INSTRUCTIES:
-- 1. Voer dit script uit in een maintenance window
-- 2. Dit script is non-breaking (update alleen functie)
-- 3. Na uitvoering worden nieuwe kolommen automatisch gearchiveerd
--
-- ============================================================================

-- ============================================================================
-- FASE 1: Update archive trigger functie
-- ============================================================================

-- REASON: Update archive_indicators_unified_cache_on_change() om alle kolommen mee te nemen
-- De volgende kolommen worden nu NIET gearchiveerd maar zitten wel in beide tabellen:
-- - ao_5_34
-- - supertrend_10_3
-- - supertrend_direction
-- - vpvr_poc, vpvr_vah, vpvr_val
-- - vpvr_hvn_upper, vpvr_hvn_lower

CREATE OR REPLACE FUNCTION kfl.archive_indicators_unified_cache_on_change()
RETURNS TRIGGER AS $$
BEGIN
    -- BEFORE UPDATE trigger: vergelijk OLD.time vs NEW.time
    -- Behoud altijd de meest recente data in de cache
    
    IF NEW.time > OLD.time THEN
        -- NEW is nieuwer: archiveer OLD naar indicators
        INSERT INTO kfl.indicators (
            asset_id, interval_min, time,
            open, high, low, close, volume,
            rsi_7, rsi_14, rsi_21,
            macd_12_26_9, macd_12_26_9_signal, macd_12_26_9_histogram, macd_12_26_9_fast, macd_12_26_9_slow,
            macd_6_13_4, macd_6_13_4_signal, macd_6_13_4_histogram, macd_6_13_4_fast, macd_6_13_4_slow,
            macd_20_50_15, macd_20_50_15_signal, macd_20_50_15_histogram, macd_20_50_15_fast, macd_20_50_15_slow,
            macd_8_24_9, macd_8_24_9_signal, macd_8_24_9_histogram, macd_8_24_9_fast, macd_8_24_9_slow,
            macd_5_35_5, macd_5_35_5_signal, macd_5_35_5_histogram, macd_5_35_5_fast, macd_5_35_5_slow,
            sma_20, sma_50, sma_100, sma_200,
            ema_10, ema_12, ema_20, ema_26, ema_50, ema_100, ema_200,
            dema_10, dema_20, dema_50, dema_100, dema_200,
            tenkan_9_26_52, kijun_9_26_52, senkou_a_9_26_52, senkou_b_9_26_52, chikou_9_26_52,
            tenkan_7_22_44, kijun_7_22_44, senkou_a_7_22_44, senkou_b_7_22_44, chikou_7_22_44,
            tenkan_6_20_52, kijun_6_20_52, senkou_a_6_20_52, senkou_b_6_20_52, chikou_6_20_52,
            tenkan_10_30_60, kijun_10_30_60, senkou_a_10_30_60, senkou_b_10_30_60, chikou_10_30_60,
            atr_14, atr_10, adx_14, dm_plus_14, dm_minus_14,
            stoch_k, stoch_d,
            bb_upper, bb_middle, bb_lower,
            cmf,
            keltner_upper, keltner_middle, keltner_lower,
            vwap_typical_price, vwap_close,
            obv, volume_flow,
            created_at, updated_at, source_script, atr_ma,
            -- NIEUWE KOLOMMEN
            ao_5_34,
            supertrend_10_3, supertrend_direction,
            vpvr_poc, vpvr_vah, vpvr_val,
            vpvr_hvn_upper, vpvr_hvn_lower
        )
        VALUES (
            OLD.asset_id, OLD.interval_min, OLD.time,
            OLD.open, OLD.high, OLD.low, OLD.close, OLD.volume,
            OLD.rsi_7, OLD.rsi_14, OLD.rsi_21,
            OLD.macd_12_26_9, OLD.macd_12_26_9_signal, OLD.macd_12_26_9_histogram, OLD.macd_12_26_9_fast, OLD.macd_12_26_9_slow,
            OLD.macd_6_13_4, OLD.macd_6_13_4_signal, OLD.macd_6_13_4_histogram, OLD.macd_6_13_4_fast, OLD.macd_6_13_4_slow,
            OLD.macd_20_50_15, OLD.macd_20_50_15_signal, OLD.macd_20_50_15_histogram, OLD.macd_20_50_15_fast, OLD.macd_20_50_15_slow,
            OLD.macd_8_24_9, OLD.macd_8_24_9_signal, OLD.macd_8_24_9_histogram, OLD.macd_8_24_9_fast, OLD.macd_8_24_9_slow,
            OLD.macd_5_35_5, OLD.macd_5_35_5_signal, OLD.macd_5_35_5_histogram, OLD.macd_5_35_5_fast, OLD.macd_5_35_5_slow,
            OLD.sma_20, OLD.sma_50, OLD.sma_100, OLD.sma_200,
            OLD.ema_10, OLD.ema_12, OLD.ema_20, OLD.ema_26, OLD.ema_50, OLD.ema_100, OLD.ema_200,
            OLD.dema_10, OLD.dema_20, OLD.dema_50, OLD.dema_100, OLD.dema_200,
            OLD.tenkan_9_26_52, OLD.kijun_9_26_52, OLD.senkou_a_9_26_52, OLD.senkou_b_9_26_52, OLD.chikou_9_26_52,
            OLD.tenkan_7_22_44, OLD.kijun_7_22_44, OLD.senkou_a_7_22_44, OLD.senkou_b_7_22_44, OLD.chikou_7_22_44,
            OLD.tenkan_6_20_52, OLD.kijun_6_20_52, OLD.senkou_a_6_20_52, OLD.senkou_b_6_20_52, OLD.chikou_6_20_52,
            OLD.tenkan_10_30_60, OLD.kijun_10_30_60, OLD.senkou_a_10_30_60, OLD.senkou_b_10_30_60, OLD.chikou_10_30_60,
            OLD.atr_14, OLD.atr_10, OLD.adx_14, OLD.dm_plus_14, OLD.dm_minus_14,
            OLD.stoch_k, OLD.stoch_d,
            OLD.bb_upper, OLD.bb_middle, OLD.bb_lower,
            OLD.cmf,
            OLD.keltner_upper, OLD.keltner_middle, OLD.keltner_lower,
            OLD.vwap_typical_price, OLD.vwap_close,
            OLD.obv, OLD.volume_flow,
            OLD.created_at, OLD.updated_at, OLD.source_script, OLD.atr_ma,
            -- NIEUWE KOLOMMEN
            OLD.ao_5_34,
            OLD.supertrend_10_3, OLD.supertrend_direction,
            OLD.vpvr_poc, OLD.vpvr_vah, OLD.vpvr_val,
            OLD.vpvr_hvn_upper, OLD.vpvr_hvn_lower
        )
        ON CONFLICT (asset_id, interval_min, time) DO UPDATE SET
            updated_at = NOW();
        
        RETURN NEW;
        
    ELSIF NEW.time < OLD.time THEN
        -- NEW is ouder: archiveer NEW naar indicators, behoud OLD in cache
        INSERT INTO kfl.indicators (
            asset_id, interval_min, time,
            open, high, low, close, volume,
            rsi_7, rsi_14, rsi_21,
            macd_12_26_9, macd_12_26_9_signal, macd_12_26_9_histogram, macd_12_26_9_fast, macd_12_26_9_slow,
            macd_6_13_4, macd_6_13_4_signal, macd_6_13_4_histogram, macd_6_13_4_fast, macd_6_13_4_slow,
            macd_20_50_15, macd_20_50_15_signal, macd_20_50_15_histogram, macd_20_50_15_fast, macd_20_50_15_slow,
            macd_8_24_9, macd_8_24_9_signal, macd_8_24_9_histogram, macd_8_24_9_fast, macd_8_24_9_slow,
            macd_5_35_5, macd_5_35_5_signal, macd_5_35_5_histogram, macd_5_35_5_fast, macd_5_35_5_slow,
            sma_20, sma_50, sma_100, sma_200,
            ema_10, ema_12, ema_20, ema_26, ema_50, ema_100, ema_200,
            dema_10, dema_20, dema_50, dema_100, dema_200,
            tenkan_9_26_52, kijun_9_26_52, senkou_a_9_26_52, senkou_b_9_26_52, chikou_9_26_52,
            tenkan_7_22_44, kijun_7_22_44, senkou_a_7_22_44, senkou_b_7_22_44, chikou_7_22_44,
            tenkan_6_20_52, kijun_6_20_52, senkou_a_6_20_52, senkou_b_6_20_52, chikou_6_20_52,
            tenkan_10_30_60, kijun_10_30_60, senkou_a_10_30_60, senkou_b_10_30_60, chikou_10_30_60,
            atr_14, atr_10, adx_14, dm_plus_14, dm_minus_14,
            stoch_k, stoch_d,
            bb_upper, bb_middle, bb_lower,
            cmf,
            keltner_upper, keltner_middle, keltner_lower,
            vwap_typical_price, vwap_close,
            obv, volume_flow,
            created_at, updated_at, source_script, atr_ma,
            -- NIEUWE KOLOMMEN
            ao_5_34,
            supertrend_10_3, supertrend_direction,
            vpvr_poc, vpvr_vah, vpvr_val,
            vpvr_hvn_upper, vpvr_hvn_lower
        )
        VALUES (
            NEW.asset_id, NEW.interval_min, NEW.time,
            NEW.open, NEW.high, NEW.low, NEW.close, NEW.volume,
            NEW.rsi_7, NEW.rsi_14, NEW.rsi_21,
            NEW.macd_12_26_9, NEW.macd_12_26_9_signal, NEW.macd_12_26_9_histogram, NEW.macd_12_26_9_fast, NEW.macd_12_26_9_slow,
            NEW.macd_6_13_4, NEW.macd_6_13_4_signal, NEW.macd_6_13_4_histogram, NEW.macd_6_13_4_fast, NEW.macd_6_13_4_slow,
            NEW.macd_20_50_15, NEW.macd_20_50_15_signal, NEW.macd_20_50_15_histogram, NEW.macd_20_50_15_fast, NEW.macd_20_50_15_slow,
            NEW.macd_8_24_9, NEW.macd_8_24_9_signal, NEW.macd_8_24_9_histogram, NEW.macd_8_24_9_fast, NEW.macd_8_24_9_slow,
            NEW.macd_5_35_5, NEW.macd_5_35_5_signal, NEW.macd_5_35_5_histogram, NEW.macd_5_35_5_fast, NEW.macd_5_35_5_slow,
            NEW.sma_20, NEW.sma_50, NEW.sma_100, NEW.sma_200,
            NEW.ema_10, NEW.ema_12, NEW.ema_20, NEW.ema_26, NEW.ema_50, NEW.ema_100, NEW.ema_200,
            NEW.dema_10, NEW.dema_20, NEW.dema_50, NEW.dema_100, NEW.dema_200,
            NEW.tenkan_9_26_52, NEW.kijun_9_26_52, NEW.senkou_a_9_26_52, NEW.senkou_b_9_26_52, NEW.chikou_9_26_52,
            NEW.tenkan_7_22_44, NEW.kijun_7_22_44, NEW.senkou_a_7_22_44, NEW.senkou_b_7_22_44, NEW.chikou_7_22_44,
            NEW.tenkan_6_20_52, NEW.kijun_6_20_52, NEW.senkou_a_6_20_52, NEW.senkou_b_6_20_52, NEW.chikou_6_20_52,
            NEW.tenkan_10_30_60, NEW.kijun_10_30_60, NEW.senkou_a_10_30_60, NEW.senkou_b_10_30_60, NEW.chikou_10_30_60,
            NEW.atr_14, NEW.atr_10, NEW.adx_14, NEW.dm_plus_14, NEW.dm_minus_14,
            NEW.stoch_k, NEW.stoch_d,
            NEW.bb_upper, NEW.bb_middle, NEW.bb_lower,
            NEW.cmf,
            NEW.keltner_upper, NEW.keltner_middle, NEW.keltner_lower,
            NEW.vwap_typical_price, NEW.vwap_close,
            NEW.obv, NEW.volume_flow,
            NEW.created_at, NEW.updated_at, NEW.source_script, NEW.atr_ma,
            -- NIEUWE KOLOMMEN
            NEW.ao_5_34,
            NEW.supertrend_10_3, NEW.supertrend_direction,
            NEW.vpvr_poc, NEW.vpvr_vah, NEW.vpvr_val,
            NEW.vpvr_hvn_upper, NEW.vpvr_hvn_lower
        )
        ON CONFLICT (asset_id, interval_min, time) DO UPDATE SET
            updated_at = NOW();
        
        -- Herstel OLD waarden om nieuwste data in cache te behouden
        NEW.time := OLD.time;
        NEW.open := OLD.open;
        NEW.high := OLD.high;
        NEW.low := OLD.low;
        NEW.close := OLD.close;
        NEW.volume := OLD.volume;
        NEW.rsi_7 := OLD.rsi_7;
        NEW.rsi_14 := OLD.rsi_14;
        NEW.rsi_21 := OLD.rsi_21;
        NEW.updated_at := OLD.updated_at;
        
        RETURN NEW;
    END IF;
    
    -- NEW.time = OLD.time: geen archivering nodig
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VERIFICATIE
-- ============================================================================

-- Check dat de functie is geüpdatet met nieuwe kolommen
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'kfl'
          AND p.proname = 'archive_indicators_unified_cache_on_change'
          AND pg_get_functiondef(p.oid) LIKE '%ao_5_34%'
          AND pg_get_functiondef(p.oid) LIKE '%supertrend_10_3%'
          AND pg_get_functiondef(p.oid) LIKE '%vpvr_poc%'
    ) THEN
        RAISE EXCEPTION 'FOUT: archive_indicators_unified_cache_on_change() functie is niet geüpdatet met nieuwe kolommen';
    END IF;
    
    RAISE NOTICE 'OK: archive_indicators_unified_cache_on_change() functie is geüpdatet met alle kolommen';
END $$;

-- Toon overzicht van kolommen die nu worden gearchiveerd
SELECT 
    'indicators_unified_cache' as tabel,
    COUNT(*) as totaal_kolommen,
    COUNT(*) FILTER (WHERE column_name IN (
        'ao_5_34', 'supertrend_10_3', 'supertrend_direction',
        'vpvr_poc', 'vpvr_vah', 'vpvr_val', 'vpvr_hvn_upper', 'vpvr_hvn_lower'
    )) as nieuwe_kolommen
FROM information_schema.columns
WHERE table_schema = 'kfl' AND table_name = 'indicators_unified_cache'
UNION ALL
SELECT 
    'indicators' as tabel,
    COUNT(*) as totaal_kolommen,
    COUNT(*) FILTER (WHERE column_name IN (
        'ao_5_34', 'supertrend_10_3', 'supertrend_direction',
        'vpvr_poc', 'vpvr_vah', 'vpvr_val', 'vpvr_hvn_upper', 'vpvr_hvn_lower'
    )) as nieuwe_kolommen
FROM information_schema.columns
WHERE table_schema = 'kfl' AND table_name = 'indicators';

