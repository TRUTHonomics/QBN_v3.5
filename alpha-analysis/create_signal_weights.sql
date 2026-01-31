-- SQL migratie voor qbn.signal_weights tabel
-- Versie: 2.0 (Horizon-specifiek)

CREATE TABLE IF NOT EXISTS qbn.signal_weights (
    signal_name VARCHAR(100) NOT NULL,
    horizon VARCHAR(5) NOT NULL,  -- '1h', '4h', '1d'
    weight REAL DEFAULT 1.0,
    mutual_information REAL,
    hit_rate REAL,
    stability_score REAL,
    oos_performance REAL,
    last_trained_at TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(10) DEFAULT '2.0',
    PRIMARY KEY (signal_name, horizon)
);

COMMENT ON TABLE qbn.signal_weights IS 'Opslag voor data-driven signaal gewichten berekend via alfa-analyse.';
COMMENT ON COLUMN qbn.signal_weights.weight IS 'Berekende alfa-score (voorspellingskracht), genormaliseerd en geclipt.';
COMMENT ON COLUMN qbn.signal_weights.stability_score IS 'Symmetrie-score tussen odd en even maanden (1.0 = perfect stabiel).';
COMMENT ON COLUMN qbn.signal_weights.oos_performance IS 'Performance (MI of Hit Rate) op de out-of-sample set (2025).';
