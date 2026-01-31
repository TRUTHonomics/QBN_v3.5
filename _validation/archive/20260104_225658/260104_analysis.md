# QBN v2 Node Analyse - 4 januari 2026

**Asset:** 1 (BTCUSDT)  
**Dataset:** 3.039.643 MTF signals, 116.126 training observations  
**Validatieperiode:** 2025-12-05 tot 2026-01-02 (672 inferences)

---

## Executive Summary

| Laag | Node | Stability | Coverage | Verdict |
|------|------|-----------|----------|---------|
| Structural | HTF_Regime | 0.40 🔴 | 100% | Instabiel fundament |
| Tactical | Leading_Composite | 0.89 ✅ | 100% | Gezond |
| Tactical | Coincident_Composite | 0.89 ✅ | 100% | Gezond |
| Tactical | Confirming_Composite | 0.84 ✅ | 100% | Gezond |
| Entry | Entry_Timing | 0.88 ✅ | 100% | Goede aggregatie, lage entropy |
| Prediction | Prediction_1h/4h/1d | 0.00 🔴 | 100% | **Kritiek: 100% Neutral output** |

**Hoofdprobleem:** Het netwerk zit vast in een "Neutral Trap" — alle voorspellingen convergeren naar de neutrale state door gebrek aan gedifferentieerd bewijs.

---

## 1. Structural Layer: `HTF_Regime`

### 1.1 Observaties
| Metric | Waarde | Target | Status |
|--------|--------|--------|--------|
| Stability | 0.40 | > 0.70 | 🔴 Fail |
| Entropy | 3.2 | 1.5-2.5 | ⚠️ Te hoog |
| Info Gain | 0.00 | > 0.10 | 🔴 Geen voorspellende waarde |
| Semantic Score | 1.00 | 1.00 | ✅ OK |

### 1.2 Root Cause
- **11 states** verdeeld over slechts ~116k observaties = gemiddeld ~10.5k per state.
- ADX-gebaseerde regime-detectie is te granulaar voor de beschikbare data.
- Entropy van 3.2 (max ~3.46 voor 11 states) wijst op near-uniform verdeling → geen informatieve prior.

### 1.3 Tuning Parameters
| Parameter | Huidige Waarde | Voorstel | Locatie |
|-----------|----------------|----------|---------|
| `num_states` | 11 | **5-7** | `inference/regime_detector.py` |
| ADX threshold `strong_trend` | 40.0 | 35.0 | `qbn.signal_discretization` |
| ADX threshold `non_trending` | 25.0 | 20.0 | `qbn.signal_discretization` |
| State collapsing | - | Merge "Emerging" + "Retracing" | `HTFRegimeDetector.detect()` |

---

## 2. Tactical Layer: `Composites`

### 2.1 Observaties (Alle 3 nodes)

| Node | Stability | Entropy | Info Gain |
|------|-----------|---------|-----------|
| Leading_Composite | 0.89 | 0.1 | 2.24 |
| Coincident_Composite | 0.89 | 0.3 | 2.03 |
| Confirming_Composite | 0.84 | 0.8 | 1.51 |

**Concordance Analyse:**
- 99.0% van de tijd = `neutral` scenario
- Slechts 1.0% = `moderate_bullish`
- 0.0% = bearish scenarios

### 2.2 Root Cause
- **Lage entropy** (0.1–0.8) betekent dat de Composites bijna altijd dezelfde output geven.
- De gewichten uit Alpha Analysis tonen extreme saturatie: 47 signalen op max weight (2.5), terwijl 4 signalen op 0.0 staan.
- De neutrale band (±0.15) is te smal voor de huidige signal densiteit.

### 2.3 Tuning Parameters
| Parameter | Huidige Waarde | Voorstel | Locatie |
|-----------|----------------|----------|---------|
| Neutral band | ±0.15 | **±0.10** of dynamisch | `signal_aggregator.py` |
| Max weight cap | 2.5 | 1.5 (compressie) | `alpha-analysis/analyze_signal_alpha.py` |
| Weight smoothing | - | Log-transform of sigmoid | Alpha Analysis |
| Active signals | 122 (60m suffix) | Prune < 0.3 weight | `qbn.signal_weights` |

**Specifieke signal issues:**
```
ao_twin_peaks_bearish_60:  weight = 0.0 (nooit actief)
ao_twin_peaks_bullish_60:  weight = 0.0 (nooit actief)
mtf_bearish_alignment_60:  weight = 0.0 (design fout: MTF op 60m heeft geen zin)
mtf_bullish_alignment_60:  weight = 0.0 (design fout)
```

---

## 3. Entry Layer: `Entry_Timing`

### 3.1 Observaties
| Metric | Waarde | Target | Status |
|--------|--------|--------|--------|
| Stability | 0.88 | > 0.70 | ✅ OK |
| Entropy | 1.8 | 1.0-2.0 | ✅ OK |
| Info Gain | 0.22 | > 0.30 | ⚠️ Marginaal |
| Semantic Score | 0.29 | > 0.50 | 🔴 Laag |

### 3.2 Root Cause
- **Semantic score 0.29** betekent dat de mapping van Composite → Entry_Timing niet logisch consistent is.
- De latente variabele ontvangt 3x near-identical "neutral" inputs van de Composites → outputdistributie plat.
- Info Gain van 0.22 is te laag om als effectief filter te dienen.

### 3.3 Tuning Parameters
| Parameter | Huidige Waarde | Voorstel | Locatie |
|-----------|----------------|----------|---------|
| State count | 4 (Poor→Excellent) | Behouden | - |
| CPT prior | Uniform | **Dirichlet α=0.5** | `qbn_v2_cpt_generator.py` |
| Parent discretization | 5 states per Composite | 3 states (collapsed) | `entry_model_inference.py` |

---

## 4. Prediction Layer: `Prediction_1h`, `Prediction_4h`, `Prediction_1d`

### 4.1 Observaties
| Metric | 1h | 4h | 1d |
|--------|----|----|-----|
| Stability | 0.00 | 0.00 | 0.00 |
| Entropy | 0.8 | 2.4 | 2.4 |
| Semantic Score | 0.67 | 0.67 | 0.67 |
| **WF Accuracy** | 47.9% | 48.1% | 47.0% |
| **WF Dir Accuracy** | 47.9% | 48.1% | 47.0% |
| **Brier Score** | 0.707 | 0.708 | 0.677 |
| **Prediction Dist** | 100% Neutral | 100% Neutral | 100% Neutral |

### 4.2 Root Cause
1. **Stability = 0.00**: De CPT vergelijking tussen windows faalt volledig. Mogelijke oorzaken:
   - Te weinig niet-neutrale observaties per window
   - Bug in stability check logica
2. **100% Neutral output**: De `argmax` van de posterior is altijd "Neutral" omdat:
   - Prior (HTF_Regime) is uniform (entropy 3.2)
   - Likelihood (Entry_Timing) geeft geen differentiatie
   - Posterior collapseert naar prior → Neutral wint bij twijfel

### 4.3 Tuning Parameters
| Parameter | Huidige Waarde | Voorstel | Locatie |
|-----------|----------------|----------|---------|
| ATR thresholds | [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25] | **[-1.5, -0.5, -0.1, 0.1, 0.5, 1.5]** | `target_generator.py` |
| State count | 7 | **5** (merge ±3 into ±2) | `node_types.py` |
| Smoothing factor | - | Laplace α=1.0 | `qbn_v2_cpt_generator.py` |
| Min observations per CPT cell | - | 50 | CPT Generator |

---

## 5. Outcome Data Quality

| Check | Resultaat |
|-------|-----------|
| Lookahead Bias | ✅ Geen violations |
| Coverage 1h | 100.0% |
| Coverage 4h | 99.9% |
| Coverage 1d | 99.3% |
| ATR Correlation | ✅ Monotoon (-3 → +3 = -2.01 → +2.05 ATR) |

**Outcome Distributie (1h):**
```
-3:  3.25%  |  0: 40.35%  |  +3:  3.30%
-2:  5.35%  |             |  +2:  5.56%
-1: 20.62%  |             |  +1: 21.57%
```
De data is correct en de distributie is symmetrisch. Het probleem ligt niet bij de targets.

---

## 6. Prioritized Action Plan

| Prio | Actie | Impact | Effort |
|------|-------|--------|--------|
| 1 | Reduceer HTF_Regime van 11 → 5 states | Hoog | Laag |
| 2 | Verklein neutrale band Composites (0.15 → 0.10) | Hoog | Laag |
| 3 | Merge Prediction states (7 → 5) | Medium | Laag |
| 4 | Prune zero-weight signalen uit signal_weights | Medium | Laag |
| 5 | Implementeer Dirichlet prior (α=0.5) voor CPT smoothing | Hoog | Medium |
| 6 | Debug stability check in `qbn_v2_cpt_generator.py` | Hoog | Medium |
| 7 | Weight compression: log-transform of cap op 1.5 | Medium | Medium |

---

## 7. Verwachte Resultaten na Tuning

| Metric | Huidig | Target na Prio 1-3 |
|--------|--------|---------------------|
| HTF_Regime Stability | 0.40 | > 0.70 |
| Prediction Stability | 0.00 | > 0.60 |
| Neutral % in WF | 100% | < 60% |
| Directional Accuracy | 48% | > 55% |
| Brier Score | 0.70 | < 0.55 |

---

*Analyse gegenereerd: 2026-01-04T21:30*

