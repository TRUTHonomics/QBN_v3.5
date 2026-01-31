# Validatie Instructies: ThresholdOptimizer Refactor

Dit document beschrijft de stappen om de gerefactorde ThresholdOptimizer te valideren.

## Samenvatting Wijzigingen

### 1. Dynamic Signal Loader (`threshold_optimizer.py`)
- Verwijderd: Hardcoded 15 signalen in SQL queries
- Toegevoegd: Dynamische loader die alle 125 signalen uit `qbn.signal_classification` haalt
- Signalen worden nu gegroepeerd per semantic_class (LEADING, COINCIDENT, CONFIRMING)

### 2. Composite Score Berekening (`threshold_optimizer.py`)
- Verwijderd: Hardcoded berekening met 9 signalen
- Toegevoegd: `_compute_class_score()` die alle signalen per class gebruikt
- Weights worden nu horizon-specifiek opgehaald uit `qbn.signal_weights`

### 3. Constrained MI Grid Search (`mutual_information_analyzer.py`)
- `NEUTRAL_BAND_VALUES`: Minimum verhoogd van 0.001 naar 0.05
- `STRONG_THRESHOLD_VALUES`: Aangepast naar [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
- Nieuwe constraints:
  - Minimaal 3 actieve states (>5% representatie)
  - Neutral state tussen 10-60%

### 4. Per-Node Output (`run_threshold_analysis.py`)
- Nieuwe flag: `--per-node` voor per-node JSON output
- Nieuwe flag: `--no-diversity-check` om constraints uit te schakelen
- Diversity metrics worden nu gelogd

---

## Validatie Stappen

### Stap 1: Re-run Threshold Optimalisatie

```bash
# In de QBN_v3.1_Training container:
docker exec -it QBN_v3.1 python scripts/run_threshold_analysis.py \
    --asset-id 1 \
    --methods mi \
    --horizons all \
    --lookback-days 180 \
    --per-node \
    --apply-results
```

### Stap 2: Controleer Diversity Metrics

Bekijk de output logs voor:
- `diversity_score` >= 0.6 (3/5 actieve states)
- `valid_combinations` > 0 (niet alle combinaties rejected)
- `neutral` state tussen 10-60%

Voorbeeld verwachte output:
```
Leading_Composite:
  diversity_score=0.60, active_states=3, valid_combos=24
  states: bearish=15%, bullish=25%, neutral=35%, strong_bearish=5%, strong_bullish=20%
```

### Stap 3: Run Node-Level Diagnostics

```bash
# In de QBN_v3.1_Validation container:
docker exec -it QBN_v3.1_Validation python -c "
from menus.validation_menu import run_node_level_diagnostics
run_node_level_diagnostics()
"
# Of via het menu: optie 14
```

### Stap 4: Controleer Verbeteringen

Vergelijk met eerdere diagnostic resultaten:

| Metric | Voor | Na (Verwacht) |
|--------|------|---------------|
| Leading_Composite bullish % | 99.6% | 20-40% |
| Leading_Composite neutral % | ~0% | 10-60% |
| Leading_Composite active states | 1 | 3+ |
| MI score | ~0.01 | >0.02 |
| Directional alignment | 31% | >45% |

### Stap 5: Walk-Forward Validation

```bash
# In de QBN_v3.1_Validation container, optie 8:
docker exec -it QBN_v3.1_Validation python -c "
from menus.validation_menu import run_walk_forward
run_walk_forward()
"
```

---

## Troubleshooting

### "No valid threshold combinations found"
- De diversity constraints zijn te streng voor de huidige data
- Probeer `--no-diversity-check` om te vergelijken
- Check of de composite scores voldoende variatie hebben

### "Insufficient data" errors
- Verhoog `--lookback-days` naar 365
- Check of `qbn.signal_outcomes` voldoende data heeft

### Alle composites nog steeds biased
- Controleer of `qbn.signal_weights` correct is gevuld (menu optie 6)
- Controleer of alle signalen in de database aanwezig zijn

---

*Laatst bijgewerkt: 2026-01-10*
