# Audit Rapport: Signaal Definities en Discretisatie Logica
**Datum:** 2026-01-04
**Onderwerp:** Synchronisatie tussen KFL v3, KFL GPU en QBN v2

## 1. Samenvatting van Bevindingen
De audit heeft aangetoond dat er momenteel drie verschillende lagen van logica en configuratie bestaan die niet automatisch gesynchroniseerd zijn. Hoewel de waarden op dit moment grotendeels overeenkomen, vormt de handmatige synchronisatie een risico voor de consistentie van het model.

| Component | Bron van Waarheid | Discretisatie Methode |
|-----------|-------------------|----------------------|
| **KFL v3 (Real-time)** | `signals.yaml` (boolean), Python code (gradueel) | Hardcoded in `DiscretizeEngine.py` |
| **KFL GPU (Backfill)** | Python code | Hardcoded in `discrete_signals.py` |
| **QBN v2 (Bayesian)** | `qbn.signal_classification` (DB) | Aggregatie in `SignalAggregator.py` |

---

## 2. Detail Analyse per Component

### 2.1 KFL Backend v3
- **Graduele Signalen:** De `DiscretizeEngine` negeert `discretization.yaml`. Drempelwaarden (RSI < 20, < 30, etc.) staan hardcoded in statische methodes.
- **Boolean Signalen:** Gebruiken `signals.yaml`. Hierin staan drempels die handmatig gelijk zijn gehouden aan de graduele drempels.
- **Numba Backfill:** De `discretize_numba.py` bevat een derde kopie van deze drempels voor geoptimaliseerde berekeningen.

### 2.2 KFL GPU v5.3
- De `DiscreteSignalCalculator` bevat een volledige herimplementatie van de v3 logica. 
- **Inconsistentie Gevaar:** Wijzigingen in de v3 YAML bestanden worden hier niet overgenomen. De GPU container heeft geen toegang tot de `S:/` schijf config.

### 2.3 QBN v2 & Validation
- **Database Afhankelijkheid:** QBN leest definities uit `qbn.signal_classification`. Deze tabel wordt handmatig gezaaid via SQL migrations (`003_seed_signal_classification.sql`).
- **Meta-Discretisatie:** QBN berekent zelf een `CompositeState` (-2 tot +2) op basis van de binnenkomende signalen. De thresholds hiervoor staan in `network_config.py`:
  - `COMPOSITE_NEUTRAL_BAND = 0.15`
  - `COMPOSITE_STRONG_THRESHOLD = 0.5`
- **Consistentie:** De `WalkForwardValidator` gebruikt exact dezelfde `SignalAggregator` als de productie pipeline, wat goed is voor de validatie-integriteit.

---

## 3. Gevonden Inconsistenties & Risico's

1.  **Dode Configuratie:** `discretization.yaml` wordt nergens actief gebruikt door de engines. Dit kan leiden tot verwarring bij aanpassingen.
2.  **Handmatige SQL Sync:** De `qbn.signal_classification` tabel moet handmatig worden bijgewerkt als er in `signals.yaml` iets verandert. Het script `validate_signal_classification.py` in QBN wijst naar verouderde `v2` paden.
3.  **Triple Coding:** Drempelwaarden voor indicators zoals RSI en MACD staan op minimaal drie plekken hardcoded (v3 Engine, v3 Numba, GPU Engine).

---

## 4. Aanbevelingen

1.  **Consolideer v3 Config:** Pas `DiscretizeEngine` aan zodat deze de waarden uit `discretization.yaml` inlaadt in plaats van magic numbers te gebruiken.
2.  **Gedeelde Config Volume:** Zorg dat de GPU container de YAML bestanden van de v3 pipeline kan inzien (bijv. via een Docker volume mount).
3.  **Auto-Sync Script:** Ontwikkel een script dat `qbn.signal_classification` in de database automatisch update op basis van `signals.yaml` en `signal_classification.yaml`.
4.  **Update Validator Paden:** Werk de paden in `validate_signal_classification.py` bij naar de `v3` structuur.

---
# TODO-verify: Is het gewenst om de 5-state aggregatie in QBN (SignalAggregator) te behouden, of moet dit direct vanuit de KFL signalen komen? (low-confidence)

